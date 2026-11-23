#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>

#include "gltf_io.h"
#include "image.h"
#include "neural/codec.h"
#include "passes.h"

#ifndef GTC_GIT_SHA
#define GTC_GIT_SHA "unknown"
#endif

namespace fs = std::filesystem;

namespace gtc {

namespace {

using tinygltf::Value;

constexpr const char* kExtension = "BABYLON_materials_neural_texture";
constexpr const char* kSpecGlossExtension = "KHR_materials_pbrSpecularGlossiness";
constexpr int GL_LINEAR = 9729, GL_LINEAR_MIPMAP_LINEAR = 9987, GL_REPEAT = 10497;
constexpr int kMinNeuralSize = 64; // smaller materials cost more in weights and latents than they save

// Loss weight per canonical channel (see NeuralChannel).
constexpr float kLossWeights[kMaxNeuralChannels] = {1, 1, 1, 0.5f, 1.5f, 1.5f, 0.5f, 1, 1, // core, as in v1
                                                    1, 1, 1,                                // emissive
                                                    1, 1, 1, 0.5f,                          // diffuse
                                                    0.75f, 0.75f, 0.75f,                    // specular
                                                    1.5f};                                  // glossiness

enum SlotId { SBase, SNormal, SOcc, SMr, SEmissive, SDiffuse, SSpecGloss, SlotCount };
enum GroupId { GBase, GNormal, GOcc, GRough, GMetal, GEmissive, GDiffuse, GSpec, GGloss, GroupCount };

struct SlotDef {
    const char* replaced; // name in replacedSlots
    const char* kind;     // placeholder kind and log label
    bool v2Only;
    bool specGloss;       // lives in KHR_materials_pbrSpecularGlossiness
};
constexpr SlotDef kSlots[SlotCount] = {
    {"baseColorTexture", "baseColor", false, false},
    {"normalTexture", "normal", false, false},
    {"occlusionTexture", "occlusion", false, false},
    {"metallicRoughnessTexture", "metallicRoughness", false, false},
    {"emissiveTexture", "emissive", true, false},
    {"KHR_materials_pbrSpecularGlossiness.diffuseTexture", "diffuse", true, true},
    {"KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture", "specularGlossiness", true, true},
};

struct GroupDef {
    const char* name; // key in the extension's outputs
    int canonical;    // first canonical channel
    int slot;
    int srcChannel;   // first source image channel
    float fill;       // target value when the channel has no source (v1 only)
};
constexpr GroupDef kGroups[GroupCount] = {
    {"baseColor", kChBaseColor, SBase, 0, 0.0f},
    {"normal", kChNormal, SNormal, 0, 0.5f},
    {"occlusion", kChOcclusion, SOcc, 0, 1.0f},
    {"roughness", kChRoughness, SMr, 1, 0.0f},
    {"metallic", kChMetallic, SMr, 2, 0.0f},
    {"emissive", kChEmissive, SEmissive, 0, 0.0f},
    {"diffuse", kChDiffuse, SDiffuse, 0, 0.0f},
    {"specular", kChSpecular, SSpecGloss, 0, 0.0f},
    {"glossiness", kChGlossiness, SSpecGloss, 3, 1.0f},
};

struct Slot {
    int* core = nullptr;       // texture index of a core slot
    Value* ext = nullptr;      // texture info object of an extension slot
    int tex = -1;
    bool active = false;
    Image img;

    void setIndex(int index) {
        if (core) *core = index;
        else if (ext) ext->Get<Value::Object>()["index"] = Value(index);
    }
};

bool isUnsupportedMapping(const Value& info) {
    if (!info.IsObject()) return false;
    auto& o = info.Get<Value::Object>();
    auto tc = o.find("texCoord");
    if (tc != o.end() && tc->second.IsNumber() && tc->second.GetNumberAsInt() != 0) return true;
    auto ex = o.find("extensions");
    return ex != o.end() && ex->second.IsObject() && ex->second.Get<Value::Object>().count("KHR_texture_transform");
}

bool isUnsupportedMapping(const tinygltf::TextureInfo& t) { return t.texCoord != 0 || t.extensions.count("KHR_texture_transform"); }
bool isUnsupportedMapping(const tinygltf::NormalTextureInfo& t) { return t.texCoord != 0 || t.extensions.count("KHR_texture_transform"); }
bool isUnsupportedMapping(const tinygltf::OcclusionTextureInfo& t) { return t.texCoord != 0 || t.extensions.count("KHR_texture_transform"); }

Value* specGlossTexture(tinygltf::Material& mat, const char* key) {
    auto it = mat.extensions.find(kSpecGlossExtension);
    if (it == mat.extensions.end() || !it->second.IsObject()) return nullptr;
    auto& o = it->second.Get<Value::Object>();
    auto t = o.find(key);
    if (t == o.end() || !t->second.IsObject()) return nullptr;
    auto& to = t->second.Get<Value::Object>();
    auto idx = to.find("index");
    if (idx == to.end() || !idx->second.IsNumber() || idx->second.GetNumberAsInt() < 0) return nullptr;
    return &t->second;
}

// Texture slots a material references and that the codec can encode. Mapping and image support
// are checked per slot by the caller.
void gatherSlots(tinygltf::Material& mat, bool v2, bool specGloss, Slot (&slots)[SlotCount], bool (&unsupported)[SlotCount]) {
    auto& pbr = mat.pbrMetallicRoughness;
    auto core = [&](int id, int& index, bool bad) {
        if (index < 0 || (kSlots[id].v2Only && !v2)) return;
        slots[id].core = &index;
        slots[id].tex = index;
        unsupported[id] = bad;
    };
    core(SBase, pbr.baseColorTexture.index, isUnsupportedMapping(pbr.baseColorTexture));
    core(SNormal, mat.normalTexture.index, isUnsupportedMapping(mat.normalTexture));
    core(SOcc, mat.occlusionTexture.index, isUnsupportedMapping(mat.occlusionTexture));
    core(SMr, pbr.metallicRoughnessTexture.index, isUnsupportedMapping(pbr.metallicRoughnessTexture));
    core(SEmissive, mat.emissiveTexture.index, isUnsupportedMapping(mat.emissiveTexture));
    if (v2 && specGloss) {
        for (auto [id, key] : {std::pair{SDiffuse, "diffuseTexture"}, std::pair{SSpecGloss, "specularGlossinessTexture"}}) {
            if (Value* info = specGlossTexture(mat, key)) {
                slots[id].ext = info;
                slots[id].tex = info->Get<Value::Object>()["index"].GetNumberAsInt();
                unsupported[id] = isUnsupportedMapping(*info);
            }
        }
    }
}

// Active output channel set of a material, in training order.
struct Channels {
    struct Ch {
        int group, comp, canonical;
        bool active;
    };
    std::vector<Ch> list;
    int start[GroupCount];
    int comps[GroupCount];
    Channels() { std::fill(start, start + GroupCount, -1), std::fill(comps, comps + GroupCount, 0); }
    bool has(int g) const { return start[g] >= 0; }
};

double psnr(const std::vector<float>& a, const std::vector<float>& b, size_t texels, int stride, int start, int count) {
    double se = 0;
    for (size_t i = 0; i < texels; ++i)
        for (int c = start; c < start + count; ++c) {
            double d = std::clamp(a[i * stride + c], 0.0f, 1.0f) - b[i * stride + c];
            se += d * d;
        }
    double mse = se / (double(texels) * count);
    return mse <= 1e-12 ? 99.0 : 10.0 * std::log10(1.0 / mse);
}

uint8_t toU8(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

// Z of a unit normal stored as XY in [0,1], remapped to [0,1].
float normalZ(float nx, float ny) {
    float x = std::clamp(nx, 0.0f, 1.0f) * 2 - 1, y = std::clamp(ny, 0.0f, 1.0f) * 2 - 1;
    return std::sqrt(std::max(0.0f, 1 - x * x - y * y)) * 0.5f + 0.5f;
}

// Writes the comparison images: base color, normal with reconstructed Z, packed
// occlusion/roughness/metallic, emissive, diffuse, specular and glossiness.
void writeComparisonPngs(const fs::path& dir, const std::string& prefix, const std::vector<float>& d, int stride, int w, int h,
                         const Channels& ch, bool alpha) {
    const size_t n = size_t(w) * h;
    auto make = [&](auto fillTexel) {
        Image img{w, h};
        img.rgba.resize(n * 4);
        for (size_t i = 0; i < n; ++i) fillTexel(&d[i * stride], &img.rgba[i * 4]);
        return img;
    };
    auto at = [&](const float* p, int g, int c) { return ch.has(g) && c < ch.comps[g] ? p[ch.start[g] + c] : 0.0f; };
    auto rgbGroup = [&](const char* name, int g, bool withAlpha) {
        Image img = make([&](const float* p, uint8_t* o) {
            for (int c = 0; c < 3; ++c) o[c] = toU8(at(p, g, c));
            o[3] = withAlpha && ch.comps[g] > 3 ? toU8(at(p, g, 3)) : 255;
        });
        writePng(fromPath(dir / (prefix + "_" + name + ".png")), img, withAlpha && ch.comps[g] > 3 ? 4 : 3);
    };
    if (ch.has(GBase)) rgbGroup("baseColor", GBase, alpha);
    if (ch.has(GNormal)) {
        Image img = make([&](const float* p, uint8_t* o) {
            o[0] = toU8(at(p, GNormal, 0));
            o[1] = toU8(at(p, GNormal, 1));
            o[2] = toU8(normalZ(at(p, GNormal, 0), at(p, GNormal, 1)));
            o[3] = 255;
        });
        writePng(fromPath(dir / (prefix + "_normal.png")), img, 3);
    }
    if (ch.has(GOcc) || ch.has(GRough) || ch.has(GMetal)) {
        Image img = make([&](const float* p, uint8_t* o) {
            o[0] = toU8(ch.has(GOcc) ? at(p, GOcc, 0) : 1.0f);
            o[1] = toU8(at(p, GRough, 0));
            o[2] = toU8(at(p, GMetal, 0));
            o[3] = 255;
        });
        writePng(fromPath(dir / (prefix + "_occlusionRoughnessMetallic.png")), img, 3);
    }
    if (ch.has(GEmissive)) rgbGroup("emissive", GEmissive, false);
    if (ch.has(GDiffuse)) rgbGroup("diffuse", GDiffuse, alpha);
    if (ch.has(GSpec)) rgbGroup("specular", GSpec, false);
    if (ch.has(GGloss)) {
        Image img = make([&](const float* p, uint8_t* o) { o[0] = o[1] = o[2] = toU8(at(p, GGloss, 0)), o[3] = 255; });
        writePng(fromPath(dir / (prefix + "_glossiness.png")), img, 3);
    }
}

struct Placeholders {
    Document& doc;
    int sampler = -1;
    std::map<std::string, int> flat; // slot kind -> texture index

    int flatTexture(const std::string& kind, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255, int channels = 3) {
        auto it = flat.find(kind);
        if (it != flat.end()) return it->second;
        Image img{1, 1, {r, g, b, a}};
        int image = doc.addImage(encodePng(img, channels), "image/png", "neural_placeholder_" + kind);
        return flat[kind] = doc.addTexture(image, sampler, "neural_placeholder_" + kind);
    }
};

// Plain PNG for one replaced slot, built from the decoded network output with the same packing as
// writeComparisonPngs. sharedOrm packs occlusion, roughness and metallic into one image (R, G, B).
Image packSlotImage(int id, const std::vector<float>& d, int stride, int w, int h, const Channels& ch, bool alpha, bool sharedOrm,
                    int& channels) {
    const size_t n = size_t(w) * h;
    Image img{w, h};
    img.rgba.resize(n * 4);
    auto at = [&](const float* p, int g, int c) { return ch.has(g) && c < ch.comps[g] ? p[ch.start[g] + c] : 0.0f; };
    auto rgb = [&](int g, bool withAlpha) {
        const bool a = withAlpha && ch.comps[g] > 3;
        channels = a ? 4 : 3;
        for (size_t i = 0; i < n; ++i) {
            const float* p = &d[i * stride];
            uint8_t* o = &img.rgba[i * 4];
            for (int c = 0; c < 3; ++c) o[c] = toU8(at(p, g, c));
            o[3] = a ? toU8(at(p, g, 3)) : 255;
        }
    };
    channels = 3;
    switch (id) {
    case SBase: rgb(GBase, alpha); break;
    case SEmissive: rgb(GEmissive, false); break;
    case SDiffuse: rgb(GDiffuse, alpha); break;
    case SNormal:
        for (size_t i = 0; i < n; ++i) {
            const float* p = &d[i * stride];
            uint8_t* o = &img.rgba[i * 4];
            o[0] = toU8(at(p, GNormal, 0));
            o[1] = toU8(at(p, GNormal, 1));
            o[2] = toU8(normalZ(at(p, GNormal, 0), at(p, GNormal, 1)));
            o[3] = 255;
        }
        break;
    case SOcc:
    case SMr:
        for (size_t i = 0; i < n; ++i) {
            const float* p = &d[i * stride];
            uint8_t* o = &img.rgba[i * 4];
            const uint8_t occ = ch.has(GOcc) ? toU8(at(p, GOcc, 0)) : 255;
            if (id == SOcc && !sharedOrm) {
                o[0] = o[1] = o[2] = occ;
            } else {
                o[0] = id == SMr && !sharedOrm ? 255 : occ;
                o[1] = toU8(at(p, GRough, 0));
                o[2] = toU8(at(p, GMetal, 0));
            }
            o[3] = 255;
        }
        break;
    case SSpecGloss:
        channels = 4;
        for (size_t i = 0; i < n; ++i) {
            const float* p = &d[i * stride];
            uint8_t* o = &img.rgba[i * 4];
            for (int c = 0; c < 3; ++c) o[c] = toU8(at(p, GSpec, c));
            o[3] = toU8(ch.has(GGloss) ? at(p, GGloss, 0) : 1.0f);
        }
        break;
    }
    return img;
}

// Replaces the active slots of a material by plain PNG textures built from the decoded output. `target`
// are the slots of the material to modify in `d`; `src` are the slots as gathered from the input (their
// texture indices are valid in `d` because textures are only ever appended).
void bakeSlots(Document& d, Slot (&target)[SlotCount], const Slot (&src)[SlotCount], const std::vector<float>& decoded, int stride, int w,
               int h, const Channels& ch, bool alpha, const std::string& label) {
    const bool sharedOrm = src[SOcc].active && src[SMr].active && src[SOcc].tex == src[SMr].tex;
    int ormTexture = -1;
    for (int id = 0; id < SlotCount; ++id) {
        if (!src[id].active) continue;
        const bool orm = sharedOrm && (id == SOcc || id == SMr);
        if (orm && ormTexture >= 0) {
            target[id].setIndex(ormTexture);
            continue;
        }
        int channels = 3;
        Image img = packSlotImage(id, decoded, stride, w, h, ch, alpha, orm, channels);
        const std::string name = label + "_baked_" + kSlots[id].kind;
        int image = d.addImage(encodePng(img, channels), "image/png", name);
        int tex = d.addTexture(image, d.model.textures[src[id].tex].sampler, name);
        target[id].setIndex(tex);
        if (orm) ormTexture = tex;
    }
}

// Minimal JSON serialization of tinygltf values for the metrics sidecar.
void writeJson(std::string& out, const Value& v, int indent) {
    const std::string pad(size_t(indent) * 2, ' '), pad2(size_t(indent + 1) * 2, ' ');
    char buf[64];
    if (v.IsString()) {
        out += '"';
        for (char c : v.Get<std::string>()) {
            if (c == '"' || c == '\\') out += '\\', out += c;
            else if (c == '\n') out += "\\n";
            else if (c == '\t') out += "\\t";
            else if ((unsigned char)c < 0x20) out += ' ';
            else out += c;
        }
        out += '"';
    } else if (v.IsBool()) {
        out += v.Get<bool>() ? "true" : "false";
    } else if (v.IsInt()) {
        out += std::to_string(v.Get<int>());
    } else if (v.IsReal()) {
        double x = v.Get<double>();
        if (std::floor(x) == x && std::fabs(x) < 1e15) std::snprintf(buf, sizeof(buf), "%.0f", x);
        else std::snprintf(buf, sizeof(buf), "%.6g", x);
        out += buf;
    } else if (v.IsArray()) {
        const auto& a = v.Get<Value::Array>();
        const bool flat = std::all_of(a.begin(), a.end(), [](const Value& e) { return !e.IsArray() && !e.IsObject(); });
        out += '[';
        for (size_t i = 0; i < a.size(); ++i) {
            if (!flat) out += "\n" + pad2;
            else if (i) out += ", ";
            writeJson(out, a[i], indent + 1);
            if (!flat && i + 1 < a.size()) out += ',';
        }
        if (!flat && !a.empty()) out += "\n" + pad;
        out += ']';
    } else if (v.IsObject()) {
        const auto& o = v.Get<Value::Object>();
        out += '{';
        size_t i = 0;
        for (const auto& [k, e] : o) {
            out += "\n" + pad2 + "\"" + k + "\": ";
            writeJson(out, e, indent + 1);
            if (++i < o.size()) out += ',';
        }
        if (!o.empty()) out += "\n" + pad;
        out += '}';
    } else {
        out += "null";
    }
}

} // namespace

bool runNeuralPass(Document& doc, const NeuralOptions& opts, const std::string& pngDir, std::string& err) {
    auto codec = createNeuralCodec(opts, err);
    if (!codec) return false;
    const bool v2 = codec->name() != "babylon-ntc-v1";
    const bool specGloss = opts.specGloss != "ignore";
    const std::string tag = opts.tag.empty() ? "neural" : opts.tag;
    std::printf("  codec %s on %s\n", codec->name().c_str(), codec->deviceDescription().c_str());

    auto& m = doc.model;
    // Plain-glTF copy of the input, filled with the decoded textures of every encoded material.
    std::unique_ptr<Document> baked;
    if (!opts.bakedPath.empty()) baked = std::make_unique<Document>(doc);
    if (!pngDir.empty()) {
        fs::create_directories(toPath(pngDir) / "original");
        fs::create_directories(toPath(pngDir) / tag);
    }
    const int latentSampler = doc.addSampler(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_REPEAT);
    Placeholders ph{doc, doc.addSampler(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_REPEAT)};
    std::map<int, Image> decodedCache;
    auto loadTexture = [&](int tex, Image& out) -> bool {
        if (tex < 0 || tex >= int(m.textures.size()) || m.textures[tex].source < 0) {
            err = "texture " + std::to_string(tex) + " has no PNG/JPEG source";
            return false;
        }
        int src = m.textures[tex].source;
        auto it = decodedCache.find(src);
        if (it == decodedCache.end()) {
            Image img;
            if (!decodeImage(doc.imageData[src], img, err)) return false;
            it = decodedCache.emplace(src, std::move(img)).first;
        }
        out = it->second;
        return true;
    };

    auto selected = [&](size_t mi) {
        if (opts.materials.empty()) return true;
        const std::string name = m.materials[mi].name.empty() ? "material" + std::to_string(mi) : m.materials[mi].name;
        return std::find(opts.materials.begin(), opts.materials.end(), name) != opts.materials.end();
    };
    if (!opts.materials.empty()) {
        for (const std::string& want : opts.materials) {
            bool found = false;
            for (size_t mi = 0; mi < m.materials.size(); ++mi)
                found = found || m.materials[mi].name == want || (m.materials[mi].name.empty() && want == "material" + std::to_string(mi));
            if (!found) {
                err = "--neural-materials: no material named '" + want + "'";
                return false;
            }
        }
    }

    size_t count = 0, index = 0;
    for (size_t mi = 0; mi < m.materials.size(); ++mi) {
        if (!selected(mi)) continue;
        Slot slots[SlotCount];
        bool unsupported[SlotCount]{};
        gatherSlots(m.materials[mi], v2, specGloss, slots, unsupported);
        for (const Slot& s : slots)
            if (s.tex >= 0) {
                ++count;
                break;
            }
    }

    Value::Array materialMetrics;
    bool anyBakeOnly = false;
    const auto passStart = std::chrono::steady_clock::now();
    for (size_t mi = 0; mi < m.materials.size(); ++mi) {
        if (!selected(mi)) continue;
        auto& mat = m.materials[mi];
        Slot slots[SlotCount];
        bool unsupported[SlotCount]{};
        gatherSlots(mat, v2, specGloss, slots, unsupported);
        if (std::none_of(std::begin(slots), std::end(slots), [](const Slot& s) { return s.tex >= 0; })) continue;
        ++index;
        const std::string label = mat.name.empty() ? "material" + std::to_string(mi) : mat.name;

        // Per-slot checks: an unsupported slot is left untouched, the others are still encoded.
        for (int id = 0; id < SlotCount; ++id) {
            Slot& s = slots[id];
            if (s.tex < 0) continue;
            if (unsupported[id]) {
                std::printf("  [%zu/%zu] %s: %s kept (texCoord != 0 or KHR_texture_transform)\n", index, count, label.c_str(), kSlots[id].kind);
                continue;
            }
            if (!loadTexture(s.tex, s.img)) {
                std::printf("  [%zu/%zu] %s: %s kept (%s)\n", index, count, label.c_str(), kSlots[id].kind, err.c_str());
                err.clear();
                continue;
            }
            s.active = true;
        }
        int w = 0, h = 0;
        bool any = false;
        for (const Slot& s : slots) {
            if (!s.active) continue;
            any = true;
            w = std::max(w, s.img.width);
            h = std::max(h, s.img.height);
        }
        if (!any) continue;
        if (v2 && std::max(w, h) < kMinNeuralSize) {
            std::printf("  [%zu/%zu] %s: skipped (textures smaller than %d px)\n", index, count, label.c_str(), kMinNeuralSize);
            continue;
        }
        for (Slot& s : slots)
            if (s.active && (s.img.width != w || s.img.height != h)) s.img = resize(s.img, w, h);

        const bool blend = mat.alphaMode != "OPAQUE" && !mat.alphaMode.empty();

        // Output channels: v1 always uses the fixed core layout (unused channels get a zero loss weight),
        // v2 packs only the channels of the encoded slots, in canonical order.
        Channels ch;
        NeuralLayout layout;
        struct Source {
            const Image* img;
            int channel;
            float fill;
        };
        std::vector<Source> sources;
        for (int g = 0; g < GroupCount; ++g) {
            const GroupDef& gd = kGroups[g];
            const Slot& s = slots[gd.slot];
            if (!v2 && g > GMetal) break;
            if (v2 && !s.active) continue;
            int comps = g == GBase || g == GDiffuse ? (v2 ? (blend ? 4 : 3) : 4) : g == GNormal ? 2 : g == GEmissive || g == GSpec ? 3 : 1;
            ch.start[g] = int(sources.size());
            ch.comps[g] = comps;
            for (int c = 0; c < comps; ++c) {
                bool alphaOff = (g == GBase || g == GDiffuse) && c == 3 && !blend;
                bool active = s.active && !alphaOff;
                float fill = gd.fill;
                if (g == GBase && c == 3) fill = 1.0f;
                sources.push_back({active ? &s.img : nullptr, gd.srcChannel + c, fill});
                ch.list.push_back({g, c, gd.canonical + c, active});
                layout.canonical[layout.count++] = gd.canonical + c;
            }
            if (!v2 && !s.active) ch.start[g] = -1;
        }
        const int N = layout.count;

        NeuralMaterialInput in;
        in.name = label;
        in.width = w;
        in.height = h;
        in.outputs = N;
        in.layout = layout;
        for (int i = 0; i < N; ++i) in.lossWeights[i] = ch.list[i].active ? kLossWeights[layout.canonical[i]] : 0.0f;
        if (codec->usesWorkDir() && !opts.workDir.empty()) {
            in.workDir = fromPath(toPath(opts.workDir) / sanitizeFileName(std::to_string(mi) + "_" + label));
            fs::create_directories(toPath(in.workDir));
        }

        const size_t n = size_t(w) * h;
        in.target.resize(n * N);
        for (size_t i = 0; i < n; ++i) {
            float* t = &in.target[i * N];
            for (int c = 0; c < N; ++c) {
                const Source& s = sources[c];
                t[c] = s.img ? s.img->rgba[i * 4 + s.channel] / 255.0f : s.fill;
            }
        }

        std::string slotList;
        for (int id = 0; id < SlotCount; ++id)
            if (slots[id].active) slotList += std::string(slotList.empty() ? "" : " ") + kSlots[id].kind;
        std::printf("  [%zu/%zu] %s %dx%d, %d outputs (%s)\n", index, count, label.c_str(), w, h, N, slotList.c_str());
        auto t0 = std::chrono::steady_clock::now();
        NeuralMaterialOutput res;
        if (!codec->encode(in, res, err)) {
            err = label + ": " + err;
            return false;
        }
        const double encodeSeconds = secondsSince(t0);
        const bool bakeOnly = res.bakeOnly;

        size_t latentBytes = 0, weightBytes = 0;
        if (bakeOnly) {
            anyBakeOnly = true;
            // No glTF representation: plain textures replace the slots in the output itself.
            bakeSlots(doc, slots, slots, res.decoded, N, w, h, ch, blend, label);
        } else {
        // Store latents, weights and the extension.
        Value::Array latents;
        static const char* kLatentNames[] = {"hi0", "hi1", "lo0", "lo1"};
        for (size_t i = 0; i < res.latentImages.size(); ++i) {
            auto png = encodePng(res.latentImages[i], 4);
            latentBytes += png.size();
            std::string nm = label + "_latent_" + (i < 4 ? kLatentNames[i] : std::to_string(i));
            int img = doc.addImage(std::move(png), "image/png", nm);
            Value::Object ti;
            ti["index"] = Value(doc.addTexture(img, latentSampler, nm));
            latents.emplace_back(std::move(ti));
        }
        weightBytes = res.weights.size();
        if (weightBytes != codec->expectedWeightCount(N) * 4) {
            err = label + ": unexpected weight count";
            return false;
        }
        Value::Object weights;
        weights["bufferView"] = Value(doc.addBufferView(std::move(res.weights), label + "_neural_weights"));
        weights["componentType"] = Value(5126);
        weights["count"] = Value(int(weightBytes / 4));

        // v1 lists the channel indices of each group, v2 gives [startChannel, componentCount].
        Value::Object outputs;
        for (int g = 0; g < GroupCount; ++g) {
            if (!ch.has(g)) continue;
            Value::Array a;
            if (v2) {
                a.emplace_back(ch.start[g]);
                a.emplace_back(ch.comps[g]);
            } else {
                for (int c = 0; c < ch.comps[g]; ++c) a.emplace_back(ch.start[g] + c);
            }
            outputs[kGroups[g].name] = Value(std::move(a));
        }
        Value::Array replaced;
        for (int id = 0; id < SlotCount; ++id)
            if (slots[id].active) replaced.emplace_back(std::string(kSlots[id].replaced));

        Value::Object ext = std::move(res.parameters);
        ext["version"] = Value(1);
        ext["codec"] = Value(codec->name());
        ext["resolution"] = Value(Value::Array{Value(w), Value(h)});
        ext["latentTextures"] = Value(std::move(latents));
        ext["weights"] = Value(std::move(weights));
        ext["outputs"] = Value(std::move(outputs));
        ext["replacedSlots"] = Value(std::move(replaced));
        mat.extensions[kExtension] = Value(std::move(ext));

        // Replace the original slots with fallbacks for loaders that ignore the extension.
        struct Fallback {
            uint8_t r, g, b, a;
            int channels;
        };
        constexpr Fallback kFlat[SlotCount] = {{128, 128, 128, 255, 3}, {128, 128, 255, 255, 3}, {255, 255, 0, 255, 3},
                                               {255, 255, 0, 255, 3},   {0, 0, 0, 255, 3},       {128, 128, 128, 255, 3},
                                               {10, 10, 10, 128, 4}};
        for (int id = 0; id < SlotCount; ++id) {
            Slot& s = slots[id];
            if (!s.active) continue;
            const std::string kind = kSlots[id].kind;
            if (opts.fallbackSize > 0) {
                int fw = std::max(1, std::min(s.img.width, opts.fallbackSize));
                int fh = std::max(1, s.img.height * fw / std::max(1, s.img.width));
                Image small = resize(s.img, fw, fh);
                int img = doc.addImage(encodePng(small, small.hasAlpha() ? 4 : 3), "image/png", label + "_fallback_" + kind);
                s.setIndex(doc.addTexture(img, ph.sampler, label + "_fallback_" + kind));
            } else {
                const Fallback& f = kFlat[id];
                s.setIndex(ph.flatTexture(kind, f.r, f.g, f.b, f.a, f.channels));
            }
        }
        }

        if (baked) {
            Slot bslots[SlotCount];
            bool bunsupported[SlotCount]{};
            gatherSlots(baked->model.materials[mi], v2, specGloss, bslots, bunsupported);
            bakeSlots(*baked, bslots, slots, res.decoded, N, w, h, ch, blend, label);
        }

        struct Psnr {
            const char* name;
            double db;
        };
        std::vector<Psnr> scores;
        auto report = [&](const char* name, int g, int first, int count) {
            if (!ch.has(g) || !ch.list[ch.start[g]].active) return;
            scores.push_back({name, psnr(res.decoded, in.target, n, N, ch.start[g] + first, count)});
        };
        report("base", GBase, 0, 3);
        if (blend) report("alpha", GBase, 3, 1);
        report("normal", GNormal, 0, 2);
        report("occlusion", GOcc, 0, 1);
        if (ch.has(GRough) && ch.has(GMetal)) scores.push_back({"roughness/metallic", psnr(res.decoded, in.target, n, N, ch.start[GRough], 2)});
        report("emissive", GEmissive, 0, 3);
        report("diffuse", GDiffuse, 0, 3);
        if (blend) report("diffuse alpha", GDiffuse, 3, 1);
        report("specular", GSpec, 0, 3);
        report("glossiness", GGloss, 0, 1);
        std::string q = "    PSNR";
        char buf[64];
        for (const Psnr& s : scores) {
            std::snprintf(buf, sizeof(buf), " %s %.1f dB", s.name, s.db);
            q += buf;
        }
        const size_t compressedBytes = res.compressedBytes ? res.compressedBytes : latentBytes + weightBytes;
        if (bakeOnly)
            std::printf("%s\n    %s (%.2f bpp, baked, %.1f s)\n", q.c_str(), humanBytes(compressedBytes).c_str(),
                        8.0 * double(compressedBytes) / double(n), encodeSeconds);
        else
            std::printf("%s\n    latents %s + weights %s (%.1f s)\n", q.c_str(), humanBytes(latentBytes).c_str(),
                        humanBytes(weightBytes).c_str(), encodeSeconds);

        if (!opts.metricsPath.empty()) {
            Value::Object mm;
            mm["name"] = Value(label);
            mm["index"] = Value(int(mi));
            mm["width"] = Value(w);
            mm["height"] = Value(h);
            mm["outputs"] = Value(N);
            Value::Array canon;
            for (int i = 0; i < N; ++i) canon.emplace_back(layout.canonical[i]);
            mm["layout"] = Value(std::move(canon));
            Value::Array slotNames;
            for (int id = 0; id < SlotCount; ++id)
                if (slots[id].active) slotNames.emplace_back(std::string(kSlots[id].kind));
            mm["slots"] = Value(std::move(slotNames));
            mm["latentBytes"] = Value(double(latentBytes));
            mm["weightBytes"] = Value(double(weightBytes));
            mm["compressedBytes"] = Value(double(compressedBytes));
            mm["bpp"] = Value(8.0 * double(compressedBytes) / double(n));
            Value::Object ps;
            for (const Psnr& s : scores) ps[s.name] = Value(s.db);
            mm["psnr"] = Value(std::move(ps));
            mm["encodeSeconds"] = Value(encodeSeconds);
            mm["bakeOnly"] = Value(bakeOnly);
            if (!res.extraMetrics.empty()) mm["codecMetrics"] = Value(res.extraMetrics);
            materialMetrics.emplace_back(std::move(mm));
        }

        if (!pngDir.empty()) {
            const std::string prefix = sanitizeFileName(std::to_string(mi) + "_" + label);
            writeComparisonPngs(toPath(pngDir) / "original", prefix, in.target, N, w, h, ch, blend);
            writeComparisonPngs(toPath(pngDir) / tag, prefix, res.decoded, N, w, h, ch, blend);
        }
    }
    if (!anyBakeOnly) doc.addExtensionUsed(kExtension, false);

    if (baked) {
        std::printf("  writing %s\n", opts.bakedPath.c_str());
        if (!saveGlb(std::move(*baked), opts.bakedPath, err)) return false;
    }
    if (!opts.metricsPath.empty()) {
        Value::Object root;
        root["tool"] = Value(std::string("gltf-texture-compressor"));
        root["version"] = Value(std::string(GTC_VERSION));
        root["gitSha"] = Value(std::string(GTC_GIT_SHA));
        root["codec"] = Value(codec->name());
        root["device"] = Value(codec->deviceDescription());
        root["tag"] = Value(tag);
        root["commandLine"] = Value(opts.commandLine);
        root["seed"] = Value(int(opts.seed));
        root["steps"] = Value(opts.steps);
        root["batch"] = Value(opts.batch);
        root["parameters"] = Value(codec->describeParameters());
        root["totalEncodeSeconds"] = Value(secondsSince(passStart));
        root["materials"] = Value(std::move(materialMetrics));
        std::string json;
        writeJson(json, Value(std::move(root)), 0);
        std::ofstream f(toPath(opts.metricsPath), std::ios::binary);
        f << json << "\n";
        if (!f) {
            err = "failed to write " + opts.metricsPath;
            return false;
        }
        std::printf("  metrics %s\n", opts.metricsPath.c_str());
    }
    return true;
}

} // namespace gtc
