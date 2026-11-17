#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>

#include "gltf_io.h"
#include "image.h"
#include "neural/codec.h"
#include "passes.h"

namespace fs = std::filesystem;

namespace gtc {

namespace {

using tinygltf::Value;

constexpr const char* kExtension = "BABYLON_materials_neural_texture";
constexpr float kLossWeights[kNeuralChannels] = {1, 1, 1, 0.5f, 1.5f, 1.5f, 0.5f, 1, 1};
constexpr int GL_LINEAR = 9729, GL_LINEAR_MIPMAP_LINEAR = 9987, GL_REPEAT = 10497;

bool usesUnsupportedMapping(const tinygltf::TextureInfo& t) { return t.index >= 0 && (t.texCoord != 0 || t.extensions.count("KHR_texture_transform")); }
bool usesUnsupportedMapping(const tinygltf::NormalTextureInfo& t) { return t.index >= 0 && (t.texCoord != 0 || t.extensions.count("KHR_texture_transform")); }
bool usesUnsupportedMapping(const tinygltf::OcclusionTextureInfo& t) { return t.index >= 0 && (t.texCoord != 0 || t.extensions.count("KHR_texture_transform")); }

double psnr(const std::vector<float>& a, const std::vector<float>& b, size_t texels, std::initializer_list<int> channels) {
    double se = 0;
    for (size_t i = 0; i < texels; ++i)
        for (int c : channels) {
            double d = std::clamp(a[i * kNeuralChannels + c], 0.0f, 1.0f) - b[i * kNeuralChannels + c];
            se += d * d;
        }
    double mse = se / (double(texels) * channels.size());
    return mse <= 1e-12 ? 99.0 : 10.0 * std::log10(1.0 / mse);
}

uint8_t toU8(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

// Builds the comparison images (base color, normal with reconstructed Z, packed occlusion/roughness/metallic).
void writeComparisonPngs(const fs::path& dir, const std::string& prefix, const std::vector<float>& d, int w, int h, bool alpha,
                         bool hasNormal, bool hasOrm) {
    Image base{w, h}, normal{w, h}, orm{w, h};
    const size_t n = size_t(w) * h;
    base.rgba.resize(n * 4);
    normal.rgba.resize(n * 4);
    orm.rgba.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        const float* p = &d[i * kNeuralChannels];
        for (int c = 0; c < 4; ++c) base.rgba[i * 4 + c] = toU8(p[c]);
        float x = std::clamp(p[4], 0.0f, 1.0f) * 2 - 1, y = std::clamp(p[5], 0.0f, 1.0f) * 2 - 1;
        float z = std::sqrt(std::max(0.0f, 1 - x * x - y * y));
        normal.rgba[i * 4 + 0] = toU8(p[4]);
        normal.rgba[i * 4 + 1] = toU8(p[5]);
        normal.rgba[i * 4 + 2] = toU8(z * 0.5f + 0.5f);
        normal.rgba[i * 4 + 3] = 255;
        orm.rgba[i * 4 + 0] = toU8(p[6]);
        orm.rgba[i * 4 + 1] = toU8(p[7]);
        orm.rgba[i * 4 + 2] = toU8(p[8]);
        orm.rgba[i * 4 + 3] = 255;
    }
    writePng(fromPath(dir / (prefix + "_baseColor.png")), base, alpha ? 4 : 3);
    if (hasNormal) writePng(fromPath(dir / (prefix + "_normal.png")), normal, 3);
    if (hasOrm) writePng(fromPath(dir / (prefix + "_occlusionRoughnessMetallic.png")), orm, 3);
}

struct Placeholders {
    Document& doc;
    int sampler = -1;
    std::map<std::string, int> flat; // slot kind -> texture index

    int flatTexture(const std::string& kind, uint8_t r, uint8_t g, uint8_t b) {
        auto it = flat.find(kind);
        if (it != flat.end()) return it->second;
        Image img{1, 1, {r, g, b, 255}};
        int image = doc.addImage(encodePng(img, 3), "image/png", "neural_placeholder_" + kind);
        return flat[kind] = doc.addTexture(image, sampler, "neural_placeholder_" + kind);
    }
};

} // namespace

bool runNeuralPass(Document& doc, const NeuralOptions& opts, const std::string& pngDir, std::string& err) {
    auto codec = createNeuralCodec(opts, err);
    if (!codec) return false;
    std::printf("  codec %s on %s\n", codec->name().c_str(), codec->deviceDescription().c_str());

    auto& m = doc.model;
    if (!pngDir.empty()) {
        fs::create_directories(toPath(pngDir) / "original");
        fs::create_directories(toPath(pngDir) / "neural");
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

    size_t count = 0, index = 0;
    for (auto& mat : m.materials) {
        auto& pbr = mat.pbrMetallicRoughness;
        if (pbr.baseColorTexture.index >= 0 || mat.normalTexture.index >= 0 || mat.occlusionTexture.index >= 0 ||
            pbr.metallicRoughnessTexture.index >= 0)
            ++count;
    }

    for (size_t mi = 0; mi < m.materials.size(); ++mi) {
        auto& mat = m.materials[mi];
        auto& pbr = mat.pbrMetallicRoughness;
        const int texBase = pbr.baseColorTexture.index, texNormal = mat.normalTexture.index;
        const int texOcc = mat.occlusionTexture.index, texMr = pbr.metallicRoughnessTexture.index;
        if (texBase < 0 && texNormal < 0 && texOcc < 0 && texMr < 0) continue;
        ++index;
        const std::string label = mat.name.empty() ? "material" + std::to_string(mi) : mat.name;
        if (usesUnsupportedMapping(pbr.baseColorTexture) || usesUnsupportedMapping(mat.normalTexture) ||
            usesUnsupportedMapping(mat.occlusionTexture) || usesUnsupportedMapping(pbr.metallicRoughnessTexture)) {
            std::printf("  [%zu/%zu] %s: skipped (texCoord != 0 or KHR_texture_transform)\n", index, count, label.c_str());
            continue;
        }

        Image base, normal, occ, mr;
        if ((texBase >= 0 && !loadTexture(texBase, base)) || (texNormal >= 0 && !loadTexture(texNormal, normal)) ||
            (texOcc >= 0 && !loadTexture(texOcc, occ)) || (texMr >= 0 && !loadTexture(texMr, mr))) {
            std::printf("  [%zu/%zu] %s: skipped (%s)\n", index, count, label.c_str(), err.c_str());
            err.clear();
            continue;
        }
        int w = 0, h = 0;
        for (const Image* img : {&base, &normal, &occ, &mr}) {
            w = std::max(w, img->width);
            h = std::max(h, img->height);
        }
        auto fit = [&](Image& img) {
            if (!img.empty() && (img.width != w || img.height != h)) img = resize(img, w, h);
        };
        fit(base), fit(normal), fit(occ), fit(mr);

        NeuralMaterialInput in;
        in.name = label;
        in.width = w;
        in.height = h;
        std::copy(std::begin(kLossWeights), std::end(kLossWeights), in.lossWeights);
        const bool blend = mat.alphaMode != "OPAQUE" && !mat.alphaMode.empty();
        if (base.empty()) std::fill(in.lossWeights, in.lossWeights + 4, 0.0f);
        if (!blend) in.lossWeights[3] = 0.0f;
        if (normal.empty()) in.lossWeights[4] = in.lossWeights[5] = 0.0f;
        if (occ.empty()) in.lossWeights[6] = 0.0f;
        if (mr.empty()) in.lossWeights[7] = in.lossWeights[8] = 0.0f;

        const size_t n = size_t(w) * h;
        in.target.assign(n * kNeuralChannels, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            float* t = &in.target[i * kNeuralChannels];
            if (!base.empty()) for (int c = 0; c < 4; ++c) t[c] = base.rgba[i * 4 + c] / 255.0f;
            else t[3] = 1.0f;
            if (!normal.empty()) t[4] = normal.rgba[i * 4] / 255.0f, t[5] = normal.rgba[i * 4 + 1] / 255.0f;
            else t[4] = t[5] = 0.5f;
            t[6] = occ.empty() ? 1.0f : occ.rgba[i * 4] / 255.0f;
            if (!mr.empty()) t[7] = mr.rgba[i * 4 + 1] / 255.0f, t[8] = mr.rgba[i * 4 + 2] / 255.0f;
        }

        std::printf("  [%zu/%zu] %s %dx%d (base:%s normal:%s occlusion:%s metallicRoughness:%s)\n", index, count,
                    label.c_str(), w, h, base.empty() ? "-" : "y", normal.empty() ? "-" : "y", occ.empty() ? "-" : "y",
                    mr.empty() ? "-" : "y");
        auto t0 = std::chrono::steady_clock::now();
        NeuralMaterialOutput res;
        if (!codec->encode(in, res, err)) {
            err = label + ": " + err;
            return false;
        }

        // Store latents, weights and the extension.
        Value::Array latents;
        size_t latentBytes = 0;
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
        const size_t weightBytes = res.weights.size();
        Value::Object weights;
        weights["bufferView"] = Value(doc.addBufferView(std::move(res.weights), label + "_neural_weights"));
        weights["componentType"] = Value(5126);
        weights["count"] = Value(int(weightBytes / 4));

        Value::Object outputs;
        auto chans = [](std::initializer_list<int> v) {
            Value::Array a;
            for (int x : v) a.emplace_back(x);
            return Value(std::move(a));
        };
        Value::Array replaced;
        if (!base.empty()) outputs["baseColor"] = chans({0, 1, 2, 3}), replaced.emplace_back(std::string("baseColorTexture"));
        if (!normal.empty()) outputs["normal"] = chans({4, 5}), replaced.emplace_back(std::string("normalTexture"));
        if (!occ.empty()) outputs["occlusion"] = chans({6}), replaced.emplace_back(std::string("occlusionTexture"));
        if (!mr.empty()) outputs["roughness"] = chans({7}), outputs["metallic"] = chans({8}),
                         replaced.emplace_back(std::string("metallicRoughnessTexture"));

        Value::Object ext = std::move(res.parameters);
        ext["version"] = Value(1);
        ext["codec"] = Value(codec->name());
        ext["resolution"] = chans({w, h});
        ext["latentTextures"] = Value(std::move(latents));
        ext["weights"] = Value(std::move(weights));
        ext["outputs"] = Value(std::move(outputs));
        ext["replacedSlots"] = Value(std::move(replaced));
        mat.extensions[kExtension] = Value(std::move(ext));

        // Replace the original slots with fallbacks for loaders that ignore the extension.
        auto fallback = [&](int& slot, const Image& src, const std::string& kind, uint8_t r, uint8_t g, uint8_t b) {
            if (slot < 0) return;
            if (opts.fallbackSize > 0) {
                int fw = std::max(1, std::min(src.width, opts.fallbackSize));
                int fh = std::max(1, src.height * fw / std::max(1, src.width));
                Image small = resize(src, fw, fh);
                int img = doc.addImage(encodePng(small, small.hasAlpha() ? 4 : 3), "image/png", label + "_fallback_" + kind);
                slot = doc.addTexture(img, ph.sampler, label + "_fallback_" + kind);
            } else {
                slot = ph.flatTexture(kind, r, g, b);
            }
        };
        fallback(pbr.baseColorTexture.index, base, "baseColor", 128, 128, 128);
        fallback(mat.normalTexture.index, normal, "normal", 128, 128, 255);
        fallback(mat.occlusionTexture.index, occ, "occlusion", 255, 255, 0);
        fallback(pbr.metallicRoughnessTexture.index, mr, "metallicRoughness", 255, 255, 0);

        std::string q = "    PSNR";
        char buf[64];
        if (!base.empty()) std::snprintf(buf, sizeof(buf), " base %.1f dB", psnr(res.decoded, in.target, n, {0, 1, 2})), q += buf;
        if (blend && !base.empty()) std::snprintf(buf, sizeof(buf), " alpha %.1f dB", psnr(res.decoded, in.target, n, {3})), q += buf;
        if (!normal.empty()) std::snprintf(buf, sizeof(buf), " normal %.1f dB", psnr(res.decoded, in.target, n, {4, 5})), q += buf;
        if (!occ.empty()) std::snprintf(buf, sizeof(buf), " occlusion %.1f dB", psnr(res.decoded, in.target, n, {6})), q += buf;
        if (!mr.empty()) std::snprintf(buf, sizeof(buf), " roughness/metallic %.1f dB", psnr(res.decoded, in.target, n, {7, 8})), q += buf;
        std::printf("%s\n    latents %s + weights %s (%.1f s)\n", q.c_str(), humanBytes(latentBytes).c_str(),
                    humanBytes(weightBytes).c_str(), secondsSince(t0));

        if (!pngDir.empty()) {
            const std::string prefix = sanitizeFileName(std::to_string(mi) + "_" + label);
            const bool alphaPng = blend && !base.empty();
            writeComparisonPngs(toPath(pngDir) / "original", prefix, in.target, w, h, alphaPng, !normal.empty(),
                                !occ.empty() || !mr.empty());
            writeComparisonPngs(toPath(pngDir) / "neural", prefix, res.decoded, w, h, alphaPng, !normal.empty(),
                                !occ.empty() || !mr.empty());
        }
    }
    doc.addExtensionUsed(kExtension, false);
    return true;
}

} // namespace gtc
