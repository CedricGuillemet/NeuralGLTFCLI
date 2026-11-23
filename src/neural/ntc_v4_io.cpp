#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include "../gltf_io.h"
#include "ntc_v4.h"

namespace gtc::v4 {

using tinygltf::Value;
namespace fs = std::filesystem;

// ---- SHA-256 ---------------------------------------------------------------------------------------------------------

std::string sha256Hex(const uint8_t* data, size_t size) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> msg(data, data + size);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = uint64_t(size) * 8;
    for (int i = 7; i >= 0; --i) msg.push_back(uint8_t(bits >> (8 * i)));
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(msg[off + 4 * i]) << 24 | uint32_t(msg[off + 4 * i + 1]) << 16 | uint32_t(msg[off + 4 * i + 2]) << 8 | msg[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    char buf[65];
    for (int i = 0; i < 8; ++i) std::snprintf(buf + 8 * i, 9, "%08x", h[i]);
    return buf;
}

// ---- extension JSON / glb ---------------------------------------------------------------------------------------------

static Value intArray(std::initializer_list<int> v) {
    Value::Array a;
    for (int x : v) a.emplace_back(x);
    return Value(std::move(a));
}

Value::Object v4Parameters(const Model& m) {
    Value::Object p;
    p["mipCount"] = Value(m.M);
    p["levels"] = Value(m.L);
    Value::Array grids;
    int tex = 0;
    for (int gi = 0; gi < 2; ++gi) {
        const GridSpec& g = m.g[gi];
        Value::Object o;
        o["resolution"] = intArray({g.w, g.h});
        o["channels"] = Value(g.c);
        o["bits"] = Value(g.bits);
        o["sampling"] = Value(std::string(g.concat ? "concat2x2" : "bilinear"));
        Value::Array t;
        for (int i = 0; i < imagesPerGrid(g); ++i) t.emplace_back(tex++);
        o["textures"] = Value(std::move(t));
        grids.emplace_back(std::move(o));
    }
    p["grids"] = Value(std::move(grids));

    Value::Object net;
    net["inputs"] = Value(m.layers[0].in);
    net["hidden"] = intArray({kHidden, kHidden});
    net["outputs"] = Value(m.nNet);
    net["activation"] = Value(std::string("hardgelu"));
    Value::Array layout;
    for (const char* s : {"grid0", "grid1", "posenc", "lod"}) layout.emplace_back(std::string(s));
    net["inputLayout"] = Value(std::move(layout));
    Value::Object pe;
    pe["type"] = Value(std::string("triangle"));
    pe["periods"] = intArray({8, 4, 2});
    net["positionalEncoding"] = Value(std::move(pe));
    net["weightFormat"] = Value(std::string("float16"));
    Value::Array layers;
    for (const LayerSpec& l : m.layers) {
        Value::Object o;
        o["in"] = Value(l.in);
        o["inStride"] = Value(l.inStride);
        o["out"] = Value(l.out);
        layers.emplace_back(std::move(o));
    }
    net["layers"] = Value(std::move(layers));
    p["network"] = Value(std::move(net));

    Value::Array channels;
    for (const ChannelSpec& c : m.channels) {
        Value::Object o;
        if (c.isConst) {
            o["const"] = Value(double(c.value));
        } else {
            o["net"] = Value(c.net);
            o["activation"] = Value(std::string(c.sigmoid ? "sigmoid" : "none"));
        }
        channels.emplace_back(std::move(o));
    }
    p["channels"] = Value(std::move(channels));
    return p;
}

namespace {

struct Group {
    const char* name;
    int start, count;
};

// Output groups for the logical channel counts used by the golden cases (canonical v2 order).
std::vector<Group> groupsFor(int N) {
    if (N == 20) return {{"baseColor", 0, 4}, {"normal", 4, 2}, {"occlusion", 6, 1}, {"roughness", 7, 1}, {"metallic", 8, 1}, {"emissive", 9, 3}, {"diffuse", 12, 4}, {"specular", 16, 3}, {"glossiness", 19, 1}};
    if (N == 9) return {{"baseColor", 0, 4}, {"normal", 4, 2}, {"occlusion", 6, 1}, {"roughness", 7, 1}, {"metallic", 8, 1}};
    if (N == 4) return {{"baseColor", 0, 4}};
    return {{"occlusion", 0, N}};
}

std::vector<std::string> replacedFor(int N) {
    if (N == 20) return {"baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture", "emissiveTexture", "KHR_materials_pbrSpecularGlossiness.diffuseTexture", "KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture"};
    if (N == 9) return {"baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture"};
    if (N == 4) return {"baseColorTexture"};
    return {"occlusionTexture"};
}

} // namespace

bool writeV4Glb(const std::string& path, const Model& m, const std::string& name, std::string& err) {
    Document doc;
    auto& md = doc.model;
    md.asset.version = "2.0";
    md.asset.generator = "gltf-texture-compressor golden";

    // One quad.
    const float pos[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const float uv[] = {0, 1, 1, 1, 1, 0, 0, 0};
    const uint16_t idx[] = {0, 1, 2, 0, 2, 3};
    tinygltf::Buffer buf;
    auto append = [&](const void* p, size_t n) {
        const size_t off = buf.data.size();
        buf.data.insert(buf.data.end(), (const uint8_t*)p, (const uint8_t*)p + n);
        while (buf.data.size() % 4) buf.data.push_back(0);
        return off;
    };
    const size_t offPos = append(pos, sizeof(pos)), offUv = append(uv, sizeof(uv)), offIdx = append(idx, sizeof(idx));
    md.buffers.push_back(buf);
    auto view = [&](size_t off, size_t len, int target) {
        tinygltf::BufferView bv;
        bv.buffer = 0;
        bv.byteOffset = off;
        bv.byteLength = len;
        bv.target = target;
        md.bufferViews.push_back(bv);
        return int(md.bufferViews.size()) - 1;
    };
    auto accessor = [&](int bv, int comp, size_t count, int type) {
        tinygltf::Accessor a;
        a.bufferView = bv;
        a.componentType = comp;
        a.count = count;
        a.type = type;
        md.accessors.push_back(a);
        return int(md.accessors.size()) - 1;
    };
    const int aPos = accessor(view(offPos, sizeof(pos), 34962), TINYGLTF_COMPONENT_TYPE_FLOAT, 4, TINYGLTF_TYPE_VEC3);
    md.accessors[aPos].minValues = {-1, -1, 0};
    md.accessors[aPos].maxValues = {1, 1, 0};
    const int aUv = accessor(view(offUv, sizeof(uv), 34962), TINYGLTF_COMPONENT_TYPE_FLOAT, 4, TINYGLTF_TYPE_VEC2);
    const int aIdx = accessor(view(offIdx, sizeof(idx), 34963), TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, 6, TINYGLTF_TYPE_SCALAR);
    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = aPos;
    prim.attributes["TEXCOORD_0"] = aUv;
    prim.indices = aIdx;
    prim.material = 0;
    prim.mode = TINYGLTF_MODE_TRIANGLES;
    tinygltf::Mesh mesh;
    mesh.primitives.push_back(prim);
    md.meshes.push_back(mesh);
    tinygltf::Node node;
    node.mesh = 0;
    md.nodes.push_back(node);
    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    md.scenes.push_back(scene);
    md.defaultScene = 0;

    // Latent atlases and weights.
    const int nearest = doc.addSampler(9728, 9728, 10497, 10497);
    Value::Array latents;
    for (int gi = 0; gi < 2; ++gi) {
        auto imgs = packAtlas(m, gi);
        for (size_t i = 0; i < imgs.size(); ++i) {
            const std::string nm = name + "_latent_g" + std::to_string(gi) + "_" + std::to_string(i);
            const int img = doc.addImage(encodePng(imgs[i], 4), "image/png", nm);
            Value::Object ti;
            ti["index"] = Value(doc.addTexture(img, nearest, nm));
            latents.emplace_back(std::move(ti));
        }
    }
    const std::vector<uint8_t> wb = serializeWeights(m);
    Value::Object weights;
    weights["bufferView"] = Value(doc.addBufferView(wb, name + "_neural_weights"));
    weights["byteLength"] = Value(int(wb.size()));

    const int N = int(m.channels.size());
    Value::Object outputs;
    for (const Group& g : groupsFor(N)) outputs[g.name] = intArray({g.start, g.count});
    Value::Array replaced;
    for (const std::string& s : replacedFor(N)) replaced.emplace_back(s);

    Value::Object ext = v4Parameters(m);
    ext["version"] = Value(1);
    ext["codec"] = Value(std::string("babylon-ntc-v4"));
    ext["resolution"] = intArray({m.W, m.H});
    ext["latentTextures"] = Value(std::move(latents));
    ext["weights"] = Value(std::move(weights));
    ext["outputs"] = Value(std::move(outputs));
    ext["replacedSlots"] = Value(std::move(replaced));

    Image gray;
    gray.width = gray.height = 4;
    gray.rgba.assign(4 * 4 * 4, 128);
    for (size_t i = 3; i < gray.rgba.size(); i += 4) gray.rgba[i] = 255;
    const int fb = doc.addTexture(doc.addImage(encodePng(gray, 3), "image/png", name + "_fallback"), -1, name + "_fallback");
    tinygltf::Material mat;
    mat.name = name;
    mat.pbrMetallicRoughness.baseColorTexture.index = fb;
    mat.extensions["BABYLON_materials_neural_texture"] = Value(std::move(ext));
    md.materials.push_back(mat);
    doc.addExtensionUsed("BABYLON_materials_neural_texture", false);
    return saveGlb(std::move(doc), path, err);
}

// ---- golden cases -----------------------------------------------------------------------------------------------------

namespace {

bool writeFile(const fs::path& p, const void* data, size_t n, std::string& err) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(n));
    if (!f) {
        err = "failed to write " + p.string();
        return false;
    }
    return true;
}

} // namespace

bool writeGoldenCase(const std::string& dirStr, const std::string& name, const Model& model, uint32_t seed, std::string& err) {
    const fs::path dir = fs::path(dirStr) / name;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const int N = int(model.channels.size());
    std::vector<std::pair<std::string, std::string>> files; // name, sha256
    auto record = [&](const std::string& fname, const void* data, size_t n) {
        if (!writeFile(dir / fname, data, n, err)) return false;
        files.emplace_back(fname, sha256Hex(static_cast<const uint8_t*>(data), n));
        return true;
    };

    // The glb carries the packed form: read it back through pack/unpack so the references describe exactly what is stored.
    Model stored = model;
    for (int g = 0; g < 2; ++g) {
        std::vector<Image> imgs = packAtlas(model, g);
        if (!unpackAtlas(stored, g, imgs, err)) return false;
    }
    if (!parseWeights(serializeWeights(model).data(), weightBytes(model), stored, err)) return false;
    if (stored.q[0] != model.q[0] || stored.q[1] != model.q[1]) {
        err = "latent pack/unpack roundtrip failed";
        return false;
    }

    if (!writeV4Glb((dir / "case.glb").string(), model, name, err)) return false;
    {
        std::ifstream f(dir / "case.glb", std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        files.emplace_back("case.glb", sha256Hex(reinterpret_cast<const uint8_t*>(b.data()), b.size()));
    }

    RefDecoder ref(stored);
    std::vector<float> mip0;
    for (int m = 0; m < stored.M; ++m) {
        std::vector<float> out;
        ref.decodeMip(m, 1, out);
        if (!record("ref_m" + std::to_string(m) + ".f32", out.data(), out.size() * 4)) return false;
        if (m == 0) mip0 = out;
    }
    for (const Group& g : groupsFor(N)) {
        Image img;
        img.width = stored.W;
        img.height = stored.H;
        img.rgba.assign(size_t(stored.W) * stored.H * 4, 0);
        for (size_t t = 0; t < size_t(stored.W) * stored.H; ++t) {
            for (int c = 0; c < g.count && c < 3; ++c) img.rgba[t * 4 + c] = uint8_t(std::lround(255.0f * std::min(1.0f, std::max(0.0f, mip0[t * N + g.start + c]))));
            if (g.count == 4) img.rgba[t * 4 + 3] = uint8_t(std::lround(255.0f * std::min(1.0f, std::max(0.0f, mip0[t * N + g.start + 3]))));
            else img.rgba[t * 4 + 3] = 255;
        }
        const std::string fname = std::string("ref_") + g.name + "_m0.png";
        const auto png = encodePng(img, 4);
        if (!record(fname, png.data(), png.size())) return false;
    }
    for (int g = 0; g < 2; ++g) {
        const GridSpec& gs = stored.g[g];
        int aw, ah;
        atlasSize(gs, stored.L, aw, ah);
        for (int lv = 0; lv < stored.L; ++lv) {
            int w, h, ox, oy;
            levelSize(gs, lv, w, h);
            atlasOrigin(gs, lv, ox, oy);
            std::vector<uint8_t> q(size_t(w) * h * gs.c);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < gs.c; ++c) q[(size_t(y) * w + x) * gs.c + c] = stored.q[g][(size_t(oy + y) * aw + ox + x) * gs.c + c];
            if (!record("latents_g" + std::to_string(g) + "_lv" + std::to_string(lv) + ".u8", q.data(), q.size())) return false;
        }
    }
    for (int l = 0; l < 3; ++l) {
        const LayerSpec& ls = stored.layers[l];
        std::vector<float> f;
        for (int o = 0; o < ls.out; ++o)
            for (int k = 0; k < ls.in; ++k) f.push_back(stored.w[l][size_t(o) * ls.inStride + k]);
        for (int o = 0; o < ls.out; ++o) f.push_back(stored.b[l][o]);
        if (!record("weights_l" + std::to_string(l) + ".f32", f.data(), f.size() * 4)) return false;
    }

    // meta.json
    std::string j = "{\n";
    auto kv = [&](const std::string& k, const std::string& v, bool last = false) { j += "  \"" + k + "\": " + v + (last ? "\n" : ",\n"); };
    kv("case", "\"" + name + "\"");
    kv("resolution", "[" + std::to_string(stored.W) + ", " + std::to_string(stored.H) + "]");
    kv("mipCount", std::to_string(stored.M));
    kv("levels", std::to_string(stored.L));
    kv("seed", std::to_string(seed));
    kv("outputsCount", std::to_string(N));
    kv("networkOutputs", std::to_string(stored.nNet));
    std::string grids = "[";
    for (int g = 0; g < 2; ++g)
        grids += std::string(g ? ", " : "") + "{\"resolution\": [" + std::to_string(stored.g[g].w) + ", " + std::to_string(stored.g[g].h) + "], \"channels\": " + std::to_string(stored.g[g].c) +
                 ", \"bits\": " + std::to_string(stored.g[g].bits) + ", \"sampling\": \"" + (stored.g[g].concat ? "concat2x2" : "bilinear") + "\"}";
    kv("grids", grids + "]");
    std::string layers = "[";
    for (int l = 0; l < 3; ++l)
        layers += std::string(l ? ", " : "") + "{\"in\": " + std::to_string(stored.layers[l].in) + ", \"inStride\": " + std::to_string(stored.layers[l].inStride) + ", \"out\": " + std::to_string(stored.layers[l].out) + "}";
    kv("layers", layers + "]");
    std::string chans = "[";
    char nb[64];
    for (int n = 0; n < N; ++n) {
        const ChannelSpec& c = stored.channels[n];
        if (n) chans += ", ";
        if (c.isConst) {
            std::snprintf(nb, sizeof(nb), "%.9g", double(c.value));
            chans += std::string("{\"const\": ") + nb + "}";
        } else {
            chans += "{\"net\": " + std::to_string(c.net) + ", \"activation\": \"" + (c.sigmoid ? "sigmoid" : "none") + "\"}";
        }
    }
    kv("channels", chans + "]");
    std::string outs = "{";
    bool first = true;
    for (const Group& g : groupsFor(N)) {
        outs += std::string(first ? "" : ", ") + "\"" + g.name + "\": [" + std::to_string(g.start) + ", " + std::to_string(g.count) + "]";
        first = false;
    }
    kv("outputs", outs + "}");
    kv("fileLayout",
       "{\"ref_m<m>.f32\": \"float32 LE [j][i][n], decode(m,i,j) before normal-Z / colour conversion\", \"latents_g<k>_lv<j>.u8\": \"unpacked q, [y][x][c]\", "
       "\"weights_l<k>.f32\": \"float32 of the fp16 weights, [out][in] (no stride padding) followed by the bias\"}");
    j += "  \"sha256\": {\n";
    for (size_t i = 0; i < files.size(); ++i) j += "    \"" + files[i].first + "\": \"" + files[i].second + "\"" + (i + 1 < files.size() ? ",\n" : "\n");
    j += "  }\n}\n";
    return writeFile(dir / "meta.json", j.data(), j.size(), err);
}

// ---- synthetic data ---------------------------------------------------------------------------------------------------

namespace {

struct Rng {
    std::mt19937 g;
    explicit Rng(uint32_t s) : g(s) {}
    float u() { return float(g() >> 8) * (1.0f / 16777216.0f); } // [0,1), portable (no std distributions)
    float u11() { return 2.0f * u() - 1.0f; }
};

struct Branches {
    double neg = 0, mid = 0, pos = 0;
};

// Evaluates the activation branch fractions of hidden layer `layer` over the mip-0 texels.
Branches hiddenBranches(const Model& m, int layer) {
    RefDecoder ref(m);
    std::vector<float> x(128), a(kHidden), z(kHidden), prev(kHidden);
    Branches br;
    size_t n = 0;
    for (int j = 0; j < m.H; ++j)
        for (int i = 0; i < m.W; ++i) {
            ref.inputsOf(0, i, j, x.data());
            const float* in = x.data();
            for (int l = 0; l <= layer; ++l) {
                const LayerSpec& ls = m.layers[l];
                for (int o = 0; o < ls.out; ++o) {
                    float acc = m.b[l][o];
                    for (int k = 0; k < ls.in; ++k) acc += m.w[l][size_t(o) * ls.inStride + k] * in[k];
                    z[o] = acc;
                }
                if (l == layer) break;
                for (int o = 0; o < ls.out; ++o) a[o] = z[o] < -1.5f ? 0.0f : (z[o] > 1.5f ? z[o] : z[o] * (z[o] + 1.5f) / 3.0f);
                prev = a;
                in = prev.data();
            }
            for (int o = 0; o < kHidden; ++o) (z[o] < -1.5f ? br.neg : z[o] > 1.5f ? br.pos : br.mid) += 1.0;
            n += kHidden;
        }
    br.neg /= double(n), br.mid /= double(n), br.pos /= double(n);
    return br;
}

struct SynthSpec {
    const char* name;
    int W, H;
    int ratio; // texture size / G0 size
    int c0, b0, c1, b1;
    std::vector<ChannelSpec> channels;
};

bool buildSynth(const SynthSpec& s, uint32_t seed, Model& m, std::string& err, Branches* report) {
    m = Model();
    m.W = s.W;
    m.H = s.H;
    const int ratio = s.ratio;
    m.g[0] = {s.W / ratio, s.H / ratio, s.c0, s.b0, true};
    m.g[1] = {s.W / ratio / 2, s.H / ratio / 2, s.c1, s.b1, false};
    m.channels = s.channels;
    if (!m.finalize(err)) return false;
    Rng rng(seed);
    for (int g = 0; g < 2; ++g) {
        int aw, ah;
        atlasSize(m.g[g], m.L, aw, ah);
        const int levels = 1 << m.g[g].bits;
        // Only the cells of the levels are filled; unused atlas pixels stay 0.
        for (int lv = 0; lv < m.L; ++lv) {
            int w, h, ox, oy;
            levelSize(m.g[g], lv, w, h);
            atlasOrigin(m.g[g], lv, ox, oy);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < m.g[g].c; ++c) m.q[g][(size_t(oy + y) * aw + ox + x) * m.g[g].c + c] = uint8_t(std::min(levels - 1, int(rng.u() * levels)));
        }
    }
    // Weights ~ 1/sqrt(in) times a gain raised until every hardGELU branch holds >= 5% of the activations.
    const uint32_t wseed = rng.g();
    for (int l = 0; l < 3; ++l) {
        m.w[l].assign(size_t(m.layers[l].out) * m.layers[l].inStride, 0.0f);
        m.b[l].assign(m.layers[l].out, 0.0f);
    }
    for (int l = 0; l < 3; ++l) {
        const LayerSpec& ls = m.layers[l];
        const int tries = l < 2 ? 12 : 1;
        for (int t = 0; t < tries; ++t) {
            const float gain = l < 2 ? 1.0f + 0.35f * float(t) : 0.5f;
            Rng wr(wseed + uint32_t(l));
            m.w[l].assign(size_t(ls.out) * ls.inStride, 0.0f);
            m.b[l].assign(ls.out, 0.0f);
            for (int o = 0; o < ls.out; ++o)
                for (int k = 0; k < ls.in; ++k) m.w[l][size_t(o) * ls.inStride + k] = halfToFloat(floatToHalf(wr.u11() * 1.7320508f * gain / std::sqrt(float(ls.in))));
            for (int o = 0; o < ls.out; ++o) m.b[l][o] = halfToFloat(floatToHalf(l < 2 ? 0.3f * wr.u11() : 0.5f + 0.25f * wr.u11()));
            if (l == 2) break;
            const Branches br = hiddenBranches(m, l);
            if (report) report[l] = br;
            if (br.neg >= 0.05 && br.mid >= 0.05 && br.pos >= 0.05) break;
            if (t + 1 == tries) {
                err = "synthetic weights: hardGELU branch coverage below 5%";
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool writeSyntheticGoldens(const std::string& dir, std::string& err) {
    auto net = [](int n, bool sig) {
        ChannelSpec c;
        c.net = n;
        c.sigmoid = sig;
        return c;
    };
    auto cst = [](float v) {
        ChannelSpec c;
        c.isConst = true;
        c.value = v;
        return c;
    };
    std::vector<SynthSpec> cases;
    {
        std::vector<ChannelSpec> ch;
        int n = 0;
        for (int i = 0; i < 20; ++i) {
            if (i == 6) ch.push_back(cst(1.0f));
            else if (i == 8) ch.push_back(cst(0.0f));
            else if (i == 19) ch.push_back(cst(128.0f / 255.0f));
            else ch.push_back(net(n++, (i & 1) != 0));
        }
        cases.push_back({"synth_s", 64, 64, 4, 12, 4, 16, 4, ch});
    }
    cases.push_back({"synth_xs2b", 64, 64, 4, 8, 2, 12, 4, {net(0, false), net(1, false), net(2, false), net(3, false)}});
    cases.push_back({"synth_m_rect", 64, 32, 2, 12, 2, 10, 4, {net(0, true)}});
    {
        std::vector<ChannelSpec> ch;
        int n = 0;
        for (int i = 0; i < 9; ++i) ch.push_back(i == 6 ? cst(1.0f) : i == 8 ? cst(0.0f) : net(n++, true));
        cases.push_back({"synth_const", 32, 32, 4, 12, 4, 16, 4, ch});
    }
    for (const SynthSpec& s : cases) {
        Model m;
        Branches br[2];
        if (!buildSynth(s, 1234, m, err, br)) {
            err = std::string(s.name) + ": " + err;
            return false;
        }
        std::printf("  %s: %dx%d M=%d L=%d IN=%d N=%zu Nnet=%d, hardGELU branches l0 %.0f/%.0f/%.0f%% l1 %.0f/%.0f/%.0f%%\n", s.name, m.W, m.H, m.M, m.L, m.inputs(), m.channels.size(),
                    m.nNet, 100 * br[0].neg, 100 * br[0].mid, 100 * br[0].pos, 100 * br[1].neg, 100 * br[1].mid, 100 * br[1].pos);
        if (!writeGoldenCase(dir, s.name, m, 1234, err)) return false;
    }
    return true;
}

} // namespace gtc::v4
