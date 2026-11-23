#pragma once

// babylon-ntc-v4 shared definitions: pyramid geometry, latent packing, fp16 weights, the reference decoder.
// Clean-room NTC-style codec after Vaidyanathan et al. 2023 (arXiv 2305.17105); the normative text is the
// "Codec babylon-ntc-v4" section of docs/BABYLON_materials_neural_texture.md.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "../image.h"
#include "tiny_gltf.h"

namespace gtc::v4 {

constexpr int kHidden = 64;
constexpr int kPeValues = 12; // 3 octaves (periods 8, 4, 2) x {cos-like, sin-like} x 2 axes

struct GridSpec {
    int w = 0, h = 0;    // level 0 size
    int c = 0;           // channels
    int bits = 4;        // 2 or 4
    bool concat = false; // true: 4 raw taps (G0), false: bilinear (G1)
};

struct LayerSpec {
    int in = 0, inStride = 0, out = 0;
};

struct ChannelSpec {
    bool isConst = false;
    float value = 0.0f; // const
    int net = -1;       // network output index
    bool sigmoid = false;
};

inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

inline int mipCountOf(int w, int h) {
    int m = 1, s = std::max(w, h);
    while (s > 1) s >>= 1, ++m;
    return m;
}

// Largest L such that w1 % 4^(L-1) == 0 and h1 % 4^(L-1) == 0.
inline int levelCountOf(int w1, int h1) {
    int l = 1;
    while (l < 15 && w1 % (1 << (2 * l)) == 0 && h1 % (1 << (2 * l)) == 0) ++l;
    return l;
}

inline int levelOf(int m, int L) { return std::min(L - 1, std::max(0, (m >> 1) - 1)); }

inline void levelSize(const GridSpec& g, int lv, int& w, int& h) {
    w = g.w >> (2 * lv);
    h = g.h >> (2 * lv);
}

inline void atlasOrigin(const GridSpec& g, int lv, int& x, int& y) {
    x = y = 0;
    if (lv == 0) return;
    x = g.w;
    for (int k = 1; k < lv; ++k) y += g.h >> (2 * k);
}

inline void atlasSize(const GridSpec& g, int L, int& w, int& h) {
    w = L > 1 ? g.w + (g.w >> 2) : g.w;
    h = g.h;
}

inline int imagesPerGrid(const GridSpec& g) { return ceilDiv(g.c * g.bits, 32); }
inline int zeroQ(int bits) { return (1 << (bits - 1)) - 1; }
inline float latentValue(int q, int bits) { return float(q - zeroQ(bits)) / float(1 << bits); }

// IEEE binary16 <-> float32 (round to nearest even).
float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);

inline int inputCount(const GridSpec& g0, const GridSpec& g1) { return 4 * g0.c + g1.c + kPeValues + 1; }

// Everything the decoder needs. q[g] holds the unpacked latent atlas, [y][x][c], size atlasW * atlasH * c.
struct Model {
    int W = 0, H = 0, M = 0, L = 0;
    GridSpec g[2];
    std::vector<uint8_t> q[2];
    LayerSpec layers[3];
    std::vector<float> w[3]; // [out][inStride], values of the fp16 weights
    std::vector<float> b[3];
    std::vector<ChannelSpec> channels;
    int nNet = 0;

    int inputs() const { return inputCount(g[0], g[1]); }
    // Fills W,H,M,L, layer strides and the net output count from the grids and channels.
    bool finalize(std::string& err);
};

std::vector<uint8_t> serializeWeights(const Model& m);                                       // fp16 bytes, spec layout
bool parseWeights(const uint8_t* data, size_t size, Model& m, std::string& err);            // into m.w / m.b
size_t weightBytes(const Model& m);

// Latent atlases <-> RGBA8 images: ceil(C*B/32) images per grid.
std::vector<Image> packAtlas(const Model& m, int g);
bool unpackAtlas(Model& m, int g, const std::vector<Image>& images, std::string& err);

// Reference decoder: spec 1.2 verbatim (float32, fixed summation order, no FMA contraction).
class RefDecoder {
public:
    explicit RefDecoder(const Model& m);
    int channels() const { return int(m_.channels.size()); }
    // out: N logical channels (constants filled in, before normal Z / colour conversion).
    void decodeTexel(int m, int i, int j, float* out) const;
    // out: Hm * Wm * N, [j][i][n].
    void decodeMip(int m, int threads, std::vector<float>& out) const;
    // Network inputs of a texel (length inputs()), exposed for tests.
    void inputsOf(int m, int i, int j, float* x) const;

private:
    void net(const float* x, float* y) const;
    const Model& m_;
    std::vector<float> wt_[3]; // transposed [k][out]
};

// Case writer shared by the golden generator: quad glb with the extension and fallback textures.
bool writeV4Glb(const std::string& path, const Model& m, const std::string& name, std::string& err);
tinygltf::Value::Object v4Parameters(const Model& m);

// Hidden --neural-golden: writes the synthetic cases of the spec into dir.
bool writeSyntheticGoldens(const std::string& dir, std::string& err);
// Writes the files of one case (everything but case.glb is derived from the model).
bool writeGoldenCase(const std::string& dir, const std::string& name, const Model& m, uint32_t seed, std::string& err);

std::string sha256Hex(const uint8_t* data, size_t size);

} // namespace gtc::v4
