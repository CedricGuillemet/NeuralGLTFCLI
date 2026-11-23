#pragma once

// babylon-ntc-v4 CUDA trainer (clean-room, after Vaidyanathan et al. 2023, arXiv 2305.17105; ours).
// Plain-C++ interface: no tinygltf / codec headers so that it can be compiled by nvcc.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gtc::v4cuda {

constexpr int kMaxLevels = 16;
constexpr int kMaxMips = 16;

struct Shape {
    int W = 0, H = 0, M = 0, L = 0;
    int gw[2]{}, gh[2]{}, gc[2]{}, gbits[2]{}; // level 0 size, channels, bits of both grids
    int nIn = 0, stride0 = 0;                  // network inputs, padded to a multiple of 4
    int nNet = 0;                              // network outputs
    bool sigmoid[32]{};                        // per network output
};

// Latents are stored level after level (level 0 first), each level [y][x][c], one byte per value.
inline size_t levelCells(const Shape& s, int g, int lv) { return size_t(s.gw[g] >> (2 * lv)) * size_t(s.gh[g] >> (2 * lv)) * size_t(s.gc[g]); }
inline size_t gridCells(const Shape& s, int g) {
    size_t n = 0;
    for (int lv = 0; lv < s.L; ++lv) n += levelCells(s, g, lv);
    return n;
}

struct Params {
    int stepsA = 40000, stepsB = 2000;
    int batch = 1 << 16;
    int device = 0;
    uint32_t seed = 1;
    float lrLatent = 0.01f, lrMlp = 0.005f, lrB = 0.001f;
    int activation = 0;           // 0 hardGELU (paper), 1 hard-swish, 2 SiLU
    float uniformMipP = 0.05f;    // fraction of steps that pick the mip uniformly
    float insideP = 0.98f;        // fraction of samples taken inside the UV mask
    float initLatent = 0.0f;      // latents start at U(-x, x)
    float noiseScale = 1.0f;      // training noise amplitude in units of the quantisation step
};

struct Weights {
    std::vector<float> w[3]; // [out][inStride]
    std::vector<float> b[3];
};

struct Result {
    std::vector<uint8_t> q[2]; // quantised latents
    Weights net;               // fp32 trained weights
    std::vector<double> mseFloat, mseQuantPre, mseFinal; // masked mip-0 MSE per network output
    float finalLoss = 0;
    double seconds = 0;
};

using Progress = std::function<void(int phase, int step, int total, float loss)>;

// target: net channels only, [mip 0][y][x][nNet], values in [0,1]. mask: W*H, empty = everywhere.
bool train(const Shape& s, const std::vector<float>& target, const float* lossWeights, const std::vector<uint8_t>& mask, const Params& p, Result& out,
           const Progress& progress, std::string& err);

// GPU forward pass of a finished model (quantised latents, given weights): samples = {m,i,j} triples; out = nNet values
// per sample after the output activation (sigmoid or clamp).
bool decodeSamples(const Shape& s, const std::vector<uint8_t> q[2], const Weights& net, int activation, int device, const std::vector<int>& samples, std::vector<float>& out,
                   std::string& err);

std::string deviceName(int device);

} // namespace gtc::v4cuda
