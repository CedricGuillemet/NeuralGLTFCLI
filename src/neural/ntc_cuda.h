#pragma once

// CUDA implementation of the "babylon-ntc" neural texture network: bilinear latent grids at fractions
// of the texture resolution plus a sin/cos positional encoding feed a small MLP.
//   v1 / v2: two 8-channel 8-bit grids (1/4 and 1/16), 24 -> 32 -> 32 -> N ReLU MLP.
//   v3:      up to three grids with 4- or 8-bit channels, wider / deeper MLP, SiLU or HGELU activation.
// Training always runs with kOutputs (20) output channels; channels with a loss weight of 0 never produce a
// gradient, so the hidden layers train exactly as with an N-output network and only the first N columns of
// the last layer are exported.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gtc::ntc {

constexpr int kOutputs = 20;
constexpr int kPosEncInputs = 8; // sin/cos at 2 frequencies, both axes
constexpr int kMaxGrids = 3;

enum Activation { kRelu = 0, kSilu = 1, kHgelu = 2 };
enum LossKind { kLossL2 = 0, kLossCharbonnier = 1 };

struct GridDesc {
    int div = 4;      // grid resolution = ceil(texture size / div)
    int channels = 8; // multiple of 4
    int bits = 8;     // 8 or 4 (quantization levels = 2^bits - 1)
};

struct NetConfig {
    std::vector<GridDesc> grids{{4, 8, 8}, {16, 8, 8}};
    int hidden = 32;
    int hiddenLayers = 2; // hidden layers (the last layer comes on top)
    int activation = kRelu;

    int inputs() const {
        int n = kPosEncInputs;
        for (const GridDesc& g : grids) n += g.channels;
        return n;
    }
    // Weight floats with `outputs` columns in the last layer: sum over layers of (in + 1) * out.
    size_t weightCount(int outputs) const {
        const size_t in = size_t(inputs()), h = size_t(hidden), l = size_t(hiddenLayers);
        return (in + 1) * h + (l - 1) * (h + 1) * h + (h + 1) * size_t(outputs);
    }
    // True for the layout understood by babylon-ntc-v1 / v2 loaders.
    bool isV2Shape() const {
        return grids.size() == 2 && grids[0].channels == 8 && grids[1].channels == 8 && grids[0].bits == 8 &&
               grids[1].bits == 8 && hidden == 32 && hiddenLayers == 2 && activation == kRelu;
    }
};

struct TrainConfig {
    NetConfig net;
    int steps = 6000;
    int batch = 1 << 17;
    float lrLatent = 1e-2f;
    float lrMlp = 5e-3f;
    float lrMin = 1e-4f;
    float lrFinetune = 1e-3f;
    float lrFinetuneEnd = -1.0f; // < 0: constant lrFinetune; otherwise cosine lrFinetune -> lrFinetuneEnd
    int finetuneSteps = -1;      // < 0: steps / 4
    float warmup = 0.0f;         // fraction of the training steps with a linear learning-rate ramp
    float beta2 = 0.999f;
    bool ste = false;            // straight-through quantization instead of additive noise
    float texelP = 0.0f;         // probability of texel-center samples (exact target) instead of continuous UVs
    float importance = 0.0f;     // 0..1 blend of uniform and gradient-energy texel sampling
    int loss = kLossL2;
    float charbonnierEps = 0.003f;
    const std::vector<uint8_t>* uvMask = nullptr; // width * height, nonzero = texel used by the mesh
    float maskInside = 0.98f;                      // fraction of the samples drawn from inside the mask
    uint32_t seed = 1;
    int device = 0;
};

struct GridData {
    int w = 0, h = 0, channels = 0, bits = 8, div = 4;
    std::vector<uint8_t> q; // h * w * channels quantization levels (0 .. 2^bits - 1)
};

struct Model {
    int width = 0, height = 0;
    NetConfig net;
    std::vector<GridData> grids;
    std::vector<float> weights; // net.weightCount(kOutputs) floats, last layer with kOutputs columns
    // Training diagnostics: PSNR (dB, active channels) of the full texture before and after latent
    // quantization and after the final fine-tune.
    float psnrFloat = 0, psnrQuant = 0, psnrFinal = 0;
};

// phase: 0 = training, 1 = fine-tuning after latent quantization.
using Progress = std::function<void(int phase, int step, int total, float loss)>;

// target: height * width * kOutputs floats in [0,1] (row 0 = top of the image). Only the first
// `outputs` channels are initialized and normalized as active; the others must have a loss weight of 0.
bool train(const std::vector<float>& target, int width, int height, int outputs, const float lossWeights[kOutputs],
           const TrainConfig& cfg, Model& out, const Progress& progress, std::string& err);

// Decodes every texel center; out: height * width * kOutputs floats.
bool decode(const Model& model, int device, std::vector<float>& out, std::string& err);

// True if train()/decode() have a kernel for this network shape.
bool supported(const NetConfig& net, std::string& err);

std::string deviceName(int device);

} // namespace gtc::ntc
