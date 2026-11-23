#pragma once

// CUDA implementation of the "babylon-ntc-v1" / "babylon-ntc-v2" neural texture network:
// two bilinear latent grids (8 channels each, at 1/4 and 1/16 resolution) plus a
// sin/cos positional encoding feed a 24 -> 32 -> 32 -> N ReLU MLP. Training always runs with
// kOutputs (20) output channels; channels with a loss weight of 0 never produce a gradient, so
// the hidden layers train exactly as with an N-output network and only the first N columns of
// the last layer are exported.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gtc::ntc {

constexpr int kLatentChannels = 8;
constexpr int kInputs = 24;
constexpr int kHidden = 32;
constexpr int kOutputs = 20;
constexpr int kHiDiv = 4;
constexpr int kLoDiv = 16;
// Weight layout: W0[in][hidden], b0, W1[hidden][hidden], b1, W2[hidden][out], b2.
constexpr int kOffW0 = 0;
constexpr int kOffB0 = kOffW0 + kInputs * kHidden;
constexpr int kOffW1 = kOffB0 + kHidden;
constexpr int kOffB1 = kOffW1 + kHidden * kHidden;
constexpr int kOffW2 = kOffB1 + kHidden;
constexpr int kOffB2 = kOffW2 + kHidden * kOutputs;
constexpr int kParams = kOffB2 + kOutputs;

struct TrainConfig {
    int steps = 6000;
    int batch = 1 << 17;
    float lrLatent = 1e-2f;
    float lrMlp = 5e-3f;
    float lrMin = 1e-4f;
    float lrFinetune = 1e-3f;
    uint32_t seed = 1;
    int device = 0;
};

struct Model {
    int width = 0, height = 0;
    int hiW = 0, hiH = 0, loW = 0, loH = 0;
    std::vector<uint8_t> hi; // hiH * hiW * 8, quantized latents
    std::vector<uint8_t> lo; // loH * loW * 8
    std::vector<float> weights; // kParams, last layer with kOutputs columns
};

// phase: 0 = training, 1 = fine-tuning after latent quantization.
using Progress = std::function<void(int phase, int step, int total, float loss)>;

// target: height * width * kOutputs floats in [0,1] (row 0 = top of the image). Only the first
// `outputs` channels are initialized and normalized as active; the others must have a loss weight of 0.
bool train(const std::vector<float>& target, int width, int height, int outputs, const float lossWeights[kOutputs],
           const TrainConfig& cfg, Model& out, const Progress& progress, std::string& err);

// Decodes every texel center; out: height * width * kOutputs floats.
bool decode(const Model& model, int device, std::vector<float>& out, std::string& err);

std::string deviceName(int device);

} // namespace gtc::ntc
