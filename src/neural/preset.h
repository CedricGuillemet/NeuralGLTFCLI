#pragma once

// Neural training presets and the flags that override them (--neural-preset, --neural-grids, ...).

#include <string>

#include "ntc_cuda.h"

namespace gtc {

struct NeuralOptions;

struct NeuralPreset {
    std::string name;
    ntc::TrainConfig cfg;                    // uvMask is filled in by the caller
    bool uvMask = false;                     // restrict sampling to the texels used by the mesh
    float lossWeight[ntc::kOutputs];         // < 0: keep the codec default of the channel
    NeuralPreset() {
        for (float& w : lossWeight) w = -1.0f;
    }
};

// Preset first, then every explicitly given flag on top.
bool resolveNeuralPreset(const NeuralOptions& opts, NeuralPreset& out, std::string& err);

// babylon-ntc-v4 training recipe: the presets v4xs/v4s/v4m/v4l of the spec plus the flags that override them.
struct V4Preset {
    std::string name;
    int div0 = 4, c0 = 12, b0 = 4;     // G0: texture / div0, channels, bits (G1 is always G0 / 2)
    int c1 = 16, b1 = 4;
    int stepsA = 40000, stepsB = 2000, batch = 1 << 16;
    int activation = 0;                // trainer: 0 hardGELU, 1 hard-swish, 2 SiLU (files are always hardGELU)
    float lrLatent = 0.01f, lrMlp = 0.005f, lrB = 0.001f;
    float initLatent = 0.0f, uniformMip = 0.05f, inside = 0.98f, noise = 1.0f;
    int levelCap = 64;
    bool constElision = true;
    bool noRef = false;                // experiments with a non-standard activation: skip the reference decode
    bool sigmoid[ntc::kOutputs]{};     // per canonical channel: sigmoid output activation (else clamp)
    uint32_t seed = 1;
    int device = 0;
};
bool resolveV4Preset(const NeuralOptions& opts, V4Preset& out, std::string& err);

} // namespace gtc
