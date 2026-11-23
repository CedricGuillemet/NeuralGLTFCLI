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

} // namespace gtc
