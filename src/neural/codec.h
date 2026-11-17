#pragma once

// Neural texture codec interface. A codec compresses the PBR texture set of one material
// into latent images plus network weights stored in the BABYLON_materials_neural_texture
// extension. "babylon-ntc-v1" is the only codec today; others (e.g. NVIDIA RTXNTC) can be
// added by implementing INeuralCodec and registering it in createNeuralCodec.

#include <memory>
#include <string>
#include <vector>

#include "../image.h"
#include "tiny_gltf.h"

namespace gtc {

struct NeuralOptions;

// Output channels shared by all codecs: base RGBA, normal XY, occlusion, roughness, metallic.
constexpr int kNeuralChannels = 9;

struct NeuralMaterialInput {
    std::string name;
    int width = 0, height = 0;
    std::vector<float> target;            // height * width * 9, values in [0,1], row 0 = top
    float lossWeights[kNeuralChannels]{}; // 0 for channels the material does not use
};

struct NeuralMaterialOutput {
    std::vector<Image> latentImages;    // stored as RGBA8 PNG textures, in order
    std::vector<uint8_t> weights;       // stored in a bufferView
    tinygltf::Value::Object parameters; // codec specific fields merged into the extension
    std::vector<float> decoded;         // height * width * 9 reconstruction at texel centers
};

class INeuralCodec {
public:
    virtual ~INeuralCodec() = default;
    virtual std::string name() const = 0;
    virtual std::string deviceDescription() const = 0;
    virtual bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) = 0;
};

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err);

} // namespace gtc
