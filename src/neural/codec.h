#pragma once

// Neural texture codec interface. A codec compresses the PBR texture set of one material
// into latent images plus network weights stored in the BABYLON_materials_neural_texture
// extension. "babylon-ntc-v1" (fixed 9 outputs) and "babylon-ntc-v2" (variable outputs) are
// implemented; others (e.g. NVIDIA RTXNTC) can be added by implementing INeuralCodec and
// registering it in createNeuralCodec.

#include <memory>
#include <string>
#include <vector>

#include "../image.h"
#include "tiny_gltf.h"

namespace gtc {

struct NeuralOptions;

// Canonical output channels shared by the babylon-ntc codecs. A material only uses the channels
// it needs: v2 packs the active ones to the front in canonical order, v1 always uses 0..8.
enum NeuralChannel {
    kChBaseColor = 0,  // 0-3 RGBA (A only when alphaMode != OPAQUE in v2)
    kChNormal = 4,     // 4-5 XY
    kChOcclusion = 6,
    kChRoughness = 7,
    kChMetallic = 8,
    kChEmissive = 9,   // 9-11 RGB
    kChDiffuse = 12,   // 12-15 RGBA, KHR_materials_pbrSpecularGlossiness diffuse
    kChSpecular = 16,  // 16-18 RGB
    kChGlossiness = 19 // specularGlossiness alpha
};
constexpr int kMaxNeuralChannels = 20;
constexpr int kV1NeuralChannels = 9;

enum class NeuralCodecVersion { V1, V2 };

// Maps each active output channel i to its canonical channel.
struct NeuralLayout {
    int count = 0;
    int canonical[kMaxNeuralChannels]{};
};

struct NeuralMaterialInput {
    std::string name;
    int width = 0, height = 0;
    int outputs = 0;                                 // N, the number of active output channels
    std::vector<float> target;                       // height * width * N, values in [0,1], row 0 = top
    float lossWeights[kMaxNeuralChannels]{};         // per output channel, 0 for unused channels
};

struct NeuralMaterialOutput {
    std::vector<Image> latentImages;    // stored as RGBA8 PNG textures, in order
    std::vector<uint8_t> weights;       // stored in a bufferView, 1856 + 33 * N floats
    tinygltf::Value::Object parameters; // codec specific fields merged into the extension
    std::vector<float> decoded;         // height * width * N reconstruction at texel centers
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
