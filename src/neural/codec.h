#pragma once

// Neural texture codec interface. A codec compresses the PBR texture set of one material
// into the BABYLON_materials_neural_texture extension. "nvidia-ntc" (the default) embeds the .ntc
// file produced by NVIDIA's ntc-cli in a bufferView. The legacy CUDA codecs "babylon-ntc-v1"
// (fixed 9 outputs), "babylon-ntc-v2" (variable outputs) and "babylon-ntc-v3" (configurable grids
// and MLP) store latent images plus network weights. New codecs implement INeuralCodec and are
// registered in createNeuralCodec (codec_factory.cpp).

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

enum class NeuralCodecVersion { V1, V2, V3 };

// Maps each active output channel i to its canonical channel.
struct NeuralLayout {
    int count = 0;
    int canonical[kMaxNeuralChannels]{};
};

struct NeuralMaterialInput {
    std::string name;
    int width = 0, height = 0;
    int outputs = 0;                                 // N, the number of active output channels
    NeuralLayout layout;                             // canonical channel of each of the N outputs
    std::vector<float> target;                       // height * width * N, values in [0,1], row 0 = top
    float lossWeights[kMaxNeuralChannels]{};         // per output channel, 0 for unused channels
    // Per-material scratch directory under the output directory (only set when the codec's
    // usesWorkDir() is true; created by the caller). Codecs must not use the system temp directory.
    std::string workDir;
    // width * height, nonzero = texel covered by the mesh UVs (dilated). Empty: sample everywhere.
    std::vector<uint8_t> uvMask;
};

struct NeuralMaterialOutput {
    std::vector<Image> latentImages;    // stored as RGBA8 PNG textures, in order
    std::vector<uint8_t> weights;       // stored in a bufferView, expectedWeightCount(N) floats
    tinygltf::Value::Object parameters; // codec specific fields merged into the extension
    std::vector<float> decoded;         // height * width * N reconstruction at texel centers
    // Compressed size of the material as the codec defines it; 0 means latent PNG bytes + weight bytes.
    size_t compressedBytes = 0;
    // The codec has no glTF representation: no extension and no placeholders are written, the
    // material's slots are replaced by plain PNG textures built from `decoded`.
    bool bakeOnly = false;
    // Container bitstream (nvidia-ntc): when non-empty it is stored unchanged in a bufferView referenced by
    // the extension's bitstream.bufferView, and no latent textures or weights are written.
    std::vector<uint8_t> bitstream;
    tinygltf::Value::Object ntcInfo; // informative "ntc" object of the extension (bitstream codecs)
    // Optional names of the latent images (suffix of the glTF image/texture names), same order as latentImages.
    std::vector<std::string> latentNames;
    // Latent textures are sampled with NEAREST and no mipmaps (the shader interpolates manually).
    bool latentPointSampling = false;
    // `weights` holds fp16 bytes (babylon-ntc-v4): the extension's weights reference is {bufferView, byteLength}.
    bool weightsFp16 = false;
    // Codec specific numbers for the metrics sidecar (e.g. the tool's own PSNR).
    tinygltf::Value::Object extraMetrics;
};

class INeuralCodec {
public:
    virtual ~INeuralCodec() = default;
    virtual std::string name() const = 0;
    // Value of the extension's "codec" field (the versioned id; defaults to name()).
    virtual std::string codecId() const { return name(); }
    virtual std::string deviceDescription() const = 0;
    // Weight floats expected in NeuralMaterialOutput::weights for N outputs (skipped when bakeOnly).
    // Largest number of output channels one material can have; 0 = no limit beyond the canonical set.
    virtual int maxOutputChannels() const { return 0; }
    virtual size_t expectedWeightCount(int outputs) const { return size_t(1856) + 33 * size_t(outputs); }
    // True if encode() uses NeuralMaterialInput::workDir.
    virtual bool usesWorkDir() const { return false; }
    // Full parameter set, written to the metrics sidecar.
    virtual tinygltf::Value::Object describeParameters() const { return {}; }
    virtual bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) = 0;
};

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err);
// Only built with GTC_WITH_NVIDIA_NTC=ON (src/neural/nvidia_ntc.cpp).
std::unique_ptr<INeuralCodec> createNvidiaNtcCodec(const NeuralOptions& opts, std::string& err);
// Only built with GTC_WITH_CUDA_CODEC=ON (src/neural/babylon_ntc.cpp).
std::unique_ptr<INeuralCodec> createBabylonNtcCodec(const NeuralOptions& opts, std::string& err);
// Only built with GTC_WITH_CUDA_CODEC=ON (src/neural/babylon_ntc_v4.cpp).
std::unique_ptr<INeuralCodec> createBabylonNtcV4Codec(const NeuralOptions& opts, std::string& err);

} // namespace gtc
