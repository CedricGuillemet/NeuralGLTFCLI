#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "tiny_gltf.h"

namespace gtc {

// A glTF model whose images are held as raw encoded bytes outside of the buffers,
// so that passes can freely replace or add images. Buffers are repacked on save.
struct Document {
    tinygltf::Model model;
    std::vector<std::vector<uint8_t>> imageData; // parallel to model.images
    std::vector<std::vector<uint8_t>> extraViews; // new bufferViews, see addBufferView

    int addImage(std::vector<uint8_t> bytes, const std::string& mime, const std::string& name);
    int addTexture(int imageIndex, int sampler, const std::string& name);
    int addSampler(int magFilter, int minFilter, int wrapS, int wrapT);
    // Adds a bufferView holding the given bytes; returns its index.
    int addBufferView(std::vector<uint8_t> bytes, const std::string& name);
    void addExtensionUsed(const std::string& ext, bool required);
};

struct LoadResult {
    uint64_t inputBytes = 0; // input file plus external resources
};

bool loadDocument(const std::string& path, Document& doc, LoadResult& info, std::string& err);
// Removes images/textures/samplers no longer referenced, repacks a single buffer and writes a GLB.
bool saveGlb(Document doc, const std::string& path, std::string& err);

// Calls fn(slot, textureIndex) for every texture referenced by a material: core slots
// ("baseColorTexture", "normalTexture"...) and texture infos inside material extensions.
void forEachMaterialTexture(tinygltf::Material& m, const std::function<void(const std::string& slot, int& index)>& fn);

} // namespace gtc
