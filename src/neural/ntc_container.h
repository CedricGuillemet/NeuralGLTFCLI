#pragma once

// Reader and validator for the NTC texture set container written by NVIDIA's ntc-cli (public format description:
// docs/TextureSetFile.md of the RTXNTC repository). Only what a decoder in the glTF extension needs is accepted:
// uncompressed views, no BCn data, no stored raw colors and Linear channels.

#include <cstdint>
#include <string>
#include <vector>

namespace gtc {

struct NtcTextureInfo {
    std::string name;
    int firstChannel = 0;
    int numChannels = 0;
};

struct NtcContainerInfo {
    uint32_t containerVersion = 0;
    int schemaVersion = 0;
    int width = 0, height = 0;
    int numChannels = 0;
    int numFeatures = 0;
    int numColorMips = 1;
    int gridSizeScale = 0;   // derived from the color and latent image sizes
    int numLatentImages = 0;
    int numViews = 0;
    int numMlpVersions = 0;
    size_t jsonBytes = 0;    // size of the JSON descriptor chunk (padding included)
    size_t dataBytes = 0;    // size of the binary data chunk
    std::vector<NtcTextureInfo> textures;
};

constexpr uint32_t kNtcContainerVersion = 0x100;
// schemaVersion written by LibNTC 0.10.x. The public document still says 1; the files carry 3.
constexpr int kNtcSchemaVersion010 = 3;

// Parses and validates a container. A failed check returns false with the offending field in err.
bool parseNtcContainer(const std::vector<uint8_t>& file, NtcContainerInfo& info, std::string& err);

} // namespace gtc
