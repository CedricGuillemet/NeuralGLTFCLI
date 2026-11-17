#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gtc {

// 8-bit image, always stored as RGBA.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;

    bool empty() const { return width == 0 || height == 0; }
    bool hasAlpha() const;
};

bool decodeImage(const std::vector<uint8_t>& encoded, Image& out, std::string& err);
std::vector<uint8_t> encodePng(const Image& img, int channels = 4);
bool writePng(const std::string& path, const Image& img, int channels = 4);

// Bilinear resample with wrap addressing (matches GL LINEAR + REPEAT).
Image resize(const Image& src, int width, int height);

// Full mip chain (level 0 = src) using a 2x2 box filter; averages in linear space when srgb is set.
std::vector<Image> buildMipChain(const Image& src, bool srgb);

std::string mimeFromBytes(const std::vector<uint8_t>& bytes);

} // namespace gtc
