#include "image.h"

#include <algorithm>
#include <cmath>

#include "stb_image.h"
#include "stb_image_write.h"

namespace gtc {

bool Image::hasAlpha() const {
    for (size_t i = 3; i < rgba.size(); i += 4) {
        if (rgba[i] != 255) return true;
    }
    return false;
}

bool decodeImage(const std::vector<uint8_t>& encoded, Image& out, std::string& err) {
    int w = 0, h = 0, n = 0;
    stbi_uc* data = stbi_load_from_memory(encoded.data(), int(encoded.size()), &w, &h, &n, 4);
    if (!data) {
        err = std::string("cannot decode image (") + mimeFromBytes(encoded) + "): " + stbi_failure_reason();
        return false;
    }
    out.width = w;
    out.height = h;
    out.rgba.assign(data, data + size_t(w) * h * 4);
    stbi_image_free(data);
    return true;
}

static void appendBytes(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
    v->insert(v->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
}

static std::vector<uint8_t> packChannels(const Image& img, int channels) {
    if (channels == 4) return img.rgba;
    std::vector<uint8_t> px(size_t(img.width) * img.height * channels);
    for (size_t i = 0, n = size_t(img.width) * img.height; i < n; ++i)
        for (int c = 0; c < channels; ++c) px[i * channels + c] = img.rgba[i * 4 + c];
    return px;
}

std::vector<uint8_t> encodePng(const Image& img, int channels) {
    std::vector<uint8_t> out;
    auto px = packChannels(img, channels);
    stbi_write_png_to_func(appendBytes, &out, img.width, img.height, channels, px.data(), img.width * channels);
    return out;
}

bool writePng(const std::string& path, const Image& img, int channels) {
    auto px = packChannels(img, channels);
    return stbi_write_png(path.c_str(), img.width, img.height, channels, px.data(), img.width * channels) != 0;
}

Image resize(const Image& src, int width, int height) {
    if (src.width == width && src.height == height) return src;
    Image dst;
    dst.width = width;
    dst.height = height;
    dst.rgba.resize(size_t(width) * height * 4);
    for (int y = 0; y < height; ++y) {
        float sy = (y + 0.5f) * src.height / height - 0.5f;
        int y0 = int(std::floor(sy));
        float fy = sy - y0;
        int ya = ((y0 % src.height) + src.height) % src.height, yb = (ya + 1) % src.height;
        for (int x = 0; x < width; ++x) {
            float sx = (x + 0.5f) * src.width / width - 0.5f;
            int x0 = int(std::floor(sx));
            float fx = sx - x0;
            int xa = ((x0 % src.width) + src.width) % src.width, xb = (xa + 1) % src.width;
            for (int c = 0; c < 4; ++c) {
                auto at = [&](int xx, int yy) { return float(src.rgba[(size_t(yy) * src.width + xx) * 4 + c]); };
                float v = (at(xa, ya) * (1 - fx) + at(xb, ya) * fx) * (1 - fy) + (at(xa, yb) * (1 - fx) + at(xb, yb) * fx) * fy;
                dst.rgba[(size_t(y) * width + x) * 4 + c] = uint8_t(std::clamp(v + 0.5f, 0.0f, 255.0f));
            }
        }
    }
    return dst;
}

static float srgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
static float linearToSrgb(float c) { return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f; }

std::vector<Image> buildMipChain(const Image& src, bool srgb) {
    float toLinear[256];
    for (int i = 0; i < 256; ++i) toLinear[i] = srgb ? srgbToLinear(i / 255.0f) : i / 255.0f;

    std::vector<Image> chain{src};
    while (chain.back().width > 1 || chain.back().height > 1) {
        const Image& s = chain.back();
        Image d;
        d.width = std::max(1, s.width / 2);
        d.height = std::max(1, s.height / 2);
        d.rgba.resize(size_t(d.width) * d.height * 4);
        for (int y = 0; y < d.height; ++y) {
            int y0 = std::min(y * 2, s.height - 1), y1 = std::min(y * 2 + 1, s.height - 1);
            for (int x = 0; x < d.width; ++x) {
                int x0 = std::min(x * 2, s.width - 1), x1 = std::min(x * 2 + 1, s.width - 1);
                const uint8_t* p[4] = {&s.rgba[(size_t(y0) * s.width + x0) * 4], &s.rgba[(size_t(y0) * s.width + x1) * 4],
                                       &s.rgba[(size_t(y1) * s.width + x0) * 4], &s.rgba[(size_t(y1) * s.width + x1) * 4]};
                uint8_t* o = &d.rgba[(size_t(y) * d.width + x) * 4];
                for (int c = 0; c < 4; ++c) {
                    if (c < 3) {
                        float v = (toLinear[p[0][c]] + toLinear[p[1][c]] + toLinear[p[2][c]] + toLinear[p[3][c]]) * 0.25f;
                        if (srgb) v = linearToSrgb(v);
                        o[c] = uint8_t(std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f));
                    } else {
                        o[c] = uint8_t((p[0][c] + p[1][c] + p[2][c] + p[3][c] + 2) / 4);
                    }
                }
            }
        }
        chain.push_back(std::move(d));
    }
    return chain;
}

std::string mimeFromBytes(const std::vector<uint8_t>& b) {
    auto starts = [&](std::initializer_list<uint8_t> sig, size_t off = 0) {
        if (b.size() < off + sig.size()) return false;
        size_t i = off;
        for (uint8_t s : sig)
            if (b[i++] != s) return false;
        return true;
    };
    if (starts({0x89, 'P', 'N', 'G'})) return "image/png";
    if (starts({0xFF, 0xD8, 0xFF})) return "image/jpeg";
    if (starts({0xAB, 'K', 'T', 'X', ' ', '2', '0'})) return "image/ktx2";
    if (starts({'R', 'I', 'F', 'F'}) && starts({'W', 'E', 'B', 'P'}, 8)) return "image/webp";
    return "application/octet-stream";
}

} // namespace gtc
