#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace gtc {

// UTF-8 std::string <-> std::filesystem::path (C++20 u8string uses char8_t).
inline std::filesystem::path toPath(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }
inline std::string fromPath(const std::filesystem::path& p) {
    auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

struct KtxOptions {
    int etc1sQuality = 255;   // 1..255, ETC1S for color textures
    int uastcLevel = 2;       // 0..4, UASTC for linear data (normals, ORM...)
    float uastcRdo = 1.0f;    // UASTC RDO quality scalar, 0 disables RDO
    int zstdLevel = 18;       // supercompression for UASTC, 0 disables
    bool uastcForColor = false;
};

struct NeuralOptions {
    std::string codec = "babylon-ntc-v2";
    std::string specGloss = "encode"; // encode: also encode KHR_materials_pbrSpecularGlossiness textures (v2), ignore: leave them
    int steps = 6000;
    int batch = 1 << 17;
    uint32_t seed = 1;
    int cudaDevice = 0;
    int fallbackSize = 0; // 0: 1x1 flat placeholders, otherwise downscaled originals
};

inline double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline std::string humanBytes(uint64_t b) {
    char buf[64];
    if (b >= (1ull << 20)) std::snprintf(buf, sizeof(buf), "%.2f MiB", b / double(1 << 20));
    else if (b >= 1024) std::snprintf(buf, sizeof(buf), "%.1f KiB", b / 1024.0);
    else std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
    return buf;
}

// Rewrites the current console line (used for step progress).
inline void progressLine(const char* text) {
    std::printf("\r%-100s", text);
    std::fflush(stdout);
}

std::string sanitizeFileName(const std::string& s);

struct Document;
bool runKtxPass(Document& doc, const KtxOptions& opts, const std::string& pngDir, std::string& err);
bool runNeuralPass(Document& doc, const NeuralOptions& opts, const std::string& pngDir, std::string& err);

} // namespace gtc
