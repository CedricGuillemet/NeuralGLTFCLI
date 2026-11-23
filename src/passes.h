#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

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
    std::string codec = "babylon-ntc-v4"; // babylon-ntc-v4 (default), nvidia-ntc (opt-in benchmark codec), or the legacy babylon-ntc-v1/v2/v3
    std::string specGloss = "encode"; // encode: also encode KHR_materials_pbrSpecularGlossiness textures (v2), ignore: leave them
    int steps = 6000;
    int batch = 1 << 17;
    uint32_t seed = 1;
    int cudaDevice = 0;
    int fallbackSize = 0; // 0: 1x1 flat placeholders, otherwise downscaled originals
    // What the replaced slots contain for loaders without the extension: png (placeholders, or downscaled PNGs with
    // fallbackSize), ktx (KTX2 via KHR_texture_basisu, full size unless fallbackSize is set) or none (no texture).
    std::string fallback = "png";
    KtxOptions ktxFallback;
    bool stepsSet = false, batchSet = false; // --steps / --batch given explicitly (override the preset)

    // Training levers (src/neural/preset.cpp). Empty / negative values come from the preset.
    std::string preset;          // empty: v4s for babylon-ntc-v4, v2best otherwise; v4xs v4s v4m v4l, v2, v2best, v3s, v3m, v3l, v3x
    std::string grids;           // "div:channels:bits,..." e.g. "4:16:4,16:16:4" (v4: "4:12:4,8:16:4")
    int hidden = 0, hiddenLayers = 0;
    std::string activation;      // relu, silu, hgelu (v3 hard-swish), hardswish; v4: hardgelu (default), hardswish, silu
    std::string outAct;          // v4: "none", "sigmoid" or "group=act,..." output activation (see --neural-out-act)
    std::string tune;            // v4 trainer levers "key=value,..." (see --neural-tune)
    bool noConst = false;        // v4: keep constant channels in the network
    std::string uvMask;          // on, off
    std::string sampling;        // continuous, texel, mixed:<p>
    float importance = -1.0f;
    std::string loss;            // l2, charbonnier
    std::string lossWeights;     // "base=1.5,normal=3"
    int finetuneSteps = -1;
    float warmup = -1.0f;
    float beta2 = -1.0f;
    std::string finetuneLr;      // "1e-3" or "1e-3:1e-5" (cosine)
    std::string quant;           // noise, ste

    // Output naming and extras. The pass writes nothing itself except the PNG folder, the work
    // directories and the optional files below; main() writes <stem>_<tag>.glb.
    std::string tag = "neural";           // output name <stem>_<tag>.glb and PNG subfolder <png>/<tag>/
    std::vector<std::string> materials;   // only encode these materials (empty: all)
    std::string bakedPath;                // if set, also write the plain-glTF (baked) version here
    std::string metricsPath;              // if set, write the metrics sidecar JSON here
    std::string workDir;                  // per-material scratch directories are created under this
    std::string commandLine;              // recorded in the metrics

    // "nvidia-ntc" codec (the default, runs the external RTXNTC ntc-cli).
    std::string ntcCli;                   // path to ntc-cli (or the NTC_CLI environment variable, or PATH)
    double ntcBpp = 4.5;                  // target bits per pixel for every material (snapped to a latent shape)
    std::string ntcMatch;                 // metrics JSON whose per-material sizes set the target bpp
    std::string ntcArgs;                  // extra arguments passed to ntc-cli
    std::string ntcSnap = "nearest";      // nearest: pick the latent shape closest to the target bpp, ntc: ntc-cli -b
    bool ntcKeepImages = false;           // keep the input/decoded PNGs in the work directory
    bool ntcMips = false;                 // ntc-cli -g: latent and color mips
    bool trainingFlagsGiven = false;      // a babylon-ntc training flag was given (ignored by nvidia-ntc)
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
// Encodes only the given images (used for the neural fallback textures); KHR_texture_basisu is then optional.
bool runKtxPassOnImages(Document& doc, const KtxOptions& opts, const std::set<int>& images, std::string& err);
bool runNeuralPass(Document& doc, const NeuralOptions& opts, const std::string& pngDir, std::string& err);

} // namespace gtc
