#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <thread>

#include <ktx.h>

#include "gltf_io.h"
#include "image.h"
#include "passes.h"

namespace fs = std::filesystem;

namespace gtc {

namespace {

// Vulkan format values used by libktx (see vkformat_enum.h).
constexpr uint32_t VK_R8G8B8_UNORM = 23, VK_R8G8B8_SRGB = 29, VK_R8G8B8A8_UNORM = 37, VK_R8G8B8A8_SRGB = 43;

bool isColorSlot(const std::string& slot) {
    static const std::set<std::string> color = {"baseColorTexture", "emissiveTexture", "sheenColorTexture",
                                                "specularColorTexture", "diffuseTexture", "specularGlossinessTexture",
                                                "diffuseTransmissionColorTexture"};
    return color.count(slot) != 0;
}

struct KtxResult {
    std::vector<uint8_t> bytes;
    Image decoded; // level 0 transcoded back to RGBA (only when requested)
};

bool encodeKtx2(const Image& img, bool srgb, bool uastc, bool normalMap, const KtxOptions& o, bool decodeBack,
                KtxResult& out, std::string& err) {
    const bool alpha = img.hasAlpha();
    const int channels = alpha ? 4 : 3;
    auto mips = buildMipChain(img, srgb);

    ktxTextureCreateInfo ci{};
    ci.vkFormat = alpha ? (srgb ? VK_R8G8B8A8_SRGB : VK_R8G8B8A8_UNORM) : (srgb ? VK_R8G8B8_SRGB : VK_R8G8B8_UNORM);
    ci.baseWidth = uint32_t(img.width);
    ci.baseHeight = uint32_t(img.height);
    ci.baseDepth = 1;
    ci.numDimensions = 2;
    ci.numLevels = uint32_t(mips.size());
    ci.numLayers = 1;
    ci.numFaces = 1;
    ci.isArray = KTX_FALSE;
    ci.generateMipmaps = KTX_FALSE;

    ktxTexture2* tex = nullptr;
    KTX_error_code rc = ktxTexture2_Create(&ci, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tex);
    if (rc != KTX_SUCCESS) {
        err = std::string("ktxTexture2_Create: ") + ktxErrorString(rc);
        return false;
    }
    for (size_t level = 0; level < mips.size(); ++level) {
        const Image& m = mips[level];
        std::vector<uint8_t> px(size_t(m.width) * m.height * channels);
        for (size_t i = 0, n = size_t(m.width) * m.height; i < n; ++i)
            std::memcpy(&px[i * channels], &m.rgba[i * 4], size_t(channels));
        rc = ktxTexture_SetImageFromMemory(ktxTexture(tex), uint32_t(level), 0, 0, px.data(), px.size());
        if (rc != KTX_SUCCESS) {
            err = std::string("ktxTexture_SetImageFromMemory: ") + ktxErrorString(rc);
            ktxTexture_Destroy(ktxTexture(tex));
            return false;
        }
    }

    ktxBasisParams params{};
    params.structSize = sizeof(params);
    params.threadCount = std::max(1u, std::thread::hardware_concurrency());
    if (uastc) {
        params.uastc = KTX_TRUE;
        params.uastcFlags = std::clamp(o.uastcLevel, 0, 4);
        params.uastcRDO = o.uastcRdo > 0 ? KTX_TRUE : KTX_FALSE;
        params.uastcRDOQualityScalar = o.uastcRdo;
    } else {
        params.uastc = KTX_FALSE;
        params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
        params.qualityLevel = uint32_t(std::clamp(o.etc1sQuality, 1, 255));
        params.normalMap = normalMap ? KTX_TRUE : KTX_FALSE;
    }
    rc = ktxTexture2_CompressBasisEx(tex, &params);
    if (rc == KTX_SUCCESS && uastc && o.zstdLevel > 0) rc = ktxTexture2_DeflateZstd(tex, uint32_t(std::clamp(o.zstdLevel, 1, 22)));
    if (rc != KTX_SUCCESS) {
        err = std::string("Basis compression: ") + ktxErrorString(rc);
        ktxTexture_Destroy(ktxTexture(tex));
        return false;
    }

    ktx_uint8_t* data = nullptr;
    ktx_size_t size = 0;
    rc = ktxTexture_WriteToMemory(ktxTexture(tex), &data, &size);
    ktxTexture_Destroy(ktxTexture(tex));
    if (rc != KTX_SUCCESS) {
        err = std::string("ktxTexture_WriteToMemory: ") + ktxErrorString(rc);
        return false;
    }
    out.bytes.assign(data, data + size);
    free(data);

    if (decodeBack) {
        ktxTexture2* t = nullptr;
        rc = ktxTexture2_CreateFromMemory(out.bytes.data(), out.bytes.size(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &t);
        if (rc == KTX_SUCCESS) rc = ktxTexture2_TranscodeBasis(t, KTX_TTF_RGBA32, 0);
        if (rc != KTX_SUCCESS) {
            err = std::string("transcoding for PNG output: ") + ktxErrorString(rc);
            if (t) ktxTexture_Destroy(ktxTexture(t));
            return false;
        }
        ktx_size_t off = 0;
        ktxTexture_GetImageOffset(ktxTexture(t), 0, 0, 0, &off);
        out.decoded.width = img.width;
        out.decoded.height = img.height;
        const uint8_t* p = ktxTexture_GetData(ktxTexture(t)) + off;
        out.decoded.rgba.assign(p, p + size_t(img.width) * img.height * 4);
        ktxTexture_Destroy(ktxTexture(t));
    }
    return true;
}

} // namespace

std::string sanitizeFileName(const std::string& s) {
    std::string r;
    for (char c : s) r += (std::isalnum(uint8_t(c)) || c == '-' || c == '_' || c == '.') ? c : '_';
    return r.empty() ? "unnamed" : r;
}

bool runKtxPass(Document& doc, const KtxOptions& opts, const std::string& pngDir, std::string& err) {
    auto& m = doc.model;

    // Classify every image by how materials use it.
    struct Usage { bool color = false, linear = false, normal = false; std::string name; };
    std::map<int, Usage> usage; // image index -> usage
    std::vector<std::vector<int>> imageTextures(m.images.size());
    for (size_t t = 0; t < m.textures.size(); ++t)
        if (m.textures[t].source >= 0) imageTextures[m.textures[t].source].push_back(int(t));
    for (auto& mat : m.materials) {
        forEachMaterialTexture(mat, [&](const std::string& slot, int& texIdx) {
            if (texIdx < 0 || texIdx >= int(m.textures.size())) return;
            int img = m.textures[texIdx].source;
            if (img < 0) return; // already uses an extension source (KTX2, WebP...)
            auto& u = usage[img];
            (isColorSlot(slot) ? u.color : u.linear) = true;
            if (slot == "normalTexture") u.normal = true;
        });
    }

    if (!pngDir.empty()) {
        fs::create_directories(toPath(pngDir) / "original");
        fs::create_directories(toPath(pngDir) / "ktx");
    }

    size_t done = 0;
    for (auto& [imgIdx, u] : usage) {
        ++done;
        auto& gimg = m.images[imgIdx];
        std::string label = gimg.name.empty() ? "image" + std::to_string(imgIdx) : gimg.name;
        const std::string mime = mimeFromBytes(doc.imageData[imgIdx]);
        if (mime != "image/png" && mime != "image/jpeg") {
            std::printf("  [%zu/%zu] %s: skipped (%s)\n", done, usage.size(), label.c_str(), mime.c_str());
            continue;
        }
        Image img;
        if (!decodeImage(doc.imageData[imgIdx], img, err)) return false;

        // Linear data (or mixed usage) gets UASTC; color data gets ETC1S unless overridden.
        const bool srgb = u.color && !u.linear;
        const bool uastc = !srgb || opts.uastcForColor;
        char line[256];
        std::snprintf(line, sizeof(line), "  [%zu/%zu] %s %dx%d -> %s%s ...", done, usage.size(), label.c_str(), img.width,
                      img.height, uastc ? "UASTC" : "ETC1S", srgb ? " sRGB" : "");
        progressLine(line);
        auto t0 = std::chrono::steady_clock::now();
        KtxResult res;
        if (!encodeKtx2(img, srgb, uastc, u.normal, opts, !pngDir.empty(), res, err)) {
            std::printf("\n");
            err = label + ": " + err;
            return false;
        }
        char tail[128];
        std::snprintf(tail, sizeof(tail), " %s -> %s (%.1f s)", humanBytes(doc.imageData[imgIdx].size()).c_str(),
                      humanBytes(res.bytes.size()).c_str(), secondsSince(t0));
        std::printf("\r%-100s\n", (std::string(line) + tail).c_str());

        if (!pngDir.empty()) {
            std::string file = sanitizeFileName(std::to_string(imgIdx) + "_" + label) + ".png";
            int ch = img.hasAlpha() ? 4 : 3;
            writePng(fromPath(toPath(pngDir) / "original" / file), img, ch);
            writePng(fromPath(toPath(pngDir) / "ktx" / file), res.decoded, ch);
        }

        doc.imageData[imgIdx] = std::move(res.bytes);
        gimg.mimeType = "image/ktx2";
        for (int t : imageTextures[imgIdx]) {
            auto& tex = m.textures[t];
            tinygltf::Value::Object ext;
            ext["source"] = tinygltf::Value(imgIdx);
            tex.extensions["KHR_texture_basisu"] = tinygltf::Value(std::move(ext));
            tex.source = -1;
        }
    }
    doc.addExtensionUsed("KHR_texture_basisu", true);
    return true;
}

} // namespace gtc
