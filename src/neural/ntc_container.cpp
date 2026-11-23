#include "neural/ntc_container.h"

#include <algorithm>
#include <cstring>

#include "json.hpp"

namespace gtc {

namespace {

using json = nlohmann::json;

uint32_t readU32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
uint64_t readU64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

bool linearOrAbsent(const json& o, const char* key) {
    auto it = o.find(key);
    return it == o.end() || (it->is_string() && *it == "Linear");
}

bool intField(const json& o, const char* key, int64_t& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->is_number_integer()) return false;
    out = it->get<int64_t>();
    return true;
}

} // namespace

bool parseNtcContainer(const std::vector<uint8_t>& file, NtcContainerInfo& info, std::string& err) {
    info = NtcContainerInfo{};
    if (file.size() < 40 || std::memcmp(file.data(), "NTEX", 4) != 0) {
        err = "not an NTC texture set container (missing NTEX signature)";
        return false;
    }
    info.containerVersion = readU32(&file[4]);
    if (info.containerVersion != kNtcContainerVersion) {
        err = "unsupported NTC container version " + std::to_string(info.containerVersion);
        return false;
    }
    const uint64_t jsonOff = readU64(&file[8]), jsonSize = readU64(&file[16]);
    const uint64_t dataOff = readU64(&file[24]), dataSize = readU64(&file[32]);
    for (uint64_t v : {jsonOff, jsonSize, dataOff, dataSize})
        if (v % 4 != 0) {
            err = "NTC chunk offset or size is not a multiple of 4";
            return false;
        }
    if (jsonOff < 40 || jsonOff > file.size() || jsonSize > file.size() - jsonOff || dataOff > file.size() ||
        dataSize > file.size() - dataOff) {
        err = "NTC chunk lies outside the file";
        return false;
    }
    info.jsonBytes = size_t(jsonSize);
    info.dataBytes = size_t(dataSize);

    size_t jsonLen = size_t(jsonSize);
    while (jsonLen > 0 && file[size_t(jsonOff) + jsonLen - 1] == 0) --jsonLen; // zero padding
    json doc = json::parse(file.begin() + std::ptrdiff_t(jsonOff), file.begin() + std::ptrdiff_t(jsonOff + jsonLen),
                           nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        err = "NTC JSON descriptor does not parse";
        return false;
    }

    int64_t v = 0;
    if (!intField(doc, "schemaVersion", v) || v != kNtcSchemaVersion010) {
        err = "unsupported NTC schemaVersion (expected " + std::to_string(kNtcSchemaVersion010) + ")";
        return false;
    }
    info.schemaVersion = int(v);
    if (!intField(doc, "width", v) || v < 1 || v > 16384) { err = "NTC width missing or out of range"; return false; }
    info.width = int(v);
    if (!intField(doc, "height", v) || v < 1 || v > 16384) { err = "NTC height missing or out of range"; return false; }
    info.height = int(v);
    if (!intField(doc, "numChannels", v) || v < 1 || v > 16) { err = "NTC numChannels missing or out of range"; return false; }
    info.numChannels = int(v);
    info.numColorMips = doc.contains("numColorMips") && intField(doc, "numColorMips", v) ? int(v) : 1;

    if (!doc.contains("latentShape") || !intField(doc["latentShape"], "numFeatures", v) ||
        (v != 4 && v != 8 && v != 12 && v != 16)) {
        err = "NTC latentShape.numFeatures missing or invalid";
        return false;
    }
    info.numFeatures = int(v);

    if (!doc.contains("views") || !doc["views"].is_array()) { err = "NTC views missing"; return false; }
    const auto& views = doc["views"];
    info.numViews = int(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
        const auto& vw = views[i];
        int64_t off = 0, sz = 0;
        if (!intField(vw, "offset", off) || !intField(vw, "storedSize", sz) || off < 0 || sz < 0 ||
            uint64_t(off) > dataSize || uint64_t(sz) > dataSize - uint64_t(off)) {
            err = "NTC view " + std::to_string(i) + " lies outside the data chunk";
            return false;
        }
        if (vw.contains("compression") && !(vw["compression"].is_string() && vw["compression"] == "None")) {
            err = "NTC view " + std::to_string(i) + " is compressed (GDeflate is not supported)";
            return false;
        }
    }
    auto viewOk = [&](const json& idx) { return idx.is_number_integer() && idx.get<int64_t>() >= 0 && idx.get<int64_t>() < int64_t(views.size()); };

    if (doc.contains("mlpVersions")) {
        if (!doc["mlpVersions"].is_array()) { err = "NTC mlpVersions is not an array"; return false; }
        info.numMlpVersions = int(doc["mlpVersions"].size());
        for (const auto& mlp : doc["mlpVersions"]) {
            if (!mlp.contains("layers") || !mlp["layers"].is_array() || mlp["layers"].empty()) { err = "NTC MLP without layers"; return false; }
            for (const auto& l : mlp["layers"]) {
                if (!l.contains("weightView") || !viewOk(l["weightView"]) || !l.contains("biasView") || !viewOk(l["biasView"]) ||
                    (l.contains("scaleView") && !viewOk(l["scaleView"]))) {
                    err = "NTC MLP layer references an invalid view";
                    return false;
                }
            }
        }
    }

    if (doc.contains("channels")) {
        const auto& ch = doc["channels"];
        if (!ch.is_array() || int(ch.size()) != info.numChannels) { err = "NTC channels count does not match numChannels"; return false; }
        for (const auto& c : ch)
            if (!linearOrAbsent(c, "colorSpace")) { err = "NTC channel with a non-Linear colorSpace"; return false; }
    }

    if (doc.contains("textures")) {
        for (const auto& t : doc["textures"]) {
            NtcTextureInfo ti;
            int64_t fc = 0, nc = 0;
            if (!t.contains("name") || !t["name"].is_string() || !intField(t, "firstChannel", fc) || !intField(t, "numChannels", nc) ||
                fc < 0 || nc < 1 || nc > 4 || fc + nc > info.numChannels) {
                err = "NTC texture entry is invalid";
                return false;
            }
            ti.name = t["name"].get<std::string>();
            // ntc-cli appends "|ntcsem:<semantics>" to the texture names
            ti.name = ti.name.substr(0, ti.name.find('|'));
            ti.firstChannel = int(fc);
            ti.numChannels = int(nc);
            if (!linearOrAbsent(t, "rgbColorSpace") || !linearOrAbsent(t, "alpaColorSpace") || !linearOrAbsent(t, "alphaColorSpace")) {
                err = "NTC texture '" + ti.name + "' has a non-Linear color space";
                return false;
            }
            if (t.contains("bcModeBuffers") && !(t["bcModeBuffers"].is_array() && t["bcModeBuffers"].empty())) {
                err = "NTC texture '" + ti.name + "' carries BC mode buffers";
                return false;
            }
            if (t.contains("bcFormat") && !(t["bcFormat"].is_string() && t["bcFormat"] == "None")) {
                err = "NTC texture '" + ti.name + "' requests BC transcoding";
                return false;
            }
            info.textures.push_back(std::move(ti));
        }
    }

    int latentW = 0;
    if (!doc.contains("latents") || !doc["latents"].is_array() || doc["latents"].empty()) { err = "NTC latents missing"; return false; }
    for (size_t i = 0; i < doc["latents"].size(); ++i) {
        const auto& l = doc["latents"][i];
        int64_t w = 0, h = 0;
        if (!intField(l, "width", w) || !intField(l, "height", h) || !l.contains("layerViews") || !l["layerViews"].is_array()) {
            err = "NTC latent image " + std::to_string(i) + " is invalid";
            return false;
        }
        for (const auto& lv : l["layerViews"])
            if (!viewOk(lv)) { err = "NTC latent image references an invalid view"; return false; }
        if (i == 0) latentW = int(w);
    }
    info.numLatentImages = int(doc["latents"].size());

    if (!doc.contains("colorMips") || !doc["colorMips"].is_array() || doc["colorMips"].empty()) { err = "NTC colorMips missing"; return false; }
    if (int(doc["colorMips"].size()) != info.numColorMips) { err = "NTC colorMips count does not match numColorMips"; return false; }
    for (const auto& m : doc["colorMips"]) {
        if (!m.contains("latentMip") || !m["latentMip"].is_number_integer() || m["latentMip"].get<int>() < 0 ||
            m["latentMip"].get<int>() >= info.numLatentImages) {
            err = "NTC color mip without a valid latentMip";
            return false;
        }
        if (m.contains("combinedColorData") || m.contains("perTextureColorData")) {
            err = "NTC color mip carries raw or BC color data";
            return false;
        }
    }

    // gridSizeScale is not stored: find the scale whose latent size matches the first latent image.
    for (int gs = 1; gs <= 32 && info.gridSizeScale == 0; ++gs)
        if (std::max(1, info.width / gs) == latentW || std::max(1, (info.width + gs - 1) / gs) == latentW) info.gridSizeScale = gs;
    if (info.gridSizeScale == 0) { err = "NTC latent size does not match any gridSizeScale"; return false; }
    return true;
}

} // namespace gtc
