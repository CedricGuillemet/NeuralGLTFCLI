#include "neural/preset.h"

#include <cstdlib>

#include "passes.h"

namespace gtc {

namespace {

// Levers shared by every preset after v2 (selected with output/quality/experiments.csv): train only on the texels
// the mesh UVs cover, on exact texel centres, for four times the default steps.
void applyBestTraining(NeuralPreset& p) {
    p.uvMask = true;
    p.cfg.texelP = 1.0f;
    p.cfg.steps = 24000;
}

bool buildPreset(const std::string& name, NeuralPreset& p, std::string& err) {
    p.name = name;
    if (name == "v2") return true;
    if (name == "v2best") {
        applyBestTraining(p);
        return true;
    }
    auto v3 = [&](std::initializer_list<ntc::GridDesc> grids, int hidden, int layers) {
        applyBestTraining(p);
        p.cfg.net.grids.assign(grids.begin(), grids.end());
        p.cfg.net.hidden = hidden;
        p.cfg.net.hiddenLayers = layers;
        p.cfg.net.activation = ntc::kSilu;
        return true;
    };
    if (name == "v3s") return v3({{4, 16, 4}, {16, 16, 4}}, 64, 2);
    if (name == "v3m") return v3({{4, 24, 4}, {16, 16, 4}}, 64, 2);
    if (name == "v3l") return v3({{2, 8, 4}, {4, 16, 4}, {16, 16, 4}}, 64, 2);
    if (name == "v3x") return v3({{2, 16, 4}, {4, 16, 4}, {16, 16, 4}}, 64, 3);
    err = "unknown --neural-preset '" + name + "' (v2, v2best, v3s, v3m, v3l, v3x)";
    return false;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    for (size_t b = 0; b <= s.size();) {
        size_t e = s.find(sep, b);
        if (e == std::string::npos) e = s.size();
        if (e > b) out.push_back(s.substr(b, e - b));
        b = e + 1;
    }
    return out;
}

bool toFloat(const std::string& s, float& v) {
    char* end = nullptr;
    v = std::strtof(s.c_str(), &end);
    return end && !*end && !s.empty();
}

bool toInt(const std::string& s, int& v) {
    char* end = nullptr;
    v = int(std::strtol(s.c_str(), &end, 10));
    return end && !*end && !s.empty();
}

// Canonical channels of every loss-weight group name (see NeuralChannel).
bool groupChannels(const std::string& g, int& first, int& count) {
    struct G {
        const char* name;
        int first, count;
    };
    static const G kGroups[] = {{"base", 0, 3},      {"alpha", 3, 1},        {"normal", 4, 2},    {"occlusion", 6, 1},
                                {"roughness", 7, 1}, {"metallic", 8, 1},     {"emissive", 9, 3},  {"diffuse", 12, 3},
                                {"diffusealpha", 15, 1}, {"specular", 16, 3}, {"glossiness", 19, 1}};
    for (const G& d : kGroups)
        if (g == d.name) {
            first = d.first;
            count = d.count;
            return true;
        }
    return false;
}

} // namespace

bool resolveNeuralPreset(const NeuralOptions& o, NeuralPreset& p, std::string& err) {
    if (!buildPreset(o.preset, p, err)) return false;
    ntc::TrainConfig& c = p.cfg;
    if (o.stepsSet) c.steps = o.steps;
    if (o.batchSet) c.batch = o.batch;
    c.seed = o.seed;
    c.device = o.cudaDevice;

    if (!o.grids.empty()) {
        std::vector<ntc::GridDesc> grids;
        for (const std::string& item : split(o.grids, ',')) {
            auto f = split(item, ':');
            ntc::GridDesc g;
            if (f.size() != 3 || !toInt(f[0], g.div) || !toInt(f[1], g.channels) || !toInt(f[2], g.bits)) {
                err = "--neural-grids expects div:channels:bits[,...], e.g. 4:16:4,16:16:4";
                return false;
            }
            grids.push_back(g);
        }
        c.net.grids = grids;
    }
    if (o.hidden > 0) c.net.hidden = o.hidden;
    if (o.hiddenLayers > 0) c.net.hiddenLayers = o.hiddenLayers;
    if (!o.activation.empty()) {
        if (o.activation == "relu") c.net.activation = ntc::kRelu;
        else if (o.activation == "silu") c.net.activation = ntc::kSilu;
        else if (o.activation == "hgelu") c.net.activation = ntc::kHgelu;
        else {
            err = "--neural-activation expects relu, silu or hgelu";
            return false;
        }
    }
    if (!o.uvMask.empty()) {
        if (o.uvMask != "on" && o.uvMask != "off") {
            err = "--neural-uv-mask expects on or off";
            return false;
        }
        p.uvMask = o.uvMask == "on";
    }
    if (!o.sampling.empty()) {
        if (o.sampling == "continuous") c.texelP = 0.0f;
        else if (o.sampling == "texel") c.texelP = 1.0f;
        else if (o.sampling.rfind("mixed:", 0) == 0 && toFloat(o.sampling.substr(6), c.texelP) && c.texelP >= 0.0f && c.texelP <= 1.0f) {
        } else {
            err = "--neural-sampling expects continuous, texel or mixed:<p> (0..1)";
            return false;
        }
    }
    if (o.importance >= 0.0f) c.importance = o.importance;
    if (!o.loss.empty()) {
        if (o.loss == "l2") c.loss = ntc::kLossL2;
        else if (o.loss == "charbonnier" || o.loss == "l1") c.loss = ntc::kLossCharbonnier;
        else {
            err = "--neural-loss expects l2 or charbonnier";
            return false;
        }
    }
    if (!o.lossWeights.empty()) {
        for (const std::string& item : split(o.lossWeights, ',')) {
            auto kv = split(item, '=');
            int first = 0, count = 0;
            float w = 0;
            if (kv.size() != 2 || !groupChannels(kv[0], first, count) || !toFloat(kv[1], w)) {
                err = "--neural-loss-weights expects name=weight pairs (base, alpha, normal, occlusion, roughness, metallic, "
                      "emissive, diffuse, specular, glossiness)";
                return false;
            }
            for (int i = 0; i < count; ++i) p.lossWeight[first + i] = w;
        }
    }
    if (o.finetuneSteps >= 0) c.finetuneSteps = o.finetuneSteps;
    if (o.warmup >= 0.0f) c.warmup = o.warmup;
    if (o.beta2 > 0.0f) c.beta2 = o.beta2;
    if (!o.finetuneLr.empty()) {
        auto f = split(o.finetuneLr, ':');
        if (f.empty() || f.size() > 2 || !toFloat(f[0], c.lrFinetune) || (f.size() == 2 && !toFloat(f[1], c.lrFinetuneEnd))) {
            err = "--neural-finetune-lr expects lr or lr:endLr";
            return false;
        }
        if (f.size() == 1) c.lrFinetuneEnd = -1.0f;
    }
    if (!o.quant.empty()) {
        if (o.quant != "noise" && o.quant != "ste") {
            err = "--neural-quant expects noise or ste";
            return false;
        }
        c.ste = o.quant == "ste";
    }
    return true;
}

} // namespace gtc
