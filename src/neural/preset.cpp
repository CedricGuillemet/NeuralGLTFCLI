#include "neural/preset.h"

#include <algorithm>
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

bool isV4Codec(const NeuralOptions& o) { return o.codec == "babylon-ntc-v4"; }

std::string presetNameOf(const NeuralOptions& o) { return o.preset.empty() ? (isV4Codec(o) ? "v4s" : "v2best") : o.preset; }

} // namespace

bool resolveNeuralPreset(const NeuralOptions& o, NeuralPreset& p, std::string& err) {
    const std::string name = presetNameOf(o);
    const bool v4Name = name.rfind("v4", 0) == 0;
    if (v4Name != isV4Codec(o) && o.codec != "nvidia-ntc") {
        err = std::string(v4Name ? "the v4 presets (v4xs, v4s, v4m, v4l) need --neural-codec babylon-ntc-v4"
                                 : "--neural-preset '" + name + "' belongs to the legacy codecs; babylon-ntc-v4 takes v4xs, v4s, v4m or v4l");
        return false;
    }
    // The v4 presets share the legacy levers that stay meaningful (uvMask, loss weights, seed, device).
    if (!buildPreset(v4Name ? "v2best" : name, p, err)) return false;
    p.name = name;
    ntc::TrainConfig& c = p.cfg;
    if (isV4Codec(o)) {
        V4Preset v;
        if (!resolveV4Preset(o, v, err)) return false;
        c.steps = v.stepsA;
        c.batch = v.batch;
    } else {
        if (o.stepsSet) c.steps = o.steps;
        if (o.batchSet) c.batch = o.batch;
    }
    c.seed = o.seed;
    c.device = o.cudaDevice;

    if (!o.grids.empty() && !isV4Codec(o)) {
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
    if (!o.activation.empty() && !isV4Codec(o)) {
        if (o.activation == "relu") c.net.activation = ntc::kRelu;
        else if (o.activation == "silu") c.net.activation = ntc::kSilu;
        else if (o.activation == "hgelu" || o.activation == "hardswish") c.net.activation = ntc::kHardSwish;
        else {
            err = "--neural-activation expects relu, silu, hgelu or hardswish (hgelu is the v3 hard-swish)";
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

bool resolveV4Preset(const NeuralOptions& o, V4Preset& v, std::string& err) {
    const std::string name = presetNameOf(o);
    v.name = name;
    // div0, c0, b0, c1, b1
    if (name == "v4xs") v.div0 = 4, v.c0 = 8, v.b0 = 2, v.c1 = 12, v.b1 = 4;
    else if (name == "v4s") v.div0 = 4, v.c0 = 12, v.b0 = 4, v.c1 = 16, v.b1 = 4;
    else if (name == "v4m") v.div0 = 2, v.c0 = 12, v.b0 = 2, v.c1 = 10, v.b1 = 4;
    else if (name == "v4l") v.div0 = 2, v.c0 = 16, v.b0 = 4, v.c1 = 12, v.b1 = 4;
    else {
        err = "unknown --neural-preset '" + name + "' for babylon-ntc-v4 (v4xs, v4s, v4m, v4l)";
        return false;
    }
    v.seed = o.seed;
    v.device = o.cudaDevice;
    v.constElision = !o.noConst;
    bool stepsBSet = false;
    if (o.stepsSet) v.stepsA = o.steps;
    if (o.batchSet) v.batch = o.batch;

    if (!o.grids.empty()) {
        const auto items = split(o.grids, ',');
        int d[2], c[2], b[2];
        bool ok = items.size() == 2;
        for (size_t k = 0; ok && k < 2; ++k) {
            const auto f = split(items[k], ':');
            ok = (f.size() == 3 || f.size() == 4) && toInt(f[0], d[k]) && toInt(f[1], c[k]) && toInt(f[2], b[k]);
            if (ok && f.size() == 4 && f[3] != (k == 0 ? "concat2x2" : "bilinear")) {
                err = "--neural-grids: v4 grid 0 is concat2x2 and grid 1 bilinear";
                return false;
            }
        }
        if (!ok || d[1] != 2 * d[0]) {
            err = "--neural-grids for babylon-ntc-v4 expects div0:channels:bits,div1:channels:bits with div1 = 2*div0, e.g. 4:12:4,8:16:4";
            return false;
        }
        v.div0 = d[0], v.c0 = c[0], v.b0 = b[0], v.c1 = c[1], v.b1 = b[1];
    }
    if (!o.activation.empty()) {
        if (o.activation == "hardgelu") v.activation = 0;
        else if (o.activation == "hardswish") v.activation = 1;
        else if (o.activation == "silu") v.activation = 2;
        else {
            err = "--neural-activation for babylon-ntc-v4 expects hardgelu (default), hardswish or silu (experiments only)";
            return false;
        }
        v.noRef = v.activation != 0;
    }
    if (!o.outAct.empty()) {
        auto setAct = [&](const std::string& act, int first, int count) {
            if (act != "none" && act != "sigmoid") return false;
            for (int i = 0; i < count; ++i) v.sigmoid[first + i] = act == "sigmoid";
            return true;
        };
        if (o.outAct.find('=') == std::string::npos) {
            if (!setAct(o.outAct, 0, ntc::kOutputs)) {
                err = "--neural-out-act expects none, sigmoid or group=act pairs";
                return false;
            }
        } else {
            for (const std::string& item : split(o.outAct, ',')) {
                const auto kv = split(item, '=');
                int first = 0, count = 0;
                if (kv.size() != 2 || !groupChannels(kv[0], first, count) || !setAct(kv[1], first, count)) {
                    err = "--neural-out-act expects none, sigmoid or group=act pairs (groups as in --neural-loss-weights)";
                    return false;
                }
            }
        }
    }
    for (const std::string& item : split(o.tune, ',')) {
        const auto kv = split(item, '=');
        float x = 0;
        if (kv.size() != 2 || !toFloat(kv[1], x)) {
            err = "--neural-tune expects key=value pairs (lrLatent, lrMlp, lrB, init, uniformMip, inside, noise, levelCap, stepsB, noRef)";
            return false;
        }
        const std::string& k = kv[0];
        if (k == "lrLatent") v.lrLatent = x;
        else if (k == "lrMlp") v.lrMlp = x;
        else if (k == "lrB") v.lrB = x;
        else if (k == "init") v.initLatent = x;
        else if (k == "uniformMip") v.uniformMip = x;
        else if (k == "inside") v.inside = x;
        else if (k == "noise") v.noise = x;
        else if (k == "levelCap") v.levelCap = int(x);
        else if (k == "stepsB") v.stepsB = int(x), stepsBSet = true;
        else if (k == "noRef") v.noRef = x != 0;
        else {
            err = "--neural-tune: unknown key '" + k + "'";
            return false;
        }
    }
    if (!stepsBSet && (o.stepsSet || v.stepsA != 40000)) v.stepsB = std::max(v.stepsA / 20, 2000);
    return true;
}

} // namespace gtc
