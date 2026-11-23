#include <cstdio>
#include <cstring>

#include "../passes.h"
#include "codec.h"
#include "ntc_cuda.h"
#include "preset.h"

namespace gtc {

namespace {

using tinygltf::Value;

Value intArray(std::initializer_list<int> v) {
    Value::Array a;
    for (int x : v) a.emplace_back(x);
    return Value(std::move(a));
}

const char* activationName(int a) { return a == ntc::kSilu ? "silu" : a == ntc::kHgelu ? "hgelu" : "relu"; }

class BabylonNtc : public INeuralCodec {
public:
    BabylonNtc(const NeuralPreset& preset, NeuralCodecVersion version) : preset_(preset), cfg_(preset.cfg), version_(version) {}

    std::string name() const override {
        return version_ == NeuralCodecVersion::V1 ? "babylon-ntc-v1" : version_ == NeuralCodecVersion::V2 ? "babylon-ntc-v2" : "babylon-ntc-v3";
    }
    std::string deviceDescription() const override { return "CUDA device " + std::to_string(cfg_.device) + ": " + ntc::deviceName(cfg_.device); }
    size_t expectedWeightCount(int outputs) const override { return cfg_.net.weightCount(outputs); }

    tinygltf::Value::Object describeParameters() const override {
        const ntc::NetConfig& n = cfg_.net;
        tinygltf::Value::Object p;
        p["preset"] = Value(preset_.name);
        Value::Array grids;
        for (const ntc::GridDesc& g : n.grids) {
            Value::Object o;
            o["div"] = Value(g.div);
            o["channels"] = Value(g.channels);
            o["bits"] = Value(g.bits);
            grids.emplace_back(std::move(o));
        }
        p["grids"] = Value(std::move(grids));
        p["hidden"] = Value(n.hidden);
        p["hiddenLayers"] = Value(n.hiddenLayers);
        p["activation"] = Value(std::string(activationName(n.activation)));
        p["steps"] = Value(cfg_.steps);
        p["batch"] = Value(cfg_.batch);
        p["finetuneSteps"] = Value(cfg_.finetuneSteps < 0 ? cfg_.steps / 4 : cfg_.finetuneSteps);
        p["finetuneLr"] = Value(double(cfg_.lrFinetune));
        p["finetuneLrEnd"] = Value(double(cfg_.lrFinetuneEnd));
        p["warmup"] = Value(double(cfg_.warmup));
        p["beta2"] = Value(double(cfg_.beta2));
        p["quant"] = Value(std::string(cfg_.ste ? "ste" : "noise"));
        p["samplingTexelP"] = Value(double(cfg_.texelP));
        p["importance"] = Value(double(cfg_.importance));
        p["loss"] = Value(std::string(cfg_.loss == ntc::kLossL2 ? "l2" : "charbonnier"));
        p["uvMask"] = Value(preset_.uvMask);
        p["maskInside"] = Value(double(cfg_.maskInside));
        return p;
    }

    bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) override {
        static_assert(ntc::kOutputs == kMaxNeuralChannels);
        const int N = in.outputs;
        if (N < 1 || N > kMaxNeuralChannels || (version_ == NeuralCodecVersion::V1 && N != kV1NeuralChannels)) {
            err = "unsupported output count " + std::to_string(N) + " for " + name();
            return false;
        }
        const size_t texels = size_t(in.width) * in.height;
        if (in.target.size() != texels * N) {
            err = "target size does not match " + std::to_string(N) + " outputs";
            return false;
        }

        // Training always runs at full width; unused channels have a target and a loss weight of 0.
        std::vector<float> target(texels * ntc::kOutputs, 0.0f);
        for (size_t i = 0; i < texels; ++i) std::memcpy(&target[i * ntc::kOutputs], &in.target[i * N], N * sizeof(float));
        float lossWeights[ntc::kOutputs]{};
        std::memcpy(lossWeights, in.lossWeights, N * sizeof(float));

        ntc::TrainConfig cfg = cfg_;
        cfg.uvMask = in.uvMask.size() == texels ? &in.uvMask : nullptr;

        ntc::Model model;
        auto t0 = std::chrono::steady_clock::now();
        auto progress = [&](int phase, int step, int total, float loss) {
            char line[200];
            const double el = secondsSince(t0);
            std::snprintf(line, sizeof(line), "    %s %s: step %d/%d  loss %.5f  %.0f s", in.name.c_str(),
                          phase == 0 ? "training" : "fine-tuning", step, total, loss, el);
            progressLine(line);
        };
        if (!ntc::train(target, in.width, in.height, N, lossWeights, cfg, model, progress, err)) return false;
        std::printf("\n");
        std::vector<float> full;
        if (!ntc::decode(model, cfg_.device, full, err)) return false;
        out.decoded.resize(texels * N);
        for (size_t i = 0; i < texels; ++i) std::memcpy(&out.decoded[i * N], &full[i * ntc::kOutputs], N * sizeof(float));
        out.extraMetrics["psnrFloat"] = Value(double(model.psnrFloat));
        out.extraMetrics["psnrQuantized"] = Value(double(model.psnrQuant));
        out.extraMetrics["psnrFinal"] = Value(double(model.psnrFinal));

        // Latent grids: every texel is bits * channels bits, stored as consecutive RGBA8 images. 8-bit channels
        // fill one byte each, 4-bit channels pair up (channel 2k in the high nibble of byte k).
        const bool v3 = version_ == NeuralCodecVersion::V3;
        int textureIndex = 0;
        Value::Array gridInfo;
        std::vector<std::vector<int>> gridTextures;
        for (size_t gi = 0; gi < model.grids.size(); ++gi) {
            const ntc::GridData& g = model.grids[gi];
            const int bytes = g.channels * g.bits / 8;
            const int images = (bytes + 3) / 4;
            const size_t gt = size_t(g.w) * g.h;
            Value::Array textures;
            for (int im = 0; im < images; ++im) {
                Image img;
                img.width = g.w;
                img.height = g.h;
                img.rgba.assign(gt * 4, 0);
                for (size_t t = 0; t < gt; ++t) {
                    const uint8_t* q = &g.q[t * g.channels];
                    for (int c = 0; c < 4; ++c) {
                        const int byte = im * 4 + c;
                        if (byte >= bytes) break;
                        img.rgba[t * 4 + c] = g.bits == 8 ? q[byte] : uint8_t((q[2 * byte] << 4) | q[2 * byte + 1]);
                    }
                }
                out.latentImages.push_back(std::move(img));
                out.latentNames.push_back(v3 ? "g" + std::to_string(gi) + "_" + std::to_string(im)
                                             : std::string(gi == 0 ? "hi" : "lo") + std::to_string(im));
                textures.emplace_back(textureIndex++);
            }
            Value::Object o;
            o["resolution"] = intArray({g.w, g.h});
            o["channels"] = Value(g.channels);
            o["bits"] = Value(g.bits);
            o["sampling"] = Value(std::string("bilinear"));
            o["textures"] = Value(std::move(textures));
            gridInfo.emplace_back(std::move(o));
        }
        out.latentPointSampling = v3;

        // Weights: every hidden layer as trained, then only the first N columns of the last layer's W and b.
        const ntc::NetConfig& net = model.net;
        const size_t in0 = size_t(net.inputs()), h = size_t(net.hidden), L = size_t(net.hiddenLayers);
        const size_t hiddenFloats = (in0 + 1) * h + (L - 1) * (h + 1) * h;
        std::vector<float> weights;
        weights.reserve(net.weightCount(N));
        weights.insert(weights.end(), model.weights.begin(), model.weights.begin() + hiddenFloats);
        for (size_t k = 0; k < h; ++k)
            for (int c = 0; c < N; ++c) weights.push_back(model.weights[hiddenFloats + k * ntc::kOutputs + c]);
        for (int c = 0; c < N; ++c) weights.push_back(model.weights[hiddenFloats + h * ntc::kOutputs + c]);
        if (weights.size() != net.weightCount(N)) {
            err = "internal error: weight export size mismatch";
            return false;
        }
        out.weights.resize(weights.size() * sizeof(float));
        std::memcpy(out.weights.data(), weights.data(), out.weights.size());

        Value::Array hidden;
        for (size_t l = 0; l < L; ++l) hidden.emplace_back(net.hidden);
        Value::Object pe;
        pe["frequencies"] = intArray({1, 2});
        Value::Object netJson;
        netJson["inputs"] = Value(net.inputs());
        netJson["hidden"] = Value(std::move(hidden));
        netJson["outputs"] = Value(N);
        netJson["activation"] = Value(std::string(activationName(net.activation)));
        if (v3) {
            Value::Array layout;
            for (size_t gi = 0; gi < model.grids.size(); ++gi) layout.emplace_back("grid" + std::to_string(gi));
            layout.emplace_back(std::string("posenc"));
            pe["grid"] = Value(0);
            netJson["inputLayout"] = Value(std::move(layout));
            netJson["positionalEncoding"] = Value(std::move(pe));
            std::string wl = "W0[in][out],b0";
            for (size_t l = 1; l <= L; ++l) wl += ",W" + std::to_string(l) + ",b" + std::to_string(l);
            netJson["weightLayout"] = Value(wl + " float32");
            out.parameters["grids"] = Value(std::move(gridInfo));
        } else {
            pe["scale"] = Value(std::string("hiResolution"));
            netJson["positionalEncoding"] = Value(std::move(pe));
            netJson["weightLayout"] = Value(std::string("W0[in][out],b0,W1,b1,W2,b2 float32"));
            out.parameters["hiResolution"] = intArray({model.grids[0].w, model.grids[0].h});
            out.parameters["loResolution"] = intArray({model.grids[1].w, model.grids[1].h});
            out.parameters["latentChannels"] = Value(model.grids[0].channels);
        }
        out.parameters["network"] = Value(std::move(netJson));
        return true;
    }

private:
    NeuralPreset preset_;
    ntc::TrainConfig cfg_;
    NeuralCodecVersion version_;
};

} // namespace

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err) {
    NeuralCodecVersion version;
    if (opts.codec == "babylon-ntc-v1") version = NeuralCodecVersion::V1;
    else if (opts.codec == "babylon-ntc-v2" || opts.codec == "babylon-ntc-v3") version = opts.codec == "babylon-ntc-v3" ? NeuralCodecVersion::V3 : NeuralCodecVersion::V2;
    else {
        err = "unknown neural codec '" + opts.codec + "' (available: babylon-ntc-v1, babylon-ntc-v2, babylon-ntc-v3)";
        return nullptr;
    }
    NeuralPreset preset;
    if (!resolveNeuralPreset(opts, preset, err)) return nullptr;
    if (!ntc::supported(preset.cfg.net, err)) return nullptr;
    if (!preset.cfg.net.isV2Shape()) {
        if (version == NeuralCodecVersion::V1) {
            err = "babylon-ntc-v1 only supports the default network shape; use babylon-ntc-v3";
            return nullptr;
        }
        version = NeuralCodecVersion::V3; // a custom network shape is only representable in v3
    }
    return std::make_unique<BabylonNtc>(preset, version);
}

} // namespace gtc
