#include <cstdio>
#include <cstring>

#include "../passes.h"
#include "codec.h"
#include "ntc_cuda.h"

namespace gtc {

namespace {

using tinygltf::Value;

class BabylonNtc : public INeuralCodec {
public:
    BabylonNtc(const NeuralOptions& o, NeuralCodecVersion version) : version_(version) {
        cfg_.steps = o.steps;
        cfg_.batch = o.batch;
        cfg_.seed = o.seed;
        cfg_.device = o.cudaDevice;
    }

    std::string name() const override { return version_ == NeuralCodecVersion::V1 ? "babylon-ntc-v1" : "babylon-ntc-v2"; }
    std::string deviceDescription() const override { return "CUDA device " + std::to_string(cfg_.device) + ": " + ntc::deviceName(cfg_.device); }
    tinygltf::Value::Object describeParameters() const override {
        tinygltf::Value::Object p;
        p["hidden"] = Value(ntc::kHidden);
        p["latentChannels"] = Value(ntc::kLatentChannels);
        p["hiDiv"] = Value(ntc::kHiDiv);
        p["loDiv"] = Value(ntc::kLoDiv);
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

        ntc::Model model;
        auto t0 = std::chrono::steady_clock::now();
        auto progress = [&](int phase, int step, int total, float loss) {
            char line[200];
            const double el = secondsSince(t0);
            std::snprintf(line, sizeof(line), "    %s %s: step %d/%d  loss %.5f  %.0f s", in.name.c_str(),
                          phase == 0 ? "training" : "fine-tuning", step, total, loss, el);
            progressLine(line);
        };
        if (!ntc::train(target, in.width, in.height, N, lossWeights, cfg_, model, progress, err)) return false;
        std::printf("\n");
        std::vector<float> full;
        if (!ntc::decode(model, cfg_.device, full, err)) return false;
        out.decoded.resize(texels * N);
        for (size_t i = 0; i < texels; ++i) std::memcpy(&out.decoded[i * N], &full[i * ntc::kOutputs], N * sizeof(float));

        // Each 8-channel grid becomes two RGBA images.
        auto split = [&](const std::vector<uint8_t>& grid, int w, int h) {
            for (int half = 0; half < 2; ++half) {
                Image img;
                img.width = w;
                img.height = h;
                img.rgba.resize(size_t(w) * h * 4);
                for (size_t i = 0, n = size_t(w) * h; i < n; ++i)
                    std::memcpy(&img.rgba[i * 4], &grid[i * ntc::kLatentChannels + half * 4], 4);
                out.latentImages.push_back(std::move(img));
            }
        };
        split(model.hi, model.hiW, model.hiH);
        split(model.lo, model.loW, model.loH);

        // Weights: W0, b0, W1, b1 as trained, then only the first N columns of W2 and b2.
        std::vector<float> weights;
        weights.reserve(ntc::kOffW2 + (ntc::kHidden + 1) * N);
        weights.insert(weights.end(), model.weights.begin(), model.weights.begin() + ntc::kOffW2);
        for (int k = 0; k < ntc::kHidden; ++k)
            for (int c = 0; c < N; ++c) weights.push_back(model.weights[ntc::kOffW2 + k * ntc::kOutputs + c]);
        for (int c = 0; c < N; ++c) weights.push_back(model.weights[ntc::kOffB2 + c]);
        out.weights.resize(weights.size() * sizeof(float));
        std::memcpy(out.weights.data(), weights.data(), out.weights.size());

        auto arr = [](std::initializer_list<int> v) {
            Value::Array a;
            for (int x : v) a.emplace_back(x);
            return Value(std::move(a));
        };
        Value::Object pe;
        pe["frequencies"] = arr({1, 2});
        pe["scale"] = Value(std::string("hiResolution"));
        Value::Object net;
        net["inputs"] = Value(ntc::kInputs);
        net["hidden"] = arr({ntc::kHidden, ntc::kHidden});
        net["outputs"] = Value(N);
        net["activation"] = Value(std::string("relu"));
        net["positionalEncoding"] = Value(std::move(pe));
        net["weightLayout"] = Value(std::string("W0[in][out],b0,W1,b1,W2,b2 float32"));
        out.parameters["hiResolution"] = arr({model.hiW, model.hiH});
        out.parameters["loResolution"] = arr({model.loW, model.loH});
        out.parameters["latentChannels"] = Value(ntc::kLatentChannels);
        out.parameters["network"] = Value(std::move(net));
        return true;
    }

private:
    NeuralCodecVersion version_;
    ntc::TrainConfig cfg_;
};

} // namespace

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err) {
    if (opts.codec == "babylon-ntc-v1") return std::make_unique<BabylonNtc>(opts, NeuralCodecVersion::V1);
    if (opts.codec == "babylon-ntc-v2") return std::make_unique<BabylonNtc>(opts, NeuralCodecVersion::V2);
    err = "unknown neural codec '" + opts.codec + "' (available: babylon-ntc-v1, babylon-ntc-v2)";
    return nullptr;
}

} // namespace gtc
