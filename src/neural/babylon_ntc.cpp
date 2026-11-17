#include <cstdio>
#include <cstring>

#include "../passes.h"
#include "codec.h"
#include "ntc_cuda.h"

namespace gtc {

namespace {

using tinygltf::Value;

class BabylonNtcV1 : public INeuralCodec {
public:
    explicit BabylonNtcV1(const NeuralOptions& o) {
        cfg_.steps = o.steps;
        cfg_.batch = o.batch;
        cfg_.seed = o.seed;
        cfg_.device = o.cudaDevice;
    }

    std::string name() const override { return "babylon-ntc-v1"; }
    std::string deviceDescription() const override { return "CUDA device " + std::to_string(cfg_.device) + ": " + ntc::deviceName(cfg_.device); }

    bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) override {
        static_assert(ntc::kOutputs == kNeuralChannels);
        ntc::Model model;
        auto t0 = std::chrono::steady_clock::now();
        auto progress = [&](int phase, int step, int total, float loss) {
            char line[200];
            const double el = secondsSince(t0);
            std::snprintf(line, sizeof(line), "    %s %s: step %d/%d  loss %.5f  %.0f s", in.name.c_str(),
                          phase == 0 ? "training" : "fine-tuning", step, total, loss, el);
            progressLine(line);
        };
        if (!ntc::train(in.target, in.width, in.height, in.lossWeights, cfg_, model, progress, err)) return false;
        std::printf("\n");
        if (!ntc::decode(model, cfg_.device, out.decoded, err)) return false;

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

        out.weights.resize(model.weights.size() * sizeof(float));
        std::memcpy(out.weights.data(), model.weights.data(), out.weights.size());

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
        net["outputs"] = Value(ntc::kOutputs);
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
    ntc::TrainConfig cfg_;
};

} // namespace

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err) {
    if (opts.codec == "babylon-ntc-v1") return std::make_unique<BabylonNtcV1>(opts);
    err = "unknown neural codec '" + opts.codec + "' (available: babylon-ntc-v1)";
    return nullptr;
}

} // namespace gtc
