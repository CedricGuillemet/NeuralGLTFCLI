// babylon-ntc-v4 codec: trains with the CUDA trainer (ntc_v4_cuda.cu), writes the pyramid atlases, the fp16 weights and the
// extension fields, and verifies the result with the CPU reference decoder (ntc_v4_ref.cpp).
// Clean-room design after Vaidyanathan et al. 2023 (arXiv 2305.17105); ours.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

#include "../passes.h"
#include "codec.h"
#include "ntc_v4.h"
#include "ntc_v4_cuda.h"
#include "preset.h"

namespace gtc {

namespace {

using tinygltf::Value;
using namespace v4;

struct GroupRange {
    const char* name;
    int first, count;
};
constexpr GroupRange kGroupRanges[] = {{"base", 0, 4},  {"normal", 4, 2},   {"occlusion", 6, 1}, {"roughness", 7, 1}, {"metallic", 8, 1},
                                       {"emissive", 9, 3}, {"diffuse", 12, 4}, {"specular", 16, 3}, {"glossiness", 19, 1}};

double psnrOf(double mse) { return mse <= 1e-12 ? 99.0 : std::min(99.0, -10.0 * std::log10(mse)); }

void boxDownFloat(const std::vector<float>& src, int w, int h, int c, std::vector<float>& dst) {
    const int w2 = std::max(1, w >> 1), h2 = std::max(1, h >> 1);
    dst.assign(size_t(w2) * h2 * c, 0.0f);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) {
            const int x1 = std::min(2 * x + 1, w - 1), y1 = std::min(2 * y + 1, h - 1);
            for (int k = 0; k < c; ++k)
                dst[(size_t(y) * w2 + x) * c + k] = 0.25f * (src[(size_t(2 * y) * w + 2 * x) * c + k] + src[(size_t(2 * y) * w + x1) * c + k] +
                                                             src[(size_t(y1) * w + 2 * x) * c + k] + src[(size_t(y1) * w + x1) * c + k]);
        }
}

void boxDownMask(const std::vector<uint8_t>& src, int w, int h, std::vector<uint8_t>& dst) {
    const int w2 = std::max(1, w >> 1), h2 = std::max(1, h >> 1);
    dst.assign(size_t(w2) * h2, 0);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) {
            const int x1 = std::min(2 * x + 1, w - 1), y1 = std::min(2 * y + 1, h - 1);
            dst[size_t(y) * w2 + x] = (src[size_t(2 * y) * w + 2 * x] | src[size_t(2 * y) * w + x1] | src[size_t(y1) * w + 2 * x] | src[size_t(y1) * w + x1]) ? 1 : 0;
        }
}

// Trainer buffers hold the levels one after another, each [y][x][c]; the file format uses the atlas.
void levelsToAtlas(Model& m, int g, const std::vector<uint8_t>& levels) {
    const GridSpec& gs = m.g[g];
    int aw, ah;
    atlasSize(gs, m.L, aw, ah);
    m.q[g].assign(size_t(aw) * ah * gs.c, 0);
    size_t off = 0;
    for (int lv = 0; lv < m.L; ++lv) {
        int w, h, ox, oy;
        levelSize(gs, lv, w, h);
        atlasOrigin(gs, lv, ox, oy);
        for (int y = 0; y < h; ++y) {
            std::memcpy(&m.q[g][(size_t(oy + y) * aw + ox) * gs.c], &levels[off], size_t(w) * gs.c);
            off += size_t(w) * gs.c;
        }
    }
}

std::vector<uint8_t> atlasToLevels(const Model& m, int g) {
    const GridSpec& gs = m.g[g];
    int aw, ah;
    atlasSize(gs, m.L, aw, ah);
    std::vector<uint8_t> out;
    for (int lv = 0; lv < m.L; ++lv) {
        int w, h, ox, oy;
        levelSize(gs, lv, w, h);
        atlasOrigin(gs, lv, ox, oy);
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = &m.q[g][(size_t(oy + y) * aw + ox) * gs.c];
            out.insert(out.end(), row, row + size_t(w) * gs.c);
        }
    }
    return out;
}

size_t pngBytes(const Model& m, int g) {
    size_t n = 0;
    for (const Image& im : packAtlas(m, g)) n += encodePng(im, 4).size();
    return n;
}

class BabylonNtcV4 : public INeuralCodec {
public:
    explicit BabylonNtcV4(const V4Preset& p) : p_(p) {}

    std::string name() const override { return "babylon-ntc-v4"; }
    std::string deviceDescription() const override { return "CUDA device " + std::to_string(p_.device) + ": " + v4cuda::deviceName(p_.device); }
    size_t expectedWeightCount(int) const override { return 0; }

    tinygltf::Value::Object describeParameters() const override {
        Value::Object o;
        o["preset"] = Value(p_.name);
        o["div0"] = Value(p_.div0);
        o["channels0"] = Value(p_.c0);
        o["bits0"] = Value(p_.b0);
        o["channels1"] = Value(p_.c1);
        o["bits1"] = Value(p_.b1);
        o["stepsA"] = Value(p_.stepsA);
        o["stepsB"] = Value(p_.stepsB);
        o["batch"] = Value(p_.batch);
        o["activation"] = Value(std::string(p_.activation == 0 ? "hardgelu" : p_.activation == 1 ? "hardswish" : "silu"));
        o["lrLatent"] = Value(double(p_.lrLatent));
        o["lrMlp"] = Value(double(p_.lrMlp));
        o["lrB"] = Value(double(p_.lrB));
        o["initLatent"] = Value(double(p_.initLatent));
        o["uniformMip"] = Value(double(p_.uniformMip));
        o["maskInside"] = Value(double(p_.inside));
        o["noise"] = Value(double(p_.noise));
        o["mip0"] = Value(double(p_.mip0));
        o["levelCap"] = Value(p_.levelCap);
        o["constElision"] = Value(p_.constElision);
        Value::Array sig;
        for (int c = 0; c < ntc::kOutputs; ++c)
            if (p_.sigmoid[c]) sig.emplace_back(c);
        o["sigmoidChannels"] = Value(std::move(sig));
        return o;
    }

    bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) override {
        const auto tStart = std::chrono::steady_clock::now();
        const int N = in.outputs, W = in.width, H = in.height;
        const size_t texels = size_t(W) * H;
        if (N < 1 || N > kMaxNeuralChannels || in.target.size() != texels * N) {
            err = "unsupported output count or target size for babylon-ntc-v4";
            return false;
        }
        if (W % 8 || H % 8) {
            err = "babylon-ntc-v4 needs a resolution that is a multiple of 8 (got " + std::to_string(W) + "x" + std::to_string(H) + ")";
            return false;
        }
        const bool haveMask = in.uvMask.size() == texels;
        std::vector<uint8_t> mask = haveMask ? in.uvMask : std::vector<uint8_t>(texels, 1);

        // Constant channels: max - min <= 1/255 over the masked texels; they cost nothing in the network.
        Model model;
        model.W = W;
        model.H = H;
        model.levelCap = p_.levelCap;
        model.g[0] = {W / p_.div0, H / p_.div0, p_.c0, p_.b0, true};
        model.g[1] = {W / p_.div0 / 2, H / p_.div0 / 2, p_.c1, p_.b1, false};
        std::vector<int> netOf;
        model.channels.resize(N);
        {
            std::vector<float> lo(N, 2.0f), hi(N, -1.0f);
            std::vector<double> sum(N, 0.0);
            size_t cnt = 0;
            for (size_t t = 0; t < texels; ++t) {
                if (!mask[t]) continue;
                ++cnt;
                for (int c = 0; c < N; ++c) {
                    const float v = in.target[t * N + c];
                    lo[c] = std::min(lo[c], v), hi[c] = std::max(hi[c], v), sum[c] += v;
                }
            }
            int nNet = 0;
            for (int c = 0; c < N; ++c) {
                ChannelSpec& ch = model.channels[c];
                const bool isConst = p_.constElision && cnt > 0 && hi[c] - lo[c] <= 1.0f / 255.0f + 1e-6f && in.lossWeights[c] > 0.0f;
                if (isConst) {
                    ch.isConst = true;
                    ch.value = std::round(float(sum[c] / double(cnt)) * 255.0f) / 255.0f;
                } else {
                    ch.net = nNet++;
                    ch.sigmoid = p_.sigmoid[in.layout.canonical[c]];
                }
            }
            if (nNet == 0) { // keep one network channel so that the network has an output
                ChannelSpec& ch = model.channels[0];
                ch = ChannelSpec{};
                ch.net = 0;
                ch.sigmoid = p_.sigmoid[in.layout.canonical[0]];
            }
        }
        if (!model.finalize(err)) return false;
        const int nNet = model.nNet;

        std::vector<int> netChannel; // network output -> logical channel
        for (int c = 0; c < N; ++c)
            if (!model.channels[c].isConst) netChannel.push_back(c);

        v4cuda::Shape s;
        s.W = W, s.H = H, s.M = model.M, s.L = model.L;
        for (int g = 0; g < 2; ++g) s.gw[g] = model.g[g].w, s.gh[g] = model.g[g].h, s.gc[g] = model.g[g].c, s.gbits[g] = model.g[g].bits;
        s.nIn = model.inputs();
        s.stride0 = model.layers[0].inStride;
        s.nNet = nNet;
        if (nNet > 32) {
            err = "too many network outputs";
            return false;
        }
        for (int k = 0; k < nNet; ++k) s.sigmoid[k] = model.channels[netChannel[k]].sigmoid;

        std::vector<float> target(texels * nNet);
        float lossWeights[32]{};
        for (int k = 0; k < nNet; ++k) lossWeights[k] = in.lossWeights[netChannel[k]] > 0 ? in.lossWeights[netChannel[k]] : 1.0f;
        for (size_t t = 0; t < texels; ++t)
            for (int k = 0; k < nNet; ++k) target[t * nNet + k] = in.target[t * N + netChannel[k]];

        v4cuda::Params tp;
        tp.stepsA = p_.stepsA, tp.stepsB = p_.stepsB, tp.batch = p_.batch, tp.device = p_.device, tp.seed = p_.seed;
        tp.lrLatent = p_.lrLatent, tp.lrMlp = p_.lrMlp, tp.lrB = p_.lrB;
        tp.activation = p_.activation, tp.uniformMipP = p_.uniformMip, tp.insideP = p_.inside, tp.initLatent = p_.initLatent, tp.noiseScale = p_.noise, tp.mip0P = p_.mip0;

        auto t0 = std::chrono::steady_clock::now();
        auto progress = [&](int phase, int step, int total, float loss) {
            char line[200];
            std::snprintf(line, sizeof(line), "    %s %s: step %d/%d  loss %.5f  %.0f s", in.name.c_str(), phase == 0 ? "training" : "fine-tuning", step, total, loss,
                          secondsSince(t0));
            progressLine(line);
        };
        v4cuda::Result res;
        if (!v4cuda::train(s, target, lossWeights, haveMask ? mask : std::vector<uint8_t>{}, tp, res, progress, err)) return false;
        std::printf("\n");

        // fp16 weights, atlases.
        for (int l = 0; l < 3; ++l) {
            model.w[l] = res.net.w[l];
            model.b[l] = res.net.b[l];
            for (float& v : model.w[l]) v = halfToFloat(floatToHalf(v));
            for (float& v : model.b[l]) v = halfToFloat(floatToHalf(v));
        }
        for (int g = 0; g < 2; ++g) levelsToAtlas(model, g, res.q[g]);
        const size_t pngBefore = pngBytes(model, 0) + pngBytes(model, 1);
        const size_t zeroed = zeroDeadCells(model, mask);
        const size_t pngAfter = pngBytes(model, 0) + pngBytes(model, 1);

        // The written bytes must decode to the same model.
        const std::vector<uint8_t> weightBytes = serializeWeights(model);
        {
            Model chk = model;
            if (!parseWeights(weightBytes.data(), weightBytes.size(), chk, err)) return false;
            if (chk.w[0] != model.w[0] || chk.w[1] != model.w[1] || chk.w[2] != model.w[2] || chk.b[2] != model.b[2]) {
                err = "internal error: fp16 weight serialisation does not round-trip";
                return false;
            }
        }

        RefDecoder ref(model);
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        double consistency = 0.0;
        if (!p_.noRef) {
            // GPU decode of the final model vs the CPU reference on random samples.
            std::mt19937 rng(12345);
            std::vector<int> samples;
            constexpr int kSamples = 4096;
            for (int k = 0; k < kSamples; ++k) {
                const int m = int(rng() % uint32_t(model.M));
                samples.push_back(m);
                samples.push_back(int(rng() % uint32_t(std::max(1, W >> m))));
                samples.push_back(int(rng() % uint32_t(std::max(1, H >> m))));
            }
            std::vector<uint8_t> q[2] = {atlasToLevels(model, 0), atlasToLevels(model, 1)};
            v4cuda::Weights gw;
            for (int l = 0; l < 3; ++l) gw.w[l] = model.w[l], gw.b[l] = model.b[l];
            std::vector<float> gpu;
            if (!v4cuda::decodeSamples(s, q, gw, p_.activation, p_.device, samples, gpu, err)) return false;
            std::vector<float> texel(N);
            for (int k = 0; k < kSamples; ++k) {
                ref.decodeTexel(samples[3 * k], samples[3 * k + 1], samples[3 * k + 2], texel.data());
                for (int n = 0; n < nNet; ++n) consistency = std::max(consistency, double(std::fabs(gpu[size_t(k) * nNet + n] - texel[netChannel[n]])));
            }
            if (!(consistency <= 1e-3)) {
                err = "GPU decode and the CPU reference decoder disagree (max |diff| " + std::to_string(consistency) + ")";
                return false;
            }
        }

        // Reference decode of the first mips: all reported PSNRs come from the written bytes.
        const auto tRef = std::chrono::steady_clock::now();
        const int mipsChecked = std::min(4, model.M);
        std::vector<float> target0 = in.target, nextTarget;
        std::vector<uint8_t> maskCur = mask, maskNext;
        Value::Object perMip;
        double mseRefNet = 0.0;
        for (int m = 0; m < mipsChecked; ++m) {
            const int wm = std::max(1, W >> m), hm = std::max(1, H >> m);
            std::vector<float> dec;
            if (p_.noRef) {
                dec.assign(size_t(wm) * hm * N, 0.0f);
            } else {
                ref.decodeMip(m, int(threads), dec);
            }
            Value::Object groups;
            for (const GroupRange& gr : kGroupRanges) {
                double se = 0.0;
                size_t cnt = 0;
                for (size_t t = 0; t < size_t(wm) * hm; ++t) {
                    if (!maskCur[t]) continue;
                    for (int c = 0; c < N; ++c) {
                        const int canon = in.layout.canonical[c];
                        if (canon < gr.first || canon >= gr.first + gr.count) continue;
                        const double d = double(dec[t * N + c]) - double(target0[t * N + c]);
                        se += d * d;
                        ++cnt;
                    }
                }
                if (cnt) groups[gr.name] = Value(psnrOf(se / double(cnt)));
            }
            if (m == 0) {
                double se = 0.0;
                size_t cnt = 0;
                for (size_t t = 0; t < texels; ++t) {
                    if (!mask[t]) continue;
                    for (int k = 0; k < nNet; ++k) {
                        const double d = double(dec[t * N + netChannel[k]]) - double(target0[t * N + netChannel[k]]);
                        se += d * d;
                    }
                    cnt += nNet;
                }
                mseRefNet = cnt ? se / double(cnt) : 0.0;
                out.decoded = dec;
            }
            perMip["mip" + std::to_string(m)] = Value(std::move(groups));
            if (m + 1 < mipsChecked) {
                boxDownFloat(target0, wm, hm, N, nextTarget);
                boxDownMask(maskCur, wm, hm, maskNext);
                target0.swap(nextTarget);
                maskCur.swap(maskNext);
            }
        }
        const double refSeconds = secondsSince(tRef);

        auto meanMse = [&](const std::vector<double>& v) {
            double t = 0.0;
            for (double x : v) t += x;
            return v.empty() ? 0.0 : t / double(v.size());
        };
        out.extraMetrics["psnrFloat"] = Value(psnrOf(meanMse(res.mseFloat)));
        out.extraMetrics["psnrQuantPre"] = Value(psnrOf(meanMse(res.mseQuantPre)));
        out.extraMetrics["psnrFinal"] = Value(psnrOf(meanMse(res.mseFinal)));
        out.extraMetrics["psnrQuantFp16W"] = Value(psnrOf(mseRefNet));
        out.extraMetrics["maskedPsnrByMip"] = Value(std::move(perMip));
        out.extraMetrics["netChannels"] = Value(nNet);
        Value::Array constCh;
        for (int c = 0; c < N; ++c)
            if (model.channels[c].isConst) constCh.emplace_back(c);
        out.extraMetrics["constChannels"] = Value(std::move(constCh));
        out.extraMetrics["consistencyMaxAbs"] = Value(consistency);
        out.extraMetrics["deadCellsZeroed"] = Value(double(zeroed));
        out.extraMetrics["latentPngBytesBeforeZeroing"] = Value(double(pngBefore));
        out.extraMetrics["latentPngBytesAfterZeroing"] = Value(double(pngAfter));
        out.extraMetrics["trainSeconds"] = Value(res.seconds);
        out.extraMetrics["refDecodeSeconds"] = Value(refSeconds);
        out.extraMetrics["levels"] = Value(model.L);
        out.extraMetrics["bppc"] = Value(8.0 * double(pngAfter + weightBytes.size()) / (double(texels) * N));
        (void)tStart;

        for (int g = 0; g < 2; ++g) {
            std::vector<Image> imgs = packAtlas(model, g);
            for (size_t i = 0; i < imgs.size(); ++i) {
                out.latentImages.push_back(std::move(imgs[i]));
                out.latentNames.push_back("g" + std::to_string(g) + "_" + std::to_string(i));
            }
        }
        out.weights = weightBytes;
        out.weightsFp16 = true;
        out.parameters = v4Parameters(model);
        out.latentPointSampling = true;
        return true;
    }

private:
    V4Preset p_;
};

} // namespace

std::unique_ptr<INeuralCodec> createBabylonNtcV4Codec(const NeuralOptions& opts, std::string& err) {
    V4Preset p;
    if (!resolveV4Preset(opts, p, err)) return nullptr;
    return std::make_unique<BabylonNtcV4>(p);
}

} // namespace gtc
