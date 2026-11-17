#include "ntc_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace gtc::ntc {

namespace {

constexpr int kThreads = 64;          // samples per block iteration
constexpr int kStride = kThreads + 1; // shared memory row stride (avoids bank conflicts)
constexpr int kRows = 64;             // max rows staged per gradient phase
constexpr int kAcc = (kParams + kThreads - 1) / kThreads;

__constant__ float cW[kParams];
__constant__ float cLossW[kOutputs];

struct Bilerp {
    int idx[4];
    float w[4];
};

// GL LINEAR + REPEAT addressing.
__device__ inline Bilerp bilerp(float u, float v, int W, int H) {
    float x = u * W - 0.5f, y = v * H - 0.5f;
    float fx0 = floorf(x), fy0 = floorf(y);
    float fx = x - fx0, fy = y - fy0;
    int x0 = int(fx0) % W, y0 = int(fy0) % H;
    if (x0 < 0) x0 += W;
    if (y0 < 0) y0 += H;
    int x1 = x0 + 1 == W ? 0 : x0 + 1;
    int y1 = y0 + 1 == H ? 0 : y0 + 1;
    Bilerp b;
    b.idx[0] = y0 * W + x0; b.w[0] = (1 - fx) * (1 - fy);
    b.idx[1] = y0 * W + x1; b.w[1] = fx * (1 - fy);
    b.idx[2] = y1 * W + x0; b.w[2] = (1 - fx) * fy;
    b.idx[3] = y1 * W + x1; b.w[3] = fx * fy;
    return b;
}

__device__ inline void sampleGrid(const float* g, const Bilerp& b, float* x) {
#pragma unroll
    for (int c = 0; c < kLatentChannels; ++c) x[c] = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4* p = reinterpret_cast<const float4*>(g + size_t(b.idx[k]) * kLatentChannels);
        float4 a = p[0], c = p[1];
        float w = b.w[k];
        x[0] += w * a.x; x[1] += w * a.y; x[2] += w * a.z; x[3] += w * a.w;
        x[4] += w * c.x; x[5] += w * c.y; x[6] += w * c.z; x[7] += w * c.w;
    }
}

// [sin(p.x), sin(p.y), cos(p.x), cos(p.y), sin(2p.x), sin(2p.y), cos(2p.x), cos(2p.y)], p = uv * hiRes * 2pi.
__device__ inline void posEnc(float u, float v, int hiW, int hiH, float* x) {
    float fx = u * hiW, fy = v * hiH;
    fx -= floorf(fx);
    fy -= floorf(fy);
    float s1x, c1x, s1y, c1y, s2x, c2x, s2y, c2y;
    sincospif(2.0f * fx, &s1x, &c1x);
    sincospif(2.0f * fy, &s1y, &c1y);
    sincospif(4.0f * fx, &s2x, &c2x);
    sincospif(4.0f * fy, &s2y, &c2y);
    x[0] = s1x; x[1] = s1y; x[2] = c1x; x[3] = c1y;
    x[4] = s2x; x[5] = s2y; x[6] = c2x; x[7] = c2y;
}

__device__ inline void forward(const float* x, float* a1, float* a2, float* o) {
#pragma unroll
    for (int j = 0; j < kHidden; ++j) {
        float h = cW[kOffB0 + j];
#pragma unroll
        for (int i = 0; i < kInputs; ++i) h = fmaf(x[i], cW[kOffW0 + i * kHidden + j], h);
        a1[j] = fmaxf(h, 0.0f);
    }
#pragma unroll
    for (int j = 0; j < kHidden; ++j) {
        float h = cW[kOffB1 + j];
#pragma unroll
        for (int i = 0; i < kHidden; ++i) h = fmaf(a1[i], cW[kOffW1 + i * kHidden + j], h);
        a2[j] = fmaxf(h, 0.0f);
    }
#pragma unroll
    for (int k = 0; k < kOutputs; ++k) {
        float h = cW[kOffB2 + k];
#pragma unroll
        for (int i = 0; i < kHidden; ++i) h = fmaf(a2[i], cW[kOffW2 + i * kOutputs + k], h);
        o[k] = h;
    }
}

// Accumulates weight gradients (sum over staged samples of activation x delta) for the
// parameters owned by this thread (p = t + k * kThreads) that belong to the given layer.
template <int LAYER>
__device__ inline void accumulate(float* acc, const float* buf, int t) {
    constexpr int begin = LAYER == 0 ? kOffW0 : LAYER == 1 ? kOffW1 : kOffW2;
    constexpr int end = LAYER == 0 ? kOffW1 : LAYER == 1 ? kOffW2 : kParams;
    constexpr int nIn = LAYER == 0 ? kInputs : kHidden;
    constexpr int nOut = LAYER == 2 ? kOutputs : kHidden;
#pragma unroll
    for (int k = 0; k < kAcc; ++k) {
        int p = t + k * kThreads;
        if (p < begin || p >= end) continue;
        int q = p - begin;
        float s = 0.0f;
        if (q < nIn * nOut) {
            const float* a = buf + (q / nOut) * kStride;
            const float* d = buf + (nIn + q % nOut) * kStride;
            for (int i = 0; i < kThreads; ++i) s = fmaf(a[i], d[i], s);
        } else {
            const float* d = buf + (nIn + q - nIn * nOut) * kStride;
            for (int i = 0; i < kThreads; ++i) s += d[i];
        }
        acc[k] += s;
    }
}

__global__ void __launch_bounds__(kThreads) trainKernel(const float* hiN, const float* loN, float* gHi, float* gLo,
                                                        const float* target, int W, int H, int hiW, int hiH, int loW,
                                                        int loH, int batch, unsigned long long seed, bool latentGrads,
                                                        float invNorm, float* gParams, float* lossOut) {
    __shared__ float buf[kRows * kStride];
    const int t = threadIdx.x;
    float acc[kAcc];
#pragma unroll
    for (int k = 0; k < kAcc; ++k) acc[k] = 0.0f;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, blockIdx.x * kThreads + t, 0, &rng);
    float lossSum = 0.0f;

    for (int base = blockIdx.x * kThreads; base < batch; base += gridDim.x * kThreads) {
        float x[kInputs], a1[kHidden], a2[kHidden], d1[kHidden], d2[kHidden], dO[kOutputs];
#pragma unroll
        for (int i = 0; i < kInputs; ++i) x[i] = 0.0f;
#pragma unroll
        for (int i = 0; i < kHidden; ++i) a1[i] = a2[i] = d1[i] = d2[i] = 0.0f;
#pragma unroll
        for (int i = 0; i < kOutputs; ++i) dO[i] = 0.0f;

        if (base + t < batch) {
            float4 r = curand_uniform4(&rng);
            float u = r.x, v = r.y;
            Bilerp bh = bilerp(u, v, hiW, hiH), bl = bilerp(u, v, loW, loH);
            sampleGrid(hiN, bh, x);
            sampleGrid(loN, bl, x + kLatentChannels);
            posEnc(u, v, hiW, hiH, x + 2 * kLatentChannels);
            float o[kOutputs];
            forward(x, a1, a2, o);

            // Target is bilinearly filtered too, so training covers every UV the GPU can sample.
            Bilerp bt = bilerp(u, v, W, H);
            float tg[kOutputs];
#pragma unroll
            for (int c = 0; c < kOutputs; ++c) tg[c] = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float* tp = target + size_t(bt.idx[k]) * kOutputs;
#pragma unroll
                for (int c = 0; c < kOutputs; ++c) tg[c] += bt.w[k] * tp[c];
            }
#pragma unroll
            for (int c = 0; c < kOutputs; ++c) {
                float e = o[c] - tg[c];
                lossSum += cLossW[c] * e * e;
                dO[c] = 2.0f * cLossW[c] * e * invNorm;
            }
#pragma unroll
            for (int j = 0; j < kHidden; ++j) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < kOutputs; ++k) s = fmaf(cW[kOffW2 + j * kOutputs + k], dO[k], s);
                d2[j] = a2[j] > 0.0f ? s : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < kHidden; ++i) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < kHidden; ++j) s = fmaf(cW[kOffW1 + i * kHidden + j], d2[j], s);
                d1[i] = a1[i] > 0.0f ? s : 0.0f;
            }
            if (latentGrads) {
#pragma unroll
                for (int i = 0; i < 2 * kLatentChannels; ++i) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < kHidden; ++j) s = fmaf(cW[kOffW0 + i * kHidden + j], d1[j], s);
                    const Bilerp& b = i < kLatentChannels ? bh : bl;
                    float* g = i < kLatentChannels ? gHi : gLo;
                    int c = i % kLatentChannels;
#pragma unroll
                    for (int k = 0; k < 4; ++k) atomicAdd(&g[size_t(b.idx[k]) * kLatentChannels + c], b.w[k] * s);
                }
            }
        }

        // Layer 0: inputs (rows 0..23) x d1 (rows 24..55).
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kInputs; ++i) buf[i * kStride + t] = x[i];
#pragma unroll
        for (int j = 0; j < kHidden; ++j) buf[(kInputs + j) * kStride + t] = d1[j];
        __syncthreads();
        accumulate<0>(acc, buf, t);
        // Layer 1: a1 (rows 0..31) x d2 (rows 32..63).
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kHidden; ++i) buf[i * kStride + t] = a1[i];
#pragma unroll
        for (int j = 0; j < kHidden; ++j) buf[(kHidden + j) * kStride + t] = d2[j];
        __syncthreads();
        accumulate<1>(acc, buf, t);
        // Layer 2: a2 (rows 0..31) x dO (rows 32..40).
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kHidden; ++i) buf[i * kStride + t] = a2[i];
#pragma unroll
        for (int k = 0; k < kOutputs; ++k) buf[(kHidden + k) * kStride + t] = dO[k];
        __syncthreads();
        accumulate<2>(acc, buf, t);
    }

#pragma unroll
    for (int k = 0; k < kAcc; ++k) {
        int p = t + k * kThreads;
        if (p < kParams) atomicAdd(&gParams[p], acc[k]);
    }
    for (int off = 16; off > 0; off >>= 1) lossSum += __shfl_down_sync(0xffffffffu, lossSum, off);
    if ((t & 31) == 0) atomicAdd(lossOut, lossSum);
}

__device__ inline uint32_t hash32(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

// dst = clamp(src + U(-0.5, 0.5) / 255): simulates 8-bit quantization during training.
__global__ void noiseKernel(const float* src, float* dst, int n, uint32_t seed, bool noise) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    if (noise) v += ((hash32(uint32_t(i) ^ hash32(seed)) >> 8) * (1.0f / 16777216.0f) - 0.5f) / 255.0f;
    dst[i] = fminf(fmaxf(v, 0.0f), 1.0f);
}

__global__ void adamKernel(float* p, float* g, float* m, float* v, int n, float lr, float bc1, float bc2, bool clamp01) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    g[i] = 0.0f;
    float mi = 0.9f * m[i] + 0.1f * gi;
    float vi = 0.999f * v[i] + 0.001f * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float r = p[i] - lr * (mi / bc1) / (sqrtf(vi / bc2) + 1e-8f);
    p[i] = clamp01 ? fminf(fmaxf(r, 0.0f), 1.0f) : r;
}

__global__ void quantizeKernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = rintf(fminf(fmaxf(p[i], 0.0f), 1.0f) * 255.0f) / 255.0f;
}

__global__ void decodeKernel(const float* hi, const float* lo, int W, int H, int hiW, int hiH, int loW, int loH, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    float u = (idx % W + 0.5f) / W, v = (idx / W + 0.5f) / H;
    float x[kInputs], a1[kHidden], a2[kHidden], o[kOutputs];
    sampleGrid(hi, bilerp(u, v, hiW, hiH), x);
    sampleGrid(lo, bilerp(u, v, loW, loH), x + kLatentChannels);
    posEnc(u, v, hiW, hiH, x + 2 * kLatentChannels);
    forward(x, a1, a2, o);
#pragma unroll
    for (int c = 0; c < kOutputs; ++c) out[size_t(idx) * kOutputs + c] = o[c];
}

// Owns device allocations for the lifetime of one call.
struct DeviceArena {
    std::vector<void*> ptrs;
    ~DeviceArena() {
        for (void* p : ptrs) cudaFree(p);
    }
    float* alloc(size_t count, cudaError_t& rc) {
        void* p = nullptr;
        rc = cudaMalloc(&p, count * sizeof(float));
        if (rc != cudaSuccess) return nullptr;
        ptrs.push_back(p);
        cudaMemset(p, 0, count * sizeof(float));
        return static_cast<float*>(p);
    }
};

#define CK(expr)                                                                         \
    do {                                                                                 \
        cudaError_t rc_ = (expr);                                                        \
        if (rc_ != cudaSuccess) {                                                        \
            err = std::string("CUDA error in ") + #expr + ": " + cudaGetErrorString(rc_); \
            return false;                                                                \
        }                                                                                \
    } while (0)

inline int blocksFor(int n, int threads) { return (n + threads - 1) / threads; }

} // namespace

std::string deviceName(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "no CUDA device";
    return prop.name;
}

bool train(const std::vector<float>& target, int width, int height, const float lossWeights[kOutputs],
           const TrainConfig& cfg, Model& out, const Progress& progress, std::string& err) {
    CK(cudaSetDevice(cfg.device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, cfg.device));

    out.width = width;
    out.height = height;
    out.hiW = std::max(1, (width + kHiDiv - 1) / kHiDiv);
    out.hiH = std::max(1, (height + kHiDiv - 1) / kHiDiv);
    out.loW = std::max(1, (width + kLoDiv - 1) / kLoDiv);
    out.loH = std::max(1, (height + kLoDiv - 1) / kLoDiv);
    const int nHi = out.hiW * out.hiH * kLatentChannels, nLo = out.loW * out.loH * kLatentChannels;

    // Initialization: latents ~ U(0.4, 0.6); MLP like torch.nn.Linear (U(-1/sqrt(fan_in), 1/sqrt(fan_in))).
    std::mt19937 gen(cfg.seed);
    std::uniform_real_distribution<float> lat(0.4f, 0.6f);
    std::vector<float> hHi(nHi), hLo(nLo), hP(kParams);
    for (auto& v : hHi) v = lat(gen);
    for (auto& v : hLo) v = lat(gen);
    auto initLayer = [&](int offW, int offB, int fanIn, int fanOut) {
        std::uniform_real_distribution<float> d(-1.0f / std::sqrt(float(fanIn)), 1.0f / std::sqrt(float(fanIn)));
        for (int i = 0; i < fanIn * fanOut; ++i) hP[offW + i] = d(gen);
        for (int i = 0; i < fanOut; ++i) hP[offB + i] = d(gen);
    };
    initLayer(kOffW0, kOffB0, kInputs, kHidden);
    initLayer(kOffW1, kOffB1, kHidden, kHidden);
    initLayer(kOffW2, kOffB2, kHidden, kOutputs);

    DeviceArena arena;
    cudaError_t rc = cudaSuccess;
    auto A = [&](size_t n) { return arena.alloc(n, rc); };
    float *dTarget = A(target.size()), *dHi = A(nHi), *dLo = A(nLo), *dHiN = A(nHi), *dLoN = A(nLo);
    float *dGHi = A(nHi), *dGLo = A(nLo), *dMHi = A(nHi), *dVHi = A(nHi), *dMLo = A(nLo), *dVLo = A(nLo);
    float *dP = A(kParams), *dGP = A(kParams), *dMP = A(kParams), *dVP = A(kParams), *dLoss = A(1);
    CK(rc);
    CK(cudaMemcpy(dTarget, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dHi, hHi.data(), nHi * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dLo, hLo.data(), nLo * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, hP.data(), kParams * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(cLossW, lossWeights, kOutputs * sizeof(float)));

    const int trainBlocks = std::min(blocksFor(cfg.batch, kThreads), prop.multiProcessorCount * 16);
    const float invNorm = 1.0f / (float(cfg.batch) * kOutputs);
    constexpr int T = 256;

    auto runPhase = [&](int phase, int steps) -> bool {
        const bool latents = phase == 0;
        for (int it = 0; it < steps; ++it) {
            const uint32_t stepSeed = cfg.seed * 7919u + uint32_t(phase * 1000003 + it);
            float lrMlp, lrLat;
            if (latents) {
                float c = 0.5f * (1.0f + std::cos(3.14159265f * it / steps));
                lrMlp = cfg.lrMin + (cfg.lrMlp - cfg.lrMin) * c;
                lrLat = cfg.lrMin + (cfg.lrLatent - cfg.lrMin) * c;
            } else {
                lrMlp = cfg.lrFinetune;
                lrLat = 0.0f;
            }
            if (latents) {
                noiseKernel<<<blocksFor(nHi, T), T>>>(dHi, dHiN, nHi, stepSeed, true);
                noiseKernel<<<blocksFor(nLo, T), T>>>(dLo, dLoN, nLo, stepSeed ^ 0x9e3779b9u, true);
            }
            CK(cudaMemcpyToSymbolAsync(cW, dP, kParams * sizeof(float), 0, cudaMemcpyDeviceToDevice));
            CK(cudaMemsetAsync(dLoss, 0, sizeof(float)));
            trainKernel<<<trainBlocks, kThreads>>>(dHiN, dLoN, dGHi, dGLo, dTarget, width, height, out.hiW, out.hiH,
                                                   out.loW, out.loH, cfg.batch, (unsigned long long)stepSeed, latents,
                                                   invNorm, dGP, dLoss);
            const float bc1 = 1.0f - std::pow(0.9f, float(it + 1)), bc2 = 1.0f - std::pow(0.999f, float(it + 1));
            adamKernel<<<blocksFor(kParams, T), T>>>(dP, dGP, dMP, dVP, kParams, lrMlp, bc1, bc2, false);
            if (latents) {
                adamKernel<<<blocksFor(nHi, T), T>>>(dHi, dGHi, dMHi, dVHi, nHi, lrLat, bc1, bc2, true);
                adamKernel<<<blocksFor(nLo, T), T>>>(dLo, dGLo, dMLo, dVLo, nLo, lrLat, bc1, bc2, true);
            }
            CK(cudaGetLastError());
            if (it % 50 == 0 || it + 1 == steps) {
                float loss = 0.0f;
                CK(cudaMemcpy(&loss, dLoss, sizeof(float), cudaMemcpyDeviceToHost));
                if (progress) progress(phase, it + 1, steps, loss * invNorm);
            }
        }
        return true;
    };

    if (!runPhase(0, cfg.steps)) return false;

    // Quantize latents to 8 bits, then fine-tune the MLP only on the quantized latents.
    quantizeKernel<<<blocksFor(nHi, T), T>>>(dHi, nHi);
    quantizeKernel<<<blocksFor(nLo, T), T>>>(dLo, nLo);
    CK(cudaMemcpy(dHiN, dHi, nHi * sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dLoN, dLo, nLo * sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaMemset(dMP, 0, kParams * sizeof(float)));
    CK(cudaMemset(dVP, 0, kParams * sizeof(float)));
    if (!runPhase(1, std::max(1, cfg.steps / 4))) return false;
    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(hHi.data(), dHi, nHi * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hLo.data(), dLo, nLo * sizeof(float), cudaMemcpyDeviceToHost));
    out.weights.resize(kParams);
    CK(cudaMemcpy(out.weights.data(), dP, kParams * sizeof(float), cudaMemcpyDeviceToHost));
    auto toU8 = [](const std::vector<float>& f, std::vector<uint8_t>& u) {
        u.resize(f.size());
        for (size_t i = 0; i < f.size(); ++i) u[i] = uint8_t(std::lround(std::clamp(f[i], 0.0f, 1.0f) * 255.0f));
    };
    toU8(hHi, out.hi);
    toU8(hLo, out.lo);
    return true;
}

bool decode(const Model& model, int device, std::vector<float>& out, std::string& err) {
    CK(cudaSetDevice(device));
    const int nHi = int(model.hi.size()), nLo = int(model.lo.size());
    std::vector<float> hHi(nHi), hLo(nLo);
    for (int i = 0; i < nHi; ++i) hHi[i] = model.hi[i] / 255.0f;
    for (int i = 0; i < nLo; ++i) hLo[i] = model.lo[i] / 255.0f;
    const int texels = model.width * model.height;

    DeviceArena arena;
    cudaError_t rc = cudaSuccess;
    float* dHi = arena.alloc(nHi, rc);
    float* dLo = arena.alloc(nLo, rc);
    float* dOut = arena.alloc(size_t(texels) * kOutputs, rc);
    CK(rc);
    CK(cudaMemcpy(dHi, hHi.data(), nHi * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dLo, hLo.data(), nLo * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(cW, model.weights.data(), kParams * sizeof(float)));
    decodeKernel<<<blocksFor(texels, 256), 256>>>(dHi, dLo, model.width, model.height, model.hiW, model.hiH, model.loW,
                                                  model.loH, dOut);
    CK(cudaGetLastError());
    out.resize(size_t(texels) * kOutputs);
    CK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

} // namespace gtc::ntc
