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
constexpr int kRows = 128;            // max rows staged per gradient phase (layer inputs + outputs)
constexpr int kMaxParams = 15360;     // constant memory budget for the weights (60 KiB)
constexpr int kTile = 4;              // texels per side of a sampling-CDF tile

__constant__ float cW[kMaxParams];
__constant__ float cLossW[kOutputs];

// Static layout of an IN -> HID (x NH hidden layers) -> kOutputs MLP: W[in][out] then b, per layer.
template <int IN, int HID, int NH>
struct Net {
    __host__ __device__ static constexpr int layerIn(int l) { return l == 0 ? IN : HID; }
    __host__ __device__ static constexpr int layerOut(int l) { return l == NH ? kOutputs : HID; }
    __host__ __device__ static constexpr int sizeOf(int l) { return layerIn(l) * layerOut(l) + layerOut(l); }
    __host__ __device__ static constexpr int offW(int l) {
        int o = 0;
        for (int m = 0; m < l; ++m) o += sizeOf(m);
        return o;
    }
    __host__ __device__ static constexpr int offB(int l) { return offW(l) + layerIn(l) * layerOut(l); }
    static constexpr int params = offW(NH + 1);
    static constexpr int acc = (params + kThreads - 1) / kThreads;
    static_assert(params <= kMaxParams, "weights do not fit in constant memory");
    static_assert(IN + HID <= kRows && 2 * HID <= kRows && HID + kOutputs <= kRows, "layer staging must fit in shared memory");
};

struct GridK {
    const float* data;
    float* grad;
    int w, h, c, off;
    float levels;
};
struct GridSet {
    GridK g[kMaxGrids];
    int n;
};
struct SampleParams {
    const float* cdf; // per-tile cumulative distribution, nullptr: uniform
    int nTiles, tilesX;
    float texelP;
};

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

__device__ inline void sampleGrid(const GridK& g, const Bilerp& b, float* x) {
    for (int c = 0; c < g.c; ++c) x[c] = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float4* p = reinterpret_cast<const float4*>(g.data + size_t(b.idx[k]) * g.c);
        const float w = b.w[k];
        for (int q = 0; q < g.c / 4; ++q) {
            float4 a = p[q];
            x[4 * q] += w * a.x; x[4 * q + 1] += w * a.y; x[4 * q + 2] += w * a.z; x[4 * q + 3] += w * a.w;
        }
    }
}

// [sin(p.x), sin(p.y), cos(p.x), cos(p.y), sin(2p.x), sin(2p.y), cos(2p.x), cos(2p.y)], p = uv * grid0Res * 2pi.
__device__ inline void posEnc(float u, float v, int gw, int gh, float* x) {
    float fx = u * gw, fy = v * gh;
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

__device__ inline void fillInputs(const GridSet& gs, float u, float v, float* x, Bilerp* bl, int posOff) {
    for (int g = 0; g < gs.n; ++g) {
        bl[g] = bilerp(u, v, gs.g[g].w, gs.g[g].h);
        sampleGrid(gs.g[g], bl[g], x + gs.g[g].off);
    }
    posEnc(u, v, gs.g[0].w, gs.g[0].h, x + posOff);
}

__device__ inline float actf(float z, int mode) {
    if (mode == kRelu) return fmaxf(z, 0.0f);
    if (mode == kSilu) return z / (1.0f + __expf(-z));
    return fminf(z, 3.0f) * fminf(fmaxf(z * (1.0f / 3.0f) + 0.5f, 0.0f), 1.0f);
}

__device__ inline float dactf(float z, int mode) {
    if (mode == kRelu) return z > 0.0f ? 1.0f : 0.0f;
    if (mode == kSilu) {
        float s = 1.0f / (1.0f + __expf(-z));
        return s * (1.0f + z * (1.0f - s));
    }
    return z < -1.5f ? 0.0f : z < 1.5f ? z * (2.0f / 3.0f) + 0.5f : z < 3.0f ? 1.0f : 0.0f;
}

// z: pre-activations, a: activations of every hidden layer, o: kOutputs outputs.
template <int IN, int HID, int NH>
__device__ inline void forwardNet(const float* x, float (&z)[NH][HID], float (&a)[NH][HID], float* o, int mode) {
    using N = Net<IN, HID, NH>;
#pragma unroll
    for (int j = 0; j < HID; ++j) z[0][j] = cW[N::offB(0) + j];
#pragma unroll
    for (int i = 0; i < IN; ++i) {
        const float xi = x[i];
#pragma unroll
        for (int j = 0; j < HID; ++j) z[0][j] = fmaf(xi, cW[N::offW(0) + i * HID + j], z[0][j]);
    }
#pragma unroll
    for (int j = 0; j < HID; ++j) a[0][j] = actf(z[0][j], mode);
#pragma unroll
    for (int l = 1; l < NH; ++l) {
#pragma unroll
        for (int j = 0; j < HID; ++j) z[l][j] = cW[N::offB(l) + j];
#pragma unroll
        for (int i = 0; i < HID; ++i) {
            const float ai = a[l - 1][i];
#pragma unroll
            for (int j = 0; j < HID; ++j) z[l][j] = fmaf(ai, cW[N::offW(l) + i * HID + j], z[l][j]);
        }
#pragma unroll
        for (int j = 0; j < HID; ++j) a[l][j] = actf(z[l][j], mode);
    }
#pragma unroll
    for (int k = 0; k < kOutputs; ++k) o[k] = cW[N::offB(NH) + k];
#pragma unroll
    for (int i = 0; i < HID; ++i) {
        const float ai = a[NH - 1][i];
#pragma unroll
        for (int k = 0; k < kOutputs; ++k) o[k] = fmaf(ai, cW[N::offW(NH) + i * kOutputs + k], o[k]);
    }
}

// Accumulates weight gradients (sum over staged samples of activation x delta) for the
// parameters owned by this thread (p = t + k * kThreads) that belong to layer L.
template <class N, int L>
__device__ inline void accumulate(float* acc, const float* buf, int t) {
    constexpr int begin = N::offW(L);
    constexpr int end = N::offW(L + 1);
    constexpr int nIn = N::layerIn(L);
    constexpr int nOut = N::layerOut(L);
#pragma unroll
    for (int k = begin / kThreads; k <= (end - 1) / kThreads; ++k) {
        int p = t + k * kThreads;
        if (p < begin || p >= end) continue;
        int q = p - begin;
        float s = 0.0f;
        if (q < nIn * nOut) {
            const float* a = buf + (q / nOut) * kStride;
            const float* d = buf + (nIn + q % nOut) * kStride;
#pragma unroll 8
            for (int i = 0; i < kThreads; ++i) s = fmaf(a[i], d[i], s);
        } else {
            const float* d = buf + (nIn + q - nIn * nOut) * kStride;
#pragma unroll 8
            for (int i = 0; i < kThreads; ++i) s += d[i];
        }
        acc[k] += s;
    }
}

template <class N, int L, int IN, int HID, int NH>
__device__ inline void stageAndAccumulate(float* acc, float* buf, int t, const float* x, const float (&a)[NH][HID],
                                          const float (&dz)[NH][HID], const float* dO) {
    constexpr int nIn = N::layerIn(L);
    constexpr int nOut = N::layerOut(L);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < nIn; ++i) buf[i * kStride + t] = L == 0 ? x[i] : a[L == 0 ? 0 : L - 1][i];
#pragma unroll
    for (int j = 0; j < nOut; ++j) buf[(nIn + j) * kStride + t] = L == NH ? dO[j] : dz[L == NH ? 0 : L][j];
    __syncthreads();
    accumulate<N, L>(acc, buf, t);
}

template <int IN, int HID, int NH>
__global__ void __launch_bounds__(kThreads)
    trainKernel(GridSet gs, const float* target, int W, int Ht, SampleParams sp, int batch, unsigned long long seed,
                bool latentGrads, float invNorm, int mode, int lossMode, float eps, float* gParams, float* lossOut) {
    using N = Net<IN, HID, NH>;
    __shared__ float buf[kRows * kStride];
    const int t = threadIdx.x;
    float acc[N::acc];
#pragma unroll
    for (int k = 0; k < N::acc; ++k) acc[k] = 0.0f;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, blockIdx.x * kThreads + t, 0, &rng);
    float lossSum = 0.0f;

    for (int base = blockIdx.x * kThreads; base < batch; base += gridDim.x * kThreads) {
        float x[IN], z[NH][HID], a[NH][HID], dz[NH][HID], dO[kOutputs];
#pragma unroll
        for (int i = 0; i < IN; ++i) x[i] = 0.0f;
#pragma unroll
        for (int l = 0; l < NH; ++l)
#pragma unroll
            for (int i = 0; i < HID; ++i) z[l][i] = a[l][i] = dz[l][i] = 0.0f;
#pragma unroll
        for (int i = 0; i < kOutputs; ++i) dO[i] = 0.0f;

        if (base + t < batch) {
            float4 r = curand_uniform4(&rng);
            float fx = r.x * W, fy = r.y * Ht;
            if (sp.cdf) {
                int lo = 0, hi = sp.nTiles - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (sp.cdf[mid] >= r.z) hi = mid;
                    else lo = mid + 1;
                }
                fx = float((lo % sp.tilesX) * kTile) + r.x * kTile;
                fy = float((lo / sp.tilesX) * kTile) + r.y * kTile;
                fx = fminf(fx, float(W));
                fy = fminf(fy, float(Ht));
            }
            const bool texel = r.w < sp.texelP;
            int ix = 0, iy = 0;
            if (texel) {
                ix = min(int(fx), W - 1);
                iy = min(int(fy), Ht - 1);
                fx = ix + 0.5f;
                fy = iy + 0.5f;
            }
            const float u = fx / W, v = fy / Ht;

            Bilerp bl[kMaxGrids];
            fillInputs(gs, u, v, x, bl, IN - kPosEncInputs);
            float o[kOutputs];
            forwardNet<IN, HID, NH>(x, z, a, o, mode);

            // Continuous samples compare against the bilinearly filtered target, so training covers every UV the GPU can sample.
            float tg[kOutputs];
            if (texel) {
                const float* tp = target + (size_t(iy) * W + ix) * kOutputs;
#pragma unroll
                for (int c = 0; c < kOutputs; ++c) tg[c] = tp[c];
            } else {
                Bilerp bt = bilerp(u, v, W, Ht);
#pragma unroll
                for (int c = 0; c < kOutputs; ++c) tg[c] = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float* tp = target + size_t(bt.idx[k]) * kOutputs;
#pragma unroll
                    for (int c = 0; c < kOutputs; ++c) tg[c] += bt.w[k] * tp[c];
                }
            }
#pragma unroll
            for (int c = 0; c < kOutputs; ++c) {
                float e = o[c] - tg[c];
                if (lossMode == kLossL2) {
                    lossSum += cLossW[c] * e * e;
                    dO[c] = 2.0f * cLossW[c] * e * invNorm;
                } else {
                    float rr = sqrtf(e * e + eps * eps);
                    lossSum += cLossW[c] * (rr - eps);
                    dO[c] = cLossW[c] * e / rr * invNorm;
                }
            }
#pragma unroll
            for (int j = 0; j < HID; ++j) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < kOutputs; ++k) s = fmaf(cW[N::offW(NH) + j * kOutputs + k], dO[k], s);
                dz[NH - 1][j] = s * dactf(z[NH - 1][j], mode);
            }
#pragma unroll
            for (int l = NH - 2; l >= 0; --l) {
#pragma unroll
                for (int i = 0; i < HID; ++i) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < HID; ++j) s = fmaf(cW[N::offW(l + 1) + i * HID + j], dz[l + 1][j], s);
                    dz[l][i] = s * dactf(z[l][i], mode);
                }
            }
            if (latentGrads) {
#pragma unroll
                for (int i = 0; i < IN - kPosEncInputs; ++i) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < HID; ++j) s = fmaf(cW[N::offW(0) + i * HID + j], dz[0][j], s);
                    int g = 0;
                    if (gs.n > 1 && i >= gs.g[1].off) g = 1;
                    if (gs.n > 2 && i >= gs.g[2].off) g = 2;
                    const GridK& gk = gs.g[g];
                    const int c = i - gk.off;
#pragma unroll
                    for (int k = 0; k < 4; ++k) atomicAdd(&gk.grad[size_t(bl[g].idx[k]) * gk.c + c], bl[g].w[k] * s);
                }
            }
        }

        stageAndAccumulate<N, 0, IN, HID, NH>(acc, buf, t, x, a, dz, dO);
        if constexpr (NH > 1) stageAndAccumulate<N, 1, IN, HID, NH>(acc, buf, t, x, a, dz, dO);
        if constexpr (NH > 2) stageAndAccumulate<N, 2, IN, HID, NH>(acc, buf, t, x, a, dz, dO);
        stageAndAccumulate<N, NH, IN, HID, NH>(acc, buf, t, x, a, dz, dO);
    }

#pragma unroll
    for (int k = 0; k < N::acc; ++k) {
        int p = t + k * kThreads;
        if (p < N::params) atomicAdd(&gParams[p], acc[k]);
    }
    for (int off = 16; off > 0; off >>= 1) lossSum += __shfl_down_sync(0xffffffffu, lossSum, off);
    if ((t & 31) == 0) atomicAdd(lossOut, lossSum);
}

// Sum of squared errors over the active channels at every texel center (against the exact texel).
template <int IN, int HID, int NH>
__global__ void __launch_bounds__(128) evalKernel(GridSet gs, const float* target, int W, int Ht, int mode, float* sumOut) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float err = 0.0f;
    if (idx < W * Ht) {
        const float u = (idx % W + 0.5f) / W, v = (idx / W + 0.5f) / Ht;
        float x[IN], z[NH][HID], a[NH][HID], o[kOutputs];
        Bilerp bl[kMaxGrids];
        fillInputs(gs, u, v, x, bl, IN - kPosEncInputs);
        forwardNet<IN, HID, NH>(x, z, a, o, mode);
#pragma unroll
        for (int c = 0; c < kOutputs; ++c)
            if (cLossW[c] > 0.0f) {
                float e = o[c] - target[size_t(idx) * kOutputs + c];
                err += e * e;
            }
    }
    for (int off = 16; off > 0; off >>= 1) err += __shfl_down_sync(0xffffffffu, err, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(sumOut, err);
}

template <int IN, int HID, int NH>
__global__ void __launch_bounds__(128) decodeKernel(GridSet gs, int W, int Ht, int mode, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * Ht) return;
    const float u = (idx % W + 0.5f) / W, v = (idx / W + 0.5f) / Ht;
    float x[IN], z[NH][HID], a[NH][HID], o[kOutputs];
    Bilerp bl[kMaxGrids];
    fillInputs(gs, u, v, x, bl, IN - kPosEncInputs);
    forwardNet<IN, HID, NH>(x, z, a, o, mode);
#pragma unroll
    for (int c = 0; c < kOutputs; ++c) out[size_t(idx) * kOutputs + c] = o[c];
}

__device__ inline uint32_t hash32(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

// dst = clamp(src); noise: + U(-0.5, 0.5) / levels (simulated quantization), ste: round to the quantization grid.
__global__ void noiseKernel(const float* src, float* dst, int n, uint32_t seed, float levels, bool ste) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fminf(fmaxf(src[i], 0.0f), 1.0f);
    if (ste) v = rintf(v * levels) / levels;
    else v += ((hash32(uint32_t(i) ^ hash32(seed)) >> 8) * (1.0f / 16777216.0f) - 0.5f) / levels;
    dst[i] = fminf(fmaxf(v, 0.0f), 1.0f);
}

__global__ void adamKernel(float* p, float* g, float* m, float* v, int n, float lr, float bc1, float bc2, float beta2, bool clamp01) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    g[i] = 0.0f;
    float mi = 0.9f * m[i] + 0.1f * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float r = p[i] - lr * (mi / bc1) / (sqrtf(vi / bc2) + 1e-8f);
    p[i] = clamp01 ? fminf(fmaxf(r, 0.0f), 1.0f) : r;
}

__global__ void quantizeKernel(float* p, int n, float levels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = rintf(fminf(fmaxf(p[i], 0.0f), 1.0f) * levels) / levels;
}

} // namespace

// Kernel launchers of one network shape. makeKernels is instantiated in a TU of its own per shape (ntc_shape_<n>.cu), so the
// slow device compilations run in parallel; every TU keeps its own copy of the constant weights, set through setWeights.
struct Kernels {
    void (*train)(int blocks, const GridSet&, const float*, int, int, const SampleParams&, int, unsigned long long, bool, float, int,
                  int, float, float*, float*);
    void (*eval)(const GridSet&, const float*, int, int, int, float*);
    void (*decode)(const GridSet&, int, int, int, float*);
    cudaError_t (*setWeights)(const float* src, size_t count, bool fromDevice);
    cudaError_t (*setLossWeights)(const float* src);
};

template <int IN, int HID, int NH>
Kernels makeKernels() {
    Kernels k;
    k.train = [](int blocks, const GridSet& gs, const float* target, int W, int Ht, const SampleParams& sp, int batch,
                 unsigned long long seed, bool latentGrads, float invNorm, int mode, int lossMode, float eps, float* gp,
                 float* loss) {
        trainKernel<IN, HID, NH><<<blocks, kThreads>>>(gs, target, W, Ht, sp, batch, seed, latentGrads, invNorm, mode, lossMode, eps, gp, loss);
    };
    k.eval = [](const GridSet& gs, const float* target, int W, int Ht, int mode, float* sum) {
        evalKernel<IN, HID, NH><<<(W * Ht + 127) / 128, 128>>>(gs, target, W, Ht, mode, sum);
    };
    k.decode = [](const GridSet& gs, int W, int Ht, int mode, float* out) {
        decodeKernel<IN, HID, NH><<<(W * Ht + 127) / 128, 128>>>(gs, W, Ht, mode, out);
    };
    k.setWeights = [](const float* src, size_t count, bool fromDevice) {
        return cudaMemcpyToSymbolAsync(cW, src, count * sizeof(float), 0, fromDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
    };
    k.setLossWeights = [](const float* src) { return cudaMemcpyToSymbol(cLossW, src, kOutputs * sizeof(float)); };
    return k;
}

// Network shapes with a compiled kernel: (inputs, hidden width, hidden layers). NTC_ONLY_V2 (dev builds)
// compiles the default v2 shape alone, inside this TU, which takes a fraction of the time.
#ifdef NTC_ONLY_V2
#define NTC_SHAPES(X) X(24, 32, 2)
#else
#define NTC_SHAPES(X) \
    X(24, 32, 2) X(40, 32, 2) X(40, 48, 2) X(40, 64, 2) X(48, 64, 2) X(48, 48, 2) X(56, 64, 2) X(56, 64, 3)
#endif

#ifdef NTC_SHAPE_TU
// ntc_shape_<n>.cu: only the kernels of shape n.
#define NTC_INSTANTIATE(IN, H, NH) template Kernels makeKernels<IN, H, NH>();
#if NTC_SHAPE_TU == 0
NTC_INSTANTIATE(24, 32, 2)
#elif NTC_SHAPE_TU == 1
NTC_INSTANTIATE(40, 32, 2)
#elif NTC_SHAPE_TU == 2
NTC_INSTANTIATE(40, 48, 2)
#elif NTC_SHAPE_TU == 3
NTC_INSTANTIATE(40, 64, 2)
#elif NTC_SHAPE_TU == 4
NTC_INSTANTIATE(48, 64, 2)
#elif NTC_SHAPE_TU == 5
NTC_INSTANTIATE(48, 48, 2)
#elif NTC_SHAPE_TU == 6
NTC_INSTANTIATE(56, 64, 2)
#elif NTC_SHAPE_TU == 7
NTC_INSTANTIATE(56, 64, 3)
#else
#error "unknown NTC_SHAPE_TU"
#endif
#undef NTC_INSTANTIATE
} // namespace gtc::ntc
#else
#ifndef NTC_ONLY_V2
#define NTC_EXTERN(IN, H, NH) extern template Kernels makeKernels<IN, H, NH>();
NTC_SHAPES(NTC_EXTERN)
#undef NTC_EXTERN
#endif

namespace {

bool findKernels(const NetConfig& net, Kernels& k, std::string& err) {
    if (net.grids.empty() || int(net.grids.size()) > kMaxGrids) {
        err = "a network needs 1 to " + std::to_string(kMaxGrids) + " latent grids";
        return false;
    }
    for (const GridDesc& g : net.grids)
        if (g.channels < 4 || g.channels % 4 || (g.bits != 4 && g.bits != 8) || g.div < 1) {
            err = "grids need a channel count that is a multiple of 4, 4 or 8 bits and div >= 1";
            return false;
        }
    const int in = net.inputs();
#define NTC_PICK(IN, H, NH)                                                         \
    if (in == IN && net.hidden == H && net.hiddenLayers == NH) {                    \
        k = makeKernels<IN, H, NH>();                                               \
        return true;                                                                \
    }
    NTC_SHAPES(NTC_PICK)
#undef NTC_PICK
    err = "no kernel for " + std::to_string(in) + " inputs, hidden width " + std::to_string(net.hidden) + " x " +
          std::to_string(net.hiddenLayers) + " layers (add it to NTC_SHAPES in ntc_cuda.cu)";
    return false;
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

float levelsOf(int bits) { return float((1 << bits) - 1); }

// Per-tile sampling distribution from the UV mask (98 / 2 inside / outside) and the gradient energy of the target.
void buildSamplingCdf(const std::vector<float>& target, int W, int H, int outputs, const float* lw, const TrainConfig& cfg,
                      std::vector<float>& cdf, int& tilesX, int& tilesY) {
    const size_t n = size_t(W) * H;
    std::vector<double> texelW(n, 1.0);
    size_t inside = 0;
    const bool useMask = cfg.uvMask && cfg.uvMask->size() == n;
    if (useMask)
        for (uint8_t m : *cfg.uvMask) inside += m ? 1 : 0;
    const bool maskWeights = useMask && inside > 0 && inside < n;
    if (maskWeights)
        for (size_t i = 0; i < n; ++i)
            texelW[i] = (*cfg.uvMask)[i] ? cfg.maskInside / double(inside) : (1.0 - cfg.maskInside) / double(n - inside);
    if (cfg.importance > 0.0f) {
        std::vector<float> energy(n, 0.0f);
        double sum = 0.0, count = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int xm = std::max(0, x - 1), xp = std::min(W - 1, x + 1), ym = std::max(0, y - 1), yp = std::min(H - 1, y + 1);
                float e = 0.0f;
                for (int c = 0; c < outputs; ++c) {
                    if (lw[c] <= 0.0f) continue;
                    e += lw[c] * (std::fabs(target[(size_t(y) * W + xp) * kOutputs + c] - target[(size_t(y) * W + xm) * kOutputs + c]) +
                                  std::fabs(target[(size_t(yp) * W + x) * kOutputs + c] - target[(size_t(ym) * W + x) * kOutputs + c]));
                }
                const size_t i = size_t(y) * W + x;
                energy[i] = e;
                if (!maskWeights || (*cfg.uvMask)[i]) {
                    sum += e;
                    count += 1.0;
                }
            }
        const double mean = std::max(1e-9, sum / std::max(1.0, count));
        for (size_t i = 0; i < n; ++i) texelW[i] *= (1.0 - cfg.importance) + cfg.importance * double(energy[i]) / mean;
    }
    tilesX = (W + kTile - 1) / kTile;
    tilesY = (H + kTile - 1) / kTile;
    std::vector<double> tileW(size_t(tilesX) * tilesY, 0.0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) tileW[size_t(y / kTile) * tilesX + x / kTile] += texelW[size_t(y) * W + x];
    double total = 0.0;
    for (double w : tileW) total += w;
    cdf.resize(tileW.size());
    double run = 0.0;
    for (size_t i = 0; i < tileW.size(); ++i) {
        run += tileW[i];
        cdf[i] = float(run / total);
    }
    cdf.back() = 1.0f;
}

} // namespace

std::string deviceName(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "no CUDA device";
    return prop.name;
}

bool supported(const NetConfig& net, std::string& err) {
    Kernels k;
    return findKernels(net, k, err);
}

bool train(const std::vector<float>& target, int width, int height, int outputs, const float lossWeights[kOutputs],
           const TrainConfig& cfg, Model& out, const Progress& progress, std::string& err) {
    CK(cudaSetDevice(cfg.device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, cfg.device));
    Kernels kern;
    if (!findKernels(cfg.net, kern, err)) return false;
    const int nGrids = int(cfg.net.grids.size());
    const int mode = cfg.net.activation;
    const int params = int(cfg.net.weightCount(kOutputs));
    if (params > kMaxParams) {
        err = "network has " + std::to_string(params) + " weights, the limit is " + std::to_string(kMaxParams);
        return false;
    }

    out.width = width;
    out.height = height;
    out.net = cfg.net;
    out.grids.assign(nGrids, GridData());
    std::vector<int> gSize(nGrids);
    for (int g = 0; g < nGrids; ++g) {
        const GridDesc& d = cfg.net.grids[g];
        GridData& gd = out.grids[g];
        gd.div = d.div;
        gd.channels = d.channels;
        gd.bits = d.bits;
        gd.w = std::max(1, (width + d.div - 1) / d.div);
        gd.h = std::max(1, (height + d.div - 1) / d.div);
        gSize[g] = gd.w * gd.h * gd.channels;
    }

    // Initialization: latents ~ U(0.4, 0.6); MLP like torch.nn.Linear (U(-1/sqrt(fan_in), 1/sqrt(fan_in))).
    std::mt19937 gen(cfg.seed);
    std::uniform_real_distribution<float> lat(0.4f, 0.6f);
    std::vector<std::vector<float>> hLat(nGrids);
    for (int g = 0; g < nGrids; ++g) {
        hLat[g].resize(gSize[g]);
        for (auto& v : hLat[g]) v = lat(gen);
    }
    std::vector<float> hP(params, 0.0f);
    {
        const int in = cfg.net.inputs(), hid = cfg.net.hidden, nh = cfg.net.hiddenLayers;
        size_t off = 0;
        for (int l = 0; l <= nh; ++l) {
            const int fanIn = l == 0 ? in : hid;
            const int fanOut = l == nh ? kOutputs : hid;
            // The last layer is initialized as an N-output layer (the active columns come first), so that a
            // network with N active outputs trains like a dedicated N-output network.
            const int active = l == nh ? outputs : fanOut;
            std::uniform_real_distribution<float> d(-1.0f / std::sqrt(float(fanIn)), 1.0f / std::sqrt(float(fanIn)));
            for (int i = 0; i < fanIn; ++i)
                for (int c = 0; c < active; ++c) hP[off + size_t(i) * fanOut + c] = d(gen);
            for (int c = 0; c < active; ++c) hP[off + size_t(fanIn) * fanOut + c] = d(gen);
            off += size_t(fanIn + 1) * fanOut;
        }
    }

    DeviceArena arena;
    cudaError_t rc = cudaSuccess;
    auto A = [&](size_t n) { return arena.alloc(n, rc); };
    float* dTarget = A(target.size());
    std::vector<float*> dLat(nGrids), dLatN(nGrids), dG(nGrids), dM(nGrids), dV(nGrids);
    for (int g = 0; g < nGrids; ++g) {
        dLat[g] = A(gSize[g]);
        dLatN[g] = A(gSize[g]);
        dG[g] = A(gSize[g]);
        dM[g] = A(gSize[g]);
        dV[g] = A(gSize[g]);
    }
    float *dP = A(params), *dGP = A(params), *dMP = A(params), *dVP = A(params), *dLoss = A(1);
    CK(rc);
    CK(cudaMemcpy(dTarget, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int g = 0; g < nGrids; ++g) CK(cudaMemcpy(dLat[g], hLat[g].data(), gSize[g] * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, hP.data(), params * sizeof(float), cudaMemcpyHostToDevice));
    CK(kern.setLossWeights(lossWeights));

    GridSet gsTrain{}, gsEval{};
    gsTrain.n = gsEval.n = nGrids;
    {
        int off = 0;
        for (int g = 0; g < nGrids; ++g) {
            const GridData& gd = out.grids[g];
            gsTrain.g[g] = {dLatN[g], dG[g], gd.w, gd.h, gd.channels, off, levelsOf(gd.bits)};
            gsEval.g[g] = {dLat[g], nullptr, gd.w, gd.h, gd.channels, off, levelsOf(gd.bits)};
            off += gd.channels;
        }
    }

    // Texel sampling distribution (UV mask / gradient energy).
    SampleParams sp{nullptr, 0, 0, std::clamp(cfg.texelP, 0.0f, 1.0f)};
    size_t maskInside = 0;
    const bool haveMask = cfg.uvMask && cfg.uvMask->size() == size_t(width) * height;
    if (haveMask)
        for (uint8_t m : *cfg.uvMask) maskInside += m ? 1 : 0;
    const bool wantMask = haveMask && maskInside > 0 && maskInside < cfg.uvMask->size();
    if (wantMask || cfg.importance > 0.0f) {
        std::vector<float> cdf;
        int tx = 0, ty = 0;
        buildSamplingCdf(target, width, height, outputs, lossWeights, cfg, cdf, tx, ty);
        float* dCdf = A(cdf.size());
        CK(rc);
        CK(cudaMemcpy(dCdf, cdf.data(), cdf.size() * sizeof(float), cudaMemcpyHostToDevice));
        sp.cdf = dCdf;
        sp.nTiles = int(cdf.size());
        sp.tilesX = tx;
    }

    const int trainBlocks = std::min(blocksFor(cfg.batch, kThreads), prop.multiProcessorCount * 16);
    const float invNorm = 1.0f / (float(cfg.batch) * outputs);
    constexpr int T = 256;
    int activeChannels = 0;
    for (int c = 0; c < kOutputs; ++c) activeChannels += lossWeights[c] > 0.0f ? 1 : 0;
    float* dEval = A(1);
    CK(rc);

    auto evalPsnr = [&](float& psnr) -> bool {
        CK(cudaMemset(dEval, 0, sizeof(float)));
        CK(kern.setWeights(dP, params, true));
        kern.eval(gsEval, dTarget, width, height, mode, dEval);
        float sum = 0.0f;
        CK(cudaMemcpy(&sum, dEval, sizeof(float), cudaMemcpyDeviceToHost));
        const double mse = double(sum) / (double(width) * height * std::max(1, activeChannels));
        psnr = float(10.0 * std::log10(1.0 / std::max(mse, 1e-12)));
        return true;
    };

    const int ftSteps = cfg.finetuneSteps >= 0 ? cfg.finetuneSteps : std::max(1, cfg.steps / 4);
    auto runPhase = [&](int phase, int steps) -> bool {
        const bool latents = phase == 0;
        const int warm = latents ? int(cfg.warmup * steps) : 0;
        for (int it = 0; it < steps; ++it) {
            const uint32_t stepSeed = cfg.seed * 7919u + uint32_t(phase * 1000003 + it);
            float lrMlp, lrLat;
            if (latents) {
                float c = 0.5f * (1.0f + std::cos(3.14159265f * it / steps));
                lrMlp = cfg.lrMin + (cfg.lrMlp - cfg.lrMin) * c;
                lrLat = cfg.lrMin + (cfg.lrLatent - cfg.lrMin) * c;
                if (it < warm) {
                    const float ramp = float(it + 1) / float(warm);
                    lrMlp *= ramp;
                    lrLat *= ramp;
                }
            } else {
                lrMlp = cfg.lrFinetune;
                if (cfg.lrFinetuneEnd >= 0.0f) {
                    float c = 0.5f * (1.0f + std::cos(3.14159265f * it / steps));
                    lrMlp = cfg.lrFinetuneEnd + (cfg.lrFinetune - cfg.lrFinetuneEnd) * c;
                }
                lrLat = 0.0f;
            }
            if (latents)
                for (int g = 0; g < nGrids; ++g)
                    noiseKernel<<<blocksFor(gSize[g], T), T>>>(dLat[g], dLatN[g], gSize[g], stepSeed ^ (0x9e3779b9u * (g + 1)),
                                                               levelsOf(cfg.net.grids[g].bits), cfg.ste);
            CK(kern.setWeights(dP, params, true));
            CK(cudaMemsetAsync(dLoss, 0, sizeof(float)));
            kern.train(trainBlocks, gsTrain, dTarget, width, height, sp, cfg.batch, (unsigned long long)stepSeed, latents, invNorm,
                       mode, cfg.loss, cfg.charbonnierEps, dGP, dLoss);
            const float bc1 = 1.0f - std::pow(0.9f, float(it + 1)), bc2 = 1.0f - std::pow(cfg.beta2, float(it + 1));
            adamKernel<<<blocksFor(params, T), T>>>(dP, dGP, dMP, dVP, params, lrMlp, bc1, bc2, cfg.beta2, false);
            if (latents)
                for (int g = 0; g < nGrids; ++g)
                    adamKernel<<<blocksFor(gSize[g], T), T>>>(dLat[g], dG[g], dM[g], dV[g], gSize[g], lrLat, bc1, bc2, cfg.beta2, true);
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
    if (!evalPsnr(out.psnrFloat)) return false;

    // Quantize latents, then fine-tune the MLP only on the quantized latents.
    for (int g = 0; g < nGrids; ++g) {
        quantizeKernel<<<blocksFor(gSize[g], T), T>>>(dLat[g], gSize[g], levelsOf(cfg.net.grids[g].bits));
        CK(cudaMemcpy(dLatN[g], dLat[g], gSize[g] * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    if (!evalPsnr(out.psnrQuant)) return false;
    CK(cudaMemset(dMP, 0, params * sizeof(float)));
    CK(cudaMemset(dVP, 0, params * sizeof(float)));
    if (ftSteps > 0 && !runPhase(1, ftSteps)) return false;
    CK(cudaDeviceSynchronize());
    if (!evalPsnr(out.psnrFinal)) return false;

    for (int g = 0; g < nGrids; ++g) {
        std::vector<float> f(gSize[g]);
        CK(cudaMemcpy(f.data(), dLat[g], gSize[g] * sizeof(float), cudaMemcpyDeviceToHost));
        const float levels = levelsOf(cfg.net.grids[g].bits);
        out.grids[g].q.resize(f.size());
        for (size_t i = 0; i < f.size(); ++i) out.grids[g].q[i] = uint8_t(std::lround(std::clamp(f[i], 0.0f, 1.0f) * levels));
    }
    out.weights.resize(params);
    CK(cudaMemcpy(out.weights.data(), dP, params * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

bool decode(const Model& model, int device, std::vector<float>& out, std::string& err) {
    CK(cudaSetDevice(device));
    Kernels kern;
    if (!findKernels(model.net, kern, err)) return false;
    const int nGrids = int(model.grids.size());
    DeviceArena arena;
    cudaError_t rc = cudaSuccess;
    GridSet gs{};
    gs.n = nGrids;
    int off = 0;
    for (int g = 0; g < nGrids; ++g) {
        const GridData& gd = model.grids[g];
        const float levels = levelsOf(gd.bits);
        std::vector<float> f(gd.q.size());
        for (size_t i = 0; i < f.size(); ++i) f[i] = gd.q[i] / levels;
        float* d = arena.alloc(f.size(), rc);
        CK(rc);
        CK(cudaMemcpy(d, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
        gs.g[g] = {d, nullptr, gd.w, gd.h, gd.channels, off, levels};
        off += gd.channels;
    }
    const int texels = model.width * model.height;
    float* dOut = arena.alloc(size_t(texels) * kOutputs, rc);
    CK(rc);
    CK(kern.setWeights(model.weights.data(), model.weights.size(), false));
    kern.decode(gs, model.width, model.height, model.net.activation, dOut);
    CK(cudaGetLastError());
    out.resize(size_t(texels) * kOutputs);
    CK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

} // namespace gtc::ntc
#endif // !NTC_SHAPE_TU
