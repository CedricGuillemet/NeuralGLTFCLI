// babylon-ntc-v4 trainer: batched forward/backward with cuBLAS SGEMM (fp32) plus small custom kernels.
// Clean-room implementation of the training recipe of Vaidyanathan et al. 2023 (arXiv 2305.17105): quantisation noise on
// the latents, frozen quantised latents while the network is fine-tuned, Adam with a cosine schedule. ours.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "ntc_v4_cuda.h"

namespace gtc::v4cuda {

namespace {

struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
#define CK(x)                                                                                                  \
    do {                                                                                                       \
        cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) throw CudaError(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)
#define CB(x)                                                                                                  \
    do {                                                                                                       \
        cublasStatus_t s_ = (x);                                                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw CudaError(std::string(#x) + ": cuBLAS error " + std::to_string(int(s_))); \
    } while (0)

constexpr int kH = 64;

struct Geom {
    int W, H, M, L, nNet, stride0, nIn, c0, c1;
    int gw[2][kMaxLevels], gh[2][kMaxLevels], goff[2][kMaxLevels];
    int mipOff[kMaxMips];
    int maskOff[kMaxMips], maskCnt[kMaxMips];
    float sig[32], lw[32];
    float invNorm;
};
__constant__ Geom g;

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__device__ __forceinline__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) { return hash32(a ^ hash32(b + 0x9e3779b9U ^ hash32(c))); }
__device__ __forceinline__ float u01(uint32_t h) { return float(h >> 8) * (1.0f / 16777216.0f); }

__device__ __forceinline__ float fractf(float t) { return t - floorf(t); }
__device__ __forceinline__ float triC(float t) { return 4.0f * fabsf(fractf(t) - 0.5f) - 1.0f; }
__device__ __forceinline__ float triS(float t) { return triC(t - 0.25f); }
__device__ __forceinline__ int wrapi(int v, int n) {
    v %= n;
    return v < 0 ? v + n : v;
}
__device__ __forceinline__ int levelOfDev(int m) { return min(g.L - 1, max(0, (m >> 1) - 1)); }

struct Tap {
    int w, h, off, cx, cy;
    float fx, fy;
};
__device__ __forceinline__ Tap tapOf(int gi, int m, int i, int j) {
    const int lv = levelOfDev(m);
    const int Wm = max(1, g.W >> m), Hm = max(1, g.H >> m);
    Tap t;
    t.w = g.gw[gi][lv];
    t.h = g.gh[gi][lv];
    t.off = g.goff[gi][lv];
    const float rx = float(t.w) / float(Wm), ry = float(t.h) / float(Hm);
    const float gx = __fadd_rn(__fmul_rn(__fadd_rn(float(i), 0.5f), rx), -0.5f);
    const float gy = __fadd_rn(__fmul_rn(__fadd_rn(float(j), 0.5f), ry), -0.5f);
    t.cx = int(floorf(gx));
    t.cy = int(floorf(gy));
    t.fx = gx - float(t.cx);
    t.fy = gy - float(t.cy);
    return t;
}

__device__ __forceinline__ float activate(float z, int act) {
    if (act == 0) return z < -1.5f ? 0.0f : (z > 1.5f ? z : z * (z + 1.5f) / 3.0f);
    if (act == 1) return z * fminf(fmaxf(z + 3.0f, 0.0f), 6.0f) / 6.0f;
    if (act == 2) return z / (1.0f + __expf(-z));
    return z;
}
__device__ __forceinline__ float dActivate(float z, int act) {
    if (act == 0) return z < -1.5f ? 0.0f : (z > 1.5f ? 1.0f : (2.0f * z + 1.5f) / 3.0f);
    if (act == 1) return z < -3.0f ? 0.0f : (z > 3.0f ? 1.0f : (2.0f * z + 3.0f) / 6.0f);
    if (act == 2) {
        const float s = 1.0f / (1.0f + __expf(-z));
        return s * (1.0f + z * (1.0f - s));
    }
    return 1.0f;
}

__global__ void sampleK(int* smp, int n, int m, uint32_t seed, uint32_t step, const int* maskIdx, float pInside, int seq, int start) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int Wm = max(1, g.W >> m), Hm = max(1, g.H >> m);
    int idx;
    if (seq) {
        idx = start + s;
    } else {
        const uint32_t h = hash3(seed, step, uint32_t(s));
        const float u = u01(hash32(h ^ 0xa511e9b3U));
        const int cnt = g.maskCnt[m];
        if (cnt > 0 && u < pInside) idx = maskIdx[g.maskOff[m] + int(h % uint32_t(cnt))];
        else idx = int(h % uint32_t(Wm * Hm));
    }
    smp[3 * s] = m;
    smp[3 * s + 1] = idx % Wm;
    smp[3 * s + 2] = idx / Wm;
}

__global__ void noiseK(const float* lat, float* out, size_t n, float amp, uint32_t seed, uint32_t step) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = lat[i] + (u01(hash3(seed, step, uint32_t(i))) - 0.5f) * amp;
}

__global__ void quantK(const float* lat, float* outVal, uint8_t* outQ, size_t n, int bits) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sc = float(1 << bits);
    const int zq = (1 << (bits - 1)) - 1;
    int q = int(rintf(lat[i] * sc)) + zq;
    q = min(max(q, 0), (1 << bits) - 1);
    outQ[i] = uint8_t(q);
    outVal[i] = float(q - zq) / sc;
}

__global__ void dequantK(const uint8_t* q, float* val, size_t n, int bits) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val[i] = float(int(q[i]) - ((1 << (bits - 1)) - 1)) / float(1 << bits);
}

__global__ void gatherK(const int* smp, const float* lat0, const float* lat1, float* X, int n) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int m = smp[3 * s], i = smp[3 * s + 1], j = smp[3 * s + 2];
    float* x = X + size_t(s) * g.stride0;
    int k = 0;
    {
        const Tap t = tapOf(0, m, i, j);
        for (int tp = 0; tp < 4; ++tp) {
            const int cx = wrapi(t.cx + (tp & 1), t.w), cy = wrapi(t.cy + (tp >> 1), t.h);
            const float* src = lat0 + t.off + (size_t(cy) * t.w + cx) * g.c0;
            for (int c = 0; c < g.c0; ++c) x[k++] = src[c];
        }
    }
    {
        const Tap t = tapOf(1, m, i, j);
        const int x0 = wrapi(t.cx, t.w), x1 = wrapi(t.cx + 1, t.w), y0 = wrapi(t.cy, t.h), y1 = wrapi(t.cy + 1, t.h);
        const float* a = lat1 + t.off + (size_t(y0) * t.w + x0) * g.c1;
        const float* b = lat1 + t.off + (size_t(y0) * t.w + x1) * g.c1;
        const float* d = lat1 + t.off + (size_t(y1) * t.w + x0) * g.c1;
        const float* e = lat1 + t.off + (size_t(y1) * t.w + x1) * g.c1;
        for (int c = 0; c < g.c1; ++c) x[k++] = (1.0f - t.fy) * ((1.0f - t.fx) * a[c] + t.fx * b[c]) + t.fy * ((1.0f - t.fx) * d[c] + t.fx * e[c]);
    }
    for (int axis = 0; axis < 2; ++axis) {
        const float p = float(axis == 0 ? i : j);
        x[k++] = triC(p / 8.0f);
        x[k++] = triS(p / 8.0f);
        x[k++] = triC(p / 4.0f);
        x[k++] = triS(p / 4.0f);
        x[k++] = triC(p / 2.0f);
        x[k++] = triS(p / 2.0f);
    }
    x[k++] = float(m) / float(max(1, g.M - 1));
}

__global__ void scatterK(const int* smp, const float* dX, float* g0, float* g1, int n) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int m = smp[3 * s], i = smp[3 * s + 1], j = smp[3 * s + 2];
    const float* d = dX + size_t(s) * g.stride0;
    {
        const Tap t = tapOf(0, m, i, j);
        for (int tp = 0; tp < 4; ++tp) {
            const int cx = wrapi(t.cx + (tp & 1), t.w), cy = wrapi(t.cy + (tp >> 1), t.h);
            float* dst = g0 + t.off + (size_t(cy) * t.w + cx) * g.c0;
            for (int c = 0; c < g.c0; ++c) atomicAdd(&dst[c], d[tp * g.c0 + c]);
        }
    }
    {
        const Tap t = tapOf(1, m, i, j);
        const int x0 = wrapi(t.cx, t.w), x1 = wrapi(t.cx + 1, t.w), y0 = wrapi(t.cy, t.h), y1 = wrapi(t.cy + 1, t.h);
        const float w00 = (1.0f - t.fx) * (1.0f - t.fy), w10 = t.fx * (1.0f - t.fy), w01 = (1.0f - t.fx) * t.fy, w11 = t.fx * t.fy;
        float* a = g1 + t.off + (size_t(y0) * t.w + x0) * g.c1;
        float* b = g1 + t.off + (size_t(y0) * t.w + x1) * g.c1;
        float* c2 = g1 + t.off + (size_t(y1) * t.w + x0) * g.c1;
        float* e = g1 + t.off + (size_t(y1) * t.w + x1) * g.c1;
        const float* dd = d + 4 * g.c0;
        for (int c = 0; c < g.c1; ++c) {
            atomicAdd(&a[c], w00 * dd[c]);
            atomicAdd(&b[c], w10 * dd[c]);
            atomicAdd(&c2[c], w01 * dd[c]);
            atomicAdd(&e[c], w11 * dd[c]);
        }
    }
}

__global__ void biasActK(float* Z, float* A, const float* b, int out, size_t total, int act) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= uint32_t(total)) return;
    const float z = Z[i] + b[i % uint32_t(out)];
    Z[i] = z;
    if (A) A[i] = activate(z, act);
}

__global__ void dActK(float* dA, const float* Z, size_t total, int act) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= uint32_t(total)) return;
    dA[i] *= dActivate(Z[i], act);
}

__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// mode 0: training loss and gradient (raw network output for `none` channels, sigmoid for the others);
// mode 1: evaluation (clamped / sigmoid prediction), squared errors accumulated into sums[0..nNet) (inside the mask) and
// sums[nNet..2 nNet) (everywhere).
__global__ void lossK(const int* smp, const float* Y, const float* tgt, float* dY, float* lossAcc, double* sums, const uint8_t* mask0, int n, int mode) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = s < n;
    float loss = 0.0f;
    float err[32];
    bool inside = false;
    const int N = g.nNet;
    if (valid) {
        const int m = smp[3 * s], i = smp[3 * s + 1], j = smp[3 * s + 2];
        const int Wm = max(1, g.W >> m);
        const float* t = tgt + g.mipOff[m] + (size_t(j) * Wm + i) * N;
        inside = mode == 1 && mask0[size_t(j) * g.W + i] != 0;
        for (int c = 0; c < N; ++c) {
            const float y = Y[size_t(s) * N + c];
            const bool sg = g.sig[c] != 0.0f;
            float p = sg ? 1.0f / (1.0f + __expf(-y)) : y;
            if (mode == 1 && !sg) p = fminf(fmaxf(p, 0.0f), 1.0f);
            const float d = p - t[c];
            err[c] = d * d;
            if (mode == 0) {
                loss += g.lw[c] * d * d;
                dY[size_t(s) * N + c] = 2.0f * g.lw[c] * d * g.invNorm / float(n) * (sg ? p * (1.0f - p) : 1.0f);
            }
        }
    } else {
        for (int c = 0; c < N; ++c) err[c] = 0.0f;
    }
    if (mode == 0) {
        loss = warpSum(loss);
        if ((threadIdx.x & 31) == 0) atomicAdd(lossAcc, loss * g.invNorm / float(n));
    } else {
        for (int c = 0; c < N; ++c) {
            const float a = warpSum(inside ? err[c] : 0.0f);
            const float b = warpSum(valid ? err[c] : 0.0f);
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&sums[c], double(a));
                atomicAdd(&sums[N + c], double(b));
            }
        }
    }
}

__global__ void outActK(const float* Y, float* out, int n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * g.nNet) return;
    const float y = Y[i];
    out[i] = g.sig[i % g.nNet] != 0.0f ? 1.0f / (1.0f + expf(-y)) : fminf(fmaxf(y, 0.0f), 1.0f);
}

__global__ void adamK(float* p, float* gr, float* mo, float* ve, size_t n, float lr, float b1, float b2, float c1, float c2, float eps, float lo, float hi, int zeroGrad) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = gr[i];
    const float mm = b1 * mo[i] + (1.0f - b1) * gv;
    const float vv = b2 * ve[i] + (1.0f - b2) * gv * gv;
    mo[i] = mm;
    ve[i] = vv;
    p[i] = fminf(fmaxf(p[i] - lr * (mm * c1) / (sqrtf(vv * c2) + eps), lo), hi);
    if (zeroGrad) gr[i] = 0.0f;
}

int blocks(size_t n, int t = 256) { return int((n + t - 1) / t); }

// ---- device state -------------------------------------------------------------------------------------------------------

class Dev {
public:
    Dev(const Shape& shape, int act, int cap, int device) : s_(shape), act_(act), cap_(cap) {
        CK(cudaSetDevice(device));
        CB(cublasCreate(&h_));
        stride_[0] = s_.stride0;
        stride_[1] = stride_[2] = kH;
        out_[0] = out_[1] = kH;
        out_[2] = s_.nNet;
        mlpN_ = 0;
        for (int l = 0; l < 3; ++l) {
            pW_[l] = mlpN_;
            mlpN_ += size_t(stride_[l]) * out_[l];
            pB_[l] = mlpN_;
            mlpN_ += out_[l];
        }
        for (int gi = 0; gi < 2; ++gi) {
            cells_[gi] = gridCells(s_, gi);
            lat_[gi] = alloc<float>(cells_[gi]);
            latN_[gi] = alloc<float>(cells_[gi]);
            qb_[gi] = alloc<uint8_t>(cells_[gi]);
        }
        mlp_ = alloc<float>(mlpN_);
        X_ = alloc<float>(size_t(cap_) * s_.stride0);
        CK(cudaMemset(X_, 0, size_t(cap_) * s_.stride0 * 4));
        Z_[0] = alloc<float>(size_t(cap_) * kH);
        Z_[1] = alloc<float>(size_t(cap_) * kH);
        A_[0] = alloc<float>(size_t(cap_) * kH);
        A_[1] = alloc<float>(size_t(cap_) * kH);
        Y_ = alloc<float>(size_t(cap_) * s_.nNet);
        smp_ = alloc<int>(size_t(cap_) * 3);
        ones_ = alloc<float>(cap_);
        std::vector<float> ones(cap_, 1.0f);
        CK(cudaMemcpy(ones_, ones.data(), cap_ * 4, cudaMemcpyHostToDevice));
        buildGeom();
    }
    ~Dev() {
        for (void* p : ptrs_) cudaFree(p);
        if (h_) cublasDestroy(h_);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;

    template <class T>
    T* alloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        ptrs_.push_back(p);
        return static_cast<T*>(p);
    }

    void buildGeom() {
        std::memset(&geom_, 0, sizeof(geom_));
        geom_.W = s_.W, geom_.H = s_.H, geom_.M = s_.M, geom_.L = s_.L, geom_.nNet = s_.nNet, geom_.stride0 = s_.stride0, geom_.nIn = s_.nIn;
        geom_.c0 = s_.gc[0], geom_.c1 = s_.gc[1];
        for (int gi = 0; gi < 2; ++gi) {
            int off = 0;
            for (int lv = 0; lv < s_.L; ++lv) {
                geom_.gw[gi][lv] = s_.gw[gi] >> (2 * lv);
                geom_.gh[gi][lv] = s_.gh[gi] >> (2 * lv);
                geom_.goff[gi][lv] = off;
                off += int(levelCells(s_, gi, lv));
            }
        }
        for (int c = 0; c < s_.nNet; ++c) geom_.sig[c] = s_.sigmoid[c] ? 1.0f : 0.0f, geom_.lw[c] = 1.0f;
        geom_.invNorm = 1.0f / float(std::max(1, s_.nNet));
        upload();
    }
    void upload() { CK(cudaMemcpyToSymbol(g, &geom_, sizeof(Geom))); }

    // Training data: per-mip targets and mask lists.
    void setTargets(const std::vector<float>& mips, const std::vector<int>& mipOff, const std::vector<uint8_t>& mask0, const std::vector<int>& maskIdx, const std::vector<int>& maskOff,
                    const std::vector<int>& maskCnt, const float* lossWeights) {
        tgt_ = alloc<float>(mips.size());
        CK(cudaMemcpy(tgt_, mips.data(), mips.size() * 4, cudaMemcpyHostToDevice));
        mask0_ = alloc<uint8_t>(mask0.size());
        CK(cudaMemcpy(mask0_, mask0.data(), mask0.size(), cudaMemcpyHostToDevice));
        maskIdx_ = alloc<int>(maskIdx.size());
        if (!maskIdx.empty()) CK(cudaMemcpy(maskIdx_, maskIdx.data(), maskIdx.size() * 4, cudaMemcpyHostToDevice));
        for (int m = 0; m < s_.M; ++m) geom_.mipOff[m] = mipOff[m], geom_.maskOff[m] = maskOff[m], geom_.maskCnt[m] = maskCnt[m];
        float sum = 0.0f;
        for (int c = 0; c < s_.nNet; ++c) geom_.lw[c] = lossWeights[c], sum += lossWeights[c];
        geom_.invNorm = sum > 0.0f ? 1.0f / sum : 1.0f;
        upload();
        sums_ = alloc<double>(2 * s_.nNet);
        lossAcc_ = alloc<float>(1);
        dY_ = alloc<float>(size_t(cap_) * s_.nNet);
        dA_[0] = alloc<float>(size_t(cap_) * kH);
        dA_[1] = alloc<float>(size_t(cap_) * kH);
        dX_ = alloc<float>(size_t(cap_) * s_.stride0);
        mlpG_ = alloc<float>(mlpN_);
        mlpM_ = alloc<float>(mlpN_);
        mlpV_ = alloc<float>(mlpN_);
        for (int gi = 0; gi < 2; ++gi) {
            latG_[gi] = alloc<float>(cells_[gi]);
            latM_[gi] = alloc<float>(cells_[gi]);
            latV_[gi] = alloc<float>(cells_[gi]);
            CK(cudaMemset(latG_[gi], 0, cells_[gi] * 4));
            CK(cudaMemset(latM_[gi], 0, cells_[gi] * 4));
            CK(cudaMemset(latV_[gi], 0, cells_[gi] * 4));
        }
        CK(cudaMemset(mlpG_, 0, mlpN_ * 4));
        resetMlpMoments();
        maskCount0_ = 0;
        for (uint8_t v : mask0) maskCount0_ += v != 0;
        if (maskCount0_ == 0) maskCount0_ = mask0.size();
    }

    void resetMlpMoments() {
        CK(cudaMemset(mlpM_, 0, mlpN_ * 4));
        CK(cudaMemset(mlpV_, 0, mlpN_ * 4));
    }

    void uploadMlp(const std::vector<float>& flat) { CK(cudaMemcpy(mlp_, flat.data(), mlpN_ * 4, cudaMemcpyHostToDevice)); }
    std::vector<float> downloadMlp() {
        std::vector<float> flat(mlpN_);
        CK(cudaMemcpy(flat.data(), mlp_, mlpN_ * 4, cudaMemcpyDeviceToHost));
        return flat;
    }
    size_t mlpSize() const { return mlpN_; }
    size_t pW(int l) const { return pW_[l]; }
    size_t pB(int l) const { return pB_[l]; }
    int strideOf(int l) const { return stride_[l]; }
    int outOf(int l) const { return out_[l]; }

    void initLatents(float x, uint32_t seed) {
        for (int gi = 0; gi < 2; ++gi) {
            std::vector<float> v(cells_[gi], 0.0f);
            if (x > 0.0f) {
                std::mt19937 rng(seed + 77u * gi);
                for (float& f : v) f = (float(rng() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * x;
            }
            CK(cudaMemcpy(lat_[gi], v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        }
    }
    void setLatentsFromQ(const std::vector<uint8_t> q[2]) {
        for (int gi = 0; gi < 2; ++gi) {
            CK(cudaMemcpy(qb_[gi], q[gi].data(), cells_[gi], cudaMemcpyHostToDevice));
            dequantK<<<blocks(cells_[gi]), 256>>>(qb_[gi], latN_[gi], cells_[gi], s_.gbits[gi]);
        }
        CK(cudaGetLastError());
    }

    void sample(int n, int m, uint32_t seed, uint32_t step, float pInside, bool seq, int start) {
        sampleK<<<blocks(n, 128), 128>>>(smp_, n, m, seed, step, maskIdx_, pInside, seq ? 1 : 0, start);
    }
    void setSamples(const std::vector<int>& smp) { CK(cudaMemcpy(smp_, smp.data(), smp.size() * 4, cudaMemcpyHostToDevice)); }

    void noise(uint32_t seed, uint32_t step, float scale) {
        for (int gi = 0; gi < 2; ++gi) noiseK<<<blocks(cells_[gi]), 256>>>(lat_[gi], latN_[gi], cells_[gi], scale / float(1 << s_.gbits[gi]), seed + 31u * gi, step);
    }
    void quantise() {
        for (int gi = 0; gi < 2; ++gi) quantK<<<blocks(cells_[gi]), 256>>>(lat_[gi], latN_[gi], qb_[gi], cells_[gi], s_.gbits[gi]);
        CK(cudaGetLastError());
    }
    std::vector<uint8_t> downloadQ(int gi) {
        std::vector<uint8_t> q(cells_[gi]);
        CK(cudaMemcpy(q.data(), qb_[gi], cells_[gi], cudaMemcpyDeviceToHost));
        return q;
    }

    void gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
        const float one = 1.0f, zero = 0.0f;
        CB(cublasSgemm(h_, ta, tb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc));
    }
    void gemv(int m, int n, const float* A, int lda, float* y) {
        const float one = 1.0f, zero = 0.0f;
        CB(cublasSgemv(h_, CUBLAS_OP_N, m, n, &one, A, lda, ones_, 1, &zero, y, 1));
    }

    // Forward pass over the samples in smp_ using the given latent buffers.
    void forward(int n, const float* l0, const float* l1) {
        gatherK<<<blocks(n, 128), 128>>>(smp_, l0, l1, X_, n);
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, kH, n, stride_[0], mlp_ + pW_[0], stride_[0], X_, stride_[0], Z_[0], kH);
        biasActK<<<blocks(size_t(n) * kH), 256>>>(Z_[0], A_[0], mlp_ + pB_[0], kH, size_t(n) * kH, act_);
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, kH, n, kH, mlp_ + pW_[1], kH, A_[0], kH, Z_[1], kH);
        biasActK<<<blocks(size_t(n) * kH), 256>>>(Z_[1], A_[1], mlp_ + pB_[1], kH, size_t(n) * kH, act_);
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, s_.nNet, n, kH, mlp_ + pW_[2], kH, A_[1], kH, Y_, s_.nNet);
        biasActK<<<blocks(size_t(n) * s_.nNet), 256>>>(Y_, nullptr, mlp_ + pB_[2], s_.nNet, size_t(n) * s_.nNet, 3);
    }

    // One optimisation step on the already sampled batch. phaseA: latents are trained as well.
    void trainStep(int n, bool phaseA, int step, float lrMlp, float lrLat, uint32_t seed) {
        const float* l0 = phaseA ? latN_[0] : latN_[0];
        const float* l1 = latN_[1];
        forward(n, l0, l1);
        lossK<<<blocks(n, 128), 128>>>(smp_, Y_, tgt_, dY_, lossAcc_, sums_, mask0_, n, 0);
        const int N = s_.nNet;
        // layer 2
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, kH, N, n, A_[1], kH, dY_, N, mlpG_ + pW_[2], kH);
        gemv(N, n, dY_, N, mlpG_ + pB_[2]);
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, kH, n, N, mlp_ + pW_[2], kH, dY_, N, dA_[1], kH);
        dActK<<<blocks(size_t(n) * kH), 256>>>(dA_[1], Z_[1], size_t(n) * kH, act_);
        // layer 1
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, kH, kH, n, A_[0], kH, dA_[1], kH, mlpG_ + pW_[1], kH);
        gemv(kH, n, dA_[1], kH, mlpG_ + pB_[1]);
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, kH, n, kH, mlp_ + pW_[1], kH, dA_[1], kH, dA_[0], kH);
        dActK<<<blocks(size_t(n) * kH), 256>>>(dA_[0], Z_[0], size_t(n) * kH, act_);
        // layer 0
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, stride_[0], kH, n, X_, stride_[0], dA_[0], kH, mlpG_ + pW_[0], stride_[0]);
        gemv(kH, n, dA_[0], kH, mlpG_ + pB_[0]);
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        if (phaseA) {
            gemm(CUBLAS_OP_N, CUBLAS_OP_N, stride_[0], n, kH, mlp_ + pW_[0], stride_[0], dA_[0], kH, dX_, stride_[0]);
            scatterK<<<blocks(n, 128), 128>>>(smp_, dX_, latG_[0], latG_[1], n);
            const float c1 = 1.0f / (1.0f - std::pow(b1, float(step + 1))), c2 = 1.0f / (1.0f - std::pow(b2, float(step + 1)));
            for (int gi = 0; gi < 2; ++gi) {
                const int bits = s_.gbits[gi], zq = (1 << (bits - 1)) - 1;
                const float lo = -float(zq) / float(1 << bits), hi = float((1 << bits) - 1 - zq) / float(1 << bits);
                adamK<<<blocks(cells_[gi]), 256>>>(lat_[gi], latG_[gi], latM_[gi], latV_[gi], cells_[gi], lrLat, b1, b2, c1, c2, eps, lo, hi, 1);
            }
        }
        const float c1 = 1.0f / (1.0f - std::pow(b1, float(mlpStep_ + 1))), c2 = 1.0f / (1.0f - std::pow(b2, float(mlpStep_ + 1)));
        ++mlpStep_;
        adamK<<<blocks(mlpN_), 256>>>(mlp_, mlpG_, mlpM_, mlpV_, mlpN_, lrMlp, b1, b2, c1, c2, eps, -1e30f, 1e30f, 0);
        (void)seed;
    }
    void resetMlpStep() { mlpStep_ = 0; }

    float takeLoss(int stepsSince) {
        float l = 0.0f;
        CK(cudaMemcpy(&l, lossAcc_, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemset(lossAcc_, 0, 4));
        return l / float(std::max(1, stepsSince));
    }
    void clearLoss() { CK(cudaMemset(lossAcc_, 0, 4)); }

    // Masked mip-0 MSE per network output of the full texture.
    std::vector<double> evalMip0(const float* l0, const float* l1) {
        const int N = s_.nNet;
        CK(cudaMemset(sums_, 0, 2 * N * sizeof(double)));
        const size_t total = size_t(s_.W) * s_.H;
        for (size_t start = 0; start < total; start += cap_) {
            const int n = int(std::min<size_t>(cap_, total - start));
            sample(n, 0, 0, 0, 0.0f, true, int(start));
            forward(n, l0, l1);
            lossK<<<blocks(n, 128), 128>>>(smp_, Y_, tgt_, dY_, lossAcc_, sums_, mask0_, n, 1);
        }
        std::vector<double> sums(2 * N);
        CK(cudaMemcpy(sums.data(), sums_, sums.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<double> mse(N);
        for (int c = 0; c < N; ++c) mse[c] = sums[c] / double(maskCount0_);
        return mse;
    }

    void decode(int n, std::vector<float>& out) {
        forward(n, latN_[0], latN_[1]);
        float* o = alloc<float>(size_t(n) * s_.nNet);
        outActK<<<blocks(size_t(n) * s_.nNet), 256>>>(Y_, o, n);
        out.resize(size_t(n) * s_.nNet);
        CK(cudaMemcpy(out.data(), o, out.size() * 4, cudaMemcpyDeviceToHost));
    }

    void sync() { CK(cudaDeviceSynchronize()); }
    void check() { CK(cudaGetLastError()); }

    float* lat(int gi) { return lat_[gi]; }
    float* latN(int gi) { return latN_[gi]; }

private:
    Shape s_;
    int act_, cap_;
    cublasHandle_t h_ = nullptr;
    std::vector<void*> ptrs_;
    Geom geom_{};
    int stride_[3]{}, out_[3]{};
    size_t pW_[3]{}, pB_[3]{}, mlpN_ = 0, cells_[2]{};
    float *lat_[2]{}, *latN_[2]{}, *latG_[2]{}, *latM_[2]{}, *latV_[2]{};
    uint8_t* qb_[2]{};
    float *mlp_ = nullptr, *mlpG_ = nullptr, *mlpM_ = nullptr, *mlpV_ = nullptr;
    float *X_ = nullptr, *Z_[2]{}, *A_[2]{}, *Y_ = nullptr, *dY_ = nullptr, *dA_[2]{}, *dX_ = nullptr, *ones_ = nullptr, *tgt_ = nullptr, *lossAcc_ = nullptr;
    int *smp_ = nullptr, *maskIdx_ = nullptr;
    uint8_t* mask0_ = nullptr;
    double* sums_ = nullptr;
    size_t maskCount0_ = 0;
    int mlpStep_ = 0;
};

void boxDown(const std::vector<float>& src, int w, int h, int c, std::vector<float>& dst, int& w2, int& h2) {
    w2 = std::max(1, w >> 1);
    h2 = std::max(1, h >> 1);
    dst.assign(size_t(w2) * h2 * c, 0.0f);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) {
            const int x1 = std::min(2 * x + 1, w - 1), y1 = std::min(2 * y + 1, h - 1);
            for (int k = 0; k < c; ++k)
                dst[(size_t(y) * w2 + x) * c + k] = 0.25f * (src[(size_t(2 * y) * w + 2 * x) * c + k] + src[(size_t(2 * y) * w + x1) * c + k] + src[(size_t(y1) * w + 2 * x) * c + k] + src[(size_t(y1) * w + x1) * c + k]);
        }
}

void maskDown(const std::vector<uint8_t>& src, int w, int h, std::vector<uint8_t>& dst) {
    const int w2 = std::max(1, w >> 1), h2 = std::max(1, h >> 1);
    dst.assign(size_t(w2) * h2, 0);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) {
            const int x1 = std::min(2 * x + 1, w - 1), y1 = std::min(2 * y + 1, h - 1);
            dst[size_t(y) * w2 + x] = (src[size_t(2 * y) * w + 2 * x] | src[size_t(2 * y) * w + x1] | src[size_t(y1) * w + 2 * x] | src[size_t(y1) * w + x1]) ? 1 : 0;
        }
}

float cosine(float lr, int step, int total) { return lr * 0.5f * (1.0f + std::cos(3.14159265358979f * float(step) / float(std::max(1, total)))); }

} // namespace

std::string deviceName(int device) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) return "unknown";
    return p.name;
}

bool train(const Shape& s, const std::vector<float>& target, const float* lossWeights, const std::vector<uint8_t>& maskIn, const Params& p, Result& out, const Progress& progress,
           std::string& err) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        const int N = s.nNet;
        const size_t texels = size_t(s.W) * s.H;
        if (target.size() != texels * N) throw CudaError("target size mismatch");
        const int cap = p.batch;
        Dev dev(s, p.activation, cap, p.device);

        // Mip chain of the targets and masks.
        std::vector<float> mips;
        std::vector<int> mipOff(s.M), maskOff(s.M), maskCnt(s.M);
        std::vector<int> maskIdx;
        std::vector<uint8_t> mask0(texels, 1);
        const bool haveMask = maskIn.size() == texels;
        if (haveMask) mask0 = maskIn;
        {
            std::vector<float> cur = target, nxt;
            std::vector<uint8_t> maskCur = mask0, maskNxt;
            int w = s.W, h = s.H;
            for (int m = 0; m < s.M; ++m) {
                mipOff[m] = int(mips.size());
                mips.insert(mips.end(), cur.begin(), cur.end());
                maskOff[m] = int(maskIdx.size());
                int cnt = 0;
                if (haveMask) {
                    for (size_t t = 0; t < size_t(w) * h; ++t)
                        if (maskCur[t]) maskIdx.push_back(int(t)), ++cnt;
                    if (size_t(cnt) == size_t(w) * h) maskIdx.resize(maskOff[m]), cnt = 0; // everything inside: sample uniformly
                }
                maskCnt[m] = cnt;
                if (m + 1 < s.M) {
                    int w2, h2;
                    boxDown(cur, w, h, N, nxt, w2, h2);
                    maskDown(maskCur, w, h, maskNxt);
                    cur.swap(nxt);
                    maskCur.swap(maskNxt);
                    w = w2, h = h2;
                }
            }
        }
        dev.setTargets(mips, mipOff, mask0, maskIdx, maskOff, maskCnt, lossWeights);

        // Initialisation: MLP U(+-1/sqrt(fan_in)), last bias at the target mean, latents at 0.
        std::mt19937 rng(p.seed);
        auto uni = [&]() { return float(rng() >> 8) * (1.0f / 16777216.0f); };
        std::vector<float> flat(dev.mlpSize(), 0.0f);
        const int fanIn[3] = {s.nIn, kH, kH};
        for (int l = 0; l < 3; ++l) {
            const float a = 1.0f / std::sqrt(float(fanIn[l]));
            for (int o = 0; o < dev.outOf(l); ++o)
                for (int k = 0; k < fanIn[l]; ++k) flat[dev.pW(l) + size_t(o) * dev.strideOf(l) + k] = (2.0f * uni() - 1.0f) * a;
        }
        {
            std::vector<double> mean(N, 0.0);
            size_t cnt = 0;
            for (size_t t = 0; t < texels; ++t) {
                if (!mask0[t]) continue;
                for (int c = 0; c < N; ++c) mean[c] += target[t * N + c];
                ++cnt;
            }
            for (int c = 0; c < N; ++c) {
                const double m = cnt ? mean[c] / double(cnt) : 0.5;
                flat[dev.pB(2) + c] = s.sigmoid[c] ? float(std::log(std::min(std::max(m, 0.02), 0.98) / (1.0 - std::min(std::max(m, 0.02), 0.98)))) : float(m);
            }
        }
        dev.uploadMlp(flat);
        dev.initLatents(p.initLatent, p.seed);

        // Phase A.
        int sinceReport = 0;
        std::mt19937 mrng(p.seed * 2654435761u + 17u);
        auto pickMip = [&]() {
            const float u = float(mrng() >> 8) * (1.0f / 16777216.0f);
            if (u < p.uniformMipP) return int(mrng() % uint32_t(s.M));
            const float u2 = float(mrng() >> 8) * (1.0f / 16777216.0f);
            const double x = (double(mrng() >> 8) + 1.0) / 16777217.0;
            if (u2 < p.mip0P || s.M == 1) return 0;
            return std::min(s.M - 1, 1 + int(std::floor(-std::log(x) / std::log(4.0))));
        };
        dev.clearLoss();
        for (int step = 0; step < p.stepsA; ++step) {
            const int m = pickMip();
            dev.sample(cap, m, p.seed, uint32_t(step), p.insideP, false, 0);
            dev.noise(p.seed, uint32_t(step), p.noiseScale);
            dev.trainStep(cap, true, step, cosine(p.lrMlp, step, p.stepsA), cosine(p.lrLatent, step, p.stepsA), p.seed);
            ++sinceReport;
            if ((step + 1) % 250 == 0 || step + 1 == p.stepsA) {
                const float l = dev.takeLoss(sinceReport);
                sinceReport = 0;
                if (progress) progress(0, step + 1, p.stepsA, l);
                out.finalLoss = l;
            }
        }
        dev.sync();
        dev.check();
        out.mseFloat = dev.evalMip0(dev.lat(0), dev.lat(1));
        dev.quantise();
        out.mseQuantPre = dev.evalMip0(dev.latN(0), dev.latN(1));

        // Phase B: frozen quantised latents, network only.
        dev.resetMlpMoments();
        dev.resetMlpStep();
        dev.clearLoss();
        sinceReport = 0;
        for (int step = 0; step < p.stepsB; ++step) {
            const int m = pickMip();
            dev.sample(cap, m, p.seed + 999u, uint32_t(step), p.insideP, false, 0);
            dev.trainStep(cap, false, step, cosine(p.lrB, step, p.stepsB), 0.0f, p.seed);
            ++sinceReport;
            if ((step + 1) % 250 == 0 || step + 1 == p.stepsB) {
                const float l = dev.takeLoss(sinceReport);
                sinceReport = 0;
                if (progress) progress(1, step + 1, p.stepsB, l);
                out.finalLoss = l;
            }
        }
        dev.sync();
        dev.check();
        out.mseFinal = dev.evalMip0(dev.latN(0), dev.latN(1));

        // Export.
        for (int gi = 0; gi < 2; ++gi) out.q[gi] = dev.downloadQ(gi);
        const std::vector<float> w = dev.downloadMlp();
        for (int l = 0; l < 3; ++l) {
            const size_t nw = size_t(dev.strideOf(l)) * dev.outOf(l);
            out.net.w[l].assign(w.begin() + dev.pW(l), w.begin() + dev.pW(l) + nw);
            out.net.b[l].assign(w.begin() + dev.pB(l), w.begin() + dev.pB(l) + dev.outOf(l));
        }
        out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    } catch (const std::exception& e) {
        err = std::string("CUDA trainer: ") + e.what();
        return false;
    }
}

bool decodeSamples(const Shape& s, const std::vector<uint8_t> q[2], const Weights& net, int activation, int device, const std::vector<int>& samples, std::vector<float>& out,
                   std::string& err) {
    try {
        const int n = int(samples.size() / 3);
        Dev dev(s, activation, std::max(n, 1), device);
        std::vector<float> flat(dev.mlpSize(), 0.0f);
        for (int l = 0; l < 3; ++l) {
            std::copy(net.w[l].begin(), net.w[l].end(), flat.begin() + dev.pW(l));
            std::copy(net.b[l].begin(), net.b[l].end(), flat.begin() + dev.pB(l));
        }
        dev.uploadMlp(flat);
        dev.setLatentsFromQ(q);
        dev.setSamples(samples);
        dev.decode(n, out);
        return true;
    } catch (const std::exception& e) {
        err = std::string("CUDA decode: ") + e.what();
        return false;
    }
}

} // namespace gtc::v4cuda
