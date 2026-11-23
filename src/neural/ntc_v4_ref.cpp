#include <cmath>
#include <cstring>
#include <thread>

#include "ntc_v4.h"

namespace gtc::v4 {

float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
    if (e == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            e = 127 - 15 + 1;
            while (!(mant & 0x400u)) mant <<= 1, --e;
            mant &= 0x3FFu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) return uint16_t(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0u));
    if (absx >= 0x477FF000u) return uint16_t(sign | 0x7C00u); // rounds to infinity (>= 65520)
    if (absx < 0x33000001u) return uint16_t(sign);            // <= 2^-25 rounds to zero
    int e = int(absx >> 23) - 127;
    if (e < -14) { // subnormal
        const uint32_t mant = (absx & 0x7FFFFFu) | 0x800000u;
        const int shift = -e - 14 + 13;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = ((uint32_t(e + 15)) << 10) | ((absx & 0x7FFFFFu) >> 13);
    const uint32_t rem = absx & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return uint16_t(sign | h);
}

bool Model::finalize(std::string& err) {
    if (W <= 0 || H <= 0 || W % 8 || H % 8) {
        err = "v4 needs a resolution that is a multiple of 8";
        return false;
    }
    for (const GridSpec& gs : g)
        if ((gs.bits != 2 && gs.bits != 4) || gs.c <= 0 || (gs.c * gs.bits) % 8) {
            err = "v4 grids need 2 or 4 bit channels filling whole bytes";
            return false;
        }
    if (g[1].w * 2 != g[0].w || g[1].h * 2 != g[0].h || (W != g[0].w * 2 && W != g[0].w * 4) || W * g[0].h != H * g[0].w) {
        err = "v4 grid sizes: G1 must be G0/2 and the texture 2x or 4x G0";
        return false;
    }
    g[0].concat = true;
    g[1].concat = false;
    M = mipCountOf(W, H);
    L = std::min(levelCountOf(g[1].w, g[1].h), std::max(1, levelCap));
    nNet = 0;
    for (const ChannelSpec& c : channels)
        if (!c.isConst) {
            if (c.net != nNet) {
                err = "v4 channels: net indices must ascend from 0";
                return false;
            }
            ++nNet;
        }
    if (nNet < 1) {
        err = "v4 needs at least one non-constant channel";
        return false;
    }
    const int in = inputs();
    layers[0] = {in, ceilDiv(in, 4) * 4, kHidden};
    layers[1] = {kHidden, kHidden, kHidden};
    layers[2] = {kHidden, kHidden, nNet};
    for (int g2 = 0; g2 < 2; ++g2) {
        int aw, ah;
        atlasSize(g[g2], L, aw, ah);
        if (q[g2].size() != size_t(aw) * ah * g[g2].c) q[g2].assign(size_t(aw) * ah * g[g2].c, 0);
    }
    return true;
}

size_t weightBytes(const Model& m) {
    size_t n = 0;
    for (const LayerSpec& l : m.layers) n += size_t(l.out) * (l.inStride + 1);
    return n * 2;
}

std::vector<uint8_t> serializeWeights(const Model& m) {
    std::vector<uint8_t> out;
    out.reserve(weightBytes(m));
    auto put = [&](float v) {
        const uint16_t h = floatToHalf(v);
        out.push_back(uint8_t(h & 0xFF));
        out.push_back(uint8_t(h >> 8));
    };
    for (int l = 0; l < 3; ++l) {
        const LayerSpec& ls = m.layers[l];
        for (int o = 0; o < ls.out; ++o)
            for (int k = 0; k < ls.inStride; ++k) put(k < ls.in ? m.w[l][size_t(o) * ls.inStride + k] : 0.0f);
        for (int o = 0; o < ls.out; ++o) put(m.b[l][o]);
    }
    return out;
}

bool parseWeights(const uint8_t* data, size_t size, Model& m, std::string& err) {
    if (size != weightBytes(m)) {
        err = "v4 weights: unexpected byte length";
        return false;
    }
    size_t p = 0;
    auto get = [&]() {
        const uint16_t h = uint16_t(data[p] | (data[p + 1] << 8));
        p += 2;
        return halfToFloat(h);
    };
    for (int l = 0; l < 3; ++l) {
        const LayerSpec& ls = m.layers[l];
        m.w[l].resize(size_t(ls.out) * ls.inStride);
        m.b[l].resize(ls.out);
        for (float& v : m.w[l]) v = get();
        for (float& v : m.b[l]) v = get();
    }
    return true;
}

std::vector<Image> packAtlas(const Model& m, int gi) {
    const GridSpec& g = m.g[gi];
    int aw, ah;
    atlasSize(g, m.L, aw, ah);
    const int nImg = imagesPerGrid(g);
    std::vector<Image> imgs(nImg);
    for (Image& im : imgs) {
        im.width = aw;
        im.height = ah;
        im.rgba.assign(size_t(aw) * ah * 4, 0);
    }
    for (int lv = 0; lv < m.L; ++lv) {
        int w, h, ox, oy;
        levelSize(g, lv, w, h);
        atlasOrigin(g, lv, ox, oy);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t px = size_t(oy + y) * aw + ox + x;
                const uint8_t* cell = &m.q[gi][px * g.c];
                for (int c = 0; c < g.c; ++c) {
                    const int bit = c * g.bits, byte = bit / 8, shift = 8 - g.bits - bit % 8;
                    imgs[byte / 4].rgba[px * 4 + byte % 4] |= uint8_t((cell[c] & ((1 << g.bits) - 1)) << shift);
                }
            }
    }
    return imgs;
}

bool unpackAtlas(Model& m, int gi, const std::vector<Image>& imgs, std::string& err) {
    const GridSpec& g = m.g[gi];
    int aw, ah;
    atlasSize(g, m.L, aw, ah);
    if (int(imgs.size()) != imagesPerGrid(g)) {
        err = "v4: wrong number of latent images";
        return false;
    }
    for (const Image& im : imgs)
        if (im.width != aw || im.height != ah || im.rgba.size() != size_t(aw) * ah * 4) {
            err = "v4: latent atlas has the wrong size";
            return false;
        }
    m.q[gi].assign(size_t(aw) * ah * g.c, 0);
    for (size_t px = 0; px < size_t(aw) * ah; ++px)
        for (int c = 0; c < g.c; ++c) {
            const int bit = c * g.bits, byte = bit / 8, shift = 8 - g.bits - bit % 8;
            m.q[gi][px * g.c + c] = uint8_t((imgs[byte / 4].rgba[px * 4 + byte % 4] >> shift) & ((1 << g.bits) - 1));
        }
    return true;
}

namespace {

inline float fractf(float t) { return t - std::floor(t); }
inline float triC(float t) { return 4.0f * std::fabs(fractf(t) - 0.5f) - 1.0f; }
inline float triS(float t) { return triC(t - 0.25f); }
inline float hardGelu(float z) { return z < -1.5f ? 0.0f : (z > 1.5f ? z : z * (z + 1.5f) / 3.0f); }

} // namespace

RefDecoder::RefDecoder(const Model& m) : m_(m) {
    for (int l = 0; l < 3; ++l) {
        const LayerSpec& ls = m.layers[l];
        wt_[l].assign(size_t(ls.in) * ls.out, 0.0f);
        for (int o = 0; o < ls.out; ++o)
            for (int k = 0; k < ls.in; ++k) wt_[l][size_t(k) * ls.out + o] = m.w[l][size_t(o) * ls.inStride + k];
    }
}

void RefDecoder::inputsOf(int m, int i, int j, float* x) const {
    const Model& md = m_;
    const int lv = levelOf(m, md.L);
    const int Wm = std::max(1, md.W >> m), Hm = std::max(1, md.H >> m);
    int n = 0;
    auto latent = [&](int g, int cx, int cy, int c) {
        const GridSpec& gs = md.g[g];
        int w, h, ox, oy, aw, ah;
        levelSize(gs, lv, w, h);
        atlasOrigin(gs, lv, ox, oy);
        atlasSize(gs, md.L, aw, ah);
        cx = ((cx % w) + w) % w;
        cy = ((cy % h) + h) % h;
        const int q = md.q[g][(size_t(oy + cy) * aw + ox + cx) * gs.c + c];
        return latentValue(q, gs.bits);
    };
    {
        int w0, h0;
        levelSize(md.g[0], lv, w0, h0);
        const float rx = float(w0) / float(Wm), ry = float(h0) / float(Hm);
        const float gx = (float(i) + 0.5f) * rx - 0.5f, gy = (float(j) + 0.5f) * ry - 0.5f;
        const int cx = int(std::floor(gx)), cy = int(std::floor(gy));
        for (int t = 0; t < 4; ++t)
            for (int c = 0; c < md.g[0].c; ++c) x[n++] = latent(0, cx + (t & 1), cy + (t >> 1), c);
    }
    {
        int w1, h1;
        levelSize(md.g[1], lv, w1, h1);
        const float rx = float(w1) / float(Wm), ry = float(h1) / float(Hm);
        const float gx = (float(i) + 0.5f) * rx - 0.5f, gy = (float(j) + 0.5f) * ry - 0.5f;
        const int cx = int(std::floor(gx)), cy = int(std::floor(gy));
        const float fx = gx - float(cx), fy = gy - float(cy);
        for (int c = 0; c < md.g[1].c; ++c) {
            const float a = latent(1, cx, cy, c), b = latent(1, cx + 1, cy, c), d = latent(1, cx, cy + 1, c), e = latent(1, cx + 1, cy + 1, c);
            x[n++] = (1.0f - fy) * ((1.0f - fx) * a + fx * b) + fy * ((1.0f - fx) * d + fx * e);
        }
    }
    for (int axis = 0; axis < 2; ++axis) {
        const float p = float(axis == 0 ? i : j);
        for (float P : {8.0f, 4.0f, 2.0f}) {
            x[n++] = triC(p / P);
            x[n++] = triS(p / P);
        }
    }
    x[n++] = float(m) / float(std::max(1, md.M - 1));
}

void RefDecoder::net(const float* x, float* y) const {
    float a[kHidden], b[kHidden];
    const LayerSpec* ls = m_.layers;
    auto dense = [&](int l, const float* v, float* o) {
        const int out = ls[l].out;
        for (int k = 0; k < out; ++k) o[k] = m_.b[l][k];
        for (int k = 0; k < ls[l].in; ++k) {
            const float vk = v[k];
            const float* w = &wt_[l][size_t(k) * out];
            for (int p = 0; p < out; ++p) o[p] += w[p] * vk;
        }
    };
    dense(0, x, a);
    for (int k = 0; k < kHidden; ++k) a[k] = hardGelu(a[k]);
    dense(1, a, b);
    for (int k = 0; k < kHidden; ++k) b[k] = hardGelu(b[k]);
    dense(2, b, y);
}

void RefDecoder::decodeTexel(int m, int i, int j, float* out) const {
    float x[128], y[32];
    inputsOf(m, i, j, x);
    net(x, y);
    for (size_t n = 0; n < m_.channels.size(); ++n) {
        const ChannelSpec& ch = m_.channels[n];
        if (ch.isConst) out[n] = ch.value;
        else if (ch.sigmoid) out[n] = 1.0f / (1.0f + std::exp(-y[ch.net]));
        else out[n] = std::min(1.0f, std::max(0.0f, y[ch.net]));
    }
}

void RefDecoder::decodeMip(int m, int threads, std::vector<float>& out) const {
    const int Wm = std::max(1, m_.W >> m), Hm = std::max(1, m_.H >> m), N = int(m_.channels.size());
    out.assign(size_t(Wm) * Hm * N, 0.0f);
    threads = std::max(1, threads);
    auto work = [&](int t) {
        for (int j = t; j < Hm; j += threads)
            for (int i = 0; i < Wm; ++i) decodeTexel(m, i, j, &out[(size_t(j) * Wm + i) * N]);
    };
    if (threads == 1) return work(0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    for (auto& th : pool) th.join();
}

size_t zeroDeadCells(Model& m, const std::vector<uint8_t>& mask0) {
    std::vector<uint8_t> used[2][16];
    for (int g = 0; g < 2; ++g)
        for (int lv = 0; lv < m.L; ++lv) {
            int w, h;
            levelSize(m.g[g], lv, w, h);
            used[g][lv].assign(size_t(w) * h, 0);
        }
    std::vector<uint8_t> cur = mask0, nxt;
    int w = m.W, h = m.H;
    for (int mip = 0; mip < m.M; ++mip) {
        const int lv = levelOf(mip, m.L);
        for (int g = 0; g < 2; ++g) {
            int gw, gh;
            levelSize(m.g[g], lv, gw, gh);
            const float rx = float(gw) / float(w), ry = float(gh) / float(h);
            for (int j = 0; j < h; ++j)
                for (int i = 0; i < w; ++i) {
                    if (!cur[size_t(j) * w + i]) continue;
                    const int cx = int(std::floor((float(i) + 0.5f) * rx - 0.5f)), cy = int(std::floor((float(j) + 0.5f) * ry - 0.5f));
                    for (int t = 0; t < 4; ++t) {
                        const int x = (((cx + (t & 1)) % gw) + gw) % gw, y = (((cy + (t >> 1)) % gh) + gh) % gh;
                        used[g][lv][size_t(y) * gw + x] = 1;
                    }
                }
        }
        if (mip + 1 < m.M) {
            const int w2 = std::max(1, w >> 1), h2 = std::max(1, h >> 1);
            nxt.assign(size_t(w2) * h2, 0);
            for (int y = 0; y < h2; ++y)
                for (int x = 0; x < w2; ++x) {
                    const int x1 = std::min(2 * x + 1, w - 1), y1 = std::min(2 * y + 1, h - 1);
                    nxt[size_t(y) * w2 + x] = (cur[size_t(2 * y) * w + 2 * x] | cur[size_t(2 * y) * w + x1] | cur[size_t(y1) * w + 2 * x] | cur[size_t(y1) * w + x1]) ? 1 : 0;
                }
            cur.swap(nxt);
            w = w2, h = h2;
        }
    }
    size_t zeroed = 0;
    for (int g = 0; g < 2; ++g) {
        const GridSpec& gs = m.g[g];
        int aw, ah;
        atlasSize(gs, m.L, aw, ah);
        for (int lv = 0; lv < m.L; ++lv) {
            int lw, lh, ox, oy;
            levelSize(gs, lv, lw, lh);
            atlasOrigin(gs, lv, ox, oy);
            for (int y = 0; y < lh; ++y)
                for (int x = 0; x < lw; ++x) {
                    if (used[g][lv][size_t(y) * lw + x]) continue;
                    uint8_t* cell = &m.q[g][(size_t(oy + y) * aw + ox + x) * gs.c];
                    for (int c = 0; c < gs.c; ++c) {
                        if (cell[c] != zeroQ(gs.bits)) ++zeroed;
                        cell[c] = uint8_t(zeroQ(gs.bits));
                    }
                }
        }
    }
    return zeroed;
}

} // namespace gtc::v4
