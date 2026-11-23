#include "uv_mask.h"

#include <algorithm>
#include <cmath>

namespace gtc {

namespace {

bool readAccessor(const tinygltf::Model& m, int index, std::vector<double>& out, int& comps) {
    if (index < 0 || index >= int(m.accessors.size())) return false;
    const auto& a = m.accessors[index];
    if (a.bufferView < 0 || a.bufferView >= int(m.bufferViews.size()) || a.sparse.isSparse) return false;
    const auto& bv = m.bufferViews[a.bufferView];
    if (bv.buffer < 0 || bv.buffer >= int(m.buffers.size())) return false;
    const auto& buf = m.buffers[bv.buffer].data;
    comps = tinygltf::GetNumComponentsInType(uint32_t(a.type));
    const int csize = tinygltf::GetComponentSizeInBytes(uint32_t(a.componentType));
    if (comps <= 0 || csize <= 0) return false;
    const size_t stride = bv.byteStride ? bv.byteStride : size_t(comps) * csize;
    const size_t base = bv.byteOffset + a.byteOffset;
    if (a.count && base + (a.count - 1) * stride + size_t(comps) * csize > buf.size()) return false;
    out.resize(a.count * comps);
    for (size_t i = 0; i < a.count; ++i)
        for (int c = 0; c < comps; ++c) {
            const uint8_t* p = &buf[base + i * stride + size_t(c) * csize];
            double v = 0;
            switch (a.componentType) {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: v = *reinterpret_cast<const float*>(p); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: v = *p / (a.normalized ? 255.0 : 1.0); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: v = *reinterpret_cast<const uint16_t*>(p) / (a.normalized ? 65535.0 : 1.0); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: v = *reinterpret_cast<const uint32_t*>(p); break;
            default: return false;
            }
            out[i * comps + c] = v;
        }
    return true;
}

struct Raster {
    int w, h;
    std::vector<uint8_t>& mask;

    // Marks the texels whose center lies within ~1 texel of the triangle (pixel coordinates, REPEAT wrapped).
    void triangle(const double (&px)[3], const double (&py)[3]) {
        const double minX = std::min({px[0], px[1], px[2]}), maxX = std::max({px[0], px[1], px[2]});
        const double minY = std::min({py[0], py[1], py[2]}), maxY = std::max({py[0], py[1], py[2]});
        if (maxX - minX > 4.0 * w || maxY - minY > 4.0 * h) {
            std::fill(mask.begin(), mask.end(), uint8_t(1));
            return;
        }
        const double area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
        const double sign = area < 0 ? -1.0 : 1.0;
        const double slack = 0.75;
        for (int y = int(std::floor(minY - slack)); y <= int(std::ceil(maxY + slack)); ++y)
            for (int x = int(std::floor(minX - slack)); x <= int(std::ceil(maxX + slack)); ++x) {
                const double cx = x + 0.5, cy = y + 0.5;
                bool inside = true;
                for (int e = 0; e < 3 && inside; ++e) {
                    const int f = (e + 1) % 3;
                    const double ex = px[f] - px[e], ey = py[f] - py[e];
                    const double len = std::sqrt(ex * ex + ey * ey);
                    // signed distance of the texel center to the edge, positive inside
                    const double d = len < 1e-9 ? 0.0 : sign * (ex * (cy - py[e]) - ey * (cx - px[e])) / len;
                    if (d < -slack) inside = false;
                }
                if (!inside) continue;
                int wx = x % w, wy = y % h;
                if (wx < 0) wx += w;
                if (wy < 0) wy += h;
                mask[size_t(wy) * w + wx] = 1;
            }
    }
};

void dilate(std::vector<uint8_t>& mask, int w, int h, int radius) {
    if (radius <= 0) return;
    std::vector<uint8_t> tmp(mask.size(), 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!mask[size_t(y) * w + x]) continue;
            for (int d = -radius; d <= radius; ++d) tmp[size_t(y) * w + ((x + d) % w + w) % w] = 1;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!tmp[size_t(y) * w + x]) continue;
            for (int d = -radius; d <= radius; ++d) mask[size_t(((y + d) % h + h) % h) * w + x] = 1;
        }
}

} // namespace

bool buildUvMask(const tinygltf::Model& m, int material, int width, int height, int dilateTexels, std::vector<uint8_t>& mask) {
    mask.assign(size_t(width) * height, 0);
    Raster r{width, height, mask};
    bool any = false;
    for (const auto& mesh : m.meshes)
        for (const auto& prim : mesh.primitives) {
            if (prim.material != material) continue;
            const int mode = prim.mode < 0 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
            if (mode != TINYGLTF_MODE_TRIANGLES && mode != TINYGLTF_MODE_TRIANGLE_STRIP && mode != TINYGLTF_MODE_TRIANGLE_FAN) continue;
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt == prim.attributes.end()) continue;
            std::vector<double> uv, idx;
            int uvComps = 0, idxComps = 0;
            if (!readAccessor(m, uvIt->second, uv, uvComps) || uvComps != 2) continue;
            const size_t vertexCount = uv.size() / 2;
            std::vector<size_t> ids;
            if (prim.indices >= 0) {
                if (!readAccessor(m, prim.indices, idx, idxComps)) continue;
                ids.assign(idx.begin(), idx.end());
            } else {
                ids.resize(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) ids[i] = i;
            }
            auto tri = [&](size_t a, size_t b, size_t c) {
                if (a >= vertexCount || b >= vertexCount || c >= vertexCount) return;
                const size_t v[3] = {a, b, c};
                double px[3], py[3];
                for (int k = 0; k < 3; ++k) {
                    px[k] = uv[v[k] * 2] * width;
                    py[k] = uv[v[k] * 2 + 1] * height;
                }
                // glTF UVs have their origin at the top-left, like the decoded images.
                const double tx = std::floor(std::min({px[0], px[1], px[2]}) / width) * width;
                const double ty = std::floor(std::min({py[0], py[1], py[2]}) / height) * height;
                for (int k = 0; k < 3; ++k) px[k] -= tx, py[k] -= ty;
                r.triangle(px, py);
                any = true;
            };
            if (mode == TINYGLTF_MODE_TRIANGLES) {
                for (size_t i = 0; i + 2 < ids.size(); i += 3) tri(ids[i], ids[i + 1], ids[i + 2]);
            } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
                for (size_t i = 0; i + 2 < ids.size(); ++i) tri(ids[i], ids[i + 1], ids[i + 2]);
            } else {
                for (size_t i = 1; i + 1 < ids.size(); ++i) tri(ids[0], ids[i], ids[i + 1]);
            }
        }
    if (any) dilate(mask, width, height, dilateTexels);
    return any;
}

} // namespace gtc
