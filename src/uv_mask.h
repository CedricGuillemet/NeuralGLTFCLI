#pragma once

#include <cstdint>
#include <vector>

#include "tiny_gltf.h"

namespace gtc {

// Rasterizes the TEXCOORD_0 triangles of every primitive that uses `material` into a width x height
// mask (nonzero = texel used by the mesh, REPEAT wrapped, dilated by `dilate` texels so bilinear
// filtering and mip levels stay covered). Returns false when no triangle was found.
bool buildUvMask(const tinygltf::Model& model, int material, int width, int height, int dilate, std::vector<uint8_t>& mask);

} // namespace gtc
