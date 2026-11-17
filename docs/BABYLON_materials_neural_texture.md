# BABYLON_materials_neural_texture

Status: experimental vendor extension, written by `gltf-texture-compressor`. Not ratified.

## Overview

This material extension replaces some or all of a material's core PBR textures with a neural texture: low-resolution latent images plus a small multilayer perceptron (MLP). A renderer samples the latent images with ordinary bilinear filtering, evaluates the MLP per pixel (in the fragment shader or once at load time), and uses the outputs in place of the original textures.

The extension is listed in `extensionsUsed` only. Replaced slots still point to valid textures (1x1 placeholders, or downscaled originals with `--neural-fallback-size`), so loaders that ignore it render a plausible approximation.

## Example

```json
"materials": [{
  "name": "GlassPlasticMat",
  "pbrMetallicRoughness": {
    "baseColorTexture": { "index": 12 },
    "metallicRoughnessTexture": { "index": 14 }
  },
  "normalTexture": { "index": 13 },
  "occlusionTexture": { "index": 14 },
  "extensions": {
    "BABYLON_materials_neural_texture": {
      "version": 1,
      "codec": "babylon-ntc-v1",
      "resolution": [2048, 2048],
      "hiResolution": [512, 512],
      "loResolution": [128, 128],
      "latentChannels": 8,
      "latentTextures": [{ "index": 0 }, { "index": 1 }, { "index": 2 }, { "index": 3 }],
      "weights": { "bufferView": 7, "componentType": 5126, "count": 2153 },
      "network": {
        "inputs": 24, "hidden": [32, 32], "outputs": 9, "activation": "relu",
        "positionalEncoding": { "frequencies": [1, 2], "scale": "hiResolution" },
        "weightLayout": "W0[in][out],b0,W1,b1,W2,b2 float32"
      },
      "outputs": {
        "baseColor": [0, 1, 2, 3], "normal": [4, 5], "occlusion": [6], "roughness": [7], "metallic": [8]
      },
      "replacedSlots": ["baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture"]
    }
  }
}]
```

## Properties

| Property | Description |
|---|---|
| `version` | Extension version, currently 1. |
| `codec` | Network and latent layout. Only `babylon-ntc-v1` is defined. |
| `resolution` | Resolution of the source textures (informative). |
| `latentTextures` | Texture infos for the latent images, in codec order. They must be sampled as linear data (not sRGB) with bilinear filtering and repeat wrapping, using `TEXCOORD_0`. |
| `weights` | `bufferView` holding little-endian float32 network parameters. |
| `outputs` | Network output channel indices for each material property. |
| `replacedSlots` | Core texture slots whose texture is superseded by the network. Their factors (`baseColorFactor`, `normalTexture.scale`...) still apply to the decoded values. |

## Codec `babylon-ntc-v1`

- **Latents:** two grids of 8 channels. The high grid is `ceil(W/4) x ceil(H/4)`, the low grid `ceil(W/16) x ceil(H/16)`. Each grid is stored as two RGBA8 images: `latentTextures` = high channels 0-3, high 4-7, low 0-3, low 4-7. Values are `byte / 255`.
- **Inputs (24):** 8 high-grid values, 8 low-grid values, then the positional encoding `sin(p.x), sin(p.y), cos(p.x), cos(p.y), sin(2p.x), sin(2p.y), cos(2p.x), cos(2p.y)` with `p = uv * hiResolution * 2pi`. Computing `fract(uv * hiResolution)` first keeps precision.
- **Network:** `24 -> 32 (ReLU) -> 32 (ReLU) -> 9` (linear output). Parameters are stored as `W0` (24x32, row-major `[input][output]`), `b0` (32), `W1` (32x32), `b1` (32), `W2` (32x9), `b2` (9): 2153 floats.
- **Outputs:** 0-3 base color RGBA (sRGB-encoded values, as in the original texture), 4-5 normal X/Y in `[0,1]` (reconstruct `z = sqrt(1 - x^2 - y^2)` after remapping to `[-1,1]`), 6 occlusion, 7 roughness, 8 metallic. Clamp to `[0,1]`.

Fragment shader decode (GLSL sketch):

```glsl
vec4 h0 = texture(lat0, uv), h1 = texture(lat1, uv), l0 = texture(lat2, uv), l1 = texture(lat3, uv);
vec2 f = fract(uv * hiRes) * 6.28318530718;
float x[24] = float[24](h0.x,h0.y,h0.z,h0.w, h1.x,h1.y,h1.z,h1.w, l0.x,l0.y,l0.z,l0.w, l1.x,l1.y,l1.z,l1.w,
                        sin(f.x), sin(f.y), cos(f.x), cos(f.y), sin(2.0*f.x), sin(2.0*f.y), cos(2.0*f.x), cos(2.0*f.y));
// a1 = relu(x * W0 + b0); a2 = relu(a1 * W1 + b1); out = a2 * W2 + b2
```

## Limitations

- Materials using `KHR_texture_transform` or a `texCoord` other than 0 on their PBR textures are left untouched.
- Other textures (emissive, extension textures) are kept as they are.
