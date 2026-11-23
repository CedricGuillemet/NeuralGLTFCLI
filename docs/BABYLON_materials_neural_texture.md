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
| `codec` | Network and latent layout: `babylon-ntc-v1` or `babylon-ntc-v2`. A reader that does not know the codec must ignore the extension (see [Unknown codecs](#unknown-codecs)). |
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

## Codec `babylon-ntc-v2`

`babylon-ntc-v2` is the default codec of the encoder. It keeps the latent layout, the 24 inputs, the positional encoding, the two hidden layers and the activation of `babylon-ntc-v1`, but the output layer has a variable number of outputs `N` (1 to 20) so that more texture slots can be replaced: emissive, and the `KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness textures, in addition to the core PBR slots.

### Channels

There is one canonical 20-channel table. A material uses only the channels it needs; the active channels are packed to the front in canonical order, and `outputs` records where each group starts.

| Canonical channels | Group (`outputs` key) | Components | Source slot | Colour space |
|---|---|---|---|---|
| 0-3 | `baseColor` | 3, or 4 when `alphaMode` is not `OPAQUE` | `pbrMetallicRoughness.baseColorTexture` | sRGB-encoded |
| 4-5 | `normal` | 2 (X, Y; Z is reconstructed as in v1) | `normalTexture` | linear |
| 6 | `occlusion` | 1 | `occlusionTexture` R | linear |
| 7 | `roughness` | 1 | `metallicRoughnessTexture` G | linear |
| 8 | `metallic` | 1 | `metallicRoughnessTexture` B | linear |
| 9-11 | `emissive` | 3 | `emissiveTexture` | sRGB-encoded |
| 12-15 | `diffuse` | 3, or 4 when `alphaMode` is not `OPAQUE` | `KHR_materials_pbrSpecularGlossiness.diffuseTexture` | sRGB-encoded |
| 16-18 | `specular` | 3 | `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` RGB | linear |
| 19 | `glossiness` | 1 | `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` A | linear |

Glossiness is the alpha channel of the spec-gloss texture, as in `KHR_materials_pbrSpecularGlossiness` (1.0 if the texture has no alpha). As in v1, decoded values are clamped to `[0,1]` and the factors of the material still apply.

### Example

```json
"BABYLON_materials_neural_texture": {
  "version": 1,
  "codec": "babylon-ntc-v2",
  "resolution": [2048, 2048],
  "hiResolution": [512, 512],
  "loResolution": [128, 128],
  "latentChannels": 8,
  "latentTextures": [{ "index": 0 }, { "index": 1 }, { "index": 2 }, { "index": 3 }],
  "weights": { "bufferView": 8, "componentType": 5126, "count": 2516 },
  "network": {
    "inputs": 24, "hidden": [32, 32], "outputs": 20, "activation": "relu",
    "positionalEncoding": { "frequencies": [1, 2], "scale": "hiResolution" },
    "weightLayout": "W0[in][out],b0,W1,b1,W2,b2 float32"
  },
  "outputs": {
    "baseColor": [0, 4], "normal": [4, 2], "occlusion": [6, 1], "roughness": [7, 1], "metallic": [8, 1],
    "emissive": [9, 3], "diffuse": [12, 4], "specular": [16, 3], "glossiness": [19, 1]
  },
  "replacedSlots": [
    "baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture", "emissiveTexture",
    "KHR_materials_pbrSpecularGlossiness.diffuseTexture",
    "KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture"
  ]
}
```

### What differs from v1

- **`network.outputs`** is `N`, the number of active channels (1 to 20), instead of the fixed 9.
- **`weights.count`** is `1856 + 33 N` floats: `W0` (24x32), `b0` (32), `W1` (32x32), `b1` (32), `W2` (32xN, row-major `[input][output]`), `b2` (N). v1 is the special case N = 9 (2153 floats). `weights` is a float32 buffer view of `count * 4` bytes.
- **`outputs`** values are `[startChannel, componentCount]` ranges into the N network outputs. Note that v1 lists channel indices (for example `"baseColor": [0, 1, 2, 3]`); a v2 reader must not reuse the v1 interpretation. Ranges do not overlap and every channel `0..N-1` belongs to exactly one group.
- **`replacedSlots`** may additionally contain `emissiveTexture` and the dotted names `KHR_materials_pbrSpecularGlossiness.diffuseTexture` and `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture`.
- Each replaced slot requires specific `outputs` keys:

| Replaced slot | Required `outputs` keys |
|---|---|
| `baseColorTexture` | `baseColor` |
| `normalTexture` | `normal` |
| `occlusionTexture` | `occlusion` |
| `metallicRoughnessTexture` | `roughness` and `metallic` |
| `emissiveTexture` | `emissive` |
| `KHR_materials_pbrSpecularGlossiness.diffuseTexture` | `diffuse` |
| `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` | `specular` and `glossiness` |

Replaced slots point to valid fallback textures (1x1 placeholders or downscaled originals), including the slots inside the `KHR_materials_pbrSpecularGlossiness` extension.

Decode: the network is evaluated exactly as in v1 (`24 -> 32 (ReLU) -> 32 (ReLU) -> N`, linear output), and the groups are read from the `N` outputs using `outputs`.

### Unknown codecs

A reader that does not recognise `codec` (for example a v1-only loader seeing `babylon-ntc-v2`) must ignore the extension for that material and use the textures referenced by the material as they are: placeholders, or the downscaled originals written with `--neural-fallback-size`. It must not attempt to decode the weights.

## Limitations

- Materials using `KHR_texture_transform` or a `texCoord` other than 0 on a texture slot are left untouched for that slot.
- `babylon-ntc-v1` only replaces the core PBR slots; emissive and extension textures are kept as they are. `babylon-ntc-v2` also replaces emissive and the spec-gloss textures.
- Slots of one material with different resolutions are resampled to the largest resolution before encoding.
- With `babylon-ntc-v2`, materials whose textures are smaller than 64 pixels are not encoded.
