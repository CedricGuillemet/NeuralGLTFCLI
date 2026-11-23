# BABYLON_materials_neural_texture

Status: experimental vendor extension, written by `gltf-texture-compressor`. Not ratified.

## Overview

This material extension replaces some or all of a material's core PBR textures with a neural texture: a codec-defined payload that a renderer decodes per pixel (in the fragment shader or once at load time) and uses in place of the original textures. The `babylon-ntc-*` codecs store low-resolution latent images plus a small multilayer perceptron (MLP) as ordinary glTF images and buffer views; `nvidia-ntc-0.10` embeds an NVIDIA Neural Texture Compression (`.ntc`) container in a buffer view.

The extension is listed in `extensionsUsed` only. Replaced slots still point to valid textures (1x1 placeholders, downscaled originals with `--neural-fallback-size`, or KTX2 textures with `--neural-fallback ktx`), so loaders that ignore it render a plausible approximation.

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
| `codec` | Payload layout: `babylon-ntc-v1`, `babylon-ntc-v2`, `babylon-ntc-v3`, `babylon-ntc-v4` (the encoder default) or `nvidia-ntc-0.10`. A reader that does not know the codec must ignore the extension (see [Unknown codecs](#unknown-codecs)). |
| `resolution` | Resolution of the source textures (informative). |
| `latentTextures` | (`babylon-ntc-*`) Texture infos for the latent images, in codec order. They must be sampled as linear data (not sRGB) with bilinear filtering (NEAREST with manual bilinear in v3) and repeat wrapping, using `TEXCOORD_0`. |
| `weights` | (`babylon-ntc-*`) `bufferView` holding the network parameters: little-endian float32 for v1-v3 (`componentType`, `count`), little-endian float16 for v4 (`byteLength`). |
| `bitstream` | (`nvidia-ntc-0.10`) `{ "bufferView": n }`: the `.ntc` file. |
| `ntc` | (`nvidia-ntc-0.10`) Informative container summary. |
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

`babylon-ntc-v2` is a legacy codec of the encoder (`--neural-codec babylon-ntc-v2`; the default is now [`babylon-ntc-v4`](#codec-babylon-ntc-v4)). It keeps the latent layout, the 24 inputs, the positional encoding, the two hidden layers and the activation of `babylon-ntc-v1`, but the output layer has a variable number of outputs `N` (1 to 20) so that more texture slots can be replaced: emissive, and the `KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness textures, in addition to the core PBR slots.

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
| 16-18 | `specular` | 3 | `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` RGB | sRGB-encoded |
| 19 | `glossiness` | 1 | `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` A | linear |

Glossiness is the alpha channel of the spec-gloss texture, as in `KHR_materials_pbrSpecularGlossiness` (1.0 if the texture has no alpha). Specular RGB is sRGB-encoded as in that extension; glossiness is linear. As in v1, decoded values are clamped to `[0,1]` and the factors of the material still apply.

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

## Codec `babylon-ntc-v3`

`babylon-ntc-v3` generalises v2 for the higher-quality presets (`v3s`, `v3m`, `v3l`, `v3x`): any number of latent grids, 4-bit or 8-bit latents, any number of hidden layers and a choice of activation. The output channel table (20 canonical channels, `outputs` ranges) and the `replacedSlots` rules are **identical to v2**. The v2 layout is unchanged; with `--neural-codec babylon-ntc-v2` the encoder keeps writing v2 (`--neural-preset v2best`, that codec's default) unless a v3 preset or a non-v2 network shape is requested. Legacy codecs need a build with `-DNTC_SHAPE_SET=all`.

### Example

```json
"BABYLON_materials_neural_texture": {
  "version": 1,
  "codec": "babylon-ntc-v3",
  "resolution": [2048, 2048],
  "latentTextures": [{ "index": 0 }, { "index": 1 }, { "index": 2 }, { "index": 3 }, { "index": 4 }, { "index": 5 }, { "index": 6 }, { "index": 7 }],
  "grids": [
    { "resolution": [512, 512], "channels": 24, "bits": 4, "sampling": "bilinear", "textures": [0, 1, 2] },
    { "resolution": [128, 128], "channels": 16, "bits": 4, "sampling": "bilinear", "textures": [3, 4] }
  ],
  "weights": { "bufferView": 8, "componentType": 5126,   "count": 8596 },
  "network": {
    "inputs": 48, "hidden": [64, 64], "outputs": 20, "activation": "silu",
    "inputLayout": ["grid0", "grid1", "posenc"],
    "positionalEncoding": { "frequencies": [1, 2], "grid": 0 },
    "weightLayout": "W0[in][out],b0,W1,b1,W2,b2 float32"
  },
  "outputs": { "baseColor": [0, 4], "normal": [4, 2], "occlusion": [6, 1], "roughness": [7, 1], "metallic": [8, 1] },
  "replacedSlots": ["baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture"]
}
```

(The numbers above are illustrative; `hiResolution`, `loResolution` and `latentChannels` of v2 are replaced by `grids`.)

### Latent grids

- `grids[]` lists the grids from the **finest (grid 0) to the coarsest**. Each entry has the grid `resolution` `[w, h]`, the number of `channels`, the `bits` per channel (4 or 8), and `textures`: the indices into `latentTextures` holding the grid.
- A grid texel is `channels * bits` bits, stored as `ceil(channels * bits / 32)` RGBA8 images (`textures`, in order). Byte `j` of the texel is component `j % 4` of image `j / 4`; bytes past the end are 0.
  - 8-bit: byte `j` is channel `j`.
  - 4-bit: byte `k` holds channel `2k` in the high nibble and channel `2k + 1` in the low nibble.
- Decoded value: `q / (2^bits - 1)`, so 8-bit is `byte / 255` and 4-bit is `nibble / 15`.
- Latent images are written with a **NEAREST** sampler without mipmaps and `REPEAT` wrapping (they must be sampled as linear data). The reader performs the bilinear interpolation itself, with the same texel-centre convention as hardware bilinear (`p = uv * resolution - 0.5`, `REPEAT` wrap on the four taps), so that 4-bit values are interpolated after unpacking. Using a hardware-filtered sampler on packed 4-bit data is incorrect.
- Because the grids are sampled at texel centres during training, the reader must use `TEXCOORD_0` unmodified.

### Network

- **Inputs:** `network.inputLayout` gives the order, `grid0`, `grid1`, ... then `posenc`; the grid channels appear in channel order. `network.inputs` is the sum of grid channels plus 8.
- **Positional encoding (8 values):** `fx = fract(u * w0)`, `fy = fract(v * h0)` with `(w0, h0)` the resolution of the grid named by `positionalEncoding.grid` (0, the finest), then
  `sin(2 pi fx), sin(2 pi fy), cos(2 pi fx), cos(2 pi fy), sin(4 pi fx), sin(4 pi fy), cos(4 pi fx), cos(4 pi fy)`.
- **Layers:** `hidden` lists the width of each hidden layer; `activation` is `relu`, `silu` (`z * sigmoid(z)`) or `hgelu` (`min(z, 3) * clamp(z / 3 + 0.5, 0, 1)`; a hard-swish-like shape that is 0 below -1.5, `z (z + 1.5) / 3` on [-1.5, 1.5], `z` on [1.5, 3] and saturates at 3. It is **not** the hardGELU of [babylon-ntc-v4](#codec-babylon-ntc-v4), which is linear above 1.5; the string is kept for v3 compatibility, and the encoder calls this mode `hardswish`); the output layer is linear and clamped to `[0,1]` after decode.
- **Weights:** float32, `W0[in][out], b0, W1, b1, ..., Wn, bn`; the count is the sum over layers of `(in_l + 1) * out_l`, with the last layer having `N = network.outputs` outputs. `network.weightLayout` repeats the layer list.

### Reader requirements

A reader supporting v3 must implement the generic packing above (not a fixed shape): it may bound the supported network (the encoder's presets use at most 56 inputs, 3 hidden layers of 64 and at most 3 grids), and must ignore the extension for the material otherwise (see [Unknown codecs](#unknown-codecs)).

## Codec `babylon-ntc-v4`

`babylon-ntc-v4` is the default codec of the encoder. It is an NTC-style codec after Vaidyanathan et al. 2023, "Random-Access Neural Compression of Material Textures" (arXiv 2305.17105). It is a clean-room implementation written from the paper and from this repository's own code; it is **not compatible with NVIDIA NTC files** (use `nvidia-ntc-0.10` for those). The output channel table (20 canonical channels, `outputs` ranges), the colour spaces, the post-processing (normal Z reconstruction) and the `replacedSlots` rules are identical to v2.

| Item | v4 choice | Source |
|---|---|---|
| Feature pyramid | `L` levels; level `j` is a grid pair G0 / G1, each grid 4x smaller per axis than level `j-1`. Level 0 predicts mips 0-3, level `j >= 1` mips `2j+2` and `2j+3`, the last level also all remaining mips | paper 4.1, Table 1 |
| G0 | high resolution (texture/4 or texture/2), **4 raw taps**: the 2x2 neighbouring cells are concatenated unweighted | paper 4.3.1 |
| G1 | half of G0 on both axes, **bilinear** | paper 4.3.1 |
| Latent quantisation | scalar, 4 or 2 bits per grid, asymmetric range with an exact zero; training uses +-Q/2 noise, then the latents are frozen and the network is fine-tuned | paper 4.2 |
| Positional encoding | triangle waves tiled 8x8 texels, 3 octaves (periods 8, 4, 2): 12 values | paper 4.3.2 |
| LOD input | one scalar `m / (M-1)` | paper 4.4 (normalisation: ours) |
| MLP | `IN -> 64 -> 64 -> N_net`, one network for all mips, hardGELU (`z < -1.5 -> 0`, `z > 1.5 -> z`, else `z (z + 1.5) / 3`) | paper 4.4 |
| Weights | fp16 (IEEE binary16) storage, fp32 math | paper 5.1 |
| Output | per channel `none` (linear, clamped to [0,1]) or `sigmoid` | paper 4.4; sigmoid: three-ntc (MIT) |
| Constant channels | channels constant over the UV-covered texels are stored as constants and removed from the network | three-ntc (MIT) |
| Filtering | not part of the decode; readers filter in software (bake per texel, or nearest / bilinear / trilinear / stochastic at runtime) | paper 5.3 |

### Example

```json
"BABYLON_materials_neural_texture": {
  "version": 1,
  "codec": "babylon-ntc-v4",
  "resolution": [2048, 2048],
  "mipCount": 12,
  "levels": 5,
  "latentTextures": [{ "index": 0 }, { "index": 1 }, { "index": 2 }, { "index": 3 }],
  "grids": [
    { "resolution": [512, 512], "channels": 12, "bits": 4, "sampling": "concat2x2", "textures": [0, 1] },
    { "resolution": [256, 256], "channels": 16, "bits": 4, "sampling": "bilinear",  "textures": [2, 3] }
  ],
  "weights": { "bufferView": 9, "byteLength": 19858 },
  "network": {
    "inputs": 77, "hidden": [64, 64], "outputs": 9, "activation": "hardgelu",
    "inputLayout": ["grid0", "grid1", "posenc", "lod"],
    "positionalEncoding": { "type": "triangle", "periods": [8, 4, 2] },
    "weightFormat": "float16",
    "layers": [ { "in": 77, "inStride": 80, "out": 64 }, { "in": 64, "inStride": 64, "out": 64 }, { "in": 64, "inStride": 64, "out": 9 } ]
  },
  "channels": [ { "net": 0, "activation": "none" }, { "net": 1, "activation": "none" }, "...", { "const": 0.0 } ],
  "outputs": { "baseColor": [0, 3], "normal": [3, 2], "occlusion": [5, 1], "roughness": [6, 1], "metallic": [7, 1] },
  "replacedSlots": ["baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture"]
}
```

(`...` stands for the remaining entries; the example is illustrative.)

### Fields

| Field | Description |
|---|---|
| `mipCount` | `M = floor(log2(max(W, H))) + 1`. Must match the rule; readers reject the extension otherwise and use the fallback. |
| `levels` | `L`, the number of pyramid levels, the largest `L` with `W1 % 4^(L-1) == 0` and `H1 % 4^(L-1) == 0` where `(W1, H1)` is the G1 level-0 size. Must match. |
| `grids[]` | Exactly 2 grids, in order: `concat2x2` then `bilinear`. `resolution` is the level-0 size of the grid, `channels` the channel count, `bits` is 2 or 4, `textures` the indices into `latentTextures` (G0 images first). G1 is exactly half of G0 and `W / W0` is 2 or 4. |
| `grids[].sampling` | `concat2x2` (4 raw taps) or `bilinear`. |
| `weights` | `bufferView` and `byteLength` (`2 * sum(out_l * (inStride_l + 1))`). |
| `network.layers` | Per layer `in`, `inStride = ceil4(in)`, `out`. |
| `network.weightFormat` | `float16`. |
| `network.positionalEncoding` | `{ "type": "triangle", "periods": [8, 4, 2] }`. |
| `network.inputs` | `4 * C0 + C1 + 13`. |
| `channels[]` | Length N (the logical channels in the v2 order). Each entry is `{ "net": k, "activation": "none" or "sigmoid" }` or `{ "const": v }` with `v` in [0,1] in the decoded (encoded-colour) space. The `net` indices are 0 ... N_net-1 in ascending order and `network.outputs = N_net`. A group whose channels are all `const` may be folded into a uniform by the reader. |

### Pyramid, packing and latent values

- `size(g, lv) = (Wg >> 2*lv, Hg >> 2*lv)`; `levelOf(m) = min(L-1, max(0, (m >> 1) - 1))`.
- Each grid is stored as one RGBA8 PNG **atlas** per `ceil(C * B / 32)` images, holding all levels: level 0 at (0, 0), levels `lv >= 1` in a column at `x = Wg`, stacked from `y = sum_{k=1}^{lv-1} (Hg >> 2k)`. The atlas is `(Wg + (Wg >> 2)) x Hg` pixels when `L > 1`, else `Wg x Hg`; unused pixels are 0. Images are NEAREST, no mips; readers fetch with `texelFetch`/`textureLoad` and wrap manually inside the level.
- A cell is `C * B` bits. Byte `k` of the cell is component `k % 4` of image `k / 4`. Channel `c` is in byte `(c * B) / 8`, MSB first: `shift = 8 - B - (c * B) % 8`, `q = (byte >> shift) & (2^B - 1)`.
- Latent value `(q - (2^(B-1) - 1)) / 2^B`: 4-bit gives [-7/16, 8/16], 2-bit gives [-1/4, 2/4]; zero is exact.
- The encoder zeroes atlas cells that no UV-covered texel reads at any mip, so that deflate removes them.

### Decode

All arithmetic is float32 (fp16 weights convert exactly). Row `j = 0` is the image top: texel `(i, j)` of mip `m` is at `uv = ((i + 0.5) / Wm, (j + 0.5) / Hm)`.

```
size(g, lv)    = (Wg >> 2*lv, Hg >> 2*lv)
levelOf(m)     = min(L-1, max(0, (m >> 1) - 1))
origin(g, lv)  = lv == 0 ? (0, 0) : (Wg, sum_{k=1}^{lv-1} (Hg >> 2k))

float latent(g, lv, x, y, c):
    (w, h) = size(g, lv) ; x = mod(x, w) ; y = mod(y, h)              // REPEAT inside the level
    q = extract(g, origin(g, lv) + (x, y), c)                         // integer in [0, 2^B - 1]
    return float(q - (2^(B-1) - 1)) / float(2^B)

inputs(m, i, j):
    x = [] ; lv = levelOf(m)
    // G0: 4 raw taps. r = grid cells per mip texel, a power of two, exact in float32
    (w0, h0) = size(0, lv) ; rx = w0 / Wm ; ry = h0 / Hm
    gx = (i + 0.5) * rx - 0.5 ; gy = (j + 0.5) * ry - 0.5 ; cx = floor(gx) ; cy = floor(gy)
    for (dx, dy) in [(0,0), (1,0), (0,1), (1,1)]:
        for c in 0..C0-1: x.push(latent(0, lv, cx+dx, cy+dy, c))
    // G1: bilinear
    (w1, h1) = size(1, lv) ; rx = w1 / Wm ; ry = h1 / Hm
    gx = (i + 0.5) * rx - 0.5 ; gy = (j + 0.5) * ry - 0.5
    cx = floor(gx) ; cy = floor(gy) ; fx = gx - cx ; fy = gy - cy
    for c in 0..C1-1:
        a = latent(1,lv,cx,cy,c) ; b = latent(1,lv,cx+1,cy,c) ; d = latent(1,lv,cx,cy+1,c) ; e = latent(1,lv,cx+1,cy+1,c)
        x.push((1-fy)*((1-fx)*a + fx*b) + fy*((1-fx)*d + fx*e))
    // triangle PE: 8x8 tile, on mip-m texel indices
    triC(t) = 4*abs(fract(t) - 0.5) - 1        // 1 at t=0, -1 at t=0.5
    triS(t) = triC(t - 0.25)
    for p in (i, j): for P in (8, 4, 2): x.push(triC(p / P)) ; x.push(triS(p / P))
    x.push(float(m) / float(max(1, M - 1)))     // LOD
    return x                                    // length IN = 4*C0 + C1 + 13

// layer l: in_l, inStride_l = ceil(in_l/4)*4, out_l; W_l fp16 [out_l][inStride_l] (padding = 0); b_l fp16 [out_l]
dense(l, v): y[o] = float(b_l[o]) + sum_{k=0}^{in_l-1} float(W_l[o][k]) * v[k]       // k ascending
hardgelu(z) = z < -1.5 ? 0 : (z > 1.5 ? z : z * (z + 1.5) / 3.0)
net(m,i,j):  h = hardgelu(dense(0, inputs(m,i,j))) ; h = hardgelu(dense(1, h)) ; return dense(2, h)   // N_net values

decode(m, i, j):                                // returns N logical channels
    y = net(m, i, j)                            // skipped entirely if N_net == 0
    for n in 0..N-1:
        ch = channels[n]
        if "const" in ch:            out[n] = ch.const
        elif ch.activation=="sigmoid": out[n] = 1 / (1 + exp(-y[ch.net]))
        else /* "none" */:           out[n] = clamp(y[ch.net], 0, 1)
    return out
```

### Weights

One buffer view (4-byte aligned start), little-endian IEEE binary16. For each layer in order: `W_l[out_l][inStride_l]` (row-major, zero padding), then `b_l[out_l]`. The layout is `[out][in]`, unlike v1-v3. fp16 weights are rounded after training, before the verification decode, so the written bytes are the model that was measured.

### Filtering and mips

`decode` is defined only at integer texels of discrete mips; filtering is the reader's job. A reader may bake the textures once at load (per texel, `generate` or `neural` mip modes) or decode at runtime with nearest, bilinear, trilinear or stochastic (jittered UV and LOD) filtering. Latent samplers must not filter: the packed bytes are fetched exactly.

### Presets

`--neural-preset` selects one (default `v4s`). Numbers are for a 2048x2048, 9-channel material with a 5-level pyramid and no constant channels. bpp is total bits per texel, bppc is bpp / 9.

| Preset | Paper profile | G0 (level 0) | G1 (level 0) | IN | Total bytes | bpp | bppc |
|---|---|---|---|---|---|---|---|
| `v4xs` | NTC 0.2 | 512^2, 8 ch, 2 bit | 256^2, 12 ch, 4 bit | 57 | 995,968 | 1.90 | 0.21 |
| `v4s` (default) | NTC 0.5 | 512^2, 24 ch, 2 bit | 256^2, 12 ch, 4 bit | 121 | 2,262,450 | 4.32 | 0.48 |
| `v4m` | NTC 1.0 | 1024^2, 12 ch, 2 bit | 512^2, 10 ch, 4 bit | 71 | 4,772,374 | 9.10 | 1.01 |
| `v4l` | NTC 2.25 | 1024^2, 16 ch, 4 bit | 512^2, 12 ch, 4 bit | 89 | 10,646,954 | 20.3 | 2.26 |

Sizes are logical (packed latents plus fp16 weights); the PNG payload is smaller wherever the atlas has unused or dead cells. The material must be a multiple of 8 in both axes.

The reference encoder (not part of the format) trains 40,000 Adam steps of 65,536 samples (latent LR 0.02, network LR 0.005, cosine decay, latents perturbed by uniform quantisation noise), where 90% of the steps train mip 0 and the others a geometrically distributed coarser mip, then fine-tunes the network on the frozen quantised latents. The default output activation is `sigmoid` for base, normal, occlusion and roughness and `none` otherwise.

### Test vectors

Cases: `synth_s`, `synth_xs2b`, `synth_m_rect` and `synth_const` (synthetic weights that exercise every branch of the decoder: constant channels, mixed activations, 2-bit latents, non-square sizes; regenerated by the hidden `--neural-golden <dir>`) and `trained_leather` (a real v4s model trained for 3000 steps on a 64x64 crop of the FlightHelmet LeatherPartsMat; its input is `_src/leather64.glb`, made by `tools/v4/make_leather_crop.py`, and it is written by setting `GTC_V4_GOLDEN_DIR` and `GTC_V4_GOLDEN_NAME` while encoding that file with `--steps 3000`).

`tests/golden/v4/<case>/` holds `case.glb` and the reference decode of every mip (`ref_m<m>.f32`, float32 `[j][i][n]` before normal-Z and colour conversion), the unpacked latents, the unpadded weights (`weights_l<k>.f32`, fp16 as f32, `[out][in]` followed by the bias) and `meta.json` with the sha256 of every file. The reference decoder is `src/neural/ntc_v4_ref.cpp`. Tolerances for other implementations: GPU readback RGBA8 |diff| <= 1/255 on >= 99.9% of texel-channels (max 3/255); a CPU port max |diff| <= 1e-5.

### v4 changelog

- 1.0 (this revision): initial layout, as above.

### Reader requirements

A v4 reader must implement the decode above exactly. A reader that does not support the `version`, `network` layout or `inputs` of a material falls back as described in [Unknown codecs](#unknown-codecs).

The encoder reads the material at its native resolution, which must be a multiple of 8 in both axes (the encoder reports an error otherwise); the output textures have the same resolution.

## Codec `nvidia-ntc-0.10`

`nvidia-ntc-0.10` is an opt-in codec of the encoder (`--neural-codec nvidia-ntc`). The material's textures are compressed by NVIDIA's [RTX Neural Texture Compression](https://github.com/NVIDIA-RTX/RTXNTC) SDK (`ntc-cli`, LibNTC 0.10) and the resulting `.ntc` file is embedded **unchanged** in the glb. The codec id names the SDK minor version whose decoder was validated: a future SDK version gets its own id (`nvidia-ntc-0.11`), and a reader only decodes the ids it has validated.

### Example

```json
"BABYLON_materials_neural_texture": {
  "version": 1,
  "codec": "nvidia-ntc-0.10",
  "resolution": [2048, 2048],
  "bitstream": { "bufferView": 65 },
  "ntc": {
    "containerVersion": 256, "schemaVersion": 3, "numChannels": 10,
    "gridSizeScale": 3, "numFeatures": 8, "numColorMips": 1,
    "encoder": "ntc-cli (LibNTC 0.10.1)"
  },
  "outputs": { "baseColor": [0, 4], "normal": [4, 2], "occlusion": [6, 1], "roughness": [7, 1], "metallic": [8, 1] },
  "replacedSlots": ["baseColorTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture"]
}
```

### Embedding

- `bitstream.bufferView` references a buffer view holding the whole `.ntc` file, byte for byte. The buffer view starts at a 16-byte aligned offset and has no `byteStride` or `target`.
- `latentTextures`, `weights`, `network` and the other `babylon-ntc-*` properties are absent.
- `ntc` is informative, except that `containerVersion` and `schemaVersion` must equal the values in the container.
- The container format is defined by NVIDIA's public [TextureSetFile.md](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/TextureSetFile.md) (header `"NTEX"`, container version `0x100`, JSON descriptor, binary views). The decoding semantics (latent sampling, the MLP and its quantization, positional encoding, colour mips) are defined by NVIDIA's NTC runtime v0.10, not by this document.

### Channels and slots

- `outputs` uses the v2 range semantics `[firstChannel, count]`, where the channels are the channels of the NTC texture set (`numChannels` is 1 to 16). The groups are the v2 groups (`baseColor`, `normal`, `occlusion`, `roughness`, `metallic`, `emissive`, `diffuse`, `specular`, `glossiness`), packed in the v2 canonical order, and each group is one texture of the texture set whose name is the group name (ntc-cli may append `|ntcsem:...`) and whose `firstChannel`/`numChannels` equal the `outputs` entry. `baseColor` is 4 channels (alpha is the fourth), `normal` is 2 (X and Y, Z is reconstructed). The v2 colour-space table and the `replacedSlots` rules apply unchanged.
- The `numChannels` of the texture set may exceed the last used channel by one (ntc-cli pads odd counts).
- At most 16 channels fit: groups are added in canonical order until the next one would not fit; the other groups are left as regular textures and are not in `replacedSlots`.
- Every channel is declared `Linear` in the texture set, and sRGB-encoded sources (base colour, emissive, diffuse, specular) are stored as data. Decoded values are therefore sRGB-encoded, exactly like the v2 outputs.

### Encoder constraints and validation

The encoder writes containers with `--gdeflate none` (no GDeflate views), without BC output (no `bcFormat` or BC mode buffers) and with `numColorMips == 1` unless `--ntc-mips` is given. A reader must reject, and fall back as for [Unknown codecs](#unknown-codecs), a container whose signature is not `"NTEX"`, whose container version is not `0x100`, whose `schemaVersion` is not 3, which uses compression or BC data, or whose channels are not `Linear`. The CLI validates every container before embedding it.

### Fallback textures

By default the replaced slots hold 1x1 placeholders. `--neural-fallback-size <n>` stores downscaled PNG originals, `--neural-fallback ktx` stores KTX2 textures (`KHR_texture_basisu` is then added to `extensionsUsed`, not to `extensionsRequired`), and `--neural-fallback none` leaves the core slots without a texture.

## Limitations

- Materials using `KHR_texture_transform` or a `texCoord` other than 0 on a texture slot are left untouched for that slot.
- `babylon-ntc-v1` only replaces the core PBR slots; emissive and extension textures are kept as they are. `babylon-ntc-v2` also replaces emissive and the spec-gloss textures.
- Slots of one material with different resolutions are resampled to the largest resolution before encoding.
- With `babylon-ntc-v2`, `babylon-ntc-v3` and `babylon-ntc-v4`, materials whose textures are smaller than 64 pixels are not encoded. `babylon-ntc-v4` additionally needs a resolution that is a multiple of 8.
- `babylon-ntc-v3` needs a reader that implements the generic grid/MLP layout; the Babylon.js loader support for v3 is tracked separately.
- `babylon-ntc-v4` needs a reader that implements the v4 decode (latent pyramid, fp16 MLP) described above, and a CUDA GPU at encode time.
- `nvidia-ntc-0.10` needs a reader that implements NVIDIA's NTC 0.10 decoding; it replaces at most 16 output channels per material, and ntc-cli needs an NVIDIA GPU at encode time.
