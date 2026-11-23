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
| `codec` | Payload layout: `babylon-ntc-v1`, `babylon-ntc-v2`, `babylon-ntc-v3` or `nvidia-ntc-0.10`. A reader that does not know the codec must ignore the extension (see [Unknown codecs](#unknown-codecs)). |
| `resolution` | Resolution of the source textures (informative). |
| `latentTextures` | (`babylon-ntc-*`) Texture infos for the latent images, in codec order. They must be sampled as linear data (not sRGB) with bilinear filtering (NEAREST with manual bilinear in v3) and repeat wrapping, using `TEXCOORD_0`. |
| `weights` | (`babylon-ntc-*`) `bufferView` holding little-endian float32 network parameters. |
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

`babylon-ntc-v3` generalises v2 for the higher-quality presets (`v3s`, `v3m`, `v3l`, `v3x`): any number of latent grids, 4-bit or 8-bit latents, any number of hidden layers and a choice of activation. The output channel table (20 canonical channels, `outputs` ranges) and the `replacedSlots` rules are **identical to v2**. The v2 layout is unchanged; the encoder keeps writing v2 (`--neural-preset v2best`, the default) unless a v3 preset or a non-v2 network shape is requested.

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
- **Layers:** `hidden` lists the width of each hidden layer; `activation` is `relu`, `silu` (`z * sigmoid(z)`) or `hgelu` (`min(z, 3) * clamp(z / 3 + 0.5, 0, 1)`); the output layer is linear and clamped to `[0,1]` after decode.
- **Weights:** float32, `W0[in][out], b0, W1, b1, ..., Wn, bn`; the count is the sum over layers of `(in_l + 1) * out_l`, with the last layer having `N = network.outputs` outputs. `network.weightLayout` repeats the layer list.

### Reader requirements

A reader supporting v3 must implement the generic packing above (not a fixed shape): it may bound the supported network (the encoder's presets use at most 56 inputs, 3 hidden layers of 64 and at most 3 grids), and must ignore the extension for the material otherwise (see [Unknown codecs](#unknown-codecs)).

## Codec `nvidia-ntc-0.10`

`nvidia-ntc-0.10` is the default codec of the encoder. The material's textures are compressed by NVIDIA's [RTX Neural Texture Compression](https://github.com/NVIDIA-RTX/RTXNTC) SDK (`ntc-cli`, LibNTC 0.10) and the resulting `.ntc` file is embedded **unchanged** in the glb. The codec id names the SDK minor version whose decoder was validated: a future SDK version gets its own id (`nvidia-ntc-0.11`), and a reader only decodes the ids it has validated.

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
- With `babylon-ntc-v2` and `babylon-ntc-v3`, materials whose textures are smaller than 64 pixels are not encoded.
- `babylon-ntc-v3` needs a reader that implements the generic grid/MLP layout; the Babylon.js loader support for v3 is tracked separately.
- `nvidia-ntc-0.10` needs a reader that implements NVIDIA's NTC 0.10 decoding; it replaces at most 16 output channels per material, and ntc-cli needs an NVIDIA GPU at encode time.
