# Design note: embedding the NVIDIA NTC bitstream in glTF (option a)

Status: design only, nothing here is implemented. The implemented comparison path is option (b): the
`nvidia-ntc` codec runs NVIDIA's `ntc-cli`, decodes the `.ntc` offline and writes plain PNG textures
(`<stem>_nvidia-ntc.glb`). That glb is only useful to compare quality, not to ship NTC data.

Reference: RTXNTC SDK v0.10.1-beta (LibNTC 0.10.1, tools `main-8311b54`), used as a black box.

## 1. Goal

Ship the `.ntc` payload inside the glb, in the same way `BABYLON_materials_neural_texture` ships our own
latents, and let the viewer decode it at load time (bake to plain textures) or at run time.

## 2. Proposed glTF representation

Reuse the existing extension and add a new codec id, so loaders that do not know it fall back to the
fallback textures ("unknown codecs" rule of the spec):

```json
"BABYLON_materials_neural_texture": {
  "codec": "nvidia-ntc-0.10",
  "bitstream": 12,
  "textures": [
    { "name": "baseColor", "channels": "RGB", "slot": "baseColorTexture" },
    { "name": "normal",    "channels": "RG",  "slot": "normalTexture"    },
    { "name": "orm",       "channels": "RGB", "slot": "occlusionTexture, metallicRoughnessTexture" }
  ],
  "resolution": [2048, 2048],
  "latentShape": { "gridSizeScale": 3, "numFeatures": 8 },
  "colorSpaces": { "baseColor": "linear" }
}
```

* `bitstream` is a bufferView holding the `.ntc` file unchanged, or, once the format is stable, a
  stripped payload (latent tensors plus int8 weights).
* The `.ntc` container embeds the texture names, channel counts and the latent shape, so most of the JSON
  above is informational. The loader still needs the slot mapping, which is why it is repeated in the
  extension.
* Mips: our baker does not need them (the CLI uses no `-g`), so they are not stored.

## 3. Decoder

The network (from `InferenceConstants.h` of the SDK) is a pyramid of two latent grids (4-bit features,
up to 16 per grid, stored as BGRA4 texture arrays), a positional encoding (3 waves, 12 inputs) and an int8
MLP 48 -> 64 -> 48 -> 32 -> 16 with an HGELU-clamp activation, about 8.2k multiply-adds per texel. A
viewer decoder would:

1. Parse the `.ntc` header and split out the latent tensors and the weights (GDeflate may have to be
   handled: `--gdeflate none` can disable it on the encoder side; in the matched runs of this
   repository it only touched BCn buffers).
2. Upload the latents as texture arrays (4-bit unpack in the shader) and the weights as a storage buffer.
3. Run the MLP. On WebGPU, `dot4I8Packed` (WGSL `packed_4x8_integer_dot_product`) gives the int8 dot
   products. On WebGL2 there is no integer dot product, so it would be a float emulation that costs
   about 4 times our v2 shader (2k MAC per texel).
4. Decode once to RGBA8 textures at load time (compute pass on WebGPU, render-to-texture on WebGL2).
   Run-time per-pixel decoding with stochastic texture filtering is possible but not needed for a
   comparison.

Bake-at-load-time through a WebGPU compute shader is the realistic first target; WebGL2 would use
a fragment-shader bake with float math.

## 4. Blockers

1. **Format stability.** The `.ntc` container and the latent layout are not publicly specified and have
   changed between betas (0.9.x -> 0.10.x). A codec id that includes the version (`nvidia-ntc-0.10`) is
   mandatory, and every SDK bump needs a new decoder.
2. **License.** The SDK ships under the "NVIDIA RTX SDKs LICENSE" and its sources are marked
   `LicenseRef-NvidiaProprietary`. Porting LibNTC's HLSL/Slang inference code into an Apache-2.0 loader
   needs an agreement with NVIDIA. A clean-room decoder written from the NTC paper cannot read NVIDIA's
   trained weights layout without reverse engineering the container, which is the same legal question.
3. **Precision.** The decoder is trained for FP8 or INT8 inference with specific rounding. Float
   emulation in WebGL2 must match INT8 rounding to avoid visible differences with the encoder-side PSNR.
4. **No spec for filtering.** NTC decodes one unfiltered texel per call; the SDK relies on stochastic
   texture filtering. A baked texture is fine, but a faithful run-time viewer would need STF and TAA.

## 5. Lessons from the option (b) integration

* Rate is discrete: the latent shape is `gridSizeScale` 1..6 and `numFeatures` 4..16 (multiples of 4),
  about `5 * nf / gs^2 + 0.04` bpp. Several shapes tie (gs1/nf4 = gs2/nf16 = 20 bpp), so the codec picks
  the coarser grid with more features. `-b <bpp>` in `ntc-cli` snaps down, so the codec selects the
  nearest shape itself and passes `--gridSizeScale/--numFeatures` (`--ntc-snap ntc` keeps the native choice).
* Every channel needs a semantic or it is dropped from the decoded PNG. Alpha uses `Transmission`
  (`Alpha`/`Mask` enable special mask handling). 1 and 2 channel textures are written as zero-padded
  RGB with a `channelSwizzle`, because 2-channel (LA) input PNGs gave a wrong second channel.
* Decoded PNGs can be gray, LA or palette-reduced; the reader loads them as RGBA8 and maps channels.
* An embedded payload would need the same texture-set/channel mapping, so the extension should record it.

## 6. Recommended path

1. Keep option (b) for the comparison (done).
2. Ask NVIDIA for permission and for a stable bitstream description before investing in (a).
3. If granted, implement the load-time bake in WebGPU only, behind `codec: "nvidia-ntc-<version>"`,
   with the fallback textures kept for other loaders.
4. If not granted, put effort in our own codec instead: its own v3 layout (16 feature channels at 4 bit,
   64-wide MLP) is architecturally close to NTC and is under our license.
