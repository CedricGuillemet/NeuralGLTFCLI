# gltf-texture-compressor

A standalone command-line tool that re-encodes the textures of a glTF or GLB asset as either:

- **KTX2** (Basis Universal, `KHR_texture_basisu`): ETC1S for color textures, UASTC + Zstandard for normal maps and other linear data, with full mip chains.
- **Neural textures** (`BABYLON_materials_neural_texture`, experimental): each material's base color, normal, occlusion, roughness, metallic, emissive and spec-gloss (`KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness) textures are replaced by low-resolution latent images plus a tiny MLP trained on the GPU with CUDA. See [docs/BABYLON_materials_neural_texture.md](docs/BABYLON_materials_neural_texture.md).

The output is written as `<name>_ktx.glb` and/or `<name>_neural.glb`, and the tool prints the input and output sizes in bytes.

The executable has no Python or DLL dependencies: libktx is linked statically and the CUDA runtime is static too, so neural encoding only needs an NVIDIA driver. KTX encoding works on any machine.

## Usage

```
gltf-texture-compressor FlightHelmet.glb --format ktx,neural --png
```

| Option | Description |
|---|---|
| `-f, --format ktx\|neural\|ktx,neural` | Output formats (default `ktx`). |
| `-o, --output-dir <dir>` | Output directory (default: next to the input). |
| `--png [dir]` | Also write the re-compressed textures and the originals as PNG (default `<output-dir>/<name>_png`), for comparison with image diff tools. |
| `--etc1s-quality`, `--uastc-level`, `--uastc-rdo`, `--zstd`, `--uastc-color` | KTX2 encoder settings. |
| `--steps`, `--batch`, `--seed`, `--cuda-device` | Neural training settings (default 6000 steps of 131072 samples per material). |
| `--neural-codec babylon-ntc-v2\|babylon-ntc-v1` | Neural codec (default `babylon-ntc-v2`). v2 also encodes emissive and `KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness textures with a variable number of outputs. v1 writes the original format, core PBR slots only. |
| `--neural-specgloss encode\|ignore` | Whether v2 encodes the spec-gloss textures (default `encode`). `ignore` keeps them as regular textures. |
| `--neural-fallback-size <n>` | Keep downscaled originals (max width `n`) in the replaced slots for viewers without the extension, instead of 1x1 placeholders. |

PNG output layout:

- `original/<image>.png` and `ktx/<image>.png`: each source image and its KTX2 version transcoded back to RGBA.
- `original/<material>_{baseColor,normal,occlusionRoughnessMetallic,emissive,diffuse,specular,glossiness}.png` and `neural/...`: the neural network inputs and the decoded network outputs, at the source resolution (only for the groups that are encoded). Normal maps have Z reconstructed from X/Y in both versions so they compare fairly.

## Building

Requirements: CMake 3.24+, a C++20 compiler, the CUDA Toolkit 12 or newer, and Git (dependencies are fetched at configure time: [tinygltf](https://github.com/syoyo/tinygltf) and [KTX-Software](https://github.com/KhronosGroup/KTX-Software)).

On Windows, from a "x64 Native Tools" command prompt:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The default CUDA targets are Turing, Ampere, Ada and Blackwell (`75-real 86-real 89-real 120`); override with `-DCMAKE_CUDA_ARCHITECTURES=...`.

## Results

Sizes in bytes, with the default settings (6000 training steps for neural). Times are wall-clock encoding times: KTX2 on an AMD Ryzen 9 5900X (CPU), neural on an NVIDIA RTX 4060 (GPU).

All neural results were produced on an NVIDIA GeForce RTX 4060.

| Asset | Input | KTX2 | Neural | Encode time KTX2 | Encode time neural | Load-time decode WebGL | Load-time decode WebGPU | Shader-decode load WebGL | Shader-decode load WebGPU | Neural coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [FlightHelmet](https://playground.babylonjs.com/#BVK9I0#102) | 54,913,288 | 31,006,336 (56.5%) | 13,078,064 (23.8%) | 181.0 s | 110.3 s | TBD | TBD | TBD | TBD | 5 full PBR sets (base color, normal, occlusion, roughness/metallic) |
| [BoomBox](https://playground.babylonjs.com/#QCU8DJ#2) | 11,249,720 | 3,840,832 (34.1%) | 2,451,256 (21.8%) | 45.8 s | 22.4 s | TBD | TBD | TBD | TBD | 1 full PBR set plus emissive |
| [aerobatic_plane](https://playground.babylonjs.com/#5NIBE1#207) | 4,616,664 | 4,310,560 (93.4%) | 3,570,864 (77.3%) | 26.8 s | 22.3 s | TBD | TBD | TBD | TBD | 1 set: normal, occlusion, spec-gloss diffuse, specular, glossiness |
| [fish](https://playground.babylonjs.com/#9CMU0F#38) | 47,570,100 | 22,412,308 (47.1%) | 20,020,108 (42.1%) | 84.0 s | 103.3 s | TBD | TBD | TBD | TBD | 4 sets: normal, occlusion, spec-gloss diffuse, specular, glossiness, plus emissive on 2 materials |
| [haunted_house](https://playground.babylonjs.com/#E8UGIC#20) | 38,976,908 | 12,822,392 (32.9%) | 5,489,832 (14.1%) | 85.1 s | 121.5 s | TBD | TBD | TBD | TBD | 2 full sets: PBR core, emissive (1 material), spec-gloss diffuse, specular, glossiness. Encoded with `--steps 20000` |

Neural sizes and times are for the default `babylon-ntc-v2` codec (haunted_house: `--steps 20000`; the others use the default 6000 steps). Before `babylon-ntc-v2`, the neural files were 13,078,600 (FlightHelmet), 2,583,640 (BoomBox), 4,192,848 (aerobatic_plane), 45,852,732 (fish) and 19,518,400 (haunted_house) bytes, covering only the core PBR slots. With v2, neural is smaller than KTX2 on all five assets.

The TBD load-time columns are pending re-measurement with a loader that supports `babylon-ntc-v2`. The previous measurements (v1 files, Babylon.js prototype loader) were: FlightHelmet 501, 570, 81 and 60 ms; BoomBox 691, 547, 48 and 44 ms; aerobatic_plane 169, 95, 94 and 55 ms; fish 1093, 645, 423 and 204 ms; haunted_house 381, 66, 42 and 38 ms (load-time decode WebGL, load-time decode WebGPU, shader-decode load WebGL, shader-decode load WebGPU).

Load times above (when measured) are for the experimental Babylon.js loader for `BABYLON_materials_neural_texture` (a prototype, not part of this repository), measured in headless Chromium on the RTX 4060 (ANGLE/D3D11 for WebGL, native adapter for WebGPU). "Load-time decode" is the glTF load time when the neural textures are baked into regular mipmapped GPU textures by a fragment pass (WebGL) or compute shaders (WebGPU). "Shader-decode load" is the load time when decoding is left to the PBR fragment shader, so nothing is baked at load. These are wall-clock times with a warm shader cache; WebGPU completion fences include deferred driver work, not pure GPU time. Shader compilation adds to the time to first frame: scene-ready times, including Sandbox startup and PBR shader compilation, were about 2-4 s in every case, and rendering an unmodified asset took about 2-2.7 s.

Neural results depend on how many texture slots of a material the network replaces. All channels of a material share one latent set (about 2.13 MiB at 2048x2048), so a full set is stored for the price of one latent set and the network can exploit the correlation between channels. With v2, emissive and the `KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness textures are encoded too, so assets like fish, aerobatic_plane and haunted_house (which use spec-gloss textures) now replace most of their textures with one latent set per material instead of only normal and occlusion, which is why fish and haunted_house shrank substantially.

Per-group PSNR (dB) of the decoded neural textures against the sources, per material:

| Asset (material) | Outputs | Per-group PSNR |
|---|---:|---|
| FlightHelmet (5 materials) | 8-9 | base color 32.5-39.8, normal 27.1-58.2, occlusion 26.1-33.9, roughness/metallic 29.8-39.1 |
| BoomBox | 11 | base color 32.2, normal 38.5, occlusion 38.9, roughness/metallic 31.1, emissive 50.1 |
| aerobatic_plane | 10 | normal 42.4, occlusion 35.2, diffuse 36.4, specular 41.9, glossiness 44.5 |
| fish (4 materials) | 10-13 | normal 38.9-46.4, occlusion 41.4-49.7, emissive 43.6-49.9, diffuse 34.4-40.3, specular 45.3-53.5, glossiness 35.6-47.3 |
| haunted_house (material 1) | 20 | base color 33.8, normal 38.8, occlusion 45.4, roughness/metallic 31.6, emissive 56.1, diffuse 34.2, specular 38.8, glossiness 39.8 |
| haunted_house (material 2) | 17 | base color 31.1, normal 31.9, occlusion 45.0, roughness/metallic 34.7, diffuse 31.2, specular 49.9, glossiness 36.0 |

## Neural quality and size

The size of a neural material is fixed by the codec layout, not by the content: two 8-channel latent grids at 1/4 and 1/16 of the texture resolution, stored as raw 8-bit values. For a 2048x2048 material that is 512x512x8 + 128x128x8 bytes = 2.13 MiB, plus about 8 KiB of MLP weights, whatever the textures contain. Measured PSNR on the results above ranges from about 26 dB to 58 dB per channel, with most channels between 30 and 45 dB.

Settings available today:

| Option | Effect |
|---|---|
| `--steps` (default 6000), `--batch` (default 131072) | More training steps or a larger batch improve PSNR. The file size does not change. |
| `--seed` | Only changes run-to-run variation. |
| `--neural-fallback-size <n>` | Keeps downscaled originals in the replaced slots for viewers without the extension. This makes the file larger. |

The size/quality balance itself is set by compile-time constants in `src/neural/ntc_cuda.h`:

| Constant | Current value | Effect |
|---|---|---|
| `kHiDiv`, `kLoDiv` | 4, 16 | Latent grid resolution divisors. Larger values give smaller files and lower quality. |
| `kLatentChannels` | 8 | Channels per latent grid. Fewer channels give smaller files and lower quality. |
| `kHidden` | 32 | MLP width. The weights are only about 8 KiB, so this mostly affects decode cost, not file size. |

Changing these requires a rebuild, and files written with different values are not compatible with the current `babylon-ntc-v1` layout.

Possible improvements, not implemented yet:

- A `--quality` / `--latent-div` option exposing the latent resolution.
- Lower latent bit depth (for example 6 or 4 bits) and entropy coding (zstd or bit packing) of the latents.
- A `--latent-channels` option.
- Skipping materials that have too few PBR slots to amortize the fixed latent cost (see the coverage column above).

## Neural codec

`babylon-ntc-v1` and `babylon-ntc-v2` are a C++/CUDA port of a PyTorch prototype. For each material it trains two 8-channel latent grids (at 1/4 and 1/16 of the texture resolution) and a 24 -> 32 -> 32 -> N ReLU MLP (N = 9 for v1; N = 1 to 20 for v2, one output per encoded channel):

- Training samples continuous random UVs against bilinearly filtered targets, so the result is correct at every UV the GPU can sample, not only at texel centers.
- Uniform noise of half a quantization step is added to the latents during training; the latents are then quantized to 8 bits and the MLP is fine-tuned on the quantized values.
- Forward and backward passes are hand-written CUDA kernels (no deep learning framework), using Adam with a cosine learning rate schedule.

The decoder is cheap enough to run in a fragment shader (about 1.9k multiply-adds per pixel) or once at load time.

Codecs implement `INeuralCodec` (`src/neural/codec.h`), so other formats can be added. NVIDIA's [RTX Neural Texture Compression](https://github.com/NVIDIA-RTX/RTXNTC) SDK is a natural candidate: it has a more elaborate encoder and runtime decoders for DirectX 12 and Vulkan, but its bitstream is specific to its own runtime.

### Why not ONNX Runtime?

ONNX Runtime was evaluated for a Python-free encoder. It is an inference engine: on-device training needs training artifacts generated ahead of time with Python, and its training operator coverage is limited. It would help for decoding (inference) only, which is not the bottleneck. A small dedicated CUDA trainer is simpler, has no runtime dependency, and is fast.

## Limitations

- Neural encoding requires an NVIDIA GPU.
- Neural textures skip individual slots that use `KHR_texture_transform` or a `texCoord` other than 0.
- Assets using `EXT_meshopt_compression` are not supported. Draco-compressed meshes are preserved.
- `babylon-ntc-v2` files need a loader that supports that codec; loaders that only know `babylon-ntc-v1` ignore the extension and show the placeholder textures. Use `--neural-codec babylon-ntc-v1` for those loaders.
- No viewer in this repository supports `BABYLON_materials_neural_texture`; the Babylon.js glTF loader extension is being developed separately.
