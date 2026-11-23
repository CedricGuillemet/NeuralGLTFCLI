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
| `--steps`, `--batch`, `--seed`, `--cuda-device` | Neural training settings (batch 131072 samples per material; default steps come from the preset: 24000 for `v2best` and the v3 presets, 6000 for legacy `v2`). |
| `--neural-preset v2best\|v2\|v3s\|v3m\|v3l\|v3x` | Quality preset (default `v2best`). `v2` is the original 6000-step configuration. `v2best` keeps the `babylon-ntc-v2` file format and size but trains with the UV-coverage mask, texel-centre sampling and 24000 steps. `v3s`/`v3m`/`v3l`/`v3x` write `babylon-ntc-v3` (4-bit latents, 64-wide SiLU network, optional third grid): increasing quality and size. Explicit flags below override the preset. |
| `--neural-grids div:ch:bits,...` | Custom latent grids from the finest to the coarsest, for example `4:16:4,16:16:4` (resolution divisor, channels, bits 4 or 8). A network that does not fit `babylon-ntc-v2` is written as `babylon-ntc-v3`. |
| `--neural-hidden <n>`, `--neural-layers <n>`, `--neural-activation relu\|silu\|hgelu` | Hidden width (32, 48 or 64), number of hidden layers (2 or 3) and activation of the decoder MLP. |
| `--neural-uv-mask on\|off` | Rasterise the UV islands of the meshes using the material and train/evaluate only inside them (plus a small weight outside, so the padding stays plausible). Also reports masked PSNR. |
| `--neural-sampling continuous\|texel\|mixed:p` | Training sample positions: continuous UVs, texel centres (what the decoder sees at load time) or a mix. |
| `--neural-importance <x>` | Weight sampling towards high-gradient regions (0 = uniform). |
| `--neural-loss l2\|charbonnier`, `--neural-loss-weights name=w,...` | Training loss and per-group weights (`base`, `alpha`, `normal`, `occlusion`, `roughness`, `metallic`, `emissive`, `diffuse`, `specular`, `glossiness`). |
| `--neural-finetune-steps`, `--neural-finetune-lr lr[:end]`, `--neural-warmup`, `--neural-beta2`, `--neural-quant noise\|ste` | Optimiser schedule and quantisation-aware training options (experimental; no measurable gain on the reference, see the experiment log). |
| `--neural-codec babylon-ntc-v2\|babylon-ntc-v1` | Neural codec (default `babylon-ntc-v2`). v2 also encodes emissive and `KHR_materials_pbrSpecularGlossiness` diffuse and specular-glossiness textures with a variable number of outputs. v1 writes the original format, core PBR slots only. |
| `--neural-specgloss encode\|ignore` | Whether v2 encodes the spec-gloss textures (default `encode`). `ignore` keeps them as regular textures. |
| `--neural-fallback-size <n>` | Keep downscaled originals (max width `n`) in the replaced slots for viewers without the extension, instead of 1x1 placeholders. |
| `--neural-suffix <s>` | Name the neural output `<name>_neural_<s>.glb`, put the `--png` decoded textures in `neural_<s>/`, and write a `<name>_neural_<s>.metrics.json` sidecar (sizes, bits per pixel, per-group PSNR, encode time, parameters, seed, git SHA). |
| `--neural-bake` | Also write `<name>_neural[_<s>]_baked.glb`: the decoded textures stored as plain PNG textures in a standard glTF (no extension needed), equivalent to the load-time bake. |
| `--neural-materials <a,b>` | Only encode the listed materials; the others keep their original textures. Useful for fast sweeps. |

PNG output layout:

- `original/<image>.png` and `ktx/<image>.png`: each source image and its KTX2 version transcoded back to RGBA.
- `original/<material>_{baseColor,normal,occlusionRoughnessMetallic,emissive,diffuse,specular,glossiness}.png` and `neural/...` (or `neural_<s>/...`): the neural network inputs and the decoded network outputs, at the source resolution (only for the groups that are encoded). Normal maps have Z reconstructed from X/Y in both versions so they compare fairly.

`tools/quality/` has the evaluation scripts for these outputs: `eval.py` (PSNR, SSIM and high-frequency PSNR per material and channel group, error heatmaps, fixed crops; needs numpy and Pillow), `capture.mjs` (playground screenshots and MAE through Playwright) and `montage.py`.

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
| [FlightHelmet](https://playground.babylonjs.com/#BVK9I0#102) | 54,913,288 | 31,006,336 (56.5%) | 13,078,064 (23.8%) | 181.0 s | 110.3 s | 192 ms | 113 ms | 81 ms | 72 ms | 5 full PBR sets (base color, normal, occlusion, roughness/metallic) |
| [BoomBox](https://playground.babylonjs.com/#QCU8DJ#2) | 11,249,720 | 3,840,832 (34.1%) | 2,451,256 (21.8%) | 45.8 s | 22.4 s | 71 ms | 70 ms | 22 ms | 18 ms | 1 full PBR set plus emissive |
| [aerobatic_plane](https://playground.babylonjs.com/#5NIBE1#207) | 4,616,664 | 4,310,560 (93.4%) | 3,570,864 (77.3%) | 26.8 s | 22.3 s | 84 ms | 85 ms | 33 ms | 28 ms | 1 set: normal, occlusion, spec-gloss diffuse, specular, glossiness |
| [fish](https://playground.babylonjs.com/#9CMU0F#38) | 47,570,100 | 22,412,308 (47.1%) | 20,020,108 (42.1%) | 84.0 s | 103.3 s | 284 ms | 172 ms | 121 ms | 114 ms | 4 sets: normal, occlusion, spec-gloss diffuse, specular, glossiness, plus emissive on 2 materials |
| [haunted_house](https://playground.babylonjs.com/#E8UGIC#20) | 38,976,908 | 12,822,392 (32.9%) | 5,489,832 (14.1%) | 85.1 s | 121.5 s | 128 ms | 59 ms | 42 ms | 32 ms | 2 full sets: PBR core, emissive (1 material), spec-gloss diffuse, specular, glossiness. Encoded with `--steps 20000` |

Neural sizes and times are for the default `babylon-ntc-v2` codec (haunted_house: `--steps 20000`; the others use the default 6000 steps). Before `babylon-ntc-v2`, the neural files were 13,078,600 (FlightHelmet), 2,583,640 (BoomBox), 4,192,848 (aerobatic_plane), 45,852,732 (fish) and 19,518,400 (haunted_house) bytes, covering only the core PBR slots. With v2, neural is smaller than KTX2 on all five assets.

Load times are the warm median of 5 runs of `LoadAssetContainerAsync` with the `babylon-ntc-v2` loader from [BabylonJS/Babylon.js#19187](https://github.com/BabylonJS/Babylon.js/pull/19187); cold first runs took 330-830 ms in load-time decode mode. The previous measurements (v1 files, older prototype loader) were: FlightHelmet 501, 570, 81 and 60 ms; BoomBox 691, 547, 48 and 44 ms; aerobatic_plane 169, 95, 94 and 55 ms; fish 1093, 645, 423 and 204 ms; haunted_house 381, 66, 42 and 38 ms (load-time decode WebGL, load-time decode WebGPU, shader-decode load WebGL, shader-decode load WebGPU).

Load times above are for the experimental Babylon.js loader for `BABYLON_materials_neural_texture` (not part of this repository), measured in headless Chromium on the RTX 4060 (ANGLE/D3D11 for WebGL, native adapter for WebGPU). "Load-time decode" is the glTF load time when the neural textures are baked into regular mipmapped GPU textures by a fragment pass (WebGL) or compute shaders (WebGPU). "Shader-decode load" is the load time when decoding is left to the PBR fragment shader, so nothing is baked at load. These are wall-clock times with a warm shader cache; WebGPU completion fences include deferred driver work, not pure GPU time. Shader compilation adds to the time to first frame: scene-ready times, including Sandbox startup and PBR shader compilation, were about 2-4 s in every case, and rendering an unmodified asset took about 2-2.7 s.

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

The size of a neural material is fixed by the preset, not by the content. The default (`v2best`) keeps two 8-channel 8-bit latent grids at 1/4 and 1/16 of the texture resolution: for a 2048x2048 material that is 512x512x8 + 128x128x8 bytes = 2.13 MiB, plus about 8 KiB of MLP weights. Measured PSNR on the results above ranges from about 26 dB to 58 dB per channel, with most channels between 30 and 45 dB.

Quality presets (`--neural-preset`), measured on FlightHelmet `LeatherPartsMat` (2048x2048, UV-masked PSNR, 24000 steps, seed 1; see `tools/quality/experiment.py` and `output/quality/experiments.csv`):

| Preset | Latents per material | Base / normal PSNR (masked) | Training time (RTX 4060) |
|---|---|---|---|
| `v2` (legacy, 6000 steps) | 2.13 MiB, 8-bit | 33.3 / 27.1 dB | 26 s |
| `v2best` (default) | 2.13 MiB, 8-bit | 33.2 / 29.3 dB (+3.2 dB normal vs `v2` on the same mask) | 95 s |
| `v3s` | 2.1 MiB, 4-bit | 34.8 / 31.7 dB | 6 min |
| `v3m` | 3.1 MiB | 36.1 / 33.4 dB | 6.5 min |
| `v3l` | 6.1 MiB | 42.7 / 37.1 dB | 7.5 min |
| `v3x` | 10.1 MiB | 47.5 / 45.8 dB | 22 min |

`v2best` gains come from training only on texels covered by the mesh UVs (`--neural-uv-mask`), sampling texel centres (`--neural-sampling texel`) and 4x more steps. STE quantisation, warmup, fine-tune LR schedules, Charbonnier loss, importance sampling and loss re-weighting showed no measurable gain. `v3*` files need a reader that supports `babylon-ntc-v3`.

Other settings:

| Option | Effect |
|---|---|
| `--steps`, `--batch` | More training steps or a larger batch improve PSNR. The file size does not change. |
| `--seed` | Only changes run-to-run variation. |
| `--neural-fallback-size <n>` | Keeps downscaled originals in the replaced slots for viewers without the extension. This makes the file larger. |
| `--neural-grids`, `--neural-hidden`, `--neural-layers` | Custom latent grids and MLP shape (see the options table). |

Possible improvements, not implemented yet:

- Entropy coding (zstd) of the latents.
- Gradient loss, PCA initialisation of the latents and packing the four textures of a material into one image.
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
