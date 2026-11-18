# gltf-texture-compressor

A standalone command-line tool that re-encodes the textures of a glTF or GLB asset as either:

- **KTX2** (Basis Universal, `KHR_texture_basisu`): ETC1S for color textures, UASTC + Zstandard for normal maps and other linear data, with full mip chains.
- **Neural textures** (`BABYLON_materials_neural_texture`, experimental): each material's base color, normal, occlusion, roughness and metallic textures are replaced by low-resolution latent images plus a tiny MLP trained on the GPU with CUDA. See [docs/BABYLON_materials_neural_texture.md](docs/BABYLON_materials_neural_texture.md).

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
| `--neural-fallback-size <n>` | Keep downscaled originals (max width `n`) in the replaced slots for viewers without the extension, instead of 1x1 placeholders. |

PNG output layout:

- `original/<image>.png` and `ktx/<image>.png`: each source image and its KTX2 version transcoded back to RGBA.
- `original/<material>_{baseColor,normal,occlusionRoughnessMetallic}.png` and `neural/...`: the neural network inputs and the decoded network outputs, at the source resolution. Normal maps have Z reconstructed from X/Y in both versions so they compare fairly.

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
| [FlightHelmet](https://playground.babylonjs.com/#BVK9I0#102) | 54,913,288 | 31,006,336 (56.5%) | 13,078,600 (23.8%) | 181.0 s | 89.2 s | 501 ms | 570 ms | 81 ms | 60 ms | 5 full PBR sets (base color, normal, occlusion, roughness/metallic) |
| [BoomBox](https://playground.babylonjs.com/#QCU8DJ#2) | 11,249,720 | 3,840,832 (34.1%) | 2,583,640 (23.0%) | 45.8 s | 18.8 s | 691 ms | 547 ms | 48 ms | 44 ms | 1 full PBR set |
| [aerobatic_plane](https://playground.babylonjs.com/#5NIBE1#207) | 4,616,664 | 4,310,560 (93.4%) | 4,192,848 (90.8%) | 26.8 s | 17.6 s | 169 ms | 95 ms | 94 ms | 55 ms | 1 partial set (normal + occlusion only) |
| [fish](https://playground.babylonjs.com/#9CMU0F#38) | 47,570,100 | 22,412,308 (47.1%) | 45,852,732 (96.4%) | 84.0 s | 71.2 s | 1093 ms | 645 ms | 423 ms | 204 ms | 4 partial sets (normal + occlusion only) |
| [haunted_house](https://playground.babylonjs.com/#E8UGIC#20) | 38,976,908 | 12,822,392 (32.9%) | 19,518,400 (50.1%) | 85.1 s | 39.9 s | 381 ms | 66 ms | 42 ms | 38 ms | 2 full PBR sets |

Load times are for the experimental Babylon.js loader for `BABYLON_materials_neural_texture` (a prototype, not part of this repository), measured in headless Chromium on the RTX 4060 (ANGLE/D3D11 for WebGL, native adapter for WebGPU). "Load-time decode" is the glTF load time when the neural textures are baked into regular mipmapped GPU textures by a fragment pass (WebGL) or compute shaders (WebGPU). "Shader-decode load" is the load time when decoding is left to the PBR fragment shader, so nothing is baked at load. These are wall-clock times with a warm shader cache; WebGPU completion fences include deferred driver work, not pure GPU time. Shader compilation adds to the time to first frame: scene-ready times, including Sandbox startup and PBR shader compilation, were about 2-4 s in every case, and rendering an unmodified asset took about 2-2.7 s.

Neural results depend on how many texture slots of a material the network replaces. All channels of a material share one latent set (about 2.13 MiB at 2048x2048), so a full PBR set is stored for the price of one latent set and the network can exploit the correlation between channels. That is the likely reason full sets compress well (not measured separately). Materials with only normal and occlusion textures (aerobatic_plane, fish) pay the same fixed latent cost for just two slots, and their base color images are left untouched, so the file barely shrinks.

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

`babylon-ntc-v1` is a C++/CUDA port of a PyTorch prototype. For each material it trains two 8-channel latent grids (at 1/4 and 1/16 of the texture resolution) and a 24 -> 32 -> 32 -> 9 ReLU MLP:

- Training samples continuous random UVs against bilinearly filtered targets, so the result is correct at every UV the GPU can sample, not only at texel centers.
- Uniform noise of half a quantization step is added to the latents during training; the latents are then quantized to 8 bits and the MLP is fine-tuned on the quantized values.
- Forward and backward passes are hand-written CUDA kernels (no deep learning framework), using Adam with a cosine learning rate schedule.

The decoder is cheap enough to run in a fragment shader (about 1.9k multiply-adds per pixel) or once at load time.

Codecs implement `INeuralCodec` (`src/neural/codec.h`), so other formats can be added. NVIDIA's [RTX Neural Texture Compression](https://github.com/NVIDIA-RTX/RTXNTC) SDK is a natural candidate: it has a more elaborate encoder and runtime decoders for DirectX 12 and Vulkan, but its bitstream is specific to its own runtime.

### Why not ONNX Runtime?

ONNX Runtime was evaluated for a Python-free encoder. It is an inference engine: on-device training needs training artifacts generated ahead of time with Python, and its training operator coverage is limited. It would help for decoding (inference) only, which is not the bottleneck. A small dedicated CUDA trainer is simpler, has no runtime dependency, and is fast.

## Limitations

- Neural encoding requires an NVIDIA GPU.
- Neural textures skip materials whose PBR textures use `KHR_texture_transform` or a `texCoord` other than 0.
- Assets using `EXT_meshopt_compression` are not supported. Draco-compressed meshes are preserved.
- No viewer supports `BABYLON_materials_neural_texture` yet; a Babylon.js glTF loader extension is the next step.
