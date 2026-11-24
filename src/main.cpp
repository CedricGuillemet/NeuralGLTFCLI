#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gltf_io.h"
#include "neural/ntc_v4.h"
#include "passes.h"

namespace fs = std::filesystem;
using namespace gtc;

namespace {

void printUsage() {
    std::printf(
        "gltf-texture-compressor %s\n"
        "Re-encodes the textures of a glTF/GLB asset as KTX2 (Basis Universal) or neural textures.\n\n"
        "Usage: gltf-texture-compressor <input.gltf|input.glb> [options]\n\n"
        "Options:\n"
        "  -f, --format <list>        ktx, neural or ktx,neural (default: ktx)\n"
        "  -o, --output-dir <dir>     output directory (default: next to the input)\n"
        "      --png [dir]            also write the re-compressed textures (and the originals) as PNG\n"
        "                             for quality comparison (default dir: <output-dir>/<name>_png)\n"
        "  -h, --help                 show this help\n"
        "      --version              print the version\n\n"
        "KTX2 options:\n"
        "      --etc1s-quality <n>    ETC1S quality for color textures, 1-255 (default 255)\n"
        "      --uastc-level <n>      UASTC level for non-color textures, 0-4 (default 2)\n"
        "      --uastc-rdo <x>        UASTC rate-distortion quality scalar, 0 disables (default 1.0)\n"
        "      --zstd <n>             Zstandard level for UASTC textures, 0 disables (default 18)\n"
        "      --uastc-color          use UASTC for color textures too (higher quality, larger)\n\n"
        "Neural options (the default babylon-ntc-v4 codec needs an NVIDIA GPU and a build with GTC_WITH_CUDA_CODEC=ON):\n"
        "      --neural-codec <name>  babylon-ntc-v4 (default; clean-room latent pyramid + 64-wide MLP, preset v4s), or the\n"
        "                             legacy CUDA codecs babylon-ntc-v2 (variable outputs), babylon-ntc-v1\n"
        "                             (core metal-rough textures only) and babylon-ntc-v3 (these need NTC_SHAPE_SET=all)\n"
        "      --neural-preset <p>    babylon-ntc-v4: v4xs, v4s (default), v4m, v4l; legacy codecs: v2best (default), v2, v3s,\n"
        "                             v3m, v3l, v3x; the flags below override a preset\n"
        "      --steps <n>            training steps per material (v4: phase A steps, default 40000, then max(n/20, 2000)\n"
        "                             MLP fine-tune steps on the frozen quantized latents)\n"
        "      --batch <n>            samples per step (v4 default 65536)\n"
        "      --neural-grids <list>  latent grids div:channels:bits,... (finest first); v4: two grids with div1 = 2*div0,\n"
        "                             e.g. 4:12:4,8:16:4 (bits 2 or 4, channels*bits a multiple of 8)\n"
        "      --neural-activation <hardgelu|hardswish|silu>   v4 experiments only: anything but hardgelu is not decodable by\n"
        "                             the format and skips the CPU reference check\n"
        "      --neural-out-act <none|sigmoid|group=act,...>   v4: output activation per channel group (base, normal, ...)\n"
        "      --neural-no-const      v4: do not elide constant channels\n"
        "      --neural-tune <k=v,...> v4 experiments: lrLatent, lrMlp, lrB, init, uniformMip, inside, noise, levelCap, stepsB\n"
        "      --ntc-suffix <s>       deprecated alias of --neural-suffix\n"
        "      --neural-fallback <ktx|png|none>\n"
        "                             what the replaced slots contain for loaders without the extension: png (default:\n"
        "                             1x1 placeholders, or downscaled PNGs with --neural-fallback-size), ktx (KTX2 via\n"
        "                             KHR_texture_basisu, full size unless --neural-fallback-size is set; the extension is\n"
        "                             listed in extensionsUsed only) or none (no texture in core slots)\n"
        "      --neural-fallback-size <n>\n"
        "                             max width of the fallback textures\n"
        "      --neural-suffix <s>    name the neural output <name>_neural_<s>.glb (PNGs go in <png dir>/neural_<s>/)\n"
        "                             and also write <name>_neural_<s>.metrics.json\n"
        "      --neural-materials <list>\n"
        "                             only encode the comma-separated material names (others are left untouched)\n"
        "      --seed <n>             random seed (default 1)\n"
        "      --cuda-device <n>      CUDA device index (default 0)\n\n"
        "Legacy babylon-ntc training options (--steps/--batch/--neural-grids/--neural-activation also apply to v4 as\n"
        "described above):\n"
        "      --neural-specgloss <encode|ignore>\n"
        "                             v2: encode KHR_materials_pbrSpecularGlossiness textures (default) or leave them\n"
        "      --neural-hidden <n>    MLP width; --neural-layers <n>: hidden layers\n"
        "      --neural-activation <relu|silu|hgelu|hardswish>   hardswish: z*clamp(z+3,0,6)/6 (hgelu is the older v3 shape)\n"
        "      --neural-uv-mask <on|off>   train on the texels covered by the mesh UVs\n"
        "      --neural-sampling <continuous|texel|mixed:p>   texel-center sample probability\n"
        "      --neural-importance <x>     0..1 gradient-energy sampling blend\n"
        "      --neural-loss <l2|charbonnier>\n"
        "      --neural-loss-weights <name=w,...>   base, alpha, normal, occlusion, roughness, metallic, ...\n"
        "      --neural-finetune-steps <n>, --neural-finetune-lr <lr[:endLr]>, --neural-warmup <fraction>,\n"
        "      --neural-beta2 <x>, --neural-quant <noise|ste>   optimizer and quantization\n\n"
        "Outputs are written as <name>_ktx.glb and <name>_neural.glb.\n",
        GTC_VERSION);
}

bool parseInt(const char* s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end) return false;
    out = int(v);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string input, outDir, pngDir, formats = "ktx";
    std::string neuralSuffix;
    bool png = false;
    KtxOptions ktxOpts;
    NeuralOptions nOpts;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s expects %s\n", a.c_str(), what);
                std::exit(2);
            }
            return argv[++i];
        };
        auto nextInt = [&](int& v) {
            if (!parseInt(next("an integer"), v)) {
                std::fprintf(stderr, "error: %s expects an integer\n", a.c_str());
                std::exit(2);
            }
        };
        if (a == "-h" || a == "--help") { printUsage(); return 0; }
        else if (a == "--version") { std::printf("%s\n", GTC_VERSION); return 0; }
        else if (a == "--neural-golden") {
            // Hidden: writes the synthetic babylon-ntc-v4 golden vectors into the given directory.
            const std::string dir = next("a directory");
            std::string err;
            if (!gtc::v4::writeSyntheticGoldens(dir, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
            return 0;
        }
        if (a == "-f" || a == "--format") formats = next("a format list");
        else if (a == "-o" || a == "--output-dir") outDir = next("a directory");
        else if (a == "--png") {
            png = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') pngDir = argv[++i];
        }
        else if (a == "--etc1s-quality") nextInt(ktxOpts.etc1sQuality);
        else if (a == "--uastc-level") nextInt(ktxOpts.uastcLevel);
        else if (a == "--uastc-rdo") ktxOpts.uastcRdo = float(std::atof(next("a number")));
        else if (a == "--zstd") nextInt(ktxOpts.zstdLevel);
        else if (a == "--uastc-color") ktxOpts.uastcForColor = true;
        else if (a == "--neural-codec") nOpts.codec = next("a codec name");
        else if (a == "--neural-specgloss") {
            nOpts.specGloss = next("encode or ignore");
            if (nOpts.specGloss != "encode" && nOpts.specGloss != "ignore") {
                std::fprintf(stderr, "error: --neural-specgloss expects encode or ignore\n");
                return 2;
            }
        }
        else if (a == "--steps") { nextInt(nOpts.steps); nOpts.stepsSet = true; }
        else if (a == "--batch") { nextInt(nOpts.batch); nOpts.batchSet = true; }
        else if (a == "--neural-preset") nOpts.preset = next("a preset name");
        else if (a == "--neural-grids") nOpts.grids = next("a grid list");
        else if (a == "--neural-hidden") nextInt(nOpts.hidden);
        else if (a == "--neural-layers") nextInt(nOpts.hiddenLayers);
        else if (a == "--neural-activation") nOpts.activation = next("hardgelu, hardswish, silu, relu or hgelu");
        else if (a == "--neural-out-act") nOpts.outAct = next("none, sigmoid or group=act pairs");
        else if (a == "--neural-tune") nOpts.tune = next("key=value pairs");
        else if (a == "--neural-no-const") nOpts.noConst = true;
        else if (a == "--neural-uv-mask") nOpts.uvMask = next("on or off");
        else if (a == "--neural-sampling") nOpts.sampling = next("continuous, texel or mixed:<p>");
        else if (a == "--neural-importance") nOpts.importance = float(std::atof(next("a number")));
        else if (a == "--neural-loss") nOpts.loss = next("l2 or charbonnier");
        else if (a == "--neural-loss-weights") nOpts.lossWeights = next("name=weight pairs");
        else if (a == "--neural-finetune-steps") nextInt(nOpts.finetuneSteps);
        else if (a == "--neural-finetune-lr") nOpts.finetuneLr = next("lr or lr:endLr");
        else if (a == "--neural-warmup") nOpts.warmup = float(std::atof(next("a fraction")));
        else if (a == "--neural-beta2") nOpts.beta2 = float(std::atof(next("a number")));
        else if (a == "--neural-quant") nOpts.quant = next("noise or ste");
        else if (a == "--seed") { int s; nextInt(s); nOpts.seed = uint32_t(s); }
        else if (a == "--cuda-device") nextInt(nOpts.cudaDevice);
        else if (a == "--neural-fallback-size") nextInt(nOpts.fallbackSize);
        else if (a == "--neural-fallback") {
            nOpts.fallback = next("ktx, png or none");
            if (nOpts.fallback != "ktx" && nOpts.fallback != "png" && nOpts.fallback != "none") {
                std::fprintf(stderr, "error: --neural-fallback expects ktx, png or none\n");
                return 2;
            }
        }
        else if (a == "--neural-suffix") neuralSuffix = next("a suffix");
        else if (a == "--ntc-suffix") neuralSuffix = next("a suffix");
        else if (a == "--neural-materials") {
            std::string list = next("a comma-separated material list");
            for (size_t s = 0; s <= list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                if (e > s) nOpts.materials.push_back(list.substr(s, e - s));
                s = e + 1;
            }
        }
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "error: unknown option %s (see --help)\n", a.c_str()); return 2; }
        else if (input.empty()) input = a;
        else { std::fprintf(stderr, "error: unexpected argument %s\n", a.c_str()); return 2; }
    }
    if (input.empty()) { printUsage(); return 2; }

    bool doKtx = false, doNeural = false;
    for (size_t s = 0; s <= formats.size();) {
        size_t e = formats.find(',', s);
        if (e == std::string::npos) e = formats.size();
        std::string f = formats.substr(s, e - s);
        if (f == "ktx" || f == "ktx2") doKtx = true;
        else if (f == "neural") doNeural = true;
        else { std::fprintf(stderr, "error: unknown format '%s' (expected ktx or neural)\n", f.c_str()); return 2; }
        s = e + 1;
    }
    nOpts.ktxFallback = ktxOpts;
    if (nOpts.steps < 1 || nOpts.batch < 64) { std::fprintf(stderr, "error: --steps must be >= 1 and --batch >= 64\n"); return 2; }

    const fs::path inPath = toPath(input);
    const std::string stem = fromPath(inPath.stem());
    const fs::path out = outDir.empty() ? inPath.parent_path() : toPath(outDir);
    if (!out.empty()) fs::create_directories(out);
    if (png && pngDir.empty()) pngDir = fromPath(out / (stem + "_png"));

    auto t0 = std::chrono::steady_clock::now();
    std::printf("Loading %s\n", input.c_str());
    Document doc;
    LoadResult info;
    std::string err;
    if (!loadDocument(input, doc, info, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::printf("  %zu materials, %zu textures, %zu images, input size %llu bytes (%s)\n", doc.model.materials.size(),
                doc.model.textures.size(), doc.model.images.size(), (unsigned long long)info.inputBytes,
                humanBytes(info.inputBytes).c_str());

    struct Report { std::string format, path; uint64_t bytes; double seconds; };
    std::vector<Report> reports;
    auto run = [&](const std::string& format, auto&& pass) -> bool {
        std::printf("\n== %s ==\n", format.c_str());
        auto ts = std::chrono::steady_clock::now();
        Document copy = doc;
        const std::string png = pngDir.empty() ? std::string() : pngDir;
        if (!pass(copy, png)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return false; }
        const fs::path file = out / (stem + "_" + format + ".glb");
        std::printf("  writing %s\n", fromPath(file).c_str());
        if (!saveGlb(std::move(copy), fromPath(file), err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return false; }
        const uint64_t bytes = fs::file_size(file);
        reports.push_back({format, fromPath(file), bytes, secondsSince(ts)});
        std::printf("  output size %llu bytes (%s), %.1f s\n", (unsigned long long)bytes, humanBytes(bytes).c_str(), secondsSince(ts));
        return true;
    };

    if (doKtx && !run("ktx", [&](Document& d, const std::string& p) { return runKtxPass(d, ktxOpts, p, err); })) return 1;
    if (doNeural) {
        nOpts.tag = neuralSuffix.empty() ? "neural" : "neural_" + neuralSuffix;
        if (!neuralSuffix.empty()) nOpts.metricsPath = fromPath(out / (stem + "_" + nOpts.tag + ".metrics.json"));
        std::string cl;
        for (int i = 1; i < argc; ++i) cl += (i > 1 ? " " : "") + std::string(argv[i]);
        nOpts.commandLine = cl;
    }
    if (doNeural && !run(nOpts.tag, [&](Document& d, const std::string& p) { return runNeuralPass(d, nOpts, p, err); })) return 1;

    std::printf("\nSummary\n  input   %12llu bytes  %s\n", (unsigned long long)info.inputBytes, input.c_str());
    for (auto& r : reports)
        std::printf("  %-7s %12llu bytes  %s  (%.1f%% of input, %.1f s)\n", r.format.c_str(), (unsigned long long)r.bytes,
                    r.path.c_str(), 100.0 * double(r.bytes) / double(info.inputBytes), r.seconds);
    if (png) std::printf("  PNG textures in %s\n", pngDir.c_str());
    std::printf("Done in %.1f s\n", secondsSince(t0));
    return 0;
}
