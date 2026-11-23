// "nvidia-ntc": default neural codec. It does not train anything itself: each material is written as PNG
// images plus a manifest, NVIDIA's ntc-cli (RTXNTC SDK, not part of this repository) compresses it to a
// .ntc file and decompresses it again, and the decoded PNGs are read back. The .ntc container is validated
// and returned unchanged as the extension's bitstream. Nothing from the SDK is linked or redistributed.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../passes.h"
#include "codec.h"
#include "json.hpp"
#include "neural/ntc_container.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

namespace gtc {

namespace {

using tinygltf::Value;

constexpr int kNtcMaxChannels = 16;

// Semantic label of each canonical channel (see the manifest docs of the SDK). The labels are metadata, except that ntc-cli
// only saves the channels covered by a label, so alpha needs one: "Transmission" (opacity) has no special processing,
// unlike Alpha/Mask/AlphaMask which switch on the SDK's mask handling that the other codec has no equivalent of.
const char* const kSemantic[kMaxNeuralChannels] = {"Albedo", "Albedo", "Albedo", "Transmission", "Normal", "Normal", "Occlusion",
                                                   "Roughness", "Metalness", "Emissive", "Emissive", "Emissive", "Albedo", "Albedo",
                                                   "Albedo", "Transmission", "SpecularColor", "SpecularColor", "SpecularColor", "Glossiness"};
// Channels that never share a texture: one NTC texture per output group (names match the extension's outputs).
const int kGroupOf[kMaxNeuralChannels] = {0, 0, 0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8};
const char* const kGroupName[] = {"baseColor", "normal", "occlusion", "roughness", "metallic", "emissive", "diffuse", "specular", "glossiness"};

struct TextureGroup {
    std::string name;
    std::vector<int> outputs; // output indices (into the material's N channels), in channel order
};

struct Shape {
    int gridSizeScale;
    int numFeatures;
    double bpp;
};

// Latent bits per pixel of a shape: two 4-bit feature grids at 1/g^2 and 1/(4 g^2) of the pixel count, plus about
// 0.04 bpp of weights and headers (measured on 2048^2 materials: 3.75 -> 3.79, 4.44 -> 4.48, 20 -> 20.0).
double estimateBpp(int gridSizeScale, int numFeatures) { return numFeatures * 5.0 / (double(gridSizeScale) * gridSizeScale) + 0.04; }

Shape nearestShape(double bpp) {
    Shape best{1, 4, estimateBpp(1, 4)};
    double bestErr = 1e30;
    // Several shapes can have the same bpp (gs1 nf4 = gs2 nf16 = 20): prefer the coarser grid with more features, which is
    // what ntc-cli picks for -b 20 and which gives the better quality.
    for (int g = 6; g >= 1; --g)
        for (int f = 4; f <= 16; f += 4) {
            const double e = std::fabs(std::log(estimateBpp(g, f) / bpp));
            if (e < bestErr - 1e-9) {
                bestErr = e;
                best = {g, f, estimateBpp(g, f)};
            }
        }
    return best;
}

std::vector<std::string> splitArgs(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool inQuote = false, any = false;
    for (char c : s) {
        if (c == '"') {
            inQuote = !inQuote;
            any = true;
        } else if (std::isspace((unsigned char)c) && !inQuote) {
            if (any || !cur.empty()) out.push_back(cur);
            cur.clear();
            any = false;
        } else {
            cur += c;
        }
    }
    if (any || !cur.empty()) out.push_back(cur);
    return out;
}

bool hasArg(const std::vector<std::string>& args, std::initializer_list<const char*> names) {
    for (const std::string& a : args)
        for (const char* n : names)
            if (a == n || a.rfind(std::string(n) + "=", 0) == 0) return true;
    return false;
}

// Looks for an executable in PATH.
std::string findOnPath(const std::string& name) {
    const char* path = std::getenv("PATH");
    if (!path) return {};
#ifdef _WIN32
    const char sep = ';';
    const std::vector<std::string> names = {name + ".exe", name};
#else
    const char sep = ':';
    const std::vector<std::string> names = {name};
#endif
    std::stringstream ss(path);
    std::string dir;
    while (std::getline(ss, dir, sep)) {
        if (dir.empty()) continue;
        for (const std::string& n : names) {
            std::error_code ec;
            const fs::path p = toPath(dir) / n;
            if (fs::is_regular_file(p, ec)) return fromPath(p);
        }
    }
    return {};
}

#ifdef _WIN32
std::wstring quoteWide(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : a) {
        if (c == L'\\') {
            ++backslashes;
        } else if (c == L'"') {
            q.append(backslashes * 2 + 1, L'\\');
            q += c;
            backslashes = 0;
        } else {
            q.append(backslashes, L'\\');
            q += c;
            backslashes = 0;
        }
    }
    q.append(backslashes * 2, L'\\');
    return q + L"\"";
}
#else
std::string quoteShell(const std::string& a) {
    std::string q = "'";
    for (char c : a) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
}
#endif

// Runs a process with stdout and stderr merged; every output line (split on \n or \r) goes to onLine.
bool runProcess(const std::string& exe, const std::vector<std::string>& args, const std::string& cwd, const std::function<void(const std::string&)>& onLine,
                int& exitCode, std::string& err) {
    std::string pending;
    auto feed = [&](const char* data, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (data[i] == '\n' || data[i] == '\r') {
                if (!pending.empty()) onLine(pending);
                pending.clear();
            } else {
                pending += data[i];
            }
        }
    };
#ifdef _WIN32
    std::wstring cmd = quoteWide(toPath(exe).wstring());
    for (const std::string& a : args) cmd += L" " + quoteWide(toPath(a).wstring());
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) {
        err = "CreatePipe failed";
        return false;
    }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    const std::wstring wcwd = cwd.empty() ? std::wstring() : toPath(cwd).wstring();
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, wcwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        err = "cannot start " + exe + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof(buf), &got, nullptr) && got) feed(buf, got);
    if (!pending.empty()) onLine(pending);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    exitCode = int(code);
    return true;
#else
    std::string cmd;
    if (!cwd.empty()) cmd += "cd " + quoteShell(cwd) + " && ";
    cmd += quoteShell(exe);
    for (const std::string& a : args) cmd += " " + quoteShell(a);
    cmd += " 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) {
        err = "cannot start " + exe;
        return false;
    }
    char buf[4096];
    size_t got;
    while ((got = fread(buf, 1, sizeof(buf), p)) > 0) feed(buf, got);
    if (!pending.empty()) onLine(pending);
    exitCode = pclose(p);
    return true;
#endif
}

class NvidiaNtc : public INeuralCodec {
public:
    explicit NvidiaNtc(const NeuralOptions& o) : opts_(o) { cli_ = o.ntcCli; }

    bool init(std::string& err) {
        if (cli_.empty()) {
            const char* env = std::getenv("NTC_CLI");
            if (env) cli_ = env;
        }
        if (cli_.empty()) cli_ = findOnPath("ntc-cli");
        if (cli_.empty()) {
            err = "nvidia-ntc: ntc-cli not found. Use --ntc-cli <path>, set the NTC_CLI environment variable, or put ntc-cli on PATH "
                  "(download the RTXNTC SDK from https://github.com/NVIDIA-RTX/RTXNTC/releases)";
            return false;
        }
        if (!fs::exists(toPath(cli_))) {
            err = "nvidia-ntc: ntc-cli not found at '" + cli_ + "'";
            return false;
        }
        cli_ = fromPath(fs::absolute(toPath(cli_)));
        int code = 0;
        std::string text;
        if (!runProcess(cli_, {"--version"}, "", [&](const std::string& l) { text += l + "; "; }, code, err) || code != 0) {
            if (err.empty()) err = "nvidia-ntc: '" + cli_ + " --version' failed: " + text;
            return false;
        }
        version_ = text;
        std::smatch vm;
        static const std::regex reVersion(R"(LibNTC version: (\d+)\.(\d+)\.(\d+))");
        if (!std::regex_search(text, vm, reVersion)) {
            err = "nvidia-ntc: cannot read the LibNTC version from '" + cli_ + " --version': " + text;
            return false;
        }
        libVersion_ = vm[1].str() + "." + vm[2].str() + "." + vm[3].str();
        if (vm[1].str() != "0" || vm[2].str() != "10") {
            err = "nvidia-ntc: ntc-cli reports LibNTC " + libVersion_ + " but this codec id (nvidia-ntc-0.10) is defined for the 0.10.x SDK "
                  "(download it from https://github.com/NVIDIA-RTX/RTXNTC/releases)";
            return false;
        }
        if (opts_.ntcBpp <= 0 && opts_.ntcMatch.empty() && !hasArg(splitArgs(opts_.ntcArgs), {"-b", "--bitsPerPixel", "-p", "--targetPsnr", "--gridSizeScale"})) {
            err = "nvidia-ntc: no bit rate given (use --ntc-bpp <x>, --ntc-match <metrics.json> or --ntc-args \"--targetPsnr <dB>\")";
            return false;
        }
        if (!opts_.ntcMatch.empty()) {
            try {
                std::ifstream f(toPath(opts_.ntcMatch));
                if (!f) throw std::runtime_error("cannot open file");
                nlohmann::json j = nlohmann::json::parse(f);
                for (const auto& m : j.at("materials")) {
                    double bpp = 0;
                    if (m.contains("compressedBytes") && m.contains("width") && m.contains("height"))
                        bpp = 8.0 * m["compressedBytes"].get<double>() / (m["width"].get<double>() * m["height"].get<double>());
                    else if (m.contains("bpp"))
                        bpp = m["bpp"].get<double>();
                    if (bpp > 0) matchBpp_[m.at("name").get<std::string>()] = bpp;
                }
            } catch (const std::exception& e) {
                err = "nvidia-ntc: cannot read --ntc-match file '" + opts_.ntcMatch + "': " + e.what();
                return false;
            }
            if (matchBpp_.empty()) {
                err = "nvidia-ntc: --ntc-match file '" + opts_.ntcMatch + "' has no materials with a size";
                return false;
            }
        }
        return true;
    }

    std::string name() const override { return "nvidia-ntc"; }
    std::string codecId() const override { return "nvidia-ntc-0.10"; }
    int maxOutputChannels() const override { return kNtcMaxChannels; }
    std::string deviceDescription() const override { return "external ntc-cli (" + version_ + ") on CUDA device " + std::to_string(opts_.cudaDevice); }
    bool usesWorkDir() const override { return true; }
    size_t expectedWeightCount(int) const override { return 0; }

    tinygltf::Value::Object describeParameters() const override {
        Value::Object p;
        p["ntcCli"] = Value(cli_);
        p["ntcVersion"] = Value(version_);
        p["bpp"] = Value(opts_.ntcBpp);
        p["match"] = Value(opts_.ntcMatch);
        p["snap"] = Value(opts_.ntcSnap);
        p["mips"] = Value(opts_.ntcMips);
        p["extraArgs"] = Value(opts_.ntcArgs);
        p["seed"] = Value(int(opts_.seed));
        p["isSRGB"] = Value(false);
        return p;
    }

    bool encode(const NeuralMaterialInput& in, NeuralMaterialOutput& out, std::string& err) override {
        const int N = in.outputs, w = in.width, h = in.height;
        const size_t texels = size_t(w) * h;
        if (N > kNtcMaxChannels) {
            err = "nvidia-ntc supports at most " + std::to_string(kNtcMaxChannels) + " channels per material, this one has " + std::to_string(N);
            return false;
        }
        if (in.workDir.empty() || in.target.size() != texels * N || in.layout.count != N) {
            err = "nvidia-ntc: invalid material input";
            return false;
        }
        const fs::path work = toPath(in.workDir);
        const fs::path imageDir = work / "input", decodedDir = work / "decoded";
        fs::create_directories(imageDir);
        fs::create_directories(decodedDir);

        // One image per run of consecutive channels of the same group.
        std::vector<TextureGroup> groups;
        for (int i = 0; i < N; ++i) {
            const int c = in.layout.canonical[i];
            const bool extend = !groups.empty() && kGroupOf[in.layout.canonical[groups.back().outputs.back()]] == kGroupOf[c] &&
                                in.layout.canonical[groups.back().outputs.back()] + 1 == c;
            if (!extend) groups.push_back({kGroupName[kGroupOf[c]], {}});
            groups.back().outputs.push_back(i);
        }

        // The target is the 8-bit data the other codec sees, so the values are written back exactly.
        Value::Array manifestTextures;
        std::string manifest = "{\n  \"textures\": [";
        for (size_t g = 0; g < groups.size(); ++g) {
            const TextureGroup& grp = groups[g];
            const int k = int(grp.outputs.size());
            // 1 and 2 channel PNGs are written as RGB and selected with channelSwizzle: ntc-cli's reader returns a wrong
            // second channel for LA files (measured: 20 dB instead of 32 dB on a normal map).
            const int kp = std::max(k, 3);
            std::vector<uint8_t> px(texels * kp, 0);
            for (size_t t = 0; t < texels; ++t)
                for (int c = 0; c < k; ++c)
                    px[t * kp + c] = uint8_t(std::lround(std::clamp(in.target[t * N + grp.outputs[c]], 0.0f, 1.0f) * 255.0f));
            const std::string file = grp.name + ".png";
            if (!stbi_write_png(fromPath(imageDir / file).c_str(), w, h, kp, px.data(), w * kp)) {
                err = "cannot write " + fromPath(imageDir / file);
                return false;
            }
            std::string scales, semantics;
            for (int c = 0; c < kp; ++c) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%s%g", c ? ", " : "", c < k ? double(in.lossWeights[grp.outputs[c]]) : 0.0);
                scales += buf;
            }
            for (int c = 0; c < k;) {
                const char* sem = kSemantic[in.layout.canonical[grp.outputs[c]]];
                int e = c + 1;
                while (e < k && std::string(kSemantic[in.layout.canonical[grp.outputs[e]]]) == sem) ++e;
                if (*sem) semantics += std::string(semantics.empty() ? "" : ", ") + "\"" + sem + "\": \"" + std::string("RGBA").substr(c, e - c) + "\"";
                c = e;
            }
            manifest += std::string(g ? ",\n" : "\n") + "    { \"fileName\": \"input/" + file + "\", \"name\": \"" + grp.name +
                        "\", \"isSRGB\": false, " + (k < 3 ? "\"channelSwizzle\": \"" + std::string("RGB").substr(0, k) + "\", " : std::string()) +
                        "\"lossFunctionScale\": [" + scales + "], \"semantics\": {" + semantics + "} }";
        }
        manifest += "\n  ]\n}\n";
        {
            std::ofstream f(work / "manifest.json", std::ios::binary);
            f << manifest;
            if (!f) {
                err = "cannot write " + fromPath(work / "manifest.json");
                return false;
            }
        }

        // Rate: the target bpp of this material, mapped to a latent shape.
        std::vector<std::string> extra = splitArgs(opts_.ntcArgs);
        double bpp = opts_.ntcBpp;
        auto it = matchBpp_.find(in.name);
        if (it != matchBpp_.end()) bpp = it->second;
        else if (!matchBpp_.empty() && bpp <= 0) {
            err = "nvidia-ntc: material '" + in.name + "' is not in the --ntc-match file";
            return false;
        }
        const bool userRate = hasArg(extra, {"-b", "--bitsPerPixel", "-p", "--targetPsnr", "--gridSizeScale", "--matchBcPsnr"});
        std::vector<std::string> args = {"--loadManifest", fromPath(work / "manifest.json"), "-c", "-D", "-o", fromPath(work / "material.ntc"), "-i",
                                         fromPath(decodedDir), "-F", "png", "-B", "none", "--gdeflate", "none", "--cudaDevice", std::to_string(opts_.cudaDevice),
                                         "--randomSeed", std::to_string(opts_.seed ? opts_.seed : 1)};
        if (opts_.ntcMips) args.push_back("-g");
        Shape shape{0, 0, 0};
        char num[32];
        if (!userRate) {
            if (opts_.ntcSnap == "nearest") {
                shape = nearestShape(bpp);
                args.insert(args.end(), {"--gridSizeScale", std::to_string(shape.gridSizeScale), "--numFeatures", std::to_string(shape.numFeatures)});
            } else {
                std::snprintf(num, sizeof(num), "%.4f", bpp);
                args.insert(args.end(), {"-b", num});
            }
        }
        args.insert(args.end(), extra.begin(), extra.end());

        std::string cmdline = fromPath(toPath(cli_).filename());
        for (const std::string& a : args) cmdline += " " + a;
        std::printf("    %s: %s\n", in.name.c_str(), cmdline.c_str());

        struct Parsed {
            double psnrInt8 = 0, psnrFp8 = 0;
            double fileBpp = 0;
            size_t fileBytes = 0;
            int gridSizeScale = 0, numFeatures = 0, steps = 0;
            std::vector<std::pair<std::string, double>> perTexture;
        } parsed;
        parsed.gridSizeScale = shape.gridSizeScale;
        parsed.numFeatures = shape.numFeatures;
        std::ofstream log(work / "ntc-cli.log", std::ios::binary);
        log << cmdline << "\n";
        static const std::regex reShape(R"(--gridSizeScale (\d+) --numFeatures (\d+))");
        static const std::regex reOverall(R"(Overall PSNR \((\w+) weights\): ([0-9.]+|inf) dB)");
        static const std::regex reTexture(R"(^\s+(.+?)\s+: ([0-9.]+|inf) dB \[[^\]]*\]\s*$)");
        static const std::regex reStep(R"(^Training: (\d+) steps)");
        static const std::regex reSize(R"(File size: (\d+) bytes, ([0-9.]+) bits per pixel)");
        std::string tail;
        const auto t0 = std::chrono::steady_clock::now();
        auto onLine = [&](const std::string& line) {
            log << line << "\n";
            std::smatch m;
            if (std::regex_search(line, m, reStep)) {
                parsed.steps = std::atoi(m[1].str().c_str());
                if (parsed.steps % 20000 != 0) return;
                char text[160];
                std::snprintf(text, sizeof(text), "    %s ntc-cli: %s  %.0f s", in.name.c_str(), line.c_str(), secondsSince(t0));
                progressLine(text);
                return;
            }
            tail += line + "\n";
            if (tail.size() > 4000) tail.erase(0, tail.size() - 4000);
            auto value = [](const std::string& s) { return s == "inf" ? 999.0 : std::atof(s.c_str()); };
            if (std::regex_search(line, m, reShape) && parsed.gridSizeScale == 0) {
                parsed.gridSizeScale = std::atoi(m[1].str().c_str());
                parsed.numFeatures = std::atoi(m[2].str().c_str());
            } else if (std::regex_search(line, m, reOverall)) {
                (m[1].str() == "INT8" ? parsed.psnrInt8 : parsed.psnrFp8) = value(m[2].str());
            } else if (std::regex_search(line, m, reSize)) {
                parsed.fileBytes = size_t(std::strtoull(m[1].str().c_str(), nullptr, 10));
                parsed.fileBpp = std::atof(m[2].str().c_str());
            } else if (std::regex_search(line, m, reTexture)) {
                parsed.perTexture.emplace_back(m[1].str(), value(m[2].str()));
            }
        };
        int code = 0;
        if (!runProcess(cli_, args, in.workDir, onLine, code, err)) return false;
        const double cliSeconds = secondsSince(t0);
        std::printf("\n");
        if (code != 0) {
            err = "ntc-cli exited with code " + std::to_string(code) + " (log: " + fromPath(work / "ntc-cli.log") + ")\n" + tail;
            return false;
        }

        std::error_code ec;
        const size_t ntcBytes = size_t(fs::file_size(work / "material.ntc", ec));
        if (ec || ntcBytes == 0) {
            err = "ntc-cli did not write " + fromPath(work / "material.ntc");
            return false;
        }

        // Read the decoded images back into the canonical channel order of this material.
        // ntc-cli writes k channels as L, LA, RGB or RGBA but may reduce the PNG (gray, palette), so always expand to
        // RGBA: 1 channel = R, 2 channels = R and A, 3/4 channels = R,G,B(,A).
        out.decoded.assign(texels * N, 0.0f);
        for (const TextureGroup& grp : groups) {
            const int k = int(grp.outputs.size());
            const std::string file = fromPath(decodedDir / (grp.name + ".png"));
            int dw = 0, dh = 0, dn = 0;
            stbi_uc* data = stbi_load(file.c_str(), &dw, &dh, &dn, 4);
            if (!data) {
                err = "cannot read decoded image " + file + ": " + stbi_failure_reason();
                return false;
            }
            const bool okSize = dw == w && dh == h;
            if (okSize) {
                const int src[4] = {0, k == 2 ? 3 : 1, 2, 3};
                for (size_t t = 0; t < texels; ++t)
                    for (int c = 0; c < k; ++c) out.decoded[t * N + grp.outputs[c]] = data[t * 4 + src[c]] / 255.0f;
            }
            stbi_image_free(data);
            if (!okSize) {
                err = "decoded image " + file + " is " + std::to_string(dw) + "x" + std::to_string(dh) + ", expected " + std::to_string(w) + "x" +
                      std::to_string(h);
                return false;
            }
        }
        std::ifstream nf(work / "material.ntc", std::ios::binary);
        out.bitstream.assign(std::istreambuf_iterator<char>(nf), std::istreambuf_iterator<char>());
        NtcContainerInfo info;
        if (!parseNtcContainer(out.bitstream, info, err)) {
            err = "invalid ntc-cli output " + fromPath(work / "material.ntc") + ": " + err;
            return false;
        }
        // ntc-cli pads numChannels up to an even count
        if (info.width != w || info.height != h || info.numChannels < N || info.numChannels > N + 1 || info.textures.size() != groups.size()) {
            err = "ntc-cli output does not match the material (" + std::to_string(info.width) + "x" + std::to_string(info.height) + ", " +
                  std::to_string(info.numChannels) + " channels, " + std::to_string(info.textures.size()) + " textures)";
            return false;
        }
        // The texture ranges are what the extension's outputs refer to.
        for (const TextureGroup& grp : groups) {
            const auto t = std::find_if(info.textures.begin(), info.textures.end(), [&](const NtcTextureInfo& x) { return x.name == grp.name; });
            if (t == info.textures.end() || t->firstChannel != grp.outputs.front() || t->numChannels != int(grp.outputs.size())) {
                err = "ntc-cli output: texture '" + grp.name + "' does not cover the expected channels";
                return false;
            }
        }

        {
            int dcode = 0;
            std::string derr;
            std::ofstream df(work / "describe.txt", std::ios::binary);
            if (runProcess(cli_, {fromPath(work / "material.ntc"), "-d"}, in.workDir, [&](const std::string& l) { df << l << "\n"; }, dcode, derr) && dcode != 0)
                std::printf("    warning: ntc-cli -d exited with code %d\n", dcode);
        }
        if (!opts_.ntcKeepImages) {
            fs::remove_all(imageDir, ec);
            fs::remove_all(decodedDir, ec);
        }

        out.compressedBytes = ntcBytes;
        Value::Object ntc;
        ntc["containerVersion"] = Value(int(info.containerVersion));
        ntc["schemaVersion"] = Value(info.schemaVersion);
        ntc["numChannels"] = Value(info.numChannels);
        ntc["gridSizeScale"] = Value(info.gridSizeScale);
        ntc["numFeatures"] = Value(info.numFeatures);
        ntc["numColorMips"] = Value(info.numColorMips);
        ntc["encoder"] = Value("ntc-cli (LibNTC " + libVersion_ + ")");
        out.ntcInfo = std::move(ntc);
        Value::Object& x = out.extraMetrics;
        x["ntcFile"] = Value(fromPath(work / "material.ntc"));
        x["ntcBytes"] = Value(double(ntcBytes));
        x["ntcJsonBytes"] = Value(double(info.jsonBytes));
        x["ntcBpp"] = Value(8.0 * double(ntcBytes) / double(texels));
        x["targetBpp"] = Value(bpp);
        x["gridSizeScale"] = Value(info.gridSizeScale);
        x["numFeatures"] = Value(info.numFeatures);
        x["numColorMips"] = Value(info.numColorMips);
        x["ntcPsnrInt8"] = Value(parsed.psnrInt8);
        x["ntcPsnrFp8"] = Value(parsed.psnrFp8);
        x["trainingSteps"] = Value(parsed.steps);
        x["cliSeconds"] = Value(cliSeconds);
        x["encoder"] = Value("ntc-cli (LibNTC " + libVersion_ + ")");
        x["command"] = Value(cmdline);
        Value::Object per;
        for (const auto& [name, db] : parsed.perTexture) per[name.substr(0, name.find('|'))] = Value(db);
        x["ntcPsnrPerTexture"] = Value(std::move(per));
        return true;
    }

private:
    NeuralOptions opts_;
    std::string cli_, version_, libVersion_;
    std::map<std::string, double> matchBpp_;
};

} // namespace

std::unique_ptr<INeuralCodec> createNvidiaNtcCodec(const NeuralOptions& opts, std::string& err) {
    auto codec = std::make_unique<NvidiaNtc>(opts);
    if (!codec->init(err)) return nullptr;
    return codec;
}

} // namespace gtc
