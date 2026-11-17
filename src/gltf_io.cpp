#include "gltf_io.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>

#include "image.h"
#include "passes.h"

namespace fs = std::filesystem;
using tinygltf::Value;

namespace gtc {

int Document::addImage(std::vector<uint8_t> bytes, const std::string& mime, const std::string& name) {
    tinygltf::Image img;
    img.name = name;
    img.mimeType = mime;
    model.images.push_back(img);
    imageData.push_back(std::move(bytes));
    return int(model.images.size()) - 1;
}

int Document::addTexture(int imageIndex, int sampler, const std::string& name) {
    tinygltf::Texture tex;
    tex.name = name;
    tex.source = imageIndex;
    tex.sampler = sampler;
    model.textures.push_back(tex);
    return int(model.textures.size()) - 1;
}

int Document::addSampler(int magFilter, int minFilter, int wrapS, int wrapT) {
    for (size_t i = 0; i < model.samplers.size(); ++i) {
        const auto& s = model.samplers[i];
        if (s.magFilter == magFilter && s.minFilter == minFilter && s.wrapS == wrapS && s.wrapT == wrapT && s.name.empty())
            return int(i);
    }
    tinygltf::Sampler s;
    s.magFilter = magFilter;
    s.minFilter = minFilter;
    s.wrapS = wrapS;
    s.wrapT = wrapT;
    model.samplers.push_back(s);
    return int(model.samplers.size()) - 1;
}

int Document::addBufferView(std::vector<uint8_t> bytes, const std::string& name) {
    // Stored in a pending buffer (index -1); resolved by saveGlb.
    tinygltf::BufferView bv;
    bv.name = name;
    bv.buffer = -1 - int(extraViews.size());
    bv.byteLength = bytes.size();
    extraViews.push_back(std::move(bytes));
    model.bufferViews.push_back(bv);
    return int(model.bufferViews.size()) - 1;
}

void Document::addExtensionUsed(const std::string& ext, bool required) {
    auto add = [&](std::vector<std::string>& list) {
        if (std::find(list.begin(), list.end(), ext) == list.end()) list.push_back(ext);
    };
    add(model.extensionsUsed);
    if (required) add(model.extensionsRequired);
}

static uint64_t fileSize(const fs::path& p) {
    std::error_code ec;
    auto s = fs::file_size(p, ec);
    return ec ? 0 : uint64_t(s);
}

static bool isDataUri(const std::string& uri) { return uri.rfind("data:", 0) == 0; }

bool loadDocument(const std::string& path, Document& doc, LoadResult& info, std::string& err) {
    tinygltf::TinyGLTF loader;
    loader.SetImagesAsIs(true);
    std::string warn;
    std::string ext = fromPath(toPath(path).extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    bool ok = ext == ".glb" ? loader.LoadBinaryFromFile(&doc.model, &err, &warn, path)
                            : loader.LoadASCIIFromFile(&doc.model, &err, &warn, path);
    if (!ok) return false;

    for (const auto& e : doc.model.extensionsUsed) {
        if (e == "EXT_meshopt_compression" || e == "KHR_meshopt_compression") {
            err = e + " is not supported (buffers cannot be repacked)";
            return false;
        }
    }

    info.inputBytes = fileSize(path);
    fs::path base = toPath(path).parent_path();
    for (const auto& b : doc.model.buffers)
        if (!b.uri.empty() && !isDataUri(b.uri)) info.inputBytes += fileSize(base / toPath(b.uri));
    for (const auto& img : doc.model.images)
        if (!img.uri.empty() && !isDataUri(img.uri)) info.inputBytes += fileSize(base / toPath(img.uri));

    // Move image bytes out of the model; images get fresh bufferViews on save.
    doc.imageData.resize(doc.model.images.size());
    for (size_t i = 0; i < doc.model.images.size(); ++i) {
        auto& img = doc.model.images[i];
        doc.imageData[i] = std::move(img.image);
        img.image.clear();
        img.as_is = false;
        img.width = img.height = img.component = img.bits = img.pixel_type = -1;
        if (img.mimeType.empty()) img.mimeType = mimeFromBytes(doc.imageData[i]);
        img.uri.clear();
        img.bufferView = -1;
    }
    return true;
}

void forEachMaterialTexture(tinygltf::Material& m, const std::function<void(const std::string&, int&)>& fn) {
    auto core = [&](const char* slot, int& idx) {
        if (idx >= 0) fn(slot, idx);
    };
    core("baseColorTexture", m.pbrMetallicRoughness.baseColorTexture.index);
    core("metallicRoughnessTexture", m.pbrMetallicRoughness.metallicRoughnessTexture.index);
    core("normalTexture", m.normalTexture.index);
    core("occlusionTexture", m.occlusionTexture.index);
    core("emissiveTexture", m.emissiveTexture.index);

    std::function<void(Value&, const std::string&)> walk = [&](Value& v, const std::string& texSlot) {
        if (v.IsObject()) {
            auto& obj = v.Get<Value::Object>();
            if (!texSlot.empty()) {
                auto it = obj.find("index");
                if (it != obj.end() && it->second.IsNumber()) {
                    int idx = it->second.GetNumberAsInt();
                    fn(texSlot, idx);
                    it->second = Value(idx);
                }
            }
            for (auto& [key, child] : obj) {
                bool texKey = key.size() > 7 && (key.ends_with("Texture") || key.ends_with("Textures"));
                walk(child, texKey ? key : std::string());
            }
        } else if (v.IsArray()) {
            for (auto& child : v.Get<Value::Array>()) walk(child, texSlot);
        }
    };
    for (auto& [name, ext] : m.extensions) walk(ext, std::string());
}

static void forEachTextureImage(tinygltf::Texture& t, const std::function<void(int&)>& fn) {
    if (t.source >= 0) fn(t.source);
    for (auto& [name, ext] : t.extensions) {
        if (!ext.IsObject()) continue;
        auto& obj = ext.Get<Value::Object>();
        auto it = obj.find("source");
        if (it != obj.end() && it->second.IsNumber()) {
            int idx = it->second.GetNumberAsInt();
            fn(idx);
            it->second = Value(idx);
        }
    }
}

template <typename T>
static std::vector<int> compact(std::vector<T>& items, const std::vector<bool>& used) {
    std::vector<int> remap(items.size(), -1);
    std::vector<T> kept;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!used[i]) continue;
        remap[i] = int(kept.size());
        kept.push_back(std::move(items[i]));
    }
    items = std::move(kept);
    return remap;
}

static void pruneTextures(Document& doc) {
    auto& m = doc.model;
    std::vector<bool> texUsed(m.textures.size(), false);
    for (auto& mat : m.materials) forEachMaterialTexture(mat, [&](const std::string&, int& i) { if (i < int(texUsed.size())) texUsed[i] = true; });
    auto texRemap = compact(m.textures, texUsed);
    for (auto& mat : m.materials) forEachMaterialTexture(mat, [&](const std::string&, int& i) { i = texRemap[i]; });

    std::vector<bool> imgUsed(m.images.size(), false), smpUsed(m.samplers.size(), false);
    for (auto& t : m.textures) {
        forEachTextureImage(t, [&](int& i) { imgUsed[i] = true; });
        if (t.sampler >= 0) smpUsed[t.sampler] = true;
    }
    std::vector<bool> imgKeep = imgUsed;
    auto imgRemap = compact(m.images, imgKeep);
    std::vector<std::vector<uint8_t>> data;
    for (size_t i = 0; i < imgUsed.size(); ++i)
        if (imgUsed[i]) data.push_back(std::move(doc.imageData[i]));
    doc.imageData = std::move(data);
    auto smpRemap = compact(m.samplers, smpUsed);
    for (auto& t : m.textures) {
        forEachTextureImage(t, [&](int& i) { i = imgRemap[i]; });
        if (t.sampler >= 0) t.sampler = smpRemap[t.sampler];
    }
}

static int* dracoView(tinygltf::Primitive& p, int& storage, Value*& holder) {
    auto it = p.extensions.find("KHR_draco_mesh_compression");
    if (it == p.extensions.end() || !it->second.IsObject()) return nullptr;
    auto& obj = it->second.Get<Value::Object>();
    auto bv = obj.find("bufferView");
    if (bv == obj.end()) return nullptr;
    storage = bv->second.GetNumberAsInt();
    holder = &bv->second;
    return &storage;
}

bool saveGlb(Document doc, const std::string& path, std::string& err) {
    pruneTextures(doc);
    auto& m = doc.model;

    // Collect bufferView references (accessors, sparse, Draco, neural weights in material extensions).
    std::vector<bool> viewUsed(m.bufferViews.size(), false);
    auto markView = [&](int v) { if (v >= 0) viewUsed[v] = true; };
    for (auto& a : m.accessors) {
        markView(a.bufferView);
        if (a.sparse.isSparse) {
            markView(a.sparse.indices.bufferView);
            markView(a.sparse.values.bufferView);
        }
    }
    for (auto& mesh : m.meshes)
        for (auto& p : mesh.primitives) {
            int v; Value* h = nullptr;
            if (dracoView(p, v, h)) markView(v);
        }
    // Material extensions may reference bufferViews through a "bufferView" key (e.g. neural weights).
    std::function<void(Value&, const std::function<void(Value&)>&)> walkViews = [&](Value& v, const std::function<void(Value&)>& fn) {
        if (v.IsObject()) {
            for (auto& [k, c] : v.Get<Value::Object>()) {
                if (k == "bufferView" && c.IsNumber()) fn(c);
                else walkViews(c, fn);
            }
        } else if (v.IsArray()) {
            for (auto& c : v.Get<Value::Array>()) walkViews(c, fn);
        }
    };
    for (auto& mat : m.materials)
        for (auto& [n, e] : mat.extensions) walkViews(e, [&](Value& c) { markView(c.GetNumberAsInt()); });

    // Build the new single buffer.
    std::vector<uint8_t> bin;
    auto align = [&](size_t a) { while (bin.size() % a) bin.push_back(0); };
    std::vector<int> viewRemap(m.bufferViews.size(), -1);
    std::vector<tinygltf::BufferView> views;
    for (size_t i = 0; i < m.bufferViews.size(); ++i) {
        if (!viewUsed[i]) continue;
        auto bv = m.bufferViews[i];
        const uint8_t* src;
        if (bv.buffer < 0) {
            src = doc.extraViews[size_t(-1 - bv.buffer)].data();
        } else {
            const auto& buf = m.buffers[bv.buffer].data;
            if (bv.byteOffset + bv.byteLength > buf.size()) {
                err = "bufferView " + std::to_string(i) + " is out of range";
                return false;
            }
            src = buf.data() + bv.byteOffset;
        }
        align(8);
        bv.buffer = 0;
        bv.byteOffset = bin.size();
        bin.insert(bin.end(), src, src + bv.byteLength);
        viewRemap[i] = int(views.size());
        views.push_back(std::move(bv));
    }
    for (size_t i = 0; i < m.images.size(); ++i) {
        align(8);
        tinygltf::BufferView bv;
        bv.buffer = 0;
        bv.byteOffset = bin.size();
        bv.byteLength = doc.imageData[i].size();
        bin.insert(bin.end(), doc.imageData[i].begin(), doc.imageData[i].end());
        m.images[i].bufferView = int(views.size());
        m.images[i].uri.clear();
        views.push_back(bv);
    }
    align(4);

    auto remap = [&](int& v) { if (v >= 0) v = viewRemap[v]; };
    for (auto& a : m.accessors) {
        remap(a.bufferView);
        if (a.sparse.isSparse) {
            remap(a.sparse.indices.bufferView);
            remap(a.sparse.values.bufferView);
        }
    }
    for (auto& mesh : m.meshes)
        for (auto& p : mesh.primitives) {
            int v; Value* h = nullptr;
            if (dracoView(p, v, h)) *h = Value(viewRemap[v]);
        }
    for (auto& mat : m.materials)
        for (auto& [n, e] : mat.extensions) walkViews(e, [&](Value& c) { c = Value(viewRemap[c.GetNumberAsInt()]); });

    m.bufferViews = std::move(views);
    m.buffers.clear();
    tinygltf::Buffer buffer;
    buffer.data = std::move(bin);
    m.buffers.push_back(std::move(buffer));

    for (auto& img : m.images) img.image.clear();

    tinygltf::TinyGLTF writer;
    if (!writer.WriteGltfSceneToFile(&m, path, true, true, false, true)) {
        err = "failed to write " + path;
        return false;
    }
    return true;
}

} // namespace gtc
