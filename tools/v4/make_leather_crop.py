"""Builds the 64x64 LeatherPartsMat crop used to train the `trained_leather` golden case.

usage: python make_leather_crop.py assets/flightHelmet.glb out.glb [x0 y0]

The output is a unit quad with the base colour, metallic-roughness (also occlusion) and
normal textures of LeatherPartsMat cropped to 64x64 at (x0, y0), default (960, 960).
"""
import io
import json
import struct
import sys

from PIL import Image


def read_glb(path):
    d = open(path, 'rb').read()
    jl = struct.unpack('<I', d[12:16])[0]
    j = json.loads(d[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack('<I', d[off:off + 4])[0]
    return j, d[off + 8:off + 8 + bl]


def image_of(j, bin_, tex):
    bv = j['bufferViews'][j['images'][j['textures'][tex]['source']]['bufferView']]
    return Image.open(io.BytesIO(bin_[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    x0, y0 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (960, 960)
    j, bin_ = read_glb(src)
    mat = next(m for m in j['materials'] if m['name'] == 'LeatherPartsMat')
    tex = [mat['pbrMetallicRoughness']['baseColorTexture']['index'],
           mat['pbrMetallicRoughness']['metallicRoughnessTexture']['index'],
           mat['normalTexture']['index']]
    blob = bytearray()
    views, images = [], []

    def add(data):
        while len(blob) % 4:
            blob.append(0)
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)})
        blob.extend(data)
        return len(views) - 1

    pos = struct.pack('<12f', 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0)
    nrm = struct.pack('<12f', *([0, 0, 1] * 4))
    uv = struct.pack('<8f', 0, 1, 1, 1, 1, 0, 0, 0)
    idx = struct.pack('<6H', 0, 1, 2, 0, 2, 3)
    bv_pos, bv_nrm, bv_uv, bv_idx = add(pos), add(nrm), add(uv), add(idx)
    for t in tex:
        im = image_of(j, bin_, t).crop((x0, y0, x0 + 64, y0 + 64))
        b = io.BytesIO()
        im.save(b, 'PNG')
        images.append({'bufferView': add(b.getvalue()), 'mimeType': 'image/png'})
    out = {
        'asset': {'version': '2.0'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3, 'material': 0}]}],
        'materials': [{'name': 'LeatherPartsMat',
                       'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicRoughnessTexture': {'index': 1}},
                       'normalTexture': {'index': 2}, 'occlusionTexture': {'index': 1}}],
        'textures': [{'source': i} for i in range(3)],
        'images': images,
        'accessors': [
            {'bufferView': bv_pos, 'componentType': 5126, 'count': 4, 'type': 'VEC3', 'min': [0, 0, 0], 'max': [1, 1, 0]},
            {'bufferView': bv_nrm, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
            {'bufferView': bv_uv, 'componentType': 5126, 'count': 4, 'type': 'VEC2'},
            {'bufferView': bv_idx, 'componentType': 5123, 'count': 6, 'type': 'SCALAR'}],
        'bufferViews': views,
        'buffers': [{'byteLength': len(blob)}],
    }
    js = json.dumps(out, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    while len(blob) % 4:
        blob.append(0)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(dst, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))


main()
