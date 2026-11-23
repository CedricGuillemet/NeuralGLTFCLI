#!/usr/bin/env python3
"""Texture-space quality evaluation for the --png output of gltf-texture-compressor.

Per material and channel group (base RGB, alpha, normal XY, occlusion, roughness, metallic) it reports
PSNR (full and UV-masked), SSIM (Gaussian window, base luminance and normal XY) and HF-PSNR, the PSNR of
x - G_sigma(x) (sigma 1.5), which isolates the loss of high-frequency detail. It also writes error heatmaps
and two fixed 512x512 crops of the most detailed area of one material, reused for every method.

Only numpy and Pillow are required.

  eval.py --ref out/flightHelmet_png/original \
          --method neural=out/flightHelmet_png/neural_v2 --method ktx=out/flightHelmet_png/ktx \
          --glb assets/flightHelmet.glb --out out/quality/eval --materials LeatherPartsMat

A method directory holds either <index>_<material>_<group>.png files (neural outputs) or <imageIndex>_image<imageIndex>.png
files (the ktx output); the latter are mapped to materials through the glTF JSON given with --glb.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np
from PIL import Image

SIGMA = 1.5
FILE_GROUPS = ["baseColor", "normal", "occlusionRoughnessMetallic", "emissive", "diffuse", "specular", "glossiness"]
# (file group, reported name, channels)
SCORED = [
    ("baseColor", "base", [0, 1, 2]),
    ("baseColor", "alpha", [3]),
    ("normal", "normal", [0, 1]),
    ("occlusionRoughnessMetallic", "occlusion", [0]),
    ("occlusionRoughnessMetallic", "roughness", [1]),
    ("occlusionRoughnessMetallic", "metallic", [2]),
    ("occlusionRoughnessMetallic", "roughness/metallic", [1, 2]),
    ("emissive", "emissive", [0, 1, 2]),
    ("diffuse", "diffuse", [0, 1, 2]),
    ("specular", "specular", [0, 1, 2]),
    ("glossiness", "glossiness", [0]),
]


def read_glb_json(path):
    if path.lower().endswith(".gltf"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, "not a GLB"
        clen, ctype = struct.unpack("<II", f.read(8))
        assert ctype == 0x4E4F534A
        return json.loads(f.read(clen))


def gaussian_kernel(sigma):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


KERNEL = gaussian_kernel(SIGMA)


def blur(img):
    """Separable Gaussian blur of an (H, W) float32 array with reflected borders."""
    r = len(KERNEL) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(KERNEL):
        out += w * p[:, i : i + img.shape[1]]
    p = np.pad(out, ((r, r), (0, 0)), mode="reflect")
    res = np.zeros_like(img)
    for i, w in enumerate(KERNEL):
        res += w * p[i : i + img.shape[0], :]
    return res


def ssim(a, b):
    """Mean SSIM (Gaussian window sigma 1.5) of two (H, W) arrays in [0, 1]."""
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a * mu_a
    sbb = blur(b * b) - mu_b * mu_b
    sab = blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (saa + sbb + c2))
    return float(m.mean())


def psnr_of(mse):
    return 99.0 if mse <= 1e-12 else float(-10 * np.log10(mse))


def load(path):
    im = Image.open(path)
    if im.mode not in ("L", "RGB", "RGBA"):
        im = im.convert("RGBA")
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a[:, :, None] if a.ndim == 2 else a


def heatmap(err):
    """|diff| * 8 mapped through a black-red-yellow-white ramp."""
    t = np.clip(err * 8.0, 0, 1)
    return (np.stack([np.clip(t * 3, 0, 1), np.clip(t * 3 - 1, 0, 1), np.clip(t * 3 - 2, 0, 1)], -1) * 255 + 0.5).astype(np.uint8)


def method_files(dir_, glb):
    """Maps (prefix, file group) -> path for one method directory."""
    files = {}
    pat = re.compile(r"^(\d+_.+?)_(%s)\.png$" % "|".join(FILE_GROUPS))
    for n in os.listdir(dir_):
        m = pat.match(n)
        if m:
            files[(m.group(1), m.group(2))] = os.path.join(dir_, n)
    # ktx-style files: <imageIndex>_image<imageIndex>.png, mapped through the glTF materials
    if glb is not None:
        for mi, mat in enumerate(glb.get("materials", [])):
            prefix = f"{mi}_{mat.get('name') or 'material' + str(mi)}"
            pbr = mat.get("pbrMetallicRoughness", {})
            slots = {
                "baseColor": pbr.get("baseColorTexture"),
                "normal": mat.get("normalTexture"),
                "occlusionRoughnessMetallic": pbr.get("metallicRoughnessTexture") or mat.get("occlusionTexture"),
            }
            for g, t in slots.items():
                if t is None or (prefix, g) in files:
                    continue
                src = glb["textures"][t["index"]]["source"]
                p = os.path.join(dir_, f"{src}_image{src}.png")
                if os.path.exists(p):
                    files[(prefix, g)] = p
    return files


def pick_crops(base, normal=None, size=512):
    """Two non-overlapping windows with the highest high-frequency energy of the reference base color and normal."""
    lum = base[:, :, :3].mean(axis=2)
    hf = lum - blur(lum)
    energy = hf * hf
    if normal is not None:
        for c in (0, 1):
            n = normal[:, :, c] - blur(normal[:, :, c])
            energy = energy + n * n
    h, w = energy.shape
    size = min(size, h, w)
    ii = np.pad(energy, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, h - size + 1, 32)
    xs = np.arange(0, w - size + 1, 32)
    score = ii[np.ix_(ys + size, xs + size)] - ii[np.ix_(ys, xs + size)] - ii[np.ix_(ys + size, xs)] + ii[np.ix_(ys, xs)]
    crops = []
    for _ in range(2):
        yi, xi = np.unravel_index(np.argmax(score), score.shape)
        y, x = int(ys[yi]), int(xs[xi])
        crops.append({"x": x, "y": y, "size": size})
        score[(np.abs(ys[:, None] - y) < size) & (np.abs(xs[None, :] - x) < size)] = -1
    return crops


def save(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def to_u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def score_pair(group, ref, test, mask):
    out = {}
    for fg, name, chans in SCORED:
        if fg != group or max(chans) >= ref.shape[2]:
            continue
        if name == "alpha" and float(ref[:, :, 3].min()) > 0.999:
            continue
        d = test[:, :, chans] - ref[:, :, chans]
        hf = np.stack([(test[:, :, c] - blur(test[:, :, c])) - (ref[:, :, c] - blur(ref[:, :, c])) for c in chans], -1)
        row = {"psnr": psnr_of(float((d * d).mean())), "hfPsnr": psnr_of(float((hf * hf).mean()))}
        hf_ref = np.stack([ref[:, :, c] - blur(ref[:, :, c]) for c in chans], -1)
        # detail retention: energy of the test's high-pass over the reference's (< 1 means blurred)
        row["hfRatio"] = float(np.sqrt(((hf_ref + hf) ** 2).mean() / max(float((hf_ref**2).mean()), 1e-12)))
        if mask is not None:
            row["psnrMasked"] = psnr_of(float((d[mask] ** 2).mean()))
            row["hfPsnrMasked"] = psnr_of(float((hf[mask] ** 2).mean()))
        if name == "base":
            row["ssim"] = ssim(ref[:, :, :3].mean(2), test[:, :, :3].mean(2))
        elif name == "normal":
            row["ssim"] = float(np.mean([ssim(ref[:, :, c], test[:, :, c]) for c in (0, 1)]))
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="the original/ PNG directory")
    ap.add_argument("--method", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--glb", help="original glTF/GLB, needed to map ktx image files to materials")
    ap.add_argument("--mask", help="directory with <index>_<material>_mask.png UV masks")
    ap.add_argument("--materials", help="comma separated material names (default: all)")
    ap.add_argument("--out", required=True, help="output directory (results.json, heatmaps, crops)")
    ap.add_argument("--crop-material", default="LeatherPartsMat")
    ap.add_argument("--no-images", action="store_true", help="skip heatmaps and crops")
    args = ap.parse_args()

    glb = read_glb_json(args.glb) if args.glb else None
    ref_files = method_files(args.ref, glb)
    wanted = set(args.materials.split(",")) if args.materials else None
    methods = []
    for m in args.method:
        name, _, d = m.partition("=")
        methods.append((name, method_files(d, glb)))

    results = {"sigma": SIGMA, "materials": {}, "crops": {}}
    os.makedirs(args.out, exist_ok=True)
    for prefix in sorted({p for p, _ in ref_files}, key=lambda p: int(p.split("_")[0])):
        label = prefix.split("_", 1)[1]
        if wanted and label not in wanted:
            continue
        mask = None
        if args.mask:
            mp = os.path.join(args.mask, prefix + "_mask.png")
            if os.path.exists(mp):
                mask = np.asarray(Image.open(mp).convert("L")) > 127
                if not mask.any():
                    mask = None
        res = {}
        base_cache = {}
        for g in FILE_GROUPS:
            if (prefix, g) not in ref_files:
                continue
            ref = load(ref_files[(prefix, g)])
            for mname, files in methods:
                if (prefix, g) not in files:
                    continue
                test = load(files[(prefix, g)])
                if test.shape[:2] != ref.shape[:2]:
                    print(f"warning: {mname} {prefix} {g}: size {test.shape} != {ref.shape}, skipped", file=sys.stderr)
                    continue
                if test.shape[2] != ref.shape[2]:
                    n = min(test.shape[2], ref.shape[2])
                    test, r = test[:, :, :n], ref[:, :, :n]
                else:
                    r = ref
                res.setdefault(mname, {}).update(score_pair(g, r, test, mask))
                if not args.no_images:
                    n = min(3, r.shape[2])
                    err = np.abs(test[:, :, :n] - r[:, :, :n])
                    if g == "normal":
                        err[:, :, 2:] = 0
                    heat = heatmap(err.max(axis=2))
                    save(heat, os.path.join(args.out, "heat", mname, f"{prefix}_{g}.png"))
                    if g == "baseColor":
                        base_cache[mname] = (test, heat)
            if g == "baseColor" and label == args.crop_material and not args.no_images:
                crops = pick_crops(ref, load(ref_files[(prefix, "normal")]) if (prefix, "normal") in ref_files else None)
                results["crops"][label] = crops
                for i, c in enumerate(crops, 1):
                    sl = (slice(c["y"], c["y"] + c["size"]), slice(c["x"], c["x"] + c["size"]))
                    save(to_u8(ref[sl][:, :, :3]), os.path.join(args.out, "crops", "original", f"crop{i}_img.png"))
                    for mname, (test, heat) in base_cache.items():
                        save(to_u8(test[sl][:, :, :3]), os.path.join(args.out, "crops", mname, f"crop{i}_img.png"))
                        save(heat[sl], os.path.join(args.out, "crops", mname, f"crop{i}_heat.png"))
        results["materials"][prefix] = res

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for prefix, res in results["materials"].items():
        for mname, groups in res.items():
            line = " ".join(
                f"{g} {r['psnr']:.2f}/{r['hfPsnr']:.2f}/{r['hfRatio']:.2f}" + (f" (m {r['psnrMasked']:.2f}/{r['hfPsnrMasked']:.2f})" if "psnrMasked" in r else "")
                for g, r in groups.items()
            )
            print(f"{prefix:30s} {mname:16s} {line}")


if __name__ == "__main__":
    main()
