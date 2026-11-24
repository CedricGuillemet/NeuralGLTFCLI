#!/usr/bin/env python3
"""Runs one neural-texture experiment end to end and appends a row to output/quality/experiments.csv.

  experiment.py --cfg a1_mask --flags "--neural-uv-mask on" [--seed 1] [--capture] [--asset flightHelmet]

Steps (every GPU job goes through the lock script when --lock is given):
  1. gltf-texture-compressor --format neural --neural-suffix <cfg>  (only the Leather material by default)
  2. tools/quality/eval.py  (PSNR / HF-PSNR, plus the UV-masked variants)
  3. tools/quality/capture.mjs  (--capture: render MAE of the load-time decoded glb at the reference viewpoint)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

GROUPS = ["base", "normal", "occlusion", "roughness/metallic"]
COLUMNS = (
    ["config", "flags", "seed", "asset", "materials"]
    + [f"{g}_{m}" for g in GROUPS for m in ("psnr", "hfPsnr", "psnrMasked", "hfPsnrMasked")]
    + ["base_hfRatio", "normal_hfRatio", "base_ssim", "psnrFloat", "psnrQuantized", "psnrFinal", "uvCoverage"]
    + ["mae_view2", "maeMesh_view2", "latentBytes", "weightBytes", "compressedBytes", "encodeSeconds", "glbBytes", "note"]
)


def run(cmd, log):
    log.write("> " + " ".join(cmd) + "\n")
    log.flush()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(p.stdout + "\n")
    log.flush()
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:])
        raise SystemExit(f"command failed ({p.returncode}): {cmd[0]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--asset", default="flightHelmet")
    ap.add_argument("--materials", default="LeatherPartsMat", help="'all' for the whole asset")
    ap.add_argument("--exe", default=os.path.join(REPO, "build-v2", "gltf-texture-compressor.exe"))
    ap.add_argument("--assets", required=True, help="directory with <asset>.glb")
    ap.add_argument("--out", required=True, help="output/quality directory")
    ap.add_argument("--lock", help="gpulock.ps1")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-run", action="store_true", help="reuse the existing outputs (eval and capture only)")
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()

    glb = os.path.join(a.assets, a.asset + ".glb")
    pngRoot = os.path.join(a.out, a.asset + "_png")
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    log = open(os.path.join(a.out, "logs", f"{a.asset}_{a.cfg}.log"), "a", encoding="utf-8")

    def gpu(cmd):
        return (["powershell", "-NoProfile", "-File", a.lock] + cmd) if a.lock else cmd

    if not a.no_run:
        cmd = [a.exe, glb, "--format", "neural", "--png", pngRoot, "-o", a.out, "--neural-suffix", a.cfg,
               "--seed", str(a.seed)]
        if a.materials != "all":
            cmd += ["--neural-materials", a.materials]
        cmd += a.flags.split()
        run(gpu(cmd), log)

    metrics = json.load(open(os.path.join(a.out, f"{a.asset}_neural_{a.cfg}.metrics.json")))
    sel = [m for m in metrics["materials"] if a.materials == "all" or m["name"] in a.materials.split(",")]
    mat = sel[0]
    for m in sel:
        if m["name"] == "LeatherPartsMat":
            mat = m

    evalDir = os.path.join(a.out, "eval", f"{a.asset}_{a.cfg}")
    ev = [sys.executable, os.path.join(HERE, "eval.py"), "--ref", os.path.join(pngRoot, "original"), "--method",
          f"{a.cfg}={os.path.join(pngRoot, 'neural_' + a.cfg)}", "--glb", glb, "--out", evalDir, "--no-images",
          "--mask", os.path.join(pngRoot, "mask"), "--materials", mat["name"]]
    run(ev, log)
    res = json.load(open(os.path.join(evalDir, "results.json")))
    groups = next(v for k, v in res["materials"].items() if k.endswith(mat["name"]))[a.cfg]

    row = {"config": a.cfg, "flags": a.flags, "seed": a.seed, "asset": a.asset, "materials": mat["name"], "note": a.note}
    for g in GROUPS:
        for m in ("psnr", "hfPsnr", "psnrMasked", "hfPsnrMasked"):
            v = groups.get(g, {}).get(m)
            row[f"{g}_{m}"] = "" if v is None else f"{v:.3f}"
    row["base_hfRatio"] = f"{groups['base']['hfRatio']:.3f}"
    row["normal_hfRatio"] = f"{groups['normal']['hfRatio']:.3f}"
    row["base_ssim"] = f"{groups['base'].get('ssim', 0):.4f}"
    cm = mat.get("codecMetrics") or {}
    for k in ("psnrFloat", "psnrQuantized", "psnrFinal"):
        row[k] = f"{cm[k]:.3f}" if k in cm else ""
    row["uvCoverage"] = f"{mat.get('uvCoverage', 0):.3f}"
    for k in ("latentBytes", "weightBytes", "compressedBytes"):
        row[k] = mat.get(k, "")
    row["encodeSeconds"] = f"{mat.get('encodeSeconds', 0):.1f}"
    glbPath = os.path.join(a.out, f"{a.asset}_neural_{a.cfg}.glb")
    row["glbBytes"] = os.path.getsize(glbPath) if os.path.exists(glbPath) else ""

    row["mae_view2"] = row["maeMesh_view2"] = ""
    if a.capture:
        cap = ["node", os.path.join(HERE, "capture.mjs"), "--out", os.path.join(a.out, "shots"), "--asset", a.asset,
               "--entry", f"{a.cfg}=loadtime:{a.cfg}"]
        run(gpu(cap), log)
        cj = json.load(open(os.path.join(a.out, "shots", f"captures_webgl_{a.asset}_v2.json")))
        st = cj[a.cfg]["compareStats"]
        row["mae_view2"] = f"{st['mae']:.3f}"
        row["maeMesh_view2"] = f"{st['maeMesh']:.3f}"

    if not a.no_csv:
        path = os.path.join(a.out, "experiments.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
    print(f"{a.cfg}: base {row['base_psnr']}/{row['base_hfPsnr']} (m {row['base_psnrMasked']}) "
          f"normal {row['normal_psnr']}/{row['normal_hfPsnr']} (m {row['normal_psnrMasked']}) "
          f"occ {row['occlusion_psnr']} rm {row['roughness/metallic_psnr']} mae {row['mae_view2']} "
          f"fl/q/f {row['psnrFloat']}/{row['psnrQuantized']}/{row['psnrFinal']} t {row['encodeSeconds']}s")


if __name__ == "__main__":
    main()
