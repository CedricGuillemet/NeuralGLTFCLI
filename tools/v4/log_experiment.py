"""Append one babylon-ntc-v4 run (a *.metrics.json written by --neural-suffix) to an experiments CSV.

usage: log_experiment.py <experiments.csv> <config> <metrics.json> [--flags "..."] [--note "..."] [--glb path]
One row per material in the metrics file; masked mip-0 PSNR per channel group (base, normal, ...) comes from the
tool's own written-bytes decode (codecMetrics.maskedPsnrByMip) so it is the number the runtime would reproduce.
"""
import argparse
import csv
import json
import os

COLS = ["config", "flags", "seed", "asset", "material", "base_psnrMasked", "normal_psnrMasked", "occlusion_psnrMasked",
        "roughness_psnrMasked", "metallic_psnrMasked", "base_psnr", "normal_psnr", "base_mip1", "base_mip2", "base_mip3",
        "normal_mip1", "normal_mip2", "normal_mip3", "psnrFloat", "psnrQuantPre", "psnrFinal", "latentBytes", "weightBytes",
        "compressedBytes", "encodeSeconds", "glbBytes", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("config")
    ap.add_argument("metrics")
    ap.add_argument("--flags", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--glb", default="")
    a = ap.parse_args()
    with open(a.metrics, encoding="utf-8") as f:
        j = json.load(f)
    asset = os.path.basename(a.metrics).split("_neural")[0]
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        for m in j["materials"]:
            cm = m.get("codecMetrics", {})
            pm = cm.get("maskedPsnrByMip", {})
            g = lambda mip, grp: round(pm.get(mip, {}).get(grp, float("nan")), 3)
            row = [a.config, a.flags, j.get("seed", ""), asset, m["name"], g("mip0", "base"), g("mip0", "normal"),
                   g("mip0", "occlusion"), g("mip0", "roughness"), g("mip0", "metallic"),
                   round(m["psnr"].get("base", float("nan")), 3), round(m["psnr"].get("normal", float("nan")), 3),
                   g("mip1", "base"), g("mip2", "base"), g("mip3", "base"), g("mip1", "normal"), g("mip2", "normal"),
                   g("mip3", "normal"), round(cm.get("psnrFloat", 0), 3), round(cm.get("psnrQuantPre", 0), 3),
                   round(cm.get("psnrFinal", 0), 3), int(m["latentBytes"]), int(m["weightBytes"]), int(m["compressedBytes"]),
                   round(m["encodeSeconds"], 1), os.path.getsize(a.glb) if a.glb and os.path.exists(a.glb) else "", a.note]
            w.writerow(row)


if __name__ == "__main__":
    main()
