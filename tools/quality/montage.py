#!/usr/bin/env python3
"""Montage of methods: rows are methods, columns are the render, the render diff, crop 1, crop 1 heatmap, crop 2
and crop 2 heatmap.

  montage.py --shots output/quality/shots --eval output/quality/eval --out output/quality/montage.png \
             --row original --row ktx --row base=neural_base

A row is label[=evalName]: <label> selects <shots>/<engine>_<asset>_v<view>_<label>_{plain,cmp}.png, <evalName>
(default: label) selects <eval>/crops/<evalName>/crop{1,2}_{img,heat}.png written by eval.py. Rows without crops
(for example the original) show the reference crops and no heatmaps.
"""
import argparse
import os

from PIL import Image, ImageDraw


def load(path, size):
    if not os.path.exists(path):
        return Image.new("RGB", size, (40, 40, 40))
    im = Image.open(path).convert("RGB")
    return im.resize(size, Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--engine", default="webgl")
    ap.add_argument("--asset", default="flightHelmet")
    ap.add_argument("--view", default="2")
    ap.add_argument("--stats", help="optional captures_*.json to print MAE on each row")
    ap.add_argument("--cell", type=int, default=420, help="crop cell size in pixels")
    args = ap.parse_args()

    stats = {}
    if args.stats:
        import json

        with open(args.stats) as f:
            stats = json.load(f)
    cs = args.cell
    render = (int(cs * 16 / 9), cs)
    cols = [("render", render), ("diff", render)] + [(n, (cs, cs)) for n in ("crop 1", "crop 1 err", "crop 2", "crop 2 err")]
    width = sum(c[1][0] for c in cols)
    sheet = Image.new("RGB", (width, cs * len(args.row)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for r, spec in enumerate(args.row):
        label, _, ev = spec.partition("=")
        ev = ev or label
        prefix = os.path.join(args.shots, f"{args.engine}_{args.asset}_v{args.view}_{label}")
        crops = os.path.join(args.eval, "crops", ev)
        ref = os.path.join(args.eval, "crops", "original")
        paths = [
            prefix + "_plain.png",
            prefix + "_cmp.png",
            os.path.join(crops if os.path.exists(crops) else ref, "crop1_img.png"),
            os.path.join(crops, "crop1_heat.png"),
            os.path.join(crops if os.path.exists(crops) else ref, "crop2_img.png"),
            os.path.join(crops, "crop2_heat.png"),
        ]
        x = 0
        for (name, size), p in zip(cols, paths):
            sheet.paste(load(p, size), (x, r * cs))
            x += size[0]
        mae = stats.get(label, {}).get("compareStats", {}).get("mae")
        text = label + (f"  MAE {mae:.2f}" if isinstance(mae, (int, float)) else "")
        d.rectangle((0, r * cs, 12 + 7 * len(text), r * cs + 16), fill=(0, 0, 0))
        d.text((4, r * cs + 3), text, fill=(255, 255, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
