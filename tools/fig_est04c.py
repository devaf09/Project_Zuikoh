"""est_04 の図（その 3）: データの相関で位置を合わせ直した画像（tools/align_gec.py）で、駐機位置の切り出しを並べ直す（docs/images/est04c_stands_aligned_<空港>.png）

合わせ直しの基準は最初の日付だが、ほかの日付のずれの中央値を「多数の日付の位置」とみなし、その分だけ切り出しの場所を動かす
（基準の日付だけがずれている場合に、OSM の駐機位置と画像を合わせるため）。
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est04c.py --airport bkk
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airport", default="bkk")
    ap.add_argument("--half-m", type=float, default=50)
    a = ap.parse_args()
    d = data_dir()
    al = [json.loads(x) for x in open(os.path.join(ROOT, "results", f"align_{a.airport}.txt"), encoding="utf-8")]
    me = float(np.median([r["shift_east_m"] for r in al[1:]]))      # 多数の日付の位置（基準の日付からのずれの中央値）
    mn = float(np.median([r["shift_north_m"] for r in al[1:]]))
    osm = json.load(open(d / "raw" / "osm" / "airports" / f"{a.airport}.geojson", encoding="utf-8"))
    st = []
    for f in osm["features"]:
        if f["properties"].get("aeroway") == "parking_position":
            g = f["geometry"]
            pts = [g["coordinates"]] if g["type"] == "Point" else (g["coordinates"] if g["type"] == "LineString" else g["coordinates"][0])
            st.append((np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])))
    files = [d / "processed" / "umbra_aligned" / a.airport / f"{r['date']}.tif" for r in al]
    picks = [18, 30, 103, 136]
    cols = [1, 4, 7, 10, 13, 16, 19, 22]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(16, 9.5))
    with rasterio.open(files[0]) as s0:
        xs, ys = transform("EPSG:4326", s0.crs, [p[0] for p in st], [p[1] for p in st])
        tr = s0.transform
    for r, i in enumerate(picks):
        # 基準の画像の座標で、多数の日付の位置に合わせた分だけ動かす（画像は基準に重ねてあるので、基準の日付のずれを引く）
        x, y = xs[i] + me, ys[i] + mn
        col, row = ~tr * (x, y)
        h = int(a.half_m)
        for c, j in enumerate(cols):
            with rasterio.open(files[j]) as s:
                im = s.read(1, window=((int(row) - h, int(row) + h), (int(col) - h, int(col) + h)))
            v = im[im > 0]
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(v, 99.5) if v.size else 1)
            ax.plot([h], [h], "+", color="#ff7f0e", ms=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"位置 {i} {al[j]['date'][2:10]}", fontsize=8)
    fig.suptitle(f"est_04（その 3）: データの相関で位置を合わせ直した {a.airport} の画像で、駐機位置（橙の +、OSM）の周り ±{int(a.half_m)} m を並べる", fontsize=10)
    fig.text(0.01, 0.005, f"多数の日付の位置 = 基準の日付から東 {me:+.0f} m・北 {mn:+.0f} m。SAR の画像 © Umbra Lab, Inc.（CC BY 4.0）。駐機位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(ROOT, "docs", "images", f"est04c_stands_aligned_{a.airport}.png")
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
