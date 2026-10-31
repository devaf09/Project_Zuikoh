"""est_01c の図: SLC（1 回分の複素数の画像）で見たタンクと、1 基の距離の向きの明るさの形（docs/images/est01c_slc_tanks.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est01c.py
est_01c が cache/ に残した切り出しと、results/est01c.txt の位置合わせのずれを使う。
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import GCPTransformer  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    d = data_dir()
    rec = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est01c.txt"), encoding="utf-8")][-1]
    r0, r1, c0, c1 = rec["window"]
    z = np.load(d / "cache" / "est01c_cushing_vv.npy")
    pw = np.abs(z) ** 2
    bg = np.median(pw[pw > 0])
    tanks = json.load(open(d / "raw" / "osm" / "cushing" / "storage_tanks.geojson", encoding="utf-8"))
    lon = [np.mean([p[0] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
    lat = [np.mean([p[1] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
    zp = sorted((d / "raw" / "s1_slc" / "cushing").glob("*.zip"))[-1]
    name = [n for n in zipfile.ZipFile(zp).namelist() if n.endswith(rec["subswath_file"])][0]
    with rasterio.open(f"/vsizip/{zp.as_posix()}/{name}") as s:
        rows, cols = GCPTransformer(s.gcps[0]).rowcol(lon, lat)
    rows = np.array(rows) - r0 + rec["shift_rows"]
    cols = np.array(cols) - c0 + rec["shift_cols"]

    look = 4                                                  # 距離の向きを 4 画素平均して、ほぼ正方形の画素に（表示だけ）
    m = ndimage.uniform_filter(pw, (1, look))[:, ::look]
    db = 10 * np.log10(m / bg + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    ax.imshow(db, cmap="gray", vmin=-5, vmax=30, aspect="auto")
    ax.plot(cols / look, rows, "o", mfc="none", mec="#ff7f0e", ms=4, mew=0.7)
    ax.set_title("SLC の明るさ（背景の中央値との比）と OSM のタンク（橙、位置合わせ後）")
    ax.set_xlabel(f"距離の向き（画素 ÷ {look}、1 目盛り ≈ 15 m）")
    ax.set_ylabel("進行方向（行、1 行 ≈ 14 m）")
    # いちばん明るい大きなタンクの、距離の向きの形
    best, bi = -1, None
    for i, (r, c) in enumerate(zip(rows, cols)):
        r, c = int(r), int(c)
        if 5 <= r < pw.shape[0] - 5 and 40 <= c < pw.shape[1] - 40:
            v = pw[r - 3:r + 4, c - 30:c + 31].max()
            if v > best:
                best, bi = v, i
    r, c = int(rows[bi]), int(cols[bi])
    prof = 10 * np.log10(pw[r - 3:r + 4, c - 40:c + 41].max(axis=0) / bg + 1e-12)
    x = (np.arange(len(prof)) - 40) * 2.33
    axes[1].plot(x, prof, color="#d95f02")
    axes[1].axhline(15, color="gray", ls="--", lw=0.8)
    axes[1].text(x[0], 16, "15 dB（est_01 で高さを約 0.25 m で測れる線）", fontsize=8, color="gray")
    axes[1].set_xlabel("距離の向きの位置（斜めの距離 m、タンクの中心が 0）")
    axes[1].set_ylabel("明るさ（背景の中央値との比、dB）")
    axes[1].set_title("1 基のタンクの距離の向きの形（進行方向 7 行の最大）")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"est_01c: クッシングの SLC（{rec['scene'][17:25]}、VV、IW3）。強い点の比の中央値 {rec['scr_db_median']} dB、"
                 f"15 dB 以上 {rec['frac_scr_ge_15db']:.0%}", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。タンクの位置 © OpenStreetMap contributors（ODbL）", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(ROOT, "docs", "images", "est01c_slc_tanks.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
