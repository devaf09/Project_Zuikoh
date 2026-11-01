"""est_04 の図（その 2）: 同じ作り方の 2023 年の画像だけで、駐機位置の切り出しを日付ごとに並べる（docs/images/est04b_stands_grid_<空港>.png）

est_04 の最初の結果で「強い日・弱い日」に分かれて見えたのは、2024 年以降の画像（作り方・明るさの物差しが違う版）が
まとめて明るかったためと分かった。同じ版の中で、飛行機がいる・いないが目で見て分かるかを確かめる。
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est04b.py --airport bkk
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from fig_est04 import crop  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airport", default="bkk")
    ap.add_argument("--n-dates", type=int, default=16, help="先頭から何日付を同じ版とみなすか（bkk は 2023 年の 16 日付）")
    a = ap.parse_args()
    d = data_dir()
    T = np.loadtxt(d / "cache" / f"est04_{a.airport}.csv", delimiter=",")
    names = open(d / "cache" / f"est04_{a.airport}.csv", encoding="utf-8").readline().strip("# \n").split(",")[2:]
    files = sorted((d / "raw" / "umbra" / a.airport).glob("*_GEC.tif"))[:a.n_dates]
    lon, lat, tab = T[:, 0], T[:, 1], T[:, 2:2 + a.n_dates]
    R = tab - np.nanmedian(tab, 0)
    rng_ = np.nanmax(R, 1) - np.nanmin(R, 1)
    stands = np.argsort(-rng_)[[0, 10, 40, 80]]
    cols = [0, 3, 6, 9, 12, 15]
    fig, axes = plt.subplots(len(stands), len(cols), figsize=(14, 10))
    for r, i in enumerate(stands):
        for c, j in enumerate(cols):
            im = crop(files[j], lon[i], lat[i], half_m=50, n=120)
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=max(1, np.percentile(im, 99.5)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"位置 {i} {names[j][2:10]}\n{R[i, j]:+.1f} dB", fontsize=8)
    fig.suptitle(f"est_04（その 2）: {a.airport} の駐機位置（OSM、中心）の周り ±50 m を、同じ版の 2023 年の画像で並べる（明るさは日付の中央値との差）", fontsize=10)
    fig.text(0.01, 0.005, "SAR の画像 © Umbra Lab, Inc.（CC BY 4.0）。駐機位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(ROOT, "docs", "images", f"est04b_stands_grid_{a.airport}.png")
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
