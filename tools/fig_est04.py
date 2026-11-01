"""est_04 の図: 駐機位置の明るさの時系列と、「強い日」「弱い日」の画像の見本（docs/images/est04_stands_<空港>.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est04.py --airport bkk
est_04 が cache/ に残した「駐機位置 × 日付」の表と、Umbra の GEC を使う。
"""
from __future__ import annotations

import argparse
import math
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


def crop(path, lon, lat, half_m=60, n=160):
    """GEC から、(lon, lat) の周り ±half_m を n × n 画素に読んで返す（8 ビットの値）。"""
    with rasterio.open(path) as s:
        (x,), (y,) = transform("EPSG:4326", s.crs, [lon], [lat])
        if s.crs.to_epsg() == 4326:
            mx, my = s.res[0] * 111320 * math.cos(math.radians(lat)), s.res[1] * 110540
        else:
            mx, my = s.res
        col, row = ~s.transform * (x, y)
        hc, hr = int(half_m / mx), int(half_m / my)
        a = s.read(1, window=((int(row) - hr, int(row) + hr), (int(col) - hc, int(col) + hc)), out_shape=(n, n))
        if s.transform.e > 0:                                 # 上下が逆さの画像は北を上に
            a = a[::-1]
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airport", default="bkk")
    a = ap.parse_args()
    d = data_dir()
    T = np.loadtxt(d / "cache" / f"est04_{a.airport}.csv", delimiter=",")
    names = open(d / "cache" / f"est04_{a.airport}.csv", encoding="utf-8").readline().strip("# \n").split(",")[2:]
    files = sorted((d / "raw" / "umbra" / a.airport).glob("*_GEC.tif"))
    lon, lat, tab = T[:, 0], T[:, 1], T[:, 2:]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(gs[0, :3])
    v = tab[np.isfinite(tab)]
    ax.hist(v, bins=60, color="#7570b3")
    ax.set_xlabel("駐機位置の半径 20 m の明るさ（その日の駐機場の中央値との比、dB）")
    ax.set_ylabel("駐機位置 × 日付の数")
    ax.set_title(f"{a.airport}: 駐機位置 {tab.shape[0]} か所 × {tab.shape[1]} 日付", fontsize=10)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 3:])
    rng = np.random.default_rng(0)
    for i in rng.choice(tab.shape[0], 12, replace=False):
        ax.plot(np.arange(tab.shape[1]), tab[i], "o-", ms=3, lw=0.7, alpha=0.8)
    ax.set_xticks(np.arange(len(names))[::3], [n[2:10] for n in names][::3], fontsize=7, rotation=30)
    ax.set_ylabel("明るさ（dB）")
    ax.set_title("駐機位置 12 か所の日付ごとの明るさ（強い日と弱い日を行き来する）", fontsize=10)
    ax.grid(alpha=0.3)
    spread = np.nanmax(tab, 1) - np.nanmin(tab, 1)
    picks = np.argsort(-spread)[[0, 5, 10]]
    for r, i in enumerate(picks[:2]):
        jb, jd = int(np.nanargmax(tab[i])), int(np.nanargmin(tab[i]))
        for c, (j, lab) in enumerate([(jb, "強い日"), (jd, "弱い日")]):
            ax = fig.add_subplot(gs[1 + r, c * 3:c * 3 + 3])
            ax.imshow(crop(files[j], lon[i], lat[i]), cmap="gray", vmin=0, vmax=np.percentile(crop(files[j], lon[i], lat[i]), 99.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"駐機位置 {i}（{lat[i]:.4f}, {lon[i]:.4f}）{lab} {names[j][:10]}: {tab[i, j]:+.1f} dB（±60 m）", fontsize=9)
    fig.suptitle("est_04: 駐機位置の明るさが日付ごとに「強い日」「弱い日」に分かれるか（飛行機がいる・いないの仮の正解の候補）", fontsize=11)
    fig.text(0.01, 0.005, "SAR の画像 © Umbra Lab, Inc.（CC BY 4.0）。駐機位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(ROOT, "docs", "images", f"est04_stands_{a.airport}.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
