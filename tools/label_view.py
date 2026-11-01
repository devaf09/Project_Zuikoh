"""格子の正解に印をつけるための下絵を作る（NAIP の航空写真に 200 m の格子と番号を重ねる）。Claude が目で見て印をつける作業用。

実行: .venv/Scripts/python tools/label_view.py --name roswell --x0 0 --y0 0 --n 10   （格子の番号 x0〜x0+n−1、y0〜y0+n−1 を拡大）
       --overview なら全体（50 × 50 の格子、粗く）
出力: <データの置き場>/cache/label/<名前>_<...>.png（公開しない作業用。航空写真は出典「USDA NAIP」）
格子の番号: 列 x = 西から、行 y = 北から、1 つ 200 m（2 m の画素で 100 × 100）
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

CELL = 100   # 200 m / 2 m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--overview", action="store_true")
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    with rasterio.open(data_dir() / "processed" / "naip" / f"{a.name}.tif") as s:
        if a.overview:
            img = s.read(out_shape=(3, s.height // 4, s.width // 4)).transpose(1, 2, 0)
            step, x0, y0, n = CELL / 4, 0, 0, s.width // CELL
        else:
            win = ((a.y0 * CELL, (a.y0 + a.n) * CELL), (a.x0 * CELL, (a.x0 + a.n) * CELL))
            img = s.read(window=win).transpose(1, 2, 0)
            step, x0, y0, n = CELL, a.x0, a.y0, a.n
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    for k in range(n + 1):
        ax.axhline(k * step, color="yellow", lw=0.4, alpha=0.6)
        ax.axvline(k * step, color="yellow", lw=0.4, alpha=0.6)
    lab_every = 5 if a.overview else 1
    for i in range(0, n, lab_every):
        ax.text(i * step + 1, -2, str(x0 + i), color="black", fontsize=7)
        ax.text(-2, i * step + step / 2, str(y0 + i), color="black", fontsize=7, ha="right")
    ax.set_xticks([])
    ax.set_yticks([])
    out = data_dir() / "cache" / "label" / (f"{a.name}_overview.png" if a.overview else f"{a.name}_{a.x0}_{a.y0}_{a.n}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
