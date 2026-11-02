"""est_05 の図（その 2）: 熊本空港のターミナル・駐機場の辺りを、PALSAR-2 の細かい画素のまま 2 日付で並べる（docs/images/est05b_kumamoto_apron.png）

明るい点が最も密集している所（ターミナル・駐機場の候補）の周り 700 × 700 画素（約 1.4 km × 1.1 km）を、2026-07-29 と 2022-12-07 で並べ、明るさの差も出す。
2 日付の重ね方は est_05 と同じ（データの相関で 2 段）。画像はレーダの座標のまま（北が上ではない）。
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est05b.py --row 5896 --col 20136
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
from scipy import ndimage  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import coherence, subpixel_shift  # noqa: E402
from est_03_point_relations import fourier_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.palsar2 import read_window  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=5896)
    ap.add_argument("--col", type=int, default=20136)
    ap.add_argument("--half", type=int, default=350)
    a = ap.parse_args()
    P = data_dir() / "processed" / "palsar2" / "kumamoto"
    rec = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est05.txt"), encoding="utf-8")][-1]
    dr0, dc0 = rec["shift_coarse"]
    r0, r1, c0, c1 = a.row - a.half, a.row + a.half, a.col - a.half, a.col + a.half
    A = read_window(P / "IMG-HH-ALOS2657822920-260729-UBSL1.1__D", r0, r1, c0, c1)
    B = read_window(P / "IMG-HH-ALOS2461172920-221207-UBSL1.1__D", r0 + dr0, r1 + dr0, c0 + dc0, c1 + dc0)
    dr, dc = subpixel_shift(np.log1p(np.abs(A) ** 2), np.log1p(np.abs(B) ** 2))
    B = fourier_shift(B, -dr, -dc)
    dA = 10 * np.log10(ndimage.uniform_filter(np.abs(A) ** 2, 2) + 1e-9)
    dB = 10 * np.log10(ndimage.uniform_filter(np.abs(B) ** 2, 2) + 1e-9)
    coh = coherence(A, B, win=(5, 5))
    lo, hi = np.percentile(dA, [2, 99.7])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    for ax, img, t, kw in [(axes[0], dA, "2026-07-29 の明るさ（dB）", dict(cmap="gray", vmin=lo, vmax=hi)),
                           (axes[1], dB, "2022-12-07 の明るさ（dB）", dict(cmap="gray", vmin=lo, vmax=hi)),
                           (axes[2], dA - dB, "明るさの差（2026 − 2022、dB）: 赤 = 増えた・青 = 減った", dict(cmap="RdBu_r", vmin=-12, vmax=12)),
                           (axes[3], coh, "位相のそろい具合（5 × 5）", dict(cmap="viridis", vmin=0, vmax=1))]:
        im = ax.imshow(img, **kw)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"est_05（その 2）: 熊本空港のターミナル・駐機場の辺り（{2 * a.half} × {2 * a.half} 画素、約 1.4 km × 1.1 km）。PALSAR-2 L バンド U2、南行", fontsize=11)
    fig.text(0.01, 0.01, "提供：JAXA（ALOS-2 PALSAR-2 オープン&フリー、熊本地震の公開データ、非商用の研究）。画像はレーダの座標のまま（北が上ではない）", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "est05b_kumamoto_apron.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
