"""est_06b の図: ロズウェルの 200 m の格子ごとの位相のそろい具合（3 組の平均）と、飛行機ありの格子（docs/images/est06b_storage_coherence.png）

実行: .venv/Scripts/python tools/fig_est06b.py --site roswell
est_06b が cache/ に残した格子の値と、NAIP の航空写真（参考、撮影日は SAR とずれる）を使う。
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

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="roswell")
    a = ap.parse_args()
    d = data_dir()
    T = np.loadtxt(d / "cache" / f"est06b_{a.site}_cells.csv", delimiter=",")
    recs = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est06b.txt"), encoding="utf-8") if f'"site": "{a.site}"' in x]
    summ = [r for r in recs if r.get("summary")][-1]
    pairs = [r for r in recs if not r.get("summary")][-3:]
    with rasterio.open(d / "processed" / "naip" / f"{a.site}.tif") as s:
        img = s.read(out_shape=(3, s.height // 10, s.width // 10)).transpose(1, 2, 0)
    G = np.full((50, 50), np.nan)
    for row in T:
        G[int(row[1]), int(row[0])] = row[3:].mean()
    x0, x1 = int(T[:, 0].min()), int(T[:, 0].max()) + 1
    y0, y1 = int(T[:, 1].min()), int(T[:, 1].max()) + 1
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    axes[0].imshow(img, extent=[0, 50, 50, 0])
    im = axes[1].imshow(G, cmap="viridis", vmin=0.2, vmax=0.5, extent=[0, 50, 50, 0])
    fig.colorbar(im, ax=axes[1], shrink=0.75, label="位相のそろい具合（3 組の平均）")
    for ax in axes[:2]:
        for row in T[T[:, 2] == 1]:
            ax.add_patch(plt.Rectangle((row[0], row[1]), 1, 1, fill=False, ec="#ff7f0e", lw=0.7))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xlabel("格子（200 m）")
    axes[0].set_title("航空写真（USDA NAIP、参考）と飛行機ありの格子（橙）", fontsize=10)
    axes[1].set_title(f"位相のそろい具合（12 日おき 3 組の平均）、AUC {summ['auc_of_mean']:.2f}", fontsize=10)
    ax = axes[2]
    lab = T[:, 2] == 1
    m = T[:, 3:].mean(1)
    ax.hist(m[~lab], bins=30, alpha=0.6, density=True, color="#7570b3", label=f"その他の格子 {(~lab).sum()}")
    ax.hist(m[lab], bins=30, alpha=0.6, density=True, color="#d95f02", label=f"飛行機ありの格子 {lab.sum()}")
    ax.set_xlabel("位相のそろい具合（3 組の平均）")
    ax.set_ylabel("割合（密度）")
    ax.set_title("格子ごとの分布: " + "・".join(f"{r['pair'][4:8]}→{r['pair'][13:17]} {r['auc_aircraft_high']:.2f}" for r in pairs), fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"est_06b: 飛行機の保管場（{a.site}）の格子ごとの位相のそろい具合。Sentinel-1C の SLC 4 日付、データの相関だけで重ねた", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。正解の格子は Claude が USDA NAIP の航空写真を見て印をつけたもの", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "est06b_storage_coherence.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
