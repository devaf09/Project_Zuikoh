"""est_06 の図: ダラス・フォートワース空港の位相のそろい具合（5 組の平均）と、OSM の地物の種類ごとの明るさ・そろい具合（docs/images/est06_dfw.png）

位置合わせは、OSM の滑走路を暗い帯に重ねる探し方で決めたずれ（行 −40・列 −20）を使う（RTC との相関は山が弱く、決まらなかった）。
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est06.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rasterio.transform import GCPTransformer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_06_airport_coherence import find_window, osm_points  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
SHIFT = (-40, -20)
NAMES = {"parking_position": "駐機位置", "apron": "駐機場", "taxiway": "誘導路", "runway": "滑走路", "terminal": "ターミナル"}


def main():
    d = data_dir()
    D = np.load(d / "cache" / "est06_dfw.npz")
    cohs, amp, (r0, r1, c0, c1), dates = D["cohs"], D["amp"], D["win"], D["dates"]
    z = sorted((d / "raw" / "s1_slc" / "dfw").glob("*.zip"), key=lambda p: p.name[17:25])[0]
    tr = GCPTransformer(find_window(z)[1])
    osm = json.load(open(d / "raw" / "osm" / "airports" / "dfw.geojson", encoding="utf-8"))["features"]
    P = osm_points(osm, list(NAMES))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    axes[0].imshow(amp, cmap="gray", vmin=np.percentile(amp, 5), vmax=np.percentile(amp, 99.5), aspect="auto")
    axes[1].imshow(cohs.mean(0), cmap="viridis", vmin=0.1, vmax=0.8, aspect="auto")
    stats = {}
    for k, col in zip(NAMES, ["#e7298a", "#ff7f0e", "#1b9e77", "#d95f02", "#7570b3"]):
        r, c = tr.rowcol(P[k][:, 0], P[k][:, 1])
        r = np.round(np.array(r) - r0 + SHIFT[0]).astype(int)
        c = np.round((np.array(c) - c0 + SHIFT[1]) / 4).astype(int)
        ok = (r >= 0) & (r < cohs.shape[1]) & (c >= 0) & (c < cohs.shape[2])
        stats[k] = (amp[r[ok], c[ok]].mean(), cohs[:, r[ok], c[ok]].mean(1))
        if k in ("runway", "terminal", "parking_position"):
            for ax in axes[:2]:
                ax.plot(c[ok], r[ok], ".", ms=0.6, color=col, label=NAMES[k])
    axes[0].set_title("明るさ（dB、2026-08-25）と OSM の地物（位置合わせ後）", fontsize=10)
    axes[1].set_title("位相のそろい具合（12 日おき 5 組の平均）", fontsize=10)
    axes[0].legend(fontsize=8, markerscale=15, loc="lower right")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[2]
    for k in NAMES:
        ax.plot(range(len(dates) - 1), stats[k][1], "o-", label=f"{NAMES[k]}（明るさ {stats[k][0]:.1f} dB）")
    ax.set_xticks(range(len(dates) - 1), [f"{dates[i][4:]}→{dates[i + 1][4:]}" for i in range(len(dates) - 1)], fontsize=8, rotation=20)
    ax.set_ylabel("位相のそろい具合（平均）")
    ax.set_title("地物の種類ごとのそろい具合の時系列", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("est_06: ダラス・フォートワース空港の位相のそろい具合（Sentinel-1D、軌道 34、12 日おき 6 日付）。データの相関だけで重ねた", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。空港の地物 © OpenStreetMap contributors（ODbL）。画像はレーダの座標のまま", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "est06_dfw.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
