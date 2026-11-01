"""実験06 の図: 見たことのない場所（ビクタービル）の 200 m の格子ごとの「飛行機の成分」の確率の地図と、正解の格子（docs/images/exp06_cells.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_exp06.py
results/exp06.txt と、cache/exp06_<条件>_<seed>.pt（日付を平均した確率）を使う。左: NAIP の航空写真と正解の格子、中・右: 条件ごとの確率の地図。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_cell_chips import load_cells  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"feat": "手で作った特徴", "cnn": "小さな CNN", "sin_amp": "Sin の板（振幅）"}


def prob_map(model, n=50):
    """3 seed・全日付を平均した、ビクタービルの格子ごとの飛行機の確率（無い格子は NaN）。"""
    acc, cnt = np.zeros((n, n)), np.zeros((n, n))
    for s in range(3):
        p = data_dir() / "cache" / f"exp06_{model}_{s}.pt"
        if not p.exists():
            continue
        D = torch.load(p, weights_only=False)
        for pr, site, x, y in zip(D["prob"][:, 2].numpy(), D["site"], D["cx"].numpy(), D["cy"].numpy()):
            if site == "victorville":
                acc[y, x] += pr
                cnt[y, x] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def main():
    recs = [json.loads(x) for x in open(os.path.join(ROOT, "results", "exp06.txt"), encoding="utf-8")]
    air = load_cells(os.path.join(ROOT, "data_labels", "victorville_aircraft_cells.csv"))
    with rasterio.open(data_dir() / "processed" / "naip" / "victorville.tif") as s:
        img = s.read(out_shape=(3, s.height // 10, s.width // 10)).transpose(1, 2, 0)
    models = [m for m in NAMES if any(r["model"] == m for r in recs)]
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(5 * (1 + len(models)), 5.6))
    ax = axes[0]
    ax.imshow(img, extent=[0, 50, 50, 0])
    for x, y in air:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, ec="#ff7f0e", lw=0.8))
    ax.set_title("NAIP の航空写真と、飛行機ありの格子（橙）", fontsize=10)
    ax.set_xlim(12, 42)
    ax.set_ylim(40, 8)
    for ax, m in zip(axes[1:], models):
        P = prob_map(m)
        im = ax.imshow(P, cmap="magma", vmin=0, vmax=1, extent=[0, 50, 50, 0])
        for x, y in air:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, ec="#1b9e77", lw=0.5))
        rs = [r for r in recs if r["model"] == m]
        a_ = np.array([r["auc_aircraft_vcv"] for r in rs])
        c_ = np.array([r["auc_car_lb"] for r in rs])
        ax.set_title(f"{NAMES[m]}: 飛行機の確率（日付・3 seed の平均）\n飛行機の AUC {a_.mean():.2f}±{a_.std(ddof=1) / np.sqrt(len(a_)):.2f}"
                     f"・車の AUC（ロングビーチ）{c_.mean():.2f}", fontsize=9)
        ax.set_xlim(12, 42)
        ax.set_ylim(40, 8)
        fig.colorbar(im, ax=ax, shrink=0.7, label="飛行機の確率")
    for ax in axes:
        ax.set_xlabel("格子（200 m）")
    fig.suptitle("実験06: 訓練していない場所（ビクタービル）の 200 m の格子ごとの「飛行機の成分」（訓練 = ロズウェルの飛行機 + ロングビーチの車）", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（Microsoft Planetary Computer, sentinel-1-rtc, CC BY 4.0）。航空写真: USDA NAIP。"
             "正解の格子は Claude が航空写真を見て印をつけたもの（撮影日は SAR とずれる）", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "exp06_cells.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
