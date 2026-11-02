"""実験06（改良）の図: 場所の入れ替え・日付どうしの変化のあり・なし・条件ごとの AUC（docs/images/exp06b_auc.png）

実行: .venv/Scripts/python tools/fig_exp06b.py
results/exp06.txt（3 seed の平均 ± 標準誤差）を使う。左: 飛行機（訓練していない場所）、右: 車（ロングビーチの新しい日付）。
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"feat": "手で作った特徴", "cnn": "小さな CNN", "sin_amp": "Sin の板（明るさの強さを残す）"}


def main():
    R = [json.loads(x) for x in open(os.path.join(ROOT, "results", "exp06.txt"), encoding="utf-8")]
    conds = [("victorville", False), ("roswell", False), ("victorville", True), ("roswell", True)]
    labels = ["試験 = ビクタービル", "試験 = ロズウェル", "ビクタービル + 日付の変化", "ロズウェル + 日付の変化"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "auc_aircraft_vcv", "飛行機の格子（訓練していない場所）の AUC"),
                           (axes[1], "auc_car_lb", "車の格子（ロングビーチの新しい 3 日付）の AUC")]:
        x = np.arange(len(conds))
        for j, (m, col) in enumerate(zip(NAMES, ["#7570b3", "#1b9e77", "#d95f02"])):
            mu, se = [], []
            for site, t in conds:
                v = np.array([r[key] for r in R if r["model"] == m and r["test_site"] == site and r["temporal"] == t])
                mu.append(v.mean())
                se.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax.bar(x + (j - 1) * 0.27, mu, 0.27, yerr=se, capsize=3, color=col, label=NAMES[m])
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.text(-0.4, 0.51, "偶然", fontsize=8, color="gray")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("AUC（3 seed の平均 ± 標準誤差）")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("実験06（改良）: 200 m の格子ごとの飛行機・車の成分。日付の変化 = 同じ格子の全日付の VV のばらつきを足したもの", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（Microsoft Planetary Computer, sentinel-1-rtc, CC BY 4.0）。正解は Claude が USDA NAIP の航空写真を見て印をつけたもの", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "exp06b_auc.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
