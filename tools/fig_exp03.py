"""実験03 の図: 切り出しの真ん中を隠して、周りだけで反射板を当てる（docs/images/exp03_mask_center.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_exp03.py
results/exp03.txt（3 seed の平均 ± 標準誤差）と、processed/cr_chips/chips.pt を使う。
上の段: 反射板の切り出しを、隠す半径ごとに見せる（隠したあとに残る最大の明るさ = 元の山との比 dB を添える）。
下の段: 隠す半径ごとの反射板の再現率と適合率（分解能そのまま / 1/4）。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp02_cr_chips as e  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"feat": "手で作った特徴", "cnn": "小さな CNN（複素数）", "sin": "Sin の板（複素数）"}
COL = {"feat": "#7570b3", "cnn": "#1b9e77", "sin": "#d95f02"}


def main():
    recs = [json.loads(x) for x in open(os.path.join(ROOT, "results", "exp03.txt"), encoding="utf-8")]
    recs = list({(r["model"], r["mask"], r["lowpass"], r["seed"]): r for r in recs}.values())
    D = torch.load(data_dir() / "processed" / "cr_chips" / "chips.pt", weights_only=False)
    X = e.crop(D["X"][D["y"] == 2])
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.5])
    k = 0
    for lp in [1, 4]:
        Xl = e.lowpass(X, lp)
        for r in [0, 2, 4, 8]:
            m = e.mask_center(Xl, r) if r else Xl
            rem = 10 * np.log10((m.abs().amax((-2, -1)) ** 2).median().item() + 1e-12)
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(20 * np.log10(m[0].abs().numpy() + 1e-6), cmap="magma", vmin=-40, vmax=0, aspect=13.9 / 2.33 / 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"分解能 1/{lp}・半径 {r}\n残る最大 {rem:.0f} dB", fontsize=8)
            k += 1
    chance = None
    for j, lp in enumerate([1, 4]):
        ax = fig.add_subplot(gs[1, 4 * j:4 * j + 4])
        for model in ["feat", "cnn", "sin"]:
            for key, ls in [("cr_recall", "-"), ("cr_precision", "--")]:
                xs, mu, se = [], [], []
                for r in [2, 4, 8]:
                    v = np.array([q[key] for q in recs if q["model"] == model and q["lowpass"] == lp and q["mask"] == r])
                    if len(v):
                        xs.append(r)
                        mu.append(v.mean())
                        se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                ax.errorbar(xs, mu, yerr=se, color=COL[model], ls=ls, marker="o", capsize=3,
                            label=f"{NAMES[model]} {'再現率' if key == 'cr_recall' else '適合率'}")
            if chance is None:
                cm = np.array([q["confusion"] for q in recs if q["model"] == model][0])
                chance = 100 * cm[2].sum() / cm.sum()
        ax.axhline(chance, color="gray", ls=":", lw=1)
        ax.text(2, chance + 2, f"テストの中の反射板の割合 {chance:.1f}%（でたらめに当てたときの適合率）", fontsize=8, color="gray")
        ax.set_xlabel("隠す半径（距離の向きの画素。進行方向はその 1/6、最低 1 行）")
        ax.set_ylabel("反射板の %（テスト、3 seed の平均 ± 標準誤差）")
        ax.set_ylim(0, 105)
        ax.set_xticks([2, 4, 8])
        ax.set_title("分解能そのまま（約 2.3 m × 14 m）" if lp == 1 else "分解能 1/4（10 m 相当）: 隠しても山が残る（上の段）ので参考", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("実験03: 反射板の真ん中を隠して、周りの成分だけで当てられるか（反射板ごとに訓練とテストを分ける）", fontsize=11)
    fig.text(0.01, 0.005, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。反射板の位置: NASA/JPL UAVSAR の校正のページ", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(ROOT, "docs", "images", "exp03_mask_center.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
