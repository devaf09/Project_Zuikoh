"""est_01e の図: 6 日付で「動く点」の割合（手前・奥・対照）と、点ごとの位置のばらつきの分布（docs/images/est01e_near_far.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est01e.py
results/est01e.txt の 2 行（探す幅 ±70 m と、タンクの半径に絞ったもの）と、cache/ の点の表を使う。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def main():
    recs = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est01e.txt"), encoding="utf-8")]
    wide = [r for r in recs if not r.get("own_radius")][-1]
    own = [r for r in recs if r.get("own_radius")][-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    labels = ["対照（動かないはずの点）", "タンクの手前の半分", "タンクの奥の半分"]
    keys = ["frac_moving_control", "frac_moving_near", "frac_moving_far"]
    x = np.arange(3)
    ax.bar(x - 0.18, [wide[k] for k in keys], 0.36, color="#7570b3", label="探す幅 ±70 m（となりのタンクの点も入る）")
    ax.bar(x + 0.18, [own[k] for k in keys], 0.36, color="#d95f02", label="探す幅 = そのタンクの半径")
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"動く点の割合（位置のばらつき > {own['move_threshold_m']:.2f} m = 対照の 3 倍）")
    ax.set_title("見立て「手前は動かず、奥だけ動く」は、この測り方では確かめられなかった", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    p = np.loadtxt(data_dir() / "cache" / "est01e_points_own.csv", delimiter=",")
    tank, off, sd = p[:, 0], p[:, 4], p[:, 5]
    bins = np.logspace(-1.5, 1.2, 40)
    axes[1].hist(sd[tank < 0], bins=bins, histtype="step", lw=1.5, color="#1b9e77", label="対照", density=True)
    axes[1].hist(sd[(tank >= 0) & (off < 0)], bins=bins, histtype="step", lw=1.5, color="#e7298a", label="タンクの手前", density=True)
    axes[1].hist(sd[(tank >= 0) & (off > 0)], bins=bins, histtype="step", lw=1.5, color="#d95f02", label="タンクの奥", density=True)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("6 日付での位置のばらつき（標準偏差、斜めの距離 m）")
    axes[1].set_ylabel("割合（密度）")
    axes[1].set_title("点ごとの位置のばらつき（探す幅 = タンクの半径）", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"est_01e: クッシングの SLC 6 日付（{own['dates'][0]}〜{own['dates'][-1]}、12 日おき）", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。タンクの位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(ROOT, "docs", "images", "est01e_near_far.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
