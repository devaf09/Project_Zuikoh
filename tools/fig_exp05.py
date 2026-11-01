"""実験05 の図: 2 本の光路の融合（振幅 + 干渉の位相）と、1 本の光路（板 8 枚・16 枚）、CNN を比べる（docs/images/exp05_fuse.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_exp05.py
results/exp04s.txt（干渉データ・振幅を平方根に、板 8 枚）と results/exp05.txt（融合と板 16 枚）を使う。3 seed の平均 ± 標準誤差。
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


def load(name):
    return [json.loads(x) for x in open(os.path.join(ROOT, "results", name), encoding="utf-8")]


def main():
    a, b = load("exp04s.txt"), load("exp05.txt")
    rows = [("小さな CNN（振幅）", [r for r in a if r["model"] == "cnn_amp"]),
            ("小さな CNN（複素数）", [r for r in a if r["model"] == "cnn"]),
            ("Sin の板（振幅、8 枚）", [r for r in a if r["model"] == "sin_amp"]),
            ("Sin の板（複素数、8 枚）", [r for r in a if r["model"] == "sin"]),
            ("Sin の板（振幅、16 枚）", [r for r in b if r["model"] == "sin_amp" and r["plates"] == 16]),
            ("Sin の板（複素数、16 枚）", [r for r in b if r["model"] == "sin" and r["plates"] == 16]),
            ("2 本の光路の融合（振幅 8 枚 + 複素数 8 枚）", [r for r in b if r["model"] == "fuse"])]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(rows))
    for j, (key, lab, col) in enumerate([("acc", "正解率（3 種類）", "#1b9e77"), ("cr_recall", "反射板の再現率", "#d95f02"),
                                         ("cr_precision", "反射板の適合率", "#7570b3")]):
        mu = [np.mean([r[key] for r in rs]) for _, rs in rows]
        se = [np.std([r[key] for r in rs], ddof=1) / np.sqrt(len(rs)) for _, rs in rows]
        ax.bar(x + (j - 1) * 0.27, mu, 0.27, yerr=se, capsize=3, color=col, label=lab)
    ax.set_xticks(x, [f"{n}\n（学習する数 {rs[0]['params']:,}）" for n, rs in rows], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%（テスト、3 seed の平均 ± 標準誤差）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("実験05: 干渉データ（振幅を平方根に）で反射板を見分ける。反射板ごとに訓練とテストを分ける", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。反射板の位置: NASA/JPL UAVSAR の校正のページ", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(ROOT, "docs", "images", "exp05_fuse.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
