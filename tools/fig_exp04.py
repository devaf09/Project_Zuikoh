"""実験04 の図: 1 日付の SLC と 2 日付の干渉データで、反射板を見分ける正解率を比べる（docs/images/exp04_ifg_chips.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_exp04.py
results/exp02.txt（1 日付、反射板ごとの分け方）と results/exp04.txt（干渉データ）、processed/cr_chips/ifg_chips.pt を使う。
上の段: 干渉データの切り出しの見本（色 = 位相、明るさ = 振幅）。反射板は位相がそろい（色が一様）、背景はでたらめ。
"""
from __future__ import annotations

import colorsys
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
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"cnn_amp": "小さな CNN（振幅）", "cnn": "小さな CNN（複素数）", "sin_amp": "Sin の板（振幅）", "sin": "Sin の板（複素数）"}


def phase_rgb(z):
    """複素数 → 色（位相を色相、振幅の dB を明るさ）。"""
    a = np.clip((20 * np.log10(np.abs(z) + 1e-6) + 30) / 30, 0, 1)
    h = (np.angle(z) + np.pi) / (2 * np.pi)
    return np.array([[colorsys.hsv_to_rgb(h[i, j], 0.9, a[i, j]) for j in range(z.shape[1])] for i in range(z.shape[0])])


def stats(path, split="cr"):
    recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    recs = list({(r["model"], r["split"], r["seed"]): r for r in recs if r["split"] == split}.values())
    out = {}
    for m in NAMES:
        v = np.array([[r["acc"], r["cr_recall"], r["cr_precision"]] for r in recs if r["model"] == m])
        if len(v):
            out[m] = (v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(3))
    return out


def main():
    D = torch.load(data_dir() / "processed" / "cr_chips" / "ifg_chips.pt", weights_only=False)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.5])
    rng = np.random.default_rng(1)
    k = 0
    for lab, title in [(2, "反射板"), (1, "ほかの明るい点"), (0, "背景")]:
        for i in rng.choice(np.nonzero((D["y"] == lab).numpy())[0], 2, replace=False):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(phase_rgb(D["X"][i].numpy()[8:40, 8:40]), aspect=13.9 / 2.33 / 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title} {D['cr'][i]}\n{D['date'][i][4:8]}→{D['date'][i][13:17]}", fontsize=8)
            k += 1
    s1, s2 = stats(os.path.join(ROOT, "results", "exp02.txt")), stats(os.path.join(ROOT, "results", "exp04.txt"))
    for j, (key, idx) in enumerate([("正解率（3 種類）", 0), ("反射板の適合率（反射板と答えたうち本物の割合）", 2)]):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        x = np.arange(len(NAMES))
        for o, (st, lab, col) in enumerate([(s1, "1 日付の SLC（実験02）", "#7570b3"), (s2, "2 日付の干渉データ（実験04）", "#d95f02")]):
            mu = [st[m][0][idx] if m in st else np.nan for m in NAMES]
            se = [st[m][1][idx] if m in st else 0 for m in NAMES]
            ax.bar(x + (o - 0.5) * 0.38, mu, 0.38, yerr=se, capsize=3, color=col, label=lab)
        ax.set_xticks(x, list(NAMES.values()), rotation=15, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%（テスト、3 seed の平均 ± 標準誤差）")
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("実験04: 2 日付の掛け算（干渉データ）で斑点のでたらめな位相を打ち消すと、Sin の板（複素数）は良くなるか（反射板ごとに訓練とテストを分ける）", fontsize=11)
    fig.text(0.01, 0.005, "上の段: 色 = 干渉の位相、明るさ = 振幅（dB）。Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。反射板の位置: NASA/JPL UAVSAR の校正のページ", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(ROOT, "docs", "images", "exp04_ifg_chips.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
