"""実験02 の図: 切り出しの見本（反射板・ほかの明るい点・背景）と、条件ごとの正解率・反射板の再現率（docs/images/exp02_cr_chips.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_exp02.py
results/exp02.txt（3 seed の平均 ± 標準誤差）と、データの置き場の processed/cr_chips/chips.pt を使う。
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
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"scr": "明るさの強さだけ", "feat": "手で作った特徴", "cnn_amp": "小さな CNN（振幅）", "cnn": "小さな CNN（複素数）",
         "sin_amp": "Sin の板（振幅）", "sin": "Sin の板（複素数）",
         "cnn_rc": "CNN（複素数・傾き除去）", "sin_rc": "Sin の板（複素数・傾き除去）"}


def main():
    recs = [json.loads(x) for x in open(os.path.join(ROOT, "results", "exp02.txt"), encoding="utf-8")]
    # 同じ (条件, 分け方, seed) が 2 回あるとき（CPU で途中まで回した分と GPU で回し直した分）は、新しいほうだけを使う
    recs = list({(r["model"], r["split"], r["seed"]): r for r in recs}.values())
    rc = os.path.join(ROOT, "results", "exp02r.txt")             # 位相の傾きを取り除いた確かめ（sweep02r.sh）
    if os.path.exists(rc):
        for x in open(rc, encoding="utf-8"):
            r = json.loads(x)
            r["model"] = r["model"] + "_rc"
            recs.append(r)
    D = torch.load(data_dir() / "processed" / "cr_chips" / "chips.pt", weights_only=False)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.6])
    rng = np.random.default_rng(0)
    for row, (lab, title) in enumerate([(2, "反射板"), (1, "ほかの明るい点")]):
        idx = rng.choice(np.nonzero((D["y"] == lab).numpy())[0], 6, replace=False)
        for j, i in enumerate(idx):
            ax = fig.add_subplot(gs[row, j])
            a = 20 * np.log10(D["X"][i].abs().numpy() + 1e-6)
            ax.imshow(a, cmap="magma", vmin=-40, vmax=0, aspect=13.9 / 2.33 / 3)   # 進行方向の画素は縦に長い（約 14 m × 2.3 m）。見やすく 1/3 に
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title} {D['cr'][i]} {D['date'][i][4:]}\n{D['scr_db'][i]:.0f} dB", fontsize=8)
    for k, split in enumerate(["cr", "date"]):
        ax = fig.add_subplot(gs[2, 3 * k:3 * k + 3])
        models = [m for m in NAMES if any(r["model"] == m and r["split"] == split for r in recs)]
        for j, key in enumerate(["acc", "cr_recall"]):
            mu, se = [], []
            for m in models:
                v = np.array([r[key] for r in recs if r["model"] == m and r["split"] == split])
                mu.append(v.mean())
                se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            x = np.arange(len(models)) + (j - 0.5) * 0.38
            ax.bar(x, mu, 0.38, yerr=se, capsize=3, color=["#1b9e77", "#d95f02"][j],
                   label=["正解率（3 種類）", "反射板の再現率"][j])
        ax.set_xticks(np.arange(len(models)), [NAMES[m] for m in models], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%（テスト、3 seed の平均 ± 標準誤差）")
        ax.set_title("反射板ごとに分ける（テストの反射板は訓練で見ない）" if split == "cr" else "日付で分ける（新しい日付をテストに）", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("実験02: オクラホマの角の反射板を見分ける（上 2 段: 切り出しの見本、明るさは dB、いちばん明るい画素 = 0 dB）", fontsize=11)
    fig.text(0.01, 0.005, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。反射板の位置: NASA/JPL UAVSAR の校正のページ", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(ROOT, "docs", "images", "exp02_cr_chips.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
