"""est_05 の図: 熊本空港の周りの PALSAR-2（L バンド、U2）。2026-07-29 と 2022-12-07 の明るさ、位相のそろい具合、明るさの差（docs/images/est05_kumamoto.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est05.py
est_05 が cache/ に残した切り出し（2 画素ごとに間引き）を使う。画像はレーダの座標（行 = 進行方向、列 = 距離の向き）のまま。
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
    rec = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est05.txt"), encoding="utf-8")][-1]
    D = np.load(data_dir() / "cache" / "est05_kumamoto.npz")
    dA, dB, coh = D["dA"], D["dB"], D["coh"]
    lo, hi = np.percentile(dA, [2, 99.5])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.3))
    for ax, img, t, kw in [(axes[0], dA, "2026-07-29（地震の翌日）の明るさ（dB）", dict(cmap="gray", vmin=lo, vmax=hi)),
                           (axes[1], dB, "2022-12-07 の明るさ（dB）", dict(cmap="gray", vmin=lo, vmax=hi)),
                           (axes[2], coh, f"2 日付の位相のそろい具合（中央値 {rec['coherence_median']:.2f}）", dict(cmap="viridis", vmin=0, vmax=1)),
                           (axes[3], dA - dB, "明るさの差（2026 − 2022、dB）", dict(cmap="RdBu_r", vmin=-8, vmax=8))]:
        im = ax.imshow(img, **kw)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("est_05: 熊本空港の周り（約 4 km 四方）を ALOS-2 PALSAR-2（L バンド、細かい撮り方 U2、南行）で見る。2 日付はデータの相関だけで重ねた", fontsize=11)
    fig.text(0.01, 0.01, "提供：JAXA（ALOS-2 PALSAR-2 オープン&フリー、熊本地震の公開データ、非商用の研究）。画像はレーダの座標のまま（北が上ではない）", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(ROOT, "docs", "images", "est05_kumamoto.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
