"""est_01d の図: 12 日でタンクの強い点が距離の向きにどれだけ動いたか（タンクと、動かないはずの対照の点）（docs/images/est01d_point_moves.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01d_slc_two_dates.py --seed 0 && .venv/Scripts/python tools/fig_est01d.py
est_01d が cache/ に残した点の表と、results/est01d.txt を使う。対照の点は表に残していないので、ばらつきの値だけを線で描く。
"""
from __future__ import annotations

import json
import math
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
    rec = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est01d.txt"), encoding="utf-8")][-1]
    p = np.loadtxt(data_dir() / "cache" / "est01d_points.csv", delimiter=",")
    move = p[:, 4] - rec["control_move_m_median"]           # 動かない点の中央値を 0 にそろえる
    cos_t = math.cos(math.radians(rec["theta_deg"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(move, bins=np.arange(-8, 8.01, 0.25), color="#d95f02", alpha=0.85,
            label=f"タンクの強い点 {rec['n_points']} 個（{rec['n_tanks_with_points']} 基）")
    sd = rec["control_move_m_robust_sd"]
    for k, ls in [(1, "-"), (3, "--")]:
        ax.axvline(k * sd, color="#1b9e77", ls=ls, lw=1)
        ax.axvline(-k * sd, color="#1b9e77", ls=ls, lw=1)
    ax.plot([], [], color="#1b9e77", label=f"動かないはずの対照の点 {rec['n_control']} 個のばらつき ±{sd:.2f} m（実線）・±3 倍（破線）")
    sec = ax.secondary_xaxis("top", functions=(lambda m: m / cos_t, lambda h: h * cos_t))
    sec.set_xlabel("屋根の高さの変化に直すと（m、屋根と壁の角の点だった場合）")
    ax.set_xlabel(f"12 日での距離の向きの動き（斜めの距離 m、{rec['date_a']} → {rec['date_b']}。負 = 衛星に近づいた）")
    ax.set_ylabel("点の数")
    ax.plot([], [], " ", label=f"|動き| > 1 m の割合: タンク {rec['tank_frac_abs_gt_1m']:.0%}、対照 {rec['control_frac_abs_gt_1m']:.0%}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。タンクの位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(ROOT, "docs", "images", "est01d_point_moves.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
