"""est_03 の図: 反射板と周りの点の位相の縞（点どうしの関係の周波数成分）と、その日付ごとの変化（docs/images/est03_point_relations.png）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_est03.py --name cushing_town（反射板なら --name N08K）
est_03 が cache/ に残した重ねた SLC と点の一覧、results/est03.txt を使う。
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
LAM = 0.05547


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cushing_town", help="results/est03.txt の cr の名前（反射板の番号か場所の名前）")
    name = ap.parse_args().name
    allr = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est03.txt"), encoding="utf-8")]
    allr = [r for r in allr if r["cr"] == name]
    summ = [r for r in allr if r.get("summary")][-1]
    recs = [r for r in allr if not r.get("summary")][-(len(allr) and 5):]
    D = np.load(data_dir() / "cache" / f"est03_{recs[0]['cr']}.npz")
    S, rr, cc, x, y, dates = D["S"], D["rr"], D["cc"], D["x"], D["y"], D["dates"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    a = 10 * np.log10(np.abs(S[0]) ** 2 / np.median(np.abs(S[0]) ** 2) + 1e-6)
    ax.imshow(a, cmap="gray", vmin=-10, vmax=25, aspect=13.9 / 2.33 / 3,
              extent=[(-300) * 2.33, 300 * 2.33, 60 * 13.9, -60 * 13.9])
    ax.scatter(x, y, s=18, facecolors="none", edgecolors="#ff7f0e", label=f"選んだ点 {len(x)} 個（動かない点の候補）")
    ax.scatter([0], [0], s=80, marker="*", color="#e7298a", label="基準の点" + (f"（反射板 {name}）" if name.startswith("N") else ""))
    ax.set_xlabel("距離の向き（m、基準の点から）")
    ax.set_ylabel("進行方向（m）")
    ax.set_title(f"基準の日付 {dates[0]} の明るさ（dB）と選んだ点", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    # 1 組の例: 位相 vs 距離の向きの位置と、見つけた縞
    t = int(np.argmax([r["coherence_after"] - r["coherence_before"] for r in recs])) + 1
    r = recs[t - 1]
    ph = np.angle(S[t, rr, cc] * np.conj(S[0, rr, cc]))
    kx, ky = 2 * np.pi * r["fringe_range_per_km"] / 1000, 2 * np.pi * r["fringe_azimuth_per_km"] / 1000
    ref = np.angle(np.exp(1j * (ph - kx * x - ky * y)))
    cr_res = ref[(x == 0) & (y == 0)][0]
    ax = axes[0, 1]
    ax.scatter(x, np.angle(np.exp(1j * (ph - ky * y - cr_res))), s=20, color="#7570b3", label="点ごとの位相の差（進行方向の縞は引いた後）")
    xs = np.linspace(x.min(), x.max(), 400)
    ax.plot(xs, np.angle(np.exp(1j * kx * xs)), color="#d95f02", lw=1, label=f"見つけた縞 {r['fringe_range_per_km']:+.2f} 本/km")
    ax.set_xlabel("距離の向き（m、基準の点から）")
    ax.set_ylabel("位相の差（ラジアン、−π〜π）")
    ax.set_title(f"例: {dates[0]} → {dates[t]} の点どうしの位相の差", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    lab = [r["date"][4:] for r in recs]
    ax.bar(np.arange(len(recs)) - 0.2, [r["fringe_range_per_km"] for r in recs], 0.4, color="#d95f02", label="距離の向き")
    ax.bar(np.arange(len(recs)) + 0.2, [r["fringe_azimuth_per_km"] for r in recs], 0.4, color="#1b9e77", label="進行方向")
    ax.set_xticks(np.arange(len(recs)), lab)
    ax.set_xlabel(f"日付（基準 {dates[0]} との組）")
    ax.set_ylabel("縞の細かさ（1 km あたりの本数）")
    ax.set_title(f"点どうしの関係の周波数成分（縞）は日付の組ごとに変わる（閉じ具合の差の中央値 {summ['closure_abs_median_per_km']:.2f} 本/km）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1, 1]
    ax.plot(lab, [r["coherence_before"] for r in recs], "o-", color="#7570b3", label="縞を引く前")
    ax.plot(lab, [r["coherence_after"] for r in recs], "o-", color="#d95f02", label="縞を引いた後")
    ax2 = ax.twinx()
    ax2.bar(lab, [r["residual_mm_sd"] for r in recs], alpha=0.25, color="gray")
    ax2.axhline(LAM / (4 * np.pi) * np.pi / np.sqrt(3) * 1000, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("残りのばらつき（mm、灰の棒）。破線 = でたらめの位相のとき")
    ax.axhline(summ["null_coherence_p95"], color="#7570b3", ls=":", lw=1)
    ax.text(0, summ["null_coherence_p95"] + 0.02, f"偶然の水準（位相をでたらめにしたときの 95%）{summ['null_coherence_p95']:.2f}", fontsize=8, color="#7570b3")
    ax.set_ylim(0, 1)
    ax.set_ylabel("点のそろい具合（1 = 完全にそろう）")
    ax.set_title("縞を引くと点がそろう = 縞は点の並び全体に共通の成分", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"est_03: {name} の周り（約 1.3 km × 1.7 km）の動かない点 {summ['n_points']} 個どうしの位相の関係（Sentinel-1D、上昇軌道 34、12 日おき）", fontsize=11)
    fig.text(0.01, 0.005, "Contains modified Copernicus Sentinel data 2026（ASF DAAC から取得）。反射板の位置: NASA/JPL UAVSAR の校正のページ", fontsize=7)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(ROOT, "docs", "images", f"est03_point_relations_{name}.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
