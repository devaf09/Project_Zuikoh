"""est_05: 熊本空港の周りを PALSAR-2（L バンド、細かい撮り方 U2 の複素数）で見る — 明るさ・2 日付の位相のそろい具合・変化

着想: 持ち主の提案（2026-11-02）の JAXA のオープン&フリーのデータ（熊本地震、非商用の研究、出所「提供：JAXA」）。
      Sentinel-1（C バンド、2.3 m × 14 m）より細かい撮り方（U2、画素 約 2 m）・長い波長（L バンド 約 24 cm）で、空港と町がどう見えるか。
      持ち主の方針: 軌道の情報は使わず、データどうしの相関だけで 2 日付を重ねる。
方法:
    データ: ALOS-2 PALSAR-2 L1.1、南行の軌道 29・枠 2920、2022-12-07 と 2026-07-29（約 3.6 年おき、地震の翌日）
    1. 四隅の経度・緯度（付属の KML）の双一次で、熊本空港のおおよその位置を画像の行・列に直す（粗い。滑走路が写る所に合わせ直す）
    2. 空港の周り（約 4 km 四方）を両方の日付で読み、明るさの相関で 2 段（粗い全体像 → 細かい窓）で重ねる（周波数の側でずらす、位相を保つ）
    3. 明るさ（dB）、2 日付の位相のそろい具合（コヒーレンス、7 × 7 の窓）、明るさの差（dB）を出す
指標: 重ねたずれ、コヒーレンスの分布（滑走路・建物・草地など）
結果: results/est05.txt（1 行 JSON）、図は tools/fig_est05.py、切り出しは cache/est05_kumamoto.npz
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_05_kumamoto_palsar2.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import coherence, subpixel_shift  # noqa: E402
from est_03_point_relations import fourier_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.palsar2 import header, read_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="est_05: 熊本空港の周りを PALSAR-2 で見る")
    ap.add_argument("--row", type=int, default=5824, help="切り出しの中心の行（全体像で滑走路が写る所に合わせた値）")
    ap.add_argument("--col", type=int, default=19840)
    ap.add_argument("--half", type=int, default=1200, help="切り出しの半分の大きさ（画素）")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est05.txt"))
    a = ap.parse_args()
    P = data_dir() / "processed" / "palsar2" / "kumamoto"
    new = P / "IMG-HH-ALOS2657822920-260729-UBSL1.1__D"
    old = P / "IMG-HH-ALOS2461172920-221207-UBSL1.1__D"
    # 1 段目: 全体像（32 画素ごと）の明るさの相関で、古い日付の大まかなずれを求める
    qn, qo = np.load(str(new) + "_ql.npy"), np.load(str(old) + "_ql.npy")
    h_ = min(qn.shape[0], qo.shape[0])
    w_ = min(qn.shape[1], qo.shape[1])
    dr0, dc0 = subpixel_shift(np.log1p(qn[:h_, :w_]), np.log1p(qo[:h_, :w_]))
    dr0, dc0 = int(round(dr0 * 32)), int(round(dc0 * 32))
    r0, r1, c0, c1 = a.row - a.half, a.row + a.half, a.col - a.half, a.col + a.half
    A = read_window(new, r0, r1, c0, c1)
    hb = header(old)
    B = read_window(old, max(0, r0 + dr0), min(hb["lines"], r1 + dr0), max(0, c0 + dc0), min(hb["pixels"], c1 + dc0))
    B = np.pad(B, ((0, A.shape[0] - B.shape[0]), (0, A.shape[1] - B.shape[1])))
    # 2 段目: 切り出しの中で、画素より細かいずれを求めて位相を保ってずらす
    dr, dc = subpixel_shift(np.log1p(np.abs(A) ** 2), np.log1p(np.abs(B) ** 2))
    Bs = fourier_shift(B, -dr, -dc)
    coh = coherence(A, Bs, win=(7, 7))
    pa, pb = np.abs(A) ** 2, np.abs(Bs) ** 2
    dA = 10 * np.log10(ndimage.uniform_filter(pa, 5) + 1e-9)
    dB = 10 * np.log10(ndimage.uniform_filter(pb, 5) + 1e-9)
    rec = dict(exp="est05", scene_new=new.name, scene_old=old.name, window=[r0, r1, c0, c1],
               shift_coarse=[dr0, dc0], shift_fine=[round(float(dr), 2), round(float(dc), 2)],
               coherence_median=round(float(np.median(coh)), 3), coherence_p90=round(float(np.percentile(coh, 90)), 3),
               frac_coherence_gt_05=round(float((coh > 0.5).mean()), 3))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savez_compressed(data_dir() / "cache" / "est05_kumamoto.npz", dA=dA[::2, ::2].astype("float32"), dB=dB[::2, ::2].astype("float32"),
                        coh=coh[::2, ::2].astype("float32"))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
