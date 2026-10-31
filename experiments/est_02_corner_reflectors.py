"""est_02: 角の反射板で測り方を確かめる — 答えの分かった点の明るさ・ぼけの形・位置のばらつき

着想: 持ち主の依頼（2026-10-31 夜）で、タンクをいったん止め、答えの分かった目印（ベンチマーク）で測り方を確かめる（docs/landmarks.md）。
      est_01e で「動かないはずの点でも 18% が動いて見える」「タンクの中心が明るい点に引っぱられる」と分かった。
      位置の分かった強い点 1 つ（NISAR の 2.8 m の反射板）なら、測り方そのものの精度が分かる。
方法:
    データ: すでに取ったクッシングの Sentinel-1D の SLC 6 日付（上昇軌道 34）。その中に入るオクラホマの反射板 4 基（N07K〜N10K、JPL の公開の座標）
    1. 反射板の座標を付属の目印の点（GCP）で画像の行・列に直し、周り（±150 行・±400 列）を読む。いちばん明るい点を反射板とする
       （GCP がずれる大きさ = 予測と実際の差も記録する。est_01c では約 1 km だった）
    2. 明るさ: 反射板の山 / 周りの中央値（SCR、dB）
    3. ぼけの形: 距離の向き・進行方向の明るさの形の、半分の明るさの幅（3 dB の幅、m）と、いちばん高い尾の高さ（PSLR、dB）
    4. 位置のばらつき: 日付ごとに、周りの明るさの相関で全体のずれを引いた後の、反射板の山の位置（8 倍の補間 + 放物線）の標準偏差
       = est_01d・01e の測り方の、理想に近い条件での精度
指標: SCR、3 dB の幅、PSLR、位置のばらつき（距離の向き m・進行方向 m）、GCP の予測とのずれ（画素）
結果: results/est02.txt（反射板 × 日付ごとに 1 行 JSON と、反射板ごとのまとめ 1 行）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_02_corner_reflectors.py --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import peak_pos, subpixel_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

PIX_R, PIX_A = 2.33, 13.9      # 距離の向き（斜めの距離）・進行方向の画素の大きさ（m、IW の SLC のおおよそ。知）


def width_and_pslr(prof_db, k, pix):
    """山 k の 3 dB の幅（m、隣の点との直線補間）と、主の山の外でいちばん高い尾の高さ（dB、山との差）。"""
    top = prof_db[k]
    lo = k
    while lo > 0 and prof_db[lo] > top - 3:
        lo -= 1
    hi = k
    while hi < len(prof_db) - 1 and prof_db[hi] > top - 3:
        hi += 1
    f = lambda a, b: a + (top - 3 - prof_db[a]) / (prof_db[b] - prof_db[a] + 1e-12) * (b - a)
    w = (f(hi - 1, hi) - f(lo + 1, lo)) * pix if 0 < lo and hi < len(prof_db) - 1 else np.nan
    # 主の山の外 = 山から最初の谷までの外側
    l, r = k, k
    while l > 0 and prof_db[l - 1] < prof_db[l]:
        l -= 1
    while r < len(prof_db) - 1 and prof_db[r + 1] < prof_db[r]:
        r += 1
    side = np.concatenate([prof_db[max(0, l - 20):l], prof_db[r + 1:r + 21]])
    return float(w), float(side.max() - top) if side.size else np.nan


def upsample(v, up=8):
    """複素数の並びを周波数の側でゼロを詰めて up 倍に細かくし、明るさの dB を返す。"""
    n = len(v)
    S = np.fft.fftshift(np.fft.fft(v))
    pad = np.zeros(n * up, dtype=complex)
    pad[(n * up - n) // 2:(n * up + n) // 2] = S
    return 20 * np.log10(np.abs(np.fft.ifft(np.fft.ifftshift(pad))) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description="est_02: 角の反射板で測り方を確かめる")
    ap.add_argument("--region", default="cushing", help="SLC を置いた地域のフォルダ")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est02.txt"))
    a = ap.parse_args()

    d = data_dir()
    crs = list(csv.DictReader(open(d / "raw" / "cr" / "oklahoma" / "nisar_cr_oklahoma.csv", encoding="utf-8")))
    zips = sorted((d / "raw" / "s1_slc" / a.region).glob("*.zip"), key=lambda p: p.name[17:25])
    recs = []
    for cr in crs:
        lon, lat = float(cr["lon"]), float(cr["lat"])
        per_date = []
        ref_amp = None
        for z in zips:
            hit = None
            for n in sorted(x for x in zipfile.ZipFile(z).namelist() if "/measurement/" in x and "-vv-" in x):
                with rasterio.open(f"/vsizip/{z.as_posix()}/{n}") as s:
                    row, col = GCPTransformer(s.gcps[0]).rowcol(lon, lat)
                    if 160 <= row < s.height - 160 and 420 <= col < s.width - 420:
                        win = ((row - 150, row + 151), (col - 400, col + 401))
                        hit = (n.split("/")[-1][:7], row, col, s.read(1, window=win).astype("complex64"))
                        break
            if hit is None:
                continue
            sub, row, col, A = hit
            pw = np.abs(A) ** 2
            bg = np.median(pw)
            kr, kc = np.unravel_index(np.argmax(pw), pw.shape)
            if not (16 <= kr < pw.shape[0] - 16 and 16 <= kc < pw.shape[1] - 16):
                continue
            prof_r = upsample(A[kr, kc - 16:kc + 17])          # 距離の向き
            prof_a = upsample(A[kr - 16:kr + 17, kc])          # 進行方向
            wr, pr = width_and_pslr(prof_r, int(np.argmax(prof_r)), PIX_R / 8)
            wa, pa_ = width_and_pslr(prof_a, int(np.argmax(prof_a)), PIX_A / 8)
            if ref_amp is None:                                # 最初の日付を基準に、周りの明るさの相関で全体のずれを測る
                ref_amp, dr, dc = np.log1p(pw), 0.0, 0.0
            else:
                dr, dc = subpixel_shift(ref_amp, np.log1p(pw))
            xr = kc - 8 + peak_pos(A[kr, kc - 8:kc + 9])          # 距離の向きの位置（画素、小数まで）
            xa = kr - 8 + peak_pos(A[kr - 8:kr + 9, kc])          # 進行方向の位置
            rec = dict(exp="est02", cr=cr["id"], cr_azimuth_deg=float(cr["azimuth_deg"]), date=z.name[17:25], subswath=sub,
                       scr_db=round(float(10 * np.log10(pw[kr, kc] / bg)), 1),
                       gcp_offset_rows=int(kr - 150), gcp_offset_cols=int(kc - 400),
                       width_range_m=round(wr, 2), width_azimuth_m=round(wa, 2), pslr_range_db=round(pr, 1), pslr_azimuth_db=round(pa_, 1),
                       pos_range_m=round(float((xr - dc) * PIX_R), 3), pos_azimuth_m=round(float((xa - dr) * PIX_A), 3))
            per_date.append(rec)
            recs.append(rec)
            print(f"{cr['id']} {rec['date']} {sub}: SCR {rec['scr_db']} dB、GCP とのずれ 行 {rec['gcp_offset_rows']:+d}・列 {rec['gcp_offset_cols']:+d}、"
                  f"幅 {wr:.2f} m × {wa:.2f} m、PSLR {pr:.1f} / {pa_:.1f} dB")
        if len(per_date) >= 3:
            g = lambda k: np.array([r[k] for r in per_date])
            recs.append(dict(exp="est02", cr=cr["id"], summary=True, n_dates=len(per_date),
                             scr_db_median=float(np.median(g("scr_db"))),
                             pos_range_sd_m=round(float(np.std(g("pos_range_m") - g("pos_range_m").mean())), 3),
                             pos_azimuth_sd_m=round(float(np.std(g("pos_azimuth_m") - g("pos_azimuth_m").mean())), 3),
                             width_range_m_median=float(np.median(g("width_range_m"))),
                             width_azimuth_m_median=float(np.median(g("width_azimuth_m")))))
            print("  まとめ:", json.dumps(recs[-1], ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
