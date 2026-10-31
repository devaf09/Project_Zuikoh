"""est_03: 点どうしの関係を周波数成分として見る — 反射板と周りの動かない点の、日付ごとの位相の縞

着想: 持ち主の問い（2026-10-31 夜）「3 点以上（2 次元の面）の点どうしの関係を周波数成分にできるか。時間ごとに変化する
      点間の周波数成分に関係があるか。1 つの基準物で実験できる」。
考え方:
    複素数の画像（SLC）の点 i の値は z_i = 振幅 × exp(位相)。2 つの日付の掛け算 z_i(t)·conj(z_i(0)) の位相 φ_i(t) は、
    点 i の「衛星までの距離の変化」を波長（5.5 cm）の半分を 1 周として表す（干渉 SAR の原理）。
    点を並べると、φ_i(t) は点の位置に対して「縞」になる。縞の細かさ（空間の周波数）は、2 つの撮影の位置の差（基線）で決まり、
    日付の組ごとに変わる。縞を引いた残り = 点ごとの本当の動き（mm）と高さの違い。
方法:
    1. 反射板 N08K（est_02 で最もきれいに写った）の周り ±60 行・±300 列（約 1.7 km × 1.4 km）を 6 日付で読む
    2. 日付ごとに、明るさの相関で画素より細かくずれを測り、周波数の側で位相を保ったままずらして重ねる
    3. 点を選ぶ: 基準の日付で背景より 25 dB 以上の山、かつ 6 日付で振幅が安定（振幅のばらつき / 平均 < 0.25、
       干渉 SAR の定番の「動かない点（PS）」の選び方）
    4. 日付ごとに、点の位相 φ_i を、点の位置 (距離の向き x_i, 進行方向 y_i) に対する縞 exp(i(kx·x + ky·y)) で説明する。
       縞の (kx, ky) は、Σ_i exp(i(φ_i − kx·x_i − ky·y_i)) の大きさが最大になるものを探す（点の並びの 2 次元の周波数成分）
    5. 縞を引いた残りの位相 → 動き（mm、= −λ/(4π)·位相）。点の組のそろい具合（残りの位相の平均の長さ、1 = 完全にそろう）
指標: 日付ごとの縞の周波数（1 km あたりの縞の数）、そろい具合、残りのばらつき（mm）
確かめ: 偶然の水準（位相をでたらめにしても同じ探し方でそろって見える度合い）と、閉じ具合（(i→j)+(j→k)=(i→k) か）
結果: results/est03.txt（日付ごとに 1 行 JSON）、図は tools/fig_est03.py
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_03_point_relations.py --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import PIX_R, subpixel_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

PIX_A = 13.9          # 進行方向の画素（m、おおよそ）
LAM = 0.05547         # Sentinel-1 の波長（m）


def fourier_shift(z, dr, dc):
    """複素数の画像 z を (dr, dc) 画素だけずらす（周波数の側で位相の傾きを掛ける。位相を保つ）。"""
    ky = np.fft.fftfreq(z.shape[0])[:, None]
    kx = np.fft.fftfreq(z.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(z) * np.exp(-2j * np.pi * (ky * dr + kx * dc)))


def main():
    ap = argparse.ArgumentParser(description="est_03: 点どうしの関係を周波数成分として見る")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--cr", default="N08K")
    ap.add_argument("--lon", type=float, help="反射板の代わりに、この場所の周りで試す（基準の点 = 範囲の中で最も明るい点）")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--name", default=None, help="結果に残す場所の名前（--lon のとき）")
    ap.add_argument("--min-db", type=float, default=25.0)
    ap.add_argument("--max-da", type=float, default=0.25, help="振幅のばらつき / 平均 の上限（動かない点の選び方）")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est03.txt"))
    a = ap.parse_args()

    d = data_dir()
    if a.lon is not None:
        cr = dict(id=a.name or f"{a.lon:.3f},{a.lat:.3f}", lon=a.lon, lat=a.lat)
        a.cr = cr["id"]
    else:
        cr = [r for r in csv.DictReader(open(d / "raw" / "cr" / "oklahoma" / "nisar_cr_oklahoma.csv", encoding="utf-8")) if r["id"] == a.cr][0]
    zips = sorted((d / "raw" / "s1_slc" / a.region).glob("*.zip"), key=lambda p: p.name[17:25])
    imgs, dates = [], []
    for z in zips:
        for n in sorted(x for x in zipfile.ZipFile(z).namelist() if "/measurement/" in x and "-vv-" in x):
            with rasterio.open(f"/vsizip/{z.as_posix()}/{n}") as s:
                row, col = GCPTransformer(s.gcps[0]).rowcol(float(cr["lon"]), float(cr["lat"]))
                if 300 <= row < s.height - 300 and 700 <= col < s.width - 700:
                    big = s.read(1, window=((row - 250, row + 251), (col - 600, col + 601))).astype("complex64")
                    # 反射板 = 予測の距離の向き ±60 画素の帯で最も明るい点（build_cr_chips と同じ）
                    pw = np.abs(big) ** 2
                    band = np.zeros_like(pw)
                    if a.lon is None:
                        band[:, 540:661] = pw[:, 540:661]
                    else:                                    # 場所を指定したとき: 中央付近（±100 行・±250 列）で最も明るい点を基準に
                        band[150:351, 350:851] = pw[150:351, 350:851]
                    if imgs:                                 # 2 日付目からは、1 日付目と同じ点を基準に（GCP のずれは日付でほぼ同じ）
                        kr, kc = kr0, kc0
                    else:
                        kr, kc = np.unravel_index(np.argmax(band), pw.shape)
                        kr0, kc0 = kr, kc
                    imgs.append(big[kr - 60:kr + 61, kc - 300:kc + 301])
                    dates.append(z.name[17:25])
                    break
    ref = imgs[0]
    amp_ref = np.log1p(np.abs(ref) ** 2)
    stack = [ref]
    for z in imgs[1:]:
        dr, dc = subpixel_shift(amp_ref, np.log1p(np.abs(z) ** 2))
        stack.append(fourier_shift(z, -dr, -dc))              # 基準に重ねる
    S = np.stack(stack)                                       # (日付, 行, 列)
    amp = np.abs(S)
    bg = np.median(amp[0] ** 2)
    db0 = 10 * np.log10(amp[0] ** 2 / bg + 1e-12)
    loc = (amp[0] == ndimage.maximum_filter(amp[0], size=(3, 7))) & (db0 >= a.min_db)
    da = amp.std(0) / (amp.mean(0) + 1e-12)                  # 振幅のばらつき / 平均
    sel = loc & (da < a.max_da)
    rr, cc = np.nonzero(sel)
    c0r, c0c = 60, 300                                       # 反射板の位置（切り出しの中心）
    x = (cc - c0c) * PIX_R                                   # 距離の向きの位置（m、反射板から）
    y = (rr - c0r) * PIX_A                                   # 進行方向の位置（m）
    is_cr = (rr == c0r) & (cc == c0c)
    print(f"点 {len(rr)} 個（反射板を含む: {bool(is_cr.any())}）、範囲 {x.min():.0f}〜{x.max():.0f} m × {y.min():.0f}〜{y.max():.0f} m")
    kxs = np.linspace(-2 * np.pi / 40, 2 * np.pi / 40, 401)  # 探す縞: 距離の向きに 40 m 周期まで
    kys = np.linspace(-2 * np.pi / 400, 2 * np.pi / 400, 201)  # 進行方向に 400 m 周期まで
    recs = []
    for t in range(1, len(dates)):
        ph = np.angle(S[t, rr, cc] * np.conj(S[0, rr, cc]))
        e = np.exp(1j * ph)
        # 点の並びの 2 次元の周波数成分（離散の点のフーリエ変換）: 大きさが最大の (kx, ky) = 縞
        P = np.abs(np.exp(-1j * (kys[:, None, None] * y[None, None, :] + kxs[None, :, None] * x[None, None, :])) @ e) / len(e)
        iy, ix = np.unravel_index(np.argmax(P), P.shape)
        kx, ky = kxs[ix], kys[iy]
        res = e * np.exp(-1j * (kx * x + ky * y))
        res = res * np.conj(res[is_cr][0]) if is_cr.any() else res   # 反射板を 0 にそろえる
        mm = -LAM / (4 * np.pi) * np.angle(res) * 1000
        rec = dict(exp="est03", cr=a.cr, date_ref=dates[0], date=dates[t], n_points=int(len(rr)),
                   fringe_range_per_km=round(float(kx / (2 * np.pi) * 1000), 3), fringe_azimuth_per_km=round(float(ky / (2 * np.pi) * 1000), 3),
                   coherence_before=round(float(np.abs(e.mean())), 3), coherence_after=round(float(np.abs(res.mean())), 3),
                   residual_mm_sd=round(float(np.std(mm)), 2), residual_mm_median_abs=round(float(np.median(np.abs(mm))), 2))
        recs.append(rec)
        print(f"{dates[0]} → {dates[t]}: 縞 距離の向き {rec['fringe_range_per_km']:+.2f} 本/km・進行方向 {rec['fringe_azimuth_per_km']:+.3f} 本/km、"
              f"そろい具合 {rec['coherence_before']:.2f} → {rec['coherence_after']:.2f}、残りのばらつき {rec['residual_mm_sd']:.1f} mm")
    # 確かめ 1（偶然の水準）: 位相をでたらめにしたとき、同じ探し方でどこまで「そろって見える」か（点が少ないと高く出る）
    rng = np.random.default_rng(a.seed)
    E = np.exp(-1j * (kys[:, None, None] * y[None, None, :] + kxs[None, :, None] * x[None, None, :]))
    null = [np.abs(E @ np.exp(1j * rng.uniform(-np.pi, np.pi, len(x)))).max() / len(x) for _ in range(100)]
    # 確かめ 2（閉じ具合）: 本物の縞なら (i→j) + (j→k) = (i→k) が成り立つ。日付の 3 つ組すべてで差を数える
    fr = {}
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            e = np.exp(1j * np.angle(S[j, rr, cc] * np.conj(S[i, rr, cc])))
            P = np.abs(E @ e)
            fr[(i, j)] = kxs[np.unravel_index(np.argmax(P), P.shape)[1]] / (2 * np.pi) * 1000
    clos = [fr[(i, j)] + fr[(j, k)] - fr[(i, k)] for i in range(len(dates)) for j in range(i + 1, len(dates)) for k in range(j + 1, len(dates))]
    summ = dict(exp="est03", cr=a.cr, summary=True, n_points=int(len(rr)), min_db=a.min_db, max_da=a.max_da,
                null_coherence_median=round(float(np.median(null)), 3), null_coherence_p95=round(float(np.percentile(null, 95)), 3),
                closure_abs_median_per_km=round(float(np.median(np.abs(clos))), 3),
                fringe_spread_per_km=round(float(np.std(list(fr.values()))), 3))
    recs.append(summ)
    print("確かめ:", json.dumps(summ, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez(d / "cache" / f"est03_{a.cr}.npz", S=S, rr=rr, cc=cc, x=x, y=y, dates=np.array(dates))


if __name__ == "__main__":
    main()
