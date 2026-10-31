"""est_01d: 2 日付の SLC で、タンクの強い点がどれだけ動いたかを見る — 屋根と一緒に動く点を見分けられるか

着想: est_01c で、1 基の中に強い点がいくつもあると分かった。高さの変わらない点（壁と地面の角）は動かず、
      屋根と壁の角は屋根と一緒に距離の向きに動くはず（est_01）。同じ軌道の 2 日付（12 日おき）で、点ごとの動きを測る。
方法:
    1. 2 日付の SLC（同じ軌道・同じ帯 IW3・VV）の同じ範囲を読む。範囲は est_01c の切り出しと同じ行・列（同じ軌道なので、ほぼ同じ画素の並び）
    2. 全体の位置合わせ: 明るさの画像の相関（周波数の側で 16 倍に細かく）で、2 日付のずれ（行・列、画素の小数まで）を測り、
       以後の比較からその分を引く（衛星の軌道のわずかな違いを消す）
    3. タンクごとに、日付 A の強い点（背景より 20 dB 以上の山）を探し、日付 B の同じ点の周り（±4 画素）で山の位置を探す。
       距離の向きの位置を、8 倍の補間 + 放物線で画素より細かく決め、動き = B − A（斜めの距離 m）
    対照: タンクから離れた強い点（町の建物など、動かないはずの点）で、同じ測り方の動きを測る（測り方の雑音の実物の値）
    4. 2 日付の位相のそろい具合（コヒーレンス、5 × 20 画素の窓）を、タンクの場所と畑で比べる
指標: 点ごとの動き（m）の分布。動かない点のばらつき（測り方の精度の実物の値）と、大きく動いた点の割合。
      動き m → 屋根の高さの変化 m = 動き / cosθ（θ ≈ 39 度、符号: 近づく = 屋根が上がる）
結果: results/est01d.txt（1 行 JSON）、点ごとの表は cache/est01d_points.csv。
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01d_slc_two_dates.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import rasterio
from scipy import ndimage
from scipy.signal import find_peaks

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01c_slc_tank_points import find_measurement  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

PIX_R = 2.33   # 距離の向きの画素（斜めの距離 m）


def read(zip_path, sub_name, win):
    """ZIP の中の、名前に sub_name（例 iw3）を含む VV の TIFF から、窓 ((r0, r1), (c0, c1)) を読む。"""
    href = [h for h in find_measurement(zip_path, "vv") if sub_name in os.path.basename(h)][0]
    with rasterio.open(href) as src:
        return src.read(1, window=win).astype("complex64")


def subpixel_shift(a, b, up=16):
    """明るさの画像 a に対する b のずれ（行, 列）を、相関の山を周波数の側で up 倍に細かくして測る。"""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    cc = np.fft.ifft2(A * np.conj(B))
    k = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    shift = [((k[i] + cc.shape[i] // 2) % cc.shape[i]) - cc.shape[i] // 2 for i in range(2)]
    # 山の周りを細かく: 山の近くの 3 × 3 を放物線で
    out = []
    for ax in range(2):
        idx = list(k)
        v = []
        for d in (-1, 0, 1):
            idx2 = list(idx)
            idx2[ax] = (idx[ax] + d) % cc.shape[ax]
            v.append(abs(cc[tuple(idx2)]))
        den = v[0] - 2 * v[1] + v[2]
        out.append(shift[ax] + (0.5 * (v[0] - v[2]) / den if den != 0 else 0.0))
    return -out[0], -out[1]          # b を a に重ねるには、b を (−行, −列) ずらす → b の位置 = a の位置 + これ


def peak_pos(prof, up=8):
    """1 次元の複素数の並びの明るさの山の位置（画素、小数まで）。8 倍に補間 + 放物線。"""
    n = len(prof)
    S = np.fft.fftshift(np.fft.fft(prof))
    pad = np.zeros(n * up, dtype=complex)
    pad[(n * up - n) // 2:(n * up + n) // 2] = S
    m = np.abs(np.fft.ifft(np.fft.ifftshift(pad)))
    k = int(np.argmax(m[1:-1])) + 1
    den = m[k - 1] - 2 * m[k] + m[k + 1]
    d = 0.5 * (m[k - 1] - m[k + 1]) / den if den != 0 else 0.0
    return (k + d) / up


def coherence(a, b, win=(5, 20)):
    """2 日付の位相のそろい具合 |Σ a·conj(b)| / sqrt(Σ|a|² Σ|b|²)（窓の中で）。0 = ばらばら、1 = 完全にそろう。"""
    num = ndimage.uniform_filter((a * np.conj(b)).real, win) + 1j * ndimage.uniform_filter((a * np.conj(b)).imag, win)
    den = np.sqrt(ndimage.uniform_filter(np.abs(a) ** 2, win) * ndimage.uniform_filter(np.abs(b) ** 2, win))
    return np.abs(num) / (den + 1e-12)


def main():
    ap = argparse.ArgumentParser(description="est_01d: 2 日付の SLC でタンクの点の動き")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--theta-deg", type=float, default=39.0)
    ap.add_argument("--min-db", type=float, default=20.0, help="強い点とみなす、背景の中央値との比（dB）")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est01d.txt"))
    a = ap.parse_args()

    d = data_dir()
    rec_c = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est01c.txt"), encoding="utf-8")][-1]
    r0, r1, c0, c1 = rec_c["window"]
    win = ((r0, r1), (c0, c1))
    zips = sorted((d / "raw" / "s1_slc" / a.region).glob("*.zip"), key=lambda p: p.name[17:25])
    za, zb = zips[0], zips[-1]                               # 古い日付 = A、新しい日付 = B
    A, B = read(za, "iw3", win), read(zb, "iw3", win)
    pa, pb = np.abs(A) ** 2, np.abs(B) ** 2
    dr, dc = subpixel_shift(np.log1p(pa), np.log1p(pb))
    print(f"2 日付の全体のずれ: 行 {dr:+.2f}・列 {dc:+.2f}（画素）")
    # B を整数画素だけ戻して重ねる（小数の部分は点の動きから引く）
    irow, icol = int(round(dr)), int(round(dc))
    Bs = np.roll(np.roll(B, -irow, 0), -icol, 1)
    frac_c = dc - icol
    bg = np.median(pa[pa > 0])

    # タンクの場所（est_01c と同じ位置合わせ）
    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    import zipfile
    from rasterio.transform import GCPTransformer
    name = [n for n in zipfile.ZipFile(za).namelist() if "/measurement/" in n and "iw3-slc-vv" in n][0]
    with rasterio.open(f"/vsizip/{za.as_posix()}/{name}") as s:
        lon = [np.mean([p[0] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
        lat = [np.mean([p[1] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
        rows, cols = GCPTransformer(s.gcps[0]).rowcol(lon, lat)
    # 日付 A の GCP のずれは est_01c（日付 B = 10/24 で測った）と同じではないので、A の画像で測り直す
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from est_01c_slc_tank_points import register
    rows, cols = np.array(rows) - r0, np.array(cols) - c0
    sr, sc, st = register(pa, rows, cols)
    rows, cols = rows + sr, cols + sc
    print(f"タンクの位置合わせ（日付 A）: 行 {sr:+d}・列 {sc:+d}（強さ {st:.1f}）")

    pts = []
    for ti, (r, c) in enumerate(zip(rows, cols)):
        r, c = int(r), int(c)
        if not (8 <= r < pa.shape[0] - 8 and 40 <= c < pa.shape[1] - 40):
            continue
        sub = pa[r - 3:r + 4, c - 30:c + 31]
        prof_db = 10 * np.log10(sub.max(axis=0) / bg + 1e-12)
        pk, _ = find_peaks(prof_db, height=a.min_db, distance=3)
        for p in pk:
            rr = r - 3 + int(np.argmax(sub[:, p]))
            cc = c - 30 + p
            ia = A[rr, cc - 8:cc + 9]
            ib = Bs[rr, cc - 8:cc + 9]
            if len(ia) < 17:
                continue
            xa, xb = peak_pos(ia), peak_pos(ib)
            if abs(xb - xa) > 4:                              # 別の山を拾った
                continue
            move_m = (xb - xa - frac_c) * PIX_R
            pts.append((ti, rr, cc, float(prof_db[p]), move_m))
    pts = np.array(pts)

    # 対照: タンクから離れた強い点（町の建物など、動かないはずの点）で、同じ測り方のばらつきを測る
    tank_zone = np.zeros(pa.shape, bool)
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        tank_zone[max(0, r - 10):r + 11, max(0, c - 60):c + 61] = True
    loc = (pa == ndimage.maximum_filter(pa, size=(5, 9))) & (10 * np.log10(pa / bg + 1e-12) >= a.min_db) & ~tank_zone
    ctrl = []
    for rr, cc in zip(*np.nonzero(loc)):
        if 8 <= cc < pa.shape[1] - 9:
            xa, xb = peak_pos(A[rr, cc - 8:cc + 9]), peak_pos(Bs[rr, cc - 8:cc + 9])
            if abs(xb - xa) <= 4:
                ctrl.append((xb - xa - frac_c) * PIX_R)
    ctrl = np.array(ctrl)
    move = pts[:, 4]
    cos_t = math.cos(math.radians(a.theta_deg))
    mad = float(np.median(np.abs(move - np.median(move))) * 1.4826)    # 外れに強いばらつき（標準偏差に相当）
    big = np.abs(move) > max(3 * mad, 0.5)
    coh = coherence(A, Bs)
    tank_mask = np.zeros(pa.shape, bool)
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        if 3 <= r < pa.shape[0] - 3 and 15 <= c < pa.shape[1] - 15:
            tank_mask[r - 3:r + 4, c - 15:c + 16] = True
    rec = dict(exp="est01d", date_a=za.name[17:25], date_b=zb.name[17:25], window=rec_c["window"],
               global_shift_rows=round(dr, 3), global_shift_cols=round(dc, 3), tank_shift_rows=int(sr), tank_shift_cols=int(sc),
               n_points=int(len(pts)), n_tanks_with_points=int(len(set(pts[:, 0].astype(int)))),
               move_m_median=round(float(np.median(move)), 3), move_m_robust_sd=round(mad, 3),
               height_m_robust_sd=round(mad / cos_t, 3), frac_big_move=round(float(big.mean()), 3),
               n_tanks_big_move=int(len(set(pts[big, 0].astype(int)))),
               n_control=int(len(ctrl)), control_move_m_median=round(float(np.median(ctrl)), 3),
               control_move_m_robust_sd=round(float(np.median(np.abs(ctrl - np.median(ctrl))) * 1.4826), 3),
               control_frac_abs_gt_1m=round(float((np.abs(ctrl) > 1).mean()), 3),
               tank_frac_abs_gt_1m=round(float((np.abs(move) > 1).mean()), 3),
               coherence_tank_median=round(float(np.median(coh[tank_mask])), 3),
               coherence_other_median=round(float(np.median(coh[~tank_mask])), 3), min_db=a.min_db, theta_deg=a.theta_deg)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savetxt(d / "cache" / "est01d_points.csv", pts, delimiter=",", header="tank,row,col,db,move_m", fmt="%.4f")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
