"""est_01e: 数日付の SLC で、タンクの点の動きの形を見る — 「手前の点は動かず、奥の点だけ動く」か

着想: est_01d で、12 日でタンクの強い点の 36% が 1 m 以上動いた（対照 11%）。見立てでは、
      手前（衛星に近い側）の強い点 = 壁と地面の角（高さが変わらない → 動かない）、
      奥の点 = 屋根と壁の角（屋根と一緒に上下 → 距離の向きに動く）。
      正解（タンクごとの在庫）がなくても、この「手前と奥の違い」は形として確かめられる。
方法:
    1. 基準の日付（いちばん新しい日）の画像で、タンクの位置を合わせ（est_01c の register）、タンクの中の強い点（背景より 20 dB 以上）を選ぶ
    2. ほかの日付ごとに、全体のずれを明るさの相関で測って引き（est_01d）、点ごとに距離の向きの位置を 8 倍の補間 + 放物線で決める
    3. 点ごとに、日付をまたいだ位置のばらつき（標準偏差、m）を出す。対照 = タンクから離れた強い点（町の建物など）
    4. 点を「タンクの中心より手前（距離が短い）」と「奥」に分け、ばらつきが対照の 3 倍を超える点（動く点）の割合を比べる
指標: 動く点の割合（手前・奥・対照）、点ごとのばらつきの中央値
結果: results/est01e.txt（1 行 JSON）、点ごとの表は cache/est01e_points.csv
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01e_slc_time_series.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer
from scipy import ndimage
from scipy.signal import find_peaks

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01c_slc_tank_points import register  # noqa: E402
from est_01d_slc_two_dates import PIX_R, peak_pos, read, subpixel_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="est_01e: 数日付の SLC でタンクの点の動きの形")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--min-db", type=float, default=20.0)
    ap.add_argument("--own-radius", action="store_true",
                    help="タンクごとの探す幅を、そのタンクの半径（斜めの距離に直した分）+ 3 画素に絞る（となりのタンクの点を混ぜない）")
    ap.add_argument("--theta-deg", type=float, default=39.0)
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est01e.txt"))
    a = ap.parse_args()

    d = data_dir()
    rec_c = [json.loads(x) for x in open(os.path.join(ROOT, "results", "est01c.txt"), encoding="utf-8")][-1]
    r0, r1, c0, c1 = rec_c["window"]
    win = ((r0, r1), (c0, c1))
    zips = sorted((d / "raw" / "s1_slc" / a.region).glob("*.zip"), key=lambda p: p.name[17:25])
    ref = zips[-1]
    R = read(ref, "iw3", win)
    pr = np.abs(R) ** 2
    bg = np.median(pr[pr > 0])

    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    lon = [np.mean([p[0] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
    lat = [np.mean([p[1] for p in f["geometry"]["coordinates"][0]]) for f in tanks["features"]]
    name = [n for n in zipfile.ZipFile(ref).namelist() if "/measurement/" in n and "iw3-slc-vv" in n][0]
    with rasterio.open(f"/vsizip/{ref.as_posix()}/{name}") as s:
        rows, cols = GCPTransformer(s.gcps[0]).rowcol(lon, lat)
    rows, cols = np.array(rows) - r0, np.array(cols) - c0
    sr, sc, st = register(pr, rows, cols)
    rows, cols = rows + sr, cols + sc
    if a.own_radius:                                          # 距離の向きは 1 画素の細かさで位置合わせをやり直す（4 画素の刻み = 約 9 m の誤差を減らす）
        best = (-1e9, 0)
        for dcol in range(-4, 5):
            sc2 = 0.0
            for r, c in zip(rows, cols + dcol):
                r, c = int(r), int(c)
                if 3 <= r < pr.shape[0] - 3 and 3 <= c < pr.shape[1] - 3:
                    sc2 += np.log1p(pr[r - 3:r + 4, c - 3:c + 4] / bg).sum()
            best = max(best, (sc2, dcol))
        cols = cols + best[1]
        sc += best[1]
    radius_px = []
    for f in tanks["features"]:                               # タンクの半径 → 斜めの距離の画素（地面の距離 × sinθ / 2.33 m）
        xs = [q[0] for q in f["geometry"]["coordinates"][0]]
        r_m = (max(xs) - min(xs)) / 2 * 111320 * np.cos(np.radians(np.mean(lat)))
        radius_px.append(r_m * np.sin(np.radians(a.theta_deg)) / PIX_R)

    # 基準の日付で点を選ぶ: タンクの中の強い点と、対照（タンクから離れた強い点）
    pts = []                                                  # (タンクの番号 or −1, 行, 列, dB, 中心からの距離の向きの位置 m)
    zone = np.zeros(pr.shape, bool)
    for ti, (r, c) in enumerate(zip(rows, cols)):
        r, c = int(r), int(c)
        if not (8 <= r < pr.shape[0] - 8 and 40 <= c < pr.shape[1] - 40):
            continue
        zone[max(0, r - 10):r + 11, max(0, c - 60):c + 61] = True
        h = int(radius_px[ti]) + 3 if a.own_radius else 30
        sub = pr[r - 3:r + 4, c - h:c + h + 1]
        prof = 10 * np.log10(sub.max(axis=0) / bg + 1e-12)
        for p in find_peaks(prof, height=a.min_db, distance=3)[0]:
            rr = r - 3 + int(np.argmax(sub[:, p]))
            pts.append((ti, rr, c - h + p, prof[p], (p - h) * PIX_R))
    loc = (pr == ndimage.maximum_filter(pr, size=(5, 9))) & (10 * np.log10(pr / bg + 1e-12) >= a.min_db) & ~zone
    for rr, cc in zip(*np.nonzero(loc)):
        if 8 <= rr < pr.shape[0] - 8 and 20 <= cc < pr.shape[1] - 20:
            pts.append((-1, rr, cc, 10 * np.log10(pr[rr, cc] / bg), 0.0))
    pts = np.array(pts)

    # 日付ごとに位置を測る（基準の日付は 0）
    pos = np.full((len(pts), len(zips)), np.nan)
    dates = []
    for j, z in enumerate(zips):
        dates.append(z.name[17:25])
        B = R if z == ref else read(z, "iw3", win)
        dr, dc = (0.0, 0.0) if z == ref else subpixel_shift(np.log1p(pr), np.log1p(np.abs(B) ** 2))
        ir, ic = int(round(dr)), int(round(dc))
        Bs = np.roll(np.roll(B, -ir, 0), -ic, 1)
        for i, (_, rr, cc, _, _) in enumerate(pts):
            rr, cc = int(rr), int(cc)
            xr, xb = peak_pos(R[rr, cc - 8:cc + 9]), peak_pos(Bs[rr, cc - 8:cc + 9])
            if abs(xb - xr) <= 4:
                pos[i, j] = (xb - xr - (dc - ic)) * PIX_R
        print(f"{dates[-1]}: 全体のずれ 行 {dr:+.2f}・列 {dc:+.2f}")
    ok = np.isfinite(pos).sum(1) >= len(zips) - 1                # ほとんどの日付で追えた点だけ
    sd = np.nanstd(pos, axis=1)
    ctrl = ok & (pts[:, 0] < 0)
    tank = ok & (pts[:, 0] >= 0)
    thr = 3 * np.median(sd[ctrl])
    near = tank & (pts[:, 4] < 0)
    far = tank & (pts[:, 4] > 0)
    frac = lambda m: round(float((sd[m] > thr).mean()), 3) if m.any() else None
    rec = dict(exp="est01e", own_radius=a.own_radius, dates=dates, n_dates=len(zips), tank_shift=[int(sr), int(sc)], min_db=a.min_db,
               n_control=int(ctrl.sum()), n_near=int(near.sum()), n_far=int(far.sum()),
               sd_m_control_median=round(float(np.median(sd[ctrl])), 3), sd_m_near_median=round(float(np.median(sd[near])), 3),
               sd_m_far_median=round(float(np.median(sd[far])), 3), move_threshold_m=round(float(thr), 3),
               frac_moving_control=frac(ctrl), frac_moving_near=frac(near), frac_moving_far=frac(far))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savetxt(d / "cache" / f"est01e_points{'_own' if a.own_radius else ''}.csv", np.column_stack([pts, sd, pos]), delimiter=",",
               header="tank,row,col,db,offset_m,sd_m," + ",".join(dates), fmt="%.4f")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
