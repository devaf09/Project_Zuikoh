"""est_04d: 位置を合わせ直した画像で、駐機位置の「いる・いない」を数え直す

着想: est_04 の「強い日・弱い日」は見かけだけだった（版の違い・日付ごとの位置のずれ）。tools/align_gec.py でデータどうしの相関で
      位置を合わせ直し（持ち主の方針の範囲）、明るさは画像ごとの順位（0〜1）にそろえて（版の違いをならす）、数え直す。
方法:
    1. 合わせ直した画像（processed/umbra_aligned/<空港>/）で、駐機位置（OSM、多数の日付の位置に動かす）の周り半径 R m の中で、
       「その画像の上位 5% より明るい画素」の割合を数える（金属の機体があれば明るい画素がまとまって増える）
    2. 対照: 駐機位置から 150 m 以上離れた、駐機場（apron）の中のでたらめな点で同じ値を数える（飛行機の来ない舗装の目安）
    3. 駐機位置ごとに、日付をまたいだ値の幅と、対照の 95% を超える日の割合（= 仮に「いる」）を出す
結果: results/est04d.txt、駐機位置 × 日付の表は cache/est04d_<空港>.csv
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_04d_stands_aligned.py --airport bkk --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform, transform_geom

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="est_04d: 位置を合わせ直した画像で駐機位置を数え直す")
    ap.add_argument("--airport", default="bkk")
    ap.add_argument("--radius-m", type=float, default=20.0)
    ap.add_argument("--top", type=float, default=0.05, help="明るい画素 = その画像の上位この割合")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est04d.txt"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = data_dir()
    al = [json.loads(x) for x in open(os.path.join(ROOT, "results", f"align_{a.airport}.txt"), encoding="utf-8")]
    me = float(np.median([r["shift_east_m"] for r in al[1:]]))
    mn = float(np.median([r["shift_north_m"] for r in al[1:]]))
    osm = json.load(open(d / "raw" / "osm" / "airports" / f"{a.airport}.geojson", encoding="utf-8"))
    st, aprons = [], []
    for f in osm["features"]:
        k, g = f["properties"].get("aeroway"), f["geometry"]
        if k == "parking_position":
            pts = [g["coordinates"]] if g["type"] == "Point" else (g["coordinates"] if g["type"] == "LineString" else g["coordinates"][0])
            st.append((np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])))
        elif k == "apron" and g["type"] == "Polygon":
            aprons.append(g)
    files = [d / "processed" / "umbra_aligned" / a.airport / f"{r['date']}.tif" for r in al]
    with rasterio.open(files[0]) as s0:
        crs, tr, H, W = s0.crs, s0.transform, s0.height, s0.width
    xs, ys = transform("EPSG:4326", crs, [p[0] for p in st], [p[1] for p in st])
    pts = np.array([~tr * (x + me, y + mn) for x, y in zip(xs, ys)])     # (列, 行)
    am = np.zeros((H, W), bool)
    for g in aprons:
        am |= ~geometry_mask([transform_geom("EPSG:4326", crs, g)], out_shape=(H, W), transform=tr)
    # 対照: 駐機場の中で、どの駐機位置からも 150 m 以上離れた点
    cand = np.argwhere(am)
    cand = cand[rng.choice(len(cand), min(20000, len(cand)), replace=False)]
    dmin = np.min(np.hypot(cand[:, 1, None] - pts[None, :, 0], cand[:, 0, None] - pts[None, :, 1]), axis=1)
    ctrl = cand[dmin > 150][:300]
    r_px = int(a.radius_m)
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disc = (yy ** 2 + xx ** 2) <= r_px ** 2

    def frac_bright(img, thr, c, r):
        c, r = int(round(c)), int(round(r))
        if not (r_px <= r < H - r_px and r_px <= c < W - r_px):
            return np.nan
        w = img[r - r_px:r + r_px + 1, c - r_px:c + r_px + 1]
        if (w[disc] == 0).mean() > 0.3:
            return np.nan
        return float((w[disc] > thr).mean())

    S = np.full((len(pts), len(files)), np.nan)
    Cn = np.full((len(ctrl), len(files)), np.nan)
    for j, f in enumerate(files):
        with rasterio.open(f) as s:
            img = s.read(1)
        v = img[img > 0]
        thr = np.quantile(v, 1 - a.top)                         # その画像の上位 5% の明るさ（版の違いをならす）
        for i, (c, r) in enumerate(pts):
            S[i, j] = frac_bright(img, thr, c, r)
        for i, (r, c) in enumerate(ctrl):
            Cn[i, j] = frac_bright(img, thr, c, r)
    c95 = np.nanpercentile(Cn, 95, axis=0)                     # 日付ごとの対照の 95%
    occ = S > c95[None, :]
    rec = dict(exp="est04d", airport=a.airport, n_dates=len(files), n_stands=len(pts), n_control=len(ctrl), radius_m=a.radius_m, top=a.top,
               stand_frac_median=round(float(np.nanmedian(S)), 3), control_frac_median=round(float(np.nanmedian(Cn)), 3),
               control_p95_median=round(float(np.median(c95)), 3),
               occupied_rate_median=round(float(np.nanmedian(np.nanmean(np.where(np.isfinite(S), occ, np.nan), 1))), 3),
               frac_stands_ever_occupied=round(float(occ.any(1).mean()), 3),
               frac_stands_mixed=round(float(((occ.sum(1) > 0) & (occ.sum(1) < np.isfinite(S).sum(1))).mean()), 3))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savetxt(d / "cache" / f"est04d_{a.airport}.csv", np.column_stack([pts, S]), delimiter=",",
               header="col,row," + ",".join(r["date"] for r in al), fmt="%.4f")
    np.savetxt(d / "cache" / f"est04d_{a.airport}_ctrl95.csv", c95[None, :], delimiter=",", fmt="%.4f")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
