"""est_06: 空港の駐機場の「位相のそろい具合」の時系列 — 飛行機の出入りで、そろい具合が下がるか

着想: 持ち主の案（2026-11-02）「位相差で数 mm の判断ができるなら、引き延ばして大きな一枚にし、その変化幅を畳み込みで強調できないか」。
      飛行機が来る・去るのは「物が入れ替わる」変化なので、位相はずれるのではなく、でたらめになる（そろい具合が下がる）はず。
      そろい具合の地図は、周りの画素を窓で畳み込んで求める（持ち主の「畳み込みで強調」と同じ考え）。
方法（持ち主の方針: 軌道の情報は使わず、データの相関だけ）:
    データ: ダラス・フォートワース空港、Sentinel-1D の SLC 6 日付（軌道 34、12 日おき）、VV。OSM の空港の地物。同じ軌道・日付の RTC（地図の座標）
    1. 基準の日付の SLC の空港の周りを読む。付属の目印の点（GCP）で、画素ごとの経度・緯度を出し、RTC の明るさをその画素に並べ直す
       → SLC の明るさ（距離の向きに 4 画素平均）との相関で、GCP のずれを求めて直す（地図と SLC を合わせる）
    2. ほかの日付を、明るさの相関で基準に重ねる（位相を保ってずらす）
    3. となりの日付の組（12 日おき）ごとに、そろい具合を窓（進行方向 5 × 距離の向き 20 画素 ≈ 70 m × 45 m）で求める
    4. OSM の地物の種類ごと（駐機位置・駐機場・誘導路・滑走路・ターミナル・格納庫）に、そろい具合の平均を出す。駐機位置ごとの時系列も
指標: 種類ごとのそろい具合、駐機位置のそろい具合のばらつき（日付の組ごと）
結果: results/est06.txt（組 × 種類ごとに 1 行 JSON）、駐機位置の表は cache/est06_stands.csv、図は tools/fig_est06.py
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_06_airport_coherence.py --seed 0
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
from rasterio.warp import transform
from scipy import ndimage
from shapely.geometry import Point, shape

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import coherence, subpixel_shift  # noqa: E402
from est_03_point_relations import fourier_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

LON, LAT = -97.04, 32.90


def find_window(z, hr=260, hc=1600):
    """ZIP の中で、空港の中心を含む帯（VV）の名前・GCP・窓 ((r0, r1), (c0, c1)) を返す。"""
    for n in sorted(x for x in zipfile.ZipFile(z).namelist() if "/measurement/" in x and "-vv-" in x):
        href = f"/vsizip/{z.as_posix()}/{n}"
        with rasterio.open(href) as s:
            gcps = s.gcps[0]
            r, c = GCPTransformer(gcps).rowcol(LON, LAT)
            if hr <= r < s.height - hr and hc <= c < s.width - hc:
                return href, gcps, ((r - hr, r + hr), (c - hc, c + hc))
    return None


def osm_points(feats, kinds, step_m=10.0):
    """OSM の地物を、種類ごとに経度・緯度の点の列にする（線・面は step_m おきに点を並べる）。"""
    out = {k: [] for k in kinds}
    dlat = step_m / 110540
    for f in feats:
        k = f["properties"].get("aeroway")
        if k not in out:
            continue
        g = shape(f["geometry"])
        if g.geom_type == "Point":
            out[k].append((g.x, g.y))
        elif g.geom_type == "LineString":
            n = max(2, int(g.length / dlat))
            out[k] += [(p.x, p.y) for p in (g.interpolate(i / (n - 1), normalized=True) for i in range(n))]
        else:
            x0, y0, x1, y1 = g.bounds
            for x in np.arange(x0, x1, dlat * 1.2):
                for y in np.arange(y0, y1, dlat):
                    if g.contains(Point(x, y)):
                        out[k].append((x, y))
    return {k: np.array(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(description="est_06: 空港の駐機場の位相のそろい具合の時系列")
    ap.add_argument("--win-az", type=int, default=5)
    ap.add_argument("--win-rg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est06.txt"))
    a = ap.parse_args()
    d = data_dir()
    zips = sorted((d / "raw" / "s1_slc" / "dfw").glob("*.zip"), key=lambda p: p.name[17:25])
    href0, gcps, win = find_window(zips[0])
    tr = GCPTransformer(gcps)
    with rasterio.open(href0) as s:
        A0 = s.read(1, window=win).astype("complex64")
    (r0, r1), (c0, c1) = win
    # 1. 地図（RTC）を SLC の画素に並べ直して、GCP のずれを相関で求める（距離の向きは 4 画素ごと）
    rr, cc = np.mgrid[r0:r1, c0:c1:4]
    xs, ys = tr.xy(rr.ravel(), cc.ravel())
    rtc = sorted((d / "raw" / "s1_rtc" / "dfw").glob(f"*{zips[0].name[17:25]}*_vv.tif"))[0]
    with rasterio.open(rtc) as R:
        X, Y = transform("EPSG:4326", R.crs, list(xs), list(ys))
        img = R.read(1)
        col, row = (~R.transform * (np.array(X), np.array(Y)))
        ok = (row >= 0) & (row < R.height) & (col >= 0) & (col < R.width)
        sim = np.zeros(len(X), "float32")
        sim[ok] = img[row[ok].astype(int), col[ok].astype(int)]
    sim = sim.reshape(rr.shape)
    amp = ndimage.uniform_filter(np.abs(A0) ** 2, (1, 4))[:, ::4][:, :sim.shape[1]]
    dr, dc4 = subpixel_shift(np.log1p(sim * 1e3), np.log1p(amp))
    gcp_shift = (float(dr), float(dc4) * 4)                   # 地図の位置 + これ = SLC の画素
    print(f"地図と SLC のずれ: 行 {gcp_shift[0]:+.1f}・列 {gcp_shift[1]:+.1f}")
    # 2. ほかの日付を重ねる
    stack, dates = [A0], [zips[0].name[17:25]]
    for z in zips[1:]:
        f = find_window(z)
        if f is None:
            continue
        with rasterio.open(f[0]) as s:
            B = s.read(1, window=win).astype("complex64")
        sr, sc = subpixel_shift(np.log1p(np.abs(A0) ** 2), np.log1p(np.abs(B) ** 2))
        stack.append(fourier_shift(B, -sr, -sc))
        dates.append(z.name[17:25])
        print(f"{dates[-1]}: 重ねたずれ 行 {sr:+.2f}・列 {sc:+.2f}")
    # 3. となりの日付の組のそろい具合
    cohs = [coherence(stack[i], stack[i + 1], win=(a.win_az, a.win_rg)) for i in range(len(stack) - 1)]
    # 4. OSM の地物の種類ごと
    osm = json.load(open(d / "raw" / "osm" / "airports" / "dfw.geojson", encoding="utf-8"))["features"]
    kinds = ["parking_position", "apron", "taxiway", "runway", "terminal", "hangar"]
    pts = osm_points(osm, kinds)

    def to_px(lonlat):
        r_, c_ = tr.rowcol(lonlat[:, 0], lonlat[:, 1])
        r_ = np.round(np.array(r_) - r0 + gcp_shift[0]).astype(int)
        c_ = np.round(np.array(c_) - c0 + gcp_shift[1]).astype(int)
        ok = (r_ >= 0) & (r_ < A0.shape[0]) & (c_ >= 0) & (c_ < A0.shape[1])
        return r_[ok], c_[ok], ok

    recs = []
    for i, coh in enumerate(cohs):
        for k, p in pts.items():
            r_, c_, _ = to_px(p)
            if len(r_) == 0:
                continue
            v = coh[r_, c_]
            recs.append(dict(exp="est06", pair=f"{dates[i]}-{dates[i + 1]}", kind=k, n=int(len(v)),
                             coherence_mean=round(float(v.mean()), 3), coherence_median=round(float(np.median(v)), 3)))
    allc = np.concatenate([c.ravel() for c in cohs])
    recs.append(dict(exp="est06", summary=True, dates=dates, gcp_shift=[round(gcp_shift[0], 1), round(gcp_shift[1], 1)],
                     win=[a.win_az, a.win_rg], coherence_all_median=round(float(np.median(allc)), 3)))
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 駐機位置ごとの時系列
    sp = pts["parking_position"]
    r_, c_, ok = to_px(sp)
    T = np.stack([c[r_, c_] for c in cohs], 1)
    np.savetxt(d / "cache" / "est06_stands.csv", np.column_stack([sp[ok], T]), delimiter=",",
               header="lon,lat," + ",".join(f"{dates[i]}-{dates[i + 1]}" for i in range(len(cohs))), fmt="%.5f")
    np.savez_compressed(d / "cache" / "est06_dfw.npz", amp=(10 * np.log10(np.abs(A0) ** 2 + 1e-6))[:, ::4].astype("float32"),
                        cohs=np.stack([c[:, ::4] for c in cohs]).astype("float32"), dates=np.array(dates), shift=np.array(gcp_shift),
                        win=np.array([r0, r1, c0, c1]))
    for r in recs[:-1]:
        print(r["pair"], r["kind"], r["n"], r["coherence_mean"])


if __name__ == "__main__":
    main()
