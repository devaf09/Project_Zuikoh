"""est_06b: 飛行機の保管場の格子の「位相のそろい具合」 — 止まっている機体は、そろい具合が高いままか

着想: 持ち主の案（2026-11-02）「位相差の変化幅を畳み込みで強調」。ASF から新しい SLC が取れない間に、手元のロズウェルの SLC 4 日付で試す。
      保管中の機体は「動かない金属」なので、そろい具合は高いままのはず（出入りの多い駐機場とは逆の見え方の見込み）。
方法（持ち主の方針: 軌道は使わず、データの相関だけ）:
    データ: ロズウェル、Sentinel-1C の SLC 4 日付（軌道 151、12 日おき）、VV。同じ日付の RTC（地図の座標）。正解の格子（data_labels/、航空写真を見て印）
    1. est_06 と同じく、RTC を SLC の画素に並べ直して相関で GCP のずれを直し、ほかの日付を重ねる
    2. となりの日付の組ごとに、そろい具合（窓 5 × 20）
    3. 200 m の格子ごとに、格子の中の 20 m おきの点（10 × 10）のそろい具合の平均 → 飛行機ありの格子とその他の格子で比べる（AUC）
結果: results/est06b.txt、図は tools/fig_est06b.py、格子の値は cache/est06b_cells.csv
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_06b_storage_coherence.py --site roswell --lon -104.53 --lat 33.30 --seed 0
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

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_cell_chips import load_cells  # noqa: E402
from est_01d_slc_two_dates import coherence, subpixel_shift  # noqa: E402
from est_03_point_relations import fourier_shift  # noqa: E402
from exp06_cells import auc  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

CELL = 100   # NAIP の 2 m の画素で 200 m


def find_window(z, lon, lat, hr=260, hc=1600):
    """ZIP の中で、(lon, lat) を含む帯（VV）の名前・GCP・窓を返す。"""
    for n in sorted(x for x in zipfile.ZipFile(z).namelist() if "/measurement/" in x and "-vv-" in x):
        href = f"/vsizip/{z.as_posix()}/{n}"
        with rasterio.open(href) as s:
            gcps = s.gcps[0]
            r, c = GCPTransformer(gcps).rowcol(lon, lat)
            if hr <= r < s.height - hr and hc <= c < s.width - hc:
                return href, gcps, ((r - hr, r + hr), (c - hc, c + hc))
    return None


def main():
    ap = argparse.ArgumentParser(description="est_06b: 保管場の格子の位相のそろい具合")
    ap.add_argument("--site", default="roswell")
    ap.add_argument("--lon", type=float, default=-104.53)
    ap.add_argument("--lat", type=float, default=33.30)
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est06b.txt"))
    a = ap.parse_args()
    d = data_dir()
    zips = sorted((d / "raw" / "s1_slc" / a.site).glob("*.zip"), key=lambda p: p.name[17:25])
    href0, gcps, win = find_window(zips[0], a.lon, a.lat)
    tr = GCPTransformer(gcps)
    with rasterio.open(href0) as s:
        A0 = s.read(1, window=win).astype("complex64")
    (r0, r1), (c0, c1) = win
    rr, cc = np.mgrid[r0:r1, c0:c1:4]
    xs, ys = tr.xy(rr.ravel(), cc.ravel())
    rtc = sorted((d / "raw" / "s1_rtc" / a.site).glob(f"*{zips[0].name[17:25]}*_vv.tif"))[0]
    with rasterio.open(rtc) as R:
        X, Y = transform("EPSG:4326", R.crs, list(xs), list(ys))
        img = R.read(1)
        col, row = ~R.transform * (np.array(X), np.array(Y))
        ok = (row >= 0) & (row < R.height) & (col >= 0) & (col < R.width)
        sim = np.zeros(len(X), "float32")
        sim[ok] = img[row[ok].astype(int), col[ok].astype(int)]
    sim = sim.reshape(rr.shape)
    amp = ndimage.uniform_filter(np.abs(A0) ** 2, (1, 4))[:, ::4][:, :sim.shape[1]]
    dr, dc4 = subpixel_shift(np.log1p(sim * 1e3), np.log1p(amp))
    shift = (float(dr), float(dc4) * 4)
    print(f"地図と SLC のずれ: 行 {shift[0]:+.1f}・列 {shift[1]:+.1f}")
    stack, dates = [A0], [zips[0].name[17:25]]
    for z in zips[1:]:
        f = find_window(z, a.lon, a.lat)
        with rasterio.open(f[0]) as s:
            B = s.read(1, window=win).astype("complex64")
        sr, sc = subpixel_shift(np.log1p(np.abs(A0) ** 2), np.log1p(np.abs(B) ** 2))
        stack.append(fourier_shift(B, -sr, -sc))
        dates.append(z.name[17:25])
    cohs = [coherence(stack[i], stack[i + 1], win=(5, 20)) for i in range(len(stack) - 1)]
    # 200 m の格子（NAIP の格子）ごとの平均
    air = load_cells(os.path.join(ROOT, "data_labels", f"{a.site}_aircraft_cells.csv"))
    with rasterio.open(d / "processed" / "naip" / f"{a.site}.tif") as nv:
        ntr, ncrs, nx, ny = nv.transform, nv.crs, nv.width // CELL, nv.height // CELL
    xs_, ys_ = [x for x, _ in air], [y for _, y in air]
    cells = [(x, y) for x in range(max(0, min(xs_) - 10), min(nx, max(xs_) + 11)) for y in range(max(0, min(ys_) - 10), min(ny, max(ys_) + 11))]
    rows = []
    sub = (np.arange(10) + 0.5) * CELL / 10
    for x, y in cells:
        px = [(ntr * ((x * CELL + u), (y * CELL + v))) for u in sub for v in sub]
        lon, lat = transform(ncrs, "EPSG:4326", [p[0] for p in px], [p[1] for p in px])
        r_, c_ = tr.rowcol(lon, lat)
        r_ = np.round(np.array(r_) - r0 + shift[0]).astype(int)
        c_ = np.round(np.array(c_) - c0 + shift[1]).astype(int)
        ok = (r_ >= 0) & (r_ < A0.shape[0]) & (c_ >= 0) & (c_ < A0.shape[1])
        if ok.mean() < 0.9:
            continue
        rows.append([x, y, int((x, y) in air)] + [float(c[r_[ok], c_[ok]].mean()) for c in cohs])
    T = np.array(rows)
    recs = []
    for i in range(len(cohs)):
        v, lab = T[:, 3 + i], T[:, 2] == 1
        recs.append(dict(exp="est06b", site=a.site, pair=f"{dates[i]}-{dates[i + 1]}", n_aircraft=int(lab.sum()), n_other=int((~lab).sum()),
                         coherence_aircraft=round(float(v[lab].mean()), 3), coherence_other=round(float(v[~lab].mean()), 3),
                         auc_aircraft_high=round(auc(v, lab), 3)))
    recs.append(dict(exp="est06b", site=a.site, summary=True, dates=dates, shift=[round(shift[0], 1), round(shift[1], 1)],
                     auc_mean_over_pairs=round(float(np.mean([r["auc_aircraft_high"] for r in recs])), 3),
                     auc_of_mean=round(auc(T[:, 3:].mean(1), T[:, 2] == 1), 3)))
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savetxt(d / "cache" / f"est06b_{a.site}_cells.csv", T, delimiter=",",
               header="x,y,aircraft," + ",".join(f"{dates[i]}-{dates[i + 1]}" for i in range(len(cohs))), fmt="%.4f")
    np.savez_compressed(d / "cache" / f"est06b_{a.site}.npz", amp=(10 * np.log10(np.abs(A0) ** 2 + 1e-6))[:, ::4].astype("float32"),
                        cohs=np.stack([c[:, ::4] for c in cohs]).astype("float32"), dates=np.array(dates))
    for r in recs:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
