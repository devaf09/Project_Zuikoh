"""Umbra の GEC を、共通の格子に置き直し、データどうしの相関で日付ごとの位置のずれを直す（est_04 の考察の次の一手）。

持ち主の方針（2026-11-01）: 軌道の情報は使わず、データどうしの相関だけ。ここでは、滑走路・誘導路・ターミナルなど
「動かない大きな形」の明るさ（対数）の相関で、基準の日付に対するずれ（東西・南北、m）を求める。
飛行機・車など動く物の影響を減らすため、相関は 4 m の格子に粗くしてから取り、明るさは日付ごとに順位（パーセンタイル）にそろえる
（版によって 8 ビットへの縮め方が違うので、値そのものではなく順位で比べる）。

手順:
    1. 全日付を、空港の UTM の共通の格子（既定 1 m、範囲は全日付が重なる部分）に置き直す（双一次の補間）
    2. 4 m に粗くし、順位にそろえた明るさの相関（周波数の側で）から、基準（最初の日付）に対するずれを求める（± 200 m まで）
    3. ずれを直した 1 m の格子の画像を、<データの置き場>/processed/umbra_aligned/<空港>/<日付>.tif に保存する
結果: results/align_<空港>.txt（日付ごとのずれ・相関の強さ）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/align_gec.py --airport bkk
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import utm_epsg  # noqa: E402


def rank01(a):
    """0 でない画素の明るさを順位（0〜1）にそろえる。0（画像の外）は 0 のまま。"""
    out = np.zeros_like(a, dtype="float32")
    m = a > 0
    if m.any():
        r = np.argsort(np.argsort(a[m]))
        out[m] = (r + 1) / m.sum()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--airport", default="bkk")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--max-shift-m", type=float, default=200.0)
    a = ap.parse_args()
    d = data_dir()
    files = sorted((d / "raw" / "umbra" / a.airport).glob("*_GEC.tif"))
    bnds = []
    for f in files:
        with rasterio.open(f) as s:
            bnds.append(transform_bounds(s.crs, 4326, *s.bounds))
    w, so, e, n = max(b[0] for b in bnds), max(b[1] for b in bnds), min(b[2] for b in bnds), min(b[3] for b in bnds)
    epsg = utm_epsg((w + e) / 2, (so + n) / 2)
    x0, y0, x1, y1 = transform_bounds(4326, epsg, w, so, e, n)
    W, H = int((x1 - x0) / a.res), int((y1 - y0) / a.res)
    tr = from_origin(x0, y1, a.res, a.res)
    print(f"共通の格子: EPSG {epsg}、{W} × {H} 画素（{a.res} m）")
    out_dir = d / "processed" / "umbra_aligned" / a.airport
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(round(4 / a.res))                                # 4 m に粗くする
    ref_small, recs = None, []
    for f in files:
        dst = np.zeros((H, W), dtype="float32")
        with rasterio.open(f) as s:
            reproject(rasterio.band(s, 1), dst, dst_transform=tr, dst_crs=f"EPSG:{epsg}", resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
        small = ndimage.zoom(rank01(dst), 1 / k, order=1)
        small = small - ndimage.uniform_filter(small, 25)     # ゆっくりした明るさの変化を除く（形の縁だけを比べる）
        if ref_small is None:
            ref_small, dy, dx, peak = small, 0.0, 0.0, float("nan")
        else:
            F = np.fft.fft2(ref_small) * np.conj(np.fft.fft2(small))
            cc = np.real(np.fft.ifft2(F / (np.abs(F) + 1e-6)))   # 位相だけの相関（明るさの物差しの違いに強い）
            lim = int(a.max_shift_m / (a.res * k))
            cc2 = np.full_like(cc, -np.inf)
            idx = np.r_[0:lim + 1, cc.shape[0] - lim:cc.shape[0]]
            idy = np.r_[0:lim + 1, cc.shape[1] - lim:cc.shape[1]]
            cc2[np.ix_(idx, idy)] = cc[np.ix_(idx, idy)]
            iy, ix = np.unravel_index(np.argmax(cc2), cc.shape)
            dy = (iy if iy <= cc.shape[0] // 2 else iy - cc.shape[0]) * k
            dx = (ix if ix <= cc.shape[1] // 2 else ix - cc.shape[1]) * k
            peak = float(cc2[iy, ix] / cc.std())
        aligned = ndimage.shift(dst, (dy, dx), order=0, cval=0)  # 画素単位で、基準に重なるようにずらす
        with rasterio.open(out_dir / f"{f.name[:19]}.tif", "w", driver="GTiff", width=W, height=H, count=1, dtype="float32",
                           crs=f"EPSG:{epsg}", transform=tr, compress="deflate", tiled=True) as o:
            o.write(aligned, 1)
        rec = dict(airport=a.airport, date=f.name[:19], shift_east_m=round(-dx * a.res, 1), shift_north_m=round(dy * a.res, 1),
                   peak_strength=round(peak, 1) if np.isfinite(peak) else None, src_crs=str(rasterio.open(f).crs))
        recs.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    with open(os.path.join(ROOT, "results", f"align_{a.airport}.txt"), "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
