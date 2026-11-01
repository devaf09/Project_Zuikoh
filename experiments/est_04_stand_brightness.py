"""est_04: 駐機位置ごとの明るさの時系列 — 飛行機が「いる日」と「いない日」に分かれるか

着想: 持ち主の案（2026-11-01）「飛行機があるエリアとないエリアで板が学習できれば、次は駐機機種の割合」。
      飛行機 1 機ずつの正解は公開データにない。駐機位置（OSM の aeroway=parking_position）の明るさが、日付ごとに
      「強い日（金属の機体がいる）」と「弱い日（なめらかな舗装だけ = 電波が鏡のように逃げて暗い）」に分かれれば、それを仮の正解にできる。
方法:
    データ: Umbra の GEC（地図の座標に置いた明るさ、0.3〜0.6 m 格子、8 ビット）、空港ごとの全日付。駐機位置 = OSM の点
    1. 日付ごとに、駐機位置の周り半径 R m（既定 20 m）の明るさの平均を取る
    2. 日付ごとの明るさの物差しの違い（8 ビットへの縮め方が画像ごとに違う）をならすため、
       その日付の駐機場（OSM の apron の面）全体の明るさの中央値で割る（dB の差にする）
    3. 駐機位置ごとに、日付をまたいだ値の分布が 2 つの山に分かれるか（2 つの正規分布の当てはめ、山の間の離れ具合）を見る
指標: 駐機位置の値の分布、2 つに分けたときの山の離れ具合（d' = 平均の差 / ばらつき）、「強い日」の割合
結果: results/est04.txt（空港ごとに 1 行 JSON）、駐機位置 × 日付の表は cache/est04_<空港>.csv
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_04_stand_brightness.py --airport bkk --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform, transform_geom

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402


def meters_per_pixel(src, lat):
    """画像の 1 画素が何 m か（経度・緯度の画像なら緯度で直す）。"""
    if src.crs.to_epsg() == 4326:
        return src.res[0] * 111320 * math.cos(math.radians(lat)), src.res[1] * 110540
    return src.res[0], src.res[1]


def disc_mean(src, x, y, r_px):
    """(x, y)（画像の座標）の周り半径 r_px 画素の明るさの平均（8 ビットの値 → 振幅に直して 2 乗 = 強さ）。外なら None。"""
    col, row = ~src.transform * (x, y)
    col, row = int(col), int(row)
    if not (r_px <= row < src.height - r_px and r_px <= col < src.width - r_px):
        return None
    a = src.read(1, window=((row - r_px, row + r_px + 1), (col - r_px, col + r_px + 1))).astype("float64")
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    m = (yy ** 2 + xx ** 2) <= r_px ** 2
    v = a[m]
    if (v == 0).mean() > 0.5:                                  # 画像の外（0 で埋まっている）
        return None
    return float((v ** 2).mean())


def main():
    ap = argparse.ArgumentParser(description="est_04: 駐機位置ごとの明るさの時系列")
    ap.add_argument("--airport", default="bkk")
    ap.add_argument("--radius-m", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0, help="2 つの山の当てはめの初期値の乱数")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est04.txt"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = data_dir()
    osm = json.load(open(d / "raw" / "osm" / "airports" / f"{a.airport}.geojson", encoding="utf-8"))
    stands = []                                               # 駐機位置は点のことも、機首の向きの線のこともある → 線なら平均の位置
    for f in osm["features"]:
        if f["properties"].get("aeroway") != "parking_position":
            continue
        g = f["geometry"]
        pts = [g["coordinates"]] if g["type"] == "Point" else (g["coordinates"] if g["type"] == "LineString" else g["coordinates"][0])
        stands.append([float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))])
    aprons = [f["geometry"] for f in osm["features"]
              if f["properties"].get("aeroway") == "apron" and f["geometry"]["type"] == "Polygon"]
    files = sorted((d / "raw" / "umbra" / a.airport).glob("*_GEC.tif"))
    lon = np.array([p[0] for p in stands])
    lat = np.array([p[1] for p in stands])
    table = np.full((len(stands), len(files)), np.nan)
    for j, f in enumerate(files):
        with rasterio.open(f) as src:
            xs, ys = transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
            mx, _ = meters_per_pixel(src, lat.mean())
            r_px = max(2, int(round(a.radius_m / mx)))
            # 駐機場の明るさの中央値（日付ごとの物差し）: 駐機場の面の中を間引いて読む
            vals = []
            for g in aprons:
                gg = transform_geom("EPSG:4326", src.crs, g)
                xs_g = [p[0] for p in gg["coordinates"][0]]
                ys_g = [p[1] for p in gg["coordinates"][0]]
                # 画像の行・列に直して窓を作る（上下が逆さの画像もあるので、from_bounds は使わない）
                cs, rs = zip(*[~src.transform * (x_, y_) for x_, y_ in zip(xs_g, ys_g)])
                c0, c1 = max(0, int(min(cs))), min(src.width, int(max(cs)) + 1)
                r0, r1 = max(0, int(min(rs))), min(src.height, int(max(rs)) + 1)
                if c1 - c0 < 2 or r1 - r0 < 2:
                    continue
                win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                step = max(1, int(max(win.width, win.height) // 400))
                arr = src.read(1, window=win, out_shape=(max(1, int(win.height // step)), max(1, int(win.width // step)))).astype("float64")
                tr = src.window_transform(win) * rasterio.Affine.scale(win.width / arr.shape[1], win.height / arr.shape[0])
                m = ~geometry_mask([gg], out_shape=arr.shape, transform=tr)
                v = arr[m & (arr > 0)]
                if v.size:
                    vals.append(v ** 2)
            ref = float(np.median(np.concatenate(vals))) if vals else np.nan
            for i, (x, y) in enumerate(zip(xs, ys)):
                v = disc_mean(src, x, y, r_px)
                if v is not None and ref > 0:
                    table[i, j] = 10 * math.log10(v / ref + 1e-12)
        print(f"{f.name[:19]}: 駐機場の中央値 {10 * math.log10(ref):.1f} dB（8 ビットの値の 2 乗）、測れた駐機位置 {np.isfinite(table[:, j]).sum()}")
    # 駐機位置ごとに 2 つの山（強い日・弱い日）に分ける: 1 次元の 2 つの正規分布の当てはめ（EM）
    seps, fracs = [], []
    allv = table[np.isfinite(table)]
    for i in range(len(stands)):
        v = table[i][np.isfinite(table[i])]
        if len(v) < 6:
            continue
        mu = np.sort(rng.choice(v, 2, replace=False))
        sd = np.array([v.std() + 1e-6] * 2)
        w = np.array([0.5, 0.5])
        for _ in range(100):
            p = np.stack([w[k] * np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) / sd[k] for k in range(2)]) + 1e-12
            r = p / p.sum(0)
            w = r.mean(1)
            mu = (r * v).sum(1) / r.sum(1)
            sd = np.sqrt((r * (v - mu[:, None]) ** 2).sum(1) / r.sum(1)) + 0.3
        k_hi = int(np.argmax(mu))
        seps.append(abs(mu[1] - mu[0]) / math.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2))
        fracs.append(float(w[k_hi]))
    seps, fracs = np.array(seps), np.array(fracs)
    rec = dict(exp="est04", airport=a.airport, n_dates=len(files), n_stands=len(stands), n_stands_used=int(len(seps)),
               radius_m=a.radius_m, value_db_median=round(float(np.median(allv)), 2),
               value_db_p10=round(float(np.percentile(allv, 10)), 2), value_db_p90=round(float(np.percentile(allv, 90)), 2),
               dprime_median=round(float(np.median(seps)), 2), frac_stands_dprime_gt2=round(float((seps > 2).mean()), 3),
               frac_bright_days_median=round(float(np.median(fracs)), 3))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savetxt(d / "cache" / f"est04_{a.airport}.csv", np.column_stack([lon, lat, table]), delimiter=",",
               header="lon,lat," + ",".join(f.name[:19] for f in files), fmt="%.5f")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
