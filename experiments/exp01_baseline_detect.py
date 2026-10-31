"""実験01: 定番の検出でタンクを見つける — CFAR とテンプレートの照合（M1 の比較相手の最初の値）

着想: M1 の設計（docs/design_m0_m1.md）。Sin レイヤフィルムと比べる前に、学習しない定番の手法がどこまで届くかを測る。
方法（クッシング 10 km 四方、Sentinel-1 RTC の VV、1 日付）:
    cfar      セル平均の CFAR（guard 3・train 8 画素）。pfa を変えて、見つけた山（近い山は min_dist 画素でまとめる）を数える
    template  明るい円（直径 6 画素 = 60 m）の見本との相関（dB の画像）。上位からしきい値を変えて山を数える
    chance    同じ数の山をでたらめな位置に置いたとき（偶然の水準）
正解: OSM の man_made=storage_tank の輪郭（333 基）。山がタンクの輪郭の中（+ 10 m の余裕）にあれば当たり。
      1 基に何個当たっても見つけた数は 1。どのタンクにも入らない山は誤報
指標: 再現率（見つけたタンク / 全タンク）、適合率（当たった山 / 全部の山）、F1、1 km² あたりの誤報
結果: results/exp01.txt（条件ごとに 1 行 JSON）。実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/exp01_baseline_detect.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.warp import transform

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402
from zuikoh.detect import ca_cfar, disk_template, peaks, template_match  # noqa: E402


def tank_disks(tanks, src):
    """タンクを (中心の行, 中心の列, 半径の画素) の配列にする（輪郭を円で近似）。"""
    out = []
    px = src.transform.a
    for f in tanks["features"]:
        lon, lat = zip(*f["geometry"]["coordinates"][0])
        x, y = transform("EPSG:4326", src.crs, lon, lat)
        c, r = ~src.transform * (np.mean(x), np.mean(y))
        out.append((r, c, (max(x) - min(x)) / 2 / px))
    return np.array(out)


def score(pts, disks, margin_px, area_km2):
    """山の位置 pts と正解の円 disks から、再現率・適合率・F1・誤報の密度を数える。"""
    if len(pts) == 0:
        return dict(n_det=0, recall=0.0, precision=0.0, f1=0.0, fa_per_km2=0.0)
    d = np.hypot(pts[:, None, 0] - disks[None, :, 0], pts[:, None, 1] - disks[None, :, 1])
    inside = d <= disks[None, :, 2] + margin_px
    hit_pt = inside.any(1)
    hit_tank = inside.any(0)
    rec, prec = hit_tank.mean(), hit_pt.mean()
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    return dict(n_det=int(len(pts)), recall=round(float(rec), 4), precision=round(float(prec), 4),
                f1=round(float(f1), 4), fa_per_km2=round(float((~hit_pt).sum() / area_km2), 3))


def main():
    ap = argparse.ArgumentParser(description="実験01: 定番の検出")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--date", default="20261024")
    ap.add_argument("--min-dist", type=int, default=4, help="近い山をまとめる距離（画素）")
    ap.add_argument("--seed", type=int, default=0, help="偶然の水準のでたらめな位置に使う")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp01.txt"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = data_dir()
    path = next((d / "raw" / "s1_rtc" / a.region).glob(f"*_{a.date}T*_vv.tif"))
    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    with rasterio.open(path) as src:
        img = src.read(1).astype("float64")
        disks = tank_disks(tanks, src)
        px = src.transform.a
    area = img.size * px * px / 1e6
    margin = 10 / px
    db = 10 * np.log10(np.where(img > 0, img, np.nan))
    base = dict(vars(a), exp="01", scene=path.name.replace("_vv.tif", ""), n_tanks=int(len(disks)), area_km2=round(area, 2),
                params=0)
    rows = []
    for pfa in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        t0 = time.time()
        pts = peaks(ca_cfar(img, guard=3, train=8, pfa=pfa) * img, min_dist=a.min_dist)
        rows.append(dict(base, method="cfar", pfa=pfa, time_s=round(time.time() - t0, 3), **score(pts, disks, margin, area)))
    t0 = time.time()
    s = template_match(db, disk_template(6))
    t_tm = time.time() - t0
    for q in [99.0, 99.5, 99.8, 99.9]:
        thr = np.nanpercentile(s, q)
        pts = peaks(np.where(s > thr, s, 0), min_dist=a.min_dist)
        rows.append(dict(base, method="template", percentile=q, time_s=round(t_tm, 3), **score(pts, disks, margin, area)))
        rnd = np.column_stack([rng.integers(0, img.shape[0], len(pts)), rng.integers(0, img.shape[1], len(pts))])
        rows.append(dict(base, method="chance", matched_to=f"template {q}", **score(rnd, disks, margin, area)))
    with open(a.out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        tag = r.get("pfa", r.get("percentile", r.get("matched_to")))
        print(f"{r['method']:8s} {str(tag):>14s}  山 {r['n_det']:5d}  再現率 {r['recall']:.3f}  適合率 {r['precision']:.3f}  F1 {r['f1']:.3f}  誤報/km² {r['fa_per_km2']:.2f}")


if __name__ == "__main__":
    main()
