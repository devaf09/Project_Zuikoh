"""est_01b: タンクの強い点は背景よりどれだけ明るいか — est_01 の「点と雑音の比（SCR）」の手がかり

着想: est_01 で、屋根の高さを測れるかは SCR（強い点の明るさ / 周りの斑点の明るさ）でほぼ決まると分かった。実物の値の目安を、手元の RTC（10 m）で測る。
方法: OSM のタンクの輪郭ごとに、中心の周り（半径 + 2 画素）で VV のいちばん明るい画素を取り、10 km 四方の VV の中央値（ほとんどが畑）との比を dB で数える。直径の帯ごとにも分ける
注意: RTC は平均して斑点を減らした画像で、SLC（1 回分の画像）の SCR とは違う。いちばん明るい点が屋根と壁の角とは限らない（壁の上端・屋根の縁かもしれない）。目安として扱う
結果: results/est01b.txt（1 行 JSON）。実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01b_tank_peak_ratio.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import transform

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="est_01b: タンクの強い点の明るさの比")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est01b.txt"))
    a = ap.parse_args()

    d = data_dir()
    vv = sorted((d / "raw" / "s1_rtc" / a.region).glob("*_vv.tif"))[-1]
    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    with rasterio.open(vv) as s:
        img, T, crs = s.read(1).astype("float64"), s.transform, s.crs
    bg = np.nanmedian(img[img > 0])                        # 背景 = 範囲全体の中央値（ほとんどが畑）
    ratio, diam = [], []
    for f in tanks["features"]:
        lon, lat = zip(*f["geometry"]["coordinates"][0])
        x, y = transform("EPSG:4326", crs, lon, lat)       # 輪郭を画像の座標（UTM、m）へ
        r = (max(x) - min(x)) / 2
        col, row = ~T * (np.mean(x), np.mean(y))
        col, row, k = int(col), int(row), int(max(2, r / 10)) + 2
        win = img[max(0, row - k):row + k + 1, max(0, col - k):col + k + 1]
        if win.size:
            ratio.append(10 * np.log10(win.max() / bg))
            diam.append(2 * r)
    ratio, diam = np.array(ratio), np.array(diam)
    rec = dict(exp="est01b", region=a.region, scene=vv.name.replace("_vv.tif", ""), n_tanks=int(len(ratio)),
               bg_db=round(float(10 * np.log10(bg)), 1), ratio_db_median=round(float(np.median(ratio)), 1),
               ratio_db_q25=round(float(np.percentile(ratio, 25)), 1), ratio_db_q75=round(float(np.percentile(ratio, 75)), 1),
               frac_ge_15db=round(float((ratio >= 15).mean()), 3), diam_m_median=round(float(np.median(diam)), 1),
               by_diam={f"{lo}-{hi}m": dict(n=int(((diam >= lo) & (diam < hi)).sum()),
                                            median_db=round(float(np.median(ratio[(diam >= lo) & (diam < hi)])), 1)
                                            if ((diam >= lo) & (diam < hi)).any() else None)
                        for lo, hi in [(0, 30), (30, 60), (60, 200)]})
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
