"""M1 の図: クッシングのタンクと畑の後方散乱の 1 年の時系列（docs/images/m1_cushing_timeseries.png）

タンク（OSM の輪郭の中の最も明るい画素）の中央値と、範囲全体（ほとんど畑）の中央値を日付ごとに並べる。
衛星（Sentinel-1A・1C・1D）ごとに印を変え、衛星が替わって明るさの物差しがずれていないかを見る。
数値は results/m1_timeseries.txt（1 日付 = 1 行 JSON）にも書く。

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_m1_timeseries.py --region cushing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def tank_cells(tanks, src):
    """タンクごとに、輪郭の中心と半径から、画像の中で調べる四角（行・列の範囲）を作る。"""
    cells = []
    for f in tanks["features"]:
        lon, lat = zip(*f["geometry"]["coordinates"][0])
        x, y = transform("EPSG:4326", src.crs, lon, lat)
        r = (max(x) - min(x)) / 2
        c, rw = ~src.transform * (np.mean(x), np.mean(y))
        k = int(max(2, r / 10)) + 1
        cells.append((int(rw) - k, int(rw) + k + 1, int(c) - k, int(c) + k + 1))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "images", "m1_cushing_timeseries.png"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "m1_timeseries.txt"))
    a = ap.parse_args()

    d = data_dir()
    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    rows = []
    for pol in ["vv", "vh"]:
        for path in sorted((d / "raw" / "s1_rtc" / a.region).glob(f"*_{pol}.tif")):
            with rasterio.open(path) as src:
                img = src.read(1).astype("float64")
                cells = tank_cells(tanks, src)
            peak = [img[r0:r1, c0:c1].max() for r0, r1, c0, c1 in cells if img[r0:r1, c0:c1].size]
            bg = np.nanmedian(img[img > 0])
            name = path.name
            rows.append(dict(pol=pol, scene=name.replace(f"_{pol}.tif", ""), sat=name[:3],
                             date=name[17:25], tank_db=float(10 * np.log10(np.median(peak))),
                             bg_db=float(10 * np.log10(bg))))
    with open(a.json, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    marks = {"S1A": "o", "S1C": "s", "S1D": "^"}
    for ax, pol in zip(axes, ["vv", "vh"]):
        for sat, m in marks.items():
            rs = [r for r in rows if r["pol"] == pol and r["sat"] == sat]
            if not rs:
                continue
            t = [dt.datetime.strptime(r["date"], "%Y%m%d") for r in rs]
            ax.plot(t, [r["tank_db"] for r in rs], m, color="#d95f02", label=f"タンク（最も明るい点の中央値）{sat}")
            ax.plot(t, [r["bg_db"] for r in rs], m, color="#1b9e77", mfc="none", label=f"範囲全体（ほぼ畑）の中央値 {sat}")
        ax.set_ylabel(f"{pol.upper()} 後方散乱（dB）")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("日付")
    fig.suptitle("クッシング 10 km 四方: タンク 333 基と畑の後方散乱の 1 年（Sentinel-1、上昇軌道 34）", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2025–2026（Microsoft Planetary Computer, sentinel-1-rtc, CC BY 4.0）。"
             "タンクの位置 © OpenStreetMap contributors（ODbL）", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
