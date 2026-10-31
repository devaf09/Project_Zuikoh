"""M0 の図: クッシングの Sentinel-1 の後方散乱（VV・VH）に OSM のタンクの輪郭を重ねる（docs/images/m0_cushing_sample.png）。

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fig_m0_sample.py --region cushing
データは tools/fetch_s1_rtc.py・tools/fetch_osm_tanks.py で取ったもの（データの置き場の raw/）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]


def to_db(a):
    """後方散乱の強さ（線形）→ dB（10·log10）。0 以下は欠け（NaN）にする。"""
    a = a.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, 10 * np.log10(a), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "images", "m0_cushing_sample.png"))
    a = ap.parse_args()

    d = data_dir() / "raw" / "s1_rtc" / a.region
    vv_path = sorted(d.glob("*_vv.tif"))[-1]                 # いちばん新しい場面
    vh_path = vv_path.with_name(vv_path.name.replace("_vv.tif", "_vh.tif"))
    scene = vv_path.name.replace("_rtc_vv.tif", "")
    tanks = json.load(open(data_dir() / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, path, title in [(axes[0], vv_path, "VV（縦に送って縦で受ける）"), (axes[1], vh_path, "VH（縦に送って横で受ける）")]:
        with rasterio.open(path) as src:
            db = to_db(src.read(1))
            crs, (left, bottom, right, top) = src.crs, src.bounds
        lo, hi = np.nanpercentile(db, [2, 98])
        im = ax.imshow(db, cmap="gray", vmin=lo, vmax=hi, extent=[left / 1e3, right / 1e3, bottom / 1e3, top / 1e3])
        for f in tanks["features"]:                       # タンクの輪郭を画像の座標（UTM）に直して描く
            lon, lat = zip(*f["geometry"]["coordinates"][0])
            x, y = warp_transform("EPSG:4326", crs, lon, lat)
            ax.plot(np.array(x) / 1e3, np.array(y) / 1e3, color="#ff7f0e", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("東西（km、UTM 14N）")
        ax.set_ylabel("南北（km、UTM 14N）")
        fig.colorbar(im, ax=ax, shrink=0.75, label="後方散乱（dB）")
    fig.suptitle(f"クッシング（オクラホマ州）10 km 四方: Sentinel-1 の後方散乱と OSM のタンク {len(tanks['features'])} 個（橙の輪郭）\n"
                 f"{scene}", fontsize=11)
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data 2026（Microsoft Planetary Computer, sentinel-1-rtc, CC BY 4.0）。"
             "タンクの輪郭 © OpenStreetMap contributors（ODbL）", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
