"""NAIP（アメリカの航空写真、0.6〜1 m、赤・緑・青・近赤外）を、Planetary Computer から場所の周りの範囲で取り、1 枚にまとめる。

格子の正解（飛行機あり・車あり）を目で印をつけるための下絵（持ち主の案の確かめ、2026-11-01）。
アカウントは要らない。ライセンス: アメリカ政府（USDA）の著作物（制限なし、出典の表記「USDA NAIP」）。
同じ場所の一番新しい年の画像を集め、UTM の共通の格子（既定 2 m）に置き直して、赤・緑・青の 3 色の GeoTIFF にする。

保存先: <データの置き場>/processed/naip/<名前>.tif（MANIFEST.md に 1 行足す）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_naip.py --name roswell --lon -104.53 --lat 33.30 --half-km 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import planetary_computer  # noqa: E402
import pystac_client  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import Resampling, reproject, transform_bounds  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import bbox_around, utm_epsg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half-km", type=float, default=5.0)
    ap.add_argument("--res", type=float, default=2.0)
    a = ap.parse_args()
    bbox = bbox_around(a.lon, a.lat, a.half_km)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["naip"], bbox=bbox, max_items=200).items())
    year = max(int(i.properties["datetime"][:4]) for i in items)
    items = [i for i in items if i.properties["datetime"].startswith(str(year))]
    epsg = utm_epsg(a.lon, a.lat)
    x0, y0, x1, y1 = transform_bounds(4326, epsg, *bbox)
    W, H = int((x1 - x0) / a.res), int((y1 - y0) / a.res)
    tr = from_origin(x0, y1, a.res, a.res)
    out = np.zeros((3, H, W), dtype="uint8")
    for it in items:                                           # 四角ごとに置き直して重ねる（0 の所だけ埋める）
        with rasterio.open(it.assets["image"].href) as s:
            for b in range(3):
                tmp = np.zeros((H, W), dtype="uint8")
                reproject(rasterio.band(s, b + 1), tmp, dst_transform=tr, dst_crs=f"EPSG:{epsg}", resampling=Resampling.average)
                out[b] = np.where(out[b] == 0, tmp, out[b])
    out_dir = data_dir() / "processed" / "naip"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=f"EPSG:{epsg}", transform=tr,
                       compress="deflate", tiled=True) as o:
        o.write(out)
    mb = path.stat().st_size / 1e6
    print(f"{a.name}: NAIP {year} 年、四角 {len(items)} 枚 → {W} × {H}（{a.res} m）、{mb:.0f} MB")
    with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
        fo.write(f"| NAIP の航空写真（{a.name}、±{a.half_km:g} km、{a.res} m に置き直し） | {path.relative_to(data_dir()).as_posix()} | "
                 f"Planetary Computer naip | {year} 年の {len(items)} 枚 | {dt.date.today()} | {mb:.0f} MB | アメリカ政府の著作物（制限なし） | "
                 f"出典「USDA NAIP」 | tools/fetch_naip.py |\n")


if __name__ == "__main__":
    main()
