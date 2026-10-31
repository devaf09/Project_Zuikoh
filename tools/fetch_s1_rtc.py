"""Sentinel-1 RTC（地形の影響をならした後方散乱）を、Planetary Computer から範囲を切り出して取る。

後方散乱 = レーダの電波がどれだけ戻ってきたか。VV・VH は偏波（電波の振動の向き）の組み合わせ（縦で送って縦で受ける・横で受ける）。
目録（STAC）で場面を探し、COG（必要な部分だけ読める GeoTIFF）から範囲だけを読むので、取る量は小さい。
アカウントは要らない（読み出しの鍵 SAS は誰でも取れる）。ライセンスは CC BY 4.0。

取ったものは <データの置き場>/raw/s1_rtc/<地域>/<場面の ID>_<偏波>.tif に保存し、MANIFEST.md に 1 行足す。

実行（例: クッシングの 10 km 四方、最新の 1 場面）:
    CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_s1_rtc.py --region cushing --lon -96.755 --lat 35.955 --half-km 5 --start 2026-10-01 --end 2026-10-31 --max-items 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import planetary_computer  # noqa: E402
import pystac_client  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import bbox_around  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search(bbox, start, end, max_items):
    """範囲と期間で sentinel-1-rtc の場面を新しい順に探す。"""
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    s = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}", max_items=max_items,
                   sortby=[{"field": "properties.datetime", "direction": "desc"}])
    return list(s.items())


def cut(href, bbox, out_path):
    """COG から経度・緯度の範囲だけを読み、同じ座標系の GeoTIFF に保存する。読んだ大きさ（画素）を返す。"""
    with rasterio.open(href) as src:
        b = transform_bounds(4326, src.crs, *bbox)          # 経度・緯度 → 画像の座標系（UTM）
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(width=data.shape[1], height=data.shape[0], transform=src.window_transform(win),
                    driver="GTiff", tiled=True, compress="deflate", blockxsize=256, blockysize=256)
        prof.pop("photometric", None)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data, 1)
    return data.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True, help="地域の名前（保存先のフォルダ名）")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half-km", type=float, default=5.0, help="中心から東西南北への距離（km）")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-items", type=int, default=1)
    ap.add_argument("--pols", default="vv,vh")
    a = ap.parse_args()

    bbox = bbox_around(a.lon, a.lat, a.half_km)
    out_dir = data_dir() / "raw" / "s1_rtc" / a.region
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir() / "MANIFEST.md"
    for item in search(bbox, a.start, a.end, a.max_items):
        for pol in a.pols.split(","):
            out = out_dir / f"{item.id}_{pol}.tif"
            t0 = time.time()
            shape = cut(item.assets[pol].href, bbox, out)
            mb = out.stat().st_size / 1e6
            print(f"{item.id} {pol}: {shape[0]}x{shape[1]} 画素, {mb:.1f} MB, {time.time() - t0:.1f} 秒")
            rel = out.relative_to(data_dir()).as_posix()
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(f"| Sentinel-1 RTC {pol.upper()}（{a.region}、{a.half_km * 2:g} km 四方の切り出し） | {rel} | "
                        f"Planetary Computer sentinel-1-rtc | {item.id} | {dt.date.today()} | {mb:.1f} MB | CC BY 4.0 | "
                        f"出典の表記をすれば再配布可（Contains modified Copernicus Sentinel data、Microsoft Planetary Computer） | tools/fetch_s1_rtc.py |\n")


if __name__ == "__main__":
    main()
