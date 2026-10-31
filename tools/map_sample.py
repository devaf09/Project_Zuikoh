"""M0 の地図: Sentinel-1 の後方散乱（VV）と OSM のタンクを、ブラウザで開ける地図（Leaflet）に重ねる。

画像を経度・緯度の格子に置き直して PNG にし、背景の地図（OpenStreetMap のタイル）の上に重ねる 1 つの HTML を作る。
保存先: <データの置き場>/processed/maps/<地域>/index.html（データを含むので git には入れない）

実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/map_sample.py --region cushing
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
from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .info{{background:#fff;padding:6px 8px;font:13px sans-serif;border-radius:4px}}</style>
</head><body><div id="map"></div><script>
const map = L.map('map');
// 開いたときに枠の大きさが 0 のことがある（裏のタブなど）ので、大きさを測り直してから範囲に合わせる
const fit = () => {{ map.invalidateSize(); map.fitBounds({bounds}); }};
fit(); window.addEventListener('load', fit);
const osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const sar = L.imageOverlay('{png}', {bounds}, {{opacity: 0.8,
  attribution: 'Contains modified Copernicus Sentinel data 2026 (Microsoft Planetary Computer, CC BY 4.0)'}}).addTo(map);
const tanks = L.geoJSON({tanks}, {{style: {{color: '#ff7f0e', weight: 1, fillOpacity: 0}}}}).addTo(map);
L.control.layers(null, {{'Sentinel-1 VV（後方散乱）': sar, 'OSM のタンク': tanks}}).addTo(map);
const info = L.control({{position: 'bottomleft'}});
info.onAdd = () => {{ const d = L.DomUtil.create('div', 'info'); d.innerHTML = '{caption}'; return d; }};
info.addTo(map);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="cushing")
    a = ap.parse_args()

    d = data_dir() / "raw" / "s1_rtc" / a.region
    vv_path = sorted(d.glob("*_vv.tif"))[-1]
    out_dir = data_dir() / "processed" / "maps" / a.region
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(vv_path) as src:                     # UTM → 経度・緯度の格子に置き直す
        tr, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(dst > 0, 10 * np.log10(dst), np.nan)
    lo, hi = np.nanpercentile(db, [2, 98])
    rgba = plt.get_cmap("gray")(np.clip((db - lo) / (hi - lo), 0, 1))
    rgba[..., 3] = np.where(np.isnan(db), 0, 1)              # 欠けは透明に
    plt.imsave(out_dir / "vv.png", rgba)
    west, north = tr.c, tr.f
    east, south = west + tr.a * w, north + tr.e * h
    bounds = json.dumps([[south, west], [north, east]])
    tanks = (data_dir() / "raw" / "osm" / a.region / "storage_tanks.geojson").read_text(encoding="utf-8")
    scene = vv_path.name.replace("_rtc_vv.tif", "")
    html = HTML.format(title=f"Zuikoh M0 見本 {a.region}", bounds=bounds, png="vv.png", tanks=tanks,
                       caption=f"Sentinel-1 VV（{scene[17:25]}）と OSM のタンク（橙）。右上で表示を切り替え")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
