"""OpenStreetMap から、格子の正解の割合を作るための面の地物を、場所の周りの範囲で取る（飛行機の成分・車の成分の学習）。

取るもの（面だけ）: aeroway（apron = 駐機場・保管場、runway、taxiway、hangar、terminal）、amenity=parking（駐車場）、
landuse（industrial・commercial・residential・farmland・port など）、building（建物）、natural（水・草地など）。
ライセンス: ODbL（出典「© OpenStreetMap contributors」）。

保存先: <データの置き場>/raw/osm/sites/<名前>.geojson（MANIFEST.md に 1 行足す）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_osm_landcover.py --name roswell --lon -104.53 --lat 33.30 --half-km 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import bbox_around  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half-km", type=float, default=5.0)
    a = ap.parse_args()
    w, s, e, n = bbox_around(a.lon, a.lat, a.half_km)
    bb = f"({s},{w},{n},{e})"
    q = (f'[out:json][timeout:240];(way["aeroway"]{bb};way["amenity"="parking"]{bb};way["landuse"]{bb};'
         f'way["building"]{bb};way["natural"]{bb};relation["landuse"]{bb};);out geom tags;')
    for wait in (0, 30, 60, 120, 240):                         # Overpass が混んでいるときは待って取り直す
        time.sleep(wait)
        r = requests.post(OVERPASS, data={"data": q}, timeout=300,
                          headers={"User-Agent": "Project_Zuikoh (research; github.com/devaf09/Project_Zuikoh)"})
        if r.status_code not in (429, 502, 503, 504):
            break
    r.raise_for_status()
    feats = []
    for el in r.json()["elements"]:
        if el["type"] != "way":                                # 関係（relation）の外周の組み立ては省く（大きな landuse の一部が抜けうる）
            continue
        pts = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            feats.append({"type": "Feature", "properties": {"osm_id": el["id"], **el.get("tags", {})},
                          "geometry": {"type": "Polygon", "coordinates": [pts]}})
    out_dir = data_dir() / "raw" / "osm" / "sites"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.geojson"
    json.dump({"type": "FeatureCollection", "features": feats, "bbox": [w, s, e, n]}, open(out, "w", encoding="utf-8"))
    kinds = {}
    for f in feats:
        p = f["properties"]
        k = p.get("aeroway") and f"aeroway={p['aeroway']}" or p.get("amenity") and f"amenity={p['amenity']}" or \
            p.get("landuse") and f"landuse={p['landuse']}" or ("building" if "building" in p else p.get("natural") and f"natural={p['natural']}")
        kinds[k] = kinds.get(k, 0) + 1
    print(a.name, len(feats), dict(sorted(kinds.items(), key=lambda x: -x[1])[:10]))
    with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
        fo.write(f"| OSM の面の地物（{a.name}、±{a.half_km:g} km、{len(feats)} 個） | {out.relative_to(data_dir()).as_posix()} | Overpass API | "
                 f"aeroway・parking・landuse・building・natural | {dt.date.today()} | {out.stat().st_size / 1e3:.0f} KB | ODbL | "
                 f"出典「© OpenStreetMap contributors」、派生した地図データは同じ条件で共有 | tools/fetch_osm_landcover.py |\n")


if __name__ == "__main__":
    main()
