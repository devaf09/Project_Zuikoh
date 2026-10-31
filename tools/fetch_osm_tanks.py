"""OpenStreetMap からタンク（man_made=storage_tank）の形を範囲で取る（M1 の正解の下書き）。

Overpass API（OSM の検索の窓口）に問い合わせ、タンクの輪郭を GeoJSON で保存する。アカウントは要らない。
ライセンスは ODbL（出典「© OpenStreetMap contributors」。作った地図データを配るときは同じ条件で共有する）。
OSM には抜けと誤りがあるので、航空写真（NAIP）で一部を数えて確かめる（docs/design_m0_m1.md）。

保存先: <データの置き場>/raw/osm/<地域>/storage_tanks.geojson（MANIFEST.md に 1 行足す）

実行:
    CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_osm_tanks.py --region cushing --lon -96.755 --lat 35.955 --half-km 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import bbox_around  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch(bbox):
    """範囲 (西, 南, 東, 北) のタンクの輪郭（way）を取り、GeoJSON の Feature の列にする。"""
    w, s, e, n = bbox
    q = f'[out:json][timeout:120];way["man_made"="storage_tank"]({s},{w},{n},{e});out geom tags;'
    r = requests.post(OVERPASS, data={"data": q}, timeout=180,
                      headers={"User-Agent": "Project_Zuikoh (research; github.com/devaf09/Project_Zuikoh)"})
    r.raise_for_status()
    feats = []
    for el in r.json()["elements"]:
        ring = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        if len(ring) < 4 or ring[0] != ring[-1]:   # 閉じていない線はタンクの輪郭として使わない
            continue
        feats.append({"type": "Feature", "properties": {"osm_id": el["id"], **el.get("tags", {})},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half-km", type=float, default=5.0)
    a = ap.parse_args()

    feats = fetch(bbox_around(a.lon, a.lat, a.half_km))
    out_dir = data_dir() / "raw" / "osm" / a.region
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "storage_tanks.geojson"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    kb = out.stat().st_size / 1e3
    print(f"タンク {len(feats)} 個, {kb:.0f} KB")
    with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as f:
        f.write(f"| OSM のタンクの輪郭（{a.region}、{a.half_km * 2:g} km 四方、{len(feats)} 個） | {out.relative_to(data_dir()).as_posix()} | "
                f"Overpass API | man_made=storage_tank | {dt.date.today()} | {kb:.0f} KB | ODbL | "
                f"出典「© OpenStreetMap contributors」、派生した地図データは同じ条件で共有 | tools/fetch_osm_tanks.py |\n")


if __name__ == "__main__":
    main()
