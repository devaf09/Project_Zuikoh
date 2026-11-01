"""OpenStreetMap から空港の地物（aeroway）を、Umbra の GEC の範囲で取る（飛行場の学習の正解の下書き）。

aeroway の種類: runway（滑走路）・taxiway（誘導路）・apron（駐機場）・terminal（ターミナル）・hangar（格納庫）・
parking_position（駐機位置）・gate など。線（滑走路・誘導路の中心線）と面（駐機場・建物）がある。
ライセンス: ODbL（出典「© OpenStreetMap contributors」）。

保存先: <データの置き場>/raw/osm/airports/<名前>.geojson（MANIFEST.md に 1 行足す）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_osm_aeroway.py --names atl,dfw,fra,dxb,pvg,hnd,bkk,syd
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

import rasterio  # noqa: E402
import requests  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"


def bounds_of(name):
    """その空港の GEC（最初の 1 件）の範囲を、経度・緯度の (西, 南, 東, 北) で返す。"""
    f = sorted((data_dir() / "raw" / "umbra" / name).glob("*_GEC.tif"))[0]
    with rasterio.open(f) as s:
        return transform_bounds(s.crs, 4326, *s.bounds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", required=True)
    a = ap.parse_args()
    out_dir = data_dir() / "raw" / "osm" / "airports"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.names.split(","):
        if (out_dir / f"{name}.geojson").exists():           # 取得済みは飛ばす（途中から再開できる）
            continue
        w, s, e, n = bounds_of(name)
        q = f'[out:json][timeout:180];(way["aeroway"]({s},{w},{n},{e});node["aeroway"="parking_position"]({s},{w},{n},{e}););out geom tags;'
        for wait in (0, 30, 60, 120, 240):                    # Overpass が混んでいるとき（502・504・429）は待って取り直す
            time.sleep(wait)
            r = requests.post(OVERPASS, data={"data": q}, timeout=240,
                              headers={"User-Agent": "Project_Zuikoh (research; github.com/devaf09/Project_Zuikoh)"})
            if r.status_code not in (429, 502, 503, 504):
                break
        r.raise_for_status()
        feats = []
        for el in r.json()["elements"]:
            tags = el.get("tags", {})
            if el["type"] == "node":
                geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
            else:
                pts = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
                closed = len(pts) >= 4 and pts[0] == pts[-1] and tags.get("aeroway") not in ("runway", "taxiway")
                geom = {"type": "Polygon", "coordinates": [pts]} if closed else {"type": "LineString", "coordinates": pts}
            feats.append({"type": "Feature", "properties": {"osm_id": el["id"], **tags}, "geometry": geom})
        out = out_dir / f"{name}.geojson"
        json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w", encoding="utf-8"))
        kinds = {}
        for f in feats:
            k = f["properties"].get("aeroway")
            kinds[k] = kinds.get(k, 0) + 1
        print(name, len(feats), dict(sorted(kinds.items(), key=lambda x: -x[1])[:8]))
        with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
            fo.write(f"| OSM の空港の地物（{name}、{len(feats)} 個） | {out.relative_to(data_dir()).as_posix()} | Overpass API | aeroway=* | "
                     f"{dt.date.today()} | {out.stat().st_size / 1e3:.0f} KB | ODbL | 出典「© OpenStreetMap contributors」、派生した地図データは同じ条件で共有 | tools/fetch_osm_aeroway.py |\n")
        time.sleep(5)                                          # Overpass に負担をかけない


if __name__ == "__main__":
    main()
