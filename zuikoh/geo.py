"""zuikoh.geo — 地図の座標の計算（場所の範囲・座標の変換・位置のぼかし）。

    from zuikoh.geo import bbox_around, to_utm, from_utm, blur_point
    box = bbox_around(-96.755, 35.955, half_km=5)     # 中心から 5 km 四方（経度・緯度の範囲）
    x, y, epsg = to_utm(-96.755, 35.955)                # メートル単位の平らな座標（UTM）へ
    lon, lat = from_utm(x, y, epsg)                     # 往復で元に戻る（tests/test_geo.py）
    blur_point(lon, lat, grid_km=5)                     # 公開の図に使う、ぼかした位置（9 章）

言葉: 経度・緯度 = 地球の上の角度（度）。UTM = 地球を細長い帯に分け、帯ごとに平らに広げた座標（メートル）。
EPSG = 座標の決め方につけた番号（4326 が経度・緯度、326xx が北半球の UTM の xx 番の帯）。

ライセンス: MIT
"""
from __future__ import annotations

import math

from pyproj import Transformer

# 地球の半径（メートル）。範囲の大まかな計算だけに使う
EARTH_R = 6_371_000.0


def bbox_around(lon: float, lat: float, half_km: float) -> tuple[float, float, float, float]:
    """中心 (lon, lat) から東西南北に half_km の範囲を (西, 南, 東, 北) の度で返す（球として近似）。"""
    dlat = math.degrees(half_km * 1000 / EARTH_R)
    dlon = dlat / math.cos(math.radians(lat))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utm_epsg(lon: float, lat: float) -> int:
    """その場所の UTM の帯の EPSG 番号（北半球 326xx、南半球 327xx）。"""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def to_utm(lon: float, lat: float, epsg: int | None = None) -> tuple[float, float, int]:
    """経度・緯度 → UTM のメートル座標 (x, y, EPSG)。"""
    epsg = epsg or utm_epsg(lon, lat)
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = t.transform(lon, lat)
    return x, y, epsg


def from_utm(x: float, y: float, epsg: int) -> tuple[float, float]:
    """UTM のメートル座標 → 経度・緯度。"""
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    return t.transform(x, y)


def blur_point(lon: float, lat: float, grid_km: float = 5.0) -> tuple[float, float]:
    """位置を grid_km の格子の中心に丸める（未知の遺跡などの正確な座標を外に出さないため。9 章）。"""
    step = math.degrees(grid_km * 1000 / EARTH_R)
    return ((math.floor(lon / step) + 0.5) * step, (math.floor(lat / step) + 0.5) * step)
