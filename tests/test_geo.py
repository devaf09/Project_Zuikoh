"""地図の座標の計算（zuikoh.geo）の確認。"""
import math

from zuikoh.geo import bbox_around, blur_point, from_utm, to_utm, utm_epsg


def test_utm_roundtrip():
    """経度・緯度 → UTM → 経度・緯度 で元に戻る（1e-9 度 ≈ 0.1 mm 以内）。"""
    for lon, lat in [(-96.755, 35.955), (131.42, 31.91), (0.1, -45.0)]:
        x, y, epsg = to_utm(lon, lat)
        lon2, lat2 = from_utm(x, y, epsg)
        assert abs(lon - lon2) < 1e-9 and abs(lat - lat2) < 1e-9


def test_utm_zone():
    """クッシング（西経 96.8 度）は 14 番の帯、宮崎（東経 131 度）は 52 番。"""
    assert utm_epsg(-96.755, 35.955) == 32614
    assert utm_epsg(131.42, 31.91) == 32652


def test_bbox_size():
    """中心から 5 km ずつの範囲を、中心を通る東西・南北の線で UTM で測ると約 10 km（1% 以内）。

    角どうしで測ると、UTM の格子が真北から少し回っている（子午線収差、この場所で約 1 度）ぶん、ずれて見える。
    """
    lon, lat = -96.755, 35.955
    w, s, e, n = bbox_around(lon, lat, half_km=5)
    xw, _, epsg = to_utm(w, lat)
    xe, _, _ = to_utm(e, lat, epsg)
    _, ys, _ = to_utm(lon, s, epsg)
    _, yn, _ = to_utm(lon, n, epsg)
    assert math.isclose(xe - xw, 10_000, rel_tol=0.01)
    assert math.isclose(yn - ys, 10_000, rel_tol=0.01)


def test_blur_same_cell():
    """同じ格子の中の 2 点は同じ位置にぼかされ、元の位置からは格子の対角の半分以内。"""
    a = blur_point(-96.7551, 35.9551, grid_km=5)
    b = blur_point(-96.7552, 35.9552, grid_km=5)
    assert a == b
    step = math.degrees(5000 / 6_371_000.0)
    assert abs(a[0] + 96.7551) <= step and abs(a[1] - 35.9551) <= step
