"""実験00: 取り込みの重さ — 切り出しの大きさごとの時間・メモリ・ディスク

着想: 開始プロンプト 9 章「大きさごとの時間・メモリ・ディスクの量を最初に測る」。M1 の見積もり（docs/design_m0_m1.md）の裏づけ。
方法:
    size_km   切り出しの一辺（km）。中心はクッシング。Sentinel-1 RTC の 1 場面（VV）を COG から範囲だけ読んで保存する
    read      ネットから範囲を読む時間（目録の検索の時間は別に search_s）
    reproject 経度・緯度の格子に置き直す時間（地図に重ねる処理と同じ）
指標: search_s・read_s・write_s・reproject_s（秒）、rss_peak_mb（プロセスのメモリの最大の目安）、file_mb（保存した大きさ）、pixels
結果: results/exp00.txt（1 回 = 1 行の JSON）。保存したファイルは cache/exp00/ に置き、測り終えたら消す（--keep で残す）
実行: CUDA_VISIBLE_DEVICES= nice -n 10 .venv/Scripts/python experiments/exp00_fetch_cost.py --size-km 10 --seed 0 --out results/exp00.txt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
from fetch_s1_rtc import cut, search  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import bbox_around  # noqa: E402


class PeakRSS:
    """別の糸（スレッド）で 0.05 秒ごとにプロセスのメモリ（RSS）を測り、最大を覚える。GDAL の中のメモリも含む。"""

    def __init__(self):
        self.p, self.peak, self.stop = psutil.Process(), 0, False

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self.stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.05)

    def __exit__(self, *exc):
        self.stop = True
        self.t.join()


def main():
    ap = argparse.ArgumentParser(description="実験00: 取り込みの重さ")
    ap.add_argument("--size-km", type=float, required=True)
    ap.add_argument("--lon", type=float, default=-96.755)
    ap.add_argument("--lat", type=float, default=35.955)
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--date", default="2026-10-24", help="この日の場面を使う（毎回同じ場面で測るため）")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu", help="GPU は使わない（決まりとして持つ）")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp00.txt"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    bbox = bbox_around(a.lon, a.lat, a.size_km / 2)
    tmp = data_dir() / "cache" / "exp00"
    tmp.mkdir(parents=True, exist_ok=True)
    with PeakRSS() as mem:
        t0 = time.time()
        item = search(bbox, a.date, a.date, 1)[0]
        t1 = time.time()
        out = tmp / f"{a.size_km:g}km_vv.tif"
        shape = cut(item.assets["vv"].href, bbox, out)
        t2 = time.time()
        with rasterio.open(out) as src:                      # 経度・緯度の格子に置き直す
            tr, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            dst = np.empty((h, w), dtype="float32")
            reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
        t3 = time.time()
    rec = dict(vars(a), exp="00", scene=item.id, region=a.region, pixels=int(shape[0] * shape[1]), shape=list(shape),
               search_s=round(t1 - t0, 2), read_write_s=round(t2 - t1, 2), reproject_s=round(t3 - t2, 2),
               rss_peak_mb=round(mem.peak / 1e6, 1), file_mb=round(out.stat().st_size / 1e6, 2),
               version="zuikoh " + __import__("zuikoh").__version__, torch=torch.__version__,
               rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__)
    rec.pop("keep")
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
