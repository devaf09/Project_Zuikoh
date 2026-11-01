"""200 m の格子ごとの切り出し（Sentinel-1 RTC、VV・VH）を作る（実験06 のデータ、持ち主の案「飛行機の成分・車の成分を格子で」）。

正解: data_labels/<場所>_aircraft_cells.csv・<場所>_car_cells.csv（NAIP の航空写真を Claude が目で見て印をつけた格子）
      2 = 飛行機あり、1 = 車あり、0 = その他（印のある格子の周り ±margin 格子の陸の格子。水は除く）
格子: NAIP の 2 m の画素で 100 × 100（tools/fetch_naip.py の格子）。格子の中心の経度・緯度から、RTC の画素を求め、
      24 × 24 画素（10 m、240 m 四方）を切り出す。RTC は日付ごと（上昇・下降の軌道が混ざる = 見る角度が違う）
保存先: <データの置き場>/processed/cell_chips/cells.pt（X: (N, 2, 24, 24) の dB、y、site、date、cx、cy）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/build_cell_chips.py --sites roswell,victorville,longbeach_cars
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import rasterio
import torch
from rasterio.warp import transform

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402

CELL, C = 100, 24


def load_cells(path):
    """印をつけた格子の (x, y) の集合（無ければ空）。"""
    if not os.path.exists(path):
        return set()
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r and not r[0].startswith("#") and r[0] != "x"]
    return {(int(x), int(y)) for x, y in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default="roswell,victorville,longbeach_cars")
    ap.add_argument("--margin", type=int, default=10, help="「その他」に使う、印のある格子の周りの幅（格子）")
    a = ap.parse_args()
    d = data_dir()
    X, meta = [], []
    for site in a.sites.split(","):
        air = load_cells(os.path.join(ROOT, "data_labels", f"{site}_aircraft_cells.csv"))
        car = load_cells(os.path.join(ROOT, "data_labels", f"{site}_car_cells.csv"))
        lab = air | car
        xs_, ys_ = [x for x, _ in lab], [y for _, y in lab]
        with rasterio.open(d / "processed" / "naip" / f"{site}.tif") as nv:
            ntr, ncrs = nv.transform, nv.crs
            nx, ny = nv.width // CELL, nv.height // CELL
        cells = [(x, y) for x in range(max(0, min(xs_) - a.margin), min(nx, max(xs_) + a.margin + 1))
                 for y in range(max(0, min(ys_) - a.margin), min(ny, max(ys_) + a.margin + 1))]
        cx = [ntr * ((x + 0.5) * CELL, (y + 0.5) * CELL) for x, y in cells]
        lon, lat = transform(ncrs, "EPSG:4326", [p[0] for p in cx], [p[1] for p in cx])
        vvs = sorted((d / "raw" / "s1_rtc" / site).glob("*_vv.tif"))
        for vv in vvs:
            vh = vv.with_name(vv.name.replace("_vv.tif", "_vh.tif"))
            with rasterio.open(vv) as s1, rasterio.open(vh) as s2:
                A = np.stack([s1.read(1), s2.read(1)]).astype("float32")
                xx, yy = transform("EPSG:4326", s1.crs, lon, lat)
                n_add = 0
                for (gx, gy), x, y in zip(cells, xx, yy):
                    col, row = ~s1.transform * (x, y)
                    col, row = int(col), int(row)
                    if not (C // 2 <= row < A.shape[1] - C // 2 and C // 2 <= col < A.shape[2] - C // 2):
                        continue
                    chip = A[:, row - C // 2:row + C // 2, col - C // 2:col + C // 2]
                    if (chip <= 0).mean() > 0.2 or np.nanmedian(chip[0]) < 10 ** (-2.0):   # 画像の外・水（VV が −20 dB 未満）は除く
                        continue
                    label = 2 if (gx, gy) in air else (1 if (gx, gy) in car else 0)
                    X.append(10 * np.log10(np.clip(chip, 1e-5, None)))
                    meta.append((label, site, vv.name[17:25], vv.name[:3], gx, gy))
                    n_add += 1
            print(f"{site} {vv.name[17:25]}: {n_add} 格子")
    y = torch.tensor([m[0] for m in meta])
    out = d / "processed" / "cell_chips"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(X=torch.tensor(np.stack(X)), y=y, site=[m[1] for m in meta], date=[m[2] for m in meta], sat=[m[3] for m in meta],
                    cx=torch.tensor([m[4] for m in meta]), cy=torch.tensor([m[5] for m in meta])), out / "cells.pt")
    for site in a.sites.split(","):
        ys = y[torch.tensor([m[1] == site for m in meta])]
        print(f"{site}: 飛行機 {(ys == 2).sum().item()}・車 {(ys == 1).sum().item()}・その他 {(ys == 0).sum().item()}（格子 × 日付）")


if __name__ == "__main__":
    main()
