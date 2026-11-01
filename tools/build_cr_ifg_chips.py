"""角の反射板の干渉データ（2 日付の掛け算）の切り出しを作る（実験04 のデータ）。

持ち主の方針（2026-11-01）: 軌道の情報は使わず、データどうしの相関（干渉データ）だけで進める。
干渉データ = A·conj(B)（同じ軌道の 2 日付の SLC を、明るさの相関で画素より細かく重ねてから掛ける）。
動かない点では、斑点のでたらめな位相が打ち消し合って、位相がそろう。畑などの背景は 12 日で反射が変わるので、でたらめのまま。
実験02 で Sin の板が複素数で崩れた理由の候補（斑点のでたらめな位相）を、データの相関だけで取り除けるかを見る。

作り方:
    同じ反射板・同じ撮影の時刻（= 同じ軌道）の日付を並べ、となりどうしの組（12 日おき）ごとに:
      1. 反射板の周り（±150 行・±400 列）を両方の日付で読み、明るさの相関で B を A に重ねる（周波数の側で位相を保ってずらす）
      2. I = A·conj(B) を作り、48 × 48 を切り出す: 2 = 反射板、1 = 日付 A で選んだほかの明るい点、0 = でたらめな位置の背景
      3. いちばん大きい |I| で割ってそろえる
保存先: <データの置き場>/processed/cr_chips/ifg_chips.pt（chips.pt と同じ形の辞書。date は "A-B"）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/build_cr_ifg_chips.py --regions cushing,oklahoma_cr
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import zipfile

import numpy as np
import rasterio
import torch
from rasterio.transform import GCPTransformer
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from est_01d_slc_two_dates import subpixel_shift  # noqa: E402
from est_03_point_relations import fourier_shift  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

C, WR, WC = 48, 150, 400


def read_window(z, cr):
    """ZIP の中から、反射板を含む帯の VV の窓と、帯の名前を返す（無ければ None）。"""
    for n in sorted(x for x in zipfile.ZipFile(z).namelist() if "/measurement/" in x and "-vv-" in x):
        with rasterio.open(f"/vsizip/{z.as_posix()}/{n}") as s:
            row, col = GCPTransformer(s.gcps[0]).rowcol(float(cr["lon"]), float(cr["lat"]))
            if WR + C <= row < s.height - WR - C and WC + C <= col < s.width - WC - C:
                return n.split("/")[-1][:7], s.read(1, window=((row - WR, row + WR + 1), (col - WC, col + WC + 1))).astype("complex64")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default="cushing,oklahoma_cr")
    ap.add_argument("--n-bright", type=int, default=12)
    ap.add_argument("--n-bg", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = data_dir()
    crs = list(csv.DictReader(open(d / "raw" / "cr" / "oklahoma" / "nisar_cr_oklahoma.csv", encoding="utf-8")))
    zips = [z for r in a.regions.split(",") for z in sorted((d / "raw" / "s1_slc" / r).glob("*.zip"))]
    by_pass = collections.defaultdict(list)                   # 撮影の時刻（hhmm）ごと = 同じ軌道・同じ場面の並び
    for z in zips:
        by_pass[z.name[26:30]].append(z)
    X, meta = [], []
    h = C // 2
    for pt, zs in by_pass.items():
        zs = sorted({z.name[17:25]: z for z in zs}.values(), key=lambda z: z.name[17:25])
        for cr in crs:
            wins = [(z.name[17:25], read_window(z, cr)) for z in zs]
            wins = [(dt, w) for dt, w in wins if w is not None]
            for (da, wa), (db_, wb) in zip(wins, wins[1:]):
                sub, A = wa
                _, B = wb
                pa = np.abs(A) ** 2
                bg = np.median(pa)
                band = np.zeros_like(pa)
                band[:, WC - 60:WC + 61] = pa[:, WC - 60:WC + 61]
                kr, kc = np.unravel_index(np.argmax(band), pa.shape)
                if 10 * np.log10(pa[kr, kc] / bg) < 30 or not (C <= kr < pa.shape[0] - C and C <= kc < pa.shape[1] - C):
                    continue
                dr, dc = subpixel_shift(np.log1p(pa), np.log1p(np.abs(B) ** 2))
                Bs = fourier_shift(B, -dr, -dc)               # B を A に重ねる（位相を保つ）
                I = A * np.conj(Bs)
                dbA = 10 * np.log10(pa / bg + 1e-12)

                def add(r, c, label):
                    chip = I[r - h:r + h, c - h:c + h]
                    X.append(chip / (np.abs(chip).max() + 1e-12))
                    meta.append((label, cr["id"], f"{da}-{db_}", sub, float(dbA[r, c]), pt))

                add(kr, kc, 2)
                loc = (pa == ndimage.maximum_filter(pa, size=9)) & (dbA >= 20)
                rr, cc = np.nonzero(loc)
                ok = (np.hypot(rr - kr, cc - kc) > 40) & (rr >= C) & (rr < pa.shape[0] - C) & (cc >= C) & (cc < pa.shape[1] - C)
                for i in np.argsort(-dbA[rr[ok], cc[ok]])[:a.n_bright]:
                    add(rr[ok][i], cc[ok][i], 1)
                for _ in range(a.n_bg):
                    r, c = int(rng.integers(C, pa.shape[0] - C)), int(rng.integers(C, pa.shape[1] - C))
                    if np.hypot(r - kr, c - kc) > 40:
                        add(r, c, 0)
                print(f"{pt} {da}-{db_} {sub} {cr['id']}: 重ねたずれ 行 {dr:+.2f}・列 {dc:+.2f}、反射板の干渉の位相のそろい "
                      f"{abs(I[kr - 1:kr + 2, kc - 1:kc + 2].sum()) / (np.abs(I[kr - 1:kr + 2, kc - 1:kc + 2]).sum() + 1e-12):.2f}")
    y = torch.tensor([m[0] for m in meta])
    out = d / "processed" / "cr_chips" / "ifg_chips.pt"
    torch.save(dict(X=torch.tensor(np.stack(X)), y=y, cr=[m[1] for m in meta], date=[m[2] for m in meta],
                    subswath=[m[3] for m in meta], scr_db=torch.tensor([m[4] for m in meta]), pass_time=[m[5] for m in meta]), out)
    print(f"干渉の切り出し {len(y)} 個（反射板 {(y == 2).sum().item()}・明るい点 {(y == 1).sum().item()}・背景 {(y == 0).sum().item()}）→ {out}")


if __name__ == "__main__":
    main()
