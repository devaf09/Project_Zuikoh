"""角の反射板の学習用の切り出しを作る（実験02 のデータ）。

SLC（複素数の画像）の中の、オクラホマの NISAR の反射板の周りを読み、次の 3 種類の切り出し（48 × 48 画素、複素数）を作る。
    2 = 反射板（いちばん明るい点、背景より 30 dB 以上を反射板とする）
    1 = ほかの明るい点（同じ範囲の、反射板から 40 画素以上離れた、背景より 20 dB 以上の山。建物・電柱・機械など）
    0 = 背景（同じ範囲のでたらめな位置）
学習では 48 × 48 から 32 × 32 をずらして切り出す（位置のずれに強くするため）。
切り出しは「いちばん明るい画素の振幅」で割ってそろえる（明るさの強さだけで見分けられないように。強さは別の列 scr_db に残す）。

保存先: <データの置き場>/processed/cr_chips/chips.pt（torch の辞書: X 複素数 (N, 48, 48)、y、cr、date、subswath、scr_db、sat、pass_time = 撮影の時刻 hhmm（同じ時刻 = 同じ軌道 = 同じ見る角度））
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/build_cr_chips.py --regions cushing,oklahoma_cr
"""
from __future__ import annotations

import argparse
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
from zuikoh.config import data_dir  # noqa: E402

C = 48           # 切り出しの一辺（画素）
WR, WC = 150, 400  # 反射板の周りに読む範囲（行・列の半分）


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default="cushing,oklahoma_cr")
    ap.add_argument("--n-bright", type=int, default=12, help="範囲ごとの、ほかの明るい点の数の上限")
    ap.add_argument("--n-bg", type=int, default=12, help="範囲ごとの背景の数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = data_dir()
    crs = list(csv.DictReader(open(d / "raw" / "cr" / "oklahoma" / "nisar_cr_oklahoma.csv", encoding="utf-8")))
    X, meta = [], []
    for region in a.regions.split(","):
        for z in sorted((d / "raw" / "s1_slc" / region).glob("*.zip")):
            date, track = z.name[17:25], None
            names = sorted(n for n in zipfile.ZipFile(z).namelist() if "/measurement/" in n and "-vv-" in n)
            for n in names:
                with rasterio.open(f"/vsizip/{z.as_posix()}/{n}") as s:
                    tr = GCPTransformer(s.gcps[0])
                    for cr in crs:
                        row, col = tr.rowcol(float(cr["lon"]), float(cr["lat"]))
                        if not (WR + C <= row < s.height - WR - C and WC + C <= col < s.width - WC - C):
                            continue
                        A = s.read(1, window=((row - WR, row + WR + 1), (col - WC, col + WC + 1))).astype("complex64")
                        pw = np.abs(A) ** 2
                        bg = np.median(pw)
                        db = 10 * np.log10(pw / bg + 1e-12)
                        # 反射板を探すのは、距離の向き（列）に予測から ±60 画素（約 140 m）の帯の中だけ。
                        # GCP の予測は、距離の向きには数十画素しかずれないが、進行方向には百行ほどずれる（est_02）。
                        # 帯を決めないと、遠くの別の明るい物を反射板と取り違える（est_02 の N10K）
                        band = np.zeros_like(pw)
                        band[:, WC - 60:WC + 61] = pw[:, WC - 60:WC + 61]
                        kr, kc = np.unravel_index(np.argmax(band), pw.shape)
                        if db[kr, kc] < 30 or not (C <= kr < pw.shape[0] - C and C <= kc < pw.shape[1] - C):
                            continue                         # 反射板が見えない（向き・帯の端など）
                        sub = n.split("/")[-1][:7]
                        h = C // 2

                        def add(r, c, label):
                            chip = A[r - h:r + h, c - h:c + h]
                            m = np.abs(chip).max()
                            X.append(chip / (m + 1e-12))
                            meta.append((label, cr["id"], date, sub, float(db[r, c]), z.name[:3], z.name[26:30]))

                        add(kr, kc, 2)
                        loc = (pw == ndimage.maximum_filter(pw, size=9)) & (db >= 20)
                        rr, cc = np.nonzero(loc)
                        far = (np.hypot(rr - kr, cc - kc) > 40) & (rr >= C) & (rr < pw.shape[0] - C) & (cc >= C) & (cc < pw.shape[1] - C)
                        idx = np.argsort(-db[rr[far], cc[far]])[:a.n_bright]   # 明るい順
                        for i in idx:
                            add(rr[far][i], cc[far][i], 1)
                        for _ in range(a.n_bg):
                            r = int(rng.integers(C, pw.shape[0] - C))
                            c = int(rng.integers(C, pw.shape[1] - C))
                            if np.hypot(r - kr, c - kc) > 40:
                                add(r, c, 0)
                        print(f"{date} {sub} {cr['id']}: 反射板 {db[kr, kc]:.1f} dB（GCP とのずれ 行 {kr - WR:+d}・列 {kc - WC:+d}）、明るい点 {len(idx)}")
    y = torch.tensor([m[0] for m in meta])
    out = d / "processed" / "cr_chips"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(X=torch.tensor(np.stack(X)), y=y, cr=[m[1] for m in meta], date=[m[2] for m in meta],
                    subswath=[m[3] for m in meta], scr_db=torch.tensor([m[4] for m in meta]), sat=[m[5] for m in meta], pass_time=[m[6] for m in meta]),
               out / "chips.pt")
    print(f"切り出し {len(y)} 個（反射板 {(y == 2).sum().item()}・明るい点 {(y == 1).sum().item()}・背景 {(y == 0).sum().item()}）→ {out / 'chips.pt'}")


if __name__ == "__main__":
    main()
