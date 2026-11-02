"""zuikoh.palsar2 — ALOS-2 PALSAR-2 の L1.1（CEOS 形式の複素数の画像）を読む部品。

L1.1 の画像ファイル（IMG-HH-…）の形:
    先頭 720 バイト = ファイルの説明（記録の長さ・行の数・1 行の画素の数が文字で入っている）
    そのあと 1 行ずつ = 頭 544 バイト（その行の時刻など）+ 画素 × 8 バイト（実部・虚部の 4 バイトの浮動小数、大きい桁が先 = big-endian）
zip のままでは途中から読めない（圧縮されている）ので、最初に 1 回だけ展開する（extract_img）。

    from zuikoh.palsar2 import extract_img, read_window, quicklook
    img = extract_img(zip_path, out_dir)                  # 展開した IMG のパス（展開済みなら何もしない）
    z = read_window(img, row0, row1, col0, col1)          # 複素数 (行, 列)
    ql = quicklook(img, step=32)                          # 間引いた明るさ（全体像を見るため）

ライセンスの注意: データは JAXA（非商用の研究、出所表示「提供：JAXA」）。

ライセンス（この部品）: MIT
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np

PREFIX = 544


def extract_img(zip_path, out_dir) -> Path:
    """zip の中の IMG-HH を out_dir に展開して、そのパスを返す（展開済みなら何もしない）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if Path(n).name.startswith("IMG-")][0]   # フォルダに入っている zip も、入っていない zip もある
        out = out_dir / Path(name).name
        if not out.exists():
            with z.open(name) as src, open(str(out) + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst, 64 << 20)
            Path(str(out) + ".part").replace(out)
    return out


def header(img) -> dict:
    """ファイルの説明から、記録の長さ・行の数・画素の数を読む。"""
    with open(img, "rb") as f:
        h = f.read(720)
    return dict(record=int(h[186:192]), lines=int(h[180:186]), pixels=int(h[248:256]))


def read_window(img, r0, r1, c0, c1) -> np.ndarray:
    """行 r0〜r1、列 c0〜c1 の複素数の画像を読む（big-endian の 4 バイト浮動小数の実部・虚部）。"""
    h = header(img)
    out = np.empty((r1 - r0, c1 - c0), dtype=np.complex64)
    with open(img, "rb") as f:
        for i, r in enumerate(range(r0, r1)):
            f.seek(720 + r * h["record"] + PREFIX + c0 * 8)
            v = np.frombuffer(f.read((c1 - c0) * 8), dtype=">f4")
            out[i] = v[0::2] + 1j * v[1::2]
    return out


def quicklook(img, step=32) -> np.ndarray:
    """step 行ごとに 1 行を読み、行の中も step 画素ごとに平均した明るさ（|z|²）の全体像。"""
    h = header(img)
    rows = range(0, h["lines"], step)
    out = np.zeros((len(rows), h["pixels"] // step), dtype=np.float32)
    with open(img, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(720 + r * h["record"] + PREFIX)
            v = np.frombuffer(f.read(h["pixels"] * 8), dtype=">f4").astype(np.float32)
            p = v[0::2] ** 2 + v[1::2] ** 2
            out[i] = p[:out.shape[1] * step].reshape(-1, step).mean(1)
    return out
