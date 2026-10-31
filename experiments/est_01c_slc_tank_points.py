"""est_01c: SLC の実物で、タンクの強い点の明るさと形を見る — est_01 の仮定（点 1 つ・SCR 15 dB 以上）が実物で成り立つか

着想: est_01（作り物）で「角の点が背景より 15 dB 明るければ屋根の高さを約 0.25 m で測れる」と見積もった。est_01b は加工済みの 10 m の画像での目安だった。
      ここでは 1 回分の撮影の複素数の画像（SLC）で、同じことを確かめる。
方法:
    1. ZIP の中の SLC（IW のどれかの帯、VV）を GDAL の /vsizip/ で読む（展開しない）。画像の位置は付属の目印の点（GCP）で経度・緯度と結ぶ
       GCP だけだとバーストの重なりのため数十〜百行ずれる（2026-10-31 に確かめた: 行 −76・列 +20）。
       OSM のタンクの点の並びと、画像の明るい点の並びの相関（ずらしながらの重なり）で、ずれを測って直す（register）
    2. OSM のタンクごとに、中心の周りの SLC の明るさ（|振幅|²）の最大を取り、周りの畑の中央値との比（SCR）を数える
    3. タンクごとに、距離の向き（横）の明るさの形を見る: 山がいくつあるか（背景より 10 dB 以上の山の数）
指標: SCR の分布（中央値・15 dB 以上の割合）、山の数の分布
結果: results/est01c.txt（1 行 JSON）。実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python experiments/est_01c_slc_tank_points.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer
from scipy import ndimage, signal
from scipy.signal import find_peaks

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from zuikoh.config import data_dir  # noqa: E402
from zuikoh.geo import to_utm  # noqa: E402


def find_measurement(zip_path, pol):
    """ZIP の中の、その偏波の measurement の TIFF を、タンクの地帯を含む帯（IW1〜3）から探す。"""
    import zipfile
    names = [n for n in zipfile.ZipFile(zip_path).namelist() if "/measurement/" in n and f"-{pol}-" in n and n.endswith(".tiff")]
    return [f"/vsizip/{zip_path.as_posix()}/{n}" for n in sorted(names)]


def register(pw, rows, cols, look=4):
    """タンクの点の並び（rows, cols）と画像の明るい点の並びの相関が最大になるずれ（行, 列）を返す。

    距離の向きは look 画素を平均して、ほぼ正方形の画素にしてから比べる（進行方向 14 m・距離の地面 約 3.7 m）。
    戻り値の強さ = 相関の山 / 相関のばらつき（10 を下回るなら当てにしない）。
    """
    m = ndimage.uniform_filter(pw, (1, look))[:, ::look]
    db = 10 * np.log10(m + 1e-12)
    bright = (db > np.percentile(db, 97)).astype("float64")
    mk = np.zeros_like(bright)
    for r, c in zip(rows, np.asarray(cols) / look):
        r, c = int(round(r)), int(round(c))
        if 0 <= r < mk.shape[0] and 0 <= c < mk.shape[1]:
            mk[r, c] = 1
    mk = ndimage.binary_dilation(mk, iterations=2).astype("float64")
    cc = signal.fftconvolve(bright - bright.mean(), (mk - mk.mean())[::-1, ::-1], mode="same")
    k = np.unravel_index(np.argmax(cc), cc.shape)
    return k[0] - cc.shape[0] // 2, (k[1] - cc.shape[1] // 2) * look, float(cc.max() / cc.std())


def main():
    ap = argparse.ArgumentParser(description="est_01c: SLC の実物でタンクの強い点を見る")
    ap.add_argument("--region", default="cushing")
    ap.add_argument("--pol", default="vv")
    ap.add_argument("--seed", type=int, default=0, help="乱数は使わない（決まりとして持つ）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "est01c.txt"))
    a = ap.parse_args()

    d = data_dir()
    zip_path = sorted((d / "raw" / "s1_slc" / a.region).glob("*.zip"))[-1]
    tanks = json.load(open(d / "raw" / "osm" / a.region / "storage_tanks.geojson", encoding="utf-8"))
    cen = []
    for f in tanks["features"]:                              # タンクの中心と直径（m）
        lon, lat = zip(*f["geometry"]["coordinates"][0])
        x, _, ep = to_utm(min(lon), np.mean(lat))
        x2, _, _ = to_utm(max(lon), np.mean(lat), ep)
        cen.append((np.mean(lon), np.mean(lat), x2 - x))
    cen = np.array(cen)

    for href in find_measurement(zip_path, a.pol):
        with rasterio.open(href) as src:
            gcps, _ = src.gcps
            tr = GCPTransformer(gcps)
            rows, cols = tr.rowcol(cen[:, 0], cen[:, 1])     # 経度・緯度 → 画像の行（進行方向）・列（距離の向き）
            rows, cols = np.array(rows), np.array(cols)
            ok = (rows > 50) & (rows < src.height - 50) & (cols > 50) & (cols < src.width - 50)
            if ok.mean() < 0.5:                             # この帯にタンクの地帯が入っていない
                continue
            r0, r1 = rows[ok].min() - 200, rows[ok].max() + 200
            c0, c1 = cols[ok].min() - 200, cols[ok].max() + 200
            win = ((r0, r1), (c0, c1))
            z = src.read(1, window=win).astype("complex64")
            sub = os.path.basename(href)
            break
    else:
        raise SystemExit("タンクの地帯を含む帯が見つからない")

    pw = np.abs(z) ** 2                                       # 1 回分の明るさ（斑点そのまま）
    dr, dc, strength = register(pw, rows[ok] - r0, cols[ok] - c0)
    rows, cols = rows + dr, cols + dc                        # GCP のずれを直す
    print(f"位置合わせ: 行 {dr:+d}・列 {dc:+d}（相関の山の強さ {strength:.1f}）")
    bg = float(np.median(pw[pw > 0]))
    scr, npk = [], []
    for (lon, lat, diam), r, c, k in zip(cen, rows, cols, ok):
        if not k or not (r0 <= r < r1 and c0 <= c < c1):
            continue
        rr, cc = r - r0, c - c0
        hr, hc = int(diam / 14 / 2) + 3, int(diam / 2.33 / 2 / 0.63) + 6   # 進行方向 14 m・距離 約 2.3 m（地面では /sinθ）の画素で半径 + 余裕
        w = pw[max(0, rr - hr):rr + hr + 1, max(0, cc - hc):cc + hc + 1]
        if w.size == 0:
            continue
        scr.append(10 * np.log10(w.max() / bg))
        prof = w.max(axis=0)                                   # 距離の向きの形（進行方向は最大で潰す）
        pk, _ = find_peaks(10 * np.log10(prof / bg + 1e-12), height=10, distance=2)
        npk.append(len(pk))
    scr, npk = np.array(scr), np.array(npk)
    rec = dict(exp="est01c", scene=zip_path.stem, subswath_file=sub, pol=a.pol, n_tanks=int(len(scr)),
               bg_db=round(10 * np.log10(bg), 1), scr_db_median=round(float(np.median(scr)), 1),
               scr_db_q25=round(float(np.percentile(scr, 25)), 1), scr_db_q75=round(float(np.percentile(scr, 75)), 1),
               frac_scr_ge_15db=round(float((scr >= 15).mean()), 3),
               peaks_median=float(np.median(npk)), peaks_hist={str(k): int((npk == k).sum()) for k in range(0, 8)},
               window=[int(r0), int(r1), int(c0), int(c1)],
               shift_rows=int(dr), shift_cols=int(dc), shift_strength=round(strength, 1))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))
    np.save(d / "cache" / f"est01c_{a.region}_{a.pol}.npy", z)   # 図と次の実験のための切り出し（cache、消してよい）


if __name__ == "__main__":
    main()
