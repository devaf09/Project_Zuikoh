"""Umbra のオープンデータ（高分解能の SAR、CC BY 4.0、AWS で匿名）から、タスクごとに GEC か SICD を取る。

GEC = 地図の座標に置いた明るさの画像（GeoTIFF、0.25〜1 m 格子、1 件 0.2〜1 GB）。
SICD = 複素数の画像（NITF、1 件 6〜17 GB）。Umbra は撮るたびに見る角度が変わるので、日付どうしの干渉は基本的にできない（持ち主の方針: 軌道の補正はしない）。
まずは GEC で、飛行場の場所ごとの見え方を学ぶ（docs/airports.md）。

保存先: <データの置き場>/raw/umbra/<名前>/（MANIFEST.md に 1 行足す）
実行（例）: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_umbra.py --task AIR/891c2a27-2f67-4059-ae39-7bddc0e15b5e --name atl --kind GEC
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
import urllib.parse

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

BUCKET = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com"


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """S3 の一覧（匿名）から、prefix の下のファイルの名前と大きさを全部取る。"""
    out, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        x = requests.get(url, timeout=120).text
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x, flags=re.S)]
        m = re.search(r"<NextContinuationToken>([^<]*)", x)
        if not m:
            return out
        token = m.group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, help="sar-data/tasks/ の下の名前（例 AIR/<uuid>、または「Sydney International Airport, Australia」）")
    ap.add_argument("--name", required=True, help="保存先のフォルダの名前（例 atl）")
    ap.add_argument("--kind", default="GEC", choices=["GEC", "SICD"])
    ap.add_argument("--max", type=int, default=100, help="取る件数の上限（日付の古い順に、均等に間引く）")
    ap.add_argument("--max-gb", type=float, default=20.0)
    a = ap.parse_args()

    suffix = "_GEC.tif" if a.kind == "GEC" else "_SICD.nitf"
    keys = sorted((k, s) for k, s in list_keys(f"sar-data/tasks/{a.task}/") if k.endswith(suffix))
    if len(keys) > a.max:                                    # 期間の全体にまんべんなく
        idx = [round(i * (len(keys) - 1) / (a.max - 1)) for i in range(a.max)] if a.max > 1 else [0]
        keys = [keys[i] for i in sorted(set(idx))]
    gb = sum(s for _, s in keys) / 1e9
    print(f"{a.task}: {a.kind} {len(keys)} 件、合計 {gb:.2f} GB")
    if gb > a.max_gb:
        raise SystemExit(f"{a.max_gb} GB を超えるので、持ち主に報告してから取る（CLAUDE.md の決まり）")
    out_dir = data_dir() / "raw" / "umbra" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, s in keys:
        out = out_dir / k.split("/")[-1]
        if out.exists() and out.stat().st_size == s:
            continue
        t0 = time.time()
        with requests.get(f"{BUCKET}/{urllib.parse.quote(k)}", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(str(out) + ".part", "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
        os.replace(str(out) + ".part", out)
        print(f"取得: {out.name} {s / 1e6:.0f} MB {time.time() - t0:.0f} 秒")
        with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
            fo.write(f"| Umbra {a.kind}（{a.name}） | {out.relative_to(data_dir()).as_posix()} | AWS umbra-open-data-catalog | {k} | {dt.date.today()} | "
                     f"{s / 1e6:.0f} MB | CC BY 4.0 | 出典「© Umbra Lab, Inc.」を書けば再配布可 | tools/fetch_umbra.py |\n")


if __name__ == "__main__":
    main()
