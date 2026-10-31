"""Sentinel-1 の SLC（複素数の画像 = 明るさと位相）を、ASF から「バースト」単位で取る。

バースト = IW モードの 1 回の撮影の中の小さな区切り（約 20 km × 85 km、1 偏波で数百 MB）。
1 場面の SLC（4〜8 GB の ZIP）を丸ごと取らずに、必要な場所だけを取れる（ASF のバーストの切り出しの仕組み）。
アカウント（NASA Earthdata）が要る。鍵は <データの置き場>/.env の EARTHDATA_TOKEN から読み、表示しない。
ライセンス: Copernicus の Sentinel データの条件（出典「Contains modified Copernicus Sentinel data [年]」）。ASF からの配布。

保存先: <データの置き場>/raw/s1_burst/<地域>/（MANIFEST.md に 1 行足す）

実行（例: クッシングの最新のバーストの一覧だけ見る / 1 つ取る）:
    CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_s1_burst.py --region cushing --lon -96.755 --lat 35.955 --start 2026-10-01 --end 2026-10-31 --list
    CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_s1_burst.py --region cushing --lon -96.755 --lat 35.955 --start 2026-10-20 --end 2026-10-25 --pol VV --max 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import asf_search as asf  # noqa: E402
import dotenv  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402


def session():
    """.env の EARTHDATA_TOKEN で ASF の窓口に入る（鍵の値はどこにも出さない）。"""
    tok = dotenv.dotenv_values(data_dir() / ".env").get("EARTHDATA_TOKEN")
    if not tok:
        raise RuntimeError("EARTHDATA_TOKEN が <データの置き場>/.env にありません")
    return asf.ASFSession().auth_with_token(tok)


def search(lon, lat, start, end, pol, max_results):
    """その点を含むバーストを新しい順に探す（アカウント不要）。"""
    res = asf.search(platform=asf.PLATFORM.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.BURST,
                     intersectsWith=f"POINT({lon} {lat})", start=start, end=end,
                     polarization=pol, maxResults=max_results)
    return sorted(res, key=lambda r: r.properties["startTime"], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--pol", default="VV")
    ap.add_argument("--max", type=int, default=1, help="取るバーストの数（新しい順）")
    ap.add_argument("--list", action="store_true", help="一覧だけ見て取らない")
    a = ap.parse_args()

    res = search(a.lon, a.lat, a.start, a.end, a.pol, 50)
    for r in res[:10]:
        p = r.properties
        print(p["sceneName"], p["startTime"], p.get("flightDirection"), "軌道", p.get("pathNumber"),
              "大きさ", round(float(p.get("bytes") or 0) / 1e6, 1), "MB")
    if a.list:
        return
    out_dir = data_dir() / "raw" / "s1_burst" / a.region
    out_dir.mkdir(parents=True, exist_ok=True)
    ses = session()
    for r in res[:a.max]:
        t0 = time.time()
        r.download(path=str(out_dir), session=ses)
        name = r.properties["fileName"]
        f = out_dir / name
        mb = f.stat().st_size / 1e6
        print(f"取得: {name} {mb:.1f} MB {time.time() - t0:.0f} 秒")
        with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
            fo.write(f"| Sentinel-1 SLC のバースト {a.pol}（{a.region}） | {f.relative_to(data_dir()).as_posix()} | ASF（NASA Earthdata） | "
                     f"{r.properties['sceneName']} | {dt.date.today()} | {mb:.1f} MB | Copernicus の Sentinel データの条件 | "
                     f"出典「Contains modified Copernicus Sentinel data」を書けば再配布可 | tools/fetch_s1_burst.py |\n")


if __name__ == "__main__":
    main()
