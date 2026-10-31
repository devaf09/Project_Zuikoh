"""Sentinel-1 の SLC（複素数の画像 = 明るさと位相）を、ASF から 1 場面まるごと（ZIP、4〜8 GB）取る。

バースト単位の取得（tools/fetch_s1_burst.py）は、ASF の仕組みの都合でトークンだけでは取れなかった（ログインの cookie が要る）。
こちらは「Authorization: Bearer <トークン>」で取れる（2026-10-31 に確かめた。最初に一度、持ち主が ASF の利用条件を承認する必要がある）。
鍵は <データの置き場>/.env の EARTHDATA_TOKEN から読み、表示しない。鍵は Earthdata と ASF の窓口にだけ送る（転送先の保管庫には送らない）。
途中で切れても、もう一度動かせば続きから取る（HTTP の Range）。
ライセンス: Copernicus の Sentinel データの条件（出典「Contains modified Copernicus Sentinel data [年]」）。

保存先: <データの置き場>/raw/s1_slc/<地域>/（MANIFEST.md に 1 行足す）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_s1_slc.py --region cushing --lon -96.755 --lat 35.955 --start 2026-10-23 --end 2026-10-25 --max 1
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
import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

AUTH_HOSTS = ("earthdata.nasa.gov", "asf.alaska.edu")   # 鍵を送ってよい窓口


def token() -> str:
    """.env から EARTHDATA_TOKEN を読む（値は返すだけで表示しない）。"""
    t = dotenv.dotenv_values(data_dir() / ".env").get("EARTHDATA_TOKEN")
    if not t:
        raise RuntimeError("EARTHDATA_TOKEN が <データの置き場>/.env にありません")
    return t


def open_stream(url: str, tok: str, start_byte: int = 0) -> requests.Response:
    """転送（リダイレクト）を手でたどり、鍵は認証の窓口にだけ付けて、本体の流れを開く。"""
    for _ in range(8):
        host = url.split("/")[2]
        h = {"Authorization": f"Bearer {tok}"} if host.endswith(AUTH_HOSTS) else {}
        if start_byte:
            h["Range"] = f"bytes={start_byte}-"
        r = requests.get(url, headers=h, stream=True, allow_redirects=False, timeout=120)
        if r.status_code in (301, 302, 303, 307, 308):
            url = r.headers["Location"]
            r.close()
            continue
        if r.status_code >= 400:
            msg = r.text[:300]
            r.close()
            raise RuntimeError(f"取得に失敗 {r.status_code} {host}: {msg}")
        return r
    raise RuntimeError("転送が多すぎる")


def download(url: str, out, tok: str):
    """out へ保存する。途中まであれば続きから。"""
    part = out.with_suffix(out.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    r = open_stream(url, tok, done)
    mode = "ab" if (done and r.status_code == 206) else "wb"
    with open(part, mode) as f:
        for chunk in r.iter_content(chunk_size=8 << 20):
            f.write(chunk)
    r.close()
    part.replace(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--cr-csv", help="点の代わりに、この CSV（lat・lon の列）の全部の点を含む場面を探す（角の反射板の一覧など）")
    ap.add_argument("--path", type=int, help="この軌道の番号（相対軌道）の場面だけ")
    a = ap.parse_args()

    if a.cr_csv:                                              # 一覧の点ごとに探して、場面を重ならないように集める
        import csv
        pts = [(float(r["lon"]), float(r["lat"])) for r in csv.DictReader(open(a.cr_csv, encoding="utf-8"))]
    else:
        pts = [(a.lon, a.lat)]
    found = {}
    for lon, lat in pts:
        for r in asf.search(platform=asf.PLATFORM.SENTINEL1, processingLevel="SLC", intersectsWith=f"POINT({lon} {lat})",
                            start=a.start, end=a.end, maxResults=100):
            if a.path is None or int(r.properties["pathNumber"]) == a.path:
                found[r.properties["sceneName"]] = r
    res = sorted(found.values(), key=lambda r: r.properties["startTime"], reverse=True)[:a.max]
    gb = sum(float(r.properties["bytes"]) for r in res) / 1e9
    print(f"{len(res)} 場面、合計 {gb:.2f} GB")
    if gb > 20:
        raise SystemExit("20 GB を超えるので、持ち主に報告してから取る（CLAUDE.md の決まり）")
    out_dir = data_dir() / "raw" / "s1_slc" / a.region
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = token()
    for r in res:
        p = r.properties
        out = out_dir / p["fileName"]
        if out.exists():
            print("取得済み:", out.name)
            continue
        t0 = time.time()
        download(p["url"], out, tok)
        mb = out.stat().st_size / 1e6
        print(f"取得: {out.name} {mb:.0f} MB {time.time() - t0:.0f} 秒")
        with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
            fo.write(f"| Sentinel-1 SLC（{a.region}、1 場面の ZIP） | {out.relative_to(data_dir()).as_posix()} | ASF（NASA Earthdata） | "
                     f"{p['sceneName']} | {dt.date.today()} | {mb:.0f} MB | Copernicus の Sentinel データの条件 | "
                     f"出典「Contains modified Copernicus Sentinel data」を書けば再配布可 | tools/fetch_s1_slc.py |\n")


if __name__ == "__main__":
    main()
