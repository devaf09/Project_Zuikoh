"""JAXA の ALOS-2 / PALSAR-2 のオープン&フリーのデータ（災害の事例など）を、URL から直接取る。

持ち主の案（2026-11-02）: PALSAR-2（L バンド、波長 約 24 cm）の細かい撮り方（U2・U3）の複素数（L1.1、CEOS 形式の zip、1 件 約 5 GB）を使う。
アカウントは要らない（このページの zip は直接取れる）。
利用条件: 無償。出所表示が必要（「提供：JAXA」）。災害の事例のデータは、大学等の研究など非商用目的に限る。
ページ: https://www.eorc.jaxa.jp/ALOS/jp/dataset/open_and_free/palsar2_l11_l22_j.htm

保存先: <データの置き場>/raw/palsar2/<名前>/（MANIFEST.md に 1 行足す）。途中で切れても続きから取る（HTTP の Range）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_jaxa_open.py --name kumamoto --urls <URL1>,<URL2>
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--urls", required=True, help="カンマで区切った zip の URL")
    a = ap.parse_args()
    out_dir = data_dir() / "raw" / "palsar2" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in a.urls.split(","):
        out = out_dir / url.rsplit("/", 1)[-1]
        if out.exists():
            print("取得済み:", out.name)
            continue
        part = out.with_suffix(".zip.part")
        done = part.stat().st_size if part.exists() else 0
        t0 = time.time()
        h = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(url, headers=h, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(part, "ab" if (done and r.status_code == 206) else "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
        part.replace(out)
        mb = out.stat().st_size / 1e6
        print(f"取得: {out.name} {mb:.0f} MB {time.time() - t0:.0f} 秒")
        with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
            fo.write(f"| ALOS-2 PALSAR-2 L1.1（{a.name}） | {out.relative_to(data_dir()).as_posix()} | JAXA オープン&フリー（災害の事例） | "
                     f"{out.stem} | {dt.date.today()} | {mb:.0f} MB | 無償・出所表示（提供：JAXA）・非商用（研究） | "
                     f"二次的著作物の配布は出所表示が必要。生データの再配布は不明（しない） | tools/fetch_jaxa_open.py |\n")


if __name__ == "__main__":
    main()
