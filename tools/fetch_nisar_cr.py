"""NISAR のオクラホマの角の反射板の一覧（座標・向き・大きさ）を、JPL の UAVSAR の校正のページから取る。

公開されている表（NISAR の形式の HTML）を読み、反射板ごとに「いちばん新しい測量」の行だけを残して CSV にする。
座標は WGS-84 の緯度・経度と楕円体の高さ（IGS20）。向き（Azimuth）は反射板の口が向いている方位、Tilt は仰ぎの角度。
アカウントは要らない。

保存先: <データの置き場>/raw/cr/oklahoma/nisar_cr_oklahoma.csv（元の HTML も同じ所に。MANIFEST.md に 1 行足す）
実行: CUDA_VISIBLE_DEVICES= .venv/Scripts/python tools/fetch_nisar_cr.py
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from zuikoh.config import data_dir  # noqa: E402

URL = "https://uavsar.jpl.nasa.gov/calibration/NISAR_ANC_OKLA_CORNER_REFLECTORS_006.html"
COLS = ["id", "lat", "lon", "h_ell", "azimuth_deg", "tilt_deg", "side_m", "survey_date", "validity",
        "ve", "vn", "vu", "shape"]


def parse(html: str) -> list[dict]:
    """表のセルを順に読み、13 個ずつで 1 行にする。"""
    cells = [re.sub(r"\s+", " ", c).strip().rstrip(",").strip()
             for c in re.findall(r"<td[^>]*>(.*?)</td>", html, flags=re.S | re.I)]
    rows = []
    for i in range(0, len(cells) - len(COLS) + 1, len(COLS)):
        r = dict(zip(COLS, cells[i:i + len(COLS)]))
        if re.match(r"^N\d", r["id"]):
            rows.append(r)
    return rows


def main():
    out_dir = data_dir() / "raw" / "cr" / "oklahoma"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = requests.get(URL, timeout=60).text
    (out_dir / "NISAR_ANC_OKLA_CORNER_REFLECTORS_006.html").write_text(html, encoding="utf-8")
    rows = parse(html)
    latest = {}
    for r in rows:                                            # 反射板ごとにいちばん新しい測量
        if r["id"] not in latest or r["survey_date"] > latest[r["id"]]["survey_date"]:
            latest[r["id"]] = r
    out = out_dir / "nisar_cr_oklahoma.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for k in sorted(latest):
            w.writerow(latest[k])
    print(f"表の行 {len(rows)}、反射板 {len(latest)} 基 → {out}")
    for k in sorted(latest):
        r = latest[k]
        print(f"  {k}: 緯度 {float(r['lat']):.4f} 経度 {float(r['lon']):.4f} 向き {r['azimuth_deg']} 仰ぎ {r['tilt_deg']} 一辺 {r['side_m']} m 測量 {r['survey_date'][:10]}")
    with open(data_dir() / "MANIFEST.md", "a", encoding="utf-8") as fo:
        fo.write(f"| NISAR のオクラホマの角の反射板の一覧（{len(latest)} 基） | {out.relative_to(data_dir()).as_posix()} | JPL UAVSAR の校正のページ | "
                 f"NISAR_ANC_OKLA_CORNER_REFLECTORS_006 | {dt.date.today()} | {out.stat().st_size / 1e3:.0f} KB | NASA/JPL の公開の資料 | "
                 f"出典を書いて利用（NASA/JPL） | tools/fetch_nisar_cr.py |\n")


if __name__ == "__main__":
    main()
