"""zuikoh.config — データの置き場などの設定を読む。

データの置き場はコードに直接書かない（開始プロンプト 5 章）。次の順に探す:
    1. 環境変数 ZUIKOH_DATA
    2. リポジトリ直下の config.local.toml の data_dir（git に入れない。見本は config.example.toml）

    from zuikoh.config import data_dir
    raw = data_dir() / "raw"

ライセンス: MIT
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

# リポジトリ直下（このファイルの 1 つ上のフォルダ）
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    """config.local.toml を読んで辞書で返す。無ければ空の辞書。"""
    path = path or ROOT / "config.local.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def data_dir(config: dict | None = None) -> Path:
    """データの置き場のフォルダを返す。環境変数 ZUIKOH_DATA が設定より優先。

    どちらも無ければ、置き場を決めるよう促すエラーにする（黙ってリポジトリの中に書かないため）。
    """
    env = os.environ.get("ZUIKOH_DATA")
    if env:
        return Path(env)
    cfg = load_config() if config is None else config
    if "data_dir" in cfg:
        return Path(cfg["data_dir"])
    raise RuntimeError("データの置き場が未設定です。環境変数 ZUIKOH_DATA か config.local.toml の data_dir を設定してください")
