"""設定の読み込み（zuikoh.config）の確認。"""
from pathlib import Path

import pytest

from zuikoh.config import data_dir, load_config


def test_env_wins(monkeypatch):
    """環境変数 ZUIKOH_DATA が設定ファイルより優先される。"""
    monkeypatch.setenv("ZUIKOH_DATA", "X:/somewhere")
    assert data_dir({"data_dir": "Y:/other"}) == Path("X:/somewhere")


def test_config_value(monkeypatch):
    """環境変数が無ければ設定の data_dir を使う。"""
    monkeypatch.delenv("ZUIKOH_DATA", raising=False)
    assert data_dir({"data_dir": "Y:/other"}) == Path("Y:/other")


def test_missing_raises(monkeypatch):
    """どちらも無ければエラー（黙ってリポジトリの中に書かない）。"""
    monkeypatch.delenv("ZUIKOH_DATA", raising=False)
    with pytest.raises(RuntimeError):
        data_dir({})


def test_load_missing_file(tmp_path):
    """設定ファイルが無ければ空の辞書。"""
    assert load_config(tmp_path / "none.toml") == {}
