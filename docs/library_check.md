# ライブラリと計算の約束の確認表（M0）

Sin-CNN3 の docs/library_check.md と同じ形。地図・衛星のライブラリを、小さな例と「別の方法」で突き合わせる。

結果の記号: ✓ 期待どおり / ✗ 期待と違う / △ 条件つき / − まだ

## 入れたもの（2026-10-31、リポジトリの .venv/。元の Python の PyTorch が見える形 `--system-site-packages`）

| ライブラリ | 版 | 何に使うか | ライセンス（知: 公式の表記を M0 の終わりに確かめる） |
|---|---|---|---|
| rasterio（GDAL 3.12.2） | 1.5.2 | GeoTIFF・COG の読み書き、座標系の置き直し | BSD |
| pyproj（PROJ 9.8.1） | 3.8.0 | 座標の変換（経度・緯度 ⇔ UTM） | MIT |
| shapely | 2.2.0 | 形の計算（タンクの輪郭など） | BSD |
| pystac-client | 0.9.1 | 衛星データの目録（STAC）の検索 | Apache-2.0 |
| planetary-computer | 1.0.0 | Planetary Computer の読み出しの鍵（SAS） | MIT |
| scipy / scikit-learn | 1.18.1 / 1.9.1 | 比較相手（CFAR の補助、ランダムフォレスト・PCA・NMF） | BSD |
| requests | 2.34.2 | Overpass API（OSM） | Apache-2.0 |
| psutil | 7.2.2 | メモリの測定（実験00） | BSD |
| すでにあったもの | Python 3.14.7、PyTorch 2.14.1+cu126、numpy 2.5.2、matplotlib 3.11.2、pytest 9.1.1 | | |

## 確認

| 関数・計算 | 期待すること | 確かめ方 | 結果 |
|---|---|---|---|
| pyproj の経度・緯度 ⇔ UTM | 往復で元に戻る | 3 地点（クッシング・宮崎・南半球）で往復（tests/test_geo.py） | ✓ 差 1e-9 度未満 |
| UTM の帯の番号 | クッシング = 14N（EPSG 32614）、宮崎 = 52N（32652） | 式と並べる（tests/test_geo.py） | ✓ |
| 範囲の大きさ（bbox_around） | 中心から 5 km ずつ → 東西・南北とも約 10 km | UTM で測る（tests/test_geo.py） | ✓ 1% 以内。角どうしで測ると UTM の格子の傾き（約 1 度）のぶんずれて見える（初めはこれでテストが落ちた） |
| COG の切り出し（rasterio の窓） | 10 m 格子で 10 km 四方 ≈ 1000 × 1000 画素 | 実際に読んだ画素の数（実験00） | ✓ 1021 × 1025（端の丸め） |
| 切り出しの位置 | 画像と OSM の道路・タンクが重なる | 地図に重ねて目で見る（tools/map_sample.py、tools/fig_m0_sample.py） | ✓ 目で見て合っている（数値の確かめは − ） |
| COG を部分で読んだ値 = 全体を読んで切った値 | 同じ | 全体を読んだ版と比べる | − （1 場面 1〜2 GB を読むので、M1 の前に 1 回） |
| STAC の検索が毎回同じ | 同じ日の検索で同じ場面 | 実験00 で 6 回 | ✓ 6 回とも同じ場面 |
| dB の変換 | 10·log10、0 以下は欠け | 式どおり（tools/fig_m0_sample.py） | ✓ |
