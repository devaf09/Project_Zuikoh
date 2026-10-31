# データの出どころの一覧（下書き、2026-10-31）

M0 の「データの出どころの一覧」の下書き。持ち主の確認を待っている。
確かめ方の記号: **実**＝2026-10-31 に実際に目録（STAC など）へアクセスして確かめた / **検**＝ウェブ検索の要約で確かめた（本文の規約は M0 で読む） / **未**＝まだ。
大きさは「1 場面（1 回の撮影）あたり」の目安。切り出して読める形（COG = 必要な部分だけ読める GeoTIFF）なら、実際に取るのはずっと小さい。

## 1. SAR（合成開口レーダ。電波を当てて、戻ってきた電波で地面を写す。雲・夜でも写る）

| 名前 | 取得元 | 分解能 | 頻度 | 期間 | ライセンス | アカウント | 大きさ | 確かめ |
|---|---|---|---|---|---|---|---|---|
| Sentinel-1 RTC（地形の影響をならした後方散乱。VV・VH） | Microsoft Planetary Computer の STAC（`sentinel-1-rtc`） | 10 m の格子 | 同じ軌道から 12 日ごと（場所により 6 日）。2026-08〜10 のアメリカ中部の 1 地点で 3 か月に 8 場面（Sentinel-1D、上昇軌道 34） | 2014-10〜現在 | **CC BY 4.0**（目録の license 欄） | **不要**（読み出しの鍵 SAS は誰でも取れる。2026-10-31 に取得できた） | 1 場面 1〜2 GB（偏波ごと、COG なので切り出せる） | 実 |
| Sentinel-1 GRD（地上の距離にならした明るさ） | 同（`sentinel-1-grd`） | 約 10 m（IW） | 同上 | 2014-10〜現在 | Copernicus の Sentinel データの法的通知（無料・自由に使える、出典表記「Contains modified Copernicus Sentinel data [年]」） | 不要 | 同上 | 実（規約の本文は未読） |
| Sentinel-1 GRD（別の取得元） | AWS の Earth Search（Element84） | 同上 | 同上 | 同上 | 同上 | **要る**（取り出し側が料金を払うバケット。AWS の鍵が要る） | 同上 | 検 |
| Sentinel-1 SLC（複素数の画像 = 振幅と位相。光の場としてそのまま板に通せる候補） | ASF（アラスカ衛星施設）、Copernicus Data Space | 約 2.3 × 14 m | 同上 | 2014〜 | 同上 | **要る**（NASA Earthdata か Copernicus のアカウント。持ち主に登録を頼む） | 1 場面 4〜8 GB（切り出せない ZIP） | 検 |
| Umbra のオープンデータ（高分解能のスポットライト。GEC・SICD（複素数）・SIDD・CPHD（生に近いデータ）） | AWS `umbra-open-data-catalog`（us-west-2） | 0.25〜1 m | 不定期（場所と時刻は Umbra が選ぶ） | 2023〜 | CC BY 4.0（「一部を変えた CC BY 4.0」と書く資料もある。M0 で本文を読む）。出典表記の例「© 年 Umbra Lab, Inc.」 | 不要（匿名で一覧が取れた） | 全体で約 17 TB、1 場面 数百 MB〜数 GB | 実（一覧）・検（規約） |
| Capella のオープンデータ（高分解能。SLC・GEO など） | AWS `capella-open-data`、STAC の目録あり | 0.5〜1 m | 四半期ごとに追加（場所は選べない） | 2020〜 | CC BY 4.0（出典表記「© Capella Space」） | 不要（目録 catalog.json を匿名で取れた） | 1 場面 数百 MB〜数 GB | 実（目録）・検（規約） |

注意:
- Sentinel-1A は 2026-06-30 に運用を終えた。いまは 1C と 1D（1D は 1C の 1 日後に同じ場所を撮る）。過去の時系列では 1A・1B（2021 年末に故障）・1C・1D が混ざるので、衛星の違いを記録する（検）。
- 高分解能（Umbra・Capella）は、撮影の場所と時刻を選べない。持ち主の案 3（撮影の時刻に合わせた写真）は Sentinel-1 の撮影計画で立てる。

## 2. 光学の衛星写真・航空写真

| 名前 | 取得元 | 分解能 | 頻度 | 期間 | ライセンス | アカウント | 確かめ |
|---|---|---|---|---|---|---|---|
| Sentinel-2 L2A（地表の反射率、13 バンド） | Planetary Computer（`sentinel-2-l2a`）、Earth Search | 10〜60 m | 約 5 日 | 2015-06〜 | Copernicus の Sentinel データの条件 | 不要 | 実 |
| Landsat 8・9（30 m、熱の帯域あり） | Planetary Computer、Earth Search | 30 m | 8〜16 日 | 1980 年代〜（機体ごと） | アメリカ政府の著作物（制限なし、USGS の出典表記） | 不要 | 未 |
| NAIP（アメリカの航空写真、赤・緑・青・近赤外） | Planetary Computer（`naip`） | 0.6〜1 m | 州ごとに 2〜3 年 | 2010〜2023（目録の期間） | アメリカ政府の著作物（目録の license 欄は "other"。M0 で確かめる） | 不要 | 実（目録） |

NAIP は「物の形と位置の正解」を作るのに使える（タンクの数と大きさを航空写真で数える）。

## 3. 正解のラベル（地図・作物・運行・経済の数字）

| 名前 | 取得元 | 中身 | ライセンス | アカウント | 確かめ |
|---|---|---|---|---|---|
| Cropland Data Layer（CDL） | USDA NASS の CroplandCROS、Planetary Computer（`usda-cdl`、2008〜2021 のみ） | アメリカ全土の作物の地図（30 m、毎年）。トウモロコシ・大豆など | パブリックドメイン（NASS のメタデータ）、data.gov では CC0。出典の表記を頼まれている | 不要 | 検（PC の目録は実） |
| AIS（船の位置の記録） | NOAA・BOEM の Marine Cadastre（日ごとの ZIP、2024・2025 は GeoParquet もある） | アメリカの沿岸の船の位置・速さ・種類（1 分ごと程度） | アメリカ政府のデータ（「現状のまま」「航海に使わない」。パブリックドメインの明記は見つからず、M0 で読む） | 不要 | 検 |
| OpenStreetMap | Geofabrik の州ごとの抜き出し、Overpass API | タンク（`man_made=storage_tank`）・港・線路・駐車場の形 | **ODbL**（出典表記と、作った地図データの共有の条件あり。公開する派生物の扱いに注意） | 不要 | 未 |
| EIA の原油在庫（週ごと） | アメリカ エネルギー情報局 | クッシング（オクラホマ州）などの原油在庫の週の数字。タンクの満ち具合の「経済の指標」との突き合わせに | アメリカ政府の著作物 | 不要（API は鍵が要る、表の CSV は不要） | 未 |
| xView3-SAR（Sentinel-1 の船の検出のラベル、約 1,000 場面） | iuu.xview.us | 船・海の構造物の位置、AIS と突き合わせ済み | 未確認（登録が要る可能性。M4 の前に確かめる） | 要るかもしれない | 検 |

## 4. 撮影の予定（持ち主の案 3 の計画に使う）

| 名前 | 取得元 | 中身 | 確かめ |
|---|---|---|---|
| Sentinel-1 の撮影計画（KML、約 20 日分ずつ） | ESA の Sentinel-1 Acquisition Plans のページ | 撮る場所と時刻の帯。いまの 1C・1D の分があるかは M0 で確かめる | 検 |
| 過去の撮影の実績 | 上の STAC の目録 | 同じ軌道は同じ時刻に通る（例: アメリカ中部の上昇軌道 34 は 現地時間 19:28 ごろ、UTC 00:28） | 実 |

## 5. 使わないもの・後回し

- Google マップ・Google Earth の画像: 自動で取らない（利用規約）。確かめたい場所は一覧にして持ち主に渡す
- 商用の高分解能（有償）: 使わない
- 軍事・防衛・重要施設の警備に関わる場所のデータは、自分から選ばない（9 章）

## 6. 出典（2026-10-31 に参照）

- Planetary Computer の目録: https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc （license 欄）、SAS の説明 https://planetarycomputer.microsoft.com/docs/concepts/sas
- Earth Search: https://github.com/Element84/earth-search
- Umbra: https://registry.opendata.aws/umbra-open-data/
- Capella: https://registry.opendata.aws/capella_opendata/ 、https://support.capellaspace.com/what-is-the-licensing-for-capellas-open-data
- CDL: https://catalog.data.gov/dataset/cropland-data-layer 、https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_Cropland-Data-Layer-2024.htm
- AIS: https://marinecadastre.gov/ais 、https://github.com/ocm-marinecadastre/ais-vessel-traffic
- Sentinel-1 の運用: https://dataspace.copernicus.eu/node/4026 （1A の終了）、撮影計画 https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans
- xView3-SAR: https://arxiv.org/abs/2206.00897
