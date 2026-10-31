# Project_Zuikoh 実験ノート

合成開口レーダ（SAR）・衛星写真・公開の運行データから、世界の変化と動きを時系列で見えるようにする。学習した Sin の板（Sin レイヤフィルム）で反応を調べ、知らない成分を連番で登録・単離していく。数字の正本。

> ファイル配置: 実験のプログラムは `experiments/expNN_名前.py`、条件は `experiments/sweeps/sweepNN.sh`、結果は `results/expNN.txt`（1 回の実行 = 1 行の JSON）、図は `tools/fig_*.py` から `docs/images/` に。リポジトリ直下で実行する。データは F: の置き場（設定 `ZUIKOH_DATA` か `config.local.toml`）、扱いに気をつけるものは D:\AI_git\Project_Zuikoh_private。決まりは [CLAUDE.md](CLAUDE.md)。

## 現時点のまとめ（2026-10-31 更新）

**いまの最良の構成**: まだない（M0 の途中。クッシングの見本を地図に重ねたところ）。

![クッシングの Sentinel-1 と OSM のタンク](docs/images/m0_cushing_sample.png)

| 目標 | 状況 |
|---|---|
| M0: 道具とデータの確認 | データの出どころの下書き（[docs/data_sources.md](docs/data_sources.md)）、先行研究の下書き（[docs/prior_art.md](docs/prior_art.md)）。ライブラリを .venv に入れて確認表（[docs/library_check.md](docs/library_check.md)）。クッシングの見本（Sentinel-1 RTC 2026-10-24、10 km 四方）と OSM のタンク 333 個を取り、図と地図（Leaflet）に重ねた。取り込みの重さは 50 km 四方でも 20 秒・93 MB（実験00）。残り: COG の部分読みと全体読みの一致 |
| M1: 持ち主の案 1（同じ形の物、Sin レイヤフィルムの反応） | 設計の提案（[docs/design_m0_m1.md](docs/design_m0_m1.md)）。浮き屋根の石油タンク・オクラホマ州クッシングに決定（2026-10-31 持ち主） |
| M2: 持ち主の案 2（衛星写真と SAR を一緒に、トウモロコシ畑） | 未着手 |
| M3: 持ち主の案 3・4（撮影時刻の写真、知らない成分の登録） | 未着手 |
| M4: 持ち主の案 5・6（登録と単離の繰り返し、船・電車） | 未着手 |
| M5: 持ち主の案 7（地図の表示と受け渡しの形） | 未着手 |

**効いた工夫**: （まだない）
**効かなかった工夫**: （まだない）

**追記 (10/31)**: 調べた範囲では、Sentinel-1（10 m）で浮き屋根の高さや 1 台ずつの車を測った研究は見つからなかった。Sentinel-1 では「物があるか・数・場所全体の明るさの時系列」までが現実的で、細かいものは高分解能のオープンデータ（Umbra・Capella、場所は選べない）で確かめる見込み（docs/prior_art.md）。

**追記 (10/31 午前)**: 10 km 四方の VV の画像で、タンクの多くは明るい点として見え、強い点のまわりに十字の尾（装置のぼけ = sinc のサイドローブ）が出ている。持ち主の「反射の特徴が畳み込まれている」の具体例（docs/design_subpixel.md 1 節）。

**未解決**: restricted/ を止める仕組みの確かめ。高分解能のデータで場所を選べないこと。
**一貫した教訓**: （まだない）

## 目標
- （2026-10-31 持ち主の依頼）世界の変化と動きを調べるシステムを作る。SAR・衛星写真・公開の運行データから、建物の変化・遺跡の捜索・船や電車の運行を時系列で見えるようにし、人の判断や AI の研究に使える形（地図の上の表示と、機械が読めるデータの形）にする。最初はガスタンク・石油タンク・トウモロコシ畑から、その後に船・電車・車へ
- （2026-10-31 持ち主の案）進め方の骨格: (1) SAR を学習した Sin レイヤフィルムに場面を通し、スクリーンのどこが光るかで物を調べる (2) 衛星写真と SAR を一緒に学ぶ (3) SAR の撮影中に駐車場などの写真を撮り、鶴亀算で車種ごとの反応を確かめる (4) 差から動く物の周波数成分を取り出し、説明できない成分を「知らない成分」として連番で登録 (5) 確信度が上がったら「学習済み」に登録、名前を調べる (6) 1〜5 の繰り返し (7) 世界地図にピン・航跡・建物の変化の色、地図に載っていない地形の拾い出し
- （2026-10-31 持ち主の答え）最初の地域はアメリカ。日本で写真を撮れるのは宮崎県内に限られる
- （2026-10-31 持ち主の答え）M1 はクッシングの石油タンク。タンクの地帯は攻撃に直結する資料ではないので公開してよい。ライブラリは .venv に
- （2026-10-31 持ち主の案）10 m の格子より細かい情報を、レーダの位相差・2 種類の周波数から取り出せないか。反射波の特徴が拡散して畳み込まれている（ぼけて混ざっている）可能性も調べる

## 環境とコードの出どころ
- PC: Windows 11、GPU は RTX 3060 12GB × 2（共通の待ち行列で使う）
- Python 3.14.7、PyTorch 2.14.1+cu126、numpy 2.5.2、matplotlib 3.11.2、pytest 9.1.1、pyarrow 26.0.0（2026-10-31）。地図・衛星のライブラリはリポジトリの .venv/（--system-site-packages）に入れた: rasterio 1.5.2（GDAL 3.12.2）、pyproj 3.8.0、shapely 2.2.0、pystac-client 0.9.1、planetary-computer 1.0.0、scipy 1.18.1、scikit-learn 1.9.1、requests 2.34.2、psutil 7.2.2（docs/library_check.md）
- 公開リポジトリ: https://github.com/devaf09/Project_Zuikoh （2026-10-31 作成、持ち主の確認の後）
- vendor: SinWave2D 0.3.0（Project_Metasequoia 811494d）、SN2-Parts-2D 0.11.0（5238c39）。バイト単位で同じ（vendor/README.md）
- 読んだだけで書き換えていないプロジェクト: Sin-CNN2、Project_Metasequoia、SN2-Parts-2D、Sin-CNN3、SinCrest
- **restricted/ を止める仕組み**: .claude/settings.json の permissions.deny に、restricted/ を読む Read・Edit と、"restricted" を含む Bash・PowerShell のコマンドを入れた。効くかの確かめ:
  - 2026-10-31（立ち上げのセッション）: **まだ確かめられていない**。立ち上げのセッションは作業フォルダが D:\AI_git\TmpFiles のままで、この settings.json が読み込まれない。試しのファイル（中身はただの合言葉）を Read・PowerShell で開けてしまい、Bash は場所が違って開けなかった。つまり「このフォルダ以外で始めたセッションには、この仕組みは効かない」。
    途中で作業フォルダを D:\AI_git\Project_Zuikoh に移してから試し直しても、Read・Bash の cat・PowerShell の Get-Content の 4 通りすべてで開けた（対照のファイルも読めた）。設定がセッションの開始時にしか読まれないためか、書き方の誤りかは、まだ分けられていない
  - 確かめ方（D:\AI_git\Project_Zuikoh で始めたセッションで最初にする）: `restricted/deny_test.txt`・`F:/AI_git/Project_Zuikoh_Data_Files/restricted/deny_test.txt` を Read・Bash の cat・PowerShell の Get-Content で開こうとし、すべて断られること。対照として `F:/AI_git/Project_Zuikoh_Data_Files/cache/open_test.txt` は読めること。結果をここに書く
  - 注意: 本物の「送らないもの」は、この確かめが済むまで restricted/ に置かないでほしい（持ち主へ）

## データの出どころ
- 一覧は [docs/data_sources.md](docs/data_sources.md)。取得したものは F: の MANIFEST.md に書く。2026-10-31: Sentinel-1 RTC（Planetary Computer、CC BY 4.0）の S1D_IW_GRDH_1SDV_20261024T002756…_rtc を、クッシングの 10 km 四方で VV・VH。OSM（Overpass、ODbL）のタンクの輪郭 333 個（同じ範囲）

## 実験00: 取り込みの重さ (exp00_fetch_cost.py、sweep00.sh)
着想: 開始プロンプト 9 章「大きさごとの時間・メモリ・ディスクを最初に測る」。M1 の見積もり（docs/design_m0_m1.md）の裏づけ。
条件: Sentinel-1 RTC の VV、同じ 1 場面（2026-10-24、クッシング中心）。COG から範囲だけ読んで保存 → 経度・緯度の格子に置き直す。CPU（4 スレッド、優先度を下げて）、2 回ずつ（乱数なし、seed 0）。

| 一辺 | 画素 | 目録の検索 | 読む + 保存 | 置き直し | 保存の大きさ | メモリの最大 |
|---|---|---|---|---|---|---|
| 1 km | 102 × 103 | 2.3 秒 | 3.0 秒 | 0.01 秒 | 0.04 MB | 592 MB |
| 10 km | 1021 × 1025 | 2.3 秒 | 7.0 / 5.8 秒 | 0.1 秒 | 3.7 MB | 628 MB |
| 50 km | 5104 × 5125 | 3.1 / 2.4 秒 | 19.9 / 18.4 秒 | 1.9 秒 | 93 MB | 1,046 MB |

- **結論: 取り込みは軽い。50 km 四方 1 偏波 1 日付でも約 20 秒・93 MB。** M1 の見積もり（3 地域 × 20 km 四方 × 60 日付 × 2 偏波）は約 15 MB × 360 = 約 5 GB、読む時間は約 1 時間（ネット次第）と考えられる
- メモリの約 590 MB は PyTorch などを読み込んだ分の土台で、50 km で +450 MB。画像の大きさ（26 M 画素 × 4 バイト = 105 MB）の約 4 倍で、読む・保存・置き直しの写しがいくつか重なっていると考えられる
- 時間の多くはネットから読む時間（1 km でも 3 秒）。同じ場所の日付を並べて取るときは、まとめて読む工夫の余地がある
- 次の一手: COG の部分読みと全体読みの一致を確かめる（docs/library_check.md の −）。M1 の時系列の取得（tools/fetch_s1_rtc.py の --max-items を増やす）
