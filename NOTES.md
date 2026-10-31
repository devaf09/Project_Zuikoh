# Project_Zuikoh 実験ノート

合成開口レーダ（SAR）・衛星写真・公開の運行データから、世界の変化と動きを時系列で見えるようにする。学習した Sin の板（Sin レイヤフィルム）で反応を調べ、知らない成分を連番で登録・単離していく。数字の正本。

> ファイル配置: 実験のプログラムは `experiments/expNN_名前.py`、条件は `experiments/sweeps/sweepNN.sh`、結果は `results/expNN.txt`（1 回の実行 = 1 行の JSON）、図は `tools/fig_*.py` から `docs/images/` に。リポジトリ直下で実行する。データは F: の置き場（設定 `ZUIKOH_DATA` か `config.local.toml`）、扱いに気をつけるものは D:\AI_git\Project_Zuikoh_private。決まりは [CLAUDE.md](CLAUDE.md)。

## 現時点のまとめ（2026-10-31 更新）

**いまの最良の構成**: まだない（骨組みと調査の段階）。

| 目標 | 状況 |
|---|---|
| M0: 道具とデータの確認 | データの出どころの下書き（[docs/data_sources.md](docs/data_sources.md)）、先行研究の下書き（[docs/prior_art.md](docs/prior_art.md)）。Sentinel-1 RTC・Umbra・Capella の目録にアカウントなしで届くことを確かめた。ライブラリの確認・見本の取得はこれから |
| M1: 持ち主の案 1（同じ形の物、Sin レイヤフィルムの反応） | 設計の提案（[docs/design_m0_m1.md](docs/design_m0_m1.md)）。推す案は浮き屋根の石油タンク・オクラホマ州クッシング。持ち主の確認待ち |
| M2: 持ち主の案 2（衛星写真と SAR を一緒に、トウモロコシ畑） | 未着手 |
| M3: 持ち主の案 3・4（撮影時刻の写真、知らない成分の登録） | 未着手 |
| M4: 持ち主の案 5・6（登録と単離の繰り返し、船・電車） | 未着手 |
| M5: 持ち主の案 7（地図の表示と受け渡しの形） | 未着手 |

**効いた工夫**: （まだない）
**効かなかった工夫**: （まだない）

**追記 (10/31)**: 調べた範囲では、Sentinel-1（10 m）で浮き屋根の高さや 1 台ずつの車を測った研究は見つからなかった。Sentinel-1 では「物があるか・数・場所全体の明るさの時系列」までが現実的で、細かいものは高分解能のオープンデータ（Umbra・Capella、場所は選べない）で確かめる見込み（docs/prior_art.md）。

**未解決**: M1 の対象と地域（持ち主の確認待ち）。高分解能のデータで場所を選べないこと。
**一貫した教訓**: （まだない）

## 目標
- （2026-10-31 持ち主の依頼）世界の変化と動きを調べるシステムを作る。SAR・衛星写真・公開の運行データから、建物の変化・遺跡の捜索・船や電車の運行を時系列で見えるようにし、人の判断や AI の研究に使える形（地図の上の表示と、機械が読めるデータの形）にする。最初はガスタンク・石油タンク・トウモロコシ畑から、その後に船・電車・車へ
- （2026-10-31 持ち主の案）進め方の骨格: (1) SAR を学習した Sin レイヤフィルムに場面を通し、スクリーンのどこが光るかで物を調べる (2) 衛星写真と SAR を一緒に学ぶ (3) SAR の撮影中に駐車場などの写真を撮り、鶴亀算で車種ごとの反応を確かめる (4) 差から動く物の周波数成分を取り出し、説明できない成分を「知らない成分」として連番で登録 (5) 確信度が上がったら「学習済み」に登録、名前を調べる (6) 1〜5 の繰り返し (7) 世界地図にピン・航跡・建物の変化の色、地図に載っていない地形の拾い出し
- （2026-10-31 持ち主の答え）最初の地域はアメリカ。日本で写真を撮れるのは宮崎県内に限られる

## 環境とコードの出どころ
- PC: Windows 11、GPU は RTX 3060 12GB × 2（共通の待ち行列で使う）
- Python 3.14.7、PyTorch 2.14.1+cu126、numpy 2.5.2、matplotlib 3.11.2、pytest 9.1.1、pyarrow 26.0.0（2026-10-31）。地図・衛星のライブラリ（rasterio など）は M0 で入れる
- vendor: SinWave2D 0.3.0（Project_Metasequoia 811494d）、SN2-Parts-2D 0.11.0（5238c39）。バイト単位で同じ（vendor/README.md）
- 読んだだけで書き換えていないプロジェクト: Sin-CNN2、Project_Metasequoia、SN2-Parts-2D、Sin-CNN3、SinCrest
- **restricted/ を止める仕組み**: .claude/settings.json の permissions.deny に、restricted/ を読む Read・Edit と、"restricted" を含む Bash・PowerShell のコマンドを入れた。効くかの確かめ:
  - 2026-10-31（立ち上げのセッション）: **まだ確かめられていない**。立ち上げのセッションは作業フォルダが D:\AI_git\TmpFiles のままで、この settings.json が読み込まれない。試しのファイル（中身はただの合言葉）を Read・PowerShell で開けてしまい、Bash は場所が違って開けなかった。つまり「このフォルダ以外で始めたセッションには、この仕組みは効かない」。
    途中で作業フォルダを D:\AI_git\Project_Zuikoh に移してから試し直しても、Read・Bash の cat・PowerShell の Get-Content の 4 通りすべてで開けた（対照のファイルも読めた）。設定がセッションの開始時にしか読まれないためか、書き方の誤りかは、まだ分けられていない
  - 確かめ方（D:\AI_git\Project_Zuikoh で始めたセッションで最初にする）: `restricted/deny_test.txt`・`F:/AI_git/Project_Zuikoh_Data_Files/restricted/deny_test.txt` を Read・Bash の cat・PowerShell の Get-Content で開こうとし、すべて断られること。対照として `F:/AI_git/Project_Zuikoh_Data_Files/cache/open_test.txt` は読めること。結果をここに書く
  - 注意: 本物の「送らないもの」は、この確かめが済むまで restricted/ に置かないでほしい（持ち主へ）

## データの出どころ
- 一覧は [docs/data_sources.md](docs/data_sources.md)。取得したものは F: の MANIFEST.md に書く。まだ何も取得していない（2026-10-31 は目録の検索だけ）
