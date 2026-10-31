# Project_Zuikoh の約束（Claude Code 向け）

このプロジェクトの決まりの正本。プロジェクトのメモリ機能は使わないので、決まったことはここか NOTES.md に書き残す。
開始プロンプトは D:\AI_git\TmpFiles\prompts\zuikoh_prompt.md（2026-10-31）。くわしい流儀（記録・コード・図・README）はそちらの 4〜7 章に従う。

## 置き場所
- 作業フォルダ（公開リポジトリ devaf09/Project_Zuikoh、MIT）: D:\AI_git\Project_Zuikoh
- データ: F:\AI_git\Project_Zuikoh_Data_Files（raw/・processed/・cache/・photos/・restricted/・MANIFEST.md）。コードは設定（config.local.toml か環境変数 ZUIKOH_DATA）から読み、F: のパスを直接書かない
- 非公開の置き場: D:\AI_git\Project_Zuikoh_private（git、リモートなし）。扱いに気をつけるもの・非公開と決めたものはここ。公開リポジトリには方法の知見だけを書く

## Claude Code に送らないもの（持ち主の方針）
- 置き場 `restricted/`（リポジトリ直下）と `F:\AI_git\Project_Zuikoh_Data_Files\restricted\` の中身は開かない・表示しない（Read、cat・head・grep、画像の表示、どれもしない）
- ファイルの名前と大きさを数えるのも、持ち主が許したときだけ。restricted/ を読むプログラムは持ち主の確認を得てから走らせ、出力は集計の数字だけにする
- 仕組みでも止めている: .claude/settings.json の permissions.deny（確かめた結果は NOTES.md の「環境」）。この仕組みは D:\AI_git\Project_Zuikoh を作業フォルダにして始めたセッションにしか効かない。ほかのフォルダから作業するときは、約束だけが頼り
- 迷ったら送らない側に倒して、持ち主に聞く

## 扱いに気をつける場所と物
- 自分から対象に選ばない: 軍事施設・軍の船や飛行機・防衛関係の場所、発電所・ダムなど重要な施設の警備に関わること、個人の特定・追跡
- 偶然入ったら: 記録は非公開の置き場に止め、公開リポジトリ・図・README・掲示板に出さない。持ち主に「こういうものが写った」とだけ報告する
- 未知かもしれない遺跡・地形の正確な座標は外に出さない（非公開の置き場だけ）。公開する図・地図は位置をぼかす（数 km の格子に丸める、地域名だけ）のを初期値にする
- 迷ったものは Project_Zuikoh_private/notes/sensitivity_log.md に日付・何が・どう判断したかを書き、持ち主に確かめる

## 写真
- 持ち主が撮った写真は F: の photos/。Claude が見てよい（2026-10-31 持ち主）。車種と台数を数える
- 顔・ナンバーが写るので、公開リポジトリ・図・掲示板には入れない。図にはぼかしたものだけ。持ち主が「送らない」と決めたものは restricted/ へ

## データの取り込み
- ライブラリ・元データ・論文へのアクセスは許可不要。20 GB 以上を 1 回で取る、または合計 200 GB を超えそうなときは前もって報告
- 取ったものは F: の MANIFEST.md に必ず書く（取得元・場面の ID・取得日・大きさ・ライセンス・再配布の条件）。取得のプログラムは tools/fetch_*.py
- アカウントが要るデータは Claude が登録しない（持ち主に頼む）。鍵は F: の .env か config.local.toml に置き、会話・記録・git に書かない
- パスワードは Claude が使わない・聞かない。Earthdata（ASF）は、持ち主が作ったトークンを F:\AI_git\Project_Zuikoh_Data_Files\.env の `EARTHDATA_TOKEN=` に持ち主自身が書き、プログラムがそこから読む（値を表示しない）。2026-10-31: 持ち主がチャットにパスワードを書いたので、使わずに、パスワードの変更とトークンの形をお願いした
- Google マップは自動で取らない。確かめたい場所は一覧にして持ち主に渡す

## 言語
- 基本は Python。遅いところだけ Rust（PyO3・maturin）か Go。書く前に測った時間を示して相談
- C・C++ は自分では書かない。ほかの言語は使わない。中身が C のライブラリ（GDAL・numpy・PyTorch）を使うのはよい

## git と公開
- 作者は devaf09 / 46590724+devaf09@users.noreply.github.com（ローカルに設定済み）
- コミットは節目ごと（「実験NN: 結論の一言」「部品 X.Y.Z: 何を足したか」）、push は節目の報告のとき。`git -c http.sslBackend=schannel push`
- push の前に `git diff --stat` で、写真・正確な座標・扱いに気をつけるものが入っていないか確かめる。一度 push したものは消せないと考える

## 全プロジェクト共通の決まり（D:\AI_git\TmpFiles\prompts\onboarding_notice_prompt.md と GPU_QUEUE.md が正）
- 掲示板 D:\AI_git\TmpFiles\BOARD.md: 全部は読まない。作業を始めるとき `grep -n '^### ' D:/AI_git/TmpFiles/BOARD.md | tail -30` で、前回より新しい投稿と関係する投稿だけ読む。書くときは末尾に `>>` で追記（「### YYYY-MM-DD 【種類】題 — Project_Zuikoh」）。前の投稿は書き換えない。識別情報・秘密・データそのもの・9 章の「送らないもの」「扱いに気をつけるもの」は書かない。コミットしない。ほかのプロジェクトへの直してほしいことは【お願い】で
  - 前回読んだ投稿: 2026-10-31「新しいプロジェクト Project_Zuikoh の開始プロンプト」まで
- GPU（RTX 3060 12GB × 2。GPU 0 = 上・x16・130W、GPU 1 = 下・x4・150W）: 共通の待ち行列 D:\AI_git\TmpFiles\gpu\queue.txt に 1 行足して予約する
  `echo "Project_Zuikoh experiments/sweeps/sweepNN.sh any # メモ（見込みの時間と出典）" >> D:/AI_git/TmpFiles/gpu/queue.txt`
  .sh の中で GPU の番号を指定しない。1 行は数時間まで。同じ .sh をもう一度回すときはメモを変えて新しい行で。取り消しはまだ始まっていない自分の行の先頭に #。途中から再開できる .sh にする。自前の待ち行列・タスクスケジューラは作らない。様子は `bash D:/AI_git/TmpFiles/gpu/gpu_status.sh` と nvidia-smi
- CPU だけの python は GPU を見せない。重い前処理は
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 GDAL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= nice -n 10 python ...`
- GPU のプログラムは割り込みを減らす（バッチを大きく、毎ステップの .item()・print・CPU への書き戻しを減らす、とても小さなモデルは CPU で）。F: から毎回大量に読む仕事は前もって小さく切り出す
- 関係するプロジェクト（Sin-CNN2・Project_Metasequoia・SN2-Parts-2D・Sin-CNN3・SinCrest）は読むだけ。D:\AI_git\Sin-CNN は許可まで見ない

## 話し方
- 悪い結果・効かなかった結果にも、必ず考察（なぜ悪いかの見立てと、それを確かめる方法）を添える（2026-10-31 持ち主）
- 日本語。結論を先に、比較相手と並べて、数字を盛らない。持ち主は席を外していることが多いので、方針の範囲内は自律的に進め、節目で短く報告する。持ち主が決めることは「未決の質問」にまとめて仮の前提で進める
