#!/bin/bash
# 実験04（GPU 版）: 2 日付の干渉データ（A·conj(B)）の切り出しで反射板を見分ける。斑点のでたらめな位相が打ち消されると Sin の板（複素数）が戻るか。
# 持ち主の方針（2026-11-01）: 軌道の情報は使わず、データどうしの相関だけ。
# 条件 4 つ（cnn_amp・cnn・sin_amp・sin）× 分け方 反射板ごと × seed 3。学習 3000 steps。共通の待ち行列で（GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  for model in sin cnn sin_amp cnn_amp; do
    n=$(grep -c "\"model\": \"$model\", .*\"seed\": $seed," results/exp04.txt 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then continue; fi
    .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split cr --seed "$seed" --steps 3000 --data ifg_chips --device cuda --out results/exp04.txt
  done
done 2>&1 | tee -a results/sweep04g.log
