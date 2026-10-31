#!/bin/bash
# 実験02 の考察の確かめ（GPU 版）: 切り出しごとに位相の傾きを取り除く（--recenter）と、複素数を入れた Sin の板・CNN が良くなるか。
# 条件 2 つ（sin・cnn）× 分け方 反射板ごと × seed 3。学習 3000 steps。共通の待ち行列で（.sh の中で GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  for model in sin cnn; do
    n=$(grep -c "\"model\": \"$model\", .*\"recenter\": true, .*\"seed\": $seed," results/exp02r.txt 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then continue; fi
    .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split cr --seed "$seed" --steps 3000 --recenter --device cuda --out results/exp02r.txt
  done
done 2>&1 | tee -a results/sweep02r.log
