#!/bin/bash
# 実験03（持ち主の問い 2「写らない物の周りに、その物の成分がまき散らされていないか」）: 切り出しの真ん中を隠し、周りだけで反射板を当てる。
# 隠す半径 0・2・4・8 画素（距離の向き。進行方向はその 1/6）× 分解能 そのまま・1/4（10 m 相当）× 条件 3 つ（feat・cnn・sin）× seed 3。
# 分け方は反射板ごと（テストの反射板は訓練で見ない）。学習 3000 steps。CPU で（GPU は見せない、4 スレッド、優先度を下げる）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8
for lp in 1 4; do
  for mask in 2 4 8; do
    for seed in 0 1 2; do
      for model in feat cnn sin; do
        n=$(grep -c "\"model\": \"$model\", .*\"mask\": $mask.0, \"lowpass\": $lp, .*\"seed\": $seed," results/exp03.txt 2>/dev/null); n=${n:-0}
        if [ "$n" -gt 0 ]; then continue; fi
        nice -n 10 .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split cr --seed "$seed" --steps 3000 --mask "$mask" --lowpass "$lp" --out results/exp03.txt
      done
    done
  done
done 2>&1 | tee -a results/sweep03.log
