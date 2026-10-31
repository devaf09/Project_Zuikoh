#!/bin/bash
# 実験03（GPU 版。持ち主の問い 2「写らない物の周りに、その物の成分がまき散らされていないか」）: 切り出しの真ん中を隠し、周りだけで反射板を当てる。
# 隠す半径 2・4・8 画素（距離の向き。進行方向はその 1/6）× 分解能 そのまま・1/4（10 m 相当）× 学習する条件 2 つ（cnn・sin）× seed 3。
# 分け方は反射板ごと（テストの反射板は訓練で見ない）。学習 3000 steps。手で作った特徴（feat）は CPU で別に回す。
# 共通の待ち行列で回す（.sh の中で GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for lp in 1 4; do
  for mask in 2 4 8; do
    for seed in 0 1 2; do
      for model in cnn sin; do
        n=$(grep -c "\"model\": \"$model\", .*\"mask\": $mask.0, \"lowpass\": $lp, .*\"seed\": $seed," results/exp03.txt 2>/dev/null); n=${n:-0}
        if [ "$n" -gt 0 ]; then continue; fi
        .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split cr --seed "$seed" --steps 3000 --mask "$mask" --lowpass "$lp" --device cuda --out results/exp03.txt
      done
    done
  done
done 2>&1 | tee -a results/sweep03g.log
