#!/bin/bash
# 実験02（CPU 版）: 角の反射板を見分ける。学習の軽い条件 2 つ（scr・feat）。学習する 4 条件は GPU 版 sweep02g.sh。分け方 2 つ（反射板ごと・日付）× seed 3。
# 学習 3000 steps、データ = processed/cr_chips/chips.pt（tools/build_cr_chips.py）。小さいので CPU で（GPU は見せない、4 スレッド、優先度を下げる）。
# 途中から再開できるよう、終わった (条件, 分け方, seed) は results/exp02.txt を数えて飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8
for split in cr date; do
  for seed in 0 1 2; do
    for model in scr feat; do
      n=$(grep -c "\"model\": \"$model\", \"split\": \"$split\", .*\"seed\": $seed," results/exp02.txt 2>/dev/null); n=${n:-0}
      if [ "$n" -gt 0 ]; then continue; fi
      nice -n 10 .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split "$split" --seed "$seed" --steps 3000
    done
  done
done 2>&1 | tee -a results/sweep02.log
