#!/bin/bash
# 実験02（GPU 版）: 角の反射板を見分ける。学習する条件 4 つ（cnn_amp・cnn・sin_amp・sin）× 分け方 2 つ（反射板ごと・日付）× seed 3。学習 3000 steps。
# 学習しない条件（scr・feat）は CPU で別に回す（sweep02.sh）。共通の待ち行列で回す（.sh の中で GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for split in cr date; do
  for seed in 0 1 2; do
    for model in cnn_amp cnn sin_amp sin; do
      n=$(grep -c "\"model\": \"$model\", \"split\": \"$split\", .*\"seed\": $seed," results/exp02.txt 2>/dev/null); n=${n:-0}
      if [ "$n" -gt 0 ]; then continue; fi
      .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split "$split" --seed "$seed" --steps 3000 --device cuda
    done
  done
done 2>&1 | tee -a results/sweep02g.log
