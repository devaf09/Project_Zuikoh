#!/bin/bash
# 実験05（GPU 版）: 2 本の光路の融合（振幅の光路 + 干渉の位相の光路、明るさを足す）。干渉データ・振幅を平方根に（実験04 の確かめと同じ入力）。
# 比較: 1 本の光路で板を 2 倍（16 枚。学習する数をそろえる）の振幅・複素数。反射板ごとに分ける × seed 3。学習 3000 steps。
# 共通の待ち行列で（GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  for cfg in "fuse 8" "sin_amp 16" "sin 16"; do
    set -- $cfg
    n=$(grep -c "\"model\": \"$1\", .*\"plates\": $2, .*\"seed\": $seed," results/exp05.txt 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then continue; fi
    .venv/Scripts/python experiments/exp02_cr_chips.py --model "$1" --plates "$2" --split cr --seed "$seed" --steps 3000 --data ifg_chips --sqrt-amp --device cuda --out results/exp05.txt
  done
done 2>&1 | tee -a results/sweep05g.log
