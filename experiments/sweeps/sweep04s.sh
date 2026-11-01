#!/bin/bash
# 実験04 の確かめ（GPU 版）: 干渉データの振幅を平方根にして（明るさの幅を 1 日付と同じに、位相はそのまま）、Sin の板・CNN を比べる。
# 条件 4 つ（sin・cnn・sin_amp・cnn_amp）× 反射板ごと × seed 3。学習 3000 steps。共通の待ち行列で（GPU の番号は指定しない）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  for model in sin cnn sin_amp cnn_amp; do
    n=$(grep -c "\"model\": \"$model\", .*\"sqrt_amp\": true, .*\"seed\": $seed," results/exp04s.txt 2>/dev/null); n=${n:-0}
    if [ "$n" -gt 0 ]; then continue; fi
    .venv/Scripts/python experiments/exp02_cr_chips.py --model "$model" --split cr --seed "$seed" --steps 3000 --data ifg_chips --sqrt-amp --device cuda --out results/exp04s.txt
  done
done 2>&1 | tee -a results/sweep04s.log
