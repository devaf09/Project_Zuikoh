#!/bin/bash
# 実験06: 200 m の格子の飛行機の成分。条件 3 つ（feat・cnn・sin_amp）× 試験の場所 2 つ（ビクタービル・ロズウェル、もう一方で訓練）× 日付どうしの変化 あり・なし × seed 3。
# 学習 1500 steps。データ = processed/cell_chips/cells.pt。小さいので CPU で（GPU は見せない、4 スレッド、優先度を下げる）。終わった組は飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8
for site in victorville roswell; do
  for tflag in "" "--temporal"; do
    tv=$([ -n "$tflag" ] && echo true || echo false)
    for seed in 0 1 2; do
      for model in feat cnn sin_amp; do
        n=$(grep -c "\"model\": \"$model\", .*\"test_site\": \"$site\", \"temporal\": $tv, .*\"seed\": $seed," results/exp06.txt 2>/dev/null); n=${n:-0}
        if [ "$n" -gt 0 ]; then continue; fi
        nice -n 10 .venv/Scripts/python experiments/exp06_cells.py --model "$model" --test-site "$site" $tflag --seed "$seed" --steps 1500
      done
    done
  done
done 2>&1 | tee -a results/sweep06.log
