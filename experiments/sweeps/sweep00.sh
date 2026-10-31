#!/bin/bash
# 実験00: 取り込みの重さを、切り出しの一辺 1・10・50 km で比べる。Sentinel-1 RTC の同じ 1 場面（2026-10-24、クッシング）。
# 乱数を使わないので seed は 0 だけ。ネットの速さで時間がぶれるので 2 回ずつ。CPU だけ（GPU は見せない）、優先度を下げる。
# 途中から再開できるよう、終わった (大きさ, 回) は results/exp00.txt を数えて飛ばす。
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 GDAL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8
for rep in 0 1; do
  for size in 1 10 50; do
    done_n=$(grep -c "\"size_km\": ${size}\.0," results/exp00.txt 2>/dev/null); done_n=${done_n:-0}
    if [ "$done_n" -gt "$rep" ]; then continue; fi
    nice -n 10 .venv/Scripts/python experiments/exp00_fetch_cost.py --size-km "$size" --seed 0 --out results/exp00.txt
  done
done 2>&1 | tee -a results/sweep00.log
