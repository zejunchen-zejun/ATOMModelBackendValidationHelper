#!/bin/bash

MODEL=/shared/data/amd_int/models/Qwen3-0.6B
RANGE_RATIO=0.8

# 1K / 1K
ISL=1000
OSL=1000
CON=4
NUM=4

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON} RANGE_RATIO=${RANGE_RATIO} Profile"

sleep 2

echo "Starting bench with model: ${MODEL}"
python bench_serving/benchmark_serving.py \
    --model=$MODEL \
    --backend=vllm \
    --base-url=http://localhost:8000 \
    --dataset-name=random \
    --random-input-len=$ISL \
    --random-output-len=$OSL \
    --random-range-ratio ${RANGE_RATIO} \
    --num-prompts=${NUM} \
    --max-concurrency=${CON} \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ \
    --profile \
    2>&1 | tee log.bench.profile.log

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON}"
