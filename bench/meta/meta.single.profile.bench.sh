#!bin/bash

# MODEL=/shared/data/amd_int/models/Qwen3-235B-A22B-Instruct-2507-FP8
# MODEL=/shared/data/amd_int/models/DeepSeek-R1-0528-MXFP4
MODEL=/shared/data/amd_int/models/DeepSeek-R1-0528

RANGE_RATIO=0.8

CON=4
# CON=16
# CON=32
# CON=64

# 1K/1K
ISL=1024
OSL=1024
NUM=$(( CON * 4 ))

# 8K/1K
#ISL=8192
#OSL=1024
#NUM=$(( CON * 5 ))

# 1K/8K
# ISL=1024
# OSL=8192
# NUM=$(( CON * 5 ))

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON} RANGE_RATIO=${RANGE_RATIO} Profile"

sleep 2

# git clone https://github.com/kimbochen/bench_serving.git
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
    2>&1 | tee log.bench.log

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON}"
rm -rf ./*.json
