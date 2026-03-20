#!/bin/bash
echo "run atom original"

rm -rf /root/.cache/

model_path=/data/models/gpt-oss-120b

echo "Starting server with model: $model_path"

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 2 \
    --enable-dp-attention \
    --enable-expert-parallel \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee log.serve.log &
