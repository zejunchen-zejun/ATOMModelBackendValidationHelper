#!/bin/bash
echo "run sgl"
export CUDA_VISIBLE_DEVICES=7

rm -rf /root/.cache/

model_path=/shared/data/amd_int/models/Qwen3-0.6B

echo "Starting server with model: $model_path"

python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.1 \
    --model-impl atom \
    --disable-cuda-graph \
    2>&1 | tee log.serve.log &
