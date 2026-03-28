#!/bin/bash
echo "run atom original"
export CUDA_VISIBLE_DEVICES=7
# export AMD_SERIALIZE_KERNEL=3

rm -rf /root/.cache/

export ATOM_TORCH_PROFILER_DIR=./profiler_traces

model_path=/shared/data/amd_int/models/Qwen3-0.6B

echo "Starting server with model: $model_path"

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.1 \
    --kv_cache_dtype fp8 \
    --torch-profiler-dir $ATOM_TORCH_PROFILER_DIR \
    2>&1 | tee log.serve.log &
