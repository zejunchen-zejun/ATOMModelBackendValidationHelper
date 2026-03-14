#!/bin/bash
echo "run atom + vllm + profile"

# for plugin
unset VLLM_ATTENTION_BACKEND

# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# for profiling
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=./

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/data/models/DeepSeek-R1-0528-MXFP4

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":%s,"torch_profiler_record_shapes":%s}' \
    "${VLLM_TORCH_PROFILER_DIR}" \
    "$([[ "${VLLM_TORCH_PROFILER_WITH_STACK:-0}" == "1" ]] && echo true || echo false)" \
    "$([[ "${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}" == "1" ]] && echo true || echo false)")

echo "profiler_config: $profiler_config"
echo "Starting server with model: $model_path"

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 18432 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --profiler-config "$profiler_config" \
    2>&1 | tee log.serve.log &

    # --kv-cache-dtype fp8 \
    # --max-model-len 32768 \
    # --max-num-batched-tokens 16384 \
    # --no-enable-chunked-prefill \
    # --enable-expert-parallel \
