#!/bin/bash
echo "run vllm amd_dev"

export VLLM_DISABLE_COMPILE_CACHE=1
export AMDGCN_USE_BUFFER_OPS=1

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_TRITON_LINEAR=1
export VLLM_USE_FUSED_ALLREDUCE_RMSNORM=1
export VLLM_ROCM_USE_AITER_MLA_PS=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=True

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/shared/data/models/DeepSeek-R1-0528

# for profiling
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=./

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":%s,"torch_profiler_record_shapes":%s}' \
    "${VLLM_TORCH_PROFILER_DIR}" \
    "$([[ "${VLLM_TORCH_PROFILER_WITH_STACK:-0}" == "1" ]] && echo true || echo false)" \
    "$([[ "${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}" == "1" ]] && echo true || echo false)")

echo "profiler_config: $profiler_config"
echo "Starting server with model: $model_path"

vllm serve "$model_path" \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --distributed-executor-backend mp \
    --async-scheduling \
    --swap-space 64 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 65536 \
    --block-size 1 \
    --gpu-memory-utilization 0.94 \
    --kv-cache-dtype fp8 \
    --max-model-len 2068 \
    --profiler-config "$profiler_config" \
    2>&1 | tee log.serve.log &

    # --max-model-len 32768 \
    # --max-num-batched-tokens 16384 \
