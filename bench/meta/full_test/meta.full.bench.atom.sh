#!/bin/bash
set -euo pipefail

# MODEL=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8
MODEL="${MODEL:-/shared/data/amd_int/models/DeepSeek-R1-0528-MXFP4}"
# MODEL=/data/models/DeepSeek-R1-0528

RANGE_RATIO="${RANGE_RATIO:-0.8}"
NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-5}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-3600}"
SERVER_SHUTDOWN_TIMEOUT_SECONDS="${SERVER_SHUTDOWN_TIMEOUT_SECONDS:-600}"
SERVER_READY_POLL_SECONDS="${SERVER_READY_POLL_SECONDS:-5}"
SERVER_SHUTDOWN_POLL_SECONDS="${SERVER_SHUTDOWN_POLL_SECONDS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/../atom/test.ds.fp4.sh"
BENCH_SERVING_DIR="${SCRIPT_DIR}/bench_serving"
BENCH_SERVING_REPO_URL="${BENCH_SERVING_REPO_URL:-https://github.com/kimbochen/bench_serving.git}"
BENCHMARK_SCRIPT="${BENCH_SERVING_DIR}/benchmark_serving.py"
SERVER_NAME="ATOM"
SERVER_PGREP_PATTERN="atom\\.entrypoints\\.openai_server"
SERVER_TEE_PGREP_PATTERN="tee .*log\\.serve\\.log"

declare -a CASE_NAMES=("1K/1K" "8K/1K" "1K/8K")
declare -a INPUT_LENS=(1024 8192 1024)
declare -a OUTPUT_LENS=(1024 1024 8192)
declare -a CONCURRENCIES=(16 32 64)

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: ${SERVER_NAME} launch script not found: ${SERVER_SCRIPT}" >&2
    exit 1
fi

ensure_benchmark_runner() {
    if [[ -f "${BENCHMARK_SCRIPT}" ]]; then
        return 0
    fi

    if [[ -e "${BENCH_SERVING_DIR}" ]]; then
        echo "ERROR: ${BENCH_SERVING_DIR} exists, but ${BENCHMARK_SCRIPT} was not found." >&2
        return 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git is not installed, unable to clone bench_serving." >&2
        return 1
    fi

    echo "Cloning bench_serving from ${BENCH_SERVING_REPO_URL}"
    git clone --depth 1 "${BENCH_SERVING_REPO_URL}" "${BENCH_SERVING_DIR}"

    if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
        echo "ERROR: cloned bench_serving but benchmark script was not found: ${BENCHMARK_SCRIPT}" >&2
        return 1
    fi
}

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
RESULT_ROOT="${SCRIPT_DIR}/results/meta.full.${RUN_TAG}"
SUMMARY_MD="${RESULT_ROOT}/summary.md"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"

mkdir -p "${RESULT_ROOT}"

declare -a SERVER_ROOT_PIDS=()

discover_server_root_pids() {
    local -a matched_pids=()
    mapfile -t matched_pids < <(pgrep -f "${SERVER_PGREP_PATTERN}" || true)
    SERVER_ROOT_PIDS=("${matched_pids[@]}")
}

collect_descendant_pids() {
    local pid="$1"
    local child_pid=""

    while IFS= read -r child_pid; do
        [[ -z "${child_pid}" ]] && continue
        echo "${child_pid}"
        collect_descendant_pids "${child_pid}"
    done < <(pgrep -P "${pid}" || true)
}

collect_server_pids() {
    local pid=""
    local child_pid=""
    declare -A seen_pids=()

    for pid in "${SERVER_ROOT_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            seen_pids["${pid}"]=1
            while IFS= read -r child_pid; do
                [[ -n "${child_pid}" ]] && seen_pids["${child_pid}"]=1
            done < <(collect_descendant_pids "${pid}")
        fi
    done

    if [[ "${#seen_pids[@]}" -eq 0 ]]; then
        return 0
    fi

    for pid in "${!seen_pids[@]}"; do
        echo "${pid}"
    done
}

http_probe() {
    python - "$@" <<'PY'
import sys
import urllib.request

for target in sys.argv[1:]:
    try:
        with urllib.request.urlopen(target, timeout=5) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"unexpected status: {response.status}")
    except Exception:
        raise SystemExit(1)

raise SystemExit(0)
PY
}

port_probe() {
    python - "${BASE_URL}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname or "127.0.0.1"
port = parsed.port or (443 if parsed.scheme == "https" else 80)

sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()

raise SystemExit(0)
PY
}

is_server_ready() {
    http_probe "${BASE_URL}/health" "${BASE_URL}/v1/models"
}

wait_for_server_pids() {
    local deadline=$((SECONDS + 60))

    while (( SECONDS < deadline )); do
        discover_server_root_pids
        if [[ "${#SERVER_ROOT_PIDS[@]}" -gt 0 ]]; then
            return 0
        fi
        sleep 1
    done

    return 1
}

wait_for_server_ready() {
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))

    while (( SECONDS < deadline )); do
        if is_server_ready; then
            echo "${SERVER_NAME} server is ready."
            return 0
        fi

        discover_server_root_pids
        if [[ "${#SERVER_ROOT_PIDS[@]}" -eq 0 ]]; then
            echo "ERROR: ${SERVER_NAME} process exited before becoming ready." >&2
            return 1
        fi

        sleep "${SERVER_READY_POLL_SECONDS}"
    done

    echo "ERROR: timed out waiting for ${SERVER_NAME} server readiness." >&2
    return 1
}

stop_server() {
    local -a graceful_pids=()
    local -a target_pids=()
    local -a tee_pids=()
    local pid=""
    local alive=0
    local deadline=0

    discover_server_root_pids
    if [[ "${#SERVER_ROOT_PIDS[@]}" -eq 0 ]]; then
        if ! port_probe; then
            echo "No running ${SERVER_NAME} server detected on ${BASE_URL}."
            return 0
        fi
    fi

    mapfile -t target_pids < <(collect_server_pids)
    if [[ "${#target_pids[@]}" -eq 0 ]]; then
        target_pids=("${SERVER_ROOT_PIDS[@]}")
    fi
    graceful_pids=("${SERVER_ROOT_PIDS[@]}")

    mapfile -t tee_pids < <(pgrep -f "${SERVER_TEE_PGREP_PATTERN}" || true)

    echo "Stopping ${SERVER_NAME} server. PID(s): ${target_pids[*]}"
    # ATOM openai_server installs a SIGINT handler that closes the engine and
    # waits for worker descendants, which is more reliable than plain SIGTERM.
    kill -INT "${graceful_pids[@]}" 2>/dev/null || true

    deadline=$((SECONDS + SERVER_SHUTDOWN_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        alive=0
        for pid in "${target_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=1
                break
            fi
        done

        if (( alive == 0 )) && ! port_probe; then
            discover_server_root_pids
            if [[ "${#SERVER_ROOT_PIDS[@]}" -eq 0 ]]; then
                echo "${SERVER_NAME} server has stopped."
                return 0
            fi
        fi

        sleep "${SERVER_SHUTDOWN_POLL_SECONDS}"
    done

    echo "${SERVER_NAME} server did not exit after SIGINT, escalating to SIGTERM." >&2
    kill -TERM "${target_pids[@]}" 2>/dev/null || true
    if [[ "${#tee_pids[@]}" -gt 0 ]]; then
        kill -TERM "${tee_pids[@]}" 2>/dev/null || true
    fi
    sleep 10

    discover_server_root_pids
    if [[ "${#SERVER_ROOT_PIDS[@]}" -eq 0 ]] && ! port_probe; then
        echo "${SERVER_NAME} server has stopped after SIGTERM."
        return 0
    fi

    echo "${SERVER_NAME} server still alive, forcing kill." >&2
    kill -KILL "${target_pids[@]}" 2>/dev/null || true
    pkill -KILL -f "${SERVER_PGREP_PATTERN}" 2>/dev/null || true
    if [[ "${#tee_pids[@]}" -gt 0 ]]; then
        kill -KILL "${tee_pids[@]}" 2>/dev/null || true
    fi
    sleep 5
    discover_server_root_pids
    if [[ "${#SERVER_ROOT_PIDS[@]}" -gt 0 ]] || port_probe; then
        echo "ERROR: ${SERVER_NAME} server still exists after force kill." >&2
        return 1
    fi

    echo "${SERVER_NAME} server has been force stopped."
}

cleanup_on_exit() {
    local exit_code=$?

    discover_server_root_pids
    if [[ "${#SERVER_ROOT_PIDS[@]}" -gt 0 ]]; then
        echo "Cleaning up running ${SERVER_NAME} server before exit..."
        stop_server || true
    fi

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "Benchmark terminated early. Partial results are under: ${RESULT_ROOT}" >&2
    fi
}

trap cleanup_on_exit EXIT INT TERM

write_summary_header() {
    cat > "${SUMMARY_MD}" <<EOF
# Benchmark Summary

Model: \`${MODEL}\`

Range ratio: \`${RANGE_RATIO}\`

Num prompts per case: \`concurrency * ${NUM_PROMPTS_FACTOR}\`

| Case | ISL | OSL | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | TTPS (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
EOF

    printf "case,isl,osl,concurrency,mean_ttft_ms,mean_tpot_ms,total_token_throughput,bench_log,result_json\n" > "${SUMMARY_CSV}"
}

extract_metrics() {
    local bench_log="$1"
    local result_json="${2:-}"

    python - "${bench_log}" "${result_json}" <<'PY'
import json
import os
import re
import sys

bench_log = sys.argv[1]
result_json = sys.argv[2] if len(sys.argv) > 2 else ""

patterns = {
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)",
    "total_token_throughput": r"Total Token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)",
}

values = {}

if os.path.isfile(bench_log):
    with open(bench_log, encoding="utf-8", errors="ignore") as infile:
        log_text = infile.read()
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            values[key] = float(match.group(1))

if result_json and os.path.isfile(result_json):
    with open(result_json, encoding="utf-8") as infile:
        data = json.load(infile)
    values.setdefault("mean_ttft_ms", float(data["mean_ttft_ms"]))
    values.setdefault("mean_tpot_ms", float(data["mean_tpot_ms"]))
    values.setdefault("total_token_throughput", float(data["total_token_throughput"]))

missing_keys = [key for key in patterns if key not in values]
if missing_keys:
    raise SystemExit(
        "Failed to extract benchmark metrics from log/json. Missing: "
        + ", ".join(missing_keys)
    )

print(f"{values['mean_ttft_ms']:.2f}\t{values['mean_tpot_ms']:.2f}\t{values['total_token_throughput']:.2f}")
PY
}

append_summary_row() {
    local case_name="$1"
    local input_len="$2"
    local output_len="$3"
    local concurrency="$4"
    local mean_ttft="$5"
    local mean_tpot="$6"
    local ttps="$7"
    local bench_log="$8"
    local result_json="$9"

    printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
        "${case_name}" \
        "${input_len}" \
        "${output_len}" \
        "${concurrency}" \
        "${mean_ttft}" \
        "${mean_tpot}" \
        "${ttps}" >> "${SUMMARY_MD}"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${case_name}" \
        "${input_len}" \
        "${output_len}" \
        "${concurrency}" \
        "${mean_ttft}" \
        "${mean_tpot}" \
        "${ttps}" \
        "${bench_log}" \
        "${result_json}" >> "${SUMMARY_CSV}"
}

launch_server() {
    local case_dir="$1"

    echo "Launching ${SERVER_NAME} server with: ${SERVER_SCRIPT}"
    (
        cd "${case_dir}"
        bash "${SERVER_SCRIPT}"
    )

    if ! wait_for_server_pids; then
        echo "ERROR: ${SERVER_NAME} process was not detected after launch." >&2
        return 1
    fi

    echo "Detected ${SERVER_NAME} root PID(s): ${SERVER_ROOT_PIDS[*]}"
    wait_for_server_ready
}

run_benchmark() {
    local input_len="$1"
    local output_len="$2"
    local concurrency="$3"
    local result_dir="$4"
    local result_json="$5"
    local num_prompts=$((concurrency * NUM_PROMPTS_FACTOR))

    python "${BENCHMARK_SCRIPT}" \
        --model="${MODEL}" \
        --backend=vllm \
        --base-url="${BASE_URL}" \
        --dataset-name=random \
        --random-input-len="${input_len}" \
        --random-output-len="${output_len}" \
        --random-range-ratio "${RANGE_RATIO}" \
        --num-prompts="${num_prompts}" \
        --max-concurrency="${concurrency}" \
        --request-rate=inf \
        --ignore-eos \
        --save-result \
        --percentile-metrics="ttft,tpot,itl,e2el" \
        --result-dir="${result_dir}" \
        --result-filename="$(basename "${result_json}")"
}

run_case() {
    local case_name="$1"
    local input_len="$2"
    local output_len="$3"
    local concurrency="$4"
    local case_slug="isl${input_len}_osl${output_len}_con${concurrency}"
    local case_dir="${RESULT_ROOT}/${case_slug}"
    local result_json="${case_dir}/result.json"
    local bench_log="${case_dir}/log.bench.log"
    local mean_ttft=""
    local mean_tpot=""
    local ttps=""

    mkdir -p "${case_dir}"

    echo
    echo "=================================================="
    echo "Running case=${case_name} concurrency=${concurrency}"
    echo "ISL=${input_len} OSL=${output_len} NUM=$((concurrency * NUM_PROMPTS_FACTOR)) RANGE_RATIO=${RANGE_RATIO}"
    echo "Artifacts=${case_dir}"
    echo "=================================================="

    stop_server
    launch_server "${case_dir}"

    run_benchmark "${input_len}" "${output_len}" "${concurrency}" "${case_dir}" "${result_json}" \
        2>&1 | tee "${bench_log}"

    if [[ ! -f "${bench_log}" ]]; then
        echo "ERROR: benchmark log not found: ${bench_log}" >&2
        return 1
    fi

    if [[ ! -f "${result_json}" ]]; then
        echo "WARN: benchmark result json not found, metrics will be parsed from log only: ${result_json}" >&2
    fi

    IFS=$'\t' read -r mean_ttft mean_tpot ttps < <(extract_metrics "${bench_log}" "${result_json}")

    echo "Collected metrics: mean_ttft_ms=${mean_ttft}, mean_tpot_ms=${mean_tpot}, ttps=${ttps}"
    append_summary_row \
        "${case_name}" \
        "${input_len}" \
        "${output_len}" \
        "${concurrency}" \
        "${mean_ttft}" \
        "${mean_tpot}" \
        "${ttps}" \
        "${bench_log}" \
        "${result_json}"

    stop_server
}

main() {
    local case_index=0
    local concurrency=""

    ensure_benchmark_runner
    write_summary_header

    echo "ATOM Model=${MODEL}"
    echo "Benchmark runner=${BENCHMARK_SCRIPT}"
    echo "${SERVER_NAME} launcher=${SERVER_SCRIPT}"
    echo "Results will be written to ${RESULT_ROOT}"

    for case_index in "${!CASE_NAMES[@]}"; do
        for concurrency in "${CONCURRENCIES[@]}"; do
            run_case \
                "${CASE_NAMES[${case_index}]}" \
                "${INPUT_LENS[${case_index}]}" \
                "${OUTPUT_LENS[${case_index}]}" \
                "${concurrency}"
        done
    done

    echo
    echo "All benchmark cases completed."
    echo "Summary markdown: ${SUMMARY_MD}"
    echo "Summary csv: ${SUMMARY_CSV}"
    echo
    cat "${SUMMARY_MD}"
}

main "$@"
