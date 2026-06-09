#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JUDGE_MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct"

JUDGE_PORT="${JUDGE_PORT:-8080}"
JUDGE_TP="${JUDGE_TP:-8}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.95}"
JUDGE_DTYPE="${JUDGE_DTYPE:-float16}"
JUDGE_SERVED_MODEL="${JUDGE_SERVED_MODEL:-JUDGE}"
JUDGE_WAIT_SECONDS="${JUDGE_WAIT_SECONDS:-3600}"

JUDGE_LOG="${JUDGE_LOG:-${ROOT}/judge_server.log}"

# Start Judge Server
echo "[judge] starting server with model $JUDGE_MODEL"
mkdir -p "$(dirname "$JUDGE_LOG")"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_SERVED_MODEL" \
    --trust-remote-code \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size "$JUDGE_TP" \
    --gpu-memory-utilization "$JUDGE_GPU_UTIL" \
    --dtype "$JUDGE_DTYPE" \
    >"$JUDGE_LOG" 2>&1 &

JUDGE_PID=$!
echo "[judge] PID $JUDGE_PID, logging to $JUDGE_LOG"

# Wait for Judge Server to Start
for elapsed in $(seq 1 "$JUDGE_WAIT_SECONDS"); do
    if python3 "${ROOT}/judge_results.py" --check-server >/dev/null 2>&1; then
        echo "[judge] server ready!"
        break
    fi
    if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then
        echo "[judge] server exited early, see $JUDGE_LOG" >&2
        exit 1
    fi
    if (( elapsed % 30 == 0 )); then
        echo "[judge] waiting for server... ${elapsed}s/${JUDGE_WAIT_SECONDS}s"
    fi
    sleep 1
done

# Judge evaluation
found=0

for dataset_root in "$ROOT"/results/*; do
  [[ -d "$dataset_root" ]] || continue

  for run_dir in "$dataset_root"/*; do
    [[ -d "$run_dir/logs" ]] || continue
    found=1
    echo "[judge] evaluating $run_dir"
    python3 "${ROOT}/judge_results.py" "$run_dir"
  done
done

if [[ "$found" == "0" ]]; then
  echo "[judge] no result directories found under $ROOT/results/*/*/logs"
fi

echo "[judge] done"