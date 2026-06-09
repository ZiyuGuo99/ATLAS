#!/bin/bash
set -euo pipefail

# ROOT should be defined by the caller script before sourcing this file.
if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ----------------------------
# vLLM and VLM image resolution
# ----------------------------
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export MIN_PIXELS="${MIN_PIXELS:-401408}"
export MAX_PIXELS="${MAX_PIXELS:-4014080}"

# ----------------------------
# NCCL
# ----------------------------
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# ----------------------------
# Host / vLLM
# ----------------------------
export HOST_IP="${HOST_IP:-0.0.0.0}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-0.0.0.0}"

# ----------------------------
# Dataset / Ray runtime
# ----------------------------
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

# ----------------------------
# Eval runtime
# ----------------------------
export ATLAS_EVAL_RUNTIME_DIR="${ATLAS_EVAL_RUNTIME_DIR:-/tmp/atlas_eval_runtime}"