#!/bin/bash
set -euo pipefail

# ----------------------------
# Arguments
# ----------------------------
CKPT="${1:-}"
TAG="${2:-$(basename "${CKPT:-run}")}"
EXTRA_ARGS=("${@:3}")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
  echo "Usage: bash $0 <checkpoint-dir> [tag] [extra_args...]" >&2
  exit 1
fi

# ----------------------------
# Environment
# ----------------------------
source "${ROOT}/env_eval.sh"

NUM_GPUS="${NUM_GPUS:-8}"

# ----------------------------
# Dataset configuration
# ----------------------------
DATASETS="${DATASETS:-vstar,wemath}"

# Mapping of dataset identifiers to their local file paths
declare -A DATASET_PATHS=(
  [vstar]="/path/to/vstar"
  [wemath]="/path/to/wemath"
  [art_style]="/path/to/art_style"
  [iq_test]="/path/to/iq_test"
)

# ----------------------------
# Optional subset arguments
# ----------------------------
SUBSET_ARGS=()
[[ "${EVAL_SUBSET_SIZE:-0}" != "0" ]] && SUBSET_ARGS+=(--eval_subset_size "$EVAL_SUBSET_SIZE")
[[ "${EVAL_SUBSET_SHUFFLE:-0}" == "1" ]] && SUBSET_ARGS+=(--eval_subset_shuffle)
[[ "${EVAL_DATALOADER_SHUFFLE:-0}" == "1" ]] && SUBSET_ARGS+=(--eval_dataloader_shuffle)

# ----------------------------
# Cleanup
# ----------------------------
cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ----------------------------
# Ray
# ----------------------------
start_ray() {
  ray stop --force >/dev/null 2>&1 || true

  ray start \
    --head \
    --num-gpus "$NUM_GPUS" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
    >/dev/null 2>&1 \
    || ray start \
      --head \
      --num-gpus "$NUM_GPUS" \
      --disable-dashboard \
      >/dev/null 2>&1
}

stop_ray() {
  ray stop --force >/dev/null 2>&1 || true
}

# ----------------------------
# Eval
# ----------------------------
run_dataset() {
  local name="$1"
  local data="${DATASET_PATHS[$name]:-}"
  local out="${ROOT}/results/${name}/${TAG}-${name}"

  if [[ -z "$data" ]]; then
    echo "[skip] ${name}: dataset is not defined in DATASET_PATHS"
    return
  fi

  if [[ ! -f "$data" ]]; then
    echo "[skip] ${name}: missing ${data}"
    return
  fi

  if compgen -G "${out}/pred/predictions*.jsonl" >/dev/null; then
    echo "[skip] ${name}: existing predictions"
    return
  fi

  mkdir -p "$out"

  echo "[eval] ${name}"
  echo "[eval] data: ${data}"
  echo "[eval] out : ${out}"

  start_ray

  if ! python3 -m openrlhf.cli.eval_ray \
    --pretrain "$CKPT" \
    --prompt_data "$data" \
    --eval_data "$data" \
    --ckpt_path "$out" \
    --save_path "$out" \
    --vllm_num_engines "$NUM_GPUS" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
    --vllm_sync_backend gloo \
    --vllm_enable_sleep \
    --train_vlm \
    --training_mode eval_only \
    --apply_chat_template \
    --input_key question \
    --system_prompt atlas \
    --prompt_max_len 10000 \
    --generate_max_len 8192 \
    --max_samples 100000 \
    --eval_batch_size_pergpu 64 \
    --n_samples_per_prompt 1 \
    --max_epochs 1 \
    --num_episodes 3 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 1024 \
    --advantage_estimator group \
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --adam_offload \
    --actor_num_nodes 0 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node "$NUM_GPUS" \
    --temperature 0.0 \
    --val_temperature 0.0 \
    --top_p 1.0 \
    --init_kl_coef 0.0 \
    --actor_learning_rate 10e-7 \
    --aux_loss_coef 0.05 \
    --entropy_loss_coef 0.0 \
    --normalize_reward \
    --rule_reward none \
    --data_version none \
    --buffer_norm 0 \
    --use_kl_estimator_k3 \
    --save_steps 5 \
    --max_ckpt_num 5 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --use_wandb null \
    --wandb_project vlm-rl-eval \
    --wandb_run_name "$TAG" \
    --micro_train_batch_size 4 \
    --train_batch_size 256 \
    --eval_dataloader_shuffle \
    "${SUBSET_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    >"${out}/eval_error.log" 2>&1; then

    echo "[error] ${name} failed. Last 80 lines:" >&2
    tail -n 80 "${out}/eval_error.log" >&2
    exit 1
  fi

  [[ "${KEEP_EVAL_LOGS:-0}" == "1" ]] || rm -f "${out}/eval_error.log"

  stop_ray
}

# ----------------------------
# Main
# ----------------------------
IFS=',' read -r -a SELECTED <<< "$DATASETS"

for dataset in "${SELECTED[@]}"; do
  dataset="$(echo "$dataset" | xargs)"
  [[ -z "$dataset" ]] && continue
  run_dataset "$dataset"
done

echo "[done] all selected datasets evaluated."