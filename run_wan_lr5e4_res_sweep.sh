#!/usr/bin/env bash
#SBATCH --job-name=lr5e4_res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-07:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-4%4

# LR 5e-4 + constant LR × resolution × view-count data tree (4 parallel jobs, 7 h each).
# Uses configs/vae/train/wan_multiview_finetune.py; each task changes bucket + processed base.
#
# Submit:  sbatch run_wan_lr5e4_res_sweep.sh
# Monitor: squeue -u $USER
#
# Shared settings (all tasks):
#   learning_rate / optim.lr = 5e-4
#   no LR scheduling (warmup_steps=0, use_exponential_decay=False)
#   profile_timing off → wandb on
#
# Tasks (2×2 grid):
#   1  lr5e4_256px_4frames   — 256px bucket, nersemble …/processed/4-frames
#   2  lr5e4_256px_8frames   — 256px bucket, nersemble …/processed/8-frames
#   3  lr5e4_512px_4frames   — 512px bucket, nersemble …/processed/4-frames
#   4  lr5e4_512px_8frames   — 512px bucket, nersemble …/processed/8-frames
#
# Data note: 4-frames/8-frames trees only ship 128-res on disk; 256/512 bucket keys load
# 128-res and resize on the fly (see opensora/utils/nersemble_bucket.py fallback).

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
WANDB_PREFIX="${WANDB_PREFIX:-lr5e4_}"
DRY_RUN="${DRY_RUN:-0}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-}"
NERSEMBLE_BASE="${NERSEMBLE_BASE:-/datasets/lindell-proj/neumayr/nersemble_v2/processed}"

# Always applied (keep wandb logging; skip one-shot profile step)
COMMON_OVERRIDES=(
  --profile_timing False
  --profile_step False
  --learning_rate 5e-4
  --optim.lr 5e-4
  --lr_scheduler.warmup_steps 0
  --lr_scheduler.use_exponential_decay False
)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  # shellcheck source=/dev/null
  source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
fi

cd "$OPEN_SORA_ROOT"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"
sweep_name=wan_lr5e4_res_sweep

if [[ -z "$DYNAMO_BACKEND" ]]; then
  DYNAMO_BACKEND=$(python3 - "$my_config" <<'PY'
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
text = config_path.read_text(encoding="utf-8")
m = re.search(r'^\s*dynamo_backend\s*=\s*["\']([^"\']+)["\']\s*$', text, re.MULTILINE)
print(m.group(1) if m else "no")
PY
)
fi

# Format: "run_key|frames_tree|bucket_key"
# bucket_key examples: 256px_ar1:1, 512px_ar1:1  (train T=9 from bucket inner key)
EXPERIMENTS=(
  "256px_4frames|4-frames|256px_ar1:1"
  "256px_8frames|8-frames|256px_ar1:1"
  "512px_4frames|4-frames|512px_ar1:1"
  "512px_8frames|8-frames|512px_ar1:1"
)

n_exp=${#EXPERIMENTS[@]}
idx=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} -> idx=$idx out of range [0,$((n_exp - 1))] (n=$n_exp)"
  exit 1
fi

IFS='|' read -r run_key frames_tree bucket_key <<< "${EXPERIMENTS[$idx]}"
processed_base="${NERSEMBLE_BASE}/${frames_tree}"
bucket_config="{'${bucket_key}': {9: (1.0, 1)}}"

override_args=(
  --nersemble_processed_base "$processed_base"
  --bucket_config "$bucket_config"
)

wandb_name="${WANDB_PREFIX}${run_key}"
experiment_name="${wandb_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_name}__job${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
fi

echo "=== ${sweep_name} task ${SLURM_ARRAY_TASK_ID:-1}/$n_exp idx=$idx ==="
echo "wandb_expr_name=$wandb_name"
echo "experiment_name (outputs dir)=$experiment_name"
echo "Config: $my_config"
echo "nersemble_processed_base=$processed_base"
echo "bucket_config=$bucket_config"
echo "Common overrides: ${COMMON_OVERRIDES[*]}"
echo "Experiment overrides: ${override_args[*]}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  nvidia-smi || true
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000) + (SLURM_ARRAY_TASK_ID % 1000)))
else
  MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
fi
export MASTER_PORT
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK=0}"

export WANDB_NAME="$wandb_name"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "DYNAMO_BACKEND=$DYNAMO_BACKEND"
echo "wandb=True (config default; profile_timing forced False in this script)"

run_cmd=(
  accelerate launch
  --num_processes 1
  --num_machines 1
  --dynamo_backend "$DYNAMO_BACKEND"
  --mixed_precision bf16
  --main_process_port "$MASTER_PORT"
  "$train_file"
  "$my_config"
  --experiment_name "$experiment_name"
  --wandb_expr_name "$wandb_name"
  "${COMMON_OVERRIDES[@]}"
  "${override_args[@]}"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${run_cmd[@]}"
  echo
  exit 0
fi

max_port_retries=3
attempt=1
while (( attempt <= max_port_retries )); do
  echo "Launch attempt ${attempt}/${max_port_retries} on port ${MASTER_PORT}"
  "${run_cmd[@]}" && exit 0
  rc=$?
  if (( attempt == max_port_retries )); then
    echo "Launch failed after ${max_port_retries} attempts (last port ${MASTER_PORT}, rc=${rc})."
    exit "$rc"
  fi

  MASTER_PORT=$((MASTER_PORT + 1))
  export MASTER_PORT
  run_cmd=(
    accelerate launch
    --num_processes 1
    --num_machines 1
    --dynamo_backend "$DYNAMO_BACKEND"
    --mixed_precision bf16
    --main_process_port "$MASTER_PORT"
    "$train_file"
    "$my_config"
    --experiment_name "$experiment_name"
    --wandb_expr_name "$wandb_name"
    "${COMMON_OVERRIDES[@]}"
    "${override_args[@]}"
  )
  echo "Retrying with port ${MASTER_PORT}..."
  attempt=$((attempt + 1))
done
