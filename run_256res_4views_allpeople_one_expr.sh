#!/usr/bin/env bash
#SBATCH --job-name=wan_mv_256res_4v_allpeople
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A.err

# Train on: 256-res, 4 views, 9 frames, all_people_one_expression (EMO-1-shout+laugh).
# Data from the new 4-view preprocessing run (415 participants).
#
# Memory scaling vs. old 128px 2-view runs:
#   128px × 2 views × batch=32: baseline
#   256px × 4 views × batch=8 : ~2× the pixel load (safe with grad_checkpoint)
#   batch=8 × accum=4 → effective batch 32
#
# Logging / eval schedule:
#   eval_at_steps  [10, 100, 500, 1000, 1500]  — milestones early in training
#   eval_every     500                          — periodic eval thereafter
#   full_eval_every 500
#   log_every      500
#   max_steps      20 000
#
# Submit:
#   sbatch run_256res_4views_allpeople_one_expr.sh
# Dry-run:
#   DRY_RUN=1 bash run_256res_4views_allpeople_one_expr.sh

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
DRY_RUN="${DRY_RUN:-0}"

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
mkdir -p slurm_logs

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

exp_name="256res_4v_allpeople_one_expr_lr5e4"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exp_name="${exp_name}__job${SLURM_JOB_ID}"
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
export LOCAL_RANK="${LOCAL_RANK:-0}"

echo "=== 256-res 4-view all_people_one_expression (415 participants) ==="
echo "experiment_name: $exp_name"
echo "MASTER_PORT: $MASTER_PORT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  nvidia-smi || true
fi

run_cmd=(
  "$train_file"
  "$my_config"
  --experiment_name "$exp_name"
  --wandb_expr_name  "256res_4v_allpeople_one_expr_lr5e4"

  # ── Data: new 4-view preprocessed data (415 participants) ─────────────────
  --nersemble_processed_base /datasets/lindell-proj/neumayr/nersemble_v2/processed/4-frames
  --bucket_config "{'256px_ar1:1':{9:(1.0,1)}}"
  --data_preset all_people_one_expression

  # ── Optimiser / scheduler ─────────────────────────────────────────────────
  --learning_rate 5e-4
  --lr_scheduler.warmup_steps 0
  --lr_scheduler.use_exponential_decay False

  # ── Loss ──────────────────────────────────────────────────────────────────
  --discriminator_choice none

  # ── Compilation ───────────────────────────────────────────────────────────
  # Disabled: 512px OOM'd even with compile off; starting conservatively.
  # Re-enable with --optimization True --optimization_compile_mode reduce-overhead
  # once confirmed it fits in memory.
  --optimization False

  # ── Training budget ───────────────────────────────────────────────────────
  # 256px × 4 views is ~2× old 128px × 2 views per pixel; batch=8 × accum=4
  # gives effective batch=32 matching prior runs.
  --batch_size 8
  --accumulation_steps 4
  --epochs 100000
  --max_steps 20000

  # ── Logging / eval schedule ───────────────────────────────────────────────
  --eval_at_steps "[10,100,500,1000,1500]"
  --eval_every 500
  --full_eval_every 500
  --log_every 500

  # ── Misc ──────────────────────────────────────────────────────────────────
  --save_ckpt True
  --fixed_seq_eval_every_epochs 0
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend no --mixed_precision bf16 --main_process_port PORT \\"
  printf '  %q \\\n' "${run_cmd[@]}"
  echo
  exit 0
fi

_launch() {
  local port="$1"
  accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    --main_process_port "$port" \
    "${run_cmd[@]}"
}

max_port_retries=3
attempt=1
while (( attempt <= max_port_retries )); do
  echo "Launch attempt ${attempt}/${max_port_retries} on port ${MASTER_PORT}"
  if _launch "$MASTER_PORT"; then
    exit 0
  fi
  rc=$?
  if (( attempt == max_port_retries )); then
    echo "Launch failed after ${max_port_retries} attempts (rc=${rc})."
    exit "$rc"
  fi
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "Retrying with port ${MASTER_PORT}..."
  attempt=$((attempt + 1))
done
