#!/usr/bin/env bash
#SBATCH --job-name=tc_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-12%12

# Temporal-compression training sweep — 12 parallel GPU jobs, 20h each.
# Submit:  sbatch run_tc_training_sweep.sh
# Monitor: squeue -u $USER
#
# Tasks:
#   1  tc_false_b32             tc=False  batch=32  accum=2  lr=5e-4  (reference)
#   ── tc=True ablations (all: batch=16, accum=2, lr=5e-4 unless noted) ──────
#   2  tc_true_baseline         baseline / current defaults
#   ── learning rate ──
#   3  tc_true_lr2e4            lr=2e-4
#   4  tc_true_lr1e4            lr=1e-4
#   ── perceptual loss weight (default 1.5) ──
#   5  tc_true_perc0p5          perceptual_loss_weight=0.5   (lower; less LPIPS emphasis)
#   6  tc_true_perc3p0          perceptual_loss_weight=3.0   (higher; stronger perceptual)
#   ── KL loss weight (default 1e-6, LDM default 5e-4) ──
#   7  tc_true_kl5e5            kl_loss_weight=5e-5          (50× higher; more posterior regularisation)
#   ── view consistency (default 0.0) ──
#   8  tc_true_vc0p01           view_consistency_weight=0.01
#   9  tc_true_vc0p1            view_consistency_weight=0.1  (10× stronger view agreement)
#   ── EMA decay (default 0.9999 — very slow) ──
#  10  tc_true_ema999           ema_decay=0.999              (faster EMA; tracks model sooner)
#   ── discriminators ──
#  11  tc_true_disc_light       discriminator_choice=TrainLight  (ndf=32,n_layers=3,grad_ckpt) batch=8 accum=2
#  12  tc_true_disc_light4d     discriminator_choice=TrainLight4D (ndf=32,n_layers=3,grad_ckpt,joint_views) batch=8 accum=2
#
# Common: data_preset=all_people_one_expression, optimization=False, wandb=True
#         log_every=100, log_schedule=[5,10,20,50,100,200], full_eval_every=250

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
fi

cd "$OPEN_SORA_ROOT"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

# ── Common overrides ───────────────────────────────────────────────────────────
COMMON_OVERRIDES=(
  --data_preset             all_people_one_expression
  --wandb                   True
  --optimization            False
  --FAST_MODE               False
  --profile_timing          False
  --profile_step            False
  --profile_memory_live     False
  # Logging: dense early (steps 5,10,20,50,100,200) then every 100 update steps
  --log_every               200
  --log_schedule_steps      "[5,10,20,50,100,200]"
  # Evaluation
  --full_eval_every         250
  --fixed_seq_eval_every_epochs 0
  --accumulation_steps      2
)

# ── Experiment table ───────────────────────────────────────────────────────────
# Format: "wandb_name|extra_overrides"
EXPERIMENTS=(
  "tc_false_b32|--model.temporal_compression False --batch_size 32"
  "tc_true_baseline_b16|--model.temporal_compression True --batch_size 16"
  "tc_true_lr2e4_b16|--model.temporal_compression True --batch_size 16 --learning_rate 2e-4 --optim.lr 2e-4"
  "tc_true_lr1e4_b16|--model.temporal_compression True --batch_size 16 --learning_rate 1e-4 --optim.lr 1e-4"
  "tc_true_perc0p5_b16|--model.temporal_compression True --batch_size 16 --perceptual_loss_weight 0.5"
  "tc_true_perc3p0_b16|--model.temporal_compression True --batch_size 16 --perceptual_loss_weight 3.0"
  "tc_true_kl5e5_b16|--model.temporal_compression True --batch_size 16 --kl_loss_weight 5e-5"
  "tc_true_vc0p01_b16|--model.temporal_compression True --batch_size 16 --view_consistency_weight 0.01"
  "tc_true_vc0p1_b16|--model.temporal_compression True --batch_size 16 --view_consistency_weight 0.1"
  "tc_true_ema999_b16|--model.temporal_compression True --batch_size 16 --ema_decay 0.999"
  "tc_true_disc_light_b8|--model.temporal_compression True --batch_size 8 --discriminator_choice TrainLight --gen_disc_weight 0.1"
  "tc_true_disc_light4d_b8|--model.temporal_compression True --batch_size 8 --discriminator_choice TrainLight4D --gen_disc_weight 0.1"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r wandb_name overrides <<< "${EXPERIMENTS[$idx]}"
read -ra override_args <<< "$overrides"

experiment_name="${wandb_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_name}__job${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_sweep task ${SLURM_ARRAY_TASK_ID:-?}/$n_exp — ${wandb_name}"
echo "  outputs dir : $experiment_name"
echo "  overrides   : ${override_args[*]}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000) + (SLURM_ARRAY_TASK_ID % 1000)))
else
  MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi
export MASTER_PORT MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE="${WORLD_SIZE:-1}" RANK="${RANK:-0}" LOCAL_RANK="${LOCAL_RANK:-0}"
export WANDB_NAME="$wandb_name"

DYNAMO_BACKEND=$(python3 - "$my_config" <<'PY'
import pathlib, re, sys
m = re.search(r'^\s*dynamo_backend\s*=\s*["\']([^"\']+)["\']\s*$', pathlib.Path(sys.argv[1]).read_text(), re.MULTILINE)
print(m.group(1) if m else "no")
PY
)
echo "MASTER_PORT=$MASTER_PORT  DYNAMO_BACKEND=$DYNAMO_BACKEND"

if [[ "$DRY_RUN" == "1" ]]; then
  echo accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend "$DYNAMO_BACKEND" \
    --mixed_precision bf16 --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name "$experiment_name" --wandb_expr_name "$wandb_name" \
    "${COMMON_OVERRIDES[@]}" "${override_args[@]}"
  exit 0
fi

max_retries=3; attempt=1
while (( attempt <= max_retries )); do
  echo "Launch attempt ${attempt}/${max_retries} on port ${MASTER_PORT}"
  accelerate launch \
    --num_processes 1 --num_machines 1 \
    --dynamo_backend "$DYNAMO_BACKEND" \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name "$experiment_name" \
    --wandb_expr_name "$wandb_name" \
    "${COMMON_OVERRIDES[@]}" \
    "${override_args[@]}" \
  && exit 0
  rc=$?
  (( attempt == max_retries )) && { echo "Failed after $max_retries attempts (rc=$rc)."; exit "$rc"; }
  MASTER_PORT=$(( MASTER_PORT + 1 )); export MASTER_PORT
  echo "Retrying with port $MASTER_PORT..."
  attempt=$(( attempt + 1 ))
done
