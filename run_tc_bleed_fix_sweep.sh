#!/usr/bin/env bash
#SBATCH --job-name=tc_bleed_fix_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-8%8

# sbatch --array=8 run_tc_bleed_fix_sweep.sh

# Temporal-compression BLEEDING-FIX sweep — follow-up to run_tc_quality_sweep.sh.
#
# Context: discriminator alone (tc_q_*) did not fix the bleeding. This sweep tests
# the remaining "keep it simple" ideas (no discriminator dependence) plus one new
# minimal-footprint module (idea8), on all_people_one_expression (real diversity,
# not single-sequence overfitting -- the bleeding only shows up with >1 sample).
#
# Submit:
#   cd /home/piado/projects/aip-lindell/piado/vae
#   sbatch run_tc_bleed_fix_sweep.sh
#
# Resubmit only failed array tasks (e.g. after OOM that exhausted the fallback ladder):
#   sbatch --array=3,6 run_tc_bleed_fix_sweep.sh
#
# Local (no SLURM) smoke test of one arm, e.g. task 2:
#   SLURM_ARRAY_TASK_ID=2 OPEN_SORA_ROOT=/home/coder/vae/Open-Sora VAE_VENV=/home/coder/vae/snth/bin/activate \
#     TRAIN_EPOCHS=1 bash run_tc_bleed_fix_sweep.sh
#
# Batch-size auto-fallback: each task STARTS at its configured batch size; if the
# launch OOMs, batch size is halved (accumulation_steps doubled to keep the SAME
# effective batch = 64 across all arms, so wandb charts stay comparable — see
# EFFECTIVE_BATCH / samples_seen below) and retried, up to MAX_OOM_RETRIES times.
# This only helps with an OOM in the first few minutes (graph capture / early
# steps); an OOM many epochs in loses that progress since this is a fresh
# (non---load) retry. If that happens, resubmit the failed task_id with a smaller
# manual batch size, or extend this script to detect+resume from a checkpoint.
#
# Tasks:
#   1  baseline_frozen_enc     tc=True, current defaults (train_spatial=False) — bleeding baseline, no other changes
#   2  unfreeze_encoder        train_spatial=True + freeze_temporal=False (encoder can finally adapt to the domain)
#   3  temporal_diff_loss      temporal_diff_loss_weight=2.0 (idea7: zero-param loss, penalizes wrong frame deltas directly)
#   4  decoder_capacity        full_finetune_decoder=True (decoder no longer LoRA-rank-bottlenecked)
#   5  unfreeze_enc+tdiff      combo of 2+3
#   6  unfreeze_enc+full_dec   combo of 2+4
#   7  disc+unfreeze_enc       discriminator (TrainLight4D) RETESTED together with an unfrozen encoder
#   8  subframe_pos_embed      idea8 (new, minimal module: ~1.5K params) + temporal_diff_loss

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"

WANDB_PROJECT="${WANDB_PROJECT:-tc_bleed_fix_sweep}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-tc_bleed_fix_sweep_v1}"
EFFECTIVE_BATCH=64
TRAIN_EPOCHS="${TRAIN_EPOCHS:-170}"   # ~1020 optimizer updates at eff.batch=64 (same budget as tc_quality_sweep)
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-3}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  # shellcheck source=/dev/null
  source "$VAE_VENV"
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export WANDB_PROJECT="$WANDB_PROJECT"
  export WANDB_RUN_GROUP="$WANDB_RUN_GROUP"
elif [[ -f "$VAE_VENV" ]]; then
  # Local (non-SLURM) smoke test: still honor VAE_VENV if provided.
  # shellcheck source=/dev/null
  source "$VAE_VENV"
fi

cd "$OPEN_SORA_ROOT"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

# Format: "wandb_name|model_overrides|batch_size"
# batch_size is the STARTING point for the OOM-fallback ladder (see run_with_oom_fallback).
EXPERIMENTS=(
  "tc_bf_baseline_frozen_enc|--model.freeze_temporal True --model.train_spatial False|8"
  "tc_bf_unfreeze_encoder|--model.freeze_temporal False --model.train_spatial True|8"
  "tc_bf_temporal_diff_loss|--temporal_diff_loss_weight 2.0|8"
  "tc_bf_decoder_capacity|--model.full_finetune_decoder True|8"
  "tc_bf_unfreeze_enc_tdiff|--model.freeze_temporal False --model.train_spatial True --temporal_diff_loss_weight 2.0|8"
  "tc_bf_unfreeze_enc_full_dec|--model.freeze_temporal False --model.train_spatial True --model.full_finetune_decoder True|8"
  "tc_bf_disc_unfreeze_enc|--model.freeze_temporal False --model.train_spatial True --discriminator_choice TrainLight4D|4"
  "tc_bf_subframe_pos_embed|--model.use_subframe_position_embedding True --temporal_diff_loss_weight 2.0|8"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r wandb_name model_overrides start_batch_size <<< "${EXPERIMENTS[$idx]}"
start_batch_size="${start_batch_size:-16}"
read -ra model_args <<< "$model_overrides"

wandb_run_name="${wandb_name}_b${start_batch_size}"
experiment_name="${wandb_run_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_run_name}__job${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
fi

MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (${SLURM_ARRAY_TASK_ID:-1} % 1000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_bleed_fix_sweep task ${SLURM_ARRAY_TASK_ID:-?}/${n_exp} — ${wandb_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && echo "  SLURM: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node: $(hostname)"
echo "  starting batch=${start_batch_size} (effective batch fixed at ${EFFECTIVE_BATCH} regardless of fallback)"
echo "  epochs=${TRAIN_EPOCHS}  wandb: project=${WANDB_PROJECT} group=${WANDB_RUN_GROUP} run=${wandb_run_name}"
echo "  model overrides: ${model_args[*]:-<none>}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true
echo ""

# run_with_oom_fallback <batch_size> -> launches training at that batch size (accum computed
# to hit EFFECTIVE_BATCH); on CUDA OOM, halves batch_size (floor 1) and retries, up to
# MAX_OOM_RETRIES times. Any non-OOM failure fails immediately (no point retrying that).
run_with_oom_fallback() {
  local batch_size="$1"
  local attempt=0
  local log_file
  log_file="$(mktemp)"

  while true; do
    local accum=$(( EFFECTIVE_BATCH / batch_size ))
    (( accum < 1 )) && accum=1
    local this_master_port=$(( MASTER_PORT + attempt ))

    echo "[attempt $((attempt + 1))/$((MAX_OOM_RETRIES + 1))] batch_size=${batch_size} accumulation_steps=${accum} (effective=$(( batch_size * accum )))"

    set +e
    accelerate launch \
      --num_processes 1 \
      --num_machines 1 \
      --dynamo_backend no \
      --mixed_precision bf16 \
      --main_process_port "$this_master_port" \
      "$train_file" \
      "$my_config" \
      --experiment_name "$experiment_name" \
      --wandb_expr_name "$wandb_run_name" \
      --data_preset all_people_one_expression \
      --model.temporal_compression True \
      --batch_size "$batch_size" \
      --accumulation_steps "$accum" \
      --epochs "$TRAIN_EPOCHS" \
      --wandb True \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_min_steps_before_init 1 \
      --optimization False \
      --FAST_MODE False \
      --profile_timing False \
      --profile_step False \
      --profile_memory_live False \
      --save_ckpt True \
      --eval_every 20 \
      --log_every 50 \
      --log_schedule_steps "[5,10,20,50,100,200]" \
      --full_eval_every 250 \
      --fixed_seq_eval_every_epochs 0 \
      ${model_args[@]+"${model_args[@]}"} 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ "$exit_code" -eq 0 ]]; then
      rm -f "$log_file"
      return 0
    fi

    if grep -qiE "out of memory|CUDA error: out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$log_file"; then
      attempt=$((attempt + 1))
      if (( attempt > MAX_OOM_RETRIES )) || (( batch_size <= 1 )); then
        echo "OOM persists after ${attempt} attempt(s) at batch_size as low as ${batch_size}; giving up." >&2
        rm -f "$log_file"
        return 1
      fi
      batch_size=$(( batch_size / 2 ))
      (( batch_size < 1 )) && batch_size=1
      echo "Detected OOM — retrying with batch_size=${batch_size} (accumulation_steps auto-adjusted to keep effective batch=${EFFECTIVE_BATCH})." >&2
      sleep 5
      continue
    fi

    echo "Training failed with exit_code=${exit_code} for a non-OOM reason; not retrying. See output above." >&2
    rm -f "$log_file"
    return "$exit_code"
  done
}

run_with_oom_fallback "$start_batch_size"

out_dir="${OPEN_SORA_ROOT}/outputs/${experiment_name}"
if [[ -f "${out_dir}/wandb_run_url.txt" ]]; then
  echo "✓ WandB: $(tr -d '\n' < "${out_dir}/wandb_run_url.txt")"
else
  echo "WARNING: no wandb_run_url.txt in ${out_dir}" >&2
fi

echo "Done: ${wandb_run_name}"
