#!/usr/bin/env bash
#SBATCH --job-name=tc_2v128
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-15:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.err

# Two TC=True, no-disc, 2-view @ 128px training runs.
#
# Prefer submitting via the helper (different wall times):
#   bash run_tc_2view_128_runs.sh submit
#
# Or submit one task manually:
#   sbatch --job-name=tc_2v128_ss  --time=0-05:00:00 --export=ALL,TASK=1 run_tc_2view_128_runs.sh
#   sbatch --job-name=tc_2v128_all --time=0-15:00:00 --export=ALL,TASK=2 run_tc_2view_128_runs.sh
#
# Tasks:
#   1  single_sequence              batch=1  accum=1   wall=5h
#   2  all_people_one_expression    batch ladder below wall=15h
#        try: 16/2 → 8/2 → 4/2 → 2/2 → 1/1  (retry only on CUDA OOM; 32 OOMs)
#
# Common: temporal_compression=True, discriminator=none, 128-res (2-view), wandb on.
#
# Note: config single_sequence points at p017/.../frames.pt (missing + V=1).
# This script overrides to p018 EMO-1 (V=2). all_people skips mismatched views.

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DRY_RUN="${DRY_RUN:-0}"

DATA_ROOT_128="${DATA_ROOT_128:-/datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res}"
# Only EMO-1 outlier with V=1 is p017; use a known 2-view clip for single_sequence.
SINGLE_SEQ_PT="${SINGLE_SEQ_PT:-${DATA_ROOT_128}/p018/EMO-1-shout+laugh/EMO-1-shout+laugh.pt}"

# ── submit helper (local invocation) ──────────────────────────────────────────
if [[ "${1:-}" == "submit" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  j1=$(sbatch --parsable --job-name=tc_2v128_ss  --time=0-05:00:00 \
    --export=ALL,TASK=1 "$SCRIPT_PATH")
  j2=$(sbatch --parsable --job-name=tc_2v128_all --time=0-15:00:00 \
    --export=ALL,TASK=2 "$SCRIPT_PATH")
  echo "Submitted TASK=1 (single_sequence, 5h):  job ${j1}"
  echo "Submitted TASK=2 (all_people, 15h):     job ${j2}"
  exit 0
fi

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Set TASK=1|2, or run: bash $SCRIPT_PATH submit"
  exit 1
fi

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
fi

cd "$OPEN_SORA_ROOT"
train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

COMMON_OVERRIDES=(
  --model.temporal_compression True
  --model.view_in 2
  --discriminator_choice none
  --wandb True
  --optimization False
  --FAST_MODE False
  --profile_timing False
  --profile_step False
  --profile_memory_live False
  --log_every 200
  --log_schedule_steps "[5,10,20,50,100,200]"
  --full_eval_every 250
  --fixed_seq_eval_every_epochs 0
  --save_ckpt True
)

case "$TASK" in
  1)
    wandb_name="tc_true_2v128_single_seq"
    data_preset="single_sequence"
    # Fixed batch; no OOM ladder
    BATCH_LADDER=( "1:1" )
    EXTRA_OVERRIDES=(
      --data_preset "$data_preset"
      --dataset_presets.single_sequence.data_path "$SINGLE_SEQ_PT"
    )
    ;;
  2)
    wandb_name="tc_true_2v128_all_people_one_expr"
    data_preset="all_people_one_expression"
    EXTRA_OVERRIDES=(
      --data_preset "$data_preset"
      --dataset_presets.all_people_one_expression.expected_views 2
      --dataset_presets.all_people_one_expression.skip_mismatched_views True
      --val_dataset_presets.all_people_one_expression.expected_views 2
      --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
    )
    # batch:accum — start large, fall back on CUDA OOM only
    BATCH_LADDER=( "16:2" "8:2" "4:2" "2:2" "1:1" )
    ;;
  *)
    echo "Unknown TASK=$TASK (expected 1 or 2)"
    exit 1
    ;;
esac

experiment_name="${wandb_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_name}__job${SLURM_JOB_ID}"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000) + (TASK % 100)))
else
  MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi
export MASTER_PORT MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export WANDB_NAME="$wandb_name"

DYNAMO_BACKEND=$(python3 - "$my_config" <<'PY'
import pathlib, re, sys
m = re.search(r'^\s*dynamo_backend\s*=\s*["\']([^"\']+)["\']\s*$', pathlib.Path(sys.argv[1]).read_text(), re.MULTILINE)
print(m.group(1) if m else "no")
PY
)

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_2v128 TASK=${TASK} — ${wandb_name}"
echo "  data_preset : ${data_preset}"
echo "  experiment  : ${experiment_name}"
echo "  SINGLE_SEQ  : ${SINGLE_SEQ_PT}"
echo "  MASTER_PORT : ${MASTER_PORT}  DYNAMO_BACKEND=${DYNAMO_BACKEND}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

launch_train() {
  local bs="$1" acc="$2"
  local run_name="${wandb_name}_b${bs}_a${acc}"
  local exp_name="${experiment_name}_b${bs}_a${acc}"
  local log_tail rc
  export WANDB_NAME="$run_name"

  echo "── launch batch=${bs} accum=${acc}  run=${run_name} ──"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend "$DYNAMO_BACKEND" \
      --mixed_precision bf16 --main_process_port "$MASTER_PORT" \
      "$train_file" "$my_config" \
      --experiment_name "$exp_name" --wandb_expr_name "$run_name" \
      "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}" \
      --batch_size "$bs" --accumulation_steps "$acc"
    return 0
  fi

  # Capture output so we can detect OOM and retry with a smaller batch.
  # IMPORTANT: keep set +e through any non-zero return. Re-enabling set -e
  # before `return 99` makes bash exit the whole job instead of falling through
  # the batch ladder (this is what killed job 4321228 after the first OOM).
  log_tail=$(mktemp)
  set +e
  accelerate launch \
    --num_processes 1 --num_machines 1 \
    --dynamo_backend "$DYNAMO_BACKEND" \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name "$exp_name" \
    --wandb_expr_name "$run_name" \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    --batch_size "$bs" \
    --accumulation_steps "$acc" \
    2>&1 | tee "$log_tail"
  rc=${PIPESTATUS[0]}

  if (( rc == 0 )); then
    rm -f "$log_tail"
    set -e
    return 0
  fi

  if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError|CUDA error: out of memory' "$log_tail"; then
    echo "CUDA OOM at batch=${bs} accum=${acc} (rc=${rc}); will try smaller batch if available."
    rm -f "$log_tail"
    return 99
  fi

  echo "Training failed with non-OOM exit code ${rc}."
  rm -f "$log_tail"
  return "$rc"
}

oom_retries=0
for spec in "${BATCH_LADDER[@]}"; do
  IFS=':' read -r bs acc <<< "$spec"
  # Call in a conditional so set -e cannot abort on a non-zero return.
  if launch_train "$bs" "$acc"; then
    echo "Success with batch=${bs} accum=${acc}"
    exit 0
  else
    rc=$?
  fi
  if (( rc == 99 )); then
    oom_retries=$((oom_retries + 1))
    MASTER_PORT=$((MASTER_PORT + 1))
    export MASTER_PORT
    continue
  fi
  exit "$rc"
done

echo "All batch ladder attempts OOM'd (${oom_retries} OOM retries). Giving up."
exit 1
