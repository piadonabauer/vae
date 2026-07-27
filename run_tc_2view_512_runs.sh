#!/usr/bin/env bash
#SBATCH --job-name=tc_2v512
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-15:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.err

# Two TC=True, no-disc, 2-view @ 512px training runs.
#
# Submit BOTH (must be from /project, not /home):
#   cd /project/6101839/piado/vae
#   bash ./run_tc_2view_512_runs.sh submit
#
# Or one task:
#   sbatch --job-name=tc_2v512_ss  --time=0-05:00:00 --export=ALL,TASK=1 ./run_tc_2view_512_runs.sh
#   sbatch --job-name=tc_2v512_all --time=0-15:00:00 --export=ALL,TASK=2 ./run_tc_2view_512_runs.sh
#
# Tasks:
#   1  single_sequence              batch=1 accum=1   wall=5h
#   2  all_people_one_expression    OOM ladder wall=15h
#        try: 8/2 → 4/2 → 2/2 → 1/1  (32/16 OOM at 512 on L40S; bucket T=9)
#
# Common: temporal_compression=True, discriminator=none, 512-res (2-view), wandb on.

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DRY_RUN="${DRY_RUN:-0}"

DATA_ROOT_512="${DATA_ROOT_512:-/datasets/lindell-proj/neumayr/nersemble_v2/processed/512-res}"
# 512-res uses frames.pt [V,T,C,H,W]; p017 EMO-1 is V=2 here.
SINGLE_SEQ_PT="${SINGLE_SEQ_PT:-${DATA_ROOT_512}/p017/EMO-1-shout+laugh/frames.pt}"
BUCKET_CONFIG="${BUCKET_CONFIG:-{'512px_ar1:1': {9: (1.0, 1)}}}"

# ── submit helper ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "submit" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  j1=$(sbatch --parsable --job-name=tc_2v512_ss  --time=0-05:00:00 \
    --export=ALL,TASK=1 "$SCRIPT_PATH")
  j2=$(sbatch --parsable --job-name=tc_2v512_all --time=0-15:00:00 \
    --export=ALL,TASK=2 "$SCRIPT_PATH")
  echo "Submitted TASK=1 (single_sequence @512, 5h):  job ${j1}"
  echo "Submitted TASK=2 (all_people @512, 15h):      job ${j2}"
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
  --bucket_config "$BUCKET_CONFIG"
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
    wandb_name="tc_true_2v512_single_seq"
    data_preset="single_sequence"
    BATCH_LADDER=( "1:1" )
    EXTRA_OVERRIDES=(
      --data_preset "$data_preset"
      --dataset_presets.single_sequence.data_path "$SINGLE_SEQ_PT"
    )
    ;;
  2)
    wandb_name="tc_true_2v512_all_people_one_expr"
    data_preset="all_people_one_expression"
    EXTRA_OVERRIDES=(
      --data_preset "$data_preset"
      --dataset_presets.all_people_one_expression.expected_views 2
      --dataset_presets.all_people_one_expression.skip_mismatched_views True
      --val_dataset_presets.all_people_one_expression.expected_views 2
      --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
    )
    # Start at 32/2; on CUDA OOM only, halve batch (keep accum=2 until batch=1).
    BATCH_LADDER=( "8:2" "4:2" "2:2" "1:1" )
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
echo "  tc_2v512 TASK=${TASK} — ${wandb_name}"
echo "  data_preset : ${data_preset}"
echo "  bucket      : ${BUCKET_CONFIG}"
echo "  experiment  : ${experiment_name}"
echo "  SINGLE_SEQ  : ${SINGLE_SEQ_PT}"
echo "  ladder      : ${BATCH_LADDER[*]}"
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

  # Keep set +e through non-zero returns so the OOM ladder can continue.
  # (Re-enabling set -e before `return 99` previously aborted the whole job.)
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
  # Conditional call so set -e cannot abort on a non-zero return.
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
