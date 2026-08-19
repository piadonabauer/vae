#!/usr/bin/env bash
#SBATCH --job-name=tc_xattn_fft_ab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-15:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-2%2

# A/B: all_people + cross_attention + full_finetune_decoder (fft), 128px, T=5, TC on.
# Only difference:
#   TASK=1  WITH  TC weight init (LOAD_CKPT) + reinit view attention
#   TASK=2  WITHOUT TC weight init (Wan pretrained only)
#
# Submit from /project (Slurm rejects /home submits):
#   cd /project/6101839/piado/vae
#   bash ./run_tc_all_xattn_fft_init_ab.sh submit
#
# Or individually:
#   sbatch --export=ALL,TASK=1 ./run_tc_all_xattn_fft_init_ab.sh
#   sbatch --export=ALL,TASK=2 ./run_tc_all_xattn_fft_init_ab.sh
#
# Override init ckpt:
#   LOAD_CKPT=/path/to/epochX-global_stepY sbatch --export=ALL,LOAD_CKPT,TASK=1 ./run_tc_all_xattn_fft_init_ab.sh

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DRY_RUN="${DRY_RUN:-0}"
BUCKET_CONFIG="${BUCKET_CONFIG:-{'128px_ar1:1': {5: (1.0, 1)}}}"
LOAD_CKPT="${LOAD_CKPT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs/tc_true_2v128_all_people_one_expr__job4327321_b16_a2/epoch600-global_step7813}"

if [[ "${1:-}" == "submit" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  j=$(sbatch --parsable --export=ALL,LOAD_CKPT="$LOAD_CKPT" "$SCRIPT_PATH")
  echo "Submitted A/B array job ${j}:"
  echo "  ${j}_1  WITH    TC init  (t5_all_xattn_fft_init)"
  echo "  ${j}_2  WITHOUT TC init  (t5_all_xattn_fft_noinit)"
  echo "LOAD_CKPT=$LOAD_CKPT"
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

# Shared: all people, xattn, fft. Start at 8:2 (matched prior fft run).
BATCH_LADDER=( "8:2" "4:2" "2:2" "1:1" )

EXTRA=(
  --bucket_config "$BUCKET_CONFIG"
  --data_preset all_people_one_expression
  --dataset_presets.all_people_one_expression.expected_views 2
  --dataset_presets.all_people_one_expression.skip_mismatched_views True
  --val_dataset_presets.all_people_one_expression.expected_views 2
  --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
  --model.temporal_compression True
  --model.view_in 2
  --model.fusion_mode cross_attention
  --model.use_lora True
  --model.use_lora_after True
  --model.full_finetune_decoder True
  --model.train_spatial False
  --discriminator_choice none
  --wandb True
  --optimization False
  --FAST_MODE False
  --save_ckpt True
  --log_every 200
  --log_schedule_steps "[5,10,20,50,100,200]"
  --full_eval_every 250
  --fixed_seq_eval_every_epochs 0
)

case "$TASK" in
  1)
    wandb_name="t5_all_xattn_fft_init"
    if [[ -z "$LOAD_CKPT" || ! -e "$LOAD_CKPT" ]]; then
      echo "ERROR: TASK=1 requires LOAD_CKPT that exists (got: '${LOAD_CKPT}')"
      exit 1
    fi
    EXTRA+=(
      --load "$LOAD_CKPT"
      --load_optimizer False
      --model.reinit_view_attention_after_load True
    )
    init_desc="WITH TC init + reinit view attn: $LOAD_CKPT"
    ;;
  2)
    wandb_name="t5_all_xattn_fft_noinit"
    # Explicitly no --load: Wan pretrained only (config default).
    init_desc="WITHOUT TC init (Wan pretrained only)"
    ;;
  *)
    echo "Unknown TASK=$TASK (expected 1 or 2)"
    exit 1
    ;;
esac

experiment_name="${wandb_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && experiment_name="${wandb_name}__job${SLURM_JOB_ID}_t${TASK}"
MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (TASK % 1000)))
export MASTER_PORT MASTER_ADDR=127.0.0.1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 WANDB_NAME="$wandb_name"

echo "════════════════════════════════════════════════════════════════════"
echo "  TASK=${TASK}  ${wandb_name}"
echo "  ${init_desc}"
echo "  data=all_people_one_expression  fusion=cross_attention  adapt=fft"
echo "  bucket=${BUCKET_CONFIG}  ladder=${BATCH_LADDER[*]}"
echo "  experiment=${experiment_name}  MASTER_PORT=${MASTER_PORT}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

launch_one() {
  local bs="$1" acc="$2"
  local run_name="${wandb_name}_b${bs}_a${acc}"
  local exp_name="${experiment_name}_b${bs}_a${acc}"
  export WANDB_NAME="$run_name"
  echo "── launch batch=${bs} accum=${acc}  run=${run_name} ──"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo accelerate launch ... "$run_name" --batch_size "$bs" --accumulation_steps "$acc" "${EXTRA[@]}"
    return 0
  fi
  local log_tail rc
  log_tail=$(mktemp)
  set +e
  accelerate launch \
    --num_processes 1 --num_machines 1 --dynamo_backend no --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name "$exp_name" --wandb_expr_name "$run_name" \
    "${EXTRA[@]}" --batch_size "$bs" --accumulation_steps "$acc" \
    2>&1 | tee "$log_tail"
  rc=${PIPESTATUS[0]}
  if (( rc == 0 )); then rm -f "$log_tail"; return 0; fi
  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$log_tail"; then
    echo "OOM at b=${bs} a=${acc}; trying smaller..."
    rm -f "$log_tail"; return 99
  fi
  rm -f "$log_tail"; return "$rc"
}

for spec in "${BATCH_LADDER[@]}"; do
  IFS=':' read -r bs acc <<< "$spec"
  if launch_one "$bs" "$acc"; then
    echo "Success b=${bs} a=${acc}"; exit 0
  else
    rc=$?
  fi
  (( rc == 99 )) || exit "$rc"
  MASTER_PORT=$((MASTER_PORT + 1)); export MASTER_PORT
done
echo "All batches OOM'd"; exit 1
