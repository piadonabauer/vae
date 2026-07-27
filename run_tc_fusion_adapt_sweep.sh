#!/usr/bin/env bash
#SBATCH --job-name=tc_fusion_adapt
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-15:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-8%8

# Fusion adaptation sweep @ 128px, T=5, TC=True, no disc, 2-view.
# Init from a TC-trained checkpoint (LOAD_CKPT). Fusion attention is re-randomized
# after load for attention modes (REINIT_FUSION_ATTN=1).
#
# Submit from /project:
#   cd /project/6101839/piado/vae
#   LOAD_CKPT=/path/to/epochX-global_stepY \
#     sbatch --export=ALL,LOAD_CKPT ./run_tc_fusion_adapt_sweep.sh
#
# Tasks (data × fusion × adapt):
#   1  one_person   + cross_attention + LoRA
#   2  one_person   + cross_attention + full_finetune_decoder
#   3  one_person   + conv4d (factorized) + LoRA
#   4  one_person   + conv4d + full_finetune_decoder
#   5  all_people_one_expression + cross_attention + LoRA
#   6  all_people_one_expression + cross_attention + full_finetune_decoder
#   7  all_people_one_expression + conv4d + LoRA
#   8  all_people_one_expression + conv4d + full_finetune_decoder

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
DRY_RUN="${DRY_RUN:-0}"
BUCKET_CONFIG="${BUCKET_CONFIG:-{'128px_ar1:1': {5: (1.0, 1)}}}"
# Default to latest successful 128 TC all-people run if present
LOAD_CKPT="${LOAD_CKPT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs/tc_true_2v128_all_people_one_expr__job4327321_b16_a2/epoch600-global_step7813}"
REINIT_FUSION_ATTN="${REINIT_FUSION_ATTN:-1}"

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
TASK="${SLURM_ARRAY_TASK_ID:-1}"

# wandb_name|data_preset|fusion|adapt_flags|batch:accum ladder start
EXPERIMENTS=(
  "t5_1p_xattn_lora|one_person|cross_attention|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder False --model.train_spatial False|16:2"
  "t5_1p_xattn_fft|one_person|cross_attention|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder True --model.train_spatial False|8:2"
  "t5_1p_conv4d_lora|one_person|conv4d|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder False --model.train_spatial False|16:2"
  "t5_1p_conv4d_fft|one_person|conv4d|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder True --model.train_spatial False|8:2"
  "t5_all_xattn_lora|all_people_one_expression|cross_attention|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder False --model.train_spatial False|16:2"
  "t5_all_xattn_fft|all_people_one_expression|cross_attention|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder True --model.train_spatial False|8:2"
  "t5_all_conv4d_lora|all_people_one_expression|conv4d|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder False --model.train_spatial False|16:2"
  "t5_all_conv4d_fft|all_people_one_expression|conv4d|--model.use_lora True --model.use_lora_after True --model.full_finetune_decoder True --model.train_spatial False|8:2"
)

idx=$(( TASK - 1 ))
IFS='|' read -r wandb_name data_preset fusion_mode adapt_flags start_batch <<< "${EXPERIMENTS[$idx]}"
read -ra adapt_args <<< "$adapt_flags"
IFS=':' read -r start_bs start_acc <<< "$start_batch"

# Build OOM ladder from start batch downward
BATCH_LADDER=()
bs=$start_bs
while (( bs >= 1 )); do
  if (( bs >= 2 )); then
    BATCH_LADDER+=( "${bs}:2" )
  else
    BATCH_LADDER+=( "1:1" )
  fi
  bs=$(( bs / 2 ))
done

EXTRA=(
  --bucket_config "$BUCKET_CONFIG"
  --data_preset "$data_preset"
  --model.temporal_compression True
  --model.view_in 2
  --model.fusion_mode "$fusion_mode"
  --discriminator_choice none
  --wandb True
  --optimization False
  --FAST_MODE False
  --save_ckpt True
  --log_every 200
  --log_schedule_steps "[5,10,20,50,100,200]"
  --full_eval_every 250
  --fixed_seq_eval_every_epochs 0
  ${adapt_args[@]+"${adapt_args[@]}"}
)

if [[ "$data_preset" == "all_people_one_expression" ]]; then
  EXTRA+=(
    --dataset_presets.all_people_one_expression.expected_views 2
    --dataset_presets.all_people_one_expression.skip_mismatched_views True
    --val_dataset_presets.all_people_one_expression.expected_views 2
    --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
  )
elif [[ "$data_preset" == "one_person" ]]; then
  EXTRA+=(
    --dataset_presets.one_person.expected_views 2
    --dataset_presets.one_person.skip_mismatched_views True
  )
fi

if [[ -n "$LOAD_CKPT" && -e "$LOAD_CKPT" ]]; then
  # Load model weights only (new run counters / fresh Adam).
  EXTRA+=( --load "$LOAD_CKPT" --load_optimizer False )
  echo "Warm-start from TC ckpt (weights only): $LOAD_CKPT"
else
  echo "WARNING: LOAD_CKPT missing ($LOAD_CKPT); training from Wan pretrained only."
fi

# Re-randomize fusion attention after load so TC backbone is kept but view-attn is fresh.
# Implemented via env flag read in train.py if present; otherwise print reminder.
if [[ "$REINIT_FUSION_ATTN" == "1" && "$fusion_mode" == "cross_attention" ]]; then
  EXTRA+=( --model.reinit_view_attention_after_load True )
fi

experiment_name="${wandb_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && experiment_name="${wandb_name}__job${SLURM_JOB_ID}_t${TASK}"
MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (TASK % 1000)))
export MASTER_PORT MASTER_ADDR=127.0.0.1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 WANDB_NAME="$wandb_name"

echo "TASK=$TASK  $wandb_name  data=$data_preset  fusion=$fusion_mode  ladder=${BATCH_LADDER[*]}"

launch_one() {
  local bs="$1" acc="$2"
  local run_name="${wandb_name}_b${bs}_a${acc}"
  local exp_name="${experiment_name}_b${bs}_a${acc}"
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
