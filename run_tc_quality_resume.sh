#!/usr/bin/env bash
#SBATCH --job-name=tc_quality_resume
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-8%8

# Resume all 8 tc_quality_sweep runs from their latest checkpoint for ~17 more hours.
#
# Submit:
#   cd /home/piado/projects/aip-lindell/piado/vae
#   sbatch run_tc_quality_resume.sh
#
# Override wall-clock budget (hours):
#   RESUME_HOURS=17 sbatch run_tc_quality_resume.sh

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"

WANDB_PROJECT="${WANDB_PROJECT:-tc_quality_sweep}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-tc_quality_sweep_v1}"
EFFECTIVE_BATCH=64
RESUME_HOURS="${RESUME_HOURS:-17}"
# Baseline (bs=16) took ~8.2h for 170 epochs in job 4217773_1 (integer math only).
EXTRA_EPOCHS=$(( (170 * RESUME_HOURS * 10 + 41) / 82 ))

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
fi

cd "$OPEN_SORA_ROOT"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"
outputs_root="${OPEN_SORA_ROOT}/outputs"

# task|wandb_name|experiment_dir|batch_size|model_overrides
EXPERIMENTS=(
  "tc_q_baseline|tc_q_baseline_b16__job4217774_t1|16|"
  "tc_q_idea1_noncausal|tc_q_idea1_noncausal_b8__job4217775_t2|8|--model.use_noncausal_decode True"
  "tc_q_idea2_refpad|tc_q_idea2_refpad_b8__job4217811_t3|8|--model.use_temporal_reflection_pad True"
  "tc_q_idea3_sidechan|tc_q_idea3_sidechan_b8__job4217777_t4|8|--model.use_temporal_side_channel True --model.side_channel_dim 4"
  "tc_q_idea4_decoattn|tc_q_idea4_decoattn_b8__job4217779_t5|8|--model.use_noncausal_decode True --model.use_decoder_temporal_attention True"
  "tc_q_idea6_gru_cache|tc_q_idea6_gru_cache_b8__job4217785_t6|8|--model.use_learned_cache_update True"
  "tc_q_combo_1_2|tc_q_combo_1_2_b8__job4217781_t7|8|--model.use_noncausal_decode True --model.use_temporal_reflection_pad True"
  "tc_q_combo_1_2_4|tc_q_combo_1_2_4_b8__job4217773_t8|8|--model.use_noncausal_decode True --model.use_temporal_reflection_pad True --model.use_decoder_temporal_attention True"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r wandb_name experiment_dir batch_size model_overrides <<< "${EXPERIMENTS[$idx]}"
read -ra model_args <<< "$model_overrides"
accum=$(( EFFECTIVE_BATCH / batch_size ))
(( accum < 1 )) && accum=1

exp_path="${outputs_root}/${experiment_dir}"
if [[ ! -d "$exp_path" ]]; then
  echo "ERROR: experiment dir not found: $exp_path" >&2
  exit 1
fi

read -r ckpt_path ckpt_epoch target_epochs <<< "$(
python3 - <<PY
import glob, json, os, re, sys

exp = "${exp_path}"
extra = int("${EXTRA_EPOCHS}")
ckpts = glob.glob(os.path.join(exp, "epoch*-global_step*"))
if not ckpts:
    print("ERROR: no checkpoints", file=sys.stderr)
    sys.exit(1)

def sort_key(p):
    m = re.search(r"epoch(\d+)-global_step(\d+)", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

latest = max(ckpts, key=sort_key)
with open(os.path.join(latest, "running_states.json"), encoding="utf-8") as f:
    rs = json.load(f)
epoch = int(rs.get("epoch", 0))
target = epoch + extra
print(latest, epoch, target)
PY
)" || { echo "Failed to resolve checkpoint for ${exp_path}" >&2; exit 1; }

wandb_run_name="${wandb_name}_b${batch_size}"

MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (${SLURM_ARRAY_TASK_ID:-1} % 1000)))
export MASTER_PORT MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_quality_resume task ${SLURM_ARRAY_TASK_ID:-?}/${n_exp} — ${wandb_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && echo "  SLURM: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node: $(hostname)"
echo "  resume from: ${ckpt_path}"
echo "  ckpt epoch=${ckpt_epoch}  target_epochs=${target_epochs}  (+${EXTRA_EPOCHS} epochs ≈ ${RESUME_HOURS}h)"
echo "  batch=${batch_size}  accum=${accum}  effective=${EFFECTIVE_BATCH}"
echo "  experiment_dir=${experiment_dir}"
echo "  wandb: project=${WANDB_PROJECT}  group=${WANDB_RUN_GROUP}  run=${wandb_run_name}"
echo "  model overrides: ${model_args[*]:-<none>}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true
echo ""

[[ -f "${ckpt_path}/running_states.json" ]] || { echo "Checkpoint invalid: $ckpt_path"; exit 1; }

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --main_process_port "$MASTER_PORT" \
  "$train_file" \
  "$my_config" \
  --experiment_name "$experiment_dir" \
  --wandb_expr_name "$wandb_run_name" \
  --load "$ckpt_path" \
  --data_preset all_people_one_expression \
  --model.temporal_compression True \
  --batch_size "$batch_size" \
  --accumulation_steps "$accum" \
  --epochs "$target_epochs" \
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
  ${model_args[@]+"${model_args[@]}"}

if [[ -f "${exp_path}/wandb_run_url.txt" ]]; then
  echo "✓ WandB: $(tr -d '\n' < "${exp_path}/wandb_run_url.txt")"
else
  echo "WARNING: no wandb_run_url.txt in ${exp_path}" >&2
fi

echo "Done: ${wandb_name} (resumed ${ckpt_epoch} → ${target_epochs})"
