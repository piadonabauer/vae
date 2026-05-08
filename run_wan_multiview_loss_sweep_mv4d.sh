#!/usr/bin/env bash
#SBATCH --job-name=wan_train_mv4d_loss_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:59:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-54

# Train + TrainMultiview4D sweep:
#   2 discriminator × 3 perceptual × 3 gen_disc_weight × 3 KL = 54 runs
#   - perceptual: 1.5, 2.0, 3.0
#   - gen_disc_weight: 0.1, 0.2, 0.3
#   - KL: 1e-6, 1e-7, 1e-8
#   - epochs: 30 (passed explicitly; overrides config)
#   - bucket: 128px 9 frames (Python literal for train.py; see opensora/utils/config.py merge_args)
# Submit:
#   sbatch /home/piado/projects/aip-lindell/piado/vae/run_wan_multiview_loss_sweep_mv4d.sh
# Optional:
#   sbatch --array=1-54%8 /home/piado/projects/aip-lindell/piado/vae/run_wan_multiview_loss_sweep_mv4d.sh
# Env overrides:
#   WANDB_PREFIX, OPEN_SORA_ROOT, DRY_RUN=1, DYNAMO_BACKEND=inductor

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
WANDB_PREFIX="${WANDB_PREFIX:-}"
DRY_RUN="${DRY_RUN:-0}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-}"

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
sweep_name=wan_multiview_train_mv4d_loss_sweep

# Must match train.py re-resolve path (no spaces inside — safe for shell word-splitting).
EPOCHS=30
BUCKET_CONFIG="{'128px_ar1:1':{9:(1.0,1)}}"

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

EXPERIMENTS=()
discs=(Train TrainMultiview4D)
percs=(1.5 2.0 3.0)
gdws=(0.1 0.2 0.3)
kls=(1e-6 1e-7 1e-8)

disc_short() {
  case "$1" in
    none) echo none ;;
    Train) echo train ;;
    TrainMultiview4D) echo mv4d ;;
    *) echo "$1" ;;
  esac
}

perc_short() {
  case "$1" in
    1.5) echo 1p5 ;;
    2.0) echo 2p0 ;;
    3.0) echo 3p0 ;;
    *) echo "$1" | tr . p ;;
  esac
}

gdw_short() {
  case "$1" in
    0.1) echo d01 ;;
    0.2) echo d02 ;;
    0.3) echo d03 ;;
    *) echo "$1" | tr . p ;;
  esac
}

kl_short() {
  case "$1" in
    1e-6) echo k1em6 ;;
    1e-7) echo k1em7 ;;
    1e-8) echo k1em8 ;;
    *) echo "$1" ;;
  esac
}

for disc in "${discs[@]}"; do
  ds=$(disc_short "$disc")
  for perc in "${percs[@]}"; do
    ps=$(perc_short "$perc")
    for gdw in "${gdws[@]}"; do
      gs=$(gdw_short "$gdw")
      for kl in "${kls[@]}"; do
        ks=$(kl_short "$kl")
        key="sweep_${ds}__perc${ps}__${gs}__${ks}"
        overrides="--discriminator_choice ${disc} --vae_loss_config.perceptual_loss_weight ${perc} --gen_disc_weight ${gdw} --sweep_gen_disc_weight ${gdw} --vae_loss_config.kl_loss_weight ${kl}"
        EXPERIMENTS+=("${key}|${overrides}")
      done
    done
  done
done

n_exp=${#EXPERIMENTS[@]}
if [[ "$n_exp" -ne 54 ]]; then
  echo "Internal error: expected 54 experiments, got $n_exp"
  exit 1
fi

idx=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
if ((idx < 0 || idx >= n_exp)); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} -> idx=$idx out of range [0,$((n_exp - 1))] (n=$n_exp)"
  exit 1
fi

IFS='|' read -r wandb_key overrides <<< "${EXPERIMENTS[$idx]}"
read -ra override_args <<< "$overrides"
wandb_name="${WANDB_PREFIX}${wandb_key}"

experiment_name="${wandb_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_name}__job${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
fi

echo "=== ${sweep_name} task ${SLURM_ARRAY_TASK_ID:-1}/$n_exp idx=$idx ==="
echo "wandb_expr_name=$wandb_name"
echo "experiment_name (outputs dir)=$experiment_name"
echo "epochs=${EPOCHS} bucket_config=${BUCKET_CONFIG}"
echo "Overrides: ${override_args[*]}"

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
export LOCAL_RANK="${LOCAL_RANK:-0}"

export WANDB_NAME="$wandb_name"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "DYNAMO_BACKEND=$DYNAMO_BACKEND"

run_cmd=(
  accelerate launch
  --num_processes 1
  --num_machines 1
  --dynamo_backend "$DYNAMO_BACKEND"
  --mixed_precision bf16
  --main_process_port "$MASTER_PORT"
  "$train_file"
  "$my_config"
  --epochs "$EPOCHS"
  --bucket_config "$BUCKET_CONFIG"
  --experiment_name "$experiment_name"
  --wandb_expr_name "$wandb_name"
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
  if "${run_cmd[@]}"; then
    exit 0
  fi

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
    --epochs "$EPOCHS"
    --bucket_config "$BUCKET_CONFIG"
    --experiment_name "$experiment_name"
    --wandb_expr_name "$wandb_name"
    "${override_args[@]}"
  )
  echo "Retrying with port ${MASTER_PORT}..."
  attempt=$((attempt + 1))
done
