#!/usr/bin/env bash
#SBATCH --job-name=tc_false_resume
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.err

# Resume tc_false_b32 from step 2444 (epoch 375) for another 20h.
# Submit: sbatch resume_tc_false.sh

set -euo pipefail

OPEN_SORA_ROOT="/home/piado/projects/aip-lindell/piado/vae/Open-Sora"
CONFIG="configs/vae/train/wan_multiview_finetune.py"
CKPT="${OPEN_SORA_ROOT}/outputs/tc_false_b32__job4062255_t1/epoch375-global_step2444"

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
[[ -f "${CKPT}/running_states.json" ]] || { echo "Checkpoint not found: $CKPT"; exit 1; }

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export MASTER_PORT MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export WANDB_NAME="tc_false_b32_resumed"

echo "Resuming from: $CKPT"
echo "MASTER_PORT=$MASTER_PORT"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

accelerate launch \
  --num_processes 1 --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --main_process_port "$MASTER_PORT" \
  "${OPEN_SORA_ROOT}/scripts/vae/train.py" \
  "${OPEN_SORA_ROOT}/${CONFIG}" \
  --experiment_name "tc_false_b32_resume_job${SLURM_JOB_ID:-local}" \
  --wandb_expr_name "tc_false_b32_resumed" \
  --load "$CKPT" \
  --data_preset             all_people_one_expression \
  --wandb                   True \
  --optimization            False \
  --FAST_MODE               False \
  --profile_timing          False \
  --profile_step            False \
  --profile_memory_live     False \
  --log_every               100 \
  --log_schedule_steps      "[5,10,20,50,100,200]" \
  --full_eval_every         250 \
  --fixed_seq_eval_every_epochs 0 \
  --accumulation_steps      2 \
  --model.temporal_compression False \
  --batch_size              32
