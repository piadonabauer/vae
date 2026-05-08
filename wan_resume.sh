#!/usr/bin/env bash
#SBATCH --job-name=wan_mv_resume
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%j.err

set -euo pipefail

OPEN_SORA_ROOT="/home/piado/projects/aip-lindell/piado/vae/Open-Sora"
CONFIG="/home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py"
CKPT="/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs/260415_022826-_home_piado_projects_aip-lindell_piado_vae_Open-Sora_vae_train_wan_multiview_finetune_lpdefault_per1.5_kl1e-06_vc0.0_discnone_gdwna/epoch35-global_step14500"

mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate

export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"

cd "$OPEN_SORA_ROOT"

MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
export MASTER_PORT
export MASTER_ADDR=127.0.0.1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend inductor \
  --mixed_precision bf16 \
  --main_process_port "$MASTER_PORT" \
  /project/6101839/piado/vae/Open-Sora/scripts/vae/train.py \
  "$CONFIG" \
  --load "$CKPT"