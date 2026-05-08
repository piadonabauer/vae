#!/bin/bash
#SBATCH --job-name=vae_fusion_modes
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-1%1
#SBATCH --exclude=kn051

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
cd /home/piado/projects/aip-lindell/piado/vae/Open-Sora

export TRITON_CACHE_DIR="$SLURM_TMPDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/.torchinductor"
export PYTORCH_KERNEL_CACHE_PATH="$SLURM_TMPDIR/.pytorch_kernels"

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

train_file=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/scripts/vae/train.py
my_config=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py
sweep_name=vae_fusion_mode_sweep

EXPERIMENTS=(
  "cross_attention|--model.fusion_mode cross_attention --experiment_name fusion_cross_attention_discr_all_seq"
  #"self_attention|--model.fusion_mode self_attention --experiment_name fusion_self_attention_discr_all_seq"
  #"conv4d|--model.fusion_mode conv4d --experiment_name fusion_conv4d_discr_all_seq"
)

echo "Total experiments: ${#EXPERIMENTS[@]}"

idx=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
IFS='|' read -r exp_suffix overrides <<< "${EXPERIMENTS[$idx]}"

read -ra override_args <<< "$overrides"

echo "Running: ${sweep_name}_${exp_suffix}"
echo "Overrides: ${override_args[@]}"

nvidia-smi

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" \
    "$my_config" \
    "${override_args[@]}"
