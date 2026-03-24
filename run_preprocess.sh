#!/bin/bash
#SBATCH --job-name=nersemble_preprocess
#SBATCH --array=0-9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Array job: 10 tasks (0-9), each gets 1 GPU. Participants are split across tasks by SLURM_ARRAY_TASK_ID.
# To use more/fewer GPUs, change --array (e.g. 0-19 for 20 GPUs).

# Run from the directory where you submitted (sbatch run_preprocess.sh from vae/). SLURM sets SLURM_SUBMIT_DIR.
VAE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$VAE_DIR" || exit 1
mkdir -p logs

module load StdEnv cuda opencv python scipy-stack
source "$VAE_DIR/snth/bin/activate"

# Paths (override via env if needed)
NERSEMBLE_ROOT="${NERSEMBLE_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2/extracted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2/processed}"

# Each array task sees SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT; the Python script assigns a chunk of participants to this task.
srun python data/processing/preprocess_nersemble.py \
  --nersemble-root "$NERSEMBLE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --image-size 128 \
  --skip-existing
