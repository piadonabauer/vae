#!/bin/bash
#SBATCH --job-name=nersemble_preprocess
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/logs/%x_%A_%a.err

# GPUs: each array task requests exactly ONE GPU (--gres=gpu:l40s:1). The line --array=0-9
# launches 10 separate tasks (when the full array runs), so up to 10 GPUs are used in parallel,
# each running a disjoint chunk of .tar archives. For a single-GPU single job, use e.g.
#   #SBATCH --array=0-0
# or submit without array and unset SLURM_ARRAY_TASK_ID (then one task processes all tars).

# Run from the directory where you submitted (sbatch run_preprocess.sh from vae/). SLURM sets SLURM_SUBMIT_DIR.
VAE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$VAE_DIR" || exit 1
mkdir -p logs

module load StdEnv cuda opencv python scipy-stack
source "$VAE_DIR/snth/bin/activate"

# --output-root is the *parent*; the script always appends "<image-size>-res" (here: 128-res).
NERSEMBLE_ROOT="${NERSEMBLE_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2/processed/4-frames}"
# Large temp for tar extract; override with TEMP_DIR=... (Compute Canada: often same as $SCRATCH).
TEMP_DIR="${TEMP_DIR:-${SCRATCH:-/scratch/${USER}}}"

srun python3 /home/piado/projects/aip-lindell/piado/vae/data/processing/preprocess_nersemble.py \
  --nersemble-root "$NERSEMBLE_ROOT" \
  --from-tars \
  --output-root "$OUTPUT_ROOT" \
  --image-size 128 \
  --frames 9 \
  --images-subdir images_fgr \
  --disable-background-removal \
  --camera-serials 222200036 220700191 222200037 222200047 \
  --only-sequences EMO-1-shout+laugh \
  --temp-dir "$TEMP_DIR" \
  --skip-existing
