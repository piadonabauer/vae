#!/bin/bash
#SBATCH --job-name=nersemble_preprocess_8v_one_expr
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/logs/%x_%A_%a.err

# Preprocess NeRSemble: all people, EMO-1-shout+laugh, 8 camera views.
# Each tar is extracted ONCE and saved at all 4 resolutions in a single pass.
# 10 array tasks split participants across GPUs.
#
# Camera serials (8 views, same set as run_preprocess.sh):
#   222200042 222200046 222200036 220700191 222200037 222200047 222200049 221501007
#
# Output tree:
#   /datasets/lindell-proj/neumayr/nersemble_v2/processed/8-views/256-res/
#   /datasets/lindell-proj/neumayr/nersemble_v2/processed/8-views/512-res/
#   /datasets/lindell-proj/neumayr/nersemble_v2/processed/8-views/1024-res/
#   /datasets/lindell-proj/neumayr/nersemble_v2/processed/8-views/2048-res/
#
# Submit:
#   sbatch run_preprocess_8v_one_expression_resolutions.sh

set -euo pipefail

VAE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$VAE_DIR" || exit 1
mkdir -p logs

NERSEMBLE_ROOT="${NERSEMBLE_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/datasets/lindell-proj/neumayr/nersemble_v2/processed/8-views}"
TEMP_DIR="${TEMP_DIR:-${SCRATCH:-/scratch/${USER}}}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module load StdEnv cuda opencv python scipy-stack
  source "$VAE_DIR/snth/bin/activate"
fi

srun python3 "$VAE_DIR/data/processing/preprocess_nersemble.py" \
  --nersemble-root "$NERSEMBLE_ROOT" \
  --from-tars \
  --output-root "$OUTPUT_ROOT" \
  --image-sizes 256 512 1024 2048 \
  --frames 9 \
  --camera-serials 222200042 222200046 222200036 220700191 222200037 222200047 222200049 221501007 \
  --images-subdir images \
  --use-tar-alpha \
  --only-sequences EMO-1-shout+laugh \
  --temp-dir "$TEMP_DIR"
