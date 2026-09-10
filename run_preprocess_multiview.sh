#!/bin/bash
#SBATCH --job-name=nersemble_preprocess_multiview
#SBATCH --output=/project/6101839/piado/vae/logs/preprocess_mv_%A_%a.out
#SBATCH --error=/project/6101839/piado/vae/logs/preprocess_mv_%A_%a.err
#SBATCH --array=0-418%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=aip-lindell

# Preprocess EMO-1-shout+laugh for 4-view and 8-view at 128px.
# Output: processed/4-views/128-res/ and processed/8-views/128-res/
# Array index maps to participant tar file (same as the all-expressions job).
# With 20 concurrent tasks and ~2 min/participant this finishes in ~45 min.

set -euo pipefail

module load gcc opencv/4.11.0
source /project/6101839/piado/vae/snth/bin/activate

PROCESSED_ROOT=/datasets/lindell-proj/neumayr/nersemble_v2/processed

# 4-view at 128px
python /project/6101839/piado/vae/data/processing/preprocess_nersemble.py \
    --nersemble-root   /datasets/lindell-proj/neumayr/nersemble_v2 \
    --from-tars \
    --output-root      "${PROCESSED_ROOT}/4-views" \
    --upper-views      4 \
    --frames           13 \
    --image-size       128 \
    --color-correction \
    --only-sequences   EMO-1-shout+laugh \
    --skip-existing \
    --temp-dir         /scratch/piado

# 8-view at 128px
python /project/6101839/piado/vae/data/processing/preprocess_nersemble.py \
    --nersemble-root   /datasets/lindell-proj/neumayr/nersemble_v2 \
    --from-tars \
    --output-root      "${PROCESSED_ROOT}/8-views" \
    --upper-views      8 \
    --frames           13 \
    --image-size       128 \
    --color-correction \
    --only-sequences   EMO-1-shout+laugh \
    --skip-existing \
    --temp-dir         /scratch/piado
