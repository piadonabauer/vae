#!/bin/bash
# Single SLURM job: one GPU, process all tasks for 10 participants (including 017) in sequence.
# No array – use this when you want one GPU job instead of many array tasks.
#
# Usage: sbatch run_single_job.sh

#SBATCH --job-name=nersemble_preprocess_single
#SBATCH --output=logs/preprocess_single_%A.out
#SBATCH --error=logs/preprocess_single_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00  # Long enough for all tasks on one GPU (adjust if needed)

set -e

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT="/home/piado/projects/aip-lindell/piado"
SCRIPT_DIR="${PROJECT_ROOT}/data/processing"
VENV_PATH="/project/6101839/piado/snth"
NERSEMBLE_ROOT="/scratch/piado/data/nersemble"
OUTPUT_ROOT="${PROJECT_ROOT}/data/preprocessed_initial_experiments"
TASK_FILE="${SCRIPT_DIR}/tasks.json"
MATTING_CHECKPOINT="${PROJECT_ROOT}/data/rvm_mobilenetv3.pth"

# ============================================================================
# Setup
# ============================================================================
source "${VENV_PATH}/bin/activate"
cd "${SCRIPT_DIR}"
mkdir -p logs
mkdir -p "${OUTPUT_ROOT}"

# ============================================================================
# 1. Discover 10 participants (017 required)
# ============================================================================
echo "Discovering 10 participants (including 017)..."
python3 preprocess_nersemble.py discover \
    --nersemble-root "${NERSEMBLE_ROOT}" \
    --output "${TASK_FILE}" \
    --target-participants 10 \
    --required-participant 17

N=$(python3 -c "import json; print(len(json.load(open('${TASK_FILE}'))))")
echo "Found ${N} tasks. Processing on one GPU..."

# ============================================================================
# 2. Process each task sequentially on this GPU
# ============================================================================
for i in $(seq 0 $((N - 1))); do
  OUT=$(python3 -c "import json; t=json.load(open(\"${TASK_FILE}\")); ti=t[${i}]; print(\"${OUTPUT_ROOT}/p\"+str(ti[\"participant_id\"])+\"_\"+ti[\"sequence_name\"])")
  echo "=========================================="
  echo "Task $((i+1))/${N} -> ${OUT}"
  echo "=========================================="
  CMD="python3 ${SCRIPT_DIR}/preprocess_nersemble.py process"
  CMD="${CMD} --task-index ${i}"
  CMD="${CMD} --task-file ${TASK_FILE}"
  CMD="${CMD} --nersemble-root ${NERSEMBLE_ROOT}"
  CMD="${CMD} --output-dir ${OUT}"
  CMD="${CMD} --device cuda"
  if [ -f "${MATTING_CHECKPOINT}" ]; then
    CMD="${CMD} --matting-checkpoint ${MATTING_CHECKPOINT} --matting-variant mobilenetv3"
  fi
  ${CMD}
done

echo "Done. Processed ${N} tasks -> ${OUTPUT_ROOT}"
