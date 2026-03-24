#!/usr/bin/env bash
# Allocate one GPU and run 5 VAE training experiments sequentially (as long as needed).
# Submit: cd /path/to/Open-Sora && sbatch scripts/vae/run_overnight.sh
# Or run directly (uses current shell; no GPU allocation): bash scripts/vae/run_overnight.sh

#SBATCH --job-name=wan_vae_overnight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/overnight_%j.out
#SBATCH --error=outputs/overnight_%j.err

# Request one GPU and up to 7 days. Adjust --time to your cluster max if needed (e.g. 3-00:00:00).
# GPU type: change to e.g. --gres=gpu:l40s:1 if your cluster requires it.

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
CONFIG="configs/vae/train/wan_multiview_finetune.py"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/overnight_${TS}"
mkdir -p "$LOG_DIR"

run() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}.log"
  echo "=============================================="
  echo "Starting run: ${name}"
  echo "Log: ${logfile}"
  echo "=============================================="
  python3 scripts/vae/train.py "$CONFIG" "$@" 2>&1 | tee "$logfile"
  echo "Finished: ${name}"
}

# 1–4: one_person, 2000 epochs (baseline + three single-variable changes)
run "one_person_2k_baseline" \
  --data_preset one_person --epochs 200

run "one_person_2k_view_embedding" \
  --data_preset one_person --epochs 200 \
  --model.use_view_embedding True

run "one_person_2k_view_group_fusion" \
  --data_preset one_person --epochs 200 \
  --model.use_view_group_fusion True

run "one_person_2k_loss_multiview" \
  --data_preset one_person --epochs 200 \
  --vae_loss_preset multiview

# 5: all_people_one_expression, 10000 epochs, default settings
run "all_people_exp1_10k_default" \
  --data_preset all_people_one_expression --epochs 10000

echo "All 5 runs finished. Logs in ${LOG_DIR}"
