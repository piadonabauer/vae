#!/bin/bash
# Sequential local runner for the paper final wave on a single L40S
# (replaces sbatch + CHAIN_LEFT chaining; no slurm on this machine).
#
# Usage:
#   bash run_local_queue.sh                    # run the default tiered queue
#   QUEUE="2 3 4" bash run_local_queue.sh      # run specific TASKs in order
#   TRAIN_EPOCHS=300 QUEUE="9 36 52" bash run_local_queue.sh   # extensions
#
# Per-task env can be injected via the queue syntax "TASK[:KEY=VAL,...]", e.g.
#   QUEUE="2:SEED=43 4:SEED=43" bash run_local_queue.sh
#
# Checkpoints (SAVE_CKPT=True) make every arm extendable later: rerunning a
# task with a larger TRAIN_EPOCHS resumes from its newest epoch checkpoint.
set -uo pipefail

VAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-$VAE_DIR/Open-Sora}"
export NERSEMBLE_BASE="${NERSEMBLE_BASE:-/home/coder/nersemble-data/processed/2view/}"
export WAN_PRETRAINED="${WAN_PRETRAINED:-$VAE_DIR/paper/audit/zeroshot/Wan2.1_VAE.pth}"
export SAVE_CKPT="${SAVE_CKPT:-True}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-170}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Training venv (accelerate/torch/wandb/lpips).
VENV_BIN="${VENV_BIN:-/home/coder/venvs/inpaint-test/bin}"
export PATH="$VENV_BIN:$PATH"

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "ERROR: wandb not configured (WANDB_API_KEY or ~/.netrc)." >&2
  exit 1
fi

# Default queue, tiered (see paper/rerun_plan_2026-10-02.md):
#   Tier 0 (minutes):  50 true zero-shot TC=on @128, 7 E1z TC=off @128
#   Tier 1 (core):     2 E1b, 3 E1c, 4 E1d, 8 E11a, 9 E11b, 36 combo, 52 E_best
#   Tier 2 (refs):     1 E1a, 24 E4h, 28 E4i
# Warm-started arms (4, 8, 9, 24, 28, 36) auto-find the newest local E1b
# checkpoint, so task 2 MUST come before them.
QUEUE="${QUEUE:-50 7 2 3 4 8 9 36 52 1 24 28}"

LOG_DIR="$OPEN_SORA_ROOT/local_logs"
mkdir -p "$LOG_DIR"

for spec in $QUEUE; do
  task="${spec%%:*}"
  extra="${spec#*:}"; [[ "$extra" == "$spec" ]] && extra=""
  log="$LOG_DIR/task${task}_$(date +%Y%m%d_%H%M%S).log"
  echo "=============================================================="
  echo "[queue] TASK=$task ${extra:+($extra)} epochs=$TRAIN_EPOCHS -> $log"
  echo "=============================================================="
  if [[ -n "$extra" ]]; then
    env ${extra//,/ } TASK="$task" bash "$VAE_DIR/run_paper_sweep.sh" 2>&1 | tee "$log"
  else
    TASK="$task" bash "$VAE_DIR/run_paper_sweep.sh" 2>&1 | tee "$log"
  fi
  rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then
    echo "[queue] TASK=$task FAILED (rc=$rc). Stopping queue -- later arms may" \
         "depend on this one (warm starts). Fix and rerun; finished arms are" \
         "skipped via their .DONE markers." >&2
    exit "$rc"
  fi
  echo "[queue] TASK=$task done."
done
echo "[queue] all tasks finished."
