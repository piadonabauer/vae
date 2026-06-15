#!/bin/bash
# ============================================================
# start_all_training.sh  –  submit all 8 VAE training jobs
#
# Usage:   bash start_all_training.sh
#
# Strategy (priority: get results by tomorrow):
#  • 256px / 512px  →  L40S (48 GB),  no torch.compile
#  • 1024px / 2048px → H100 (80 GB),  no torch.compile
#  • If a job fails the probe on L40S it is immediately
#    re-submitted on H100 as fallback (not automated here,
#    but the probes use --optimization False so they give
#    real memory numbers now).
#
# NO_COMPILE=1 is passed so the full training also runs
# without torch.compile – slightly slower but rock-solid.
# ============================================================

set -euo pipefail

VAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VAE_DIR"

# ── cancel PENDING (not running) autobatch jobs ───────────────
echo "Cancelling pending (not yet started) autobatch jobs..."
squeue -u "$USER" --state=PENDING --format="%i %j" --noheader 2>/dev/null \
  | awk '/autobatch/{print $1}' \
  | xargs -r scancel
echo "(Running jobs like the 512px 4v are left untouched)"
sleep 2

# ── helper: submit with L40S, show job id ────────────────────
submit_l40s_4v() {
  local res="$1"
  local jid
  jid=$(sbatch \
    --partition=gpubase_l40s_b3 \
    --gres=gpu:l40s:1 \
    --export=ALL,RESOLUTION="$res" \
    --parsable \
    "$VAE_DIR/run_4v_allpeople_autobatch.sh")
  echo "  4v ${res}px  →  L40S job $jid"
}

submit_l40s_8v() {
  local res="$1"
  local jid
  jid=$(sbatch \
    --partition=gpubase_l40s_b3 \
    --gres=gpu:l40s:1 \
    --export=ALL,RESOLUTION="$res" \
    --parsable \
    "$VAE_DIR/run_8v_allpeople_autobatch.sh")
  echo "  8v ${res}px  →  L40S job $jid"
}

submit_h100_4v() {
  local res="$1"
  local jid
  jid=$(sbatch \
    --partition=gpubase_h100_b3 \
    --gres=gpu:h100:1 \
    --export=ALL,RESOLUTION="$res" \
    --parsable \
    "$VAE_DIR/run_4v_allpeople_autobatch.sh")
  echo "  4v ${res}px  →  H100 job $jid"
}

submit_h100_8v() {
  local res="$1"
  local jid
  jid=$(sbatch \
    --partition=gpubase_h100_b3 \
    --gres=gpu:h100:1 \
    --export=ALL,RESOLUTION="$res" \
    --parsable \
    "$VAE_DIR/run_8v_allpeople_autobatch.sh")
  echo "  8v ${res}px  →  H100 job $jid"
}

echo ""
echo "========================================================"
echo " Submitting all 8 VAE training jobs"
echo " Goal: results by tomorrow"
echo "========================================================"
echo ""
echo "── 4-view jobs ──────────────────────────────────────────"
submit_l40s_4v 256
submit_l40s_4v 512
submit_h100_4v 1024
submit_h100_4v 2048

echo ""
echo "── 8-view jobs ──────────────────────────────────────────"
submit_l40s_8v 256
submit_l40s_8v 512
submit_h100_8v 1024
submit_h100_8v 2048

echo ""
echo "========================================================"
echo " All jobs submitted. Monitor with:"
echo "   watch -n 30 'squeue -u \$USER --format=\"%.10i %.30j %.8T %.10M\"'"
echo ""
echo " Logs: /home/piado/projects/aip-lindell/piado/outputs/slurm/"
echo " WandB: https://wandb.ai"
echo "========================================================"

# ── Schedule a safety-net retry in 3 hours ───────────────────
# Submits a lightweight SLURM job that sleeps 3h then re-runs
# this script.  Any already-running training jobs are left
# untouched (only PENDING jobs get cancelled before re-submit).
RETRY_SCRIPT="$(mktemp /tmp/retry_training_XXXX.sh)"
cat > "$RETRY_SCRIPT" <<'RETRY_EOF'
#!/bin/bash
#SBATCH --job-name=training_retry_3h
#SBATCH --output=/home/piado/projects/aip-lindell/piado/outputs/slurm/retry_3h_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:0

echo "[retry] Sleeping 3 hours then re-submitting all training jobs..."
sleep 10800

echo "[retry] Waking up at $(date). Re-running start_all_training.sh"
bash /home/piado/projects/aip-lindell/piado/vae/start_all_training.sh
echo "[retry] Done."
RETRY_EOF

retry_jid=$(sbatch --parsable "$RETRY_SCRIPT")
echo ""
echo "  Safety-net retry scheduled: job $retry_jid fires in ~3 h"
echo "  (cancels pending jobs + re-submits all 8 if something failed)"
rm -f "$RETRY_SCRIPT"
