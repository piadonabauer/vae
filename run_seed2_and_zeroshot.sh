#!/usr/bin/env bash
# Seed-2 repeats for headline arms + Wan zero-shot floor evals.
# Seed jobs: SEED=43, SAVE_CKPT=True (need weights for later).
# Zero-shot: TASK=7 at 128; separate 256 via bucket override if supported.
set -euo pipefail
ROOT="${ROOT:-/home/piado/projects/aip-lindell/piado/vae}"
OUT="${ROOT}/Open-Sora/outputs"
SCRIPT="${ROOT}/run_paper_sweep.sh"
E1B_CKPT="${E1B_CKPT:-${OUT}/paper_E1b_perview_tcT__job5556206_t2/epoch169-global_step1083}"
LOG="${ROOT}/paper/seed2_jobs.txt"
: > "$LOG"

submit() {
  local label="$1"; shift
  local jid
  jid=$(sbatch --parsable "$@")
  echo "${jid}: ${label}" | tee -a "$LOG"
}

# --- Zero-shot floor @ 128 (existing E1z) ---
submit "E1z_zeroshot_128" \
  --partition=gpubase_l40s_b3 --time=0-03:00:00 \
  --export=ALL,TASK=7,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 "$SCRIPT"

# --- Zero-shot floor @ 256 ---
submit "E1z_zeroshot_256" \
  --partition=gpubase_l40s_b3 --time=0-03:00:00 \
  --export=ALL,TASK=49,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 "$SCRIPT"

# --- Seed-2 headline arms ---
# E1c: fused TC off, no warmstart
submit "E1c_seed43" \
  --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=3,OVERFIT=0,CHAIN_LEFT=0,SEED=43,SAVE_CKPT=True,INIT_CKPT=none \
  --array=1 "$SCRIPT"

# E1d: fused TC on, warmstart E1b
submit "E1d_seed43" \
  --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=4,OVERFIT=0,CHAIN_LEFT=0,SEED=43,SAVE_CKPT=True,INIT_CKPT="${E1B_CKPT}" \
  --array=1 "$SCRIPT"

# E11a: widen 32 (warmstart E1b; widen init handles 16→32)
submit "E11a_seed43" \
  --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=8,OVERFIT=0,CHAIN_LEFT=0,SEED=43,SAVE_CKPT=True,INIT_CKPT="${E1B_CKPT}" \
  --array=1 "$SCRIPT"

# E_combo: diff + 32ch (same warmstart convention as clean wave)
submit "E_combo_seed43" \
  --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=36,OVERFIT=0,CHAIN_LEFT=0,SEED=43,SAVE_CKPT=True,INIT_CKPT="${E1B_CKPT}" \
  --array=1 "$SCRIPT"

echo "Wrote $LOG"
cat "$LOG"
squeue -u piado -n paper_sweep -o '%i %P %T %l %R' | head -20
