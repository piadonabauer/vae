#!/bin/bash
# Post-controls chain (armed 2026-10-06): E5b refill + scoped seed-43 repeats.
#   1. E5b (task 5, unfreeze-encoder trainability arm, @170, seed 42)
#   2. E1b_s43 (task 2, SEED=43, @170)            - per-view reference repeat
#   3. E1d_s43 (task 4, SEED=43, @170)            - warm from E1b_s43 (resolved below)
#   4. combo_s43 (task 36, SEED=43, @300)         - protects the combo-vs-E_best claim
# Purpose: seed std for the pivotal arms; full protocol parity with seed 42.
set -uo pipefail
VAE=/home/coder/vae
OUT=$VAE/Open-Sora/outputs
LOG=$VAE/Open-Sora/local_logs
mkdir -p "$LOG"

# 1. E5b @170 (default budget)
TRAIN_EPOCHS=170 QUEUE="5" bash "$VAE/run_local_queue.sh" >> "$LOG/queue_m5.log" 2>&1 || exit 1

# 2. E1b seed 43 @170
TRAIN_EPOCHS=170 QUEUE="2:SEED=43" bash "$VAE/run_local_queue.sh" >> "$LOG/queue_m5.log" 2>&1 || exit 1

# 3. E1d seed 43 @170, warm from the newest E1b_s43 checkpoint
CKPT=""
best_ep=-1
for d in "$OUT"/paper_E1b_perview_tcT_s43__job*/; do
  [[ -d "$d" ]] || continue
  for c in "$d"epoch*/; do
    ep=$(basename "$c" | grep -oP 'epoch\K[0-9]+' || echo -1)
    if (( ep > best_ep )); then best_ep=$ep; CKPT="${c%/}"; fi
  done
done
if [[ -z "$CKPT" ]]; then
  echo "[seed43-chain] FATAL: no E1b_s43 checkpoint found" >> "$LOG/queue_m5.log"
  exit 1
fi
echo "[seed43-chain] E1d_s43 warm start: $CKPT" >> "$LOG/queue_m5.log"
TRAIN_EPOCHS=170 QUEUE="4:SEED=43,INIT_CKPT=$CKPT" bash "$VAE/run_local_queue.sh" >> "$LOG/queue_m5.log" 2>&1 || exit 1

# 4. combo seed 43 @300 (from scratch, INIT_CKPT=none is task 36's default)
TRAIN_EPOCHS=300 QUEUE="36:SEED=43" bash "$VAE/run_local_queue.sh" >> "$LOG/queue_m5.log" 2>&1 || exit 1

echo "[seed43-chain] all done" >> "$LOG/queue_m5.log"
