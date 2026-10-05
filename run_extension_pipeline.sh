#!/bin/bash
# Extension pipeline: M2 (E11b/combo/E_best -> 300 ep) then M3 (E1d/E11a/E1b -> 300 ep).
# Idempotent across reboots: finished tasks are skipped via .DONE markers, and the
# M3 marker-clearing happens exactly once (guarded by .M3_STARTED sentinel).
set -u
OUT=/home/coder/vae/Open-Sora/outputs
LOG=/home/coder/vae/Open-Sora/local_logs
mkdir -p "$LOG"

# M2: markers were cleared when M2 first launched; on reboot, finished arms skip.
TRAIN_EPOCHS=300 QUEUE="9 36 52" bash /home/coder/vae/run_local_queue.sh >> "$LOG/queue_m2.log" 2>&1 || exit 1

# M3: all capacity-table rows to 300 ep (16-ch, 32-ch), plus E1b reference insurance.
if [ ! -f "$OUT/.M3_STARTED" ]; then
  rm -f "$OUT/paper_E1d_fused_tcT.DONE" \
        "$OUT/paper_E11a_fused_tcT_widen32.DONE" \
        "$OUT/paper_E1b_perview_tcT.DONE"
  touch "$OUT/.M3_STARTED"
fi
TRAIN_EPOCHS=300 QUEUE="4 8 2" bash /home/coder/vae/run_local_queue.sh >> "$LOG/queue_m3.log" 2>&1
