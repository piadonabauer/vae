#!/usr/bin/env bash
# Clean paper retrain under ONE eval pipeline (current code → GT XView ≈ 0.978).
# Always submits --array=1 (never the sweep's default 1-26).
#
# Already finished this round: E1b (TASK=2), E3d (TASK=16). E1d (TASK=4) may be running.
# Skips: E8b (broken), E7_discstack (dropped), E1z (eval-only), E9 (optional later).
#
# Usage:
#   bash run_paper_retrain_clean.sh           # prepare + submit
#   DRY_RUN=1 bash run_paper_retrain_clean.sh

set -euo pipefail

ROOT="${ROOT:-/home/piado/projects/aip-lindell/piado/vae}"
OUT="${ROOT}/Open-Sora/outputs"
SCRIPT="${ROOT}/run_paper_sweep.sh"
DRY_RUN="${DRY_RUN:-0}"
# Cap concurrent new jobs (Slurm fairshare); warmstart arms use E1b below.
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Fresh E1b warmstart (protocol for fused+TC arms)
E1B_CKPT="${E1B_CKPT:-${OUT}/paper_E1b_perview_tcT__job5556206_t2/epoch169-global_step1083}"
if [[ ! -f "${E1B_CKPT}/model/model-00001.safetensors" ]]; then
  echo "ERROR: E1b warmstart missing: $E1B_CKPT"
  exit 1
fi

# TASK list: table-critical arms only (name for logs)
# Format: TASK:needs_warmstart(0/1):run_name
ARMS=(
  "1:0:paper_E1a_perview_tcF"
  "3:0:paper_E1c_fused_tcF"
  "10:0:paper_E0_perview_ceiling"
  "11:0:paper_E2b_fused_tcF_self_attn"
  "12:0:paper_E2c_fused_tcF_conv3d"
  "13:0:paper_E2d_fused_tcF_conv4d"
  "27:1:paper_E2e_fused_tcT_self_attn"
  "14:0:paper_E3a_view_emb_noLora"
  "15:0:paper_E3c_view_emb_plus_lora"
  "17:0:paper_E3e_full_dec_finetune"
  "8:1:paper_E11a_fused_tcT_widen32"
  "9:1:paper_E11b_fused_tcT_widen64"
  "18:1:paper_E4b_noncausal_decode"
  "24:1:paper_E4h_temporal_diff_loss"
  "28:1:paper_E4i_diff_loss_plus_cache"
  "36:1:paper_E_combo_diffLoss_widen32"
  "5:1:paper_E5b_fused_tcT_unfreeze_enc"
  "6:1:paper_E5c_fused_tcT_unfreeze_all"
  "29:0:paper_E6b_4view_tcF"
  "30:1:paper_E6c_4view_tcT"
  "31:0:paper_E6d_8view_tcF"
  "32:1:paper_E6e_8view_tcT"
  "43:0:paper_E10_rank8"
  "47:0:paper_E10_rank16"
  "44:0:paper_E10_rank128"
  "48:0:paper_E10_rank64"
  "40:1:paper_E7_perc0p5"
  "41:1:paper_E7_perc3p0"
  "42:1:paper_E7_kl1e7"
)

echo "=== clean retrain prepare ==="
echo "E1b warmstart: $E1B_CKPT"

prepare_arm() {
  local name="$1"
  # Keep DONE for arms we intentionally skip re-running this wave
  rm -f "${OUT}/${name}.DONE"
  # Rename any leftover epoch* dirs (empty or old) so resume won't pick them.
  # Keep freshly completed E1b/E3d/E1d job dirs that still have .safetensors.
  local d
  for d in "${OUT}/${name}__job"*/; do
    [[ -d "$d" ]] || continue
    local ep
    for ep in "$d"epoch*; do
      [[ -e "$ep" ]] || continue
      local base; base=$(basename "$ep")
      [[ "$base" == purged_* ]] && continue
      if [[ -f "$ep/model/model-00001.safetensors" ]]; then
        # Fresh weights from this retrain wave — keep
        continue
      fi
      mv "$ep" "$d/purged_${base}"
      echo "  purged empty/old $d$base"
    done
  done
}

# Skip prepare for E1b / E3d (done). Prepare E1d only if not running with weights yet.
SKIP_PREP=(paper_E1b_perview_tcT paper_E3d_no_emb_no_lora)

for spec in "${ARMS[@]}"; do
  IFS=':' read -r task warm name <<< "$spec"
  skip=0
  for s in "${SKIP_PREP[@]}"; do [[ "$name" == "$s" ]] && skip=1; done
  if (( skip )); then continue; fi
  echo "prepare $name (TASK=$task warm=$warm)"
  prepare_arm "$name"
done

# E1d: only purge empties; don't remove DONE if job still running
if [[ -d "${OUT}/paper_E1d_fused_tcT__job5562110_t4" ]]; then
  echo "E1d job5562110 present — leave running; only purge empties in older dirs"
  for d in "${OUT}/paper_E1d_fused_tcT__job"*/; do
    [[ "$d" == *job5562110* ]] && continue
    for ep in "$d"epoch*; do
      [[ -e "$ep" ]] || continue
      base=$(basename "$ep"); [[ "$base" == purged_* ]] && continue
      [[ -f "$ep/model/model-00001.safetensors" ]] && continue
      mv "$ep" "$d/purged_${base}"
    done
  done
fi

echo "=== submit (${#ARMS[@]} arms, max ${MAX_PARALLEL} via sbatch) ==="
SUBMITTED=()
for spec in "${ARMS[@]}"; do
  IFS=':' read -r task warm name <<< "$spec"
  export_args="ALL,TASK=${task},OVERFIT=0,CHAIN_LEFT=0"
  if [[ "$warm" == "1" ]]; then
    export_args="${export_args},INIT_CKPT=${E1B_CKPT}"
  else
    export_args="${export_args},INIT_CKPT=none"
  fi
  echo "sbatch --array=1 TASK=$task $name warm=$warm"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  jid=$(sbatch --parsable --array=1 --export="$export_args" "$SCRIPT")
  SUBMITTED+=("$jid:$task:$name")
  echo "  -> job $jid"
done

printf '%s\n' "${SUBMITTED[@]}" > "${ROOT}/paper/clean_retrain_jobs.txt"
echo "Wrote ${ROOT}/paper/clean_retrain_jobs.txt (${#SUBMITTED[@]} jobs)"
echo "Monitor: squeue -u \$USER -n paper_sweep"
echo "ETA: ~${#ARMS[@]} arms × ~4h / ${MAX_PARALLEL} ≈ $(( (${#ARMS[@]} + MAX_PARALLEL - 1) / MAX_PARALLEL * 4 ))h if fairshare allows ${MAX_PARALLEL} at a time (often slower)."
