#!/usr/bin/env bash
# Temporal-compression QUALITY sweep — 8 parallel GPU jobs.
#
# Tests the 6 new ideas for improving tc=True reconstruction quality.
#
# Submit:  sbatch run_tc_quality_sweep.sh
# Monitor: squeue -u $USER
#
# Tasks — all: tc=True, effective_batch=64, ~1000 optimizer update steps
# ─────────────────────────────────────────────────────────────────────────────
#   1  baseline         tc=True, all new flags off  (control)
#   2  idea1_noncausal  use_noncausal_decode=True
#   3  idea2_refpad     use_temporal_reflection_pad=True
#   4  idea3_sidechan   use_temporal_side_channel=True
#   5  idea4_decoattn   use_noncausal_decode + use_decoder_temporal_attention
#   6  idea6_gru_cache  use_learned_cache_update=True
#   7  combo_1_2        use_noncausal_decode + use_temporal_reflection_pad
#   8  combo_1_2_4      use_noncausal_decode + reflection pad + temporal attn
#
# Idea 5 (teacher distillation): requires a tc=False teacher checkpoint;
# uncomment the EXPERIMENTS entry below once you have a path.
#
# ── Batch-size and fairness strategy ────────────────────────────────────────
# Different ideas have different memory footprints (e.g. non-causal decode
# must hold all T' frames at once).  Each job auto-detects the largest batch
# that fits by running a 1-epoch probe, then adjusts gradient-accumulation so
# that every optimizer update ALWAYS processes exactly EFFECTIVE_BATCH=64
# samples:
#
#   actual_batch  →  accumulation_steps  (batch × accum = 64)
#      32         →  2
#      16         →  4
#       8         →  8
#       4         →  16
#       2         →  32
#       1         →  64
#
# Because each update always consumes 64 samples, update_steps_per_epoch ≈ 6
# (390 train files ÷ 64 ≈ 6) for ALL runs.  Setting --epochs=170 therefore
# gives ≈ 1020 optimizer updates in every run, regardless of which batch size
# the GPU accepted.  The runs are directly comparable in WandB.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=tc_quality_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-8%8

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
DRY_RUN="${DRY_RUN:-0}"

# Effective batch kept constant; only the actual batch / accumulation ratio changes.
EFFECTIVE_BATCH=64
BATCH_CANDIDATES=(32 16 8 4 2 1)

# ── epochs for ~1000 optimizer updates at EFFECTIVE_BATCH=64 ─────────────────
# 390 train files ÷ 64 effective ≈ 6 updates/epoch  →  170 epochs ≈ 1020 steps
TRAIN_EPOCHS=170

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
fi

cd "$OPEN_SORA_ROOT"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

# ── Batch-size probe ──────────────────────────────────────────────────────────
# Runs one epoch silently; on success prints the batch size and returns 0.
# Cleans up the probe output directory afterwards.
probe_batch_size() {
  local port="$1"; shift
  local extra_args=("$@")  # per-experiment model flags

  for b in "${BATCH_CANDIDATES[@]}"; do
    local probe_name="_probe_${wandb_name}_b${b}"
    echo "  → probing batch_size=${b} ..."

    if accelerate launch \
        --num_processes 1 --num_machines 1 \
        --dynamo_backend no \
        --mixed_precision bf16 \
        --main_process_port "$port" \
        "$train_file" "$my_config" \
        --experiment_name  "$probe_name" \
        --wandb            False \
        --data_preset      all_people_one_expression \
        --model.temporal_compression True \
        --batch_size       "$b" \
        --accumulation_steps 1 \
        --epochs           1 \
        --log_every        999999 \
        --log_schedule_steps "[]" \
        --full_eval_every  999999 \
        --fixed_seq_eval_every_epochs 0 \
        --FAST_MODE        False \
        --optimization     False \
        --profile_timing   False \
        --profile_step     False \
        --profile_memory_live False \
        --train_psnr_guard False \
        "${extra_args[@]}" \
        > "/tmp/${probe_name}.log" 2>&1
    then
      rm -f "/tmp/${probe_name}.log"
      rm -rf "${OPEN_SORA_ROOT}/outputs/${probe_name}" 2>/dev/null || true
      echo "$b"   # ← caller reads this as the found batch size
      return 0
    else
      echo "    batch_size=${b} failed (OOM or other error); trying smaller..."
      rm -f "/tmp/${probe_name}.log"
      rm -rf "${OPEN_SORA_ROOT}/outputs/${probe_name}" 2>/dev/null || true
      # bump port to avoid port-reuse issues between probe launches
      port=$(( port + 1 ))
    fi
  done

  echo "ERROR: all batch sizes failed for ${wandb_name}." >&2
  return 1
}

# ── Common overrides shared by every experiment ───────────────────────────────
COMMON_OVERRIDES=(
  --data_preset             all_people_one_expression
  --model.temporal_compression True
  --wandb                   True
  --optimization            False
  --FAST_MODE               False
  --profile_timing          False
  --profile_step            False
  --profile_memory_live     False
  --epochs                  "$TRAIN_EPOCHS"
  # Log dense early then every 50 update steps
  --log_every               50
  --log_schedule_steps      "[5,10,20,50,100,200]"
  # Full val eval 4× during the run (~every 250 optimizer updates)
  --full_eval_every         250
  --fixed_seq_eval_every_epochs 0
)

# ── Experiment table ──────────────────────────────────────────────────────────
# Format: "wandb_base_name|extra_model_overrides"
EXPERIMENTS=(
  "tc_q_baseline|\
"
  "tc_q_idea1_noncausal|\
--model.use_noncausal_decode True"
  "tc_q_idea2_refpad|\
--model.use_temporal_reflection_pad True"
  "tc_q_idea3_sidechan|\
--model.use_temporal_side_channel True --model.side_channel_dim 4"
  "tc_q_idea4_decoattn|\
--model.use_noncausal_decode True \
--model.use_decoder_temporal_attention True"
  "tc_q_idea6_gru_cache|\
--model.use_learned_cache_update True"
  "tc_q_combo_1_2|\
--model.use_noncausal_decode True \
--model.use_temporal_reflection_pad True"
  "tc_q_combo_1_2_4|\
--model.use_noncausal_decode True \
--model.use_temporal_reflection_pad True \
--model.use_decoder_temporal_attention True"
)

# Idea 5 (teacher distillation): uncomment + set path when you have a ckpt.
# "tc_q_idea5_distill|\
# --distill_teacher_ckpt /path/to/tc_false_checkpoint.pt \
# --distill_weight 1.0"

# ── Task dispatch ─────────────────────────────────────────────────────────────
n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r wandb_name overrides <<< "${EXPERIMENTS[$idx]}"
read -ra override_args <<< "$overrides"

# ── Port allocation ───────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  BASE_PORT=$((20000 + (SLURM_JOB_ID % 20000) + (SLURM_ARRAY_TASK_ID % 1000)))
else
  BASE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE="${WORLD_SIZE:-1}" RANK="${RANK:-0}" LOCAL_RANK="${LOCAL_RANK:-0}"

DYNAMO_BACKEND=$(python3 - "$my_config" <<'PY'
import pathlib, re, sys
m = re.search(r'^\s*dynamo_backend\s*=\s*["\']([^"\']+)["\']\s*$', pathlib.Path(sys.argv[1]).read_text(), re.MULTILINE)
print(m.group(1) if m else "no")
PY
)

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_quality_sweep task ${SLURM_ARRAY_TASK_ID:-?}/$n_exp — ${wandb_name}"
echo "  base port : $BASE_PORT  dynamo: $DYNAMO_BACKEND"
echo "  model overrides: ${override_args[*]:-<none>}"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

# ── Find the largest batch that fits ─────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
  found_batch=32
  echo "[dry-run] skipping probe, assuming batch=${found_batch}"
else
  echo "Probing batch sizes: ${BATCH_CANDIDATES[*]} ..."
  found_batch=$(probe_batch_size "$BASE_PORT" "${override_args[@]}")
  echo "✓ Using batch_size=${found_batch}"
fi

# Compute accumulation_steps so every update sees exactly EFFECTIVE_BATCH samples.
accum=$(( EFFECTIVE_BATCH / found_batch ))
(( accum < 1 )) && accum=1

# Embed the actual batch in the WandB run name so it's visible in the dashboard.
wandb_run_name="${wandb_name}_b${found_batch}"
experiment_name="${wandb_run_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  experiment_name="${wandb_run_name}__job${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
fi
export WANDB_NAME="$wandb_run_name"

echo "────────────────────────────────────────────────────────────────────"
echo "  actual batch : ${found_batch}"
echo "  accumulation : ${accum}  (effective batch = $((found_batch * accum)))"
echo "  optimizer updates ≈ $(( TRAIN_EPOCHS * ( 390 / found_batch / accum ) ))"
echo "  experiment   : ${experiment_name}"
echo "────────────────────────────────────────────────────────────────────"

# ── Launch real training ──────────────────────────────────────────────────────
MASTER_PORT=$(( BASE_PORT + 100 ))  # offset from probe ports
export MASTER_PORT

if [[ "$DRY_RUN" == "1" ]]; then
  echo accelerate launch --num_processes 1 --num_machines 1 \
    --dynamo_backend "$DYNAMO_BACKEND" --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name "$experiment_name" --wandb_expr_name "$wandb_run_name" \
    --batch_size "$found_batch" --accumulation_steps "$accum" \
    "${COMMON_OVERRIDES[@]}" "${override_args[@]}"
  exit 0
fi

max_retries=3; attempt=1
while (( attempt <= max_retries )); do
  echo "Launch attempt ${attempt}/${max_retries} on port ${MASTER_PORT}"
  accelerate launch \
    --num_processes 1 --num_machines 1 \
    --dynamo_backend "$DYNAMO_BACKEND" \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    "$train_file" "$my_config" \
    --experiment_name  "$experiment_name" \
    --wandb_expr_name  "$wandb_run_name" \
    --batch_size       "$found_batch" \
    --accumulation_steps "$accum" \
    "${COMMON_OVERRIDES[@]}" \
    "${override_args[@]}" \
  && exit 0
  rc=$?
  (( attempt == max_retries )) && { echo "Failed after $max_retries attempts (rc=$rc)."; exit "$rc"; }
  MASTER_PORT=$(( MASTER_PORT + 1 )); export MASTER_PORT
  echo "Retrying with port $MASTER_PORT..."
  attempt=$(( attempt + 1 ))
done
