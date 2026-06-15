#!/usr/bin/env bash
#SBATCH --job-name=4v_autobatch
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/outputs/slurm/%x_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/outputs/slurm/%x_%j.err

# Train on new 4-view preprocessed data with automatic batch-size selection.
#
# Resolution is set via RESOLUTION env var (default 512):
#   sbatch --export=RESOLUTION=512  run_4v_allpeople_autobatch.sh
#   sbatch --export=RESOLUTION=1024 run_4v_allpeople_autobatch.sh
#   sbatch --export=RESOLUTION=2048 run_4v_allpeople_autobatch.sh
#
# Batch-size probe:
#   Tries batch sizes 32 → 16 → 8 → 4 → 2 (each for 5 optimizer steps).
#   For each attempt, records peak GPU memory and OOM/success to:
#     outputs/<exp>/batch_probe_results.jsonl
#   Picks the largest batch that fits; accumulation_steps = 32 // batch_size
#   to keep effective batch = 32.
#
# Full run: no max_steps (runs until 20 h SLURM wall time).

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
RESOLUTION="${RESOLUTION:-512}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
  # The venv was created at the old path; prepend the real bin dir so
  # 'accelerate' resolves to the correct snth binary (not ~/.local/bin).
  SNTH_BIN="/project/6101839/piado/vae/snth/bin"
  export PATH="$SNTH_BIN:$PATH"
  export VIRTUAL_ENV="/project/6101839/piado/vae/snth"
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
fi

# Reduce allocator fragmentation (recommended by OOM messages)
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Force Python stdout/stderr unbuffered so log files update in real-time
export PYTHONUNBUFFERED=1

cd "$OPEN_SORA_ROOT"
mkdir -p slurm_logs

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

exp_name="4v_${RESOLUTION}px_allpeople_one_expr_lr5e4_compile"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exp_name="${exp_name}__job${SLURM_JOB_ID}"
fi
probe_log="${OPEN_SORA_ROOT}/outputs/${exp_name}/batch_probe_results.jsonl"
mkdir -p "${OPEN_SORA_ROOT}/outputs/${exp_name}"

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000) + (SLURM_ARRAY_TASK_ID % 1000)))
else
  MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
fi
export MASTER_PORT
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0

# Rename the SLURM job to include resolution (can't use shell vars in #SBATCH headers)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update jobid="${SLURM_JOB_ID}" name="4v_${RESOLUTION}px_autobatch" 2>/dev/null || true
fi

echo "=== 4-view ${RESOLUTION}px all_people_one_expression (auto batch) ==="
echo "experiment_name: $exp_name"
echo "probe log: $probe_log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then nvidia-smi || true; fi

# ── Common training args (probe + full run share these) ──────────────────────
# Compile mode: "default" (inductor, no CUDA graphs) is compatible with gradient
# checkpointing. "reduce-overhead" (CUDA graphs) fragments the allocator when
# checkpointing is active and causes OOM even at batch=1 for 512px+.
COMPILE_MODE="reduce-overhead"

common_args=(
  "$train_file" "$my_config"
  --experiment_name "$exp_name"
  --wandb_expr_name  "4v_${RESOLUTION}px_allpeople_one_expr_lr5e4_compile"
  --nersemble_processed_base /datasets/lindell-proj/neumayr/nersemble_v2/processed/4-frames
  --bucket_config "{'${RESOLUTION}px_ar1:1':{9:(1.0,1)}}"
  --data_preset all_people_one_expression
  --learning_rate 5e-4
  --lr_scheduler.warmup_steps 0
  --lr_scheduler.use_exponential_decay False
  --discriminator_choice none
  --optimization True
  --optimization_compile_mode "$COMPILE_MODE"
  --epochs 100000
  --eval_at_steps "[10,100,500,1000,1500]"
  --eval_every 500
  --full_eval_every 500
  --log_every 500
  --save_ckpt True
  --fixed_seq_eval_every_epochs 0
  --log_memory True
  --num_reconstruction_vis_samples 1
)

# ── Batch-size probe ─────────────────────────────────────────────────────────
# Runs 5 optimizer steps.  On OOM, parses peak allocated memory from stderr.
# Writes one JSON line per attempt to $probe_log.

PROBE_STEPS=5
BATCH_SIZES=(32 16 8 4 2 1)
selected_batch=""
selected_accum=""

probe_port=$MASTER_PORT

run_probe() {
  local bs="$1"
  local accum=$(( 32 / bs ))
  local ts; ts=$(date -Iseconds)

  echo ""
  echo "──────────────────────────────────────────────"
  echo " PROBE  batch_size=${bs}  accum=${accum}  (${PROBE_STEPS} steps)"
  echo "──────────────────────────────────────────────"

  # Reset GPU memory stats before probe
  python3 -c "import torch; torch.cuda.reset_peak_memory_stats()" 2>/dev/null || true

  probe_out=$(mktemp)
  probe_port=$((probe_port + 1))

  set +e
  accelerate launch \
    --num_processes 1 --num_machines 1 \
    --dynamo_backend no --mixed_precision bf16 \
    --main_process_port "$probe_port" \
    "${common_args[@]}" \
    --optimization False \
    --batch_size "$bs" \
    --accumulation_steps "$accum" \
    --max_steps "$PROBE_STEPS" \
    --wandb False \
    2>&1 | tee "$probe_out"
  local rc=$?
  set -e

  # Parse peak GPU memory from output (success: log_memory line; OOM: error message)
  local peak_gb="null"
  local oom_line
  oom_line=$(grep -oP 'of the allocated memory [0-9.]+ GiB is allocated by PyTorch' "$probe_out" | tail -1 || true)
  if [[ -n "$oom_line" ]]; then
    peak_gb=$(echo "$oom_line" | grep -oP '[0-9.]+' | head -1)
  else
    local mem_line
    mem_line=$(grep -oP 'mem_alloc=[0-9.]+GB' "$probe_out" | tail -1 || true)
    if [[ -n "$mem_line" ]]; then
      peak_gb=$(echo "$mem_line" | grep -oP '[0-9.]+')
    fi
  fi

  local status="oom"
  if [[ $rc -eq 0 ]]; then status="success"; fi

  # Write to probe log
  echo "{\"timestamp\":\"${ts}\",\"resolution\":${RESOLUTION},\"batch_size\":${bs},\"accum\":${accum},\"status\":\"${status}\",\"peak_allocated_gb\":${peak_gb},\"exit_code\":${rc}}" \
    >> "$probe_log"

  echo " → status=${status}  peak_gpu=${peak_gb}GB  (exit ${rc})"
  rm -f "$probe_out"

  return $rc
}

echo ""
echo "=== Batch-size probe (${PROBE_STEPS} steps each) ==="
echo "Results will be saved to: $probe_log"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] Would probe batch sizes: ${BATCH_SIZES[*]}"
  echo "[dry-run] Then run full training (no max_steps, 20 h wall time)"
  exit 0
fi

for bs in "${BATCH_SIZES[@]}"; do
  if run_probe "$bs"; then
    selected_batch=$bs
    selected_accum=$(( 32 / bs ))
    echo ""
    echo "✓ Selected batch_size=${selected_batch}  accumulation_steps=${selected_accum}  (effective batch=32)"
    break
  fi
done

if [[ -z "$selected_batch" ]]; then
  echo "ERROR: All batch sizes OOM'd. Cannot train at ${RESOLUTION}px on this GPU."
  echo "Probe results: $probe_log"
  exit 1
fi

# ── Full training run (no max_steps → runs until 20 h SLURM wall time) ───────
echo ""
echo "=== Full training: batch=${selected_batch}  accum=${selected_accum}  res=${RESOLUTION}px ==="

MASTER_PORT=$((probe_port + 1))

_launch() {
  local port="$1"
  accelerate launch \
    --num_processes 1 --num_machines 1 \
    --dynamo_backend no --mixed_precision bf16 \
    --main_process_port "$port" \
    "${common_args[@]}" \
    --optimization False \
    --batch_size "$selected_batch" \
    --accumulation_steps "$selected_accum"
}

max_port_retries=3
attempt=1
while (( attempt <= max_port_retries )); do
  echo "Launch attempt ${attempt}/${max_port_retries} on port ${MASTER_PORT}"
  set +e
  _launch "$MASTER_PORT"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "Training completed successfully (rc=0)."
    exit 0
  fi
  if (( attempt == max_port_retries )); then
    echo "Launch failed after ${max_port_retries} attempts (rc=${rc})."
    exit "$rc"
  fi
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "Retrying with port ${MASTER_PORT}..."
  attempt=$((attempt + 1))
done
