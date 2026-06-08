#!/usr/bin/env bash
#SBATCH --job-name=compile_allpeople
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-07:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-5%5

# torch.compile benchmark on all_people_one_expression — one GPU per compile mode.
#
# Differences from run_compile_benchmark.sh:
#   data_preset  = all_people_one_expression  (was: single_sequence)
#   batch_size   = 32                         (was: 1; 64 OOM'd with CUDA graphs)
#   epochs       = 20   → ~240 steps          (was: 70 epochs = 70 steps)
#   learning_rate = 5e-4, no LR scheduling    (was: config default 1e-4 with decay)
#   results_dir  = outputs/compile_benchmark_allpeople_b32
#
# Data:  all_people_one_expression, EMO-1-shout+laugh, 8-frames/128-res tree
#        ~390 train clips, ~12 steps/epoch at batch 32, drop_last=True
# GPU:   NVIDIA L40S (44 GB). batch 32 safe for all compile modes (CUDA graph
#        pool ~8 GB at batch 32, vs ~30 GB that OOM'd at batch 64).
#
# Submit:  sbatch run_compile_benchmark_allpeople.sh
# Monitor: squeue -u $USER
# Results: outputs/compile_benchmark_allpeople_b32/summary.json
#
# Modes (5 tasks):
#   1  no_compile                  — baseline, torch.compile disabled
#   2  compile_default             — mode="default"
#   3  compile_reduce_overhead     — mode="reduce-overhead"    (CUDA graphs)
#   4  compile_max_autotune        — mode="max-autotune"       (CUDA graphs + full tuning)
#   5  compile_max_autotune_no_cg  — mode="max-autotune-no-cudagraphs"
#
# Metrics per run:
#   compile_load_time_s    — wall time until Step 1 log line
#   step10_avg_time_s      — avg of first 10 steps (includes compile warmup)
#   step50_total_step_s    — step time at global_step 50 (clean post-warmup)
#   gpu_util_warm_*_pct    — GPU SM utilisation during Steps 40–60

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
RESULTS_DIR="${RESULTS_DIR:-${OPEN_SORA_ROOT}/outputs/compile_benchmark_allpeople_b32}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  # shellcheck source=/dev/null
  source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_KERNEL_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/.pytorch_kernels"
fi

cd "$OPEN_SORA_ROOT"
mkdir -p "$RESULTS_DIR"

train_file="${OPEN_SORA_ROOT}/scripts/vae/train.py"
my_config="${OPEN_SORA_ROOT}/${CONFIG}"

# ── Benchmark settings ────────────────────────────────────────────────────────
# all_people_one_expression @ batch 32: ~390 clips → ~12 steps/epoch (drop_last).
# 20 epochs → ~240 optimizer steps. Profile fires at step 50 (~epoch 4-5).
BENCH_EPOCHS=20
BENCH_BATCH=32

# ── Experiment table (1-indexed to match SLURM_ARRAY_TASK_ID) ─────────────────
EXPERIMENTS=(
  "no_compile|--optimization False"
  "compile_default|--optimization True --optimization_compile_mode default"
  "compile_reduce_overhead|--optimization True --optimization_compile_mode reduce-overhead"
  "compile_max_autotune|--optimization True --optimization_compile_mode max-autotune"
  "compile_max_autotune_no_cg|--optimization True --optimization_compile_mode max-autotune-no-cudagraphs"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r exp_name exp_overrides <<< "${EXPERIMENTS[$idx]}"
read -ra extra_args <<< "$exp_overrides"

log_file="${RESULTS_DIR}/${exp_name}.log"
gpu_file="${RESULTS_DIR}/${exp_name}_gpu.csv"
result_file="${RESULTS_DIR}/${exp_name}_result.json"

# ── Write helper Python scripts to a temp dir ─────────────────────────────────
TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# gpu_poller.py — samples nvidia-smi every second
cat > "${TMPDIR_BENCH}/gpu_poller.py" <<'PYEOF'
import subprocess, time, sys

out_file = sys.argv[1]
t0 = time.time()
with open(out_file, 'w') as f:
    f.write("elapsed_s,gpu_util_pct,mem_util_pct\n")
    while True:
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,utilization.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(',')
                if len(parts) >= 2:
                    f.write(f"{time.time()-t0:.3f},{parts[0].strip()},{parts[1].strip()}\n")
                    f.flush()
        except Exception:
            pass
        time.sleep(1)
PYEOF

# timestamper.py — prepends [T+XX.XXXs] to every stdin line
cat > "${TMPDIR_BENCH}/timestamper.py" <<'PYEOF'
import sys, time
t0 = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[T+{time.time()-t0:.3f}] {line}")
    sys.stdout.flush()
PYEOF

# parse_run.py — extracts metrics and writes summary
cat > "${TMPDIR_BENCH}/parse_run.py" <<'PYEOF'
import sys, json, re

exp_name     = sys.argv[1]
log_file     = sys.argv[2]
gpu_file     = sys.argv[3]
result_file  = sys.argv[4]
results_dir  = sys.argv[5]

result = {
    "name":                        exp_name,
    "compile_load_time_s":         None,
    "step10_avg_time_s":           None,
    "step50_total_step_s":         None,
    "gpu_util_warm_avg_pct":       None,
    "gpu_util_warm_min_pct":       None,
    "gpu_util_warm_max_pct":       None,
    "mem_util_warm_avg_pct":       None,
    "confirmed_settings":          [],
}

RE_TS         = re.compile(r'^\[T\+([\d.]+)\]')
RE_STEP_LOG   = re.compile(r'Step (\d+) \| Loss:.*?time/total_step: ([\d.]+)s')
RE_STEP_BENCH = re.compile(r'\[step_time\] Average wall time over first 10 training steps: ([\d.]+) s')

CONFIRM_PATTERNS = [
    re.compile(r'\[optimization\] torch\.compile'),
    re.compile(r'\[optimization\] channels_last'),
    re.compile(r'\[optimization\] gradient checkpointing'),
    re.compile(r'optimization='),
    re.compile(r'VAE trainable params'),
    re.compile(r'Starting training'),
    re.compile(r'dynamo_backend'),
]

step_timestamps = {}
step_times      = {}

try:
    with open(log_file, errors='replace') as f:
        lines = f.readlines()
except Exception as e:
    print(f"  [parse] cannot read log: {e}")
    lines = []

for line in lines:
    ts_m    = RE_TS.match(line)
    elapsed = float(ts_m.group(1)) if ts_m else None
    bare    = line[ts_m.end():].strip() if ts_m else line.strip()

    m = RE_STEP_LOG.search(line)
    if m:
        sn, st = int(m.group(1)), float(m.group(2))
        step_times[sn] = st
        if elapsed is not None:
            step_timestamps[sn] = elapsed

    m = RE_STEP_BENCH.search(line)
    if m:
        result["step10_avg_time_s"] = round(float(m.group(1)), 4)

    for pat in CONFIRM_PATTERNS:
        if pat.search(bare):
            result["confirmed_settings"].append(bare)
            break

if 1 in step_timestamps:
    result["compile_load_time_s"] = round(step_timestamps[1], 3)
if 50 in step_times:
    result["step50_total_step_s"] = round(step_times[50], 4)

print(f"  compile_load_time_s    = {result['compile_load_time_s']} s")
print(f"  step10_avg_time_s      = {result['step10_avg_time_s']} s  (avg of first 10 steps, incl. compile)")
print(f"  step50_total_step_s    = {result['step50_total_step_s']} s")

if result["confirmed_settings"]:
    print("  Confirmed settings (echoed by training script):")
    for ln in result["confirmed_settings"]:
        print(f"    ✓ {ln}")
else:
    print("  [warn] no setting-confirmation lines found in log")

# GPU utilisation window (steps 40-60)
try:
    t_lo = step_timestamps.get(40)
    t_hi = step_timestamps.get(60)
    if t_lo is None or t_hi is None:
        all_ts = sorted(step_timestamps.values())
        n = len(all_ts)
        if n >= 6:
            t_lo = all_ts[n // 3]
            t_hi = all_ts[2 * n // 3]

    gpu_s, mem_s = [], []
    with open(gpu_file) as f:
        next(f)
        for row in f:
            parts = row.strip().split(',')
            if len(parts) == 3:
                try:
                    t, g, m_ = float(parts[0]), float(parts[1]), float(parts[2])
                    if (t_lo is None or t >= t_lo) and (t_hi is None or t <= t_hi):
                        gpu_s.append(g)
                        mem_s.append(m_)
                except ValueError:
                    pass

    if gpu_s:
        result["gpu_util_warm_avg_pct"] = round(sum(gpu_s) / len(gpu_s), 1)
        result["gpu_util_warm_min_pct"] = int(min(gpu_s))
        result["gpu_util_warm_max_pct"] = int(max(gpu_s))
        result["mem_util_warm_avg_pct"] = round(sum(mem_s) / len(mem_s), 1)
        print(f"  gpu_util (steps 40-60)  "
              f"avg={result['gpu_util_warm_avg_pct']}%  "
              f"min={result['gpu_util_warm_min_pct']}%  "
              f"max={result['gpu_util_warm_max_pct']}%  "
              f"({len(gpu_s)} samples, window={t_lo:.1f}s–{t_hi:.1f}s)")
    else:
        print("  gpu_util               = n/a (no samples in window)")
except Exception as e:
    print(f"  [parse] GPU error: {e}")

# Write individual result file
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  → {result_file}")

# If all 5 result files present, print summary table
import glob, os
expected = [
    "no_compile",
    "compile_default",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "compile_max_autotune_no_cg",
]
all_files = [os.path.join(results_dir, f"{n}_result.json") for n in expected]
if all(os.path.exists(p) for p in all_files):
    print()
    print("════════════════════════════════════════════════════════════════════")
    print("  All 5 runs complete — final summary")
    print("════════════════════════════════════════════════════════════════════")

    results = []
    for p in all_files:
        with open(p) as f:
            results.append(json.load(f))

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    def fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "N/A"

    cols = [
        ("Experiment",         "name",                    30, "<", ""),
        ("Load+Compile (s)",   "compile_load_time_s",     18, ">", ""),
        ("Step10 avg (s)",     "step10_avg_time_s",       14, ">", ""),
        ("Step50 iter (s)",    "step50_total_step_s",     14, ">", ""),
        ("GPU avg%",           "gpu_util_warm_avg_pct",   10, ">", "%"),
        ("GPU min%",           "gpu_util_warm_min_pct",   10, ">", "%"),
        ("GPU max%",           "gpu_util_warm_max_pct",   10, ">", "%"),
    ]
    sep = "  "
    header = sep.join(f"{h:{a}{w}}" for h, _, w, a, _ in cols)
    print(header)
    print("─" * len(header))
    for r in results:
        print(sep.join(f"{fmt(r.get(k), sfx):{a}{w}}" for _, k, w, a, sfx in cols))
    print()
    print(f"Full JSON: {summary_path}")
PYEOF

# ── Per-task banner ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  compile benchmark (all_people) — task ${SLURM_ARRAY_TASK_ID:-?}/${n_exp}: ${exp_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "  SLURM array job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node: $(hostname)"
fi
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "════════════════════════════════════════════════════════════════════"
echo "  Experiment-specific overrides : ${exp_overrides}"
echo "  Common overrides (all tasks)  :"
echo "    --data_preset all_people_one_expression  --batch_size ${BENCH_BATCH}  --accumulation_steps 1"
echo "    --learning_rate 5e-4  --optim.lr 5e-4  (no LR scheduling)"
echo "    --epochs ${BENCH_EPOCHS}  (~$((BENCH_EPOCHS * 12)) steps at batch ${BENCH_BATCH} with ~390 clips)"
echo "    --log_every 1  --log_bottleneck_every 1  --profile_timing True (step 50)"
echo "    --wandb False  --save_ckpt False  --eval_every 999999"
echo "  Log  → ${log_file}"
echo "  GPU  → ${gpu_file}"
echo ""

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  nvidia-smi || true
fi

MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (${SLURM_ARRAY_TASK_ID:-1} % 1000)))
export MASTER_PORT
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
echo "MASTER_PORT=${MASTER_PORT}"
echo ""

# ── Start GPU poller ───────────────────────────────────────────────────────────
python3 -u "${TMPDIR_BENCH}/gpu_poller.py" "$gpu_file" &
GPU_POLLER_PID=$!

# ── Run training ───────────────────────────────────────────────────────────────
set +e
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --main_process_port "$MASTER_PORT" \
  "$train_file" \
  "$my_config" \
  --experiment_name "allpeople_b32_${exp_name}" \
  --wandb_expr_name  "allpeople_b32_${exp_name}" \
  --data_preset all_people_one_expression \
  --batch_size "$BENCH_BATCH" \
  --accumulation_steps 1 \
  --learning_rate 5e-4 \
  --optim.lr 5e-4 \
  --lr_scheduler.warmup_steps 0 \
  --lr_scheduler.use_exponential_decay False \
  --epochs "$BENCH_EPOCHS" \
  --log_every 1 \
  --log_bottleneck_every 1 \
  --profile_timing True \
  --profile_timing_step 50 \
  --wandb False \
  --save_ckpt False \
  --eval_every 999999 \
  --full_eval_every 999999 \
  --fixed_seq_eval_every_epochs 0 \
  --final_eval False \
  --profile_step False \
  --log_step_time True \
  --log_training_design_summary False \
  --deterministic True \
  "${extra_args[@]}" \
  2>&1 | python3 -u "${TMPDIR_BENCH}/timestamper.py" | tee "$log_file"
TRAIN_RC=$?
set -e

# ── Stop GPU poller ────────────────────────────────────────────────────────────
kill "$GPU_POLLER_PID" 2>/dev/null || true
wait "$GPU_POLLER_PID" 2>/dev/null || true

# ── Parse and save results ─────────────────────────────────────────────────────
echo ""
echo "Training finished (exit code: ${TRAIN_RC}) — parsing results…"
python3 "${TMPDIR_BENCH}/parse_run.py" \
  "$exp_name" "$log_file" "$gpu_file" "$result_file" "$RESULTS_DIR"
