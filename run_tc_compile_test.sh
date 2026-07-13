#!/usr/bin/env bash
#SBATCH --job-name=tc_compile
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:30:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-10%10

# torch.compile benchmark for tc=False and tc=True — 10 parallel GPU tasks, ~2h each.
# Submit:  sbatch run_tc_compile_test.sh
# Monitor: squeue -u $USER
# Results: outputs/compile_test_tc/{tc_false,tc_true}/summary.json
#
# Tasks 1–5  — temporal_compression=False, 5 compile modes
# Tasks 6–10 — temporal_compression=True,  5 compile modes
#
# Modes (repeated for each tc setting):
#   +0  no_compile                 — optimization=False
#   +1  compile_default            — mode="default"                   (Triton op-fusion)
#   +2  compile_reduce_overhead    — mode="reduce-overhead"            (CUDA graphs; may OOM)
#   +3  compile_max_autotune       — mode="max-autotune"               (full tuning + CUDA graphs)
#   +4  compile_max_autotune_no_cg — mode="max-autotune-no-cudagraphs" (full tuning, no graphs)
#
# Protocol:
#   • single_sequence preset, batch=1, accum=1 — exactly 1 update step per epoch
#   • 100 epochs  → 100 update steps
#   • Warm-up window: steps 1–49  (compile, jit, cache fills)
#   • Measurement window: steps 50–100 (stable throughput)
#   • No wandb; results saved locally only
#
# NOTE: reduce-overhead (CUDA graphs) may OOM or recompile-loop for tc=True due to dynamic
#       temporal shapes. That is expected — the task will fail gracefully and still write a
#       partial result JSON so the summary script can note the failure.

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
RESULTS_BASE="${RESULTS_BASE:-${OPEN_SORA_ROOT}/outputs/compile_test_tc}"

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

BENCH_EPOCHS=100          # default: 100 steps (single_sequence = 1 step/epoch)

# ── Experiment table ───────────────────────────────────────────────────────────
# Format:  "tc_group|mode_name|tc_override|compile_overrides|extra_data_overrides"
#
# extra_data_overrides (5th field) appends AFTER the launch-command defaults and
# wins on duplicate flags.  Use it to override data_preset/batch_size/epochs per
# task.  Leave empty (trailing |) for tasks that are fine with the defaults.
#
# tc=False compile modes:
#   • add --model.crossview_grad_checkpoint_decoder False  (fixes the SubgraphTracer
#     crash: torch.compile is incompatible with torch.utils.checkpoint on the per-frame
#     decoder path; disabling it restores the pre-checkpoint behaviour that used to work)
#   • run at batch=32, all_people_one_expression, 15 epochs (~180 steps at 12 steps/epoch)
#     to validate memory at the real training batch size — not just timing at batch=1
EXPERIMENTS=(
  # --- tc=False (tasks 1-5) ---
  "tc_false|no_compile|--model.temporal_compression False|--optimization False|"
  "tc_false|compile_default|--model.temporal_compression False|--optimization True --optimization_compile_mode default --optimization_compile_dynamic True --model.crossview_grad_checkpoint_decoder False|"
  "tc_false|compile_reduce_overhead|--model.temporal_compression False|--optimization True --optimization_compile_mode reduce-overhead --optimization_compile_dynamic False --model.crossview_grad_checkpoint_decoder False|"
  "tc_false|compile_max_autotune|--model.temporal_compression False|--optimization True --optimization_compile_mode max-autotune --optimization_compile_dynamic True --model.crossview_grad_checkpoint_decoder False|"
  "tc_false|compile_max_autotune_no_cg|--model.temporal_compression False|--optimization True --optimization_compile_mode max-autotune-no-cudagraphs --optimization_compile_dynamic True --model.crossview_grad_checkpoint_decoder False|"
  # --- tc=True (tasks 6-10) — already ran cleanly; re-use as-is ---
  "tc_true|no_compile|--model.temporal_compression True|--optimization False|"
  "tc_true|compile_default|--model.temporal_compression True|--optimization True --optimization_compile_mode default --optimization_compile_dynamic True|"
  "tc_true|compile_reduce_overhead|--model.temporal_compression True|--optimization True --optimization_compile_mode reduce-overhead --optimization_compile_dynamic False|"
  "tc_true|compile_max_autotune|--model.temporal_compression True|--optimization True --optimization_compile_mode max-autotune --optimization_compile_dynamic True|"
  "tc_true|compile_max_autotune_no_cg|--model.temporal_compression True|--optimization True --optimization_compile_mode max-autotune-no-cudagraphs --optimization_compile_dynamic True|"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

IFS='|' read -r tc_group mode_name tc_override compile_override extra_data_override <<< "${EXPERIMENTS[$idx]}"

RESULTS_DIR="${RESULTS_BASE}/${tc_group}"
mkdir -p "$RESULTS_DIR"

log_file="${RESULTS_DIR}/${mode_name}.log"
gpu_file="${RESULTS_DIR}/${mode_name}_gpu.csv"
result_file="${RESULTS_DIR}/${mode_name}_result.json"

# All CLI overrides for this task
read -ra tc_args         <<< "$tc_override"
read -ra compile_args    <<< "$compile_override"
read -ra extra_data_args <<< "$extra_data_override"   # may be empty; overrides defaults when set

echo "════════════════════════════════════════════════════════════════════"
echo "  tc_compile task ${SLURM_ARRAY_TASK_ID:-?}/${n_exp}  —  ${tc_group}/${mode_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && echo "  SLURM: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node: $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  tc     : ${tc_override}"
echo "  compile: ${compile_override}"
[[ -n "${extra_data_override}" ]] && echo "  data   : ${extra_data_override} (overrides defaults)"
echo "  steps  : ${BENCH_EPOCHS} epochs default (measure steps 50–100)"
echo "  output : ${RESULTS_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
[[ -n "${SLURM_JOB_ID:-}" ]] && nvidia-smi || true

MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000) + (${SLURM_ARRAY_TASK_ID:-1} % 1000)))
export MASTER_PORT MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
echo "MASTER_PORT=${MASTER_PORT}"
echo ""

# ── Temp helpers ───────────────────────────────────────────────────────────────
TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# gpu_poller.py
cat > "${TMPDIR_BENCH}/gpu_poller.py" <<'PYEOF'
import subprocess, time, sys
out_file = sys.argv[1]
t0 = time.time()
with open(out_file, 'w') as f:
    f.write("elapsed_s,gpu_util_pct,mem_util_pct\n")
    while True:
        try:
            r = subprocess.run(
                ['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = r.stdout.strip().split(',')
                if len(parts) >= 2:
                    f.write(f"{time.time()-t0:.3f},{parts[0].strip()},{parts[1].strip()}\n")
                    f.flush()
        except Exception:
            pass
        time.sleep(1)
PYEOF

# timestamper.py
cat > "${TMPDIR_BENCH}/timestamper.py" <<'PYEOF'
import sys, time
t0 = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[T+{time.time()-t0:.3f}] {line}")
    sys.stdout.flush()
PYEOF

# parse_run.py — extracts stable step times from the measurement window (steps 50-100)
cat > "${TMPDIR_BENCH}/parse_run.py" <<'PYEOF'
import sys, json, re, os, math

tc_group     = sys.argv[1]
mode_name    = sys.argv[2]
log_file     = sys.argv[3]
gpu_file     = sys.argv[4]
result_file  = sys.argv[5]
results_dir  = sys.argv[6]

MEASURE_START = 50   # first step counted in stable-throughput stats
MEASURE_END   = 100

result = {
    "tc_group":           tc_group,
    "mode":               mode_name,
    "status":             "ok",
    "oom":                False,
    "compile_load_time_s":      None,   # wall-clock until step 1 log
    "step10_avg_time_s":        None,   # avg of steps 1-10 (from [step_time] banner)
    "stable_avg_time_s":        None,   # mean of steps 50-100
    "stable_median_time_s":     None,
    "stable_min_time_s":        None,
    "stable_max_time_s":        None,
    "stable_steps_counted":     0,
    "gpu_util_warm_avg_pct":    None,
    "gpu_util_warm_min_pct":    None,
    "gpu_util_warm_max_pct":    None,
    "mem_util_warm_avg_pct":    None,
    "notes":              [],
}

RE_TS        = re.compile(r'^\[T\+([\d.]+)\]')
RE_STEP_LOG  = re.compile(r'Step\s+(\d+)\s*\|.*?time/total_step:\s*([\d.]+)\s*s')
RE_STEP10    = re.compile(r'\[step_time\]\s+Average wall time over first 10.*?:\s*([\d.]+)\s*s')
RE_OOM       = re.compile(r'OutOfMemoryError|CUDA out of memory', re.I)
RE_RECOMPILE = re.compile(r'torch._dynamo.*recompile|Recompiling|recompile_limit', re.I)

step_times      = {}   # step_num → iter_time_s
step_timestamps = {}   # step_num → wall_elapsed_s

try:
    with open(log_file, errors='replace') as f:
        lines = f.readlines()
except Exception as e:
    result["status"] = f"log_unreadable: {e}"
    lines = []

for line in lines:
    ts_m    = RE_TS.match(line)
    elapsed = float(ts_m.group(1)) if ts_m else None

    if RE_OOM.search(line):
        result["oom"] = True
        result["status"] = "oom"
        result["notes"].append(line.strip()[:200])

    if RE_RECOMPILE.search(line):
        result["notes"].append(f"recompile: {line.strip()[:150]}")

    m = RE_STEP_LOG.search(line)
    if m:
        sn, st = int(m.group(1)), float(m.group(2))
        step_times[sn] = st
        if elapsed is not None:
            step_timestamps[sn] = elapsed

    m = RE_STEP10.search(line)
    if m:
        result["step10_avg_time_s"] = round(float(m.group(1)), 4)

# compile load time = wall clock at step 1
if 1 in step_timestamps:
    result["compile_load_time_s"] = round(step_timestamps[1], 3)

# stable window stats
stable = [step_times[s] for s in range(MEASURE_START, MEASURE_END+1) if s in step_times]
result["stable_steps_counted"] = len(stable)
if stable:
    result["stable_avg_time_s"]    = round(sum(stable)/len(stable), 4)
    result["stable_min_time_s"]    = round(min(stable), 4)
    result["stable_max_time_s"]    = round(max(stable), 4)
    s_sorted = sorted(stable)
    n = len(s_sorted)
    mid = n // 2
    result["stable_median_time_s"] = round((s_sorted[mid-1]+s_sorted[mid])/2 if n%2==0 else s_sorted[mid], 4)

# GPU util in stable window
try:
    t_lo = step_timestamps.get(MEASURE_START)
    t_hi = step_timestamps.get(MEASURE_END)
    gpu_s, mem_s = [], []
    with open(gpu_file) as f:
        next(f)
        for row in f:
            parts = row.strip().split(',')
            if len(parts) == 3:
                try:
                    t, g, m_ = float(parts[0]), float(parts[1]), float(parts[2])
                    in_window = True
                    if t_lo is not None and t < t_lo: in_window = False
                    if t_hi is not None and t > t_hi: in_window = False
                    if in_window:
                        gpu_s.append(g); mem_s.append(m_)
                except ValueError:
                    pass
    if gpu_s:
        result["gpu_util_warm_avg_pct"] = round(sum(gpu_s)/len(gpu_s), 1)
        result["gpu_util_warm_min_pct"] = int(min(gpu_s))
        result["gpu_util_warm_max_pct"] = int(max(gpu_s))
        result["mem_util_warm_avg_pct"] = round(sum(mem_s)/len(mem_s), 1)
except Exception as e:
    result["notes"].append(f"gpu_parse_error: {e}")

with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n  [{tc_group}/{mode_name}] status={result['status']}  oom={result['oom']}")
print(f"    compile_load_time_s  = {result['compile_load_time_s']} s")
print(f"    step10_avg_time_s    = {result['step10_avg_time_s']} s")
print(f"    stable_avg_time_s    = {result['stable_avg_time_s']} s  (steps {MEASURE_START}-{MEASURE_END}, n={result['stable_steps_counted']})")
print(f"    stable_median_time_s = {result['stable_median_time_s']} s")
print(f"    gpu_util (stable)    = avg={result['gpu_util_warm_avg_pct']}%  min={result['gpu_util_warm_min_pct']}%  max={result['gpu_util_warm_max_pct']}%")
if result["notes"]:
    print(f"    notes ({len(result['notes'])}):")
    for n in result["notes"][:5]:
        print(f"      • {n}")
print(f"  → {result_file}")

# ── Summary table if all 5 modes for this tc_group are done ───────────────────
MODES = ["no_compile","compile_default","compile_reduce_overhead",
         "compile_max_autotune","compile_max_autotune_no_cg"]
all_files = [os.path.join(results_dir, f"{m}_result.json") for m in MODES]
if all(os.path.exists(p) for p in all_files):
    print()
    print("════════════════════════════════════════════════════════════════════")
    print(f"  All 5 {tc_group} tasks complete — speed summary (steps {MEASURE_START}-{MEASURE_END})")
    print("════════════════════════════════════════════════════════════════════")

    results = []
    for p in all_files:
        with open(p) as f: results.append(json.load(f))

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f: json.dump(results, f, indent=2)

    def fmt(v, suf="", na="N/A"): return f"{v}{suf}" if v is not None else na

    baseline = next((r["stable_avg_time_s"] for r in results if r["mode"]=="no_compile"), None)

    header_row = f"  {'Mode':<35}{'Load(s)':>10}{'S10avg(s)':>11}{'Stable avg(s)':>15}{'Median':>10}{'Speedup':>9}{'GPU%':>8}{'OOM':>5}"
    print(header_row)
    print("  " + "─"*(len(header_row)-2))
    for r in results:
        spdup = (f"{baseline/r['stable_avg_time_s']:.2f}×"
                 if baseline and r["stable_avg_time_s"] else "N/A")
        oom_s = " OOM" if r["oom"] else "    "
        print(f"  {r['mode']:<35}"
              f"{fmt(r['compile_load_time_s'],suf='s'):>10}"
              f"{fmt(r['step10_avg_time_s'],suf='s'):>11}"
              f"{fmt(r['stable_avg_time_s'],suf='s'):>15}"
              f"{fmt(r['stable_median_time_s'],suf='s'):>10}"
              f"{spdup:>9}"
              f"{fmt(r['gpu_util_warm_avg_pct'],suf='%'):>8}"
              f"{oom_s:>5}")
    print()
    print(f"  Full JSON: {summary_path}")
PYEOF

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
  --experiment_name "compile_test_${tc_group}_${mode_name}" \
  --FAST_MODE False \
  --data_preset single_sequence \
  --batch_size 1 \
  --accumulation_steps 1 \
  --epochs "$BENCH_EPOCHS" \
  --log_every 1 \
  --log_schedule_steps "[]" \
  --wandb False \
  --save_ckpt False \
  --eval_every 999999 \
  --full_eval_every 999999 \
  --fixed_seq_eval_every_epochs 0 \
  --profile_timing False \
  --profile_step False \
  --profile_memory_live False \
  --log_step_time True \
  "${tc_args[@]}" \
  "${compile_args[@]}" \
  "${extra_data_args[@]}" \
  2>&1 | python3 -u "${TMPDIR_BENCH}/timestamper.py" | tee "$log_file"
TRAIN_RC=$?
set -e

# ── Stop GPU poller ────────────────────────────────────────────────────────────
kill "$GPU_POLLER_PID" 2>/dev/null || true
wait "$GPU_POLLER_PID" 2>/dev/null || true

echo ""
echo "Training finished (exit code: ${TRAIN_RC}) — parsing results…"

python3 "${TMPDIR_BENCH}/parse_run.py" \
  "$tc_group" "$mode_name" "$log_file" "$gpu_file" "$result_file" "$RESULTS_DIR"

echo ""
echo "Done. Results in: ${RESULTS_DIR}/"
