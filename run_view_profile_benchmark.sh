#!/usr/bin/env bash
#SBATCH --job-name=view_profile
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-1%1

# View-count profiling grid — one SLURM array task per view count, all run in parallel.
#
# Submit:   sbatch run_view_profile_benchmark.sh
# Monitor:  squeue -u $USER
# Results:  outputs/view_profile_benchmark/
#
# Tasks (3 GPUs in parallel):
#   1  2 views
#   2  4 views
#   3  8 views
#
# Each task:
#   - scans NeRSemble DATA_ROOT for the first .pt with exactly V views
#   - trains on that single clip (1 step/epoch) with profile_timing at step 50
#   - saves profile_timing_step50.{txt,json} + a parsed summary row
#   - disables wandb; stops shortly after the profile step (55 epochs)
#
# Optional env overrides:
#   NERSEMBLE_PROCESSED_BASE — parent of 2-frames / 4-frames / 8-frames
#   VIEW_PROFILE_DATA_ROOT   — optional full override for DATA_ROOT (rare)
#   PROFILE_STEP           — global_step to profile (default: 50)
#   BENCH_EPOCHS           — stop after this many epochs (default: PROFILE_STEP + 5)
#
# Data layout (NeRSemble processed):
#   V=2 → .../processed/128-res          (no 2-frames subtree — legacy layout)
#   V=4 → .../processed/4-frames/128-res
#   V=8 → .../processed/8-frames/128-res

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
RESULTS_DIR="${RESULTS_DIR:-${OPEN_SORA_ROOT}/outputs/view_profile_benchmark}"
PROFILE_STEP="${PROFILE_STEP:-50}"
BENCH_EPOCHS="${BENCH_EPOCHS:-$((PROFILE_STEP + 5))}"

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

# ── Experiment table (1-indexed → SLURM_ARRAY_TASK_ID) ───────────────────────
EXPERIMENTS=(
  "2"
  #"4"
  #"8"
)

n_exp=${#EXPERIMENTS[@]}
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if (( idx < 0 || idx >= n_exp )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1} → idx=$idx out of range [0,$((n_exp-1))]"
  exit 1
fi

NUM_VIEWS="${EXPERIMENTS[$idx]}"
exp_name="views_${NUM_VIEWS}v"

log_file="${RESULTS_DIR}/${exp_name}.log"
result_file="${RESULTS_DIR}/${exp_name}_result.json"
exp_dir="${OPEN_SORA_ROOT}/outputs/${exp_name}"

# ── Resolve DATA_ROOT from view count ─────────────────────────────────────────
# Each view count lives under its own tree. V=2 is the exception: clips are under
# processed/128-res directly (not processed/2-frames/128-res).
# Do NOT use a pre-exported DATA_ROOT from the shell — it often points at 4-frames.
NERSEMBLE_PROCESSED_BASE="${NERSEMBLE_PROCESSED_BASE:-/datasets/lindell-proj/neumayr/nersemble_v2/processed}"
if [[ "$NUM_VIEWS" == "2" ]]; then
  VIEW_FRAMES_ROOT="${NERSEMBLE_PROCESSED_BASE}"
  DATA_ROOT="${VIEW_PROFILE_DATA_ROOT:-${NERSEMBLE_PROCESSED_BASE}/128-res}"
else
  VIEW_FRAMES_ROOT="${NERSEMBLE_PROCESSED_BASE}/${NUM_VIEWS}-frames"
  DATA_ROOT="${VIEW_PROFILE_DATA_ROOT:-${VIEW_FRAMES_ROOT}/128-res}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
  if [[ "$NUM_VIEWS" == "2" ]]; then
    echo "  Expected layout for V=2: ${NERSEMBLE_PROCESSED_BASE}/128-res"
  else
    echo "  Expected layout: ${NERSEMBLE_PROCESSED_BASE}/<V>-frames/128-res"
  fi
  echo "  (override with VIEW_PROFILE_DATA_ROOT=... if needed)"
  exit 1
fi
echo "NUM_VIEWS=${NUM_VIEWS}"
echo "NERSEMBLE_PROCESSED_BASE=${NERSEMBLE_PROCESSED_BASE}"
echo "VIEW_FRAMES_ROOT=${VIEW_FRAMES_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"

# ── Find first .pt file with exactly NUM_VIEWS views ─────────────────────────
echo "Scanning ${DATA_ROOT} for a .pt clip with V=${NUM_VIEWS}..."
PT_PATH=$(python3 - "$DATA_ROOT" "$NUM_VIEWS" <<'PY'
import sys
from pathlib import Path
import torch

data_root = Path(sys.argv[1])
target_v = int(sys.argv[2])

for pt in sorted(data_root.rglob("*.pt")):
    try:
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            if "video" not in obj:
                continue
            video = obj["video"]
        else:
            video = obj
        if video.dim() == 5:
            v = int(video.shape[0])
        elif video.dim() == 4:
            v = 1
        else:
            continue
        if v == target_v:
            print(pt)
            raise SystemExit(0)
    except Exception:
        continue

print(f"No .pt with V={target_v} found under {data_root}", file=sys.stderr)
raise SystemExit(1)
PY
) || {
  echo "ERROR: Could not find any .pt with ${NUM_VIEWS} views under ${DATA_ROOT}"
  exit 1
}
echo "Using data: ${PT_PATH}"

# Temp config: patch paths so train.py bucket resolver does not revert to 4-frames
TMP_CONFIG=$(mktemp --suffix=.py)
python3 - "$my_config" "$PT_PATH" "$DATA_ROOT" "$VIEW_FRAMES_ROOT" "$TMP_CONFIG" <<'PY'
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
pt_path = sys.argv[2]
data_root = sys.argv[3]
view_frames_root = sys.argv[4]
out = pathlib.Path(sys.argv[5])
out.write_text(
    base.read_text(encoding="utf-8")
    + f"""

# --- injected by run_view_profile_benchmark.sh ---
data_preset = "single_sequence"
nersemble_processed_base = {view_frames_root!r}
DATA_ROOT = {data_root!r}
dataset_presets["single_sequence"] = dict(
    type="pt_video",
    data_path={pt_path!r},
    repeat=1,
)
# view_in is set via CLI (--model.view_in); do not re-assign model= here (ConfigDict **model breaks).
"""
)
PY
my_config="$TMP_CONFIG"

# ── Helper scripts ─────────────────────────────────────────────────────────────
TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"; rm -f "${TMP_CONFIG:-}"' EXIT

cat > "${TMPDIR_BENCH}/parse_profile.py" <<'PYEOF'
import json
import shutil
import sys
from pathlib import Path

exp_name = sys.argv[1]
num_views = int(sys.argv[2])
pt_path = sys.argv[3]
exp_dir = Path(sys.argv[4])
results_dir = Path(sys.argv[5])
profile_step = int(sys.argv[6])
result_file = Path(sys.argv[7])

profile_json = exp_dir / f"profile_timing_step{profile_step}.json"
profile_txt = exp_dir / f"profile_timing_step{profile_step}.txt"

result = {
    "name": exp_name,
    "num_views": num_views,
    "data_path": pt_path,
    "profile_step": profile_step,
    "profile_json": None,
    "profile_txt": None,
    "train_forward_ms": None,
    "train_backward_ms": None,
    "train_loss_ms": None,
    "train_optimizer_ms": None,
    "attention_view_sdpa_ms": None,
    "encode_cross_attention_ms": None,
    "decode_temporal_loop_total_ms": None,
    "step_total_ms": None,
}

def _block(data, name, *, use_total=False):
    for row in data.get("blocks", []):
        if row.get("name") == name:
            return row.get("ms_total") if use_total else row.get("ms_mean")
    return None

if profile_json.exists():
    dest_json = results_dir / f"{exp_name}_profile_step{profile_step}.json"
    dest_txt = results_dir / f"{exp_name}_profile_step{profile_step}.txt"
    shutil.copy2(profile_json, dest_json)
    result["profile_json"] = str(dest_json)
    if profile_txt.exists():
        shutil.copy2(profile_txt, dest_txt)
        result["profile_txt"] = str(dest_txt)

    with open(profile_json, encoding="utf-8") as f:
        data = json.load(f)

    result["train_forward_ms"] = _block(data, "train.forward")
    result["train_backward_ms"] = _block(data, "train.backward")
    result["train_loss_ms"] = _block(data, "train.loss")
    result["train_optimizer_ms"] = _block(data, "train.optimizer")
    result["attention_view_sdpa_ms"] = _block(data, "attention.view.sdpa")
    result["encode_cross_attention_ms"] = _block(data, "encode.fusion.cross_attention")
    result["decode_temporal_loop_total_ms"] = _block(
        data, "decode.temporal_loop", use_total=True
    )

    tops = [
        result["train_forward_ms"],
        result["train_backward_ms"],
        result["train_loss_ms"],
        result["train_optimizer_ms"],
    ]
    if all(v is not None for v in tops):
        result["step_total_ms"] = round(sum(tops), 3)

    print(f"  profile saved     → {dest_json}")
    if result["profile_txt"]:
        print(f"  readable report   → {dest_txt}")
    print(f"  step_total_ms     = {result['step_total_ms']}")
    print(f"  train.forward     = {result['train_forward_ms']} ms")
    print(f"  train.backward    = {result['train_backward_ms']} ms")
    print(f"  attention.view.sdpa = {result['attention_view_sdpa_ms']} ms")
    print(f"  decode.temporal_loop total = {result['decode_temporal_loop_total_ms']} ms")
else:
    print(f"  [warn] missing profile file: {profile_json}")

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"  → {result_file}")

expected = ["views_2v", "views_4v", "views_8v"]
all_files = [results_dir / f"{n}_result.json" for n in expected]
if all(p.exists() for p in all_files):
    print()
    print("════════════════════════════════════════════════════════════════════")
    print("  All 3 view-profile runs complete — comparison")
    print("════════════════════════════════════════════════════════════════════")
    rows = []
    for p in all_files:
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    cols = [
        ("Views", "num_views", 6, "d"),
        ("Step (ms)", "step_total_ms", 11, "f"),
        ("Forward (ms)", "train_forward_ms", 13, "f"),
        ("Backward (ms)", "train_backward_ms", 14, "f"),
        ("View SDPA (ms)", "attention_view_sdpa_ms", 15, "f"),
        ("Decode loop (ms)", "decode_temporal_loop_total_ms", 16, "f"),
    ]
    header = "  ".join(f"{h:>{w}}" for h, _, w, _ in cols)
    print(header)
    print("─" * len(header))
    for r in rows:
        parts = []
        for _, key, width, kind in cols:
            val = r.get(key)
            if val is None:
                parts.append(f"{'N/A':>{width}}")
            elif kind == "d":
                parts.append(f"{val:>{width}d}")
            else:
                parts.append(f"{val:>{width}.1f}")
        print("  ".join(parts))
    print()
    print(f"Full JSON: {summary_path}")
    print("Per-run profile reports:")
    for r in rows:
        print(f"  V={r['num_views']}: {r.get('profile_txt') or r.get('profile_json')}")
PYEOF

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  View profile benchmark — task ${SLURM_ARRAY_TASK_ID:-?}/${n_exp}: ${exp_name}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "  SLURM job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node: $(hostname)"
fi
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "════════════════════════════════════════════════════════════════════"
echo "  num_views=${NUM_VIEWS}"
echo "  data_path=${PT_PATH}"
echo "  profile_timing_step=${PROFILE_STEP}  epochs=${BENCH_EPOCHS}"
echo "  experiment_dir=${exp_dir}"
echo "  results_dir=${RESULTS_DIR}"
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
  --experiment_name "$exp_name" \
  --data_preset single_sequence \
  --model.view_in "$NUM_VIEWS" \
  --batch_size 1 \
  --accumulation_steps 1 \
  --epochs "$BENCH_EPOCHS" \
  --log_every 1 \
  --log_bottleneck_every 0 \
  --wandb False \
  --save_ckpt False \
  --eval_every 999999 \
  --full_eval_every 999999 \
  --fixed_seq_eval_every_epochs 0 \
  --final_eval False \
  --profile_step False \
  --profile_timing True \
  --profile_timing_step "$PROFILE_STEP" \
  --log_step_time True \
  --log_training_design_summary False \
  --deterministic True \
  2>&1 | tee "$log_file"
TRAIN_RC=$?
set -e

echo ""
echo "Training finished (exit code: ${TRAIN_RC}) — collecting profile results…"
python3 "${TMPDIR_BENCH}/parse_profile.py" \
  "$exp_name" "$NUM_VIEWS" "$PT_PATH" "$exp_dir" "$RESULTS_DIR" \
  "$PROFILE_STEP" "$result_file"
