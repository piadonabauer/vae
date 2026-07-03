#!/bin/bash
# Preprocess NeRSemble EMO-1-shout+laugh into 8 VAE datasets.
#
# For BOTH 4-view and 8-view camera selections, and for EACH of the resolutions
# 128 / 256 / 512 / 740 px, this produces one dataset -> 2 x 4 = 8 datasets total.
#
# Per camera the pipeline runs exactly once (decode is shared across resolutions):
#   1. select the N upper-middle cameras (frontal upper row)
#   2. keep 9 evenly-spaced temporal frames
#   3. apply NeRSemble per-camera color correction (Cheung2004 CCM)
#   4. remove the background -> solid white (RobustVideoMatting)
#   5. centre square-crop + resize to each target resolution
# Each sequence is saved as a merged tensor frames.pt of shape [V, T, C, H, W] in [0, 1].
#
# Output tree:
#   ${OUTPUT_ROOT}/4view/128-res/p<id>/EMO-1-shout+laugh/frames.pt   ([4, 9, 3, 128, 128])
#   ${OUTPUT_ROOT}/4view/256-res/...                                   ([4, 9, 3, 256, 256])
#   ${OUTPUT_ROOT}/4view/512-res/...                                   ([4, 9, 3, 512, 512])
#   ${OUTPUT_ROOT}/4view/740-res/...                                   ([4, 9, 3, 740, 740])
#   ${OUTPUT_ROOT}/8view/{128,256,512,740}-res/...                     ([8, 9, 3, *, *])
#
# Usage:
#   bash run_preprocess_nersemble_4and8_views.sh
#
# Override defaults via env vars, e.g.:
#   NERSEMBLE_ROOT=/path/to/data OUTPUT_ROOT=/path/out VIEWS="4 8" \
#   ONLY_PARTICIPANTS="17 18" bash run_preprocess_nersemble_4and8_views.sh

set -euo pipefail

VAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VAE_DIR"

# ----------------------------------------------------------------------------
# Config (override via environment)
# ----------------------------------------------------------------------------
NERSEMBLE_ROOT="${NERSEMBLE_ROOT:-/home/coder/nersemble-data/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/coder/nersemble-data/processed}"
TEMP_DIR="${TEMP_DIR:-/tmp/nersemble_preprocess}"
RVM_CHECKPOINT="${RVM_CHECKPOINT:-$VAE_DIR/data/rvm_mobilenetv3.pth}"

SEQUENCE="${SEQUENCE:-EMO-1-shout+laugh}"
FRAMES="${FRAMES:-9}"
IMAGE_SIZES="${IMAGE_SIZES:-128 256 512 740}"
VIEWS="${VIEWS:-4 8}"
BG_METHOD="${BG_METHOD:-rvm}"        # rvm | alpha | none
WORKERS="${WORKERS:-8}"             # parallel participants (uses CPU cores + shared GPU)

# Explicit upper-middle camera serials per view count, in the desired left/right
# view order (the tensor view axis follows this order). These are the upper row of
# the NeRSemble rig; override via env if needed.
SERIALS_4="${SERIALS_4:-222200036 220700191 222200037 222200047}"
SERIALS_8="${SERIALS_8:-222200042 222200046 222200036 220700191 222200037 222200047 222200049 221501007}"

# Optional: limit which participants run (space-separated ids). Empty = all downloaded.
ONLY_PARTICIPANTS="${ONLY_PARTICIPANTS:-}"

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
if [[ -f "$VAE_DIR/snth/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VAE_DIR/snth/bin/activate"
fi
mkdir -p "$OUTPUT_ROOT" "$TEMP_DIR" logs

NUM_PART_ARG=()
if [[ -n "$ONLY_PARTICIPANTS" ]]; then
  # The extracted-mode pipeline iterates all participant folders; restrict by
  # temporarily pointing at a filtered view is overkill, so we rely on the
  # python --num-participants/skip-existing flags. To target specific ids, the
  # simplest robust path is to pass them through a small filter below.
  echo "[run] NOTE: ONLY_PARTICIPANTS set ($ONLY_PARTICIPANTS); processing just those ids."
fi

run_one() {
  local n_views="$1"
  local serials="$2"
  local out_root="$OUTPUT_ROOT/${n_views}view"
  echo "=============================================================="
  echo "[run] ${n_views} views -> $out_root  (sizes: $IMAGE_SIZES)"
  echo "[run] cameras (view order): $serials"
  echo "=============================================================="
  python3 "$VAE_DIR/data/processing/preprocess_nersemble.py" \
    --nersemble-root "$NERSEMBLE_ROOT" \
    --output-root "$out_root" \
    --only-sequences "$SEQUENCE" \
    --camera-serials $serials \
    --frames "$FRAMES" \
    --image-sizes $IMAGE_SIZES \
    --color-correction \
    --bg-removal-method "$BG_METHOD" \
    --rvm-checkpoint "$RVM_CHECKPOINT" \
    --save-merged-pt \
    --num-workers "$WORKERS" \
    --temp-dir "$TEMP_DIR" \
    --skip-existing
}

serials_for_views() {
  case "$1" in
    4) echo "$SERIALS_4" ;;
    8) echo "$SERIALS_8" ;;
    *) echo "" ;;
  esac
}

if [[ -n "$ONLY_PARTICIPANTS" ]]; then
  # Build a temporary root containing symlinks to just the requested participants,
  # so the all-participants extracted-mode loop only sees those ids.
  FILTERED_ROOT="$TEMP_DIR/filtered_root"
  rm -rf "$FILTERED_ROOT"; mkdir -p "$FILTERED_ROOT"
  for pid in $ONLY_PARTICIPANTS; do
    src="$NERSEMBLE_ROOT/$(printf '%03d' "$pid")"
    [[ -d "$src" ]] && ln -s "$src" "$FILTERED_ROOT/$(printf '%03d' "$pid")" || echo "[run] WARN missing $src"
  done
  NERSEMBLE_ROOT="$FILTERED_ROOT"
fi

for v in $VIEWS; do
  serials="$(serials_for_views "$v")"
  if [[ -z "$serials" ]]; then
    echo "[run] No serials configured for ${v} views (set SERIALS_${v}); skipping." >&2
    continue
  fi
  run_one "$v" "$serials"
done

echo "[run] Done. Datasets under: $OUTPUT_ROOT/{4view,8view}/{128,256,512,740}-res/"
