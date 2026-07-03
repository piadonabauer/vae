#!/bin/bash
# Process exactly ONE participant/sequence through the full pipeline
# (upper-middle views -> 9 frames -> color correction -> white background -> resize)
# and save a grid image (rows = views, columns = temporal frames) to a PNG.
#
# Usage:
#   bash /home/coder/vae/preview_one_sequence.sh
#
# Override defaults via env vars:
#   PID=17 VIEWS=4 SIZE=256 OUT=/home/coder/test.png \
#   bash /home/coder/vae/preview_one_sequence.sh

set -euo pipefail

VAE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VAE_DIR"

NERSEMBLE_ROOT="${NERSEMBLE_ROOT:-/home/coder/nersemble-data/data}"
SEQUENCE="${SEQUENCE:-EMO-1-shout+laugh}"
FRAMES="${FRAMES:-9}"
SIZE="${SIZE:-256}"
VIEWS="${VIEWS:-4}"                 # 4 or 8
PID="${PID:-auto}"                  # participant id, or "auto" = first one fully downloaded
BG_METHOD="${BG_METHOD:-rvm}"       # rvm | none
OUT="${OUT:-/home/coder/test.png}"
RVM_CHECKPOINT="${RVM_CHECKPOINT:-$VAE_DIR/data/rvm_mobilenetv3.pth}"

# Upper-middle camera serials, in view order (must match run_preprocess_nersemble_4and8_views.sh).
SERIALS_4="${SERIALS_4:-222200036 220700191 222200037 222200047}"
SERIALS_8="${SERIALS_8:-222200042 222200046 222200036 220700191 222200037 222200047 222200049 221501007}"

if [[ "$VIEWS" == "4" ]]; then SERIALS="$SERIALS_4"; else SERIALS="$SERIALS_8"; fi

if [[ -f "$VAE_DIR/snth/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VAE_DIR/snth/bin/activate"
fi

NERSEMBLE_ROOT="$NERSEMBLE_ROOT" SEQUENCE="$SEQUENCE" FRAMES="$FRAMES" SIZE="$SIZE" \
PID="$PID" BG_METHOD="$BG_METHOD" OUT="$OUT" RVM_CHECKPOINT="$RVM_CHECKPOINT" \
SERIALS="$SERIALS" python3 - <<'PY'
import os, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "data/processing")
import preprocess_nersemble as P

root = Path(os.environ["NERSEMBLE_ROOT"])
seq = os.environ["SEQUENCE"]
frames_n = int(os.environ["FRAMES"])
size = int(os.environ["SIZE"])
bg_method = os.environ["BG_METHOD"]
out = Path(os.environ["OUT"])
serials = os.environ["SERIALS"].split()
pid_env = os.environ["PID"]


def has_sequence(pid: int) -> bool:
    base = root / f"{pid:03d}" / "sequences" / seq / "images"
    return base.is_dir() and all((base / f"cam_{s}.mp4").exists() for s in serials)


# Pick the participant.
if pid_env != "auto":
    pid = int(pid_env)
    if not has_sequence(pid):
        raise SystemExit(f"Participant {pid:03d} is missing {seq} or some of {serials} under {root}")
else:
    candidates = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    pid = next((c for c in candidates if has_sequence(c)), None)
    if pid is None:
        raise SystemExit(f"No downloaded participant has {seq} with all cameras {serials}")
print(f"[preview] participant p{pid:03d}, {len(serials)} views, {frames_n} frames @ {size}px, bg={bg_method}")

converter = None
if bg_method == "rvm":
    device = "cuda" if P.torch.cuda.is_available() else "cpu"
    converter = P.Converter("mobilenetv3", os.environ["RVM_CHECKPOINT"], device=device)

ccm_map = P.load_color_calibration_map(root, pid)
seq_images = root / f"{pid:03d}" / "sequences" / seq / "images"

# rows = views, cols = temporal frames
rows = []
for s in serials:
    full = P.process_camera_to_square_frames(
        seq_images / f"cam_{s}.mp4",
        converter,
        target_frames=frames_n,
        ccm=ccm_map.get(s),
        bg_method=bg_method,
    )
    rows.append([f.resize((size, size), Image.BILINEAR) for f in full])

n_cols = max(len(r) for r in rows)
pad = 4
bg = (210, 210, 210)  # light gray gridlines between cells
W = n_cols * size + (n_cols + 1) * pad
H = len(rows) * size + (len(rows) + 1) * pad
grid = Image.new("RGB", (W, H), bg)
for ri, r in enumerate(rows):
    for ci, img in enumerate(r):
        x = pad + ci * (size + pad)
        y = pad + ri * (size + pad)
        grid.paste(img, (x, y))

out.parent.mkdir(parents=True, exist_ok=True)
grid.save(out)
print(f"[preview] saved grid {grid.size} ({len(rows)} views x {n_cols} frames) -> {out}")
PY
