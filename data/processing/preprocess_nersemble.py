#!/usr/bin/env python3
"""
Preprocess NeRSemble sequences into .pt files for VAE training.

Core logic:
- Read NeRSemble multi-view video data (via nersemble_data).
- For each (participant_id, sequence_name) task:
  - Pick top-k "upper" cameras by height from calibration.
  - Load frames at ORIGINAL resolution with color correction.
  - Center square crop at original resolution.
  - Temporal downsample (e.g. 81 -> 13 frames).
  - Run RobustVideoMatting (RVM) on the high-res cropped frames.
  - Composite onto a WHITE background.
  - Resize to 128x128 and save as a tensor [V, T, C, H, W] in [0,1].

Batching / array-job support:
- tasks are defined in data/processing/tasks.json
- Modes:
  * Single GPU with limit:    --max-tasks 10
  * Array job (SLURM):       use --task-index or SLURM_ARRAY_TASK_ID

The notebook debugging-bg-remove.ipynb imports:
    from preprocess_nersemble import remove_background, Converter
This module provides that API as well.


RUN on single GPU (processes all participants; use --num-participants N to limit):

  python data/processing/preprocess_nersemble.py \\
    --nersemble-root /path/to/nersemble \\
    --output-root /path/to/processed \\
    --image-size 128 \\
    --skip-existing

  With --skip-existing, sequences that already have all camera MP4s or a .pt file are skipped.

Array job:

#SBATCH --array=0-9        # 10 array tasks, each on one GPU
#SBATCH --gres=gpu:1

srun python data/processing/preprocess_nersemble.py \
  --nersemble-root /home/piado/scratch/data/nersemble \
  --output-root /home/piado/projects/aip-lindell/piado/data/preprocessed_initial_experiments \
  --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import time

# -----------------------------------------------------------------------------
# Paths and helpers
# -----------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_OUTPUT = REPO_ROOT / "data/preprocessed_initial_experiments"
ARRAY_OUTPUT_ROOT = Path("/datasets/lindell-proj/neumayr/nersemble_v2/processed")
DEFAULT_RVM_CHECKPOINT = REPO_ROOT / "data/rvm_mobilenetv3.pth"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def crop_pad_to_square(img: Image.Image, target_size: int) -> Image.Image:
    """
    Crop the largest center square from PIL image and resize to target_size x target_size
    without skewing.
    """
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))  # PIL crop

    if side != target_size:
        cropped = cropped.resize((target_size, target_size), Image.BILINEAR)
    return cropped


# -----------------------------------------------------------------------------
# RobustVideoMatting integration
# -----------------------------------------------------------------------------


class Converter:
    """
    Thin wrapper around RobustVideoMatting's MattingNetwork.

    Matches the API used in debugging-bg-remove.ipynb:

        matting_model = Converter("mobilenetv3", str(checkpoint_path), device)
    """

    def __init__(self, variant: str, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.variant = variant
        self.checkpoint_path = str(checkpoint_path)

        rvm_root = REPO_ROOT / "RobustVideoMatting"
        if not rvm_root.exists():
            raise RuntimeError(f"RobustVideoMatting repo not found at {rvm_root}")

        if str(rvm_root) not in sys.path:
            sys.path.insert(0, str(rvm_root))

        from model import MattingNetwork  # type: ignore

        self.model = MattingNetwork(variant=self.variant).to(self.device).eval()
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)


def _frames_to_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    """
    List of PIL RGB images -> tensor [1, T, 3, H, W] in [0,1] float32.
    """
    tensors: List[torch.Tensor] = []
    for img in frames:
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
        tensors.append(t)
    if not tensors:
        raise ValueError("No frames provided to _frames_to_tensor.")
    stacked = torch.stack(tensors, dim=1)  # C, T, H, W
    stacked = stacked.unsqueeze(0)  # B=1, C, T, H, W
    # RVM expects [B, T, C, H, W] or [B, C, H, W]; we use [B, T, C, H, W]
    return stacked.permute(0, 2, 1, 3, 4).contiguous()


def remove_background(
    frames: Sequence[Image.Image],
    converter: Converter,
    temp_dir: Path,
    tag: str,
    downsample_ratio: float = 0.25,
) -> List[Image.Image]:
    """
    Apply RobustVideoMatting to a short clip of frames.

    - Input: list of PIL RGB frames (all same size).
    - Processing: run RVM on ORIGINAL resolution (no pre-resize).
    - Output: list of PIL RGB frames with WHITE background.

    Arguments match usage in debugging-bg-remove.ipynb.
    The temp_dir and tag are kept for API compatibility (no heavy I/O).
    """
    del temp_dir, tag  # not used in the current simple implementation

    if len(frames) == 0:
        return []

    device = converter.device
    model = converter.model

    src = _frames_to_tensor(frames).to(device)  # [1, T, 3, H, W]
    T = src.shape[1]

    rec: List[torch.Tensor | None] = [None] * 4
    with torch.no_grad():
        fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)

    # fgr, pha: [1, T, 3, H, W] and [1, T, 1, H, W] in [0,1]
    fgr = fgr.clamp(0.0, 1.0)
    pha = pha.clamp(0.0, 1.0)

    # WHITE background
    bgr = torch.ones_like(fgr, device=device)
    comp = fgr * pha + bgr * (1.0 - pha)  # [1, T, 3, H, W]
    comp = comp[0]  # [T, 3, H, W]

    out_frames: List[Image.Image] = []
    for t in range(T):
        frame = comp[t].permute(1, 2, 0).cpu().numpy()  # HWC
        frame = np.clip(frame, 0.0, 1.0)
        out = (frame * 255.0).astype(np.uint8)
        out_frames.append(Image.fromarray(out))

    return out_frames


# -----------------------------------------------------------------------------
# NeRSemble data
# -----------------------------------------------------------------------------

@dataclass
class SequenceInfo:
    participant_id: int
    sequence_name: str


def build_nersemble_managers(nersemble_root: Path):
    """
    Import and construct NeRSembleParticipantDataManager from local nersemble-data.
    """
    pkg_root = (
        REPO_ROOT
        / "DiffSynth-Studio"
        / "diffsynth"
        / "core"
        / "data"
        / "nersemble-data"
        / "src"
    )
    if not pkg_root.exists():
        raise RuntimeError(f"nersemble-data src not found at {pkg_root}")

    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from nersemble_data.data.nersemble_data import (  # type: ignore
        NeRSembleDataManager,
        NeRSembleParticipantDataManager,
    )

    data_folder = NeRSembleDataManager(str(nersemble_root))
    return data_folder, NeRSembleParticipantDataManager

def select_middle_upper_cameras(nersemble_root: Path, participant_id: int, upper_views: int = 2) -> List[str]:
    """
    Select the two middle cameras among the top `upper_views` cameras by height.
    """
    calib_path = nersemble_root / f"{participant_id:03d}/calibration/camera_params.json"
    if not calib_path.exists():
        raise RuntimeError(f"Camera calibration not found: {calib_path}")
    
    import json
    camera_params = json.loads(calib_path.read_text())
    centers = {s: np.linalg.inv(np.array(w2c, dtype=np.float32))[:3,3] for s,w2c in camera_params["world_2_cam"].items()}
    
    # Sort cameras by height descending (highest first)
    sorted_serials = sorted(centers.keys(), key=lambda s: centers[s][1], reverse=True)
    
    # Take top N upper cameras
    top_cameras = sorted_serials[:upper_views]
    
    # Pick middle two adjacent cameras from top_cameras
    if len(top_cameras) < 2:
        return top_cameras
    mid = len(top_cameras)//2
    if len(top_cameras) % 2 == 0:
        return top_cameras[mid-1:mid+1]  # even -> middle two
    else:
        # odd -> middle and next one
        return top_cameras[mid:mid+2] if mid+2 <= len(top_cameras) else top_cameras[mid-1:mid+1]

import cv2
import numpy as np

def save_video_mp4(
    input_path: Path,
    output_path: Path,
    converter: Converter,
    image_size: int | None = None,
    target_fps: float = 24.0,
    target_frames: int = 13,
):
    """
    Load a video, subsample to target FPS, downsample to target_frames, 
    run RVM background removal, resize, and save to MP4.
    """
    import shutil
    start_total = time.time()

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    subsample_factor = max(1, int(fps_orig // target_fps))

    # Determine which frames to keep for temporal downsampling
    if total_frames // subsample_factor > target_frames:
        indices_to_keep = np.linspace(0, total_frames // subsample_factor - 1, target_frames, dtype=int)
    else:
        indices_to_keep = np.arange(total_frames // subsample_factor)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        target_fps,
        (image_size or width, image_size or height)
    )

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {output_path}")

    # Step 1 & 2: Read & subsample frames while loading
    frames_all: List[Image.Image] = []
    frame_idx = 0
    keep_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % subsample_factor == 0:
            if keep_idx in indices_to_keep:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_all.append(Image.fromarray(frame_rgb))
            keep_idx += 1
        frame_idx += 1
    cap.release()
    print(f"[timing] Loaded & subsampled ~{len(frames_all)} frames in {time.time()-t0:.2f}s")

    if len(frames_all) == 0:
        raise RuntimeError(f"No frames loaded from {input_path}")

    # Step 3: Run RVM background removal
    t0 = time.time()
    temp_dir = Path(tempfile.mkdtemp())
    matted_frames = remove_background(frames_all, converter, temp_dir, tag=input_path.stem)
    shutil.rmtree(temp_dir)  # cleanup
    print(f"[timing] Background removal: {len(matted_frames)} frames in {time.time()-t0:.2f}s")

    # Step 4: Resize & write video
    t0 = time.time()
    for img in matted_frames:
        if image_size:
            #img = img.resize((image_size, image_size), Image.BILINEAR)
            img = crop_pad_to_square(img, image_size)
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"[timing] Writing MP4 done in {time.time()-t0:.2f}s, total {time.time()-start_total:.2f}s")


def process_sequence(
    nersemble_root: Path,
    participant_id: int,
    sequence_name: str,
    converter: Converter,
    output_root: Path,
    image_size: int | None = None,
    upper_views: int | None = None,  # NEW: optional argument
    skip_existing: bool = False,
):
    """
    Process a NeRSemble sequence: pick the two middle-upper cameras
    or all top N if upper_views not specified, and encode as MP4s after background removal.
    """
    start_seq = time.time()
    seq_path = nersemble_root / f"{participant_id:03d}/sequences/{sequence_name}/images"
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence images folder not found: {seq_path}")

    # Decide which cameras to process
    if upper_views == 2:
        # Use exactly the two middle-upper cameras
        serials = select_middle_upper_cameras(nersemble_root, participant_id, upper_views=2)
    else:
        # Pick all upper cameras (or top `upper_views` if provided)
        serials = select_middle_upper_cameras(nersemble_root, participant_id, upper_views=upper_views or 1000)

    participant_dir = output_root / f"p{participant_id:03d}"
    print(f"participant_dir: {participant_dir}")
    seq_dir = participant_dir / sequence_name

    # Skip entire sequence if already processed
    if skip_existing and seq_dir.exists():
        # Consider done if any .pt exists (downstream bundle) or all camera MP4s exist
        existing_pt = list(seq_dir.glob("*.pt"))
        if existing_pt:
            print(f"[preprocess] Skipping (already has .pt): p{participant_id:03d} {sequence_name}")
            return seq_dir
        expected_mp4s = [seq_dir / f"cam_{serial}_processed.mp4" for serial in serials]
        if all(p.exists() for p in expected_mp4s):
            print(f"[preprocess] Skipping (all camera MP4s exist): p{participant_id:03d} {sequence_name}")
            return seq_dir

    seq_dir.mkdir(parents=True, exist_ok=True)

    for serial in serials:
        cam_video_in = seq_path / f"cam_{serial}.mp4"
        if not cam_video_in.exists():
            print(f"[preprocess] WARNING: Camera video not found: {cam_video_in}")
            continue
        cam_video_out = seq_dir / f"cam_{serial}_processed.mp4"

        # Skip if already processed
        if cam_video_out.exists():
            print(f"[preprocess] Skipping already processed video: {cam_video_out}")
            continue

        print(f"[preprocess] Processing camera {serial} for {sequence_name}")
        t0 = time.time()
        save_video_mp4(cam_video_in, cam_video_out, converter, image_size=image_size)
        print(f"[timing] Camera {serial} processed in {time.time()-t0:.2f}s")

    print(f"[timing] Sequence {sequence_name} done in {time.time()-start_seq:.2f}s")
    return seq_dir

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess NeRSemble sequences into .pt tensors for VAE training."
    )
    p.add_argument(
        "--nersemble-root",
        type=Path,
        default="/datasets/lindell-proj/neumayr/nersemble_v2/extracted",
        help="Root folder of extracted NeRSemble data (per-participant folders).",
    )
    p.add_argument(
        "--rvm-checkpoint",
        type=Path,
        default=DEFAULT_RVM_CHECKPOINT,
        help="Path to rvm_mobilenetv3.pth checkpoint.",
    )
    p.add_argument(
        "--upper-views",
        type=int,
        default=2,
        help="Number of upper cameras to use per sequence.",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=13,
        help="Number of frames per sequence after temporal downsampling.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Final spatial resolution (image_size x image_size).",
    )
    p.add_argument(
        "--num-participants",
        type=int,
        default=None,
        help="Limit to first N participants (default: all).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root (default: dataset path when array, else repo data/preprocessed_initial_experiments).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sequences that are already processed (all camera MP4s or a .pt file present).",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "array" if "SLURM_ARRAY_TASK_ID" in os.environ else "local"
    print(f"[preprocess] Mode: {mode}, Device: {device}")
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        output_root = ARRAY_OUTPUT_ROOT if mode == "array" else DEFAULT_LOCAL_OUTPUT
    folder_name = f"{args.image_size}-res" if args.image_size else "default-res"
    output_root = output_root / folder_name
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] Output root: {output_root}")
    if args.skip_existing:
        print("[preprocess] Skip-existing: on (skipping sequences that already have all MP4s or a .pt)")

    converter = Converter("mobilenetv3", str(args.rvm_checkpoint), device=device)
    data_folder, ParticipantManager = build_nersemble_managers(args.nersemble_root)
    participants = sorted(data_folder.list_participants())
    if args.num_participants:
        participants = participants[:args.num_participants]

    # Array job splitting
    if mode=="array":
        total_gpus = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        chunk_size = (len(participants)+total_gpus-1)//total_gpus
        start, end = task_id*chunk_size, (task_id+1)*chunk_size
        participants = participants[start:end]

    total_sequences = sum(len(ParticipantManager(str(args.nersemble_root), pid).list_sequences()) for pid in participants)
    print(f"[preprocess] Processing {len(participants)} participants, ~{total_sequences} sequences")

    for pid in participants:
        pm = ParticipantManager(str(args.nersemble_root), pid)
        pm.nersemble_root = Path(args.nersemble_root)
        sequences = pm.list_sequences()
        for seq in sequences:
            print(f"[preprocess] Processing p{pid:03d} {seq}")
            try:
                nersemble_root=args.nersemble_root
                out_path = process_sequence(
                    nersemble_root=nersemble_root,
                    participant_id=pid,
                    sequence_name=seq,
                    converter=converter,
                    output_root=output_root,
                    image_size=args.image_size,
                    upper_views=args.upper_views,
                    skip_existing=args.skip_existing,
                )
            except Exception as e:
                print(f"[preprocess] ERROR: p{pid:03d} {seq}: {e}")
                continue
            print(f"[preprocess] Saved: {out_path}")


if __name__ == "__main__":
    main()

