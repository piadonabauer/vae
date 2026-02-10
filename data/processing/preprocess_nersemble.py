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


RUN on single GPU

python data/processing/preprocess_nersemble.py \
  --nersemble-root /home/piado/scratch/data/nersemble \
  --output-root /home/piado/projects/aip-lindell/piado/data/preprocessed_initial_experiments \
  --max-tasks 10 \
  --skip-existing

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


# -----------------------------------------------------------------------------
# Paths and helpers
# -----------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_JSON = REPO_ROOT / "data/processing/tasks.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/preprocessed_initial_experiments"
DEFAULT_NERSEMBLE_ROOT = Path("/home/piado/scratch/data/nersemble")
DEFAULT_RVM_CHECKPOINT = REPO_ROOT / "data/rvm_mobilenetv3.pth"


def center_square_crop_uint8(img: np.ndarray) -> np.ndarray:
    """
    Center square crop on HWC uint8 image (no resizing).
    """
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top : top + side, left : left + side]


def temporal_downsample_indices(n_frames: int, target_frames: int) -> List[int]:
    """
    Evenly-spaced indices from [0, n_frames-1] to length target_frames.
    """
    if n_frames <= 0:
        return []
    target_frames = max(1, target_frames)
    if n_frames <= target_frames:
        return list(range(n_frames))
    idx = np.linspace(0, n_frames - 1, target_frames, dtype=np.int64)
    return [int(i) for i in idx]


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
# NeRSemble data access
# -----------------------------------------------------------------------------


@dataclass
class TaskSpec:
    participant_id: int
    sequence_name: str
    tar_paths: Sequence[str]


def load_tasks(tasks_json: Path) -> List[TaskSpec]:
    data = json.loads(tasks_json.read_text())
    tasks: List[TaskSpec] = []
    for item in data:
        tasks.append(
            TaskSpec(
                participant_id=int(item["participant_id"]),
                sequence_name=str(item["sequence_name"]),
                tar_paths=item.get("tar_paths", []),
            )
        )
    return tasks


def select_upper_cameras(
    nersemble_root: Path,
    participant_id: int,
    top_k: int,
) -> List[str]:
    """
    Select top_k cameras by world-space height using camera_params.json.
    """
    calib_path = nersemble_root / f"{participant_id:03d}/calibration/camera_params.json"
    if not calib_path.exists():
        raise RuntimeError(f"Camera calibration not found: {calib_path}")
    camera_params = json.loads(calib_path.read_text())
    world_2_cam = camera_params["world_2_cam"]

    centers = {}
    for serial, w2c in world_2_cam.items():
        w2c_mat = np.array(w2c, dtype=np.float32)
        c2w = np.linalg.inv(w2c_mat)
        centers[serial] = c2w[:3, 3]  # camera center

    ordered = sorted(centers.keys(), key=lambda s: centers[s][1], reverse=True)
    return ordered[:top_k]


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


def process_sequence(
    nersemble_root: Path,
    task: TaskSpec,
    upper_views: int,
    target_frames: int,
    image_size: int,
    converter: Converter,
    output_root: Path,
) -> Path:
    """
    Process a single (participant, sequence) into a .pt file.
    """
    participant_id = task.participant_id
    sequence_name = task.sequence_name

    data_folder, ParticipantManager = build_nersemble_managers(nersemble_root)

    participants = sorted(data_folder.list_participants())
    if participant_id not in participants:
        raise RuntimeError(
            f"Participant {participant_id} not found under {nersemble_root}. "
            f"Available: {participants}"
        )

    pm = ParticipantManager(str(nersemble_root), participant_id)
    sequences = pm.list_sequences()
    if sequence_name not in sequences:
        raise RuntimeError(
            f"Sequence '{sequence_name}' not found for participant {participant_id}. "
            f"Available: {sequences}"
        )

    serials = select_upper_cameras(nersemble_root, participant_id, upper_views)

    # Determine timesteps based on full sequence length
    n_frames = pm.get_n_timesteps(sequence_name)
    t_indices = temporal_downsample_indices(n_frames, target_frames)

    # Temporary directory root for RVM (not heavily used, but kept for API parity)
    tmp_root = output_root / "_rvm_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    view_tensors: List[torch.Tensor] = []

    for v_idx, serial in enumerate(serials):
        frames: List[Image.Image] = []
        for t in t_indices:
            img = pm.load_image(
                sequence_name,
                serial,
                int(t),
                as_uint8=False,
                apply_color_correction=True,
                downscale_factor=None,
            )  # numpy HWC in [0,1]

            # Convert to uint8 and center-crop at ORIGINAL resolution
            img_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            cropped = center_square_crop_uint8(img_u8)
            frames.append(Image.fromarray(cropped))

        # Background removal on high-res cropped frames, WHITE background
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f"rvm_{participant_id:03d}_{sequence_name}_cam{serial}_",
                dir=str(tmp_root),
            )
        )
        matted_frames = remove_background(
            frames,
            converter,
            temp_dir,
            f"p{participant_id}_{sequence_name}_cam{serial}",
        )

        # Resize to final resolution (e.g. 128x128) and convert to tensor [T, C, H, W]
        resized_tensors: List[torch.Tensor] = []
        for img_out in matted_frames:
            img_resized = img_out.resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(img_resized).astype(np.float32) / 255.0  # [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
            resized_tensors.append(t)

        if not resized_tensors:
            raise RuntimeError(
                f"No frames produced after background removal for "
                f"participant {participant_id}, sequence {sequence_name}, serial {serial}"
            )

        view_tensor = torch.stack(resized_tensors, dim=0)  # [T, C, H, W]
        view_tensors.append(view_tensor)

    video = torch.stack(view_tensors, dim=0)  # [V, T, C, H, W]

    # Output path
    safe_seq = sequence_name.replace("/", "_")
    seq_dir = output_root / f"p{participant_id:02d}_{safe_seq}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    out_path = seq_dir / f"{participant_id:03d}_{sequence_name}.pt"

    payload = {
        "video": video,  # [V, T, C, H, W], float32 in [0,1]
        "participant_id": participant_id,
        "sequence_name": sequence_name,
        "serials": serials,
        "timesteps": t_indices,
    }
    torch.save(payload, out_path)
    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess NeRSemble sequences into .pt tensors for VAE training."
    )
    p.add_argument(
        "--tasks-json",
        type=Path,
        default=DEFAULT_TASKS_JSON,
        help="Path to tasks.json describing (participant_id, sequence_name, tar_paths).",
    )
    p.add_argument(
        "--nersemble-root",
        type=Path,
        default=DEFAULT_NERSEMBLE_ROOT,
        help="Root folder of extracted NeRSemble data (per-participant folders).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root where .pt files will be stored.",
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
        default=128,
        help="Final spatial resolution (image_size x image_size).",
    )
    p.add_argument(
        "--task-index",
        type=int,
        default=None,
        help=(
            "Index into tasks.json to process (0-based). "
            "If omitted, will use SLURM_ARRAY_TASK_ID if present, otherwise process all tasks."
        ),
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help=(
            "If set and not using task-index/SLURM_ARRAY_TASK_ID, "
            "limit processing to the first N tasks (useful for single-GPU testing)."
        ),
    )
    p.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help=(
            "If set (and not using task-index/SLURM_ARRAY_TASK_ID), "
            "select tasks for up to this many distinct participant_ids, "
            "including all sequences for each selected participant."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output .pt already exists.",
    )
    return p


def resolve_tasks_to_run(tasks: List[TaskSpec], args: argparse.Namespace) -> List[Tuple[int, TaskSpec]]:
    """
    Decide which tasks to run based on CLI args and SLURM environment.
    Returns list of (global_index, TaskSpec).
    """
    if args.task_index is not None:
        idx = int(args.task_index)
        if idx < 0 or idx >= len(tasks):
            raise IndexError(f"--task-index {idx} out of range for {len(tasks)} tasks.")
        return [(idx, tasks[idx])]

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if idx < 0 or idx >= len(tasks):
            raise IndexError(
                f"SLURM_ARRAY_TASK_ID={idx} out of range for {len(tasks)} tasks."
            )
        return [(idx, tasks[idx])]

    # Single-process mode: optionally restrict by participants or by count
    indices = list(range(len(tasks)))

    if args.max_participants is not None:
        seen = set()
        selected_indices: List[int] = []
        for i, t in enumerate(tasks):
            if t.participant_id not in seen:
                if len(seen) >= args.max_participants:
                    continue
                seen.add(t.participant_id)
            if t.participant_id in seen:
                selected_indices.append(i)
        indices = selected_indices

    if args.max_tasks is not None:
        n = max(0, min(args.max_tasks, len(indices)))
        indices = indices[:n]

    return [(i, tasks[i]) for i in indices]


def main() -> None:
    args = build_arg_parser().parse_args()

    tasks = load_tasks(args.tasks_json)
    tasks_to_run = resolve_tasks_to_run(tasks, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[preprocess] Using device: {device}")
    print(f"[preprocess] NeRSemble root: {args.nersemble_root}")
    print(f"[preprocess] Tasks file: {args.tasks_json}")
    print(f"[preprocess] Output root: {args.output_root}")
    print(f"[preprocess] RVM checkpoint: {args.rvm_checkpoint}")
    sys.stdout.flush()

    if not args.rvm_checkpoint.exists():
        raise FileNotFoundError(f"RVM checkpoint not found at {args.rvm_checkpoint}")

    converter = Converter("mobilenetv3", str(args.rvm_checkpoint), device=device)

    args.output_root.mkdir(parents=True, exist_ok=True)

    total = len(tasks_to_run)
    for local_idx, (global_idx, task) in enumerate(tasks_to_run, start=1):
        safe_seq = task.sequence_name.replace("/", "_")
        seq_dir = args.output_root / f"p{task.participant_id:02d}_{safe_seq}"
        out_path = seq_dir / f"{task.participant_id:03d}_{task.sequence_name}.pt"

        if args.skip_existing and out_path.exists():
            print(
                f"[preprocess] Skipping existing ({local_idx}/{total}, global {global_idx}): "
                f"p{task.participant_id} {task.sequence_name}"
            )
            continue

        print(
            f"[preprocess] Processing ({local_idx}/{total}, global {global_idx}): "
            f"p{task.participant_id} {task.sequence_name}"
        )
        sys.stdout.flush()

        try:
            out = process_sequence(
                nersemble_root=args.nersemble_root,
                task=task,
                upper_views=args.upper_views,
                target_frames=args.frames,
                image_size=args.image_size,
                converter=converter,
                output_root=args.output_root,
            )
        except Exception as exc:  # keep going for array jobs
            print(
                f"[preprocess] ERROR for participant {task.participant_id}, "
                f"sequence {task.sequence_name}: {exc}"
            )
            continue

        print(f"[preprocess] Saved: {out}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

