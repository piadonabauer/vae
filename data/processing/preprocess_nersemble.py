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

  Two input layouts, two outputs:
  * **Extracted folders** (``--nersemble-root`` points at unpacked ``017/``, ``p018/``, …):
    writes one MP4 per camera under the output tree (``cam_<serial>_processed.mp4``). No merged
    ``.pt`` in that mode.
  * **Tar archives** (add ``--from-tars``; root directory contains ``*.tar``): each sequence is
    unpacked only in a temp directory, then saved as one merged ``frames.pt`` tensor
    ``[V,T,C,H,W]`` in ``[0,1]``.

  ``--test`` / ``-test``: for each processed sequence, save **one** PNG under ``--test-dir`` with
  all time frames of the **first** processed camera laid out **side by side** (a single filmstrip).

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
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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
DEFAULT_TEST_DUMP_DIR = Path(__file__).resolve().parent / "test"
NERSEMBLE_PKG_SRC = (
    REPO_ROOT
    / "DiffSynth-Studio"
    / "diffsynth"
    / "core"
    / "data"
    / "nersemble-data"
    / "src"
)

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


def _ensure_nersemble_pkg_on_path() -> Path:
    if not NERSEMBLE_PKG_SRC.exists():
        raise RuntimeError(f"nersemble-data src not found at {NERSEMBLE_PKG_SRC}")
    if str(NERSEMBLE_PKG_SRC) not in sys.path:
        sys.path.insert(0, str(NERSEMBLE_PKG_SRC))
    return NERSEMBLE_PKG_SRC


def build_nersemble_managers(nersemble_root: Path):
    """
    Import and construct NeRSembleParticipantDataManager from local nersemble-data.
    """
    _ensure_nersemble_pkg_on_path()

    from nersemble_data.data.nersemble_data import (  # type: ignore
        NeRSembleDataManager,
        NeRSembleParticipantDataManager,
    )

    data_folder = NeRSembleDataManager(str(nersemble_root))
    return data_folder, NeRSembleParticipantDataManager


# -----------------------------------------------------------------------------
# Tar archives (temporary extract, no persistent extracted tree)
# -----------------------------------------------------------------------------


def _norm_tar_path(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def participant_id_from_tar_filename(path: Path) -> int | None:
    """Map ``017.tar`` / ``p017.tar`` / ``foo_042_bar.tar`` to participant id when possible."""
    _ensure_nersemble_pkg_on_path()
    from nersemble_data.data.nersemble_data import parse_participant_dir_name  # type: ignore

    stem = path.stem
    pid = parse_participant_dir_name(stem)
    if pid is not None:
        return pid
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(?<![0-9])(\d{3})(?![0-9])", stem)
    if m:
        return int(m.group(1))
    return None


def list_participant_tars(tar_root: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for p in sorted(tar_root.glob("*.tar")):
        pid = participant_id_from_tar_filename(p)
        if pid is not None:
            items.append((p, pid))
    return items


def _infer_archive_layout(
    names: Sequence[str], participant_id: int
) -> Tuple[Path, str] | None:
    """
    Find (relative root under extract dir, archive path prefix) for this participant.

    Archive members look like ``017/calibration/camera_params.json`` or
    ``NeRSemble/017/calibration/...``.
    """
    from nersemble_data.data.nersemble_data import parse_participant_dir_name  # type: ignore

    suffix = "calibration/camera_params.json"
    for raw in names:
        n = _norm_tar_path(raw)
        if not n.endswith(suffix) or "/../" in n:
            continue
        parts = n.split("/")
        if len(parts) < 3:
            continue
        p_dir = parts[-3]
        if parse_participant_dir_name(p_dir) != participant_id:
            continue
        root_parts = parts[:-3]
        rel_root = Path(*root_parts) if root_parts else Path(".")
        prefix = "/".join(root_parts + [p_dir]) if root_parts else p_dir
        return rel_root, prefix
    return None


def _tar_skip_topdir_names() -> frozenset:
    """Same idea as nersemble_data ``_SKIP_FLAT_SEQUENCE_ROOT_NAMES`` (avoid false sequences)."""
    return frozenset({"calibration", "sequences", "metadata", "color_calibration"})


# NeRSemble ``BACKGROUND`` holds ``image_*.jpg`` stills only—no ``cam_*.mp4`` for this pipeline.
TAR_IGNORE_SEQUENCE_NAMES = frozenset({"BACKGROUND"})


def _tar_sequence_has_cam_mp4(
    norm_names: Sequence[str], archive_prefix: str, seq_name: str, images_subdir: str
) -> bool:
    """
    True if the archive has at least one ``cam_*.mp4`` under this sequence.

    NeRSemble's sequence **named** ``BACKGROUND`` is reference stills (``image_*.jpg``), not
    the composited white backdrop from RVM—that white fill happens in ``remove_background``
    without reading this folder. This job only ingests ``cam_*.mp4`` clips, so that sequence
    name is skipped here.
    """
    ap = archive_prefix + "/"
    for n in norm_names:
        if "/../" in n or not n.endswith(".mp4"):
            continue
        if not n.startswith(ap):
            continue
        if f"/sequences/{seq_name}/{images_subdir}/cam_" in n:
            return True
        if n.startswith(f"{archive_prefix}/{seq_name}/{images_subdir}/cam_"):
            return True
    return False


def list_sequences_in_tar(
    tar_path: Path, participant_id: int, images_subdir: str = "images"
) -> List[str]:
    """
    Sequence names that contain at least one ``cam_*.mp4`` (video clips).

    ``BACKGROUND`` and other JPEG-only assets are omitted; this script matches
    ``nersemble_data`` ``per_cam`` layout, not ``per_person_cam`` backgrounds.
    """
    with tarfile.open(tar_path, "r:*") as tf:
        norm_names = [_norm_tar_path(x) for x in tf.getnames()]
        layout = _infer_archive_layout(norm_names, participant_id)
        if layout is None:
            return []
        _, archive_prefix = layout
        skip = _tar_skip_topdir_names()
        prefix_slash = f"{archive_prefix}/"
        seen: set[str] = set()

        # Nested: <prefix>/sequences/<seq>/...
        seq_root = f"{archive_prefix}/sequences/"
        for n in norm_names:
            if not n.startswith(seq_root):
                continue
            rest = n[len(seq_root) :]
            if not rest or rest.startswith("."):
                continue
            seq_name = rest.split("/")[0]
            if seq_name and seq_name not in TAR_IGNORE_SEQUENCE_NAMES:
                seen.add(seq_name)

        # Flat export: <prefix>/<seq>/images/cam_*.mp4 (no ``sequences/`` segment)
        for n in norm_names:
            if not n.startswith(prefix_slash) or "/../" in n:
                continue
            rest = n[len(prefix_slash) :]
            parts = rest.split("/")
            if len(parts) < 3:
                continue
            if (
                parts[1] != images_subdir
                or not parts[2].startswith("cam_")
                or not parts[2].endswith(
                ".mp4"
                )
            ):
                continue
            top = parts[0]
            if top in skip or top.startswith("."):
                continue
            if top in TAR_IGNORE_SEQUENCE_NAMES:
                continue
            seen.add(top)

        # Drop sequences with no cam MP4s (e.g. BACKGROUND = JPGs only per NeRSemble layout).
        return sorted(
            s for s in seen if _tar_sequence_has_cam_mp4(norm_names, archive_prefix, s, images_subdir)
        )


def _tar_extract_member(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    try:
        tf.extract(member, dest, set_attrs=False, filter="data")
    except TypeError:
        tf.extract(member, dest, set_attrs=False)


def _find_cam_mp4_member(
    tf: tarfile.TarFile,
    archive_prefix: str,
    sequence_name: str,
    serial: str,
    images_subdir: str = "images",
) -> tarfile.TarInfo | None:
    """
    Locate ``cam_<serial>.mp4`` for this sequence inside the archive.

    Tries official nested paths, flat ``<p>/<seq>/images/``, and suffix matches so minor
    path differences still extract.
    """
    prefix = archive_prefix + "/"
    exact_nested = f"{archive_prefix}/sequences/{sequence_name}/{images_subdir}/cam_{serial}.mp4"
    exact_flat = f"{archive_prefix}/{sequence_name}/{images_subdir}/cam_{serial}.mp4"
    nested_suffix = f"/sequences/{sequence_name}/{images_subdir}/cam_{serial}.mp4"
    flat_suffix = f"/{sequence_name}/{images_subdir}/cam_{serial}.mp4"

    for m in tf.getmembers():
        if not m.isfile():
            continue
        n = _norm_tar_path(m.name)
        if "/../" in n:
            continue
        if n == exact_nested or n == exact_flat:
            return m

    for m in tf.getmembers():
        if not m.isfile():
            continue
        n = _norm_tar_path(m.name)
        if not n.startswith(prefix) or "/../" in n:
            continue
        if n.endswith(nested_suffix) or n.endswith(flat_suffix):
            return m

    nested_pat = f"/sequences/{sequence_name}/{images_subdir}/cam_{serial}"
    flat_pat = f"/{sequence_name}/{images_subdir}/cam_{serial}"
    for m in tf.getmembers():
        if not m.isfile():
            continue
        n = _norm_tar_path(m.name)
        if not n.startswith(prefix):
            continue
        if not n.endswith(".mp4"):
            continue
        if nested_pat in n or (flat_pat in n and "/sequences/" not in n):
            return m
    return None


def extract_sequence_camera_mp4s(
    tar_path: Path,
    extract_root: Path,
    participant_id: int,
    sequence_name: str,
    camera_serials: Sequence[str],
    images_subdir: str = "images",
) -> None:
    """Extract only the requested ``cam_<serial>.mp4`` files (archive already has calibration)."""
    with tarfile.open(tar_path, "r:*") as tf:
        names = [_norm_tar_path(x) for x in tf.getnames()]
        layout = _infer_archive_layout(names, participant_id)
        if layout is None:
            raise RuntimeError(
                f"Could not resolve layout for participant {participant_id:03d} in {tar_path}"
            )
        _, archive_prefix = layout
        extracted = 0
        missing: List[str] = []
        for serial in camera_serials:
            m = _find_cam_mp4_member(
                tf, archive_prefix, sequence_name, serial, images_subdir=images_subdir
            )
            if m is None:
                missing.append(serial)
                continue
            _tar_extract_member(tf, m, extract_root)
            extracted += 1
        if extracted == 0:
            hint = ""
            sample = [n for n in names if sequence_name in n and n.endswith(".mp4")][:8]
            if sample:
                hint = f" Example member paths containing {sequence_name!r}: {sample}"
            raise RuntimeError(
                f"No camera MP4s extracted for sequence {sequence_name!r} "
                f"(tried serials {list(camera_serials)}; missing {missing}).{hint}"
            )
        if missing:
            print(
                f"[preprocess] WARNING: missing {len(missing)} / {len(camera_serials)} "
                f"camera files in tar for {sequence_name}: {missing}"
            )


def extract_tar_calibration_only(
    tar_path: Path, extract_root: Path, participant_id: int
) -> Path:
    """Extract only ``…/calibration/*`` for this participant; returns nersemble root."""
    with tarfile.open(tar_path, "r:*") as tf:
        names = [_norm_tar_path(x) for x in tf.getnames()]
        layout = _infer_archive_layout(names, participant_id)
        if layout is None:
            raise RuntimeError(
                f"Could not find calibration for participant {participant_id:03d} in {tar_path}"
            )
        rel_root, archive_prefix = layout
        calib_prefix = f"{archive_prefix}/calibration/"
        for m in tf.getmembers():
            if m.isfile() and _norm_tar_path(m.name).startswith(calib_prefix):
                _tar_extract_member(tf, m, extract_root)
        nersemble_root = extract_root / rel_root
        return nersemble_root.resolve()

def select_middle_upper_cameras(nersemble_root: Path, participant_id: int, upper_views: int = 2) -> List[str]:
    """
    Select the two middle cameras among the top `upper_views` cameras by height.
    """
    _ensure_nersemble_pkg_on_path()
    from nersemble_data.data.nersemble_data import resolve_participant_subdir  # type: ignore

    p_dir = resolve_participant_subdir(str(nersemble_root), participant_id)
    calib_path = nersemble_root / p_dir / "calibration" / "camera_params.json"
    if not calib_path.exists():
        raise RuntimeError(f"Camera calibration not found: {calib_path}")

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


def camera_serials_for_upper_views(
    nersemble_root: Path,
    participant_id: int,
    upper_views: int | None,
    explicit_serials: Sequence[str] | None = None,
) -> List[str]:
    if explicit_serials is not None and len(explicit_serials) > 0:
        return list(explicit_serials)
    if upper_views == 2:
        return select_middle_upper_cameras(nersemble_root, participant_id, upper_views=2)
    return select_middle_upper_cameras(
        nersemble_root, participant_id, upper_views=upper_views or 1000
    )


def prepare_temp_nersemble_from_tar(
    tar_path: Path,
    temp_dir: Path,
    participant_id: int,
    sequence_name: str,
    upper_views: int | None,
    explicit_camera_serials: Sequence[str] | None = None,
    images_subdir: str = "images",
) -> Path:
    """
    Extract calibration + required MP4s into ``temp_dir`` and return the on-disk
    NeRSemble root (parent of the participant folder).
    """
    nersemble_local = extract_tar_calibration_only(tar_path, temp_dir, participant_id)
    serials = camera_serials_for_upper_views(
        nersemble_local,
        participant_id,
        upper_views,
        explicit_serials=explicit_camera_serials,
    )
    extract_sequence_camera_mp4s(
        tar_path,
        temp_dir,
        participant_id,
        sequence_name,
        serials,
        images_subdir=images_subdir,
    )
    return nersemble_local


import cv2


def process_mp4_to_tensor(
    input_path: Path,
    converter: Converter | None,
    image_size: int | None = None,
    target_fps: float = 24.0,
    target_frames: int = 13,
    remove_bg: bool = True,
) -> torch.Tensor:
    """
    Load an MP4, temporally subsample, run RVM, optional square resize.

    Returns float tensor ``[T, C, H, W]`` in ``[0, 1]``.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    subsample_factor = max(1, int(fps_orig // target_fps))

    if total_frames // subsample_factor > target_frames:
        indices_to_keep = np.linspace(
            0, total_frames // subsample_factor - 1, target_frames, dtype=int
        )
    else:
        indices_to_keep = np.arange(total_frames // subsample_factor)

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

    matted_frames = frames_all
    if remove_bg:
        if converter is None:
            raise ValueError("remove_bg=True requires a valid Converter instance.")
        t0 = time.time()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            matted_frames = remove_background(
                frames_all, converter, temp_dir, tag=input_path.stem
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[timing] Background removal: {len(matted_frames)} frames in {time.time()-t0:.2f}s")

    rows: List[torch.Tensor] = []
    for img in matted_frames:
        if image_size is not None:
            img = crop_pad_to_square(img, image_size)
        arr = np.asarray(img).astype(np.float32) / 255.0
        rows.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(rows, dim=0).contiguous()


def write_tensor_preview_mp4(
    frames_tc_hw: torch.Tensor,
    output_path: Path,
    fps: float = 24.0,
) -> None:
    """Write ``[T, C, H, W]`` float RGB tensor to an MP4 (per-camera pipeline output)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t, c, h, w = frames_tc_hw.shape
    if c != 3:
        raise ValueError(f"Expected 3 RGB channels, got {c}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed for {output_path}")
    for i in range(t):
        rgb = (frames_tc_hw[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()


def write_timeline_strip_png(frames_tc_hw: torch.Tensor, output_path: Path) -> None:
    """Save ``[T, C, H, W]`` float RGB as one PNG: frames concatenated left-to-right."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t, c, h, w = frames_tc_hw.shape
    if c != 3:
        raise ValueError(f"Expected 3 RGB channels, got {c}")
    strips: List[np.ndarray] = []
    for i in range(t):
        rgb = (frames_tc_hw[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        strips.append(rgb)
    row = np.concatenate(strips, axis=1)
    Image.fromarray(row).save(output_path)


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
    start_total = time.time()
    tensor = process_mp4_to_tensor(
        input_path, converter, image_size=image_size,
        target_fps=target_fps, target_frames=target_frames,
    )
    _, _, h, w = tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {output_path}")
    t0 = time.time()
    for i in range(tensor.shape[0]):
        rgb = (tensor[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"[timing] Writing MP4 done in {time.time()-t0:.2f}s, total {time.time()-start_total:.2f}s")


def process_sequence(
    nersemble_root: Path,
    participant_id: int,
    sequence_name: str,
    converter: Converter | None,
    output_root: Path,
    image_size: int | None = None,
    upper_views: int | None = None,
    skip_existing: bool = False,
    target_frames: int = 13,
    save_merged_pt: bool = False,
    write_mp4_per_camera: bool = True,
    test_dump_dir: Path | None = None,
    explicit_camera_serials: Sequence[str] | None = None,
    images_subdir: str = "images",
    remove_bg: bool = True,
):
    """
    Process a NeRSemble sequence: pick middle-upper cameras, run RVM, then either
    write per-camera MP4s (extracted-folder mode) or a single merged ``frames.pt`` (tar mode).

    When ``test_dump_dir`` is set, write **one** PNG per sequence: time frames of the **first**
    processed camera in a single horizontal strip (before writing ``frames.pt`` when applicable).
    """
    start_seq = time.time()
    _ensure_nersemble_pkg_on_path()
    from nersemble_data.data.nersemble_data import NeRSembleParticipantDataManager  # type: ignore

    serials = camera_serials_for_upper_views(
        nersemble_root,
        participant_id,
        upper_views,
        explicit_serials=explicit_camera_serials,
    )
    pm = NeRSembleParticipantDataManager(str(nersemble_root), participant_id)
    seq_images_default = Path(pm.get_sequence_images_dir(sequence_name))

    # Resolve sequence image dir robustly for both extracted and temp-extracted tar layouts.
    # Some tar exports differ in where sequence folders are rooted.
    seq_candidates: List[Path] = []
    seq_candidates.append(
        seq_images_default if images_subdir == "images" else (seq_images_default.parent / images_subdir)
    )
    seq_candidates.append(seq_images_default)

    try:
        from nersemble_data.data.nersemble_data import resolve_participant_subdir  # type: ignore

        p_dir = resolve_participant_subdir(str(nersemble_root), participant_id)
    except Exception:
        p_dir = f"{participant_id:03d}"

    participant_root = Path(nersemble_root) / p_dir
    seq_candidates.extend(
        [
            participant_root / "sequences" / sequence_name / images_subdir,
            participant_root / sequence_name / images_subdir,
            participant_root / "sequences" / sequence_name / "images",
            participant_root / sequence_name / "images",
        ]
    )

    seq_path: Path | None = None
    for cand in seq_candidates:
        if cand.exists():
            seq_path = cand
            break

    if seq_path is None:
        # Last resort: search for one requested camera file and infer folder.
        search_serials = list(serials) if len(serials) > 0 else ["220700191"]
        for serial in search_serials:
            found = list(participant_root.rglob(f"cam_{serial}.mp4"))
            if found:
                seq_path = found[0].parent
                print(
                    f"[preprocess] WARNING: inferred sequence folder from discovered cam_{serial}.mp4: {seq_path}"
                )
                break

    if seq_path is None:
        raise FileNotFoundError(
            f"Sequence images folder not found for p{participant_id:03d} {sequence_name}. "
            f"Tried: {[str(p) for p in seq_candidates]}"
        )

    participant_dir = output_root / f"p{participant_id:03d}"
    print(f"participant_dir: {participant_dir}")
    seq_dir = participant_dir / sequence_name

    merged_pt_path = seq_dir / "frames.pt"
    if skip_existing and seq_dir.exists():
        if save_merged_pt and merged_pt_path.exists():
            print(f"[preprocess] Skipping (frames.pt exists): p{participant_id:03d} {sequence_name}")
            return seq_dir
        if not save_merged_pt:
            existing_pt = list(seq_dir.glob("*.pt"))
            if existing_pt:
                print(f"[preprocess] Skipping (already has .pt): p{participant_id:03d} {sequence_name}")
                return seq_dir
            expected_mp4s = [seq_dir / f"cam_{serial}_processed.mp4" for serial in serials]
            if all(p.exists() for p in expected_mp4s):
                print(
                    f"[preprocess] Skipping (all camera MP4s exist): p{participant_id:03d} {sequence_name}"
                )
                return seq_dir

    seq_dir.mkdir(parents=True, exist_ok=True)

    views: List[torch.Tensor] = []
    test_strip_written = False
    for serial in serials:
        # Some extracted layouts point seq_path to the sequence root while others
        # point directly to the images folder. Try both robustly.
        cam_candidates = [
            seq_path / f"cam_{serial}.mp4",
            seq_path / images_subdir / f"cam_{serial}.mp4",
            seq_path / "images" / f"cam_{serial}.mp4",
            seq_path / "images_fgr" / f"cam_{serial}.mp4",
        ]
        cam_video_in = next((p for p in cam_candidates if p.exists()), None)
        if cam_video_in is None:
            print(
                "[preprocess] WARNING: Camera video not found (tried: "
                + ", ".join(str(p) for p in cam_candidates)
                + ")"
            )
            continue
        cam_video_out = seq_dir / f"cam_{serial}_processed.mp4"

        if write_mp4_per_camera and cam_video_out.exists():
            print(f"[preprocess] Skipping already processed video: {cam_video_out}")
            continue

        print(f"[preprocess] Processing camera {serial} for {sequence_name}")
        t0 = time.time()
        tensor = process_mp4_to_tensor(
            cam_video_in,
            converter,
            image_size=image_size,
            target_frames=target_frames,
            remove_bg=remove_bg,
        )
        views.append(tensor)
        print(f"[timing] Camera {serial} processed in {time.time()-t0:.2f}s")

        if write_mp4_per_camera:
            write_tensor_preview_mp4(tensor, cam_video_out)

        if test_dump_dir is not None and not test_strip_written:
            strip_name = f"p{participant_id:03d}__{sequence_name}__cam_{serial}_frames.png"
            write_timeline_strip_png(tensor, test_dump_dir / strip_name)
            test_strip_written = True

    if save_merged_pt:
        if len(views) == 0:
            raise RuntimeError(
                f"No camera views processed for p{participant_id:03d} {sequence_name}"
            )
        stacked = torch.stack(views, dim=0)
        torch.save(stacked.cpu(), merged_pt_path)
        print(f"[preprocess] Wrote merged tensor {stacked.shape} -> {merged_pt_path}")

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
        default="/datasets/lindell-proj/neumayr/nersemble_v2/",
        help="Extracted NeRSemble root (per-participant folders), or a directory of *.tar when using --from-tars.",
    )
    p.add_argument(
        "--from-tars",
        action="store_true",
        help="Treat --nersemble-root as a directory of participant .tar archives; extract per sequence to a temp dir, save frames.pt only (no full extracted dataset).",
    )
    p.add_argument(
        "--test",
        "-test",
        action="store_true",
        dest="test_previews",
        help=(
            "Per sequence, save one PNG under --test-dir (first camera, frames in a row). "
            "Does not limit which archives are opened; with --from-tars use --only-participants "
            "to avoid touching other .tar files (e.g. unreadable ones)."
        ),
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DUMP_DIR,
        help="Directory for --test filmstrip PNGs.",
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
        help="Final spatial resolution (image_size x image_size). Use --image-sizes for multiple.",
    )
    p.add_argument(
        "--image-sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="PX",
        dest="image_sizes",
        help=(
            "One or more target resolutions processed in a single tar extraction pass "
            "(e.g. --image-sizes 256 512 1024 2048). Each resolution is saved under "
            "<output-root>/<px>-res/. Takes precedence over --image-size when both are given."
        ),
    )
    p.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Sequence image/video subfolder name (e.g. images or images_fgr).",
    )
    p.add_argument(
        "--camera-serials",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Explicit camera serials (cam_<serial>.mp4). When set, calibration-based "
            "--upper-views selection is bypassed."
        ),
    )
    p.add_argument(
        "--only-sequences",
        type=str,
        nargs="+",
        default=None,
        metavar="SEQ",
        help="Only process listed sequence names (exact match).",
    )
    p.add_argument(
        "--disable-background-removal",
        action="store_true",
        help="Skip RVM and process raw RGB frames only.",
    )
    p.add_argument(
        "--num-participants",
        type=int,
        default=None,
        help=(
            "Limit to first N participants (extracted mode) or first N .tar files in sorted "
            "filename order (--from-tars). For a specific id (e.g. 17 only), prefer "
            "--only-participants 17 so other archives are never opened."
        ),
    )
    p.add_argument(
        "--only-participants",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "With --from-tars: only process archives whose participant id is listed (parsed from "
            "the .tar filename). Other .tar files are never opened—use this if some archives are "
            "unreadable or you want a smoke test on one id."
        ),
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="With --from-tars, process at most this many sequences per archive (after sorting).",
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
    p.add_argument(
        "--temp-dir",
        type=Path,
        default="/home/piado/scratch",
        help=(
            "Parent directory for tempfile (tar extract dirs, RVM scratch). "
            "Example: /scratch/$USER. "
            "Does not change read access to archives on /datasets/."
        ),
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.temp_dir is not None:
        td = Path(args.temp_dir).expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(td)
        print(f"[preprocess] Temp parent (TMPDIR + tar extract): {td}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "array" if "SLURM_ARRAY_TASK_ID" in os.environ else "local"
    print(f"[preprocess] Mode: {mode}, Device: {device}")

    base_output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else (ARRAY_OUTPUT_ROOT if mode == "array" else DEFAULT_LOCAL_OUTPUT)
    )

    # Resolve list of (image_size, output_root) pairs.
    # --image-sizes takes precedence; fall back to --image-size; fall back to [None] (no resize).
    if args.image_sizes:
        resolved_sizes = sorted(set(args.image_sizes))
    elif args.image_size is not None:
        resolved_sizes = [args.image_size]
    else:
        resolved_sizes = [None]

    size_roots: list[tuple[int | None, Path]] = []
    for sz in resolved_sizes:
        folder_name = f"{sz}-res" if sz is not None else "default-res"
        out = base_output_root / folder_name
        out.mkdir(parents=True, exist_ok=True)
        size_roots.append((sz, out))
        print(f"[preprocess] Output root ({folder_name}): {out}")
    if args.from_tars:
        print("[preprocess] Tar mode: temp extract per sequence -> frames.pt (no persistent extract tree)")
    if args.skip_existing:
        print("[preprocess] Skip-existing: on")
    if len(size_roots) > 1:
        print(f"[preprocess] Multi-resolution pass: {[sz for sz, _ in size_roots]}")
    if args.camera_serials:
        deduped_serials = list(dict.fromkeys(args.camera_serials))
        if len(deduped_serials) != len(args.camera_serials):
            print("[preprocess] NOTE: duplicate --camera-serials removed while preserving order.")
        args.camera_serials = deduped_serials

    test_dump_dir: Path | None = None
    if args.test_previews:
        test_dump_dir = Path(args.test_dir)
        test_dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[preprocess] Test strip: one PNG per sequence (first camera) -> {test_dump_dir}")

    remove_bg = not args.disable_background_removal
    converter: Converter | None = None
    if remove_bg:
        converter = Converter("mobilenetv3", str(args.rvm_checkpoint), device=device)
    else:
        print("[preprocess] Background removal disabled; using raw frames only.")

    if args.from_tars:
        _ensure_nersemble_pkg_on_path()
        tar_items = list_participant_tars(args.nersemble_root)
        if args.only_participants is not None:
            allow = set(args.only_participants)
            tar_items = [(p, pid) for p, pid in tar_items if pid in allow]
            if not tar_items:
                print(
                    "[preprocess] No .tar matched --only-participants "
                    f"{sorted(allow)} (check filenames vs ids)."
                )
                return
            print(f"[preprocess] Tar filter: only participant id(s) {sorted(allow)}")
        if args.num_participants:
            tar_items = tar_items[: args.num_participants]
        if mode == "array":
            total_gpus = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
            task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
            chunk_size = (len(tar_items) + total_gpus - 1) // total_gpus
            start, end = task_id * chunk_size, (task_id + 1) * chunk_size
            tar_items = tar_items[start:end]

        tar_plans: List[Tuple[Path, int, List[str]]] = []
        skipped_names: List[str] = []
        for t, pid in tar_items:
            try:
                sequences = list_sequences_in_tar(t, pid, images_subdir=args.images_subdir)
            except OSError as _tar_err:
                print(f"[preprocess] Cannot open {t.name}: {type(_tar_err).__name__}: {_tar_err}")
                skipped_names.append(t.name)
                continue
            if args.only_sequences:
                allow_seq = set(args.only_sequences)
                sequences = [s for s in sequences if s in allow_seq]
            if args.max_sequences is not None:
                sequences = sequences[: args.max_sequences]
            if not sequences:
                print(
                    f"[preprocess] WARNING: no cam_*.mp4 sequences in {t.name} "
                    f"(p{pid:03d}); skipped (e.g. only BACKGROUND stills)."
                )
                continue
            tar_plans.append((t, pid, sequences))

        n_seq = sum(len(seqs) for _, _, seqs in tar_plans)
        print(
            f"[preprocess] Tar mode: {len(tar_plans)} readable archives, "
            f"~{n_seq} sequences (this chunk)"
        )
        if skipped_names:
            print(
                "[preprocess] Skipping unreadable archive(s) (OS denies read on these paths; "
                "`--temp-dir` only changes where we *write* temp files, not read access to "
                f"/datasets/): {', '.join(sorted(skipped_names))}"
            )

        for tar_path, pid, sequences in tar_plans:
            for seq in sequences:
                print(f"[preprocess] Processing p{pid:03d} {seq} from {tar_path.name}")
                try:
                    with tempfile.TemporaryDirectory(
                        dir=os.environ.get("TMPDIR") or None
                    ) as tmp:
                        tmp_path = Path(tmp)
                        nersemble_local = prepare_temp_nersemble_from_tar(
                            tar_path,
                            tmp_path,
                            pid,
                            seq,
                            args.upper_views,
                            explicit_camera_serials=args.camera_serials,
                            images_subdir=args.images_subdir,
                        )
                        for img_sz, out_root in size_roots:
                            out_path = process_sequence(
                                nersemble_root=nersemble_local,
                                participant_id=pid,
                                sequence_name=seq,
                                converter=converter,
                                output_root=out_root,
                                image_size=img_sz,
                                upper_views=args.upper_views,
                                skip_existing=args.skip_existing,
                                target_frames=args.frames,
                                save_merged_pt=True,
                                write_mp4_per_camera=False,
                                test_dump_dir=test_dump_dir,
                                explicit_camera_serials=args.camera_serials,
                                images_subdir=args.images_subdir,
                                remove_bg=remove_bg,
                            )
                            print(f"[preprocess] Saved ({img_sz}px): {out_path}")
                except Exception as e:
                    print(f"[preprocess] ERROR: p{pid:03d} {seq}: {e}")
                    continue
        return

    data_folder, ParticipantManager = build_nersemble_managers(args.nersemble_root)
    participants = sorted(data_folder.list_participants())
    if args.num_participants:
        participants = participants[: args.num_participants]

    if mode == "array":
        total_gpus = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        chunk_size = (len(participants) + total_gpus - 1) // total_gpus
        start, end = task_id * chunk_size, (task_id + 1) * chunk_size
        participants = participants[start:end]

    total_sequences = sum(
        len(ParticipantManager(str(args.nersemble_root), pid).list_sequences())
        for pid in participants
    )
    print(f"[preprocess] Processing {len(participants)} participants, ~{total_sequences} sequences")

    for pid in participants:
        pm = ParticipantManager(str(args.nersemble_root), pid)
        pm.nersemble_root = Path(args.nersemble_root)
        sequences = pm.list_sequences()
        if args.only_sequences:
            allow_seq = set(args.only_sequences)
            sequences = [s for s in sequences if s in allow_seq]
        for seq in sequences:
            if seq in TAR_IGNORE_SEQUENCE_NAMES:
                continue
            print(f"[preprocess] Processing p{pid:03d} {seq}")
            try:
                nersemble_root = args.nersemble_root
                for img_sz, out_root in size_roots:
                    out_path = process_sequence(
                        nersemble_root=nersemble_root,
                        participant_id=pid,
                        sequence_name=seq,
                        converter=converter,
                        output_root=out_root,
                        image_size=img_sz,
                        upper_views=args.upper_views,
                        skip_existing=args.skip_existing,
                        target_frames=args.frames,
                        save_merged_pt=False,
                        write_mp4_per_camera=True,
                        test_dump_dir=test_dump_dir,
                        explicit_camera_serials=args.camera_serials,
                        images_subdir=args.images_subdir,
                        remove_bg=remove_bg,
                    )
            except Exception as e:
                print(f"[preprocess] ERROR: p{pid:03d} {seq}: {e}")
                continue


if __name__ == "__main__":
    main()

