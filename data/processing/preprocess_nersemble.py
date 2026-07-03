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
from typing import Dict, List, Optional, Sequence, Tuple

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

# Candidate locations for the ``nersemble_data`` package source, tried in order when
# the package is not already importable (e.g. not pip-installed). Override with the
# ``NERSEMBLE_PKG_SRC`` environment variable.
NERSEMBLE_PKG_SRC_CANDIDATES = [
    NERSEMBLE_PKG_SRC,
    REPO_ROOT.parent / "nersemble-data" / "src",
    Path.home() / "nersemble-data" / "src",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def center_square_crop(img: Image.Image) -> Image.Image:
    """Crop the largest centered square from a PIL image (no resize)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def crop_pad_to_square(img: Image.Image, target_size: int) -> Image.Image:
    """
    Crop the largest center square from PIL image and resize to target_size x target_size
    without skewing.
    """
    cropped = center_square_crop(img)
    if cropped.size[0] != target_size:
        cropped = cropped.resize((target_size, target_size), Image.BILINEAR)
    return cropped


def resize_square(img: Image.Image, target_size: Optional[int]) -> Image.Image:
    """Resize an already-square image to ``target_size`` (no-op when ``target_size`` is None)."""
    if target_size is None or img.size[0] == target_size:
        return img
    return img.resize((target_size, target_size), Image.BILINEAR)


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
# Color correction (NeRSemble Cheung2004 per-camera CCM)
# -----------------------------------------------------------------------------


def load_color_calibration_map(
    nersemble_root: Path, participant_id: int
) -> Dict[str, np.ndarray]:
    """
    Load the per-camera color-correction matrices (``{serial: 3x3 CCM}``) for a participant.

    Reads ``<root>/<pid>/calibration/color_calibration.json`` directly (same content as
    ``NeRSembleParticipantDataManager.load_color_calibration``); matches the color-calibration
    usage documented in the NeRSemble repo README (§4.3).
    """
    p_dir = _resolve_participant_dir_name(nersemble_root, participant_id)
    calib_path = Path(nersemble_root) / p_dir / "calibration" / "color_calibration.json"
    if not calib_path.exists():
        raise RuntimeError(f"Color calibration not found: {calib_path}")
    raw = json.loads(calib_path.read_text())
    return {serial: np.array(ccm, dtype=np.float64) for serial, ccm in raw.items()}


def _srgb_to_linear_torch(x: "torch.Tensor") -> "torch.Tensor":
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_torch(x: "torch.Tensor") -> "torch.Tensor":
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _color_correct_batch_torch(
    frames: Sequence[Image.Image], ccm: np.ndarray, device: str
) -> List[Image.Image]:
    """
    GPU color correction for a 3x3 (linear) Cheung2004 CCM.

    Numerically matches ``nersemble_data``'s ``correct_color`` (sRGB EOTF/OETF + CCM in
    linear space) to within 8-bit rounding, but processes all frames in one batched pass.
    """
    arrs = np.stack([np.asarray(f.convert("RGB")) for f in frames]).astype(np.float32) / 255.0
    t = torch.from_numpy(arrs).to(device)  # [N, H, W, 3]
    ccm_t = torch.as_tensor(np.asarray(ccm, dtype=np.float32), device=device)
    lin = _srgb_to_linear_torch(t)
    corr = lin @ ccm_t.T  # per-pixel (CCM @ rgb)
    out = _linear_to_srgb_torch(corr).clamp(0.0, 1.0)
    out = (out * 255.0).round().to(torch.uint8).cpu().numpy()
    return [Image.fromarray(o) for o in out]


def apply_color_correction_frames(
    frames: Sequence[Image.Image],
    ccm: np.ndarray,
    device: Optional[str] = None,
) -> List[Image.Image]:
    """
    Apply a Cheung2004 CCM to each RGB PIL frame (NeRSemble ``correct_color``).

    When ``device`` is a CUDA device and the CCM is the standard 3x3 (linear) matrix, color
    correction runs batched on the GPU (much faster than the per-frame CPU path); otherwise
    it falls back to ``nersemble_data``'s ``correct_color`` exactly.
    """
    ccm_arr = np.asarray(ccm)
    if device is not None and str(device).startswith("cuda") and ccm_arr.shape == (3, 3):
        return _color_correct_batch_torch(frames, ccm_arr, str(device))

    _ensure_nersemble_pkg_on_path()
    from nersemble_data.util.color_correction import correct_color  # type: ignore

    out: List[Image.Image] = []
    for img in frames:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(correct_color(arr, ccm_arr)))
    return out


# -----------------------------------------------------------------------------
# Background removal dispatch (white background)
# -----------------------------------------------------------------------------

BG_REMOVAL_METHODS = ("rvm", "alpha", "none")


def composite_white_with_alpha(
    frames: Sequence[Image.Image], alpha_frames: Sequence[Image.Image]
) -> List[Image.Image]:
    """
    Composite RGB ``frames`` onto a white background using precomputed alpha mattes.

    ``alpha_frames`` are single-channel (or grayscale) PIL images in ``[0, 255]`` that are
    spatially aligned with ``frames`` (same H x W) and in the same temporal order.
    """
    if len(alpha_frames) != len(frames):
        raise ValueError(
            f"alpha/rgb frame count mismatch: {len(alpha_frames)} alpha vs {len(frames)} rgb"
        )
    out: List[Image.Image] = []
    for rgb, a in zip(frames, alpha_frames):
        rgb_arr = np.asarray(rgb.convert("RGB")).astype(np.float32) / 255.0
        a_arr = np.asarray(a.convert("L")).astype(np.float32) / 255.0
        if a_arr.shape[:2] != rgb_arr.shape[:2]:
            a_img = Image.fromarray((a_arr * 255).astype(np.uint8)).resize(
                rgb.size, Image.BILINEAR
            )
            a_arr = np.asarray(a_img).astype(np.float32) / 255.0
        a_arr = a_arr[..., None]
        comp = rgb_arr * a_arr + (1.0 - a_arr)  # white = 1.0
        out.append(Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)))
    return out


def apply_background_removal(
    frames: Sequence[Image.Image],
    method: str,
    converter: "Converter | None",
    *,
    alpha_frames: Optional[Sequence[Image.Image]] = None,
    tag: str = "cam",
) -> List[Image.Image]:
    """
    Replace the background with solid white using the selected method.

    - ``rvm``:   RobustVideoMatting alpha matte, composited on white (no precomputed mattes
                 needed; runs the matting network on the frames directly).
    - ``alpha``: composite using precomputed per-frame alpha mattes supplied via
                 ``alpha_frames`` (e.g. NeRSemble alpha videos). Use this when matte videos
                 are available alongside the RGB videos.
    - ``none``:  pass frames through unchanged.
    """
    if method == "none":
        return list(frames)
    if method == "rvm":
        if converter is None:
            raise ValueError("bg-removal-method=rvm requires a valid Converter instance.")
        tmp = Path(tempfile.mkdtemp())
        try:
            return remove_background(frames, converter, tmp, tag=tag)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if method == "alpha":
        if not alpha_frames:
            raise ValueError(
                "bg-removal-method=alpha requires precomputed alpha mattes, but none were "
                "found for this camera (expected an alpha video alongside the RGB video). "
                "This dataset has no alpha frames; use --bg-removal-method rvm instead."
            )
        return composite_white_with_alpha(frames, alpha_frames)
    raise ValueError(
        f"Unknown bg-removal method {method!r}; expected one of {BG_REMOVAL_METHODS}."
    )


# -----------------------------------------------------------------------------
# NeRSemble data
# -----------------------------------------------------------------------------

@dataclass
class SequenceInfo:
    participant_id: int
    sequence_name: str


def _ensure_nersemble_pkg_on_path() -> Optional[Path]:
    """
    Make ``nersemble_data`` importable.

    Prefers an already-importable package (e.g. ``pip install nersemble_data``); otherwise
    falls back to known source checkouts (``NERSEMBLE_PKG_SRC`` env var, then a few common
    locations).
    """
    try:
        import nersemble_data  # noqa: F401

        return None
    except Exception:
        pass

    env_src = os.environ.get("NERSEMBLE_PKG_SRC")
    candidates = ([Path(env_src)] if env_src else []) + NERSEMBLE_PKG_SRC_CANDIDATES
    for cand in candidates:
        if cand and Path(cand).exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return Path(cand)

    raise RuntimeError(
        "nersemble_data package not found. Install it (`pip install nersemble_data`) or set "
        f"the NERSEMBLE_PKG_SRC env var. Tried: {[str(c) for c in candidates]}"
    )


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

def _resolve_participant_dir_name(nersemble_root: Path, participant_id: int) -> str:
    _ensure_nersemble_pkg_on_path()
    try:
        from nersemble_data.data.nersemble_data import resolve_participant_subdir  # type: ignore

        return resolve_participant_subdir(str(nersemble_root), participant_id)
    except Exception:
        return f"{participant_id:03d}"


def load_camera_centers(nersemble_root: Path, participant_id: int) -> Dict[str, np.ndarray]:
    """Return ``{serial: camera_center_xyz}`` (world space) from ``camera_params.json``."""
    p_dir = _resolve_participant_dir_name(nersemble_root, participant_id)
    calib_path = Path(nersemble_root) / p_dir / "calibration" / "camera_params.json"
    if not calib_path.exists():
        raise RuntimeError(f"Camera calibration not found: {calib_path}")

    camera_params = json.loads(calib_path.read_text())
    return {
        s: np.linalg.inv(np.array(w2c, dtype=np.float64))[:3, 3]
        for s, w2c in camera_params["world_2_cam"].items()
    }


def select_upper_middle_cameras(
    nersemble_root: Path, participant_id: int, n_views: int
) -> List[str]:
    """
    Select ``n_views`` upper, horizontally-central cameras, ordered left -> right.

    The NeRSemble rig (16 cameras) splits cleanly into an upper and a lower row of 8
    cameras each by camera height (world ``y``). We keep the upper row, take the
    ``n_views`` cameras closest to the horizontal centre (smallest ``|x|``), and finally
    sort them left-to-right (ascending ``x``) for a stable, view-consistent order.

    With ``n_views=4`` this yields the four frontal upper cameras
    (``222200047, 222200037, 220700191, 222200036``); ``n_views=8`` yields the full
    upper row.
    """
    if n_views <= 0:
        raise ValueError(f"n_views must be positive, got {n_views}")

    centers = load_camera_centers(nersemble_root, participant_id)
    serials = list(centers.keys())

    # Upper row = top half by height, but never fewer than the requested view count.
    n_upper = max(n_views, len(serials) // 2)
    upper = sorted(serials, key=lambda s: centers[s][1], reverse=True)[:n_upper]

    # Most horizontally-central among the upper row, then stable left-to-right order.
    central = sorted(upper, key=lambda s: abs(centers[s][0]))[:n_views]
    central.sort(key=lambda s: centers[s][0])
    return central


def select_middle_upper_cameras(
    nersemble_root: Path, participant_id: int, upper_views: int = 2
) -> List[str]:
    """Backwards-compatible alias for :func:`select_upper_middle_cameras`."""
    return select_upper_middle_cameras(nersemble_root, participant_id, n_views=upper_views)


def camera_serials_for_upper_views(
    nersemble_root: Path,
    participant_id: int,
    upper_views: int | None,
    explicit_serials: Sequence[str] | None = None,
) -> List[str]:
    if explicit_serials is not None and len(explicit_serials) > 0:
        return list(explicit_serials)
    n = upper_views if (upper_views and upper_views > 0) else 2
    return select_upper_middle_cameras(nersemble_root, participant_id, n_views=n)


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


def decode_subsampled_frames(
    input_path: Path,
    target_fps: float = 24.0,
    target_frames: int = 13,
    grayscale: bool = False,
) -> List[Image.Image]:
    """
    Decode a video, temporally subsample to ~``target_frames`` evenly-spaced frames.

    Returns full-resolution PIL frames (RGB, or ``L`` when ``grayscale`` for alpha mattes).
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
    keep_set = set(int(i) for i in indices_to_keep)

    frames_all: List[Image.Image] = []
    frame_idx = 0
    keep_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % subsample_factor == 0:
            if keep_idx in keep_set:
                if grayscale:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames_all.append(Image.fromarray(gray))
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_all.append(Image.fromarray(frame_rgb))
            keep_idx += 1
        frame_idx += 1
    cap.release()
    print(f"[timing] Loaded & subsampled ~{len(frames_all)} frames in {time.time()-t0:.2f}s")

    if len(frames_all) == 0:
        raise RuntimeError(f"No frames loaded from {input_path}")
    return frames_all


def frames_to_tchw(frames: Sequence[Image.Image]) -> torch.Tensor:
    """List of RGB PIL frames -> float tensor ``[T, C, H, W]`` in ``[0, 1]``."""
    rows = [
        torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        for img in frames
    ]
    return torch.stack(rows, dim=0).contiguous()


def process_camera_to_square_frames(
    input_path: Path,
    converter: Converter | None,
    *,
    target_fps: float = 24.0,
    target_frames: int = 13,
    ccm: Optional[np.ndarray] = None,
    bg_method: str = "rvm",
    alpha_path: Optional[Path] = None,
    cc_device: Optional[str] = None,
) -> List[Image.Image]:
    """
    Full-resolution per-camera pipeline (run once per camera):

      decode + temporal subsample -> color correction -> centre square crop ->
      background removal (white).

    Returns full-resolution **square** PIL frames; callers resize to each target size.
    The expensive steps (decode + matting) therefore happen a single time regardless of
    how many output resolutions are requested.
    """
    frames = decode_subsampled_frames(input_path, target_fps, target_frames)

    if ccm is not None:
        t0 = time.time()
        if cc_device is None and converter is not None:
            cc_device = converter.device
        frames = apply_color_correction_frames(frames, ccm, device=cc_device)
        print(f"[timing] Color correction: {len(frames)} frames in {time.time()-t0:.2f}s")

    frames = [center_square_crop(f) for f in frames]

    alpha_frames: Optional[List[Image.Image]] = None
    if bg_method == "alpha":
        if alpha_path is None or not Path(alpha_path).exists():
            raise ValueError(
                f"bg-removal-method=alpha but no alpha matte video found at {alpha_path}."
            )
        alpha_frames = decode_subsampled_frames(
            Path(alpha_path), target_fps, target_frames, grayscale=True
        )
        alpha_frames = [center_square_crop(a) for a in alpha_frames]

    if bg_method != "none":
        t0 = time.time()
        frames = apply_background_removal(
            frames, bg_method, converter, alpha_frames=alpha_frames, tag=input_path.stem
        )
        print(f"[timing] Background removal ({bg_method}): {len(frames)} frames in {time.time()-t0:.2f}s")

    return frames


def process_mp4_to_tensor(
    input_path: Path,
    converter: Converter | None,
    image_size: int | None = None,
    target_fps: float = 24.0,
    target_frames: int = 13,
    remove_bg: bool = True,
    ccm: Optional[np.ndarray] = None,
    bg_method: Optional[str] = None,
    alpha_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load an MP4, temporally subsample, color-correct, remove background, optional resize.

    Returns float tensor ``[T, C, H, W]`` in ``[0, 1]``. ``bg_method`` (rvm/alpha/none)
    takes precedence over the legacy ``remove_bg`` boolean when provided.
    """
    if bg_method is None:
        bg_method = "rvm" if remove_bg else "none"

    frames = process_camera_to_square_frames(
        input_path,
        converter,
        target_fps=target_fps,
        target_frames=target_frames,
        ccm=ccm,
        bg_method=bg_method,
        alpha_path=alpha_path,
    )
    frames = [resize_square(f, image_size) for f in frames]
    return frames_to_tchw(frames)


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
    size_roots: Optional[Sequence[Tuple[Optional[int], Path]]] = None,
    bg_method: Optional[str] = None,
    color_correction: bool = False,
    alpha_subdir: str = "alpha",
    cc_device: Optional[str] = None,
):
    """
    Process a NeRSemble sequence end-to-end for one or more output resolutions.

    Pipeline (per camera, executed exactly once regardless of how many resolutions are
    requested):

      pick upper-middle cameras -> decode + keep ``target_frames`` -> per-camera color
      correction -> centre square crop -> background removal (white) -> resize to each
      target size.

    ``size_roots`` is a list of ``(image_size, output_root)`` pairs; each resolution is
    written under its own ``output_root``. When omitted it defaults to a single
    ``[(image_size, output_root)]`` pair (backwards-compatible call style). For each
    resolution this either writes per-camera MP4s (``write_mp4_per_camera``) or a single
    merged ``frames.pt`` tensor ``[V, T, C, H, W]`` (``save_merged_pt``).

    When ``test_dump_dir`` is set, write one filmstrip PNG (first camera, smallest size).
    """
    start_seq = time.time()
    _ensure_nersemble_pkg_on_path()
    from nersemble_data.data.nersemble_data import NeRSembleParticipantDataManager  # type: ignore

    if bg_method is None:
        bg_method = "rvm" if remove_bg else "none"

    if size_roots is None:
        size_roots = [(image_size, Path(output_root))]
    size_roots = [(sz, Path(root)) for sz, root in size_roots]

    serials = camera_serials_for_upper_views(
        nersemble_root,
        participant_id,
        upper_views,
        explicit_serials=explicit_camera_serials,
    )

    ccm_map: Dict[str, np.ndarray] = {}
    if color_correction:
        ccm_map = load_color_calibration_map(nersemble_root, participant_id)

    pm = NeRSembleParticipantDataManager(str(nersemble_root), participant_id)

    # Default sequence-images dir. Newer nersemble_data exposes ``get_sequence_images_dir``;
    # the public package (v0.0.6) only has ``get_images_path`` -> derive the folder from it.
    seq_images_default: Optional[Path] = None
    try:
        seq_images_default = Path(pm.get_sequence_images_dir(sequence_name))
    except AttributeError:
        try:
            probe_serial = serials[0] if serials else "220700191"
            seq_images_default = Path(pm.get_images_path(sequence_name, probe_serial)).parent
        except Exception:
            seq_images_default = None

    # Resolve sequence image dir robustly for both extracted and temp-extracted tar layouts.
    # Some tar exports differ in where sequence folders are rooted.
    seq_candidates: List[Path] = []
    if seq_images_default is not None:
        seq_candidates.append(
            seq_images_default
            if images_subdir == "images"
            else (seq_images_default.parent / images_subdir)
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

    # Per-resolution output dirs.
    def _seq_dir(root: Path) -> Path:
        return Path(root) / f"p{participant_id:03d}" / sequence_name

    def _is_done(root: Path) -> bool:
        sd = _seq_dir(root)
        if not sd.exists():
            return False
        if save_merged_pt:
            return (sd / "frames.pt").exists()
        if list(sd.glob("*.pt")):
            return True
        return all((sd / f"cam_{serial}_processed.mp4").exists() for serial in serials)

    pending = list(size_roots)
    if skip_existing:
        pending = [(sz, root) for sz, root in size_roots if not _is_done(root)]
        if not pending:
            print(f"[preprocess] Skipping (all sizes done): p{participant_id:03d} {sequence_name}")
            return _seq_dir(size_roots[0][1])

    for _, root in pending:
        _seq_dir(root).mkdir(parents=True, exist_ok=True)

    # Smallest pending size gets the optional filmstrip preview.
    preview_size = min((sz for sz, _ in pending if sz is not None), default=None)
    per_size_views: Dict[Path, List[torch.Tensor]] = {root: [] for _, root in pending}
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

        alpha_path: Optional[Path] = None
        if bg_method == "alpha":
            alpha_candidates = [
                cam_video_in.with_name(f"cam_{serial}_alpha.mp4"),
                seq_path / alpha_subdir / f"cam_{serial}.mp4",
                seq_path.parent / alpha_subdir / f"cam_{serial}.mp4",
            ]
            alpha_path = next((p for p in alpha_candidates if p.exists()), None)

        print(f"[preprocess] Processing camera {serial} for {sequence_name}")
        t0 = time.time()
        # Decode + color correct + crop + matte ONCE at full resolution.
        full_frames = process_camera_to_square_frames(
            cam_video_in,
            converter,
            target_frames=target_frames,
            ccm=ccm_map.get(serial),
            bg_method=bg_method,
            alpha_path=alpha_path,
            cc_device=cc_device,
        )
        print(f"[timing] Camera {serial} matted in {time.time()-t0:.2f}s")

        # Emit every requested resolution from the single matted clip.
        for sz, root in pending:
            tensor = frames_to_tchw([resize_square(f, sz) for f in full_frames])
            per_size_views[root].append(tensor)

            if write_mp4_per_camera:
                write_tensor_preview_mp4(tensor, _seq_dir(root) / f"cam_{serial}_processed.mp4")

            if (
                test_dump_dir is not None
                and not test_strip_written
                and sz == preview_size
            ):
                strip_name = f"p{participant_id:03d}__{sequence_name}__cam_{serial}_frames.png"
                write_timeline_strip_png(tensor, test_dump_dir / strip_name)
                test_strip_written = True

    if save_merged_pt:
        for sz, root in pending:
            views = per_size_views[root]
            if len(views) == 0:
                raise RuntimeError(
                    f"No camera views processed for p{participant_id:03d} {sequence_name}"
                )
            stacked = torch.stack(views, dim=0)
            merged_pt_path = _seq_dir(root) / "frames.pt"
            torch.save(stacked.cpu(), merged_pt_path)
            print(f"[preprocess] Wrote merged tensor {stacked.shape} -> {merged_pt_path}")

    print(f"[timing] Sequence {sequence_name} done in {time.time()-start_seq:.2f}s")
    return _seq_dir(size_roots[0][1])

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
        help="Skip background removal and process raw RGB frames only (alias for --bg-removal-method none).",
    )
    p.add_argument(
        "--bg-removal-method",
        type=str,
        choices=list(BG_REMOVAL_METHODS),
        default="rvm",
        help=(
            "How to replace the background with white: 'rvm' (RobustVideoMatting, no "
            "precomputed mattes needed), 'alpha' (composite precomputed alpha-matte videos -- "
            "this dataset has none, so it errors unless matte videos are provided), or 'none'. "
            "Overridden to 'none' by --disable-background-removal."
        ),
    )
    p.add_argument(
        "--color-correction",
        action="store_true",
        help=(
            "Apply NeRSemble per-camera color correction (Cheung2004 CCM from "
            "color_calibration.json) before background removal, as in the NeRSemble README."
        ),
    )
    p.add_argument(
        "--save-merged-pt",
        action="store_true",
        help=(
            "In extracted-folder mode, save one merged frames.pt tensor [V,T,C,H,W] per "
            "sequence instead of per-camera processed MP4s (default in --from-tars mode)."
        ),
    )
    p.add_argument(
        "--alpha-subdir",
        type=str,
        default="alpha",
        help="Subfolder holding per-camera alpha-matte videos for --bg-removal-method alpha.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Process this many participants in parallel (extracted-folder mode). Each worker "
            "uses CPU cores for video decode and shares the GPU for matting/color-correction. "
            "Good for large GPUs; e.g. 8 on a 96GB card."
        ),
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


# ---------------------------------------------------------------------------
# Parallel participant processing (extracted-folder mode)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(cfg: dict) -> None:
    """Pool initializer: build one RVM converter per worker and stash shared config."""
    converter = None
    if cfg["bg_method"] == "rvm":
        converter = Converter("mobilenetv3", cfg["rvm_checkpoint"], device=cfg["device"])
    _WORKER["cfg"] = cfg
    _WORKER["converter"] = converter


def _worker_process_participant(pid: int) -> Tuple[int, str]:
    """Process all selected sequences for one participant (runs inside a pool worker)."""
    cfg = _WORKER["cfg"]
    converter = _WORKER["converter"]
    _ensure_nersemble_pkg_on_path()
    from nersemble_data.data.nersemble_data import NeRSembleParticipantDataManager  # type: ignore

    nersemble_root = cfg["nersemble_root"]
    size_roots = cfg["size_roots"]
    pm = NeRSembleParticipantDataManager(str(nersemble_root), pid)
    try:
        sequences = pm.list_sequences()
    except Exception as e:  # pragma: no cover - defensive
        return pid, f"list_sequences failed: {e}"
    if cfg["only_sequences"]:
        allow = set(cfg["only_sequences"])
        sequences = [s for s in sequences if s in allow]

    msgs: List[str] = []
    for seq in sequences:
        if seq in TAR_IGNORE_SEQUENCE_NAMES:
            continue
        try:
            process_sequence(
                nersemble_root=nersemble_root,
                participant_id=pid,
                sequence_name=seq,
                converter=converter,
                output_root=size_roots[0][1],
                size_roots=size_roots,
                upper_views=cfg["upper_views"],
                skip_existing=cfg["skip_existing"],
                target_frames=cfg["frames"],
                save_merged_pt=cfg["save_merged_pt"],
                write_mp4_per_camera=not cfg["save_merged_pt"],
                test_dump_dir=None,
                explicit_camera_serials=cfg["camera_serials"],
                images_subdir=cfg["images_subdir"],
                bg_method=cfg["bg_method"],
                color_correction=cfg["color_correction"],
                alpha_subdir=cfg["alpha_subdir"],
                cc_device=cfg["cc_device"],
            )
            msgs.append(f"{seq} ok")
        except Exception as e:
            msgs.append(f"{seq} ERROR: {e}")
    return pid, ("; ".join(msgs) if msgs else "no sequences")


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

    bg_method = "none" if args.disable_background_removal else args.bg_removal_method
    cc_device = device if device.startswith("cuda") else None
    # In parallel extracted mode each worker builds its own RVM model, so skip the
    # main-process model (still needed for sequential and --from-tars runs).
    use_workers = (not args.from_tars) and args.num_workers > 1
    converter: Converter | None = None
    if bg_method == "rvm" and not use_workers:
        converter = Converter("mobilenetv3", str(args.rvm_checkpoint), device=device)
    if bg_method == "rvm":
        print("[preprocess] Background removal: RVM -> white")
    elif bg_method == "alpha":
        print("[preprocess] Background removal: precomputed alpha mattes -> white")
    else:
        print("[preprocess] Background removal disabled; using raw frames only.")
    if args.color_correction:
        cc_where = "GPU" if cc_device else "CPU"
        print(f"[preprocess] Color correction: on (per-camera Cheung2004 CCM, {cc_where})")
    if use_workers:
        print(f"[preprocess] Parallel participants: {args.num_workers} workers")

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
                        out_path = process_sequence(
                            nersemble_root=nersemble_local,
                            participant_id=pid,
                            sequence_name=seq,
                            converter=converter,
                            output_root=size_roots[0][1],
                            size_roots=size_roots,
                            upper_views=args.upper_views,
                            skip_existing=args.skip_existing,
                            target_frames=args.frames,
                            save_merged_pt=True,
                            write_mp4_per_camera=False,
                            test_dump_dir=test_dump_dir,
                            explicit_camera_serials=args.camera_serials,
                            images_subdir=args.images_subdir,
                            bg_method=bg_method,
                            color_correction=args.color_correction,
                            alpha_subdir=args.alpha_subdir,
                            cc_device=cc_device,
                        )
                        print(f"[preprocess] Saved {[sz for sz, _ in size_roots]}px: {out_path}")
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

    print(f"[preprocess] Processing {len(participants)} participants")

    if use_workers:
        import multiprocessing as _mp

        worker_cfg = {
            "nersemble_root": Path(args.nersemble_root),
            "size_roots": size_roots,
            "upper_views": args.upper_views,
            "skip_existing": args.skip_existing,
            "frames": args.frames,
            "save_merged_pt": args.save_merged_pt,
            "camera_serials": args.camera_serials,
            "images_subdir": args.images_subdir,
            "bg_method": bg_method,
            "color_correction": args.color_correction,
            "alpha_subdir": args.alpha_subdir,
            "cc_device": cc_device,
            "device": device,
            "rvm_checkpoint": str(args.rvm_checkpoint),
            "only_sequences": list(args.only_sequences) if args.only_sequences else None,
        }
        n_workers = max(1, min(args.num_workers, len(participants)))
        ctx = _mp.get_context("spawn")  # required for CUDA in subprocesses
        with ctx.Pool(
            processes=n_workers, initializer=_worker_init, initargs=(worker_cfg,)
        ) as pool:
            for pid, status in pool.imap_unordered(
                _worker_process_participant, participants
            ):
                print(f"[preprocess] p{pid:03d}: {status}")
        return

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
                process_sequence(
                    nersemble_root=args.nersemble_root,
                    participant_id=pid,
                    sequence_name=seq,
                    converter=converter,
                    output_root=size_roots[0][1],
                    size_roots=size_roots,
                    upper_views=args.upper_views,
                    skip_existing=args.skip_existing,
                    target_frames=args.frames,
                    save_merged_pt=args.save_merged_pt,
                    write_mp4_per_camera=not args.save_merged_pt,
                    test_dump_dir=test_dump_dir,
                    explicit_camera_serials=args.camera_serials,
                    images_subdir=args.images_subdir,
                    bg_method=bg_method,
                    color_correction=args.color_correction,
                    alpha_subdir=args.alpha_subdir,
                    cc_device=cc_device,
                )
            except Exception as e:
                print(f"[preprocess] ERROR: p{pid:03d} {seq}: {e}")
                continue


if __name__ == "__main__":
    main()

