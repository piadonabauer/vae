"""
Map Open-Sora `bucket_config` keys to NeRSemble processed `DATA_ROOT` and training shapes.

Rules:
- Bucket key prefix `2048px_...` → train at 2048×2048; load from `.../2048-res` (fallback: `128-res`).
- Bucket key prefix `1024px_...` → train at 1024×1024; load from `.../1024-res` (fallback: `128-res`).
- Bucket key prefix `512px_...` → train at 512×512; load from `.../512-res` (fallback: `128-res`).
- Bucket key prefix `256px_...` → train at 256×256; load from `.../256-res` (fallback: `128-res`).
- Bucket key prefix `128px_...` → train at 128×128; load from `.../128-res`.
- Bucket key prefix `64px_...` → train at 64×64; load from `.../64-res` when temporal length ≤ 9.
- 64-res clips only expose 9 temporal frames. If you request 64px with more than 9 frames (e.g. 13),
  load tensors from `128-res` (full length) and downsample spatially to 64 in the training loop.
- Under ``4-frames/`` or ``8-frames/`` trees only ``128-res`` may exist for 8-frames; higher target
  resolutions then load from ``128-res`` and resize on the fly in ``train.py``.
  The ``4-frames/`` tree ships 128, 256, 512, 1024, and 2048-res.
"""

from __future__ import annotations

import os
import re
from typing import Any

DEFAULT_NERSEMBLE_PROCESSED_BASE = "/datasets/lindell-proj/neumayr/nersemble_v2/processed"
# Preprocessed 64-res dataset max temporal length (128-res typically has 13).
MAX_TEMPORAL_FRAMES_64_RES = 9


def resolve_nersemble_bucket(
    bucket_config: dict[str, Any],
    processed_base: str | None = None,
) -> dict[str, Any]:
    """
    Args:
        bucket_config: e.g. ``{"128px_ar1:1": {13: (1.0, 1)}}``.
        processed_base: Root containing ``64-res``, ``128-res``, and optionally ``256-res``
            (default: NeRSemble v2 path).

    Returns:
        data_root: Directory passed to ``pt_video`` ``data_path`` (scan or single file).
        train_target_hw: ``(H, W)`` after optional on-the-fly downsample.
        train_target_frames: Target ``T`` (temporal subsample via ``downsample_video_tensor``).
        bucket_key: The bucket name string (for logging / wandb).
    """
    if not bucket_config:
        raise ValueError("bucket_config must be non-empty")

    base = (processed_base or DEFAULT_NERSEMBLE_PROCESSED_BASE).rstrip("/")
    bucket_key = next(iter(bucket_config))
    inner = bucket_config[bucket_key]
    if not inner:
        raise ValueError(f"bucket_config[{bucket_key!r}] is empty")
    train_target_frames = next(iter(inner.keys()))
    if not isinstance(train_target_frames, int) or train_target_frames < 1:
        raise ValueError(f"Invalid frame count in bucket: {train_target_frames!r}")

    m = re.match(r"^(\d+)px_", bucket_key)
    if not m:
        raise ValueError(
            f"Bucket key must start with e.g. '256px_', '128px_', or '64px_', got: {bucket_key!r}"
        )
    target_px = int(m.group(1))
    if target_px not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError(f"Unsupported training resolution {target_px}px (use 64, 128, 256, 512, 1024, or 2048).")

    if target_px == 2048:
        preferred_root = f"{base}/2048-res"
        train_target_hw = (2048, 2048)
    elif target_px == 1024:
        preferred_root = f"{base}/1024-res"
        train_target_hw = (1024, 1024)
    elif target_px == 512:
        preferred_root = f"{base}/512-res"
        train_target_hw = (512, 512)
    elif target_px == 256:
        preferred_root = f"{base}/256-res"
        train_target_hw = (256, 256)
    elif target_px == 128:
        preferred_root = f"{base}/128-res"
        train_target_hw = (128, 128)
    else:
        train_target_hw = (64, 64)
        if train_target_frames > MAX_TEMPORAL_FRAMES_64_RES:
            preferred_root = f"{base}/128-res"
        else:
            preferred_root = f"{base}/64-res"

    data_root = preferred_root
    load_res_fallback = None
    if not os.path.isdir(data_root):
        for fallback_px in (512, 256, 128, 64):
            candidate = f"{base}/{fallback_px}-res"
            if os.path.isdir(candidate):
                data_root = candidate
                load_res_fallback = fallback_px
                break

    return {
        "data_root": data_root,
        "train_target_hw": train_target_hw,
        "train_target_frames": train_target_frames,
        "bucket_key": bucket_key,
        "load_from_128_for_64_high_t": target_px == 64
        and train_target_frames > MAX_TEMPORAL_FRAMES_64_RES,
        "load_res_fallback": load_res_fallback,
        "target_px": target_px,
    }
