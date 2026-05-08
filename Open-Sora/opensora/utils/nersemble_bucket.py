"""
Map Open-Sora `bucket_config` keys to NeRSemble processed `DATA_ROOT` and training shapes.

Rules:
- Bucket key prefix `256px_...` → train at 256×256; load from `.../256-res`.
- Bucket key prefix `128px_...` → train at 128×128; load from `.../128-res`.
- Bucket key prefix `64px_...` → train at 64×64; load from `.../64-res` when temporal length ≤ 9.
- 64-res clips only expose 9 temporal frames. If you request 64px with more than 9 frames (e.g. 13),
  load tensors from `128-res` (full length) and downsample spatially to 64 in the training loop.
"""

from __future__ import annotations

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
    if target_px not in (64, 128, 256):
        raise ValueError(f"Unsupported training resolution {target_px}px (use 64, 128, or 256).")

    if target_px == 256:
        data_root = f"{base}/256-res"
        train_target_hw = (256, 256)
    elif target_px == 128:
        data_root = f"{base}/128-res"
        train_target_hw = (128, 128)
    else:
        train_target_hw = (64, 64)
        if train_target_frames > MAX_TEMPORAL_FRAMES_64_RES:
            data_root = f"{base}/128-res"
        else:
            data_root = f"{base}/64-res"

    return {
        "data_root": data_root,
        "train_target_hw": train_target_hw,
        "train_target_frames": train_target_frames,
        "bucket_key": bucket_key,
        "load_from_128_for_64_high_t": target_px == 64
        and train_target_frames > MAX_TEMPORAL_FRAMES_64_RES,
    }
