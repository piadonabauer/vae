#!/usr/bin/env python3
"""
Downsample NeRSemble processed .pt files (spatial + temporal) while preserving
the original directory structure.

Assumptions (based on preprocess_nersemble.py output):
- Each .pt file stores either:
  - a dict with key 'video' holding a tensor shaped [V, T, C, H, W] (common), or
  - a dict with key 'tensor', or
  - a raw tensor.

This script:
- Recursively scans input_root for .pt files.
- Loads each .pt on CPU.
- Converts to a canonical layout ([V,C,T,H,W] or [C,T,H,W]) for processing.
- Uniformly selects target_t frames from the original T (e.g. 13 -> 9).
- Resizes H,W to target_hw (e.g. 128 -> 64).
- Restores the original layout of the stored tensor (e.g. [V,T,C,H,W]) and saves
  to output_root with the exact same relative path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def _select_frames_uniform(t: int, target_t: int, device: torch.device) -> torch.Tensor:
    if t == target_t:
        return torch.arange(t, device=device, dtype=torch.long)
    if t < 1:
        raise ValueError(f"Invalid T={t}")
    # Evenly spaced indices from [0, t-1]
    return torch.linspace(0, t - 1, target_t, device=device).long()


def _to_canonical(
    video: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    """
    Convert supported shapes to canonical layout for processing.

    Returns:
      canonical tensor and a layout tag to restore original layout.
    """
    if video.dim() == 5:
        # Either [V, T, C, H, W] or [V, C, T, H, W]
        if video.shape[1] > video.shape[2]:
            return video.permute(0, 2, 1, 3, 4).contiguous(), "V_T_C_H_W"  # -> [V,C,T,H,W]
        return video.contiguous(), "V_C_T_H_W"
    if video.dim() == 4:
        # Either [C, T, H, W] or [T, C, H, W]
        if video.shape[0] > video.shape[1]:
            return video.permute(1, 0, 2, 3).contiguous(), "T_C_H_W"  # -> [C,T,H,W]
        return video.contiguous(), "C_T_H_W"
    raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")


def _from_canonical(video_cthw: torch.Tensor, layout_tag: str) -> torch.Tensor:
    if layout_tag == "V_T_C_H_W":
        # input is [V,C,T,H,W] -> output [V,T,C,H,W]
        return video_cthw.permute(0, 2, 1, 3, 4).contiguous()
    if layout_tag == "V_C_T_H_W":
        return video_cthw.contiguous()
    if layout_tag == "T_C_H_W":
        # input is [C,T,H,W] -> output [T,C,H,W]
        return video_cthw.permute(1, 0, 2, 3).contiguous()
    if layout_tag == "C_T_H_W":
        return video_cthw.contiguous()
    raise ValueError(f"Unknown layout_tag: {layout_tag}")


def downsample_video_tensor(video: torch.Tensor, target_hw: int, target_t: int) -> torch.Tensor:
    video = video.detach().to(dtype=torch.float32, device="cpu")
    canonical, layout_tag = _to_canonical(video)

    if canonical.dim() == 5:
        # [V, C, T, H, W]
        v, c, t, h, w = canonical.shape
        idx = _select_frames_uniform(t, target_t, canonical.device)
        canonical = canonical.index_select(2, idx)
        # resize per-frame with 2D interpolation
        canonical_perm = canonical.permute(0, 2, 1, 3, 4).reshape(v * target_t, c, h, w)  # [V*T,C,H,W]
        canonical_perm = F.interpolate(
            canonical_perm, size=(target_hw, target_hw), mode="bilinear", align_corners=False
        )
        canonical = canonical_perm.view(v, target_t, c, target_hw, target_hw).permute(0, 2, 1, 3, 4).contiguous()
        out = _from_canonical(canonical, layout_tag)
        return out

    # [C, T, H, W]
    c, t, h, w = canonical.shape
    idx = _select_frames_uniform(t, target_t, canonical.device)
    canonical = canonical.index_select(1, idx)
    canonical_perm = canonical.permute(1, 0, 2, 3).reshape(target_t, c, h, w)  # [T,C,H,W]
    canonical_perm = F.interpolate(
        canonical_perm, size=(target_hw, target_hw), mode="bilinear", align_corners=False
    )
    canonical = canonical_perm.view(target_t, c, target_hw, target_hw).permute(1, 0, 2, 3).contiguous()
    return _from_canonical(canonical, layout_tag)


def extract_video_container(obj: Any) -> Tuple[torch.Tensor, Any, str]:
    """
    Returns (video_tensor, container_obj, key_or_marker)
    - If obj is dict and contains 'video' or 'tensor', return that and the dict and key.
    - If obj is a Tensor, return it and None and 'RAW_TENSOR'.
    """
    if isinstance(obj, dict):
        if "video" in obj and isinstance(obj["video"], torch.Tensor):
            return obj["video"], obj, "video"
        if "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"], obj, "tensor"
        # Fall back to first tensor-like value
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 4:
                return v, obj, k
        raise ValueError("No video tensor found in dict.")
    if isinstance(obj, torch.Tensor):
        return obj, None, "RAW_TENSOR"
    raise ValueError(f"Unsupported .pt contents type: {type(obj)}")


def process_one(pt_path: Path, input_root: Path, output_root: Path, target_hw: int, target_t: int) -> Path:
    rel = pt_path.relative_to(input_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt_path, map_location="cpu")
    video, container, key = extract_video_container(data)
    video_ds = downsample_video_tensor(video, target_hw=target_hw, target_t=target_t)

    if container is None and key == "RAW_TENSOR":
        torch.save(video_ds, out_path)
    else:
        container[key] = video_ds
        torch.save(container, out_path)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("/datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/datasets/lindell-proj/neumayr/nersemble_v2/processed/64-res"),
    )
    p.add_argument("--target-hw", type=int, default=64, help="Target spatial size (H=W).")
    p.add_argument("--target-t", type=int, default=9, help="Target number of frames (e.g. 9 = 8+1).")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    target_hw = int(args.target_hw)
    target_t = int(args.target_t)

    if not input_root.exists():
        raise FileNotFoundError(f"input-root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_root.rglob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt files found under {input_root}")

    print(f"[downsample] Found {len(pt_files)} .pt files under {input_root}")
    print(f"[downsample] Writing to {output_root} (target {target_hw}x{target_hw}, T={target_t})")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for pt_path in pt_files:
        try:
            rel = pt_path.relative_to(input_root)
            out_path = output_root / rel
            if args.skip_existing and out_path.exists():
                n_skip += 1
                continue
            process_one(pt_path, input_root, output_root, target_hw=target_hw, target_t=target_t)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[downsample] ERROR {pt_path}: {e}")

    print(f"[downsample] Done. ok={n_ok} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()

