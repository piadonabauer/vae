#!/usr/bin/env python3
"""
Precompute VAE latents for EMO-1-shout+laugh .pt clips.

Example:
  python3 scripts/vae/precompute_emo1_latents.py \
    --config /home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py \
    --output-dir /home/piado/scratch/precomputed/128px
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from mmengine.config import Config


SCRIPT_DIR = Path(__file__).resolve().parent
OPEN_SORA_ROOT = SCRIPT_DIR.parent.parent
if str(OPEN_SORA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SORA_ROOT))

from opensora.registry import MODELS, build_module
from opensora.models.vae.wan_video_vae import build_multiview_wan_video_vae  # noqa: F401
from opensora.utils.nersemble_bucket import resolve_nersemble_bucket
from opensora.utils.misc import to_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Wan VAE latents for EMO-1-shout+laugh.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py",
        help="Training config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/piado/scratch/precomputed/128px",
        help="Directory where latent .pt files are written.",
    )
    parser.add_argument(
        "--expression",
        type=str,
        default="EMO-1-shout+laugh",
        help="Expression folder / file stem to match.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=[None, "bf16", "fp16", "fp32"],
        help="Override dtype. If omitted, uses cfg.dtype.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, precompute only first N files.",
    )
    return parser.parse_args()


def downsample_video_tensor(x: torch.Tensor, target_h: int, target_w: int, target_t: int) -> torch.Tensor:
    """Match training bucket spatiotemporal shape."""
    if x.dim() != 6:
        raise ValueError(f"Expected [B,V,C,T,H,W], got shape {tuple(x.shape)}")
    b, v, c, t, h, w = x.shape
    if t != target_t:
        idx = torch.linspace(0, t - 1, target_t, device=x.device).long()
        x = x.index_select(3, idx)
        t = target_t
    if (h, w) != (target_h, target_w):
        x_flat = x.reshape(b * v * t, c, h, w)
        x_flat = F.interpolate(x_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = x_flat.reshape(b, v, c, t, target_h, target_w)
    return x


def load_pt_as_vcthw(path: Path) -> torch.Tensor:
    """Load .pt and normalize to [V,C,T,H,W], clamped to [0,1]."""
    data = torch.load(str(path), map_location="cpu")
    if isinstance(data, dict):
        if "video" in data:
            video = data["video"]
        elif "tensor" in data:
            video = data["tensor"]
        else:
            video = None
            for value in data.values():
                if isinstance(value, torch.Tensor) and value.dim() >= 4:
                    video = value
                    break
            if video is None:
                raise ValueError(f"No video-like tensor found in {path}")
    elif isinstance(data, torch.Tensor):
        video = data
    else:
        raise ValueError(f"Unsupported tensor container in {path}: {type(data)}")

    # Convert to [V,C,T,H,W].
    if video.dim() == 5:
        # [V,T,C,H,W] -> [V,C,T,H,W]
        if video.shape[1] > video.shape[2]:
            video = video.permute(0, 2, 1, 3, 4)
    elif video.dim() == 4:
        # [T,C,H,W] or [C,T,H,W] -> add view dim
        if video.shape[0] > video.shape[1]:
            video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
        else:
            video = video.unsqueeze(0)
    elif video.dim() == 6:
        # [B,V,T,C,H,W] or [B,V,C,T,H,W] -> take first batch
        video = video[0]
        if video.dim() == 5 and video.shape[1] > video.shape[2]:
            video = video.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected video dims in {path}: shape={tuple(video.shape)}")

    return video.clamp(0.0, 1.0)


def discover_expression_files(data_root: Path, expression: str) -> list[Path]:
    # Expected layout: data_root/pXXX/<expression>/<expression>.pt
    files = sorted(data_root.glob(f"p*/{expression}/{expression}.pt"))
    # Fallback: any .pt containing expression in path.
    if not files:
        files = sorted([p for p in data_root.glob("**/*.pt") if expression in str(p)])
    return files


def path_digest(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def encode_video(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    # x is [B,V,C,T,H,W] on model device.
    if getattr(model, "use_crossview_encoder", False):
        if not hasattr(model, "crossview_vae"):
            raise RuntimeError("Model reports crossview mode but has no crossview_vae.")
        scale = [
            torch.zeros(model.z_dim, dtype=x.dtype, device=x.device),
            torch.ones(model.z_dim, dtype=x.dtype, device=x.device),
        ]
        mu, logvar = model.crossview_vae.encode(x, scale)
        return {"mu": mu.detach().cpu(), "logvar": logvar.detach().cpu()}
    # Non-crossview fallback.
    z = model.encode(x)
    return {"z": z.detach().cpu()}


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config, lazy_import=False)

    # Resolve DATA_ROOT from bucket config exactly like training config.
    bucket_cfg = cfg.get("bucket_config", None)
    processed_base = cfg.get("nersemble_processed_base", None)
    if bucket_cfg is None:
        raise ValueError("Config has no bucket_config; cannot resolve DATA_ROOT.")
    resolved = resolve_nersemble_bucket(bucket_cfg, processed_base=processed_base)
    data_root = Path(resolved["data_root"])
    target_hw = tuple(resolved["train_target_hw"])
    target_t = int(resolved["train_target_frames"])

    files = discover_expression_files(data_root, args.expression)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .pt files found for expression '{args.expression}' under {data_root}")

    device = torch.device(args.device)
    dtype_name = args.dtype if args.dtype is not None else cfg.get("dtype", "bf16")
    dtype = to_torch_dtype(dtype_name)

    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_target_range = cfg.get("vae_target_range", None)
    expected_views = int(cfg.model.get("view_in", 2))

    print(f"[precompute] data_root={data_root}")
    print(f"[precompute] expression={args.expression}, files={len(files)}")
    print(f"[precompute] output_dir={out_dir}")
    print(f"[precompute] device={device}, dtype={dtype_name}")

    with torch.no_grad():
        for idx, pt_path in enumerate(files, start=1):
            video_vcthw = load_pt_as_vcthw(pt_path)  # [V,C,T,H,W]
            if video_vcthw.shape[0] != expected_views:
                print(
                    f"[skip {idx}/{len(files)}] {pt_path} has V={video_vcthw.shape[0]}, expected {expected_views}"
                )
                continue
            x = video_vcthw.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)  # [B,V,C,T,H,W]
            x = downsample_video_tensor(x, target_h=int(target_hw[0]), target_w=int(target_hw[1]), target_t=target_t)
            if x.dim() == 6 or vae_target_range == "[-1,1]":
                x = 2.0 * x - 1.0

            latents = encode_video(model, x)
            rel = pt_path.relative_to(data_root)
            safe_name = f"{rel.parent.parent.name}__{rel.parent.name}__{path_digest(pt_path)}.pt"
            out_path = out_dir / safe_name
            payload = {
                "source_path": str(pt_path),
                "relative_path": str(rel),
                "input_shape_bvcthw": tuple(x.shape),
                "target_hw": tuple(int(v) for v in target_hw),
                "target_t": int(target_t),
                "vae_target_range": "[-1,1]" if (x.min().item() < 0.0) else "[0,1]",
                **latents,
            }
            torch.save(payload, out_path)
            print(f"[ok {idx}/{len(files)}] saved {out_path}")

    print("[done] latent precompute complete.")


if __name__ == "__main__":
    main()
