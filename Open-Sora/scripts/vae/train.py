# START with: python3 scripts/vae/train.py /home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py

# resume training with: python scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py \
#  --load /home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs/260228_150719-.../epoch0-global_step200

import gc
import os
import random
import subprocess
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
import torch.nn.functional as F

# Add Open-Sora directory to Python path so we can import opensora
# The script is in Open-Sora/scripts/vae/, so we need to go up two levels
import sys
script_dir = Path(__file__).resolve().parent  # scripts/vae/
open_sora_root = script_dir.parent.parent  # Open-Sora/
if str(open_sora_root) not in sys.path:
    sys.path.insert(0, str(open_sora_root))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()

# Workaround for flash_attn binary incompatibility
# The issue: flash_attn is imported through colossalai -> peft -> transformers -> flash_attn
# and there's a binary mismatch with PyTorch/CUDA versions
# 
# The error shows: flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so has undefined symbols
# This means it was compiled against a different PyTorch/CUDA version than what's loaded
#
# Since VAE training doesn't need flash_attn, the best solution is to uninstall it:
#   pip uninstall flash-attn
#
# If you can't uninstall it, we try to disable it via environment variables
# (but this may not work if the binary is loaded directly)

# Try to disable flash_attn in transformers before any imports
os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTN", "1")

# Check if flash_attn is installed and warn the user
try:
    import importlib.util
    flash_attn_spec = importlib.util.find_spec("flash_attn")
    if flash_attn_spec is not None:
        print(
            "WARNING: flash_attn is installed and may cause import errors.\n"
            "VAE training doesn't need flash_attn. Recommended fix:\n"
            "  pip uninstall flash-attn\n"
            "If you need flash_attn for other models, reinstall it to match your PyTorch version:\n"
            "  pip install --force-reinstall flash-attn --no-build-isolation\n"
            "Continuing anyway, but if you see import errors, uninstall flash_attn first.\n"
        )
except Exception:
    pass  # If check fails, continue anyway

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
from colossalai.booster import Booster
from colossalai.utils import set_seed
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import tqdm

import torchvision
import numpy as np

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.models.vae.losses import DiscriminatorLoss, GeneratorLoss, VAELoss
from opensora.models.vae.utils import DiagonalGaussianDistribution
from opensora.models.vae.wan_video_vae import build_multiview_wan_video_vae  # Register multi-view VAE model
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import CheckpointIO, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    Timer,
    all_reduce_sum,
    is_log_process,
    log_model_params,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.train import create_colossalai_plugin, set_lr, set_warmup_steps, setup_device, update_ema

torch.backends.cudnn.benchmark = True

WAIT = 1
WARMUP = 10
ACTIVE = 20

my_schedule = schedule(
    wait=WAIT,  # number of warmup steps
    warmup=WARMUP,  # number of warmup steps with profiling
    active=ACTIVE,  # number of active steps with profiling
)


# Evaluation Metrics and Visualization Functions


def downsample_video_tensor(x: torch.Tensor, target_h: int = 64, target_w: int = 64, target_t: int = 9) -> torch.Tensor:
    """
    Downsample a batch of videos to fixed spatial and temporal resolution.

    Supports:
    - [B, V, C, T, H, W] (multi-view)
    - [B, C, T, H, W] (single-view)
    Other shapes are returned unchanged.
    """
    if x.dim() == 6:
        b, v, c, t, h, w = x.shape
        if t != target_t:
            idx = torch.linspace(0, t - 1, target_t, device=x.device).long()
            x = x.index_select(3, idx)
            t = target_t
        x_flat = x.view(b * v * t, c, h, w)
        x_flat = F.interpolate(x_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = x_flat.view(b, v, c, t, target_h, target_w)
    elif x.dim() == 5:
        b, c, t, h, w = x.shape
        if t != target_t:
            idx = torch.linspace(0, t - 1, target_t, device=x.device).long()
            x = x.index_select(2, idx)
            t = target_t
        x_perm = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x_flat = x_perm.reshape(b * t, c, h, w)
        x_flat = F.interpolate(x_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x_perm = x_flat.view(b, t, c, target_h, target_w)
        x = x_perm.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    return x


def compute_psnr(img1, img2, max_val=1.0):
    """
    Compute PSNR between two images/videos.
    Measures the ratio between the maximum possible signal value and the power of the noise (error).
    Higher PSNR values = better reconstruction quality.
    
    Args:
        img1: Original image/video tensor, shape [..., C, H, W] or [..., T, C, H, W]
        img2: Reconstructed image/video tensor, same shape as img1
        max_val: Maximum possible pixel value (1.0 for normalized [0,1] images)
    
    Returns:
        PSNR value in dB (decibels)
    """
    # Compute MSE (Mean Squared Error) across all dimensions
    mse = torch.mean((img1 - img2) ** 2)
    
    # Avoid log(0) by clamping MSE to a small epsilon
    mse = torch.clamp(mse, min=1e-10)
    
    # PSNR formula: 20 * log10(max_val / sqrt(MSE))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return psnr.item()


def compute_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Compute SSIM between two images/videos.
    
    SSIM measures structural similarity, considering luminance, contrast, and structure.
    More perceptually aligned than MSE/PSNR. Returns values in [-1, 1], where 1 means identical images.
    
    (Simplified SSIM implementation, use torchmetrics or a more complete implementation later)
    
    Args:
        img1: Original image/video tensor
        img2: Reconstructed image/video tensor
        window_size: Size of the Gaussian window for local SSIM computation
        max_val: Maximum possible pixel value
    
    Returns:
        SSIM value (typically in [0, 1] for normalized images)
    """
    # Flatten spatial and temporal dimensions for easier computation
    # We'll compute SSIM per-channel and average
    if img1.dim() == 5:  # [B, C, T, H, W] or [B, V, C, T, H, W]
        # Reshape to [B*T, C, H, W] or similar
        if img1.dim() == 6:  # Multi-view: [B, V, C, T, H, W]
            b, v, c, t, h, w = img1.shape
            img1 = img1.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
            img2 = img2.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        else:  # Single-view: [B, C, T, H, W]
            b, c, t, h, w = img1.shape
            img1 = img1.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
            img2 = img2.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
    
    # For simplicity, compute a per-pixel SSIM approximation
    # Full SSIM with Gaussian window would be more accurate but computationally expensive
    # This gives a reasonable approximation for training monitoring
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    
    return ssim_map.mean().item()


def compute_mse(img1, img2):
    """
    Compute MSE between two images/videos.
    
    Simple pixel-wise error metric. Lower = better.
    
    Args:
        img1: Original image/video tensor
        img2: Reconstructed image/video tensor
    
    Returns:
        MSE value
    """
    return torch.mean((img1 - img2) ** 2).item()


def compute_metrics(x_orig, x_rec):
    """
    Compute all reconstruction metrics for a batch.

    Args:
        x_orig: Original video tensor, shape [B, C, T, H, W] or [B, V, C, T, H, W]
        x_rec: Reconstructed video tensor, same shape as x_orig
    
    Returns:
        Dictionary with metric names and values (both per-sample and averaged)
    """
    # Ensure values are in [0, 1] range for proper metric computation
    x_orig = torch.clamp(x_orig, 0, 1)
    x_rec = torch.clamp(x_rec, 0, 1)
    
    # Ensure temporal dimensions match before flattening; some sequences can be shorter.
    if x_orig.dim() == 6 and x_rec.dim() == 6:  # Multi-view: [B, V, C, T, H, W]
        b_o, v_o, c_o, t_o, h_o, w_o = x_orig.shape
        b_r, v_r, c_r, t_r, h_r, w_r = x_rec.shape
        # Align along time; assume same B,V,C,H,W, but allow T to differ by truncating to min
        t = min(t_o, t_r)
        x_orig_use = x_orig[:, :, :, :t, :, :]
        x_rec_use = x_rec[:, :, :, :t, :, :]
        b, v, c, _, h, w = x_orig_use.shape
        x_orig_flat = x_orig_use.view(b * v, c, t, h, w)
        x_rec_flat = x_rec_use.view(b * v, c, t, h, w)
    else:  # Single-view: [B, C, T, H, W] (or already flattened)
        if x_orig.dim() == 5 and x_rec.dim() == 5:
            b_o, c_o, t_o, h_o, w_o = x_orig.shape
            b_r, c_r, t_r, h_r, w_r = x_rec.shape
            t = min(t_o, t_r)
            x_orig_flat = x_orig[:, :, :t, :, :]
            x_rec_flat = x_rec[:, :, :t, :, :]
        else:
            # Fallback: assume shapes already match
            x_orig_flat = x_orig
            x_rec_flat = x_rec
    
    # Compute metrics per sample in the batch
    batch_size = x_orig_flat.shape[0]
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    for i in range(batch_size):
        psnr_val = compute_psnr(x_orig_flat[i], x_rec_flat[i])
        ssim_val = compute_ssim(x_orig_flat[i], x_rec_flat[i])
        mse_val = compute_mse(x_orig_flat[i], x_rec_flat[i])
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        mse_values.append(mse_val)
    
    # Return both per-sample and averaged metrics
    return {
        "psnr": np.mean(psnr_values),
        "psnr_std": np.std(psnr_values),
        "ssim": np.mean(ssim_values),
        "ssim_std": np.std(ssim_values),
        "mse": np.mean(mse_values),
        "mse_std": np.std(mse_values),
        "psnr_per_sample": psnr_values,  # For detailed analysis
        "ssim_per_sample": ssim_values,
    }


def compute_metrics_per_frame(x_orig, x_rec, max_val=1.0):
    """
    Compute PSNR, SSIM, and MSE per frame (no temporal averaging).

    Args:
        x_orig: Original video tensor, shape [B, C, T, H, W] or [B, V, C, T, H, W]
        x_rec: Reconstructed video tensor, same shape as x_orig
        max_val: Maximum possible pixel value (1.0 for [0,1] range)

    Returns:
        Dictionary with lists of per-frame metric values:
            - psnr_per_frame: list[float] of length T
            - ssim_per_frame: list[float] of length T
            - mse_per_frame: list[float] of length T
    """
    # Clamp to valid range for metrics (assume inputs already roughly in [0,1])
    x_orig = torch.clamp(x_orig, 0, 1)
    x_rec = torch.clamp(x_rec, 0, 1)

    # Align temporal dimension and flatten views if present
    if x_orig.dim() == 6 and x_rec.dim() == 6:  # [B, V, C, T, H, W]
        b_o, v_o, c_o, t_o, h_o, w_o = x_orig.shape
        b_r, v_r, c_r, t_r, h_r, w_r = x_rec.shape
        t = min(t_o, t_r)
        x_orig_use = x_orig[:, :, :, :t, :, :]
        x_rec_use = x_rec[:, :, :, :t, :, :]
        b, v, c, _, h, w = x_orig_use.shape
        # Flatten batch and view: [B, V, C, T, H, W] -> [B*V, C, T, H, W]
        x_orig_flat = x_orig_use.view(b * v, c, t, h, w)
        x_rec_flat = x_rec_use.view(b * v, c, t, h, w)
    elif x_orig.dim() == 5 and x_rec.dim() == 5:  # [B, C, T, H, W]
        b_o, c_o, t_o, h_o, w_o = x_orig.shape
        b_r, c_r, t_r, h_r, w_r = x_rec.shape
        t = min(t_o, t_r)
        x_orig_flat = x_orig[:, :, :t, :, :]
        x_rec_flat = x_rec[:, :, :t, :, :]
    else:
        # Fallback: assume shapes already match and treat third dim as time
        x_orig_flat = x_orig
        x_rec_flat = x_rec
        t = x_orig_flat.shape[2]

    # Vectorized MSE and PSNR per frame (average over batch, channels, and spatial dims)
    diff = x_orig_flat - x_rec_flat  # [B', C, T, H, W]
    mse_per_frame = torch.mean(diff ** 2, dim=(0, 1, 3, 4))  # [T]

    eps = 1e-10
    mse_safe = torch.clamp(mse_per_frame, min=eps)
    psnr_per_frame = 10.0 * torch.log10((max_val ** 2) / mse_safe)  # [T]

    # SSIM per frame: reuse compute_ssim on single-frame "videos"
    ssim_per_frame = []
    for frame_idx in range(t):
        # Shape [B', C, H, W] -> [B', C, 1, H, W] so compute_ssim can treat it as a 1-frame video
        x_o_frame = x_orig_flat[:, :, frame_idx, :, :].unsqueeze(2)
        x_r_frame = x_rec_flat[:, :, frame_idx, :, :].unsqueeze(2)
        ssim_val = compute_ssim(x_o_frame, x_r_frame, max_val=max_val)
        ssim_per_frame.append(ssim_val)

    return {
        "psnr_per_frame": psnr_per_frame.detach().cpu().tolist(),
        "ssim_per_frame": ssim_per_frame,
        "mse_per_frame": mse_per_frame.detach().cpu().tolist(),
    }


def _draw_label_strip(labels, col_widths, strip_height=28, font_size=14):
    """Draw a horizontal strip with one label per column. Returns RGB numpy array [strip_height, total_width, 3]."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    total_width = sum(col_widths)
    img = Image.new("RGB", (total_width, strip_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for i, text in enumerate(labels):
        w = col_widths[i]
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 80, strip_height  # fallback: don't center
        tx = x + max(0, (w - tw) // 2)
        ty = (strip_height - th) // 2
        draw.text((tx, ty), text, fill=(220, 220, 220), font=font)
        x += w
    return np.array(img)


def create_visualization_grid(x_orig, x_rec, num_samples=4, num_frames=None, value_range="[0,1]"):
    """
    Create a visualization where:
    - All (downsampled) temporal frames are laid out horizontally (time left -> right),
    - An arrow indicates temporal direction,
    - Rows correspond to:
        * multi-view:  View 1 input, View 2 input, View 1 recon., View 2 recon.
        * single-view: Input, Reconstruction
    - A small metric panel below shows per-frame PSNR/SSIM/MSE curves.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        Image = None

    is_multiview = x_orig.dim() == 6
    num_views = x_orig.shape[1] if is_multiview else 1

    # Compute per-frame metrics on CPU tensors in [0,1]
    x_o_metrics = x_orig
    x_r_metrics = x_rec
    if value_range == "[-1,1]":
        x_o_metrics = 0.5 * (x_o_metrics + 1.0)
        x_r_metrics = 0.5 * (x_r_metrics + 1.0)
    metrics = compute_metrics_per_frame(x_o_metrics, x_r_metrics, max_val=1.0)

    # Move to numpy for rendering
    x_orig = x_o_metrics.detach().float().cpu().numpy()
    x_rec = x_r_metrics.detach().float().cpu().numpy()
    x_orig = np.clip(x_orig, 0.0, 1.0).astype(np.float32)
    x_rec = np.clip(x_rec, 0.0, 1.0).astype(np.float32)

    if is_multiview:
        b, v, c, t, h, w = x_orig.shape
    else:
        b, c, t, h, w = x_orig.shape
        v = 1

    num_samples = min(num_samples, b)
    # Use all available (downsampled) frames by default
    if num_frames is None:
        num_frames = t
    num_frames = min(num_frames, t)
    frame_indices = np.linspace(0, t - 1, num_frames).astype(int)

    # Row labels:
    if is_multiview and num_views >= 2:
        row_labels = [
            "View 1 – input",
            "View 2 – input",
            "View 1 – reconstr.",
            "View 2 – reconstr.",
        ]
    else:
        row_labels = ["Input", "Reconstruction"]

    images = []
    for sample_idx in range(num_samples):
        # Build row-major grid: rows = views × {input,recon}, cols = time frames
        rows = []
        if is_multiview and num_views >= 2:
            # Inputs per view
            for view_idx in range(num_views):
                frames = []
                for frame_idx in frame_indices:
                    orig_frame = x_orig[sample_idx, view_idx, :, frame_idx, :, :]
                    orig_frame = (
                        np.transpose(orig_frame, (1, 2, 0))
                        if c == 3
                        else np.repeat(orig_frame[0:1].transpose(1, 2, 0), 3, axis=-1)
                    )
                    frames.append(orig_frame)
                rows.append(np.concatenate(frames, axis=1))
            # Reconstructions per view
            for view_idx in range(num_views):
                frames = []
                for frame_idx in frame_indices:
                    rec_frame = x_rec[sample_idx, view_idx, :, frame_idx, :, :]
                    rec_frame = (
                        np.transpose(rec_frame, (1, 2, 0))
                        if c == 3
                        else np.repeat(rec_frame[0:1].transpose(1, 2, 0), 3, axis=-1)
                    )
                    frames.append(rec_frame)
                rows.append(np.concatenate(frames, axis=1))
        else:
            # Single-view: two rows (input, recon)
            frames_orig = []
            frames_rec = []
            for frame_idx in frame_indices:
                o = x_orig[sample_idx, :, frame_idx, :, :]
                r = x_rec[sample_idx, :, frame_idx, :, :]
                o = (
                    np.transpose(o, (1, 2, 0))
                    if c == 3
                    else np.repeat(o[0:1].transpose(1, 2, 0), 3, axis=-1)
                )
                r = (
                    np.transpose(r, (1, 2, 0))
                    if c == 3
                    else np.repeat(r[0:1].transpose(1, 2, 0), 3, axis=-1)
                )
                frames_orig.append(o)
                frames_rec.append(r)
            rows.append(np.concatenate(frames_orig, axis=1))
            rows.append(np.concatenate(frames_rec, axis=1))

        # Convert to RGB uint8 image
        grid = np.clip(np.concatenate(rows, axis=0), 0.0, 1.0)
        grid = (grid * 255.0).astype(np.uint8)

        # Convert to PIL for annotations if available
        if Image is not None:
            #img = Image.fromarray(grid)
            #draw = ImageDraw.Draw(img)
            base_img = Image.fromarray(grid)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
                )
            except Exception:
                font = ImageFont.load_default()

            H, W = grid.shape[0], grid.shape[1]
            row_h = H // len(row_labels)

            """
            # Left-side row labels
            for i, label in enumerate(row_labels):
                y_center = i * row_h + row_h // 2
                draw.text(
                    (5, y_center - 7),
                    label,
                    fill=(255, 255, 255),
                    font=font,
                )

            # Temporal arrow on top
            arrow_y = 5
            start_x = 80
            end_x = W - 10
            draw.line((start_x, arrow_y, end_x, arrow_y), fill=(255, 255, 0), width=2)
            draw.polygon(
            """
            # --- Top band with time arrow (white background, black text) ---
            top_h = 30
            top = Image.new("RGB", (W, top_h), color=(255, 255, 255))
            tdraw = ImageDraw.Draw(top)
            arrow_y = top_h // 2
            start_x = 60
            end_x = W - 20
            tdraw.line((start_x, arrow_y, end_x, arrow_y), fill=(0, 0, 0), width=2)
            tdraw.polygon(
                [
                    (end_x, arrow_y),
                    (end_x - 8, arrow_y - 4),
                    (end_x - 8, arrow_y + 4),
                ],
                #fill=(255, 255, 0),
                fill=(0, 0, 0),
            )
            #draw.text(
            #    (start_x, arrow_y + 4),
            tdraw.text(
                (10, arrow_y - 7),
                "time ",
                #fill=(255, 255, 0),
                fill=(0, 0, 0),
                font=font,
            )

            # Metric panel below: simple line plots of per-frame PSNR/SSIM/MSE
                        # --- Left band with row labels (white background, black text) ---
            left_w = 160
            left = Image.new("RGB", (left_w, H), color=(255, 255, 255))
            ldraw = ImageDraw.Draw(left)
            for i, label in enumerate(row_labels):
                y_center = i * row_h + row_h // 2
                ldraw.text(
                    (10, y_center - 7),
                    label,
                    fill=(0, 0, 0),
                    font=font,
                )

            # --- Metric panel below (white background, numeric per-frame values) ---
            psnr = np.array(metrics["psnr_per_frame"], dtype=np.float32)
            ssim = np.array(metrics["ssim_per_frame"], dtype=np.float32)
            mse = np.array(metrics["mse_per_frame"], dtype=np.float32)

            num_t = len(psnr)
            panel_h = 50
            panel = Image.new("RGB", (W + left_w, panel_h), color=(255, 255, 255))
            pdraw = ImageDraw.Draw(panel)

            # Per-frame metrics stacked under each temporal frame (small font)
            try:
                small_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
                )
            except Exception:
                small_font = font

            # Width of a single temporal frame on the grid
            frame_w = max(1, W // max(1, num_frames))
            for i in range(num_t):
                x_center = left_w + int((i + 0.5) * frame_w)
                y0 = 4
                txts = [
                    f"P={psnr[i]:.1f}",
                    f"S={ssim[i]:.3f}",
                    f"M={mse[i]:.4f}",
                ]
                for j, txt in enumerate(txts):
                    try:
                        bbox = pdraw.textbbox((0, 0), txt, font=small_font)
                        w_txt = bbox[2] - bbox[0]
                        h_txt = bbox[3] - bbox[1]
                    except Exception:
                        w_txt, h_txt = 40, 10
                    pdraw.text(
                        (x_center - w_txt // 2, y0 + j * (h_txt + 1)),
                        txt,
                        fill=(0, 0, 0),
                        font=small_font,
                    )


            # _plot_curve(psnr, (0, 200, 0))
            # _plot_curve(ssim, (0, 150, 255))
            # _plot_curve(mse, (255, 80, 80))

            # --- Compose final image ---
            main = Image.new("RGB", (W + left_w, H), color=(255, 255, 255))
            main.paste(left, (0, 0))
            main.paste(base_img, (left_w, 0))


            # pdraw.text((10, 2), "PSNR", fill=(0, 200, 0), font=font)
            # pdraw.text((70, 2), "SSIM", fill=(0, 150, 255), font=font)
            # pdraw.text((130, 2), "MSE", fill=(255, 80, 80), font=font)

            combined = Image.new(
                "RGB", (W + left_w, top_h + H + panel_h), color=(255, 255, 255)
            )
            combined.paste(top, (0, 0))
            combined.paste(main, (0, top_h))
            combined.paste(panel, (0, top_h + H))


            # Combine main grid and metric panel
            # combined = Image.new("RGB", (W, H + panel_h), color=(0, 0, 0))
            # combined.paste(img, (0, 0))
            # combined.paste(panel, (0, H))
            combined_grid = np.array(combined)
        else:
            combined_grid = grid

        caption = f"Sample {sample_idx + 1}"
        images.append(wandb.Image(combined_grid, caption=caption))
    return images


def evaluate_model(model, dataloader, device, dtype, num_eval_samples=32, 
                   view_flatten_in_loss=True, use_ema=False, value_range="[0,1]"):
    """
    Run a full evaluation pass over the dataset (or a subset).
    
    Args:
        model: The VAE model to evaluate (can be booster-wrapped or unwrapped)
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        dtype: Data type for evaluation
        num_eval_samples: Number of samples to evaluate (for speed)
        view_flatten_in_loss: Whether to flatten views for loss computation
        use_ema: Whether to use EMA model if available
        value_range: "[0,1]" or "[-1,1]" - if "[-1,1]", scale batch to [-1,1] before model and use for vis
    Returns:
        Dictionary with aggregated metrics and visualization images
    """
    # Handle both wrapped (booster) and unwrapped models
    # Booster-wrapped models have a .module attribute
    if hasattr(model, "module"):
        model.module.eval()
    else:
        model.eval()
    
    all_metrics = {
        "psnr": [],
        "ssim": [],
        "mse": [],
    }
    
    visualization_samples = []
    num_collected = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_collected >= num_eval_samples:
                break
            
            x = batch["video"].to(device, dtype)
            is_multiview = x.dim() == 6
            if value_range == "[-1,1]":
                x = 2.0 * x - 1.0
            
            # Forward pass
            x_rec, posterior, z = model(x)
            
            # Handle posterior wrapping if needed
            if isinstance(posterior, (tuple, list)) and len(posterior) == 2:
                posterior = DiagonalGaussianDistribution(torch.cat(posterior, dim=1))
            
            # Compute metrics
            metrics = compute_metrics(x, x_rec)
            
            # Accumulate metrics
            all_metrics["psnr"].extend(metrics["psnr_per_sample"])
            all_metrics["ssim"].extend(metrics["ssim_per_sample"])
            all_metrics["mse"].append(metrics["mse"])
            
            # Collect samples for visualization (first batch only, or until we have enough)
            if len(visualization_samples) == 0:
                vis_images = create_visualization_grid(x, x_rec, num_samples=min(4, x.shape[0]), value_range=value_range)
                visualization_samples.extend(vis_images)
            
            num_collected += x.shape[0] if not is_multiview else x.shape[0] * x.shape[1]
    
    # Aggregate metrics
    aggregated = {
        "psnr_mean": np.mean(all_metrics["psnr"]),
        "psnr_std": np.std(all_metrics["psnr"]),
        "ssim_mean": np.mean(all_metrics["ssim"]),
        "ssim_std": np.std(all_metrics["ssim"]),
        "mse_mean": np.mean(all_metrics["mse"]),
        "mse_std": np.std(all_metrics["mse"]),
    }
    
    # Return to training mode
    if hasattr(model, "module"):
        model.module.train()
    else:
        model.train()
    
    return {
        "metrics": aggregated,
        "visualizations": visualization_samples,
    }


def evaluate_fixed_sequence_per_frame(model, dataset, index, device, dtype, vae_target_range=None):
    """
    Evaluate a single fixed sequence and compute per-frame metrics.

    This mirrors the pattern in `evaluate_model`: we switch the (possibly
    booster-wrapped) model into eval mode, run a no-grad forward, then
    restore the original training state.

    Args:
        model: VAE model (may be Booster-wrapped)
        dataset: PTVideoDataset (or similar) providing dicts with "video" and "path"
        index: Integer index into dataset.pt_files for the fixed sequence
        device: torch.device
        dtype: torch.dtype to run the model in
        vae_target_range: "[0,1]" or "[-1,1]" (controls input/output scaling)

    Returns:
        metrics: dict with per-frame lists for PSNR, SSIM, and MSE
    """
    # Remember original training state
    if hasattr(model, "module"):
        was_training = model.module.training
        model.module.eval()
    else:
        was_training = model.training
        model.eval()

    # Load single sequence from dataset (no batching in dataset itself)
    sample = dataset[index]
    x = sample["video"]  # [V, C, T, H, W] or [C, T, H, W]

    # Add batch dimension: [1, V, C, T, H, W] or [1, C, T, H, W]
    if x.dim() == 5:
        x = x.unsqueeze(0)
    elif x.dim() == 4:
        x = x.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected video shape for fixed-sequence eval: {x.shape}")

    x = x.to(device, dtype)
    is_multiview = x.dim() == 6

    # Prepare input for model according to training range
    x_input = x
    if is_multiview or vae_target_range == "[-1,1]":
        x_input = 2.0 * x - 1.0

    with torch.no_grad():
        # IMPORTANT: call the (possibly booster-wrapped) model exactly as in training
        x_rec, _, _ = model(x_input)

    # Map reconstructed output back to [0,1] for human-interpretable metrics
    if is_multiview or vae_target_range == "[-1,1]":
        x_rec_metrics = 0.5 * (x_rec + 1.0)
    else:
        x_rec_metrics = x_rec

    x_orig_metrics = x  # already in [0,1]

    metrics = compute_metrics_per_frame(x_orig_metrics, x_rec_metrics, max_val=1.0)

    # Attach original sample path if available so we can persist metrics alongside it.
    if "path" in sample:
        metrics["sample_path"] = sample["path"]

    # Restore original training/eval state
    if hasattr(model, "module"):
        if was_training:
            model.module.train()
        else:
            model.module.eval()
    else:
        if was_training:
            model.train()
        else:
            model.eval()

    return metrics


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin = (
        create_colossalai_plugin(
            plugin=plugin_type,
            dtype=cfg.get("dtype", "bf16"),
            grad_clip=cfg.get("grad_clip", 0),
            **plugin_config,
        )
        if plugin_type != "none"
        else None
    )
    booster = Booster(plugin=plugin)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
    )
    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        import os
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    



    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))
    # == optional val/held-out dataset for evaluation and reconstruction plots ==
    val_dataset = None
    if cfg.get("val_dataset") is not None:
        val_dataset = build_module(cfg.val_dataset, DATASETS)
        logger.info("Val (held-out) dataset contains %s samples.", len(val_dataset))

    # == pick fixed train/test sequences for per-frame metrics ==
    # We choose one sequence name that exists in both train and val (if possible),
    # and then fix one sample index for train and one for val with that name.
    fixed_train_index = None
    fixed_val_index = None
    fixed_seq_name = None

    def _seq_name_from_path(path):
        # path: .../pXXX/<sequence_name>/something.pt
        return os.path.basename(os.path.dirname(path))

    if hasattr(dataset, "pt_files"):
        train_seq_names = [_seq_name_from_path(p) for p in dataset.pt_files]
    else:
        train_seq_names = []

    if val_dataset is not None and hasattr(val_dataset, "pt_files"):
        val_seq_names = [_seq_name_from_path(p) for p in val_dataset.pt_files]
    else:
        val_seq_names = []

    if train_seq_names and val_seq_names:
        common_names = sorted(set(train_seq_names) & set(val_seq_names))
        if common_names:
            fixed_seq_name = common_names[0]
            fixed_train_index = train_seq_names.index(fixed_seq_name)
            fixed_val_index = val_seq_names.index(fixed_seq_name)
        else:
            # Fallback: use first sample from each split independently
            fixed_train_index = 0
            fixed_val_index = 0 if val_seq_names else None
            fixed_seq_name = train_seq_names[0]
    elif train_seq_names:
        fixed_train_index = 0
        fixed_val_index = None
        fixed_seq_name = train_seq_names[0]

    if fixed_seq_name is not None:
        logger.info(
            "Fixed per-frame eval sequence name: %s (train idx=%s, val idx=%s)",
            fixed_seq_name,
            fixed_train_index,
            fixed_val_index,
        )

    # == build dataloader ==
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build vae model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    log_model_params(model)

    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    vae_loss_fn = VAELoss(**cfg.vae_loss_config, device=device, dtype=dtype)

    # == build EMA model ==
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")

    # == build discriminator model ==
    use_discriminator = cfg.get("discriminator", None) is not None
    if use_discriminator:
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype).train()
        log_model_params(discriminator)
        generator_loss_fn = GeneratorLoss(**cfg.gen_loss_config)
        discriminator_loss_fn = DiscriminatorLoss(**cfg.disc_loss_config)

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer, num_steps_per_epoch=num_steps_per_epoch, epochs=cfg.get("epochs", 1000), **cfg.lr_scheduler
    )

    # == setup discriminator optimizer ==
    if use_discriminator:
        disc_optimizer = create_optimizer(discriminator, cfg.optim_discriminator)
        disc_lr_scheduler = create_lr_scheduler(
            optimizer=disc_optimizer,
            num_steps_per_epoch=num_steps_per_epoch,
            epochs=cfg.get("epochs", 1000),
            **cfg.disc_lr_scheduler,
        )
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", total_trainable)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )

    if use_discriminator:
        discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
            model=discriminator,
            optimizer=disc_optimizer,
            lr_scheduler=disc_lr_scheduler,
        )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    mixed_strategy = cfg.get("mixed_strategy", None)
    mixed_image_ratio = cfg.get("mixed_image_ratio", 0.0)
    # Multi-view input controls. These are opt-in and safe for single-view.
    view_flatten_in_loss = cfg.get("view_flatten_in_loss", True)
    view_flatten_in_disc = cfg.get("view_flatten_in_disc", True)
    # modulate mixed image ratio since we force rank 0 to be video
    num_ranks = dist.get_world_size()
    modulated_mixed_image_ratio = (
        num_ranks * mixed_image_ratio / (num_ranks - 1) if num_ranks > 1 else mixed_image_ratio
    )
    if is_log_process(plugin_type, plugin_config):
        print("modulated mixed image ratio:", modulated_mixed_image_ratio)

    start_epoch = start_step = log_step = acc_step = 0
    running_loss = dict(  # loss accumulated over config.log_every steps
        all=0.0,
        nll=0.0,
        nll_rec=0.0,
        nll_per=0.0,
        kl=0.0,
        gen=0.0,
        gen_w=0.0,
        disc=0.0,
        debug=0.0,
    )
    
    # Timing accumulators for bottleneck analysis
    # We track time spent in each major operation to identify where training is slow
    timing_stats = {
        "data_load": [],  # Time to load data from dataloader
        "forward": [],  # Time for model forward pass (encode + decode)
        "loss_compute": [],  # Time to compute losses (reconstruction, KL, perceptual)
        "discriminator": [],  # Time for discriminator forward (if used)
        "backward": [],  # Time for backward pass (gradient computation)
        "optimizer": [],  # Time for optimizer step (weight update)
        "total_step": [],  # Total time per training step
    }
    
    # Memory tracking (optional, can be expensive to log frequently)
    log_memory = cfg.get("log_memory", False)  # Set to True in config to enable
    if log_memory:
        timing_stats["memory_allocated"] = []
        timing_stats["memory_reserved"] = []

    def log_loss(name, loss, loss_dict, use_video):
        # only calculate loss for video
        if use_video == 0:
            loss.data = torch.tensor(0.0, device=device, dtype=dtype)
        all_reduce_sum(loss.data)
        num_video = torch.tensor(use_video, device=device, dtype=dtype)
        all_reduce_sum(num_video)
        loss_item = loss.item() / num_video.item()
        loss_dict[name] = loss_item
        running_loss[name] += loss_item

    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    # To resume training, run with: --load /path/to/epochX-global_stepY
    # (path to a checkpoint subdir that contains running_states.json and model/)
    if cfg.get("load", None) is not None:
        logger.info("Resuming: loading checkpoint from %s", cfg.load)
        start_epoch = cfg.get("start_epoch", None)
        start_step = cfg.get("start_step", None)
        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=(
                None if start_step is not None else sampler
            ),  # if specify start step, set last_micro_batch_access_index of a new sampler instead
        )
        if start_step is not None:
            # if start step exceeds data length, go to next epoch
            if start_step > num_steps_per_epoch:
                start_epoch = (
                    start_epoch + start_step // num_steps_per_epoch
                    if start_epoch is not None
                    else start_step // num_steps_per_epoch
                )
                start_step = start_step % num_steps_per_epoch
            sampler.set_step(start_step)

        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]

        if (
            use_discriminator
            and os.path.exists(os.path.join(cfg.load, "discriminator"))
            and not cfg.get("restart_disc", False)
        ):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            if cfg.get("load_optimizer", True):
                booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
                if disc_lr_scheduler is not None:
                    booster.load_lr_scheduler(disc_lr_scheduler, os.path.join(cfg.load, "disc_lr_scheduler"))
                if cfg.get("disc_lr", None) is not None:
                    set_lr(disc_optimizer, disc_lr_scheduler, cfg.disc_lr)

        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

        if cfg.get("lr", None) is not None:
            set_lr(optimizer, lr_scheduler, cfg.lr, cfg.get("initial_lr", None))

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.lr_scheduler.get("warmup_steps", None) is not None
            ), "you need to set lr_scheduler.warmup_steps in order to pass --update-warmup-steps True"
            set_warmup_steps(lr_scheduler, cfg.lr_scheduler.warmup_steps)
            if use_discriminator:
                assert (
                    cfg.disc_lr_scheduler.get("warmup_steps", None) is not None
                ), "you need to set disc_lr_scheduler.warmup_steps in order to pass --update-warmup-steps True"
                set_warmup_steps(disc_lr_scheduler, cfg.disc_lr_scheduler.warmup_steps)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)

    if cfg.get("freeze_layers", None) == "all":
        for param in model.module.parameters():
            param.requires_grad = False
        print("all layers frozen")

    # model.module.requires_grad_(False)
    # =======================================================
    # 5. Initialize wandb (after all setup is complete to avoid empty runs during debugging)
    # =======================================================
    if coordinator.is_master() and cfg.get("wandb", False):
        # Auto-generate wandb name if not explicitly set
        # Format: {model_name}_{dataset_name}_{timestamp} or use exp_name
        wandb_name = cfg.get("wandb_expr_name", None)
        if wandb_name is None:
            # Try to create a descriptive name from config
            model_name = cfg.model.get("model_name", "model")
            dataset_path = cfg.dataset.get("data_path", "dataset")
            # Extract meaningful part from dataset path (e.g., "p17_EXP-1-head")
            if isinstance(dataset_path, str):
                dataset_name = os.path.basename(dataset_path.rstrip("/"))
                if not dataset_name or dataset_name == "dataset":
                    dataset_name = os.path.basename(os.path.dirname(dataset_path))
            else:
                dataset_name = "data"
            # Use timestamp prefix from exp_name (first part before underscore)
            timestamp = exp_name.split("_")[0] if "_" in exp_name else exp_name[:8]
            wandb_name = f"{model_name}_{dataset_name}_{timestamp}"
        
        logger.info(f"Initializing wandb with run name: {wandb_name}")
        wandb.init(
            project=cfg.get("wandb_project", "vae"),
            name=wandb_name,
            config=cfg.to_dict(),
            dir=exp_dir,
        )
        # Use epochs as the x-axis in W&B charts.
        # We still keep `step=...` in `wandb.log` for monotonic ordering, but define all
        # training/eval metrics to use `epoch_float` as their step metric.
        wandb.define_metric("epoch_float")
        for pattern in (
            "loss/*",
            "metrics/*",
            "time/*",
            "memory/*",
            "lr",
            "global_grad_norm",
            "reconstructions",
            "val_reconstructions",
            "fixed_seq/*",
            "eval/*",
            "final_eval/*",
        ):
            wandb.define_metric(pattern, step_metric="epoch_float")
        logger.info("Wandb initialized successfully. Training will begin shortly...")
    

    # =======================================================
    # 6. training loop
    # =======================================================
    dist.barrier()
    accumulation_steps = int(cfg.get("accumulation_steps", 1))
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)
        random.seed(1024 + dist.get_rank())  # load vid/img for each rank

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataiter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            pbar_iter = iter(pbar)

            def fetch_data():
                step, batch = next(pbar_iter)
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                return batch, step, pinned_video

            batch_, step_, pinned_video_ = fetch_data()

            profiler_ctxt = (
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=my_schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                if cfg.get("profile", False)
                else nullcontext()
            )

            with profiler_ctxt:
                for _ in range(start_step, num_steps_per_epoch):
                    if cfg.get("profile", False) and _ == WARMUP + ACTIVE + WAIT + 3:
                        break

                    # Start timing the entire step - this helps us see overall throughput
                    step_start_time = time.time()
                    
                    # == load data ===
                    # Data loading can be a major bottleneck, especially with large videos
                    # We time this separately to see if we're GPU-starved (waiting for data)
                    data_load_start = time.time()
                    batch, step, pinned_video = batch_, step_, pinned_video_
                    
                    import sys
                    sys.stdout.flush()
                    if step + 1 < num_steps_per_epoch:
                        batch_, step_, pinned_video_ = fetch_data()
                    data_load_time = time.time() - data_load_start
                    timing_stats["data_load"].append(data_load_time)

                    # == log config ==
                    global_step = epoch * num_steps_per_epoch + step
                    actual_update_step = (global_step + 1) // accumulation_steps
                    log_step += 1
                    acc_step += 1
                    # Epoch coordinate for plotting in W&B.
                    # global_step is 0-indexed; +1 makes the first update land at ~1/steps_per_epoch.
                    epoch_float = (global_step + 1) / max(1, num_steps_per_epoch)

                    # == mixed strategy ==
                    x = batch["video"]
                    # Multi-view videos are shaped [B, V, C, T, H, W]. Single-view stays [B, C, T, H, W].
                    is_multiview = x.dim() == 6
                    time_dim = 3 if is_multiview else 2
                    t_length = x.size(time_dim)
                    use_video = 1
                    if mixed_strategy == "mixed_video_image":
                        if random.random() < modulated_mixed_image_ratio and dist.get_rank() != 0:
                            # NOTE: enable the first rank to use video
                            t_length = 1
                            use_video = 0
                    elif mixed_strategy == "mixed_video_random":
                        t_length = random.randint(1, x.size(time_dim))
                    # Slice time dimension regardless of view layout.
                    if is_multiview:
                        x = x[:, :, :, :t_length, :, :]
                    else:
                        x = x[:, :, :t_length, :, :]

                    # Wan (and many VAEs) expect targets in [-1, 1]; dataset is [0, 1].
                    # Always scale for multi-view so loss and decoder range match (fixes white output).
                    vae_target_range = cfg.get("vae_target_range", None)
                    if is_multiview or vae_target_range == "[-1,1]":
                        x = 2.0 * x - 1.0

                    # Optional: log latent shapes every N steps (from VAE encode/decode; 0 = off)
                    log_latent_shapes_every = int(cfg.get("log_latent_shapes_every", 0))
                    if log_latent_shapes_every and step % log_latent_shapes_every == 0:
                        _m = model.module if hasattr(model, "module") else model
                        if hasattr(_m, "debug_shapes"):
                            _m.debug_shapes = True
                    elif log_latent_shapes_every and hasattr(model, "module") and hasattr(model.module, "debug_shapes"):
                        model.module.debug_shapes = False
                    elif log_latent_shapes_every and hasattr(model, "debug_shapes"):
                        model.debug_shapes = False

                    # == forward pass ==
                    # Initialize loss_dict and vae_loss at the start of each step to ensure they're always available
                    loss_dict = {}  # loss at every step
                    vae_loss = torch.tensor(0.0, device=device, dtype=dtype)  # total VAE loss
                    
                    # The forward pass (encode + decode) is usually the most expensive operation
                    # We time it carefully to see if it's the bottleneck
                    forward_start = time.time()
                    with Timer("model", log=True) if cfg.get("profile", False) else nullcontext():
                        x_rec, posterior, z = model(x)
                    forward_time = time.time() - forward_start
                    timing_stats["forward"].append(forward_time)

                    # Step 0 diagnostic: catch constant (white) output early
                    if coordinator.is_master() and step == 0:
                        with torch.no_grad():
                            x_min, x_max = x.min().item(), x.max().item()
                            r_min, r_max = x_rec.min().item(), x_rec.max().item()
                        #print(f"[step 0] input  x   range: [{x_min:.3f}, {x_max:.3f}] (expect ~[-1, 1])")
                        #print(f"[step 0] output x_rec range: [{r_min:.3f}, {r_max:.3f}] (expect ~[-1, 1]; if constant -> white vis)")
                        if abs(r_max - r_min) < 0.01:
                            print("[step 0] WARNING: x_rec is nearly constant -> white output. Check: checkpoint loaded? expansion init?")

                    # High-level shape summary every N steps (0 = off)
                    # if coordinator.is_master() and log_latent_shapes_every and step % log_latent_shapes_every == 0:
                    #     sh = x.shape
                    #     if is_multiview:
                    #         print(f"[step {step}] input  [B,V,C,T,H,W] = {list(sh)}")
                    #     else:
                    #         print(f"[step {step}] input  [B,C,T,H,W] = {list(sh)}")
                    #     if z is not None:
                    #         print(f"[step {step}] latent z [B,Z,T',H',W'] = {list(z.shape)}")
                    #     print(f"[step {step}] output [B,V,C,T,H,W] = {list(x_rec.shape)}")
                    
                    # Log memory usage if enabled (can help identify OOM issues or memory leaks)
                    if log_memory and coordinator.is_master() and step % 10 == 0:
                        if torch.cuda.is_available():
                            timing_stats["memory_allocated"].append(torch.cuda.memory_allocated() / 1e9)  # GB
                            timing_stats["memory_reserved"].append(torch.cuda.memory_reserved() / 1e9)  # GB

                    # If a model returns (mu, logvar) instead of a posterior object,
                    # wrap it for downstream KL computation.
                    if isinstance(posterior, (tuple, list)) and len(posterior) == 2:
                        posterior = DiagonalGaussianDistribution(torch.cat(posterior, dim=1))
                    elif isinstance(posterior, torch.Tensor):
                        # If posterior is just a tensor (mu), create a dummy logvar
                        # This handles the case where the model returns (x_rec, mu, logvar) as tensors
                        mu = posterior
                        logvar = torch.zeros_like(mu)
                        posterior = DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

                    # Default loss inputs keep the original tensor shapes.
                    x_loss = x
                    x_rec_loss = x_rec
                    posterior_loss = posterior

                    # For multi-view, optionally flatten the view axis into batch so
                    # losses and discriminators can stay 5D (B,C,T,H,W).
                    if is_multiview and view_flatten_in_loss:
                        b, v, c, t, h, w = x.shape
                        # x is [B, V, C, T, H, W], flatten to [B*V, C, T, H, W]
                        x_loss = x.view(b * v, c, t, h, w)
                        x_rec_loss = x_rec.view(b * v, c, t, h, w)
                        # CRITICAL: The posterior from MultiViewWanVideoVAE is already [B, 2*C, T, H, W]
                        # (no view dimension because views are compressed), so we need to replicate it
                        # for each flattened view to match the batch size
                        if hasattr(posterior, "parameters"):
                            post_shape = posterior.parameters.shape
                            post_dims = posterior.parameters.dim()
                            
                            if post_dims == 5:
                                # Shape is [B, 2*C, T, H, W], need to replicate for each view
                                post_b, post_c, post_t, post_h, post_w = post_shape
                                if post_b == b:  # Original batch size, need to replicate for views
                                    # Repeat posterior for each view: [B, 2*C, T, H, W] -> [B*V, 2*C, T, H, W]
                                    params = posterior.parameters.repeat(v, 1, 1, 1, 1)
                                    posterior_loss = DiagonalGaussianDistribution(
                                        params, deterministic=posterior.deterministic
                                    )
                                elif post_b == b * v:  # Already the right size
                                    posterior_loss = posterior
                                else:
                                    # Unexpected batch size, use as-is
                                    posterior_loss = posterior
                            elif post_dims == 6:
                                # Has view dimension, need to check actual shape
                                post_b, post_v, post_c, post_t, post_h, post_w = post_shape
                                # Flatten view dimension into batch: [B, V, C, T, H, W] -> [B*V, C, T, H, W]
                                # Use the posterior's own dimensions, not the input video dimensions
                                params = posterior.parameters.view(post_b * post_v, post_c, post_t, post_h, post_w)
                                posterior_loss = DiagonalGaussianDistribution(
                                    params, deterministic=posterior.deterministic
                                )
                            else:
                                # Unexpected shape, use as-is
                                posterior_loss = posterior
                        else:
                            posterior_loss = posterior

                    if cfg.get("profile", False):
                        profiler_ctxt.step()

                    if cache_pin_memory:
                        dataiter.remove_cache(pinned_video)

                    # == loss computation ==
                    # Loss computation includes reconstruction, KL divergence, and perceptual loss
                    # Perceptual loss can be expensive (uses VGG features), so we time it
                    loss_start = time.time()
                    # Reset vae_loss (it was initialized at the start of the step)
                    vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                    # loss_dict is already initialized at the start of the step

                    ret = vae_loss_fn(x_loss, x_rec_loss, posterior_loss)

                    # View consistency loss: encourage different views to be similar
                    # This helps prevent the model from learning to ignore views
                    view_loss = 0.0
                    if is_multiview and x_rec.shape[1] > 1:
                        # Compute MSE between consecutive views
                        view_losses = []
                        for i in range(x_rec.shape[1] - 1):
                            view_losses.append(F.mse_loss(x_rec[:, i], x_rec[:, i + 1]))
                        view_loss = sum(view_losses) / len(view_losses)
                    
                    # Add view consistency loss to total loss
                    view_consistency_weight = cfg.vae_loss_config.get("view_consistency_weight", 0.01)
                    vae_loss = vae_loss + view_consistency_weight * view_loss
                    loss_dict["view_loss"] = view_loss.item()

                    nll_loss = ret["nll_loss"]
                    kl_loss = ret["kl_loss"]
                    recon_loss = ret["recon_loss"]
                    perceptual_loss = ret["perceptual_loss"]
                    vae_loss += nll_loss + kl_loss
                    loss_time = time.time() - loss_start
                    timing_stats["loss_compute"].append(loss_time)

                    # == generator loss ==
                    # Discriminator forward pass can be expensive, especially with 3D convolutions
                    # We time it separately to see if it's slowing down training
                    if use_discriminator:
                        disc_start = time.time()
                        # turn off grad update for disc
                        discriminator.requires_grad_(False)
                        disc_input = x_rec
                        # Discriminator expects 5D, so flatten views if requested.
                        if is_multiview and view_flatten_in_disc:
                            b, v, c, t, h, w = disc_input.shape
                            disc_input = disc_input.view(b * v, c, t, h, w)
                        fake_logits = discriminator(disc_input.contiguous())

                        generator_loss, g_loss = generator_loss_fn(
                            fake_logits,
                            nll_loss,
                            model.module.get_last_layer(),
                            actual_update_step,
                            is_training=model.training,
                        )

                        vae_loss += generator_loss
                        # turn on disc training
                        discriminator.requires_grad_(True)
                        disc_time = time.time() - disc_start
                        timing_stats["discriminator"].append(disc_time)
                    else:
                        timing_stats["discriminator"].append(0.0)  # No discriminator, no time spent

                        # == generator backward & update ==
                        # Backward pass computes gradients - can be slow with large models
                        # We time it to see if gradient computation is the bottleneck
                        backward_start = time.time()
                        ctx = (
                            booster.no_sync(model, optimizer)
                            if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                            and (step + 1) % accumulation_steps != 0
                            else nullcontext()
                        )
                        with Timer("backward", log=True) if cfg.get("profile", False) else nullcontext():
                            with ctx:
                                booster.backward(loss=vae_loss / accumulation_steps, optimizer=optimizer)
                        backward_time = time.time() - backward_start
                        timing_stats["backward"].append(backward_time)

                        # Optimizer step updates weights - usually fast but can be slow with large models
                        # or complex optimizers (e.g., Adam with many parameters)
                        optimizer_start = time.time()
                        with Timer("optimizer", log=True) if cfg.get("profile", False) else nullcontext():
                            if (step + 1) % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                                if lr_scheduler is not None:
                                    lr_scheduler.step(
                                        actual_update_step,
                                    )
                                # == update EMA ==
                                # EMA update is usually fast but we include it in optimizer timing
                                if ema is not None:
                                    update_ema(
                                        ema,
                                        model.unwrap(),
                                        optimizer=optimizer,
                                        decay=cfg.get("ema_decay", 0.9999),
                                    )
                        optimizer_time = time.time() - optimizer_start
                        timing_stats["optimizer"].append(optimizer_time)
                        
                        # Track total step time (helps identify if we're missing something)
                        step_time = time.time() - step_start_time
                        timing_stats["total_step"].append(step_time)

                        # -- logging --
                        log_loss("all", vae_loss, loss_dict, use_video)
                        log_loss("nll", nll_loss, loss_dict, use_video)
                        log_loss("nll_rec", recon_loss, loss_dict, use_video)
                        log_loss("nll_per", perceptual_loss, loss_dict, use_video)
                        log_loss("kl", kl_loss, loss_dict, use_video)
                        if use_discriminator:
                            log_loss("gen_w", generator_loss, loss_dict, use_video)
                            log_loss("gen", g_loss, loss_dict, use_video)
                        
                        # -- compute reconstruction metrics on current batch (every eval_every) --
                        eval_every = cfg.get("eval_every", 0)
                        compute_batch_metrics = (
                            eval_every > 0
                            and (global_step + 1) % accumulation_steps == 0
                            and actual_update_step % eval_every == 0
                            and coordinator.is_master()
                            and use_video == 1
                        )
                        if compute_batch_metrics:
                            batch_metrics = compute_metrics(x, x_rec)
                            loss_dict["psnr"] = batch_metrics["psnr"]
                            loss_dict["ssim"] = batch_metrics["ssim"]
                            loss_dict["mse"] = batch_metrics["mse"]

                        # -- plot train reconstruction every log_every steps (fixed train samples: 1 or 3 people) --
                        log_every_steps = cfg.get("log_every", 10)
                        plot_reconstruction = (
                            (global_step + 1) % accumulation_steps == 0
                            and actual_update_step % log_every_steps == 0
                            and coordinator.is_master()
                            and use_video == 1
                        )
                        if plot_reconstruction:
                            vis_range = "[-1,1]" if (vae_target_range == "[-1,1]" or (vae_target_range is None and is_multiview)) else "[0,1]"
                            dataset = dataloader.dataset
                            participants_cfg = getattr(dataset, "participants", None)
                            # 1 sample if single person, 3 samples from 3 people if multi-person
                            num_vis = 3 if (participants_cfg is not None and len(participants_cfg) > 1) else 1
                            num_vis = min(num_vis, len(dataset))
                            vis_items = []
                            if participants_cfg is not None and len(participants_cfg) > 1 and num_vis > 1:
                                seen_pids = set()
                                for idx in range(len(dataset)):
                                    sample = dataset[idx]
                                    pt_path = sample.get("path", "")
                                    pid = None
                                    for part in os.path.normpath(pt_path).split(os.sep):
                                        if len(part) == 4 and part[0] == "p" and part[1:].isdigit():
                                            pid = part
                                            break
                                    if pid is None or pid in seen_pids:
                                        continue
                                    seen_pids.add(pid)
                                    vis_items.append(sample)
                                    if len(vis_items) >= num_vis:
                                        break
                            if len(vis_items) < num_vis:
                                vis_items = [dataset[i] for i in range(num_vis)]
                            n_vis = len(vis_items)
                            vis_batch = default_collate(vis_items[:n_vis])
                            x_vis = vis_batch["video"].to(device, dtype)
                            if vae_target_range == "[-1,1]" or (vae_target_range is None and is_multiview):
                                x_vis = x_vis * 2.0 - 1.0
                            with torch.no_grad():
                                x_rec_vis, _, _ = model(x_vis)
                            vis_images = create_visualization_grid(x_vis, x_rec_vis, num_samples=n_vis, value_range=vis_range)
                            loss_dict["reconstruction_samples"] = vis_images
                            # Also plot val/test reconstructions every log_every when val_dataset exists
                            if val_dataset is not None and len(val_dataset) > 0:
                                n_val = min(3, len(val_dataset))
                                val_items = [val_dataset[i] for i in range(n_val)]
                                val_batch = default_collate(val_items)
                                x_val = val_batch["video"].to(device, dtype)
                                if vae_target_range == "[-1,1]" or (vae_target_range is None and is_multiview):
                                    x_val = x_val * 2.0 - 1.0
                                with torch.no_grad():
                                    x_rec_val, _, _ = model(x_val)
                                val_vis_images = create_visualization_grid(x_val, x_rec_val, num_samples=n_val, value_range=vis_range)
                                loss_dict["val_reconstruction_samples"] = val_vis_images

                    # == loss: discriminator adversarial ==
                    if use_discriminator:
                        real_input = x.detach()
                        fake_input = x_rec.detach()
                        if is_multiview and view_flatten_in_disc:
                            b, v, c, t, h, w = real_input.shape
                            real_input = real_input.view(b * v, c, t, h, w)
                            fake_input = fake_input.view(b * v, c, t, h, w)
                        real_logits = discriminator(real_input.contiguous())
                        fake_logits = discriminator(fake_input.contiguous())
                        disc_loss = discriminator_loss_fn(
                            real_logits,
                            fake_logits,
                            actual_update_step,
                        )

        
                        # == discriminator backward & update ==
                        ctx = (
                            booster.no_sync(discriminator, disc_optimizer)
                            if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                            and (step + 1) % accumulation_steps != 0
                            else nullcontext()
                        )
                        with ctx:
                            booster.backward(loss=disc_loss / accumulation_steps, optimizer=disc_optimizer)
                        if (step + 1) % accumulation_steps == 0:
                            disc_optimizer.step()
                            disc_optimizer.zero_grad()
                            if disc_lr_scheduler is not None:
                                disc_lr_scheduler.step(actual_update_step)

                        # log
                        log_loss("disc", disc_loss, loss_dict, use_video)

                    # == logging ==
                    # We log periodically to avoid overwhelming the logs, but include timing stats
                    # to help identify bottlenecks. Logging itself is fast, so we don't time it.
                    if (global_step + 1) % accumulation_steps == 0:
                        if coordinator.is_master() and actual_update_step % cfg.get("log_every", 1) == 0:
                            avg_loss = {k: v / log_step for k, v in running_loss.items()}
                            
                            # Compute average timing stats over the logged steps
                            # This tells us where time is being spent (data loading, forward, backward, etc.)
                            avg_timing = {}
                            for key, times in timing_stats.items():
                                if times and key not in ["memory_allocated", "memory_reserved"]:
                                    # Average over recent steps (last log_every steps)
                                    recent_times = times[-log_step:] if len(times) >= log_step else times
                                    if recent_times:
                                        avg_timing[f"time/{key}"] = sum(recent_times) / len(recent_times)
                            
                            # Log timing breakdown to help identify bottlenecks
                            # Format: "time/data_load: 0.05s, time/forward: 0.15s, ..."
                            timing_str = ", ".join([f"{k}: {v:.3f}s" for k, v in avg_timing.items()])
                            
                            # BOTTLENECK summary: print every log_bottleneck_every steps (0 = off)
                            log_bottleneck_every = int(cfg.get("log_bottleneck_every", 0))
                            if log_bottleneck_every and actual_update_step % log_bottleneck_every == 0:
                                total = avg_timing.get("time/total_step", 0.0) or 1e-6
                                order = ["time/data_load", "time/forward", "time/loss_compute", "time/discriminator", "time/backward", "time/optimizer", "time/total_step"]
                                parts = []
                                for k in order:
                                    v = avg_timing.get(k, 0.0)
                                    if v > 0:
                                        pct = 100.0 * v / total
                                        parts.append(f"{k}: {v:.3f}s ({pct:.0f}%)")
                                print("[BOTTLENECK] step {} | total_step {:.3f}s | {}".format(
                                    actual_update_step, total, " | ".join(parts)))
                                est_1k = (total * 1000) / 60.0  # minutes for 1k steps
                                print(f"  -> at this rate 1000 steps ≈ {est_1k:.1f} min")
                            
                            logger.info(
                                f"Step {actual_update_step} | Loss: {avg_loss['all']:.6f} | "
                                f"Timing: {timing_str}"
                            )
                            
                            # progress bar
                            pbar.set_postfix(
                                {
                                    **{k: f"{v:.2f}" for k, v in avg_loss.items()},
                                }
                            )
                            
                            # wandb
                            if cfg.get("wandb", False):
                                wandb_log_dict = {
                                        "iter": global_step,
                                        "epoch": epoch,
                                        "epoch_float": epoch_float,
                                        "lr": optimizer.param_groups[0]["lr"],
                                        # Average losses over log_every steps
                                        "loss/total": avg_loss["all"],
                                        "loss/nll": avg_loss.get("nll", 0.0),
                                        "loss/nll_rec": avg_loss.get("nll_rec", 0.0),
                                        "loss/nll_per": avg_loss.get("nll_per", 0.0),
                                        "loss/kl": avg_loss.get("kl", 0.0),
                                        "global_grad_norm": optimizer.get_grad_norm(),
                                    }
                                
                                # Add timing stats to wandb - super useful for bottleneck analysis!
                                # You can plot these in wandb to see which operation takes the most time
                                wandb_log_dict.update(avg_timing)
                                
                                # Add memory stats if enabled
                                if log_memory and timing_stats["memory_allocated"]:
                                    recent_mem = timing_stats["memory_allocated"][-log_step:]
                                    if recent_mem:
                                        wandb_log_dict["memory/allocated_gb"] = sum(recent_mem) / len(recent_mem)
                                        wandb_log_dict["memory/reserved_gb"] = (
                                            sum(timing_stats["memory_reserved"][-log_step:]) / len(recent_mem)
                                            if timing_stats["memory_reserved"] else 0
                                        )
                                
                                # Add reconstruction metrics if computed
                                if "psnr" in loss_dict:
                                    wandb_log_dict["metrics/psnr"] = loss_dict["psnr"]
                                    wandb_log_dict["metrics/ssim"] = loss_dict["ssim"]
                                    wandb_log_dict["metrics/mse"] = loss_dict["mse"]
                                
                                # Add visualizations if available (train + val so both appear in Media)
                                if "reconstruction_samples" in loss_dict:
                                    wandb_log_dict["reconstructions"] = loss_dict["reconstruction_samples"]
                                if "val_reconstruction_samples" in loss_dict:
                                    wandb_log_dict["val_reconstructions"] = loss_dict["val_reconstruction_samples"]
                                
                                wandb.log(wandb_log_dict, step=actual_update_step)

                            running_loss = {k: 0.0 for k in running_loss}
                            log_step = 0
                            
                            # Clear old timing stats to avoid memory buildup (keep last 100 steps)
                            for key in timing_stats:
                                if key not in ["memory_allocated", "memory_reserved"]:
                                    if len(timing_stats[key]) > 100:
                                        timing_stats[key] = timing_stats[key][-100:]

                        # == checkpoint saving ==
                        ckpt_every = cfg.get("ckpt_every", 0)
                        if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                            # mannually garbage collection
                            gc.collect()

                            # Disable async_io if tensornvme is not available to avoid AsyncFileWriter errors
                            # ColossalAI tries to use AsyncFileWriter even when async_io is enabled, causing errors
                            # if tensornvme is not installed
                            use_async_io = False  # Set to False to avoid AsyncFileWriter dependency
                            
                            save_dir = checkpoint_io.save(
                                booster,
                                exp_dir,
                                model=model,
                                ema=ema,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                sampler=sampler,
                                epoch=epoch,
                                step=step + 1,
                                global_step=global_step + 1,
                                batch_size=cfg.get("batch_size", None),
                                actual_update_step=actual_update_step,
                                ema_shape_dict=ema_shape_dict,
                                async_io=use_async_io,
                            )

                            if use_discriminator:
                                booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                                booster.save_optimizer(
                                    disc_optimizer,
                                    os.path.join(save_dir, "disc_optimizer"),
                                    shard=True,
                                    size_per_shard=4096,
                                )
                                if disc_lr_scheduler is not None:
                                    booster.save_lr_scheduler(
                                        disc_lr_scheduler, os.path.join(save_dir, "disc_lr_scheduler")
                                    )
                            dist.barrier()

                            logger.info(
                                "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                                epoch,
                                step + 1,
                                actual_update_step,
                                save_dir,
                            )

                            # remove old checkpoints
                            rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                            logger.info(
                                "Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1)
                            )
                            
                            # -- periodic full evaluation --
                            # Run a more comprehensive evaluation pass periodically to get
                            # better statistics than single-batch metrics. This evaluates
                            # over multiple samples to get more reliable metrics.
                            full_eval_every = cfg.get("full_eval_every", 0)  # 0 means disabled
                            if (
                                full_eval_every > 0
                                and actual_update_step % full_eval_every == 0
                                and coordinator.is_master()
                            ):
                                # Evaluate on held-out val data when present, else on train dataset
                                eval_ds = val_dataset if val_dataset is not None else dataset
                                eval_ds_label = "val" if val_dataset is not None else "train"
                                logger.info("Running full evaluation at step %s (%s)...", actual_update_step, eval_ds_label)
                                
                                eval_model = model
                                eval_dataloader, _ = prepare_dataloader(
                                    bucket_config=cfg.get("bucket_config", None),
                                    num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
                                    dataset=eval_ds,
                                    batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 4)),
                                    num_workers=cfg.get("num_workers", 4),
                                    seed=cfg.get("seed", 1024) + 9999,
                                    shuffle=True,
                                    drop_last=False,
                                    pin_memory=True,
                                    process_group=get_data_parallel_group(),
                                    prefetch_factor=cfg.get("prefetch_factor", None),
                                    cache_pin_memory=False,
                                )
                                eval_val_range = cfg.get("vae_target_range") or ("[-1,1]" if cfg.model.get("type") == "multiview_wan_video_vae" else "[0,1]")
                                eval_results = evaluate_model(
                                    eval_model,
                                    eval_dataloader,
                                    device,
                                    dtype,
                                    num_eval_samples=cfg.get("eval_num_samples", 32),
                                    view_flatten_in_loss=view_flatten_in_loss,
                                    use_ema=(ema is not None and cfg.get("eval_use_ema", True)),
                                    value_range=eval_val_range,
                                )
                                eval_metrics = eval_results["metrics"]
                                logger.info(
                                    "Evaluation (%s) - PSNR: %.2f ± %.2f, SSIM: %.4f ± %.4f, MSE: %.6f ± %.6f",
                                    eval_ds_label,
                                    eval_metrics["psnr_mean"],
                                    eval_metrics["psnr_std"],
                                    eval_metrics["ssim_mean"],
                                    eval_metrics["ssim_std"],
                                    eval_metrics["mse_mean"],
                                    eval_metrics["mse_std"],
                                )
                                if cfg.get("wandb", False):
                                    prefix = f"eval/{eval_ds_label}"
                                    log_dict = {
                                        "epoch_float": epoch_float,
                                        f"{prefix}/psnr_mean": eval_metrics["psnr_mean"],
                                        f"{prefix}/psnr_std": eval_metrics["psnr_std"],
                                        f"{prefix}/ssim_mean": eval_metrics["ssim_mean"],
                                        f"{prefix}/ssim_std": eval_metrics["ssim_std"],
                                        f"{prefix}/mse_mean": eval_metrics["mse_mean"],
                                        f"{prefix}/mse_std": eval_metrics["mse_std"],
                                        f"{prefix}/reconstructions": eval_results["visualizations"],
                                    }
                                    if eval_ds_label == "val":
                                        log_dict["val_reconstructions"] = eval_results["visualizations"]
                                    wandb.log(log_dict, step=actual_update_step)

            if cfg.get("profile", False):
                profiler_ctxt.export_chrome_trace("./log/profile/trace.json")

        # Reset sampler for next epoch (if it has the reset method)
        if sampler is not None and hasattr(sampler, 'reset'):
            sampler.reset()
        start_step = 0
        
        # =======================================================
        # Per-frame metrics on fixed train/val sequences every N epochs
        # =======================================================
        fixed_epoch_interval = int(cfg.get("fixed_seq_eval_every_epochs", 10))
        if (
            fixed_epoch_interval > 0
            and ((epoch + 1) % fixed_epoch_interval == 0)
            and coordinator.is_master()
            and cfg.get("wandb", False)
            and fixed_train_index is not None
        ):
            logger.info(
                "Running fixed per-frame evaluation at epoch %s (sequence name: %s)...",
                epoch + 1,
                fixed_seq_name,
            )
            vae_target_range = cfg.get("vae_target_range", None)

            # Fixed train sequence
            train_metrics_pf = evaluate_fixed_sequence_per_frame(
                model,
                dataset,
                fixed_train_index,
                device,
                dtype,
                vae_target_range=vae_target_range,
            )

            wandb_log_pf = {
                "fixed_seq/name": fixed_seq_name,
                "fixed_seq/epoch": epoch + 1,
                "epoch_float": float(epoch + 1),
            }

            for i, val in enumerate(train_metrics_pf["psnr_per_frame"]):
                wandb_log_pf[f"fixed_seq/train/psnr_frame_{i}"] = val
            for i, val in enumerate(train_metrics_pf["ssim_per_frame"]):
                wandb_log_pf[f"fixed_seq/train/ssim_frame_{i}"] = val
            for i, val in enumerate(train_metrics_pf["mse_per_frame"]):
                wandb_log_pf[f"fixed_seq/train/mse_frame_{i}"] = val

            # Fixed val/test sequence (same sequence name where possible)
            if fixed_val_index is not None and val_dataset is not None:
                val_metrics_pf = evaluate_fixed_sequence_per_frame(
                    model,
                    val_dataset,
                    fixed_val_index,
                    device,
                    dtype,
                    vae_target_range=vae_target_range,
                )
                for i, val in enumerate(val_metrics_pf["psnr_per_frame"]):
                    wandb_log_pf[f"fixed_seq/val/psnr_frame_{i}"] = val
                for i, val in enumerate(val_metrics_pf["ssim_per_frame"]):
                    wandb_log_pf[f"fixed_seq/val/ssim_frame_{i}"] = val
                for i, val in enumerate(val_metrics_pf["mse_per_frame"]):
                    wandb_log_pf[f"fixed_seq/val/mse_frame_{i}"] = val

            table = wandb.Table(columns=["frame", "psnr", "ssim", "mse"])

            for i in range(len(train_metrics_pf["psnr_per_frame"])):
                table.add_data(
                    i,
                    train_metrics_pf["psnr_per_frame"][i],
                    train_metrics_pf["ssim_per_frame"][i],
                    train_metrics_pf["mse_per_frame"][i],
                )

            wandb.log({"fixed_seq/train_metrics": table, "epoch_float": float(epoch + 1)}, step=actual_update_step)
    
    # =======================================================
    # 6. Final evaluation after training
    # =======================================================
    # After all training epochs are complete, run a final comprehensive
    # evaluation to assess the final model quality. This gives us the
    # definitive metrics for the trained model.
    if coordinator.is_master():
        final_eval_enabled = cfg.get("final_eval", True)
        if final_eval_enabled:
            logger.info("Training complete. Running final evaluation...")
            final_eval_model = model
            final_val_range = cfg.get("vae_target_range") or ("[-1,1]" if cfg.model.get("type") == "multiview_wan_video_vae" else "[0,1]")
            num_final = cfg.get("final_eval_num_samples", 100)
            # Always run final eval on train data (reconstructions always plotted)
            eval_sets = [("train", dataset)]
            if val_dataset is not None:
                eval_sets.append(("val", val_dataset))
            for label, ds in eval_sets:
                final_eval_dataloader, _ = prepare_dataloader(
                    bucket_config=cfg.get("bucket_config", None),
                    num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
                    dataset=ds,
                    batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 4)),
                    num_workers=cfg.get("num_workers", 4),
                    seed=cfg.get("seed", 1024) + 8888,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=True,
                    process_group=get_data_parallel_group(),
                    prefetch_factor=cfg.get("prefetch_factor", None),
                    cache_pin_memory=False,
                )
                final_eval_results = evaluate_model(
                    final_eval_model,
                    final_eval_dataloader,
                    device,
                    dtype,
                    num_eval_samples=num_final,
                    view_flatten_in_loss=view_flatten_in_loss,
                    use_ema=(ema is not None and cfg.get("eval_use_ema", True)),
                    value_range=final_val_range,
                )
                final_metrics = final_eval_results["metrics"]
                logger.info("=" * 80)
                logger.info("FINAL EVALUATION (%s)", label.upper())
                logger.info("=" * 80)
                logger.info("PSNR: %.2f ± %.2f dB", final_metrics["psnr_mean"], final_metrics["psnr_std"])
                logger.info("SSIM: %.4f ± %.4f", final_metrics["ssim_mean"], final_metrics["ssim_std"])
                logger.info("MSE:  %.6f ± %.6f", final_metrics["mse_mean"], final_metrics["mse_std"])
                logger.info("=" * 80)
                if cfg.get("wandb", False):
                    prefix = f"final_eval/{label}"
                    final_log = {
                        "epoch_float": float(epoch + 1),
                        f"{prefix}/psnr_mean": final_metrics["psnr_mean"],
                        f"{prefix}/psnr_std": final_metrics["psnr_std"],
                        f"{prefix}/ssim_mean": final_metrics["ssim_mean"],
                        f"{prefix}/ssim_std": final_metrics["ssim_std"],
                        f"{prefix}/mse_mean": final_metrics["mse_mean"],
                        f"{prefix}/mse_std": final_metrics["mse_std"],
                        f"{prefix}/reconstructions": final_eval_results["visualizations"],
                    }
                    if label == "val":
                        final_log["val_reconstructions"] = final_eval_results["visualizations"]
                    wandb.log(final_log, step=actual_update_step)
            logger.info("Final evaluation complete.")
    
    # =======================================================
    # 7. Save final checkpoint after training
    # =======================================================
    # Save the final checkpoint after all training and evaluation is complete
    if coordinator.is_master():
        logger.info("Saving final checkpoint...")
        use_async_io = False  # Disable async_io to avoid AsyncFileWriter dependency
        
        final_save_dir = checkpoint_io.save(
            booster,
            exp_dir,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=sampler,
            epoch=epoch,
            step=step + 1,
            global_step=global_step + 1,
            batch_size=cfg.get("batch_size", None),
            actual_update_step=actual_update_step,
            ema_shape_dict=ema_shape_dict,
            async_io=use_async_io,
        )

        if use_discriminator:
            booster.save_model(discriminator, os.path.join(final_save_dir, "discriminator"), shard=True)
            booster.save_optimizer(
                disc_optimizer,
                os.path.join(final_save_dir, "disc_optimizer"),
                shard=True,
                size_per_shard=4096,
            )
            if disc_lr_scheduler is not None:
                booster.save_lr_scheduler(
                    disc_lr_scheduler, os.path.join(final_save_dir, "disc_lr_scheduler")
                )
        
        logger.info(f"Final checkpoint saved to {final_save_dir}")
    
    dist.barrier()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()


                            