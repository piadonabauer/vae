# START with: python3 scripts/vae/train.py /home/piado/projects/aip-lindell/piado/vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py

# resume training with: python scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py \
#  --load /home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs/260228_150719-.../epoch0-global_step200

import gc
import json
import math
import os
import random
import shutil
import subprocess
import time
import warnings
from contextlib import nullcontext
from torch.profiler import ProfilerActivity, profile, record_function
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
from colossalai.utils import get_current_device, set_seed
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
from opensora.models.vae.stylegan2_disc_loader import build_stylegan2_ada_discriminator_from_state_dict
from opensora.models.vae.utils import DiagonalGaussianDistribution
from opensora.models.vae.wan_video_vae import build_multiview_wan_video_vae, ProfileTimer  # Register multi-view VAE model
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import CheckpointIO, load_checkpoint, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    Timer,
    all_reduce_sum,
    is_log_process,
    log_model_params,
    log_trainable_param_overview,
    log_wan_multiview_training_design_summary,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.train import create_colossalai_plugin, set_lr, set_warmup_steps, setup_device, update_ema

WAIT = 1
WARMUP = 10
ACTIVE = 20

my_schedule = schedule(
    wait=WAIT,  # number of warmup steps
    warmup=WARMUP,  # number of warmup steps with profiling
    active=ACTIVE,  # number of active steps with profiling
)


def _loss_config_dict(cfg) -> dict:
    """Structured VAE + GAN loss settings for run names, JSONL, and wandb."""
    vlc = cfg.get("vae_loss_config") if isinstance(cfg.get("vae_loss_config"), dict) else {}
    disc_choice = cfg.get("discriminator_choice")
    if disc_choice is None:
        dr = cfg.get("discriminator")
        if isinstance(dr, dict):
            disc_choice = dr.get("type", "unknown")
        else:
            disc_choice = "none" if dr is None else str(dr)
    gdw = cfg.get("gen_disc_weight")
    if gdw is None:
        gcfg = cfg.get("gen_loss_config")
        if isinstance(gcfg, dict):
            gdw = gcfg.get("disc_weight")
    if cfg.get("discriminator", None) is None:
        gdw = None
    return {
        "vae_loss_preset": cfg.get("vae_loss_preset", "default"),
        "perceptual_loss_weight": vlc.get(
            "perceptual_loss_weight", cfg.get("perceptual_loss_weight", None)
        ),
        "kl_loss_weight": vlc.get("kl_loss_weight", cfg.get("kl_loss_weight", None)),
        "view_consistency_weight": vlc.get(
            "view_consistency_weight", cfg.get("view_consistency_weight", None)
        ),
        "logvar_init": vlc.get("logvar_init", cfg.get("logvar_init", None)),
        "discriminator_choice": disc_choice,
        "gen_disc_weight": gdw,
    }


def _loss_signature_slug(cfg) -> str:
    """Filesystem- and wandb-safe short string describing loss weights."""
    d = _loss_config_dict(cfg)

    def _tok(x):
        if x is None:
            return "na"
        s = str(x).replace(" ", "")
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)[:32]

    parts = [
        f"lp{_tok(d['vae_loss_preset'])}",
        f"per{_tok(d['perceptual_loss_weight'])}",
        f"kl{_tok(d['kl_loss_weight'])}",
        f"vc{_tok(d['view_consistency_weight'])}",
        f"disc{_tok(d['discriminator_choice'])}",
        f"gdw{_tok(d['gen_disc_weight'])}",
    ]
    return "_".join(parts)[:200]


def append_eval_metrics_jsonl(exp_dir: str, record: dict) -> None:
    """Append one JSON object per line to eval_metrics.jsonl (master process only)."""
    path = os.path.join(exp_dir, "eval_metrics.jsonl")
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def append_training_debug_jsonl(exp_dir: str, record: dict) -> None:
    """Append one JSON object per line to train_debug_stats.jsonl (master process only)."""
    path = os.path.join(exp_dir, "train_debug_stats.jsonl")
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _to_float_scalar(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().item())
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_precise_float_str(x, digits: int = 12):
    """Render scalar values with explicit precision for debug logging."""
    v = _to_float_scalar(x)
    if v is None:
        return None
    return format(v, f".{digits}g")


def _extract_batch_sample_ids(batch: dict):
    """Best-effort extraction of per-sample IDs from a dataloader batch."""
    if not isinstance(batch, dict):
        return None

    index_keys = ("index", "indices", "idx", "sample_idx", "sample_index")
    for key in index_keys:
        if key not in batch:
            continue
        val = batch[key]
        if torch.is_tensor(val):
            if val.ndim == 0:
                return [int(val.detach().item())]
            return [int(x) for x in val.detach().cpu().tolist()]
        if isinstance(val, (list, tuple)):
            out = []
            for x in val:
                try:
                    out.append(int(x))
                except Exception:
                    out.append(str(x))
            return out
        try:
            return [int(val)]
        except Exception:
            return [str(val)]

    path_keys = ("path", "paths", "file", "files")
    for key in path_keys:
        if key not in batch:
            continue
        val = batch[key]
        if isinstance(val, str):
            return [val]
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        return [str(val)]

    return None


def collect_trainable_param_stats(model, include_weight_stats: bool = True, optimizer=None) -> dict:
    """Collect grad/weight stats for trainable parameters.

    With ColossalAI ZeRO / LowLevelZero, ``p.grad`` is often **unset** on the module's
    ``nn.Parameter`` objects after ``backward`` (gradients are sharded internally). In that
    case we fall back to ``optimizer.get_grad_norm()`` when available (same idea as W&B
    ``global_grad_norm``).
    """
    m = model
    if hasattr(model, "unwrap"):
        try:
            m = model.unwrap()
        except Exception:
            m = model.module if hasattr(model, "module") else model
    elif hasattr(model, "module"):
        m = model.module

    grad_abs_mean_sum = 0.0
    grad_abs_max = 0.0
    grad_sq_sum = 0.0
    grad_numel = 0
    grad_nonfinite = 0
    grad_tensors = 0

    weight_abs_mean_sum = 0.0
    weight_abs_max = 0.0
    weight_sq_sum = 0.0
    weight_numel = 0
    weight_nonfinite = 0
    weight_tensors = 0

    for p in m.parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            g = p.grad.detach()
            g_abs = g.abs()
            grad_abs_mean_sum += g_abs.sum().item()
            grad_abs_max = max(grad_abs_max, g_abs.max().item())
            grad_sq_sum += torch.sum(g * g).item()
            grad_numel += g.numel()
            grad_nonfinite += int((~torch.isfinite(g)).sum().item())
            grad_tensors += 1
        if include_weight_stats:
            w = p.detach()
            w_abs = w.abs()
            weight_abs_mean_sum += w_abs.sum().item()
            weight_abs_max = max(weight_abs_max, w_abs.max().item())
            weight_sq_sum += torch.sum(w * w).item()
            weight_numel += w.numel()
            weight_nonfinite += int((~torch.isfinite(w)).sum().item())
            weight_tensors += 1

    # ZeRO / sharded optimizers: recover global norm from optimizer when per-param grads missing.
    grad_fallback_norm = None
    if grad_numel == 0 and optimizer is not None and hasattr(optimizer, "get_grad_norm"):
        try:
            gn = optimizer.get_grad_norm()
            grad_fallback_norm = float(gn) if gn is not None else None
        except Exception:
            grad_fallback_norm = None

    if grad_numel > 0:
        l2 = grad_sq_sum**0.5
        grad_stats_source = "per_parameter_grad"
        out = {
            "grad_abs_mean": (grad_abs_mean_sum / max(1, grad_numel)),
            "grad_abs_max": grad_abs_max,
            "grad_l2_norm": l2,
            "global_grad_norm": l2,
            "grad_numel": int(grad_numel),
            "grad_nonfinite": int(grad_nonfinite),
            "grad_tensors_with_grad": int(grad_tensors),
            "grad_stats_source": grad_stats_source,
        }
    elif grad_fallback_norm is not None:
        out = {
            "grad_abs_mean": None,
            "grad_abs_max": None,
            "grad_l2_norm": grad_fallback_norm,
            "grad_numel": None,
            "grad_nonfinite": 0,
            "grad_tensors_with_grad": 0,
            "grad_stats_source": "optimizer_get_grad_norm (sharded ZeRO; per-param .grad empty on module)",
            "global_grad_norm": grad_fallback_norm,
        }
    else:
        out = {
            "grad_abs_mean": 0.0,
            "grad_abs_max": 0.0,
            "grad_l2_norm": 0.0,
            "global_grad_norm": 0.0,
            "grad_numel": int(grad_numel),
            "grad_nonfinite": int(grad_nonfinite),
            "grad_tensors_with_grad": int(grad_tensors),
            "grad_stats_source": "none (no grads and get_grad_norm unavailable)",
        }
    if include_weight_stats:
        out.update(
            {
                "weight_abs_mean": (weight_abs_mean_sum / max(1, weight_numel)),
                "weight_abs_max": weight_abs_max,
                "weight_l2_norm": weight_sq_sum ** 0.5,
                "weight_numel": int(weight_numel),
                "weight_nonfinite": int(weight_nonfinite),
                "weight_tensors": int(weight_tensors),
            }
        )
    return out


def unwrap_model_safe(model):
    """Return the underlying nn.Module for wrapped or plain models."""
    if hasattr(model, "unwrap"):
        try:
            model = model.unwrap()
        except Exception:
            pass
    if hasattr(model, "module"):
        model = model.module
    # torch.compile wraps the original module as _orig_mod
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _iter_optimizer_grad_tensors(optimizer):
    """Yield unique gradient tensors visible to the optimizer.

    Under ZeRO, gradients may live on master/sharded params instead of module parameters.
    """
    seen = set()

    def _yield_from_params(params):
        for p in params:
            if p is None:
                continue
            g = getattr(p, "grad", None)
            if g is None:
                continue
            gid = id(g)
            if gid in seen:
                continue
            seen.add(gid)
            yield g

    for group in getattr(optimizer, "param_groups", []):
        yield from _yield_from_params(group.get("params", []))

    for group in getattr(optimizer, "_master_param_groups", []):
        yield from _yield_from_params(group)


@torch.no_grad()
def _safe_optimizer_get_grad_norm(optimizer) -> float | None:
    """Best-effort fallback when optimizer-managed grad tensors are not directly visible."""
    if optimizer is None or not hasattr(optimizer, "get_grad_norm"):
        return None
    try:
        gn = optimizer.get_grad_norm()
    except Exception:
        return None
    if gn is None:
        return None
    if isinstance(gn, torch.Tensor):
        try:
            return float(gn.detach().float().item())
        except Exception:
            return None
    try:
        return float(gn)
    except Exception:
        return None


@torch.no_grad()
def compute_optimizer_global_grad_norm(optimizer, dp_group=None) -> float | None:
    """Compute true global grad norm across DP ranks for optimizer-visible grads."""
    local_sq_sum = 0.0
    local_found_grad = False
    for g in _iter_optimizer_grad_tensors(optimizer):
        local_found_grad = True
        gg = g.detach()
        if gg.is_sparse:
            gg = gg.coalesce().values()
        gg = gg.float()
        local_sq_sum += torch.sum(gg * gg).item()
    device = get_current_device()
    sq_sum = torch.tensor(local_sq_sum, dtype=torch.float64, device=device)
    found_grad_any = torch.tensor(1 if local_found_grad else 0, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(found_grad_any, op=dist.ReduceOp.SUM, group=dp_group)
    if int(found_grad_any.item()) == 0:
        return _safe_optimizer_get_grad_norm(optimizer)
    return float(torch.sqrt(sq_sum).item())


@torch.no_grad()
def clip_optimizer_grad_norm_global_(optimizer, max_norm: float, dp_group=None, eps: float = 1e-6):
    """Clip optimizer gradients using a truly global L2 norm across DP ranks."""
    if max_norm is None or max_norm <= 0:
        return None, None, 1.0

    pre_clip_norm = compute_optimizer_global_grad_norm(optimizer, dp_group=dp_group)
    if pre_clip_norm is None:
        return None, None, 1.0

    clip_coef = float(max_norm) / (pre_clip_norm + eps)
    clip_coef_clamped = clip_coef if clip_coef < 1.0 else 1.0
    if clip_coef_clamped < 1.0:
        for g in _iter_optimizer_grad_tensors(optimizer):
            g.mul_(clip_coef_clamped)

    post_clip_norm = compute_optimizer_global_grad_norm(optimizer, dp_group=dp_group)
    return pre_clip_norm, post_clip_norm, clip_coef_clamped


def apply_pytorch_determinism(cfg) -> bool:
    """CUDA/PyTorch settings for best-effort reproducibility. Call before ``setup_device()`` so
    ``CUBLAS_WORKSPACE_CONFIG`` can take effect. Returns the effective ``deterministic`` flag.

    Bit-exact runs are still not guaranteed under distributed ZeRO, BF16, or non-deterministic
    kernels; use ``dtype=\"fp32\"`` and ``plugin=\"none\"`` on one GPU for stricter replay.
    """
    deterministic = bool(cfg.get("deterministic", True))
    if deterministic:
        # Deterministic cuBLAS matmul selection (must be set before CUDA context init when possible).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                print(f"[determinism] torch.use_deterministic_algorithms unavailable: {exc}")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return deterministic


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
        x_flat = x.contiguous().view(b * v * t, c, h, w)
        x_flat = F.interpolate(x_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = x_flat.view(b, v, c, t, target_h, target_w)
    elif x.dim() == 5:
        b, c, t, h, w = x.shape
        if t != target_t:
            idx = torch.linspace(0, t - 1, target_t, device=x.device).long()
            x = x.index_select(2, idx)
            t = target_t
        x_perm = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x_flat = x_perm.reshape(b * t, c, h, w)
        x_flat = F.interpolate(x_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x_perm = x_flat.view(b, t, c, target_h, target_w)
        x = x_perm.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    return x


def apply_train_bucket_spatiotemporal(x: torch.Tensor, cfg) -> torch.Tensor:
    """If ``train_target_hw`` / ``train_target_frames`` are set (Nersemble bucket), match spatial/temporal."""
    th = cfg.get("train_target_hw")
    tt = cfg.get("train_target_frames")
    if th is None or tt is None:
        return x
    if not isinstance(th, (list, tuple)) or len(th) < 2:
        return x
    return downsample_video_tensor(
        x, target_h=int(th[0]), target_w=int(th[1]), target_t=int(tt)
    )


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
    # Ensure values are in [0, 1] range for proper metric computation.
    # nan_to_num first: clamp() does NOT fix NaN (NaN stays NaN after clamp).
    # Replacing NaN/inf with 0 gives a valid but very-bad-PSNR signal rather than silently NaN.
    x_orig = torch.nan_to_num(x_orig.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    x_rec = torch.nan_to_num(x_rec.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    
    # Ensure temporal dimensions match before flattening; some sequences can be shorter.
    if x_orig.dim() == 6 and x_rec.dim() == 6:  # Multi-view: [B, V, C, T, H, W]
        b_o, v_o, c_o, t_o, h_o, w_o = x_orig.shape
        b_r, v_r, c_r, t_r, h_r, w_r = x_rec.shape
        # Align along time; assume same B,V,C,H,W, but allow T to differ by truncating to min
        t = min(t_o, t_r)
        x_orig_use = x_orig[:, :, :, :t, :, :]
        x_rec_use = x_rec[:, :, :, :t, :, :]
        b, v, c, _, h, w = x_orig_use.shape
        # reshape (not view): torch.compile outputs may be non-contiguous
        x_orig_flat = x_orig_use.reshape(b * v, c, t, h, w)
        x_rec_flat = x_rec_use.reshape(b * v, c, t, h, w)
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


def compute_intra_chunk_bleed_metrics(x_orig, x_rec, chunk_size=4):
    """
    Diagnose temporal "bleeding" from temporal_compression=True.

    Wan's causal chunked layout: frame 0 is decoded standalone from its own latent
    frame, then every subsequent group of `chunk_size` real frames shares ONE
    latent frame. If the decoder can't disentangle that one latent frame into
    genuinely different output frames, consecutive frames *within* a chunk will
    look far more similar to each other than the ground truth does at those same
    positions ("bleeding"), while frames straddling a chunk boundary (each backed
    by a different latent frame) are less affected.

    This compares mean |frame[t] - frame[t-1]| in the reconstruction vs the
    ground truth, split into "within-chunk" pairs and "across-chunk-boundary"
    pairs:
      - bleed_ratio_within  ~= rec-frame-to-frame-delta / gt-frame-to-frame-delta,
        restricted to pairs inside the same chunk.
      - bleed_ratio_across  ~= same, restricted to pairs that cross a chunk
        boundary (acts as a rough control/baseline).
    ~1.0 = reconstruction varies frame-to-frame as much as the ground truth
    (healthy). ~0.0 = the reconstruction is nearly frozen across those frames
    while the ground truth is not (severe bleeding). If bleed_ratio_within is
    much lower than bleed_ratio_across, the bleeding is specifically a
    within-chunk (temporal-compression) effect rather than generic blur.
    """
    with torch.no_grad():
        xo = x_orig.detach().float()
        xr = x_rec.detach().float()
        if xo.dim() == 6:  # [B, V, C, T, H, W]
            b, v, c, t, h, w = xo.shape
            xo = xo.reshape(b * v, c, t, h, w)
            xr = xr.reshape(xo.shape[0], c, xr.shape[-3] if xr.dim() == 6 else t, h, w) if xr.dim() != 6 else xr.reshape(b * v, c, t, h, w)
        t = min(xo.shape[2], xr.shape[2])
        if t < 2:
            return {}
        xo = xo[:, :, :t]
        xr = xr[:, :, :t]

        gt_diff = (xo[:, :, 1:] - xo[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))  # [T-1]
        rec_diff = (xr[:, :, 1:] - xr[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))  # [T-1]

        def chunk_id(tt):
            return 0 if tt == 0 else 1 + (tt - 1) // chunk_size

        within_idx = [i for i in range(t - 1) if chunk_id(i) == chunk_id(i + 1)]
        across_idx = [i for i in range(t - 1) if chunk_id(i) != chunk_id(i + 1)]

        eps = 1e-6
        out = {}
        if within_idx:
            idx = torch.tensor(within_idx, device=xo.device)
            gt_w = gt_diff[idx].mean()
            rec_w = rec_diff[idx].mean()
            out["bleed_ratio_within"] = (rec_w / (gt_w + eps)).item()
            out["gt_diff_within"] = gt_w.item()
            out["rec_diff_within"] = rec_w.item()
        if across_idx:
            idx = torch.tensor(across_idx, device=xo.device)
            gt_a = gt_diff[idx].mean()
            rec_a = rec_diff[idx].mean()
            out["bleed_ratio_across"] = (rec_a / (gt_a + eps)).item()
            out["gt_diff_across"] = gt_a.item()
            out["rec_diff_across"] = rec_a.item()
        return out


def compute_cross_view_similarity(x):
    """
    Mean pairwise cosine similarity between views of a multi-view clip.

    Ghosting diagnostic for the paper: if the decoder collapses the views onto
    their mean, the similarity of the *reconstruction* rises towards 1 while the
    ground truth stays at its natural level (nearby cameras are similar but not
    identical). Always report rec AND gt so the gap is interpretable.

    x: [B, V, C, T, H, W]; returns a float, or None for single-view input.
    """
    if x.dim() != 6 or x.shape[1] < 2:
        return None
    with torch.no_grad():
        b, v = x.shape[0], x.shape[1]
        flat = x.detach().float().reshape(b, v, -1)
        flat = F.normalize(flat, dim=-1)
        sims = []
        for i in range(v):
            for j in range(i + 1, v):
                sims.append((flat[:, i] * flat[:, j]).sum(-1).mean())
        return torch.stack(sims).mean().item()


def compute_temporal_diff_loss(x, x_rec, is_multiview):
    """
    L1 between consecutive-frame differences of ground truth vs reconstruction.

    Zero new parameters. Directly penalizes the decoder for producing frames
    that don't change from one timestep to the next when the ground truth does
    change -- the failure mode of temporal-compression "bleeding" -- rather than
    relying on per-frame L1/LPIPS, which is blind to whether frames vary
    correctly relative to *each other*.
    """
    t_dim = 3 if is_multiview else 2
    if x.shape[t_dim] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    gt_diff = x.diff(dim=t_dim)
    rec_diff = x_rec.diff(dim=t_dim)
    return F.l1_loss(rec_diff, gt_diff)


# Growing-interval cache for should_log_images: {(schedule, growth, max_gap): {...}}
_IMAGE_LOG_STEPS_CACHE = {}


def _get_image_log_steps(cfg, upto_step):
    schedule = tuple(
        sorted({int(s) for s in (cfg.get("image_log_schedule_steps", None) or cfg.get("log_schedule_steps", None) or []) if int(s) > 0})
    )
    growth = float(cfg.get("image_log_growth_factor", 1.5))
    max_gap = int(cfg.get("image_log_max_interval", 2000))
    base_gap = max(1, int(cfg.get("log_every", 10)))
    cache_key = (schedule, growth, max_gap, base_gap)
    cached = _IMAGE_LOG_STEPS_CACHE.get(cache_key)
    if cached is None:
        cached = {"steps": set(schedule), "last": max(schedule) if schedule else 0, "gap": base_gap}
        _IMAGE_LOG_STEPS_CACHE[cache_key] = cached
    while cached["last"] < upto_step:
        cached["gap"] = min(max_gap, max(1, int(round(cached["gap"] * growth))))
        cached["last"] += cached["gap"]
        cached["steps"].add(cached["last"])
    return cached["steps"]


def should_log_images(step, cfg):
    """
    Like `should_log_update`, but for wandb reconstruction *images* specifically.

    Uses the same early `log_schedule_steps`, then backs off geometrically
    (multiply the gap by `image_log_growth_factor` each time, capped at
    `image_log_max_interval`) instead of the constant `log_every` interval used
    for scalar metrics. Long runs otherwise log full-resolution reconstruction
    grids forever at a fixed cadence, which is what fills up local/cloud wandb
    storage on multi-day sweeps. Scalars stay on the original fixed cadence
    (cheap); only images taper off.
    """
    if step is None or step <= 0:
        return False
    return step in _get_image_log_steps(cfg, step)


def should_log_update(step, cfg):
    """Decide whether to emit wandb logs / reconstruction plots at this optimizer update.

    Supports an *escalating* schedule via cfg["log_schedule_steps"] (a list of early
    one-off update steps) plus cfg["log_every"] (steady interval after the last early
    point). Example: log_schedule_steps=[3, 9, 15], log_every=21 -> log at updates
    3, 9, 15, then every 21 (36, 57, ...). With ~20s/update that's roughly 1, 3, 5 min
    then every ~7 min. If log_schedule_steps is unset/empty, falls back to the plain
    `step % log_every == 0` behavior.
    """
    if step is None or step <= 0:
        return False
    every = int(cfg.get("log_every", 10)) or 1
    early = cfg.get("log_schedule_steps", None)
    if early:
        early = sorted({int(s) for s in early if int(s) > 0})
        if step in early:
            return True
        last = early[-1]
        if step < last:
            return False
        return (step - last) % every == 0
    return step % every == 0


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
        x_orig_flat = x_orig_use.reshape(b * v, c, t, h, w)
        x_rec_flat = x_rec_use.reshape(b * v, c, t, h, w)
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

    # Row labels: one input row per view, then one reconstruction row per view.
    if is_multiview and num_views >= 2:
        row_labels = (
            [f"View {v + 1} – input" for v in range(num_views)]
            + [f"View {v + 1} – reconstr." for v in range(num_views)]
        )
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


def evaluate_model(
    model,
    dataloader,
    device,
    dtype,
    num_eval_samples=32,
    view_flatten_in_loss=True,
    use_ema=False,
    value_range="[0,1]",
    train_target_hw=None,
    train_target_frames=None,
    vis_max_samples=4,
):
    """
    Run a full evaluation pass over the dataset (or a subset).
    
    Args:
        model: The VAE model to evaluate (can be booster-wrapped or unwrapped)
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        dtype: Data type for evaluation
        num_eval_samples: Clip count to score (sum of batch sizes). None or <=0 = entire dataloader.
        vis_max_samples: Max clips in the W&B visualization grid (first batch only).
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
    # Paper diagnostics, accumulated per batch (see Sec. "diagnostics" in the paper):
    # bleed ratios (temporal), cross-view similarity (ghosting), per-frame PSNR profile.
    diag = {
        "bleed_ratio_within": [],
        "bleed_ratio_across": [],
        "xview_sim_rec": [],
        "xview_sim_gt": [],
        "psnr_per_frame": [],
    }

    visualization_samples = []
    num_collected = 0
    eval_all = num_eval_samples is None or (
        isinstance(num_eval_samples, (int, float)) and num_eval_samples <= 0
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not eval_all and num_collected >= int(num_eval_samples):
                break

            x = batch["video"].to(device, dtype)
            if train_target_hw is not None and train_target_frames is not None:
                x = downsample_video_tensor(
                    x,
                    target_h=int(train_target_hw[0]),
                    target_w=int(train_target_hw[1]),
                    target_t=int(train_target_frames),
                )
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

            # Diagnostics. Metrics expect [0,1]; undo the [-1,1] scaling first.
            if value_range == "[-1,1]":
                x01 = ((x.float() + 1.0) / 2.0).clamp(0, 1)
                xr01 = ((x_rec.float() + 1.0) / 2.0).clamp(0, 1)
            else:
                x01 = x.float().clamp(0, 1)
                xr01 = x_rec.float().clamp(0, 1)
            bleed = compute_intra_chunk_bleed_metrics(x01, xr01)
            if "bleed_ratio_within" in bleed:
                diag["bleed_ratio_within"].append(bleed["bleed_ratio_within"])
            if "bleed_ratio_across" in bleed:
                diag["bleed_ratio_across"].append(bleed["bleed_ratio_across"])
            if is_multiview:
                sim_rec = compute_cross_view_similarity(xr01)
                sim_gt = compute_cross_view_similarity(x01)
                if sim_rec is not None:
                    diag["xview_sim_rec"].append(sim_rec)
                    diag["xview_sim_gt"].append(sim_gt)
            pf = compute_metrics_per_frame(x01, xr01)
            diag["psnr_per_frame"].append([float(p) for p in pf["psnr_per_frame"]])

            # Visualization: first batch only; clip count is separate from metric coverage.
            if len(visualization_samples) == 0:
                n_vis = min(max(1, int(vis_max_samples)), int(x.shape[0]))
                vis_images = create_visualization_grid(
                    x, x_rec, num_samples=n_vis, value_range=value_range
                )
                visualization_samples.extend(vis_images)

            num_collected += x.shape[0]
    
    # Aggregate metrics
    aggregated = {
        "psnr_mean": np.mean(all_metrics["psnr"]),
        "psnr_std": np.std(all_metrics["psnr"]),
        "ssim_mean": np.mean(all_metrics["ssim"]),
        "ssim_std": np.std(all_metrics["ssim"]),
        "mse_mean": np.mean(all_metrics["mse"]),
        "mse_std": np.std(all_metrics["mse"]),
    }
    for key in ("bleed_ratio_within", "bleed_ratio_across", "xview_sim_rec", "xview_sim_gt"):
        if diag[key]:
            aggregated[key] = float(np.mean(diag[key]))
    if diag["psnr_per_frame"]:
        # Mean profile over batches; batches can have different T (buckets), so
        # average only over batches with the most common length.
        lengths = [len(p) for p in diag["psnr_per_frame"]]
        t_common = max(set(lengths), key=lengths.count)
        profile = np.mean([p for p in diag["psnr_per_frame"] if len(p) == t_common], axis=0)
        aggregated["psnr_per_frame"] = [float(v) for v in profile]
    
    # Return to training mode
    if hasattr(model, "module"):
        model.module.train()
    else:
        model.train()
    
    return {
        "metrics": aggregated,
        "visualizations": visualization_samples,
    }


def evaluate_fixed_sequence_per_frame(
    model, dataset, index, device, dtype, vae_target_range=None, train_target_hw=None, train_target_frames=None
):
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
    if train_target_hw is not None and train_target_frames is not None:
        x = downsample_video_tensor(
            x,
            target_h=int(train_target_hw[0]),
            target_w=int(train_target_hw[1]),
            target_t=int(train_target_frames),
        )
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

    # Normalize discriminator-related overrides immediately so run naming, W&B labels,
    # and training behavior all reflect the effective (post-CLI) configuration.
    from opensora.utils.vae_discriminator_presets import apply_discriminator_bundle_to_cfg, resolve_vae_discriminator_bundle

    _disc_choice = cfg.get("discriminator_choice", None)
    _disc_raw = cfg.get("discriminator", None)
    # Important: parser may coerce CLI `--discriminator_choice none` into Python None.
    # If user provided discriminator_choice explicitly (even None), it must override
    # any prebuilt dict from the base config.
    _has_disc_choice_key = False
    try:
        _has_disc_choice_key = "discriminator_choice" in cfg
    except Exception:
        _has_disc_choice_key = hasattr(cfg, "discriminator_choice")

    if _has_disc_choice_key:
        apply_discriminator_bundle_to_cfg(cfg, resolve_vae_discriminator_bundle(_disc_choice))
    elif not isinstance(_disc_raw, dict):
        # Backward compatibility for configs that set `discriminator` directly.
        apply_discriminator_bundle_to_cfg(cfg, resolve_vae_discriminator_bundle(_disc_raw))

    # Apply optional top-level GAN weight override only when discriminator is enabled.
    # Prefer gen_disc_weight: CLI passes --gen_disc_weight, which merge_args updates on cfg only
    # for that key. The config file also sets sweep_gen_disc_weight for compatibility; if we read
    # sweep first, a stale file value shadows every sweep job (all runs keep the same disc_weight).
    _gdw_top = cfg.get("gen_disc_weight", None)
    _gdw_sweep = cfg.get("sweep_gen_disc_weight", None)
    _sweep_gdw = _gdw_top if _gdw_top is not None else _gdw_sweep

    if cfg.get("discriminator", None) is None:
        cfg.discriminator_choice = "none"
        cfg.gen_disc_weight = None
        cfg.sweep_gen_disc_weight = None
    elif _sweep_gdw is not None and cfg.get("gen_loss_config") is not None:
        _g = dict(cfg.gen_loss_config)
        _g["disc_weight"] = float(_sweep_gdw)
        cfg.gen_loss_config = _g
        cfg.gen_disc_weight = float(_sweep_gdw)

    # Keep multi-view discriminator input layout consistent with the effective preset
    # after CLI overrides (base config may have been authored for a different default).
    _disc_choice_norm = str(cfg.get("discriminator_choice", "none")).strip().lower()
    if _disc_choice_norm in ("none", ""):
        cfg.disc_multiview_mode = "joint_4d"
        cfg.view_flatten_in_disc = False
    elif _disc_choice_norm == "train":
        cfg.disc_multiview_mode = "flatten_batch"
        cfg.view_flatten_in_disc = True
    elif _disc_choice_norm in ("trainmultiview4d", "train_multiview_4d", "train_mv4d"):
        cfg.disc_multiview_mode = "joint_4d"
        cfg.view_flatten_in_disc = False
    elif _disc_choice_norm in ("trainmultiviewstack", "train_multiview_stack", "train_mv_stack"):
        cfg.disc_multiview_mode = "stack_channels"
        cfg.view_flatten_in_disc = False

    # Keep top-level loss fields in sync with the effective nested config so
    # logs/config dumps do not show stale defaults after CLI nested overrides.
    _vlc = cfg.get("vae_loss_config", None)
    if isinstance(_vlc, dict):
        if "perceptual_loss_weight" in _vlc:
            cfg.perceptual_loss_weight = _vlc["perceptual_loss_weight"]
        if "kl_loss_weight" in _vlc:
            cfg.kl_loss_weight = _vlc["kl_loss_weight"]
        if "view_consistency_weight" in _vlc:
            cfg.view_consistency_weight = _vlc["view_consistency_weight"]
        if "logvar_init" in _vlc:
            cfg.logvar_init = _vlc["logvar_init"]

    # NeRSemble: ``bucket_config`` / ``DATA_ROOT`` are fixed at config import time. CLI
    # ``--bucket_config`` updates only the dict unless we re-run the resolver and reroot
    # ``dataset_presets`` paths that still point at the old ``DATA_ROOT``.
    _bucket_cfg = cfg.get("bucket_config", None)
    if _bucket_cfg:
        from opensora.utils.nersemble_bucket import resolve_nersemble_bucket

        _old_root = cfg.get("DATA_ROOT", None)
        _resolved = resolve_nersemble_bucket(
            dict(_bucket_cfg),
            processed_base=cfg.get("nersemble_processed_base"),
        )
        cfg.DATA_ROOT = _resolved["data_root"]
        cfg.train_target_hw = _resolved["train_target_hw"]
        cfg.train_target_frames = _resolved["train_target_frames"]
        if _old_root is not None and isinstance(_old_root, str) and _old_root != cfg.DATA_ROOT:
            for _preset_name in ("dataset_presets", "val_dataset_presets"):
                _presets = cfg.get(_preset_name)
                if not isinstance(_presets, dict):
                    continue
                for _ds in _presets.values():
                    if _ds is None or not isinstance(_ds, dict):
                        continue
                    _p = _ds.get("data_path")
                    if isinstance(_p, str) and _p.startswith(_old_root):
                        _ds["data_path"] = cfg.DATA_ROOT + _p[len(_old_root) :]
        print(
            "[wan_multiview] effective data: "
            f"epochs={cfg.get('epochs')} "
            f"DATA_ROOT={cfg.get('DATA_ROOT')} "
            f"train_target_hw={cfg.get('train_target_hw')} "
            f"train_target_frames={cfg.get('train_target_frames')} "
            f"bucket_config={dict(_bucket_cfg)}"
        )

    # Some configs construct `cfg.dataset` during initial config parsing, but
    # CLI overrides are merged afterwards. Rebuild dataset configs here so
    # overrides take effect.
    dataset_presets = cfg.get("dataset_presets", None) if hasattr(cfg, "get") else getattr(cfg, "dataset_presets", None)
    val_dataset_presets = cfg.get("val_dataset_presets", None) if hasattr(cfg, "get") else getattr(cfg, "val_dataset_presets", None)
    if isinstance(dataset_presets, dict):
        preset = cfg.get("data_preset", None) if hasattr(cfg, "get") else getattr(cfg, "data_preset", None)
        default_key = "__default__"
        key = preset if preset in dataset_presets else default_key
        if key in dataset_presets:
            cfg.dataset = dataset_presets[key]
        if isinstance(val_dataset_presets, dict) and key in val_dataset_presets:
            cfg.val_dataset = val_dataset_presets[key]

    # Backward compatibility: some configs may expose a callable hook.
    build_dataset_fn = None
    if hasattr(cfg, "get"):
        build_dataset_fn = cfg.get("build_dataset", None)
    if build_dataset_fn is None:
        build_dataset_fn = getattr(cfg, "build_dataset", None)
    if callable(build_dataset_fn):
        dataset_cfg, val_dataset_cfg = build_dataset_fn(cfg)
        cfg.dataset = dataset_cfg
        cfg.val_dataset = val_dataset_cfg

    # Subsample T inside the dataset (before DataLoader collate). Bucket target is
    # usually 9 frames while many .pt files still store T=13; mixing them in one
    # batch crashes workers with "Trying to resize storage that is not resizable".
    _target_frames = cfg.get("train_target_frames", None)
    if _target_frames is not None:
        for _ds_key in ("dataset", "val_dataset"):
            _ds = cfg.get(_ds_key, None)
            if isinstance(_ds, dict) and _ds.get("target_frames", None) is None:
                _ds = dict(_ds)
                _ds["target_frames"] = int(_target_frames)
                cfg[_ds_key] = _ds
        print(
            f"[wan_multiview] dataset target_frames={int(_target_frames)} "
            "(normalize T before collate)"
        )

    apply_pytorch_determinism(cfg)

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
    configured_grad_clip = float(cfg.get("grad_clip", 0) or 0.0)
    # ZeRO2 clipping: ColossalAI's LowLevelZeroOptimizer (stage=2) clips grads internally inside
    # step() via _unscale_and_clip_grads over the grad_store shards — the only place where the
    # actual gradient tensors are accessible. The manual path (torch.clip_grad_norm_ +
    # clip_optimizer_grad_norm_global_) cannot reach those tensors and silently does nothing.
    # Therefore zero2 must use plugin-internal clipping (force_manual=False).
    # zero1 shards only optimizer states, not gradients, so param.grad is visible → manual works.
    # plugin="none" (no ColossalAI plugin): plain optimizer, param.grad is always visible → manual.
    force_manual_global_grad_clip = bool(
        cfg.get(
            "force_manual_global_grad_clip",
            configured_grad_clip > 0 and plugin_type in ("none", "zero1", "zero1-seq"),
        )
    )
    plugin_grad_clip = 0.0 if force_manual_global_grad_clip else configured_grad_clip
    plugin = (
        create_colossalai_plugin(
            plugin=plugin_type,
            dtype=cfg.get("dtype", "bf16"),
            grad_clip=plugin_grad_clip,
            **plugin_config,
        )
        if plugin_type != "none"
        else None
    )
    booster = Booster(plugin=plugin)

    # == init exp_dir (folder name includes loss weights unless experiment_name is set) ==
    loss_slug = _loss_signature_slug(cfg)
    model_name_for_dir = f"{config_to_name(cfg)}_{loss_slug}"[:200]
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=model_name_for_dir,
        config=cfg.to_dict(),
        exp_name=cfg.get("experiment_name"),
    )
    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        import os
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger ==
    logger = create_logger(exp_dir)
    if cfg.get("profile_timing", False):
        cfg.wandb = False
        if cfg.get("profile_step", False):
            logger.info("profile_timing enabled: disabling profile_step (Kineto trace)")
            cfg.profile_step = False
        logger.info(
            "profile_timing enabled: wandb disabled; will profile once at global_step=%s",
            int(cfg.get("profile_timing_step", 50)),
        )
    if cfg.get("deterministic", True):
        logger.info(
            "Determinism mode: cuBLAS workspace, cudnn deterministic, TF32 disabled, "
            "torch.use_deterministic_algorithms(warn_only=True); data workers use seed+worker_id; "
            "epoch Python RNG uses seed+rank. Multi-GPU ZeRO and BF16 may still break bit-for-bit replay."
        )
    if force_manual_global_grad_clip and configured_grad_clip > 0:
        logger.info(
            "Using explicit global grad clipping (max_norm=%s) with DP all-reduce; "
            "plugin-internal clipping disabled to avoid ZeRO shard-local clipping ambiguity.",
            configured_grad_clip,
        )
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        append_eval_metrics_jsonl(
            exp_dir,
            {
                "kind": "meta",
                "loss_config": _loss_config_dict(cfg),
                "loss_signature_slug": loss_slug,
                "exp_dir": os.path.abspath(exp_dir),
            },
        )
        with open(os.path.join(exp_dir, "loss_config.json"), "w", encoding="utf-8") as _lf:
            json.dump(_loss_config_dict(cfg), _lf, indent=2)




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
    _persistent_dl_extras = {}
    if int(cfg.get("num_workers", 0) or 0) > 0 and cfg.get("persistent_workers", False):
        _persistent_dl_extras["persistent_workers"] = True
    dataloader_args.update(_persistent_dl_extras)
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

    # Auto-detect view_in from the first dataset sample when not explicitly set (or set to 0).
    # This makes the config portable: the model always matches the actual data shape.
    if not cfg.model.get("view_in"):
        try:
            _sample_video = dataset[0]["video"]  # [V, C, T, H, W]
            if _sample_video.dim() == 5:
                _detected_views = int(_sample_video.shape[0])
                cfg.model["view_in"] = _detected_views
                logger.info("Auto-detected model.view_in=%d from first dataset sample", _detected_views)
        except Exception as _e:
            logger.warning("Could not auto-detect view_in from dataset: %s", _e)

    # == build vae model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    log_model_params(model)
    if is_log_process(plugin_type, plugin_config):
        if cfg.get("log_training_design_summary", False):
            pass  # deferred to first training step (after booster.wrap) — see training loop
        elif cfg.get("log_training_param_overview", True):
            log_trainable_param_overview(model, "VAE")

    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)

    _optimize = cfg.get("optimization", False)
    if _optimize:
        # 1. Gradient checkpointing: trade compute for memory, allows bigger batches.
        if not cfg.get("grad_checkpoint", False):
            logger.info("[optimization] grad_checkpoint: ON (was not set in config)")
            set_grad_checkpoint(model)

    vae_loss_fn = VAELoss(**cfg.vae_loss_config, device=device, dtype=dtype)

    # == optional distillation teacher (Idea 5) ==
    # A frozen temporal_compression=False copy of the model. Its reconstructions are
    # clean along the temporal axis, so distill_weight * L1(student_rec, teacher_rec)
    # gives the compressed student a dense per-frame signal for what good temporal
    # reconstruction looks like. No inference-time cost (training-only loss).
    teacher_model = None
    distill_weight = float(cfg.get("distill_weight", 1.0))
    if cfg.get("distill_teacher_ckpt", None):
        teacher_cfg = dict(cfg.model)
        teacher_cfg["temporal_compression"] = False
        # The teacher is a plain baseline: strip any round-2 idea flags.
        for _k in (
            "use_noncausal_decode",
            "use_temporal_reflection_pad",
            "use_temporal_side_channel",
            "side_channel_dim",
            "use_decoder_temporal_attention",
            "use_learned_cache_update",
        ):
            teacher_cfg.pop(_k, None)
        logger.info(
            "[distill] Building frozen teacher (temporal_compression=False) and loading %s",
            cfg.distill_teacher_ckpt,
        )
        teacher_model = build_module(teacher_cfg, MODELS, device_map=device, torch_dtype=dtype)
        load_checkpoint(teacher_model, cfg.distill_teacher_ckpt)
        teacher_model = teacher_model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        logger.info("[distill] Teacher ready (frozen, eval mode); distill_weight=%s", distill_weight)

    # == build EMA model ==
    # EMA is deepcopied BEFORE channels-last conversion: model_sharding calls
    # param.data.view(-1) which requires contiguous tensors.
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")

    if _optimize:
        # 2. channels_last_3d is disabled: it causes .view() failures throughout the codebase
        #    whenever a data tensor (not just a weight) ends up with non-contiguous strides,
        #    and it requires restoring strides after every checkpoint save to keep torch.compile
        #    happy. The ~32ms/step saving is not worth the fragility. torch.compile alone
        #    (step 1) gives the main speedup.
        logger.info("[optimization] channels_last_3d: disabled (fragile with view() ops and compile cache)")

    def _freeze_first_child_modules(module, num_children: int):
        """Freeze parameters in the first N immediate child modules."""
        if num_children <= 0:
            return
        child_modules = list(module.named_children())
        for _, child in child_modules[:num_children]:
            child.requires_grad_(False)

    # == build discriminator model ==

    use_discriminator = cfg.get("discriminator", None) is not None
    if use_discriminator:
        disc_cfg = cfg.discriminator
        disc_type = disc_cfg.get("type")
        if disc_type == "pretrained_stylegan2_discriminator":
            from huggingface_hub import hf_hub_download

            repo_id = disc_cfg.get("repo_id", "mukhbiir/StyleGAN2_Discriminator")
            filename = disc_cfg.get("filename", "model.pt")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.nn.Module):
                discriminator = loaded.to(device).train()
            else:
                discriminator = build_stylegan2_ada_discriminator_from_state_dict(
                    loaded,
                    device,
                    img_resolution=int(disc_cfg.get("img_resolution", 512)),
                    img_channels=int(disc_cfg.get("img_channels", 3)),
                    stylegan2_ada_root=disc_cfg.get("stylegan2_ada_root"),
                ).train()
            freeze_layers = int(disc_cfg.get("freeze_layers", 0) or 0)
            # Pretrained path may wrap NVlabs D in a module with a single child `core`;
            # freeze sub-blocks on the inner core so freeze_layers matches StyleGAN blocks.
            _freeze_first_child_modules(getattr(discriminator, "core", discriminator), freeze_layers)
            logger.info(
                "Loaded pretrained discriminator from %s/%s (preset=%s, freeze_layers=%d)",
                repo_id,
                filename,
                disc_cfg.get("pretrained", "unknown"),
                freeze_layers,
            )
        elif disc_type == "pretrained_sd_vae_nlayer_discriminator":
            from opensora.models.vae.sd_ldm_discriminator import build_pretrained_sd_vae_nlayer_discriminator

            disc_kwargs = {k: v for k, v in disc_cfg.items() if k != "type"}
            discriminator = build_pretrained_sd_vae_nlayer_discriminator(**disc_kwargs).to(device, dtype).train()
            freeze_layers = int(disc_cfg.get("freeze_layers", 0) or 0)
            logger.info(
                "Loaded LDM PatchGAN (SD VAE) discriminator repo_id=%s filename=%s freeze_layers=%d",
                disc_cfg.get("repo_id", "stabilityai/sd-vae-ft-mse"),
                disc_cfg.get("filename", "diffusion_pytorch_model.bin"),
                freeze_layers,
            )
        else:
            # Ensure discriminator classes are registered in the MODELS registry.
            # Some training entrypoints may not import the discriminator module by default.
            import opensora.models.vae.discriminator  # noqa: F401

            # For the multiview disc, auto-detect num_views from the first dataset sample so the
            # view_merge conv is sized correctly regardless of how many cameras are in the data.
            if disc_type == "N_Layer_discriminator_multiview_4d" and not disc_cfg.get("num_views"):
                try:
                    _probe = dataset[0]
                    _probe_vid = _probe["video"] if isinstance(_probe, dict) else _probe
                    _detected_views = int(_probe_vid.shape[0])
                    disc_cfg["num_views"] = _detected_views
                    logger.info("[disc] auto-detected num_views=%d from dataset probe", _detected_views)
                except Exception as _probe_err:
                    logger.warning(
                        "[disc] Could not auto-detect num_views (%s); falling back to constructor default", _probe_err
                    )

            discriminator = build_module(disc_cfg, MODELS).to(device, dtype).train()
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
    if is_log_process(plugin_type, plugin_config):
        logger.info("VAE trainable params (optimizer): %s", f"{total_trainable:,}")

    # 3. torch.compile: fuses small ops, eliminates Python/autograd overhead.
    #    Applied before booster.boost() so ColossalAI wraps the compiled graph.
    #    First run is slow (compilation); subsequent steps get the full speedup.
    if _optimize:
        _compile_mode = cfg.get("optimization_compile_mode", "reduce-overhead")
        # dynamic=None -> torch auto-detects (legacy behavior). dynamic=True compiles a
        # single shape-flexible graph, which avoids the per-chunk recompiles caused by the
        # temporal feat_cache loop (variable chunk sizes). Use with mode="default" (no CUDA
        # graphs) so it stays compatible with gradient checkpointing.
        _compile_dynamic = cfg.get("optimization_compile_dynamic", None)
        logger.info(
            "[optimization] torch.compile: mode=%s dynamic=%s (first step will be slow)",
            _compile_mode,
            _compile_dynamic,
        )
        try:
            if _compile_dynamic is None:
                model = torch.compile(model, mode=_compile_mode)
            else:
                model = torch.compile(model, mode=_compile_mode, dynamic=_compile_dynamic)
        except Exception as _e:
            logger.warning("[optimization] torch.compile failed, continuing without it: %s", _e)

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
    optimizer_has_booster_backward = hasattr(optimizer, "backward")
    if is_log_process(plugin_type, plugin_config) and not optimizer_has_booster_backward:
        logger.warning(
            "Optimizer %s has no `.backward`; using booster.backward(loss=...) compatibility path.",
            optimizer.__class__.__name__,
        )
    disc_optimizer_has_booster_backward = False
    if use_discriminator:
        disc_optimizer_has_booster_backward = hasattr(disc_optimizer, "backward")
        if is_log_process(plugin_type, plugin_config) and not disc_optimizer_has_booster_backward:
            logger.warning(
                "Discriminator optimizer %s has no `.backward`; using booster.backward(loss=...) compatibility path.",
                disc_optimizer.__class__.__name__,
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
    # Multi-view 3D discriminator input: flatten_batch | stack_channels | joint_4d (see wan_multiview_finetune.py).
    disc_multiview_mode = cfg.get("disc_multiview_mode", "flatten_batch")
    disc_per_frame_2d = cfg.get("disc_per_frame_2d", False)
    # Safety guard: plain 3D PatchGAN discriminator expects 5D input [B,C,T,H,W].
    # If multiview clips are used, force view flattening into batch to avoid 6D conv3d errors.
    if use_discriminator and not disc_per_frame_2d:
        _disc_cfg = cfg.get("discriminator", None)
        _disc_type = _disc_cfg.get("type") if isinstance(_disc_cfg, dict) else None
        if _disc_type == "N_Layer_discriminator_3D":
            if disc_multiview_mode != "flatten_batch" or not view_flatten_in_disc:
                logger.warning(
                    "Overriding discriminator input layout for N_Layer_discriminator_3D: "
                    "forcing disc_multiview_mode='flatten_batch' and view_flatten_in_disc=True."
                )
            disc_multiview_mode = "flatten_batch"
            view_flatten_in_disc = True
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
        distill=0.0,
        temporal_diff=0.0,
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
            load_optimizer=cfg.get("load_optimizer", True),
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
            if sampler is not None and hasattr(sampler, "set_step"):
                sampler.set_step(start_step)
            elif sampler is not None:
                logger.warning(
                    "Sampler %s has no set_step(); resume will continue from epoch boundary without sampler micro-step offset.",
                    sampler.__class__.__name__,
                )

        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]

        # If the resumed step is at or beyond the end of an epoch, roll over to the next epoch.
        # This happens when a checkpoint was saved at the last step of an epoch (step == num_steps_per_epoch).
        if start_step >= num_steps_per_epoch:
            start_epoch += start_step // num_steps_per_epoch
            start_step = start_step % num_steps_per_epoch
            if sampler is not None and hasattr(sampler, "set_step"):
                sampler.set_step(start_step)

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

        # Optionally re-randomize view-fusion attention after loading a TC checkpoint,
        # so the temporal backbone is kept but cross-view attention starts fresh.
        if cfg.model.get("reinit_view_attention_after_load", False):
            core = model.module if hasattr(model, "module") else model
            cv = getattr(core, "crossview_vae", None)
            n_reset = 0
            for attr in ("cross_attn", "joint_attn"):
                attn = getattr(cv, attr, None) if cv is not None else None
                if attn is None:
                    continue
                for name, mod in attn.named_modules():
                    if hasattr(mod, "reset_parameters"):
                        mod.reset_parameters()
                        n_reset += 1
                # Keep residual identity at init for ViewAttention.proj if present
                if hasattr(attn, "proj"):
                    import torch.nn as nn
                    nn.init.zeros_(attn.proj.weight)
                    if getattr(attn.proj, "bias", None) is not None:
                        nn.init.zeros_(attn.proj.bias)
            logger.info(
                "reinit_view_attention_after_load=True: reset fusion attention modules "
                "(%d submodules with reset_parameters)",
                n_reset,
            )

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
    # model_sharding calls param.data.view(-1) which requires contiguous tensors.
    # channels_last_3d conversion (applied earlier for optimization) makes 5D tensors
    # non-contiguous in the default stride order, so we make them contiguous first.
    if ema is not None:
        for _p in ema.parameters():
            if not _p.data.is_contiguous():
                _p.data = _p.data.contiguous()
        for _b in ema.buffers():
            if not _b.is_contiguous():
                _b.data = _b.data.contiguous()
        model_sharding(ema)
        ema = ema.to(device)

    if cfg.get("freeze_layers", None) == "all":
        for param in model.module.parameters():
            param.requires_grad = False
        print("all layers frozen")

    # model.module.requires_grad_(False)
    # =======================================================
    # 5. Wandb: initialized lazily after wandb_min_steps_before_init (default 10) so short
    #    smoke runs do not create empty online runs. Charts use optimizer step (wandb.log step=...).
    # =======================================================
    def _build_wandb_run_name():
        explicit = cfg.get("wandb_expr_name")
        if explicit:
            # Keep user label but append loss signature so runs remain distinguishable.
            return f"{str(explicit).strip()}__{loss_slug}"[:256]

        def _tok(s):
            t = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))
            return "_".join(x for x in t.split("_") if x)

        m = cfg.model
        bucket_key = next(iter(cfg.get("bucket_config") or {}), "")
        bucket_s = _tok(bucket_key.replace(":", "").replace("/", ""))

        # View decoding: view-wise decoder LoRA vs per-view latent embeddings vs neither
        if m.get("use_viewwise_decoder_lora"):
            view_dec = "vwlora"
        elif m.get("use_view_embedding"):
            view_dec = "vemb"
        else:
            view_dec = "novw"

        thw = cfg.get("train_target_hw")
        if thw is not None and len(thw) >= 2:
            h, w = int(thw[0]), int(thw[1])
            px_part = f"{h}x{w}px" if h != w else f"{h}px"
        else:
            px_part = bucket_s or "hwunk"

        tf = cfg.get("train_target_frames")
        t_part = f"{int(tf)}t" if tf is not None else "tunk"

        disc_cfg = cfg.get("discriminator")
        if disc_cfg is None:
            vlp = _tok(cfg.get("vae_loss_preset", "default"))
            loss_part = f"loss_{vlp}"
        elif disc_cfg.get("type") in (
            "pretrained_stylegan2_discriminator",
            "pretrained_sd_vae_nlayer_discriminator",
        ):
            loss_part = "loss_disc_pretrained"
        else:
            loss_part = "loss_disc_scratch"

        parts = [
            _tok(m.get("fusion_mode", "fusion")),
            "lb" if m.get("use_lora_before") else "nolb",
            "la" if m.get("use_lora_after") else "nola",
            view_dec,
            f"r{m.get('lora_rank', 16)}",
            px_part,
            t_part,
            _tok(cfg.get("data_preset", "") or "data"),
            loss_part,
            loss_slug,
            f"{m.get('view_in', 2)}v",
        ]
        name = "_".join(str(p) for p in parts if str(p))
        ts = exp_name.split("_")[0] if "_" in exp_name else exp_name[:8]
        return f"{name}_{ts}"[:256]

    def maybe_init_wandb(actual_update_step):
        if not coordinator.is_master() or not cfg.get("wandb", False):
            return
        if wandb.run is not None:
            return
        min_w = int(cfg.get("wandb_min_steps_before_init", 10))
        if actual_update_step <= min_w:
            return
        wandb_name = _build_wandb_run_name()
        logger.info(
            "Initializing wandb after step %s (threshold %s) with run name: %s",
            actual_update_step,
            min_w,
            wandb_name,
        )
        wandb_init_kwargs = dict(
            project=cfg.get("wandb_project", "vae"),
            name=wandb_name,
            config=cfg.to_dict(),
            dir=exp_dir,
        )
        wandb_group = cfg.get("wandb_group", None) or os.environ.get("WANDB_RUN_GROUP")
        if wandb_group:
            wandb_init_kwargs["group"] = wandb_group
        run_id_path = os.path.join(exp_dir, "wandb_run_id.txt")
        if cfg.get("load", None) is not None and os.path.isfile(run_id_path):
            with open(run_id_path, encoding="utf-8") as f:
                prev_run_id = f.read().strip()
            if prev_run_id:
                wandb_init_kwargs["id"] = prev_run_id
                wandb_init_kwargs["resume"] = "allow"
                logger.info("Resuming wandb run id=%s", prev_run_id)
        wandb.init(**wandb_init_kwargs)
        # Register samples_seen as the primary x-axis so all charts are comparable
        # across runs with different batch sizes (e.g. bs=1 disc runs vs bs=16 idea runs).
        wandb.define_metric("samples_seen")
        wandb.define_metric("*", step_metric="samples_seen")
        wandb.config.update(
            {"loss_config": _loss_config_dict(cfg), "loss_signature_slug": loss_slug},
            allow_val_change=True,
        )
        run_id = wandb.run.id
        run_url = getattr(wandb.run, "url", None) or ""
        out_abs = os.path.abspath(exp_dir)
        with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as f:
            f.write(run_id + "\n")
        with open(os.path.join(exp_dir, "wandb_run_url.txt"), "w", encoding="utf-8") as f:
            f.write(run_url + "\n")
        with open(os.path.join(exp_dir, "server_output_dir.txt"), "w", encoding="utf-8") as f:
            f.write(out_abs + "\n")
        cfg_path = getattr(cfg, "config_path", None)
        if cfg_path and os.path.isfile(cfg_path):
            snap = os.path.join(exp_dir, "training_config_snapshot.py")
            shutil.copy2(cfg_path, snap)
            wandb.save(snap)
        wandb.config.update(
            {
                "experiment_folder": out_abs,
                "wandb_run_id": run_id,
                "wandb_run_url": run_url,
                "config_path": cfg_path or "",
                "sweep_config": cfg.get("sweep_config", cfg_path or ""),
            },
            allow_val_change=True,
        )
        logger.info("Wandb run URL: %s", run_url)

    # =======================================================
    # 5b. warm up torch.compile eval graph before the training loop
    # =======================================================
    # torch.compile caches one graph per (function, mode). The train graph is compiled on
    # the first training forward. The eval graph (.eval() mode) is a *different* graph and
    # would compile on the first full_eval call — causing a multi-minute stall mid-training.
    # We trigger that compilation now, before the loop, so it's out of the way.
    if _optimize and cfg.get("full_eval_every", 0) > 0:
        logger.info("[optimization] Warming up torch.compile eval graph (one-time, before training loop)...")
        _sample = dataset[0]["video"].unsqueeze(0).to(device, dtype)
        _sample = apply_train_bucket_spatiotemporal(_sample, cfg)
        if cfg.get("vae_target_range") == "[-1,1]" or cfg.model.get("type") == "multiview_wan_video_vae":
            _sample = _sample * 2.0 - 1.0
        _inner = unwrap_model_safe(model)
        _inner.eval()
        with torch.no_grad():
            try:
                model(_sample)
                logger.info("[optimization] Eval graph warm-up done.")
            except Exception as _we:
                logger.warning("[optimization] Eval graph warm-up failed (non-fatal): %s", _we)
        _inner.train()
        del _sample, _inner

    # =======================================================
    # 6. training loop
    # =======================================================
    first_training_global_step = start_epoch * num_steps_per_epoch + start_step

    dist.barrier()
    accumulation_steps = int(cfg.get("accumulation_steps", 1))
    # Samples seen per optimizer update = micro-batch × accumulation × world_size.
    # Used as the canonical x-axis in WandB so runs with different batch sizes are comparable.
    num_processes = dist.get_world_size() if dist.is_initialized() else 1
    effective_batch_size = cfg.get("batch_size", 1) * accumulation_steps * num_processes
    actual_update_step = 0
    debug_stats_start_step = int(cfg.get("debug_stats_start_step", 0))
    debug_stats_every = max(1, int(cfg.get("debug_stats_every", 500)))
    debug_stats_weight_every = max(1, int(cfg.get("debug_stats_weight_every", 500)))
    # Train-batch PSNR guard (rank 0): enforce on optimizer update boundaries (not micro-steps)
    # so behavior matches logged train_batch snapshots when accumulation_steps > 1.
    train_psnr_guard_threshold = float(
        cfg.get("train_psnr_guard_threshold", cfg.get("psnr_guard_threshold", 15.0))
    )
    train_psnr_guard_consecutive = max(
        1, int(cfg.get("train_psnr_guard_consecutive", cfg.get("psnr_guard_required_consecutive", 3)))
    )
    train_psnr_guard_start_step = max(0, int(cfg.get("train_psnr_guard_start_step", 15000)))
    train_psnr_guard_start_epoch = cfg.get("train_psnr_guard_start_epoch", None)
    if train_psnr_guard_start_epoch is None:
        # Backward compatibility: if only step-based start is provided, convert to epoch index.
        train_psnr_guard_start_epoch = train_psnr_guard_start_step // max(1, num_steps_per_epoch)
    train_psnr_guard_start_epoch = max(0, int(train_psnr_guard_start_epoch))
    train_psnr_guard_min_epochs = max(0, int(cfg.get("train_psnr_guard_min_epochs", 0)))
    train_psnr_guard_min_updates = max(0, int(cfg.get("train_psnr_guard_min_updates", 0)))
    train_psnr_low_streak = 0
    # Mirror image of the guard, for the Stage-1 overfit gate: stop as soon as the
    # epoch-mean train PSNR holds ABOVE a target for K consecutive epochs. 0.0 = off.
    # Do NOT set this on generalization runs -- those keep the fixed budget so that
    # all arms are compared at identical optimizer-update counts.
    stop_at_train_psnr = float(cfg.get("stop_at_train_psnr", 0.0))
    stop_at_train_psnr_consecutive = max(1, int(cfg.get("stop_at_train_psnr_consecutive", 3)))
    train_psnr_target_streak = 0
    train_psnr_bad_for_ckpt = False
    last_epoch_psnr_mean = float("nan")
    last_saved_ckpt_epoch = -1
    early_stop_requested = False
    _psnr_stop_log_once = False
    _profile_timing_done = False
    _profile_timing_step = int(cfg.get("profile_timing_step", 50))
    # Live per-block GPU memory printing from step 0 (independent of profile_timing).
    # Use this to find WHERE an OOM happens: the last "[mem] ..." line printed before
    # the crash is the block that pushed allocation over the GPU limit. Requires eager
    # mode (optimization=False) for meaningful numbers; torch.compile fuses/reorders ops
    # and graph-breaks on the timer's cuda.synchronize, so per-block attribution is lost.
    if cfg.get("profile_memory_live", False) and coordinator.is_master():
        ProfileTimer.enable(reset=True)
        ProfileTimer.set_live_memory(True)
        logger.warning(
            "[profile] profile_memory_live=True: printing per-block GPU memory every step "
            "from step 0. Set optimization=False for valid per-method numbers, and turn this "
            "OFF for real training (the per-block cuda.synchronize adds overhead)."
        )
    # One-time average wall time over first 10 steps; set log_step_time False to disable.
    _log_step_time_once = cfg.get("log_step_time", True)
    _step_time_bench_done = False
    _step_time_bench_samples = []
    if coordinator.is_master():
        append_training_debug_jsonl(
            exp_dir,
            {
                "kind": "meta",
                "dtype_config": str(cfg.get("dtype", "bf16")),
                "torch_dtype_runtime": str(dtype),
                "debug_stats_start_step": debug_stats_start_step,
                "debug_stats_every": debug_stats_every,
                "debug_stats_weight_every": debug_stats_weight_every,
                "train_psnr_guard_threshold": train_psnr_guard_threshold,
                "train_psnr_guard_consecutive": train_psnr_guard_consecutive,
                "train_psnr_guard_start_epoch": train_psnr_guard_start_epoch,
                "train_psnr_guard_start_step": train_psnr_guard_start_step,
                "train_psnr_guard_min_epochs": train_psnr_guard_min_epochs,
                "train_psnr_guard_min_updates": train_psnr_guard_min_updates,
            },
        )
    for epoch in range(start_epoch, cfg_epochs):
        epoch_psnr_sum = 0.0
        epoch_psnr_count = 0
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        random.seed(int(cfg.get("seed", 1024)) + dist.get_rank())

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataiter, start=start_step),
            desc="train",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            if coordinator.is_master():
                pbar.set_postfix(epoch=epoch)
            pbar_iter = iter(pbar)
            current_update_sample_ids = []

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
                    if early_stop_requested:
                        break
                    if cfg.get("profile", False) and _ == WARMUP + ACTIVE + WAIT + 3:
                        break

                    # Start timing the entire step - this helps us see overall throughput
                    step_start_time = time.time()
                    
                    # == load data ===
                    # Data loading can be a major bottleneck, especially with large videos
                    # We time this separately to see if we're GPU-starved (waiting for data)
                    data_load_start = time.time()
                    batch, step, pinned_video = batch_, step_, pinned_video_
                    global_step = epoch * num_steps_per_epoch + step
                    if (global_step + 1) % accumulation_steps == 1:
                        current_update_sample_ids = []
                    step_sample_ids = _extract_batch_sample_ids(batch)
                    if step_sample_ids is not None:
                        current_update_sample_ids.append(step_sample_ids)
                    
                    import sys
                    sys.stdout.flush()
                    if step + 1 < num_steps_per_epoch:
                        batch_, step_, pinned_video_ = fetch_data()
                    data_load_time = time.time() - data_load_start
                    timing_stats["data_load"].append(data_load_time)

                    # == log config ==
                    # global_step already computed above (needed before accumulation checks)
                    actual_update_step = (global_step + 1) // accumulation_steps
                    log_step += 1
                    acc_step += 1
                    # Epoch coordinate for plotting in W&B (logged as a metric; charts use optimizer step).
                    # global_step is 0-indexed; +1 makes the first update land at ~1/steps_per_epoch.
                    epoch_float = (global_step + 1) / max(1, num_steps_per_epoch)

                    maybe_init_wandb(actual_update_step)

                    if (
                        is_log_process(plugin_type, plugin_config)
                        and cfg.get("log_training_design_summary", False)
                        and global_step == first_training_global_step
                    ):
                        if cfg.model.get("type") == "multiview_wan_video_vae":
                            log_wan_multiview_training_design_summary(model, cfg.model, emit=logger.info)
                        else:
                            logger.info(
                                "log_training_design_summary: model.type is not multiview_wan_video_vae; "
                                "printing generic trainable overview only."
                            )
                            log_trainable_param_overview(
                                model, "VAE", emit=logger.info, detailed_lora_buckets=True
                            )

                    # == mixed strategy ==
                    x = batch["video"]
                    x = apply_train_bucket_spatiotemporal(x, cfg)
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

                    # Full-step profiler: fires once on step 5 (warmed up) when profile_step=True.
                    # Use profile_step=True instead of profile=True — they cannot run simultaneously
                    # because profile=True already holds a live Kineto session over the whole epoch.
                    _do_full_profile = cfg.get("profile_step", False) and global_step == 14
                    _do_profile_timing = (
                        cfg.get("profile_timing", False)
                        and not _profile_timing_done
                        and global_step == _profile_timing_step
                    )
                    if _do_profile_timing and coordinator.is_master():
                        ProfileTimer.enable(reset=True)
                        logger.info(
                            "[ProfileTiming] Profiling global_step=%s (CUDA-synchronized block timers)",
                            global_step,
                        )
                    _prof = None
                    if _do_full_profile:
                        _prof = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                        )
                        _prof.__enter__()

                    # == forward pass ==
                    # Initialize loss_dict and vae_loss at the start of each step to ensure they're always available
                    loss_dict = {}  # loss at every step
                    vae_loss = torch.tensor(0.0, device=device, dtype=dtype)  # total VAE loss

                    # channels-last input: convert 5-D inputs directly; for 6-D
                    # multiview [B,V,C,T,H,W] convert each view slice since the
                    # model slices them to 5-D before any conv anyway.
                    if _optimize:
                        if x.dim() == 5:
                            x = x.contiguous(memory_format=torch.channels_last_3d)
                        elif x.dim() == 6:
                            b, v, c, t, h, w = x.shape
                            x = x.reshape(b * v, c, t, h, w).contiguous(
                                memory_format=torch.channels_last_3d
                            ).reshape(b, v, c, t, h, w)

                    # The forward pass (encode + decode) is usually the most expensive operation
                    # We time it carefully to see if it's the bottleneck
                    forward_start = time.time()
                    _forward_ctx = ProfileTimer.block("train.forward") if _do_profile_timing else nullcontext()
                    with _forward_ctx:
                        with record_function("forward") if _do_full_profile else nullcontext():
                            x_rec, posterior, z = model(x)
                    forward_time = time.time() - forward_start
                    timing_stats["forward"].append(forward_time)

                    # Train-batch PSNR/SSIM/MSE every step (master + video): same as wandb train_batch/*.
                    if coordinator.is_master() and use_video == 1:
                        with torch.no_grad():
                            _bm = compute_metrics(x, x_rec)
                        loss_dict["psnr"] = _bm["psnr"]
                        loss_dict["ssim"] = _bm["ssim"]
                        loss_dict["mse"] = _bm["mse"]
                        if is_multiview and cfg.get("temporal_compression", False):
                            _bleed = compute_intra_chunk_bleed_metrics(
                                x, x_rec, chunk_size=int(cfg.get("temporal_bleed_chunk_size", 4))
                            )
                            loss_dict.update(_bleed)
                        on_update_boundary = ((global_step + 1) % accumulation_steps == 0)
                        if cfg.get("train_psnr_guard", True) and on_update_boundary:
                            try:
                                _tb = float(_bm["psnr"])
                            except (TypeError, ValueError):
                                _tb = float("nan")
                            if not math.isnan(_tb):
                                # Keep per-update quality gate for checkpoint eligibility.
                                train_psnr_bad_for_ckpt = _tb < train_psnr_guard_threshold
                                # Epoch-level guard aggregation for early stop decision.
                                epoch_psnr_sum += _tb
                                epoch_psnr_count += 1

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
                    _loss_ctx = ProfileTimer.block("train.loss") if _do_profile_timing else nullcontext()
                    with _loss_ctx:
                        loss_start = time.time()
                        # Reset vae_loss (it was initialized at the start of the step)
                        vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                        # loss_dict is already initialized at the start of the step

                        ret = vae_loss_fn(x_loss, x_rec_loss, posterior_loss)

                        # View consistency loss: encourage reconstructions to stay coherent
                        # across all view pairs (all-to-all).
                        view_loss = 0.0
                        if is_multiview and x_rec.shape[1] > 1:
                            # Compute pairwise MSE over all distinct view pairs.
                            view_losses = []
                            num_views = x_rec.shape[1]
                            for i in range(num_views):
                                for j in range(i + 1, num_views):
                                    view_losses.append(F.mse_loss(x_rec[:, i], x_rec[:, j]))
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

                        # Idea 5 — teacher distillation: match the compressed student's
                        # output to the frozen temporal_compression=False teacher's.
                        distill_loss = None
                        if teacher_model is not None:
                            with torch.no_grad():
                                x_rec_teacher, _, _ = teacher_model(x)
                            distill_loss = F.l1_loss(x_rec, x_rec_teacher)
                            vae_loss = vae_loss + distill_weight * distill_loss

                        # Idea 7 — temporal-difference loss: penalize wrong frame-to-frame
                        # deltas directly (see compute_temporal_diff_loss docstring). Zero new
                        # parameters; complements (does not replace) the discriminator/encoder
                        # unfreeze experiments for the temporal-compression bleeding problem.
                        temporal_diff_loss = None
                        temporal_diff_loss_weight = float(cfg.get("temporal_diff_loss_weight", 0.0))
                        if temporal_diff_loss_weight > 0.0:
                            temporal_diff_loss = compute_temporal_diff_loss(x, x_rec, is_multiview)
                            vae_loss = vae_loss + temporal_diff_loss_weight * temporal_diff_loss
                            loss_dict["temporal_diff_loss"] = temporal_diff_loss.item()
                        loss_time = time.time() - loss_start
                    timing_stats["loss_compute"].append(loss_time)

                    # == generator loss ==
                    # Discriminator forward pass can be expensive, especially with 3D convolutions
                    # We time it separately to see if it's slowing down training
                    adaptive_w = None
                    g_adv_loss = None
                    if use_discriminator:
                        disc_start = time.time()
                        # turn off grad update for disc
                        discriminator.requires_grad_(False)
                        disc_input = x_rec
                        # For pretrained 2D discriminator, flatten views+time to frame batches.
                        if disc_per_frame_2d:
                            if disc_input.dim() == 6:
                                b, v, c, t, h, w = disc_input.shape
                                disc_input = disc_input.reshape(b * v * t, c, h, w)
                            elif disc_input.dim() == 5:
                                b, c, t, h, w = disc_input.shape
                                disc_input = disc_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                        elif is_multiview and not disc_per_frame_2d:
                            b, v, c, t, h, w = disc_input.shape
                            if disc_multiview_mode == "joint_4d":
                                pass
                            elif disc_multiview_mode == "stack_channels":
                                disc_input = disc_input.reshape(b, v * c, t, h, w)
                            elif disc_multiview_mode == "flatten_batch":
                                if view_flatten_in_disc:
                                    disc_input = disc_input.view(b * v, c, t, h, w)
                                else:
                                    raise ValueError(
                                        "disc_multiview_mode='flatten_batch' requires view_flatten_in_disc=True, "
                                        "or use stack_channels / joint_4d with view_flatten_in_disc=False."
                                    )
                            else:
                                raise ValueError(
                                    f"Unknown disc_multiview_mode={disc_multiview_mode!r} "
                                    "(use flatten_batch, stack_channels, or joint_4d)."
                                )
                        fake_logits = discriminator(disc_input.contiguous())

                        _model_inner = getattr(model, "module", model)
                        generator_loss, g_loss = generator_loss_fn(
                            fake_logits,
                            nll_loss,
                            _model_inner.get_last_layer(),
                            actual_update_step,
                            is_training=model.training,
                        )
                        g_adv_loss = g_loss.detach()
                        adaptive_w = None
                        # Optional: two extra autograd.grad calls on the last layer (expensive); keep off for speed.
                        if cfg.get("gan_log_adaptive_grad_metrics", False):
                            last_layer = _model_inner.get_last_layer()
                            g_recon = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                            g_adv = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                            adaptive_w = torch.norm(g_recon) / (torch.norm(g_adv) + 1e-4)

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
                        and (global_step + 1) % accumulation_steps != 0
                        else nullcontext()
                    )
                    _backward_ctx = ProfileTimer.block("train.backward") if _do_profile_timing else nullcontext()
                    with _backward_ctx:
                        with Timer("backward", log=True) if cfg.get("profile", False) else nullcontext():
                            with ctx:
                                loss_scaled = vae_loss / accumulation_steps
                                if optimizer_has_booster_backward:
                                    booster.backward(loss=loss_scaled, optimizer=optimizer)
                                else:
                                    loss_scaled.backward()
                    backward_time = time.time() - backward_start
                    timing_stats["backward"].append(backward_time)

                    # Grad stats: collect right after backward (before step / zero_grad). Under ZeRO,
                    # per-parameter .grad on the module is often empty; collect_trainable_param_stats
                    # falls back to optimizer.get_grad_norm().
                    pending_train_stats = None
                    pre_clip_global_grad_norm = None
                    if (global_step + 1) % accumulation_steps == 0:
                        should_log_train_debug = (
                            coordinator.is_master()
                            and actual_update_step >= debug_stats_start_step
                            and actual_update_step % debug_stats_every == 0
                        )
                        if should_log_train_debug:
                            include_weight_stats = actual_update_step % debug_stats_weight_every == 0
                            pending_train_stats = collect_trainable_param_stats(
                                model,
                                include_weight_stats=include_weight_stats,
                                optimizer=optimizer,
                            )
                        if force_manual_global_grad_clip and configured_grad_clip > 0:
                            pre_clip_global_grad_norm = compute_optimizer_global_grad_norm(
                                optimizer,
                                dp_group=get_data_parallel_group(),
                            )
                            if pending_train_stats is not None and pre_clip_global_grad_norm is not None:
                                pending_train_stats["global_grad_norm"] = pre_clip_global_grad_norm
                                pending_train_stats["grad_l2_norm"] = pre_clip_global_grad_norm
                                pending_train_stats["grad_stats_source"] = (
                                    "optimizer_grad_tensors_global_dp_allreduce (pre-clip)"
                                )

                    # Optimizer step updates weights - usually fast but can be slow with large models
                    # or complex optimizers (e.g., Adam with many parameters)
                    optimizer_start = time.time()
                    post_clip_global_grad_norm = None
                    _optimizer_ctx = ProfileTimer.block("train.optimizer") if _do_profile_timing else nullcontext()
                    with _optimizer_ctx:
                        with Timer("optimizer", log=True) if cfg.get("profile", False) else nullcontext():
                            if (global_step + 1) % accumulation_steps == 0:
                                #print(f"[GradClip] force_manual={force_manual_global_grad_clip}, configured_grad_clip={configured_grad_clip}, plugin_grad_clip={plugin_grad_clip}")
                                if force_manual_global_grad_clip and configured_grad_clip > 0:
                                    # Manual clipping path (zero1 / non-ZeRO): param.grad is visible,
                                    # so torch.clip_grad_norm_ + clip_optimizer_grad_norm_global_ work.
                                    _ = torch.nn.utils.clip_grad_norm_(
                                        model.parameters(),
                                        max_norm=configured_grad_clip,
                                    )
                                    (
                                        _manual_pre_clip_norm,
                                        _manual_post_clip_norm,
                                        _,
                                    ) = clip_optimizer_grad_norm_global_(
                                        optimizer,
                                        max_norm=configured_grad_clip,
                                        dp_group=get_data_parallel_group(),
                                    )
                                    if _manual_pre_clip_norm is not None:
                                        pre_clip_global_grad_norm = _manual_pre_clip_norm
                                    if _manual_post_clip_norm is not None:
                                        post_clip_global_grad_norm = _manual_post_clip_norm
                                    #print(f"[GradClip manual] pre={pre_clip_global_grad_norm}, post={post_clip_global_grad_norm}")

                                optimizer.step()

                                if not force_manual_global_grad_clip and configured_grad_clip > 0:
                                    # Plugin-internal clipping path (ZeRO2+): ColossalAI's step() calls
                                    # _compute_grad_norm (all-reduce), stores it in _current_grad_norm,
                                    # then calls _unscale_and_clip_grads — all before returning.
                                    # So get_grad_norm() immediately after step() gives the CURRENT
                                    # step's pre-clip norm. Post-clip is analytically min(pre, max_norm).
                                    _plugin_pre = _safe_optimizer_get_grad_norm(optimizer)
                                    if _plugin_pre is not None:
                                        pre_clip_global_grad_norm = _plugin_pre
                                        post_clip_global_grad_norm = min(_plugin_pre, float(configured_grad_clip))
                                        if pending_train_stats is not None:
                                            pending_train_stats["grad_l2_norm"] = _plugin_pre
                                            pending_train_stats["global_grad_norm"] = _plugin_pre
                                            pending_train_stats["grad_stats_source"] = (
                                                "optimizer_get_grad_norm (plugin-internal clip; pre-clip from step)"
                                            )
                                    print(f"[GradClip plugin] pre={pre_clip_global_grad_norm}, post={post_clip_global_grad_norm}, max_norm={configured_grad_clip}")

                                if lr_scheduler is not None:
                                    lr_scheduler.step(
                                        actual_update_step,
                                    )
                                # == update EMA ==
                                # EMA update is usually fast but we include it in optimizer timing
                                if ema is not None:
                                    update_ema(
                                        ema,
                                        unwrap_model_safe(model),
                                        optimizer=optimizer,
                                        decay=cfg.get("ema_decay", 0.9999),
                                        sharded=plugin_type not in ("none",),
                                    )
                                optimizer.zero_grad()
                    optimizer_time = time.time() - optimizer_start
                    timing_stats["optimizer"].append(optimizer_time)

                    # -- user-friendly block timing report (once, after warmup) --
                    if _do_profile_timing and coordinator.is_master():
                        if x.is_cuda:
                            torch.cuda.synchronize()
                        ProfileTimer.disable()
                        summary = ProfileTimer.summarize()
                        report = ProfileTimer.format_report(summary, step=global_step)
                        print(report)
                        logger.info("\n%s", report)
                        json_path, txt_path = ProfileTimer.save_report(summary, exp_dir, step=global_step)
                        print(f"[ProfileTiming] Saved → {json_path}")
                        print(f"[ProfileTiming] Saved → {txt_path}")
                        _profile_timing_done = True

                    # -- full-step profiler report (step 5 only) --
                    if _do_full_profile and _prof is not None:
                        if x.is_cuda:
                            torch.cuda.synchronize()
                        _prof.__exit__(None, None, None)
                        sort_key = "cuda_time_total" if x.is_cuda else "cpu_time_total"
                        print(
                            f"\n{'='*80}\n"
                            f"[Profiler] Full training step — top 30 ops by {sort_key}\n"
                            f"{'='*80}\n"
                            + _prof.key_averages(group_by_input_shape=True).table(
                                sort_by=sort_key, row_limit=30
                            )
                        )
                        trace_path = "/home/piado/projects/aip-lindell/piado/vae/tmp/train_step_profile.json"
                        _prof.export_chrome_trace(trace_path)
                        print(f"[Profiler] Chrome trace saved → {trace_path}")
                        print(f"[Profiler] Open with chrome://tracing or https://ui.perfetto.dev")
                        _prof = None  # prevent accidental re-use

                    # -- logging --
                    log_loss("all", vae_loss, loss_dict, use_video)
                    log_loss("nll", nll_loss, loss_dict, use_video)
                    log_loss("nll_rec", recon_loss, loss_dict, use_video)
                    log_loss("nll_per", perceptual_loss, loss_dict, use_video)
                    log_loss("kl", kl_loss, loss_dict, use_video)
                    if distill_loss is not None:
                        log_loss("distill", distill_loss, loss_dict, use_video)
                    if temporal_diff_loss is not None:
                        log_loss("temporal_diff", temporal_diff_loss, loss_dict, use_video)
                    if use_discriminator:
                        log_loss("gen_w", generator_loss, loss_dict, use_video)
                        log_loss("gen", g_loss, loss_dict, use_video)
                    if (global_step + 1) % accumulation_steps == 0 and pending_train_stats is not None:
                        flat_update_sample_ids = [
                            sid for microbatch_ids in current_update_sample_ids for sid in microbatch_ids
                        ]
                        train_stats_record = {
                            "kind": "train_update_stats",
                            "actual_update_step": int(actual_update_step),
                            "global_step": int(global_step),
                            "epoch": int(epoch),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "dtype_config": str(cfg.get("dtype", "bf16")),
                            "torch_dtype_runtime": str(dtype),
                            "loss_total": _to_float_scalar(loss_dict.get("all")),
                            "loss_nll": _to_float_scalar(loss_dict.get("nll")),
                            "loss_nll_rec": _to_float_scalar(loss_dict.get("nll_rec")),
                            "loss_nll_per": _to_float_scalar(loss_dict.get("nll_per")),
                            "loss_kl": _to_float_scalar(loss_dict.get("kl")),
                            "loss_vc": _to_float_scalar(loss_dict.get("vc")),
                            "loss_total_precise": _to_precise_float_str(loss_dict.get("all")),
                            "loss_nll_precise": _to_precise_float_str(loss_dict.get("nll")),
                            "loss_nll_rec_precise": _to_precise_float_str(loss_dict.get("nll_rec")),
                            "loss_nll_per_precise": _to_precise_float_str(loss_dict.get("nll_per")),
                            "loss_kl_precise": _to_precise_float_str(loss_dict.get("kl")),
                            "loss_vc_precise": _to_precise_float_str(loss_dict.get("vc")),
                            # Batch PSNR from the same training step, if computed.
                            "train_batch_psnr": _to_float_scalar(loss_dict.get("psnr")),
                            "train_batch_psnr_precise": _to_precise_float_str(loss_dict.get("psnr")),
                            # Explicitly measured global grad norm before clipping for this update.
                            "pre_clip_global_grad_norm": pre_clip_global_grad_norm,
                            # Best-effort post-clip norm captured after optimizer.step() and before zero_grad().
                            "post_clip_global_grad_norm": post_clip_global_grad_norm,
                            # Sample IDs/paths used by this optimizer update (across accumulation microbatches).
                            "update_microbatch_sample_ids": current_update_sample_ids,
                            "update_flat_sample_ids": flat_update_sample_ids,
                            "update_flat_sample_count": int(len(flat_update_sample_ids)),
                        }
                        train_stats_record.update(pending_train_stats)
                        append_training_debug_jsonl(exp_dir, train_stats_record)
                    
                    # -- JSONL train_batch snapshot (optional; throttled by eval_every) --
                    eval_every = cfg.get("eval_every", 0)
                    batch_eval_this_step = (
                        eval_every > 0
                        and (global_step + 1) % accumulation_steps == 0
                        and should_log_update(actual_update_step, cfg)
                        and coordinator.is_master()
                        and use_video == 1
                        and "psnr" in loss_dict
                    )

                    # -- plot train reconstruction on the (tapering) image log schedule: fixed train
                    # samples (1 or 3 people). Uses should_log_images (not should_log_update) so image
                    # logging frequency backs off over a long run instead of staying at a fixed cadence
                    # forever -- that's what fills up wandb/local storage on multi-day sweeps.
                    plot_reconstruction = (
                        (global_step + 1) % accumulation_steps == 0
                        and should_log_images(actual_update_step, cfg)
                        and coordinator.is_master()
                        and use_video == 1
                    )
                    if plot_reconstruction:
                        vis_range = "[-1,1]" if (vae_target_range == "[-1,1]" or (vae_target_range is None and is_multiview)) else "[0,1]"
                        dataset = dataloader.dataset
                        participants_cfg = getattr(dataset, "participants", None)
                        # 1 sample if single person; else up to num_reconstruction_vis_samples distinct people
                        _nrv = int(cfg.get("num_reconstruction_vis_samples", 3))
                        num_vis = _nrv if (participants_cfg is not None and len(participants_cfg) > 1) else 1
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
                        x_vis = apply_train_bucket_spatiotemporal(x_vis, cfg)
                        if vae_target_range == "[-1,1]" or (vae_target_range is None and is_multiview):
                            x_vis = x_vis * 2.0 - 1.0
                        with torch.no_grad():
                            x_rec_vis, _, _ = model(x_vis)
                        vis_images = create_visualization_grid(x_vis, x_rec_vis, num_samples=n_vis, value_range=vis_range)
                        loss_dict["reconstruction_samples"] = vis_images
                        # Also plot val/test reconstructions every log_every when val_dataset exists
                        if val_dataset is not None and len(val_dataset) > 0:
                            n_val = min(int(cfg.get("num_reconstruction_vis_samples", 3)), len(val_dataset))
                            val_items = [val_dataset[i] for i in range(n_val)]
                            val_batch = default_collate(val_items)
                            x_val = val_batch["video"].to(device, dtype)
                            x_val = apply_train_bucket_spatiotemporal(x_val, cfg)
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
                        if disc_per_frame_2d:
                            if real_input.dim() == 6:
                                b, v, c, t, h, w = real_input.shape
                                real_input = real_input.reshape(b * v * t, c, h, w)
                                fake_input = fake_input.reshape(b * v * t, c, h, w)
                            elif real_input.dim() == 5:
                                b, c, t, h, w = real_input.shape
                                real_input = real_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                                fake_input = fake_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                        elif is_multiview and not disc_per_frame_2d:
                            b, v, c, t, h, w = real_input.shape
                            if disc_multiview_mode == "joint_4d":
                                pass
                            elif disc_multiview_mode == "stack_channels":
                                real_input = real_input.reshape(b, v * c, t, h, w)
                                fake_input = fake_input.reshape(b, v * c, t, h, w)
                            elif disc_multiview_mode == "flatten_batch":
                                if view_flatten_in_disc:
                                    real_input = real_input.view(b * v, c, t, h, w)
                                    fake_input = fake_input.view(b * v, c, t, h, w)
                                else:
                                    raise ValueError(
                                        "disc_multiview_mode='flatten_batch' requires view_flatten_in_disc=True, "
                                        "or use stack_channels / joint_4d with view_flatten_in_disc=False."
                                    )
                            else:
                                raise ValueError(
                                    f"Unknown disc_multiview_mode={disc_multiview_mode!r} "
                                    "(use flatten_batch, stack_channels, or joint_4d)."
                                )
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
                            and (global_step + 1) % accumulation_steps != 0
                            else nullcontext()
                        )
                        with ctx:
                            disc_loss_scaled = disc_loss / accumulation_steps
                            if disc_optimizer_has_booster_backward:
                                booster.backward(loss=disc_loss_scaled, optimizer=disc_optimizer)
                            else:
                                disc_loss_scaled.backward()
                        if (global_step + 1) % accumulation_steps == 0:
                            disc_optimizer.step()
                            disc_optimizer.zero_grad()
                            if disc_lr_scheduler is not None:
                                disc_lr_scheduler.step(actual_update_step)

                        # log
                        log_loss("disc", disc_loss, loss_dict, use_video)
                        # if (
                        #     coordinator.is_master()
                        #     and (global_step + 1) % accumulation_steps == 0
                        #     and actual_update_step % 100 == 0
                        #     and g_adv_loss is not None
                        #     and adaptive_w is not None
                        # ):
                        #     print("g_adv_loss:", g_adv_loss.item())
                        #     print("d_loss:", disc_loss.item())
                        #     print("adaptive_weight:", adaptive_w.item())

                    # Persist train-batch metrics + loss snapshot (after disc, so loss_dict is complete)
                    if batch_eval_this_step:
                        _keys = (
                            "all",
                            "nll",
                            "nll_rec",
                            "nll_per",
                            "kl",
                            "view_loss",
                            "gen",
                            "gen_w",
                            "disc",
                            "distill",
                        )
                        def _loss_item(v):
                            if v is None:
                                return None
                            return float(v.item()) if hasattr(v, "item") else float(v)

                        losses_snapshot = {
                            k: _loss_item(loss_dict[k]) for k in _keys if k in loss_dict and loss_dict[k] is not None
                        }
                        append_eval_metrics_jsonl(
                            exp_dir,
                            {
                                "kind": "train_batch",
                                "actual_update_step": int(actual_update_step),
                                "global_step": int(global_step),
                                "epoch": int(epoch),
                                "metrics": {
                                    "psnr": float(loss_dict["psnr"]),
                                    "ssim": float(loss_dict["ssim"]),
                                    "mse": float(loss_dict["mse"]),
                                },
                                "losses": losses_snapshot,
                                "loss_config": _loss_config_dict(cfg),
                            },
                        )

                    # Wall-clock for full training iteration (generator ± discriminator paths)
                    step_time = time.time() - step_start_time
                    timing_stats["total_step"].append(step_time)

                    if dist.is_initialized() and dist.get_world_size() > 1:
                        _sync_stop = torch.zeros(1, dtype=torch.int32, device=device)
                        if coordinator.is_master():
                            _sync_stop[0] = 1 if early_stop_requested else 0
                        dist.broadcast(_sync_stop, src=0)
                        early_stop_requested = bool(_sync_stop.item())

                    if (
                        _log_step_time_once
                        and not _step_time_bench_done
                        and coordinator.is_master()
                    ):
                        _step_time_bench_samples.append(step_time)
                        if len(_step_time_bench_samples) >= 10:
                            avg_step = sum(_step_time_bench_samples) / 10.0
                            _bench_msg = (
                                f"[step_time] Average wall time over first 10 training steps: {avg_step:.4f} s"
                            )
                            logger.info(_bench_msg)
                            # tqdm.write avoids clobbering the progress bar (plain print often hides this line)
                            tqdm.write(_bench_msg)
                            _step_time_bench_done = True

                    # == logging ==
                    # We log periodically to avoid overwhelming the logs, but include timing stats
                    # to help identify bottlenecks. Logging itself is fast, so we don't time it.
                    if (global_step + 1) % accumulation_steps == 0:
                        if coordinator.is_master() and should_log_update(actual_update_step, cfg):
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
                            
                            # wandb (lazy init: see maybe_init_wandb)
                            if cfg.get("wandb", False) and wandb.run is not None:
                                wandb_log_dict = {
                                        "iter": global_step,
                                        "global_step": actual_update_step,
                                        "samples_seen": actual_update_step * effective_batch_size,
                                        "epoch": epoch,
                                        "epoch_float": epoch_float,
                                        "lr": optimizer.param_groups[0]["lr"],
                                        # Average losses over log_every steps
                                        "loss/total": avg_loss["all"],
                                        "loss/nll": avg_loss.get("nll", 0.0),
                                        "loss/nll_rec": avg_loss.get("nll_rec", 0.0),
                                        "loss/nll_per": avg_loss.get("nll_per", 0.0),
                                        "loss/kl": avg_loss.get("kl", 0.0),
                                        "global_grad_norm": _safe_optimizer_get_grad_norm(optimizer),
                                    }
                                
                                # Discriminator / GAN losses (only when enabled)
                                if use_discriminator:
                                    wandb_log_dict["loss/disc"] = avg_loss.get("disc", 0.0)
                                    wandb_log_dict["loss/gen"] = avg_loss.get("gen", 0.0)
                                    wandb_log_dict["loss/gen_w"] = avg_loss.get("gen_w", 0.0)

                                if teacher_model is not None:
                                    wandb_log_dict["loss/distill"] = avg_loss.get("distill", 0.0)

                                if temporal_diff_loss_weight > 0.0:
                                    wandb_log_dict["loss/temporal_diff"] = avg_loss.get("temporal_diff", 0.0)

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
                                
                                # Train-batch recon metrics (current minibatch only; full eval uses eval/* keys)
                                if "psnr" in loss_dict:
                                    wandb_log_dict["train_batch/psnr"] = loss_dict["psnr"]
                                    wandb_log_dict["train_batch/ssim"] = loss_dict["ssim"]
                                    wandb_log_dict["train_batch/mse"] = loss_dict["mse"]

                                # Temporal-compression "bleeding" diagnostic (see
                                # compute_intra_chunk_bleed_metrics): ~1.0 = healthy, ~0.0 = severe
                                # bleeding. Only present when temporal_compression=True (multiview).
                                if "bleed_ratio_within" in loss_dict:
                                    wandb_log_dict["train_batch/bleed_ratio_within"] = loss_dict["bleed_ratio_within"]
                                    wandb_log_dict["train_batch/bleed_ratio_across"] = loss_dict.get("bleed_ratio_across", 0.0)
                                    wandb_log_dict["train_batch/gt_diff_within"] = loss_dict.get("gt_diff_within", 0.0)
                                    wandb_log_dict["train_batch/rec_diff_within"] = loss_dict.get("rec_diff_within", 0.0)

                                # Add train/test reconstruction visualizations as separate media keys.
                                if "reconstruction_samples" in loss_dict:
                                    wandb_log_dict["train_reconstructions"] = loss_dict["reconstruction_samples"]
                                if "val_reconstruction_samples" in loss_dict:
                                    wandb_log_dict["test_reconstructions"] = loss_dict["val_reconstruction_samples"]
                                
                                wandb.log(wandb_log_dict, step=actual_update_step)

                            running_loss = {k: 0.0 for k in running_loss}
                            log_step = 0
                            
                            # Clear old timing stats to avoid memory buildup (keep last 100 steps)
                            for key in timing_stats:
                                if key not in ["memory_allocated", "memory_reserved"]:
                                    if len(timing_stats[key]) > 100:
                                        timing_stats[key] = timing_stats[key][-100:]

                        # -- periodic full evaluation (runs even when save_ckpt is False) --
                        full_eval_every = cfg.get("full_eval_every", 0)
                        if (
                            full_eval_every > 0
                            and actual_update_step % full_eval_every == 0
                            and coordinator.is_master()
                        ):
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
                                **_persistent_dl_extras,
                            )
                            eval_val_range = cfg.get("vae_target_range") or (
                                "[-1,1]" if cfg.model.get("type") == "multiview_wan_video_vae" else "[0,1]"
                            )
                            eval_results = evaluate_model(
                                eval_model,
                                eval_dataloader,
                                device,
                                dtype,
                                num_eval_samples=cfg.get("eval_num_samples", 32),
                                view_flatten_in_loss=view_flatten_in_loss,
                                use_ema=(ema is not None and cfg.get("eval_use_ema", True)),
                                value_range=eval_val_range,
                                train_target_hw=cfg.get("train_target_hw"),
                                train_target_frames=cfg.get("train_target_frames"),
                                vis_max_samples=cfg.get("num_reconstruction_vis_samples", 3),
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
                            append_eval_metrics_jsonl(
                                exp_dir,
                                {
                                    "kind": "full_eval",
                                    "split": eval_ds_label,
                                    "actual_update_step": int(actual_update_step),
                                    "global_step": int(global_step),
                                    "epoch": int(epoch),
                                    # Dump everything numeric (incl. bleed ratios, cross-view
                                    # similarity, per-frame PSNR profile) so collect_results.py
                                    # never needs a code change when a metric is added.
                                    "metrics": {
                                        k: (float(v) if not isinstance(v, list) else v)
                                        for k, v in eval_metrics.items()
                                    },
                                    "loss_config": _loss_config_dict(cfg),
                                },
                            )
                            if cfg.get("wandb", False) and wandb.run is not None:
                                prefix = f"eval/{eval_ds_label}"
                                log_dict = {
                                    "global_step": actual_update_step,
                                    "samples_seen": actual_update_step * effective_batch_size,
                                    "epoch_float": epoch_float,
                                    f"{prefix}/psnr_mean": eval_metrics["psnr_mean"],
                                    f"{prefix}/psnr_std": eval_metrics["psnr_std"],
                                    f"{prefix}/ssim_mean": eval_metrics["ssim_mean"],
                                    f"{prefix}/ssim_std": eval_metrics["ssim_std"],
                                    f"{prefix}/mse_mean": eval_metrics["mse_mean"],
                                    f"{prefix}/mse_std": eval_metrics["mse_std"],
                                    f"{prefix}/reconstructions": eval_results["visualizations"],
                                }
                                # Paper diagnostics (present when applicable: bleed ratios
                                # need T>=2, cross-view similarity needs V>=2).
                                for k in (
                                    "bleed_ratio_within",
                                    "bleed_ratio_across",
                                    "xview_sim_rec",
                                    "xview_sim_gt",
                                ):
                                    if k in eval_metrics:
                                        log_dict[f"{prefix}/{k}"] = eval_metrics[k]
                                if "psnr_per_frame" in eval_metrics:
                                    for fi, pv in enumerate(eval_metrics["psnr_per_frame"]):
                                        log_dict[f"{prefix}/psnr_frame{fi}"] = pv
                                wandb.log(log_dict, step=actual_update_step)

            if cfg.get("profile", False):
                profiler_ctxt.export_chrome_trace("./log/profile/trace.json")
        epoch_psnr_mean = float("nan")
        if epoch_psnr_count > 0:
            epoch_psnr_mean = epoch_psnr_sum / max(1, epoch_psnr_count)
            last_epoch_psnr_mean = epoch_psnr_mean
            train_psnr_bad_for_ckpt = epoch_psnr_mean < train_psnr_guard_threshold
        if cfg.get("train_psnr_guard", True) and epoch_psnr_count > 0 and not early_stop_requested:
            monitor_epoch = (epoch + 1) >= (train_psnr_guard_start_epoch + 1)
            warmup_done = (epoch + 1) >= (train_psnr_guard_start_epoch + train_psnr_guard_min_epochs + 1)
            if not monitor_epoch or not warmup_done:
                train_psnr_low_streak = 0
            else:
                if epoch_psnr_mean < train_psnr_guard_threshold:
                    train_psnr_low_streak += 1
                else:
                    train_psnr_low_streak = 0
                if train_psnr_low_streak >= train_psnr_guard_consecutive:
                    early_stop_requested = True
                    if coordinator.is_master() and not _psnr_stop_log_once:
                        _psnr_stop_log_once = True
                        logger.error(
                            "Train PSNR guard: %s consecutive epochs with mean train_batch PSNR < %.3f "
                            "after start epoch %s + warmup %s epochs (epoch_mean=%.3f). "
                            "Stopping training.",
                            train_psnr_guard_consecutive,
                            train_psnr_guard_threshold,
                            train_psnr_guard_start_epoch,
                            train_psnr_guard_min_epochs,
                            epoch_psnr_mean,
                        )
        # Overfit-gate target stop (see init above): success counterpart of the guard.
        if stop_at_train_psnr > 0 and epoch_psnr_count > 0 and not early_stop_requested:
            if epoch_psnr_mean >= stop_at_train_psnr:
                train_psnr_target_streak += 1
            else:
                train_psnr_target_streak = 0
            if train_psnr_target_streak >= stop_at_train_psnr_consecutive:
                early_stop_requested = True
                if coordinator.is_master():
                    logger.info(
                        "Overfit gate PASSED: %s consecutive epochs with mean train PSNR >= %.2f "
                        "(epoch %s, epoch_mean=%.3f). Stopping early.",
                        train_psnr_target_streak,
                        stop_at_train_psnr,
                        epoch,
                        epoch_psnr_mean,
                    )
        # == checkpoint saving at epoch end (optional; default off via save_ckpt) ==
        save_ckpt = cfg.get("save_ckpt", False)
        save_every_n_epochs = max(1, int(cfg.get("save_every_n_epochs", 1)))
        is_ckpt_epoch = (epoch % save_every_n_epochs == 0)
        if save_ckpt and is_ckpt_epoch and coordinator.is_master():
            if epoch_psnr_count <= 0:
                logger.warning(
                    "No train PSNR samples at epoch %s (NaN output or eval not reached yet); "
                    "saving checkpoint anyway.",
                    epoch,
                )
            elif train_psnr_bad_for_ckpt:
                logger.warning(
                    "Low epoch-mean train PSNR at epoch %s (%.3f < %.3f); "
                    "saving checkpoint anyway (use early-stop guard to halt training).",
                    epoch,
                    epoch_psnr_mean,
                    train_psnr_guard_threshold,
                )
            # Always save — PSNR quality is logged above but never blocks the save.
            # The early-stop guard (train_psnr_guard) is the right place to halt bad training.
            if save_ckpt:
                gc.collect()
                use_async_io = False
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
                last_saved_ckpt_epoch = epoch

                logger.info(
                    "Saved epoch-end checkpoint at epoch %s, step %s, global_step %s (epoch_mean_psnr=%.3f) to %s",
                    epoch,
                    step + 1,
                    actual_update_step,
                    epoch_psnr_mean,
                    save_dir,
                )

                keep_n_latest = int(cfg.get("keep_n_latest", 5))
                rm_checkpoints(exp_dir, keep_n_latest=keep_n_latest)
                logger.info("Removed old checkpoints and kept %s latest one(s).", keep_n_latest)
        if early_stop_requested:
            logger.info(
                "Exited training loop early (PSNR guard tripped or overfit-gate target reached)."
            )
            break

        # Reset sampler for next epoch (if it has the reset method)
        if sampler is not None and hasattr(sampler, 'reset'):
            sampler.reset()
        start_step = 0
        
        # =======================================================
        # Per-frame metrics on fixed train/val sequences every N epochs
        # =======================================================
        fixed_epoch_interval = int(cfg.get("fixed_seq_eval_every_epochs", 10))
        maybe_init_wandb(actual_update_step)
        if (
            fixed_epoch_interval > 0
            and ((epoch + 1) % fixed_epoch_interval == 0)
            and coordinator.is_master()
            and fixed_train_index is not None
        ):
            logger.info(
                "Running fixed per-frame evaluation at epoch %s (sequence name: %s)...",
                epoch + 1,
                fixed_seq_name,
            )
            vae_target_range = cfg.get("vae_target_range", None)

            train_metrics_pf = evaluate_fixed_sequence_per_frame(
                model,
                dataset,
                fixed_train_index,
                device,
                dtype,
                vae_target_range=vae_target_range,
                train_target_hw=cfg.get("train_target_hw"),
                train_target_frames=cfg.get("train_target_frames"),
            )

            val_metrics_pf = None
            if fixed_val_index is not None and val_dataset is not None:
                val_metrics_pf = evaluate_fixed_sequence_per_frame(
                    model,
                    val_dataset,
                    fixed_val_index,
                    device,
                    dtype,
                    vae_target_range=vae_target_range,
                    train_target_hw=cfg.get("train_target_hw"),
                    train_target_frames=cfg.get("train_target_frames"),
                )

            fixed_record = {
                "kind": "fixed_seq",
                "epoch": int(epoch + 1),
                "global_step": int(actual_update_step),
                "sequence_name": fixed_seq_name,
                "train": {
                    "psnr_per_frame": train_metrics_pf["psnr_per_frame"],
                    "ssim_per_frame": train_metrics_pf["ssim_per_frame"],
                    "mse_per_frame": train_metrics_pf["mse_per_frame"],
                    "psnr_mean": float(np.mean(train_metrics_pf["psnr_per_frame"])),
                    "ssim_mean": float(np.mean(train_metrics_pf["ssim_per_frame"])),
                    "mse_mean": float(np.mean(train_metrics_pf["mse_per_frame"])),
                },
                "loss_config": _loss_config_dict(cfg),
            }
            if val_metrics_pf is not None:
                fixed_record["val"] = {
                    "psnr_per_frame": val_metrics_pf["psnr_per_frame"],
                    "ssim_per_frame": val_metrics_pf["ssim_per_frame"],
                    "mse_per_frame": val_metrics_pf["mse_per_frame"],
                    "psnr_mean": float(np.mean(val_metrics_pf["psnr_per_frame"])),
                    "ssim_mean": float(np.mean(val_metrics_pf["ssim_per_frame"])),
                    "mse_mean": float(np.mean(val_metrics_pf["mse_per_frame"])),
                }
            append_eval_metrics_jsonl(exp_dir, fixed_record)

            if cfg.get("wandb", False) and wandb.run is not None:
                table = wandb.Table(columns=["frame", "psnr", "ssim", "mse"])
                for i in range(len(train_metrics_pf["psnr_per_frame"])):
                    table.add_data(
                        i,
                        train_metrics_pf["psnr_per_frame"][i],
                        train_metrics_pf["ssim_per_frame"][i],
                        train_metrics_pf["mse_per_frame"][i],
                    )
                wandb_log_pf = {
                    "fixed_seq/name": fixed_seq_name,
                    "fixed_seq/epoch": epoch + 1,
                    "epoch_float": float(epoch + 1),
                    "global_step": actual_update_step,
                    "samples_seen": actual_update_step * effective_batch_size,
                    "fixed_seq/train/psnr_mean": float(np.mean(train_metrics_pf["psnr_per_frame"])),
                    "fixed_seq/train/ssim_mean": float(np.mean(train_metrics_pf["ssim_per_frame"])),
                    "fixed_seq/train/mse_mean": float(np.mean(train_metrics_pf["mse_per_frame"])),
                    "fixed_seq/train_metrics_table": table,
                }
                if val_metrics_pf is not None:
                    wandb_log_pf["fixed_seq/val/psnr_mean"] = float(
                        np.mean(val_metrics_pf["psnr_per_frame"])
                    )
                    wandb_log_pf["fixed_seq/val/ssim_mean"] = float(
                        np.mean(val_metrics_pf["ssim_per_frame"])
                    )
                    wandb_log_pf["fixed_seq/val/mse_mean"] = float(
                        np.mean(val_metrics_pf["mse_per_frame"])
                    )
                wandb.log(wandb_log_pf, step=actual_update_step)
    
    # =======================================================
    # 6. Final evaluation after training
    # =======================================================
    # After all training epochs are complete, run a final comprehensive
    # evaluation to assess the final model quality. This gives us the
    # definitive metrics for the trained model.
    if coordinator.is_master():
        maybe_init_wandb(actual_update_step)
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
                    **_persistent_dl_extras,
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
                    train_target_hw=cfg.get("train_target_hw"),
                    train_target_frames=cfg.get("train_target_frames"),
                    vis_max_samples=cfg.get("num_reconstruction_vis_samples", 3),
                )
                final_metrics = final_eval_results["metrics"]
                logger.info("=" * 80)
                logger.info("FINAL EVALUATION (%s)", label.upper())
                logger.info("=" * 80)
                logger.info("PSNR: %.2f ± %.2f dB", final_metrics["psnr_mean"], final_metrics["psnr_std"])
                logger.info("SSIM: %.4f ± %.4f", final_metrics["ssim_mean"], final_metrics["ssim_std"])
                logger.info("MSE:  %.6f ± %.6f", final_metrics["mse_mean"], final_metrics["mse_std"])
                logger.info("=" * 80)
                append_eval_metrics_jsonl(
                    exp_dir,
                    {
                        "kind": "final_eval",
                        "split": label,
                        "actual_update_step": int(actual_update_step),
                        # Generic dump, same as full_eval: includes the paper diagnostics.
                        "metrics": {
                            k: (float(v) if not isinstance(v, list) else v)
                            for k, v in final_metrics.items()
                        },
                        "loss_config": _loss_config_dict(cfg),
                    },
                )
                if cfg.get("wandb", False) and wandb.run is not None:
                    prefix = f"final_eval/{label}"
                    final_log = {
                        "global_step": actual_update_step,
                        "samples_seen": actual_update_step * effective_batch_size,
                        "epoch_float": float(epoch + 1),
                        f"{prefix}/psnr_mean": final_metrics["psnr_mean"],
                        f"{prefix}/psnr_std": final_metrics["psnr_std"],
                        f"{prefix}/ssim_mean": final_metrics["ssim_mean"],
                        f"{prefix}/ssim_std": final_metrics["ssim_std"],
                        f"{prefix}/mse_mean": final_metrics["mse_mean"],
                        f"{prefix}/mse_std": final_metrics["mse_std"],
                        f"{prefix}/reconstructions": final_eval_results["visualizations"],
                    }
                    wandb.log(final_log, step=actual_update_step)
            logger.info("Final evaluation complete.")
    
    # =======================================================
    # 7. Save final checkpoint after training (optional)
    # =======================================================
    if coordinator.is_master():
        save_ckpt = cfg.get("save_ckpt", False)
        if save_ckpt and last_saved_ckpt_epoch != epoch and not train_psnr_bad_for_ckpt:
            logger.info("Saving final checkpoint...")
            use_async_io = False

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

            logger.info("Final checkpoint saved to %s", final_save_dir)
            keep_n_latest = int(cfg.get("keep_n_latest", 5))
            rm_checkpoints(exp_dir, keep_n_latest=keep_n_latest)
            logger.info("Removed old checkpoints and kept %s latest one(s).", keep_n_latest)
        else:
            if not save_ckpt:
                logger.info("Skipping final checkpoint save (save_ckpt=False).")
            elif last_saved_ckpt_epoch == epoch:
                logger.info("Skipping final checkpoint save (already saved at epoch end).")
            elif train_psnr_bad_for_ckpt:
                logger.info(
                    "Skipping final checkpoint save due to low epoch-mean train PSNR (last=%.3f, threshold=%.3f).",
                    last_epoch_psnr_mean,
                    train_psnr_guard_threshold,
                )
    
    dist.barrier()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()


                            