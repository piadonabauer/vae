"""Unified, range-explicit reconstruction metrics for the multiview-VAE paper.

Why this module exists
----------------------
The eval bug fixed in commit 40295bb came from an implicit range contract:
``compute_metrics`` in ``Open-Sora/scripts/vae/train.py`` clamps its inputs to
[0,1], but with ``vae_target_range="[-1,1]"`` the eval loop handed it tensors in
[-1,1]; the clamp crushed the negative half of the signal and under-reported
PSNR by 1.7-3.4 dB (and biased SSIM). Every function in this module therefore
takes an explicit ``value_range`` argument and converts ONCE, at the entry
point, before any metric math.

Declared conventions (use these everywhere in the paper)
--------------------------------------------------------
* Working range: all metrics are computed on float32 tensors in **[0,1]**.
  ``value_range`` declares the range of the tensors you pass in
  ("[0,1]" or "[-1,1]"); the module converts internally.
* Clamping: the reconstruction may legitimately overshoot its range; both GT
  and rec are clamped to the working range AFTER conversion (standard for
  8-bit-referenced PSNR). GT from the dataloader is already in range, so the
  clamp is a no-op for it.
* **PSNR aggregation = mean of per-(clip, view) PSNRs.** A "sample" is one
  (clip, view) video of shape [C, T, H, W]; PSNR is computed from the MSE
  pooled over C,T,H,W of that sample, then averaged ACROSS samples.
  This matches the corrected ``psnr_mean`` and the mean of ``psnr_per_view``
  in train.py. We do NOT use PSNR of the pooled MSE (``psnr_pooled`` is
  returned for reference only; it sits ~0.3 dB below the mean-of-PSNRs here).
  Headline, per-view, and per-frame numbers must all state this convention.
* Per-view PSNR: mean of per-clip PSNRs restricted to one view.
* Per-frame PSNR: PSNR of the MSE pooled over all samples for one frame index
  (this matches ``compute_metrics_per_frame`` in train.py: a batch-pooled
  profile, not a mean of per-sample per-frame PSNRs).
* SSIM: windowed SSIM, 11x11 Gaussian (sigma 1.5), data_range = 1.0, computed
  per (clip, view, frame) and averaged. NOTE: train.py's ``compute_ssim`` is a
  *global-moment* approximation (one mean/variance per frame, no local
  window); it reads systematically higher. ``ssim_global`` reproduces that
  legacy definition for comparability with old logs.
* LPIPS: the LPIPS net consumes [-1,1]; this module converts from the [0,1]
  working tensors internally. Pass an ``lpips_fn`` (pip ``lpips`` or
  opensora's LPIPS - both expect [-1,1]).
* Bleed ratios: chunk layout for Wan causal decode with chunk_size=4 and T=9
  is frame 0 alone, then frames 1-4 (chunk 1) and 5-8 (chunk 2). Within-chunk
  consecutive pairs are (1,2),(2,3),(3,4) and (5,6),(6,7),(7,8); pairs (0,1)
  and (4,5) cross chunk boundaries. Ratios are scale-invariant but are
  computed on the converted [0,1] tensors (NOT on clamped [-1,1] tensors).
* Cross-view cosine (XView): cosine similarity of flattened per-view videos,
  averaged over view pairs and clips. Scale-sensitive to the clamp, so it is
  also computed on the converted [0,1] tensors.

Shapes
------
``gt`` and ``rec``: [B, V, C, T, H, W] (multi-view) or [B, C, T, H, W]
(single view; treated as V=1).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "to_unit_range",
    "compute_all_metrics",
    "psnr_per_sample",
    "ssim_windowed",
    "ssim_global",
    "bleed_pair_indices",
    "bleed_ratios",
    "cross_view_cosine",
]

_VALID_RANGES = ("[0,1]", "[-1,1]")


def to_unit_range(x: torch.Tensor, value_range: str) -> torch.Tensor:
    """Convert a tensor whose values are declared to live in ``value_range``
    into float32 [0,1], clamping AFTER the conversion."""
    if value_range not in _VALID_RANGES:
        raise ValueError(f"value_range must be one of {_VALID_RANGES}, got {value_range!r}")
    x = x.float()
    if value_range == "[-1,1]":
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _as_bvcthw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:  # [B, C, T, H, W] -> V=1
        return x.unsqueeze(1)
    if x.dim() == 6:
        return x
    raise ValueError(f"expected 5D or 6D video tensor, got shape {tuple(x.shape)}")


def psnr_per_sample(gt01: torch.Tensor, rec01: torch.Tensor) -> torch.Tensor:
    """Per-(clip, view) PSNR in dB. Inputs [B,V,C,T,H,W] in [0,1].
    Returns [B*V]. Identical inputs give +inf."""
    b, v = gt01.shape[0], gt01.shape[1]
    mse = ((gt01 - rec01) ** 2).reshape(b * v, -1).mean(dim=1).double()
    return torch.where(
        mse == 0,
        torch.full_like(mse, math.inf),
        10.0 * torch.log10(1.0 / mse),
    )


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)  # [K, K]


def ssim_windowed(
    gt01: torch.Tensor,
    rec01: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Mean windowed SSIM over all (clip, view, frame) images.
    Standard Wang et al. SSIM: 11x11 Gaussian window, C1=(0.01*L)^2,
    C2=(0.03*L)^2 with L=data_range. Inputs [B,V,C,T,H,W] in [0,1]."""
    b, v, c, t, h, w = gt01.shape
    x = gt01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w)
    y = rec01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w)
    win = _gaussian_window(window_size, sigma).to(x.device, x.dtype)
    win = win.expand(c, 1, window_size, window_size).contiguous()

    def filt(z):
        # valid convolution (no padding): canonical Wang et al. SSIM; zero
        # padding would corrupt the local statistics at image borders
        return F.conv2d(z, win, groups=c)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = filt(x * x) - mu_x2
    sig_y2 = filt(y * y) - mu_y2
    sig_xy = filt(x * y) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    )
    return float(ssim_map.mean().item())


def ssim_global(gt01: torch.Tensor, rec01: torch.Tensor, data_range: float = 1.0) -> float:
    """Legacy global-moment SSIM exactly as train.py's ``compute_ssim``:
    one mean/variance per (image, channel), no local window. Reads higher than
    windowed SSIM; kept only to compare against historical logs."""
    b, v, c, t, h, w = gt01.shape
    x = gt01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w)
    y = rec01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu1 = x.mean(dim=[2, 3], keepdim=True)
    mu2 = y.mean(dim=[2, 3], keepdim=True)
    s1 = ((x - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    s2 = ((y - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    s12 = ((x - mu1) * (y - mu2)).mean(dim=[2, 3], keepdim=True)
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean().item())


def bleed_pair_indices(t: int, chunk_size: int = 4) -> tuple[list[int], list[int]]:
    """Consecutive-frame pair indices split into within-chunk / across-chunk.

    Pair ``i`` is (frame i, frame i+1). Wan causal layout: frame 0 is decoded
    alone from latent frame 0; frames 1..chunk_size share latent frame 1, etc.
    For T=9, chunk_size=4: within = [1,2,3,5,6,7] (pairs 1-2,2-3,3-4 and
    5-6,6-7,7-8); across = [0,4] (pairs 0-1 and 4-5). Frame 0 participates in
    no within-chunk pair.
    """

    def chunk_id(frame: int) -> int:
        return 0 if frame == 0 else 1 + (frame - 1) // chunk_size

    within = [i for i in range(t - 1) if chunk_id(i) == chunk_id(i + 1)]
    across = [i for i in range(t - 1) if chunk_id(i) != chunk_id(i + 1)]
    return within, across


def bleed_ratios(gt01: torch.Tensor, rec01: torch.Tensor, chunk_size: int = 4) -> dict:
    """bleed_ratio_within / bleed_ratio_across on [0,1] tensors.
    rec frame-to-frame L1 delta over gt delta, averaged over the pair set.
    ~1 healthy, <<1 = frozen frames (bleeding), >1 = rec changes more than GT."""
    b, v, c, t, h, w = gt01.shape
    x = gt01.reshape(b * v, c, t, h, w)
    y = rec01.reshape(b * v, c, t, h, w)
    gt_d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))  # [T-1]
    rec_d = (y[:, :, 1:] - y[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
    within, across = bleed_pair_indices(t, chunk_size)
    out = {}
    eps = 1e-6
    if within:
        idx = torch.tensor(within, device=x.device)
        out["bleed_ratio_within"] = float((rec_d[idx].mean() / (gt_d[idx].mean() + eps)).item())
    if across:
        idx = torch.tensor(across, device=x.device)
        out["bleed_ratio_across"] = float((rec_d[idx].mean() / (gt_d[idx].mean() + eps)).item())
    return out


def cross_view_cosine(x01: torch.Tensor) -> float | None:
    """Mean pairwise cosine similarity between flattened views; None if V<2."""
    if x01.shape[1] < 2:
        return None
    b, v = x01.shape[0], x01.shape[1]
    flat = F.normalize(x01.reshape(b, v, -1), dim=-1)
    sims = []
    for i in range(v):
        for j in range(i + 1, v):
            sims.append((flat[:, i] * flat[:, j]).sum(-1).mean())
    return float(torch.stack(sims).mean().item())


@torch.no_grad()
def compute_all_metrics(
    gt: torch.Tensor,
    rec: torch.Tensor,
    *,
    value_range: str,
    chunk_size: int = 4,
    lpips_fn=None,
    lpips_batch: int = 32,
) -> dict:
    """All paper metrics from one declared-range entry point.

    Args:
        gt, rec: [B,V,C,T,H,W] or [B,C,T,H,W] videos, SAME declared range.
        value_range: "[0,1]" or "[-1,1]" - the range of gt/rec as passed in.
        chunk_size: Wan temporal chunk size (4).
        lpips_fn: optional callable taking two [N,C,H,W] tensors in [-1,1]
            and returning per-image distances (pip ``lpips`` or opensora LPIPS).

    Returns dict with:
        psnr (mean of per-(clip,view) PSNRs - THE headline convention),
        psnr_pooled (PSNR of pooled MSE, reference only), psnr_per_sample,
        psnr_per_view, psnr_per_frame, ssim (windowed), ssim_global (legacy),
        lpips (if lpips_fn given), bleed_ratio_within/across,
        xview_rec, xview_gt.
    """
    gt01 = to_unit_range(_as_bvcthw(gt), value_range)
    rec01 = to_unit_range(_as_bvcthw(rec), value_range)
    if gt01.shape != rec01.shape:
        raise ValueError(f"shape mismatch: gt {tuple(gt01.shape)} vs rec {tuple(rec01.shape)}")
    b, v, c, t, h, w = gt01.shape

    per_sample = psnr_per_sample(gt01, rec01)  # [B*V]
    pooled_mse = float(((gt01 - rec01) ** 2).double().mean().item())
    out = {
        "psnr": float(per_sample.mean().item()),
        "psnr_pooled": math.inf if pooled_mse == 0 else 10.0 * math.log10(1.0 / pooled_mse),
        "psnr_per_sample": [float(p) for p in per_sample],
        "psnr_per_view": [
            float(per_sample.reshape(b, v)[:, vi].mean().item()) for vi in range(v)
        ],
    }

    # per-frame: PSNR of the MSE pooled over samples at each frame index
    mse_f = ((gt01 - rec01) ** 2).double().mean(dim=(0, 1, 2, 4, 5))  # [T]
    out["psnr_per_frame"] = [
        math.inf if m == 0 else float(10.0 * torch.log10(1.0 / m)) for m in mse_f
    ]

    out["ssim"] = ssim_windowed(gt01, rec01)
    out["ssim_global"] = ssim_global(gt01, rec01)

    if lpips_fn is not None:
        a = gt01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w) * 2 - 1
        bb = rec01.permute(0, 1, 3, 2, 4, 5).reshape(b * v * t, c, h, w) * 2 - 1
        vals = []
        for i in range(0, a.shape[0], lpips_batch):
            d = lpips_fn(a[i : i + lpips_batch], bb[i : i + lpips_batch])
            vals.append(d.reshape(-1).detach().float().cpu())
        out["lpips"] = float(torch.cat(vals).mean().item())

    if t >= 2:
        out.update(bleed_ratios(gt01, rec01, chunk_size))
    xr = cross_view_cosine(rec01)
    if xr is not None:
        out["xview_rec"] = xr
        out["xview_gt"] = cross_view_cosine(gt01)
    return out
