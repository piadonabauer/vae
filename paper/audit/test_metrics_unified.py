"""Unit tests for metrics_unified.py (run: python -m pytest test_metrics_unified.py -q).

Covers exactly the audit checklist:
  1. identical inputs  -> PSNR = inf, SSIM = 1, LPIPS ~ 0
  2. Gaussian noise of known sigma -> analytic PSNR = -10*log10(sigma^2)
  3. pure white vs pure black -> PSNR 0 dB, SSIM ~ C1/(1+C1)
  4. a [-1,1] tensor and its [0,1] remap give IDENTICAL results
  5. Bleed-W within-chunk pairs are {1-2,2-3,3-4} and {5-6,6-7,7-8};
     frame 0 is in no within pair
  6. aggregation: psnr (mean of per-sample) != psnr_pooled on heterogeneous
     batches, and per-view mean reproduces the headline mean
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_unified import (  # noqa: E402
    bleed_pair_indices,
    compute_all_metrics,
    to_unit_range,
)

B, V, C, T, H, W = 2, 2, 3, 9, 32, 32


def _rand01(seed=0, shape=(B, V, C, T, H, W)):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g)


def _lpips_fn():
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def test_identical_inputs():
    x = _rand01(0)
    m = compute_all_metrics(x, x.clone(), value_range="[0,1]")
    assert math.isinf(m["psnr"]) and m["psnr"] > 0
    assert math.isinf(m["psnr_pooled"])
    assert all(math.isinf(p) for p in m["psnr_per_sample"])
    assert all(math.isinf(p) for p in m["psnr_per_frame"])
    assert m["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert m["ssim_global"] == pytest.approx(1.0, abs=1e-6)
    lp = _lpips_fn()
    if lp is not None:
        m2 = compute_all_metrics(x, x.clone(), value_range="[0,1]", lpips_fn=lp)
        assert m2["lpips"] == pytest.approx(0.0, abs=1e-5)


def test_gaussian_noise_analytic_psnr():
    # mid-gray base so +-4 sigma of noise cannot clip at 0/1 (clipping would
    # make measured PSNR slightly HIGHER than analytic)
    sigma = 0.01
    base = torch.full((B, V, C, T, H, W), 0.5)
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(base.shape, generator=g) * sigma
    m = compute_all_metrics(base, base + noise, value_range="[0,1]")
    expected = -10.0 * math.log10(sigma**2)  # 40 dB for sigma=0.01
    assert m["psnr"] == pytest.approx(expected, abs=0.05)
    assert m["psnr_pooled"] == pytest.approx(expected, abs=0.05)


def test_white_vs_black():
    white = torch.ones(1, 1, C, T, H, W)
    black = torch.zeros(1, 1, C, T, H, W)
    m = compute_all_metrics(white, black, value_range="[0,1]")
    assert m["psnr"] == pytest.approx(0.0, abs=1e-6)  # MSE = 1 -> 0 dB
    # SSIM(1, 0) = (2*1*0+C1)(0+C2) / ((1+0+C1)(0+0+C2)) = C1/(1+C1)
    c1 = 0.01**2
    assert m["ssim"] == pytest.approx(c1 / (1 + c1), abs=1e-6)
    assert m["ssim_global"] == pytest.approx(c1 / (1 + c1), abs=1e-6)


def test_range_equivalence_minus11_vs_01():
    gt01 = _rand01(2)
    rec01 = (gt01 + 0.05 * torch.randn(gt01.shape, generator=torch.Generator().manual_seed(3))).clamp(0, 1)
    m01 = compute_all_metrics(gt01, rec01, value_range="[0,1]")
    m11 = compute_all_metrics(gt01 * 2 - 1, rec01 * 2 - 1, value_range="[-1,1]")
    for k in ("psnr", "psnr_pooled", "ssim", "ssim_global",
              "bleed_ratio_within", "bleed_ratio_across", "xview_rec", "xview_gt"):
        assert m01[k] == pytest.approx(m11[k], rel=1e-5, abs=1e-6), k
    assert m01["psnr_per_view"] == pytest.approx(m11["psnr_per_view"], rel=1e-5)
    assert m01["psnr_per_frame"] == pytest.approx(m11["psnr_per_frame"], rel=1e-5)


def test_clamp_bug_is_reproduced_by_wrong_declaration():
    """Sanity: declaring [-1,1] data as [0,1] (the old bug) must change PSNR."""
    gt01 = _rand01(4)
    rec01 = (gt01 + 0.05 * torch.randn(gt01.shape, generator=torch.Generator().manual_seed(5))).clamp(0, 1)
    correct = compute_all_metrics(gt01, rec01, value_range="[0,1]")["psnr"]
    # feed [-1,1] tensors but (wrongly) declare them [0,1] -> internal clamp
    # crushes negatives, exactly the pre-40295bb behavior
    buggy = compute_all_metrics(gt01 * 2 - 1, rec01 * 2 - 1, value_range="[0,1]")["psnr"]
    assert buggy != pytest.approx(correct, abs=0.2)


def test_bleed_chunk_indices_T9():
    within, across = bleed_pair_indices(9, chunk_size=4)
    # pair i is (frame i, frame i+1)
    assert within == [1, 2, 3, 5, 6, 7]  # 1-2, 2-3, 3-4, 5-6, 6-7, 7-8
    assert across == [0, 4]              # 0-1 and 4-5 cross boundaries
    assert 0 not in within               # frame-0 pair excluded from within


def test_bleed_detects_frozen_chunk():
    """A reconstruction frozen inside chunks but matching across boundaries
    must give bleed_within << bleed_across."""
    g = torch.Generator().manual_seed(6)
    gt = torch.rand(1, 1, C, T, H, W, generator=g)
    rec = gt.clone()
    for chunk in ([1, 2, 3, 4], [5, 6, 7, 8]):
        rec[:, :, :, chunk] = gt[:, :, :, chunk[0]].unsqueeze(3)  # freeze chunk
    m = compute_all_metrics(gt, rec, value_range="[0,1]")
    assert m["bleed_ratio_within"] == pytest.approx(0.0, abs=1e-5)
    assert m["bleed_ratio_across"] > 0.5


def test_aggregation_convention():
    """Headline = mean of per-sample PSNRs; differs from pooled-MSE PSNR on
    heterogeneous batches, and equals the mean over psnr_per_view."""
    gt = _rand01(7)
    # view 0 nearly perfect, view 1 noisy -> conventions must diverge
    rec = gt.clone()
    rec[:, 0] = (gt[:, 0] + 0.001 * torch.randn(gt[:, 0].shape, generator=torch.Generator().manual_seed(8)))
    rec[:, 1] = (gt[:, 1] + 0.100 * torch.randn(gt[:, 1].shape, generator=torch.Generator().manual_seed(9)))
    rec = rec.clamp(0, 1)
    m = compute_all_metrics(gt, rec, value_range="[0,1]")
    import numpy as np
    assert m["psnr"] == pytest.approx(float(np.mean(m["psnr_per_sample"])), abs=1e-6)
    assert m["psnr"] == pytest.approx(float(np.mean(m["psnr_per_view"])), abs=1e-6)
    assert abs(m["psnr"] - m["psnr_pooled"]) > 1.0  # conventions truly differ


def test_to_unit_range_validation():
    with pytest.raises(ValueError):
        to_unit_range(torch.zeros(2), "[0,255]")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
