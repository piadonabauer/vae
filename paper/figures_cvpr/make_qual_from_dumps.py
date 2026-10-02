#!/usr/bin/env python3
"""Rebuild qualitative grids from clean best_val_eval_dump.pt files."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

OUT = Path(__file__).resolve().parent
DUMP_ROOT = Path("/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs")

DUMPS = {
    "E1a": "paper_E1a_perview_tcF__job5562190_t1",
    "E1c": "paper_E1c_fused_tcF__job5562191_t3",
    "E1d": "paper_E1d_fused_tcT__job5562192_t4",
    "E11a": "paper_E11a_fused_tcT_widen32__job5562201_t8",
    "E11b": "paper_E11b_fused_tcT_widen64__job5562202_t9",
    "combo": "paper_E_combo_diffLoss_widen32__job5562206_t36",
}

# ONE identity + ONE frame shared by the ghosting (failure) and capacity
# (repair) figures, so a reader can line the two up as a single story.
# f5 is the hardest frame (chunk-2 interior, per-frame PSNR minimum).
# The bleeding grid intentionally differs: it shows frames f1-f4 (chunk 1)
# of the SAME clip, because its subject is within-chunk dynamics.
QUAL_CLIP = 0
QUAL_FRAME = 5


def load_clip(arm: str, clip_idx: int = QUAL_CLIP, _gt_ref={}):
    d = DUMP_ROOT / DUMPS[arm]
    data = torch.load(d / "best_val_eval_dump.pt", map_location="cpu", weights_only=False)
    clip = data["clips"][clip_idx]
    gt = clip["gt"]
    rec = clip["rec"]
    if gt.dtype == torch.uint8:
        gt = gt.float() / 255.0
        rec = rec.float() / 255.0
    # Safety: every arm's dump must contain the SAME clip at this index
    # (deterministic eval order). Guards against silently comparing identities.
    ref = _gt_ref.setdefault(clip_idx, gt)
    assert torch.allclose(ref, gt, atol=2 / 255), \
        f"{arm}: GT at clip_idx={clip_idx} differs from reference dump -- eval order mismatch"
    return gt, rec  # (V,C,T,H,W)


def frame_to_img(x_chw):
    arr = (x_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return arr


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    if mse < 1e-10:
        return 99.0
    return 10 * np.log10(1.0 / mse)


def show(ax, img, title=None, xlabel=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=1)


def fig_qual_ghosting():
    """2 views x methods: GT, E1c, E1d, E11b — same clip+frame as capacity fig."""
    arms = [("GT", None), ("E1c", "E1c"), ("E1d", "E1d"), ("E11b", "E11b")]
    gt, _ = load_clip("E1d")
    t = QUAL_FRAME
    fig, axes = plt.subplots(2, 4, figsize=(6.875, 3.4))
    fig.subplots_adjust(wspace=0.04, hspace=0.08, left=0.04, right=0.99, top=0.92, bottom=0.08)
    for v in range(2):
        for c, (lab, arm) in enumerate(arms):
            if arm is None:
                img = frame_to_img(gt[v, :, t])
                xlab = None
            else:
                _, rec = load_clip(arm)
                img = frame_to_img(rec[v, :, t])
                xlab = f"{psnr(gt[v, :, t], rec[v, :, t]):.1f} dB"
            show(axes[v, c], img, title=lab if v == 0 else None, xlabel=xlab)
        axes[v, 0].set_ylabel(f"view {v}", fontsize=7)
    save(fig, "qual_ghosting.pdf")


def fig_qual_bleeding():
    """Rows: GT / 16-ch / 32-ch / 64-ch; cols: frames in chunk 1 (f1..f4).
    (64-ch row added 2026-10-02; regenerate on the cluster, then rerun
    figures_cvpr/make_figures_cvpr.py which re-crops this PDF.)"""
    arms = [("GT", None), ("16ch", "E1d"), ("32ch", "E11a"), ("64ch", "E11b")]
    frames = [1, 2, 3, 4]
    gt, _ = load_clip("E1d")
    v = 0
    fig, axes = plt.subplots(len(arms), 4, figsize=(6.875, 6.6))
    fig.subplots_adjust(wspace=0.03, hspace=0.12, left=0.06, right=0.99, top=0.95, bottom=0.04)
    for r, (lab, arm) in enumerate(arms):
        for c, t in enumerate(frames):
            if arm is None:
                img = frame_to_img(gt[v, :, t])
                xlab = f"$f_{t}$"
            else:
                _, rec = load_clip(arm)
                img = frame_to_img(rec[v, :, t])
                xlab = f"{psnr(gt[v, :, t], rec[v, :, t]):.1f} dB"
            show(axes[r, c], img, title=(f"$f_{t}$" if r == 0 else None), xlabel=xlab)
        axes[r, 0].set_ylabel(lab, fontsize=7)
    save(fig, "qual_bleeding.pdf")


def fig_qual_capacity():
    """GT / E1d / E11a / E11b — same clip+frame as the ghosting fig (failure
    there, repair here: one story)."""
    arms = [("GT", None), ("16-ch", "E1d"), ("32-ch", "E11a"), ("64-ch", "E11b")]
    gt, _ = load_clip("E1d")
    t, v = QUAL_FRAME, 0
    fig, axes = plt.subplots(1, 4, figsize=(6.875, 1.85))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.99, top=0.88, bottom=0.12)
    for c, (lab, arm) in enumerate(arms):
        if arm is None:
            img = frame_to_img(gt[v, :, t])
            xlab = None
        else:
            _, rec = load_clip(arm)
            img = frame_to_img(rec[v, :, t])
            xlab = f"{psnr(gt[v, :, t], rec[v, :, t]):.1f} dB"
        show(axes[c], img, title=lab, xlabel=xlab)
    save(fig, "qual_capacity.pdf")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    print("Saved", name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_qual_ghosting()
    fig_qual_bleeding()
    fig_qual_capacity()
    print("done")
