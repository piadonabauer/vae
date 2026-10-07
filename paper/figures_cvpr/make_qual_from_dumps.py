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
# 2026-10-04: repointed from the dead Vector cluster to the local re-run wave
# (uniform 170-epoch budget, seed 42). Uses final_eval_dump_val.pt so images
# match the reported final_eval/val numbers exactly.
DUMP_ROOT = Path("/home/coder/vae/Open-Sora/outputs")

DUMPS = {
    # E1a still training locally; add "paper_E1a_perview_tcF__job1791135751_t1"
    # once its final dump exists (not used by any figure below yet).
    "zeroshot": "paper_E1z_perview_zeroshot_tcON__job1791026450_t50",
    "E1b": "paper_E1b_perview_tcT__job1791027540_t2",
    "E1c": "paper_E1c_fused_tcF__job1791042863_t3",
    "E1d": "paper_E1d_fused_tcT__job1791058355_t4",
    "E11a": "paper_E11a_fused_tcT_widen32__job1791073606_t8",
    "E11b": "paper_E11b_fused_tcT_widen64__job1791088848_t9",
    "combo": "paper_E_combo_diffLoss_widen32__job1791104094_t36",
    "Ebest": "paper_Ebest_allcombined__job1791119530_t52",
    # @300 dumps -- capacity figure only (Table-3 budget; everything else @170).
    "E1d_300": "paper_E1d_fused_tcT__job1791225734_t4",
    "E11a_300": "paper_E11a_fused_tcT_widen32__job1791237549_t8",
    "E11b_300": "paper_E11b_fused_tcT_widen64__job1791189563_t9",
}

DUMP_FILE = "final_eval_dump_val.pt"

# ONE identity + ONE frame shared by the ghosting (failure) and capacity
# (repair) figures, so a reader can line the two up as a single story.
# f5 is the hardest frame (chunk-2 interior, per-frame PSNR minimum).
# The bleeding grid intentionally differs: it shows frames f1-f4 (chunk 1)
# of the SAME clip, because its subject is within-chunk dynamics.
QUAL_CLIP = 0
QUAL_FRAME = 5


def load_clip(arm: str, clip_idx: int = QUAL_CLIP, _gt_ref={}):
    d = DUMP_ROOT / DUMPS[arm]
    data = torch.load(d / DUMP_FILE, map_location="cpu", weights_only=False)
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
    """2 views x methods: GT, E1c, E1d, E11b — same clip+frame as capacity fig.
    NOTE: figure labels must stay descriptive — never show experiment codenames."""
    arms = [
        ("GT", None),
        ("Fused, TC off", "E1c"),
        ("Fused, TC on", "E1d"),
        ("Fused, TC on, 64-ch", "E11b"),
    ]
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
    arms = [("GT", None), ("16-ch", "E1d"), ("32-ch", "E11a"), ("64-ch", "E11b")]
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
    """GT / fused 16/32/64-ch — same clip+frame as the ghosting fig (failure
    there, repair here: one story). Uses the @300 dumps (Table-3 budget);
    the @170 dumps had 64-ch below 32-ch, an artifact of the shorter budget."""
    arms = [("GT", None), ("16-ch", "E1d_300"), ("32-ch", "E11a_300"), ("64-ch", "E11b_300")]
    gt, _ = load_clip("E1d_300")
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


def err_to_img(gt_chw, rec_chw, gain=5.0):
    """Amplified absolute-error heatmap (inferno), mean over channels."""
    err = (gt_chw - rec_chw).abs().mean(0).numpy() * gain
    cmap = plt.get_cmap("inferno")
    return (cmap(err.clip(0, 1))[..., :3] * 255).astype(np.uint8)


def fig_qual_overview():
    """All final-wave arms on one clip+frame: both views + x5 error map.

    Not a paper figure (too wide) -- a working visual so quality differences
    behind the PSNR table are directly inspectable.
    """
    arms = [
        ("GT", None),
        ("Zero-shot\nper-view, 36x", "zeroshot"),
        ("Per-view\nTC on, 36x", "E1b"),
        ("Fused\nTC off, 36x", "E1c"),
        ("Fused\nTC on, 72x", "E1d"),
        ("Fused TC on\n32-ch, 36x", "E11a"),
        ("32-ch +\ndiff-loss, 36x", "combo"),
        ("Fused TC on\n64-ch, 18x", "E11b"),
        ("All tweaks\n18x", "Ebest"),
    ]
    gt, _ = load_clip("E1d")
    t = QUAL_FRAME
    fig, axes = plt.subplots(3, len(arms), figsize=(1.45 * len(arms), 4.9))
    fig.subplots_adjust(wspace=0.04, hspace=0.26, left=0.05, right=0.995, top=0.88, bottom=0.05)
    for c, (lab, arm) in enumerate(arms):
        rec = None if arm is None else load_clip(arm)[1]
        for v in range(2):
            if arm is None:
                img, xlab = frame_to_img(gt[v, :, t]), None
            else:
                img = frame_to_img(rec[v, :, t])
                xlab = f"{psnr(gt[v, :, t], rec[v, :, t]):.1f} dB"
            show(axes[v, c], img, title=(lab if v == 0 else None), xlabel=(xlab if v == 1 else None))
        if arm is None:
            em = np.zeros((*gt.shape[-2:], 3), dtype=np.uint8)
        else:
            em = err_to_img(gt[0, :, t], rec[0, :, t])
        show(axes[2, c], em)
    axes[0, 0].set_ylabel("view 0", fontsize=7)
    axes[1, 0].set_ylabel("view 1", fontsize=7)
    axes[2, 0].set_ylabel("|err| x5 (v0)", fontsize=7)
    save(fig, "qual_overview_finalwave.pdf")


def fig_qual_best_temporal():
    """GT vs no-tweaks fused (E1d) vs E_best across all 9 frames: shows where
    in the chunk structure the tweak stack helps (chunk-interior frames)."""
    arms = [("GT", None), ("Fused TC on\n(no tweaks)", "E1d"), ("Fused TC on\n(all tweaks)", "Ebest")]
    gt, _ = load_clip("E1d")
    v = 0
    T = gt.shape[2]
    fig, axes = plt.subplots(len(arms), T, figsize=(1.1 * T, 3.9))
    fig.subplots_adjust(wspace=0.03, hspace=0.3, left=0.07, right=0.995, top=0.93, bottom=0.07)
    for r, (lab, arm) in enumerate(arms):
        rec = None if arm is None else load_clip(arm)[1]
        for t in range(T):
            if arm is None:
                img, xlab = frame_to_img(gt[v, :, t]), None
            else:
                img = frame_to_img(rec[v, :, t])
                xlab = f"{psnr(gt[v, :, t], rec[v, :, t]):.1f}"
            show(axes[r, t], img, title=(f"$f_{t}$" if r == 0 else None), xlabel=xlab)
        axes[r, 0].set_ylabel(lab, fontsize=7)
    save(fig, "qual_best_temporal.pdf")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    print("Saved", name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_qual_ghosting()
    fig_qual_bleeding()
    fig_qual_capacity()
    fig_qual_overview()
    fig_qual_best_temporal()
    print("done")
