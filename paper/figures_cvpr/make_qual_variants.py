#!/usr/bin/env python3
"""Draft figure variants (saved under *_v2 / new names; originals untouched).

- qual_ghosting_v2: GT | per-view TC on | fused TC off | fused TC on; both
  views, zoom insets (mouth, hair parting), |diff| map. No 64-ch column.
- qual_capacity_v2: GT | 16 | 32 | 64 | all-tweaks; same identity+frame as
  the ghosting figure, same insets.
- perframe_psnr_v2: per-frame PSNR (mean over val clips+views), legend
  outside the axes, per-view TC-on reference line, chunk boundaries marked.
- perframe_motion: mean |x_k - x_{k-1}| per frame for GT vs reconstructions
  (the temporal-bleeding evidence: flattened motion inside chunks).

HARD RULE: no experiment codenames in rendered text (see 03_figures.md).
NOTE: currently reads the uniform 170-epoch final dumps. After the 300-epoch
extensions land, switch DUMPS to the @300 job dirs and rerun.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
DUMP_ROOT = Path("/home/coder/vae/Open-Sora/outputs")

# Budget rule: each figure uses the budget of the table it accompanies.
# qual_ghosting_v2 (Table 1 arms) -> @170; capacity + per-frame figures
# (Table 3 arms) -> @300. Keys are internal only.
DUMPS_170 = {
    "E1b": "paper_E1b_perview_tcT__job1791027540_t2",
    "E1c": "paper_E1c_fused_tcF__job1791042863_t3",
    "E1d": "paper_E1d_fused_tcT__job1791058355_t4",
    "E11a": "paper_E11a_fused_tcT_widen32__job1791073606_t8",
    "E11b": "paper_E11b_fused_tcT_widen64__job1791088848_t9",
    "combo": "paper_E_combo_diffLoss_widen32__job1791104094_t36",
    "Ebest": "paper_Ebest_allcombined__job1791119530_t52",
}
DUMPS_300 = {
    "E1b": "paper_E1b_perview_tcT__job1791249364_t2",
    "E1d": "paper_E1d_fused_tcT__job1791225734_t4",
    "E11a": "paper_E11a_fused_tcT_widen32__job1791237549_t8",
    "E11b": "paper_E11b_fused_tcT_widen64__job1791189563_t9",
    "combo": "paper_E_combo_diffLoss_widen32__job1791201374_t36",
    "Ebest": "paper_Ebest_allcombined__job1791213193_t52",
}
DUMPS = DUMPS_170

QUAL_CLIP = 0
QUAL_FRAME = 5
# Inset boxes (y0, y1, x0, x1) on the 128x128 frame.
BOX_MOUTH = (76, 110, 46, 82)
# subject's left eye (right side of the image)
BOX_EYE = (44, 72, 52, 84)

_cache: dict = {}


def load_clip(arm: str, clip_idx: int = QUAL_CLIP):
    key = (arm, clip_idx)
    if key not in _cache:
        d = DUMP_ROOT / DUMPS[arm]
        data = torch.load(d / "final_eval_dump_val.pt", map_location="cpu", weights_only=False)
        clip = data["clips"][clip_idx]
        gt = clip["gt"].float() / 255.0
        rec = clip["rec"].float() / 255.0
        _cache[key] = (gt, rec)
    return _cache[key]


def set_budget(dumps):
    global DUMPS
    DUMPS = dumps
    _cache.clear()


def load_all(arm: str):
    """All clips as (gt, rec) float tensors [N,V,C,T,H,W]."""
    d = DUMP_ROOT / DUMPS[arm]
    data = torch.load(d / "final_eval_dump_val.pt", map_location="cpu", weights_only=False)
    gt = torch.stack([c["gt"].float() / 255.0 for c in data["clips"]])
    rec = torch.stack([c["rec"].float() / 255.0 for c in data["clips"]])
    return gt, rec


def to_img(x_chw):
    return (x_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def diff_img(gt_chw, rec_chw, gain=5.0):
    err = (gt_chw - rec_chw).abs().mean(0).numpy() * gain
    cmap = plt.get_cmap("inferno")
    return (cmap(err.clip(0, 1))[..., :3] * 255).astype(np.uint8)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def show(ax, img, title=None, xlabel=None, boxes=False):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if boxes:
        for box, color in [(BOX_MOUTH, "#00d0ff"), (BOX_EYE, "#ffd000")]:
            y0, y1, x0, x1 = box
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                            fill=False, edgecolor=color, linewidth=1.0))
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=1)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    print("Saved", name)


def fig_ghosting_v2():
    arms = [
        ("GT", None),
        ("Per-view\nTC on", "E1b"),
        ("Fused\nTC off", "E1c"),
        ("Fused\nTC on", "E1d"),
    ]
    gt, _ = load_clip("E1d")
    t = QUAL_FRAME
    rows = ["view 0", "view 1", "mouth (v0)", "eye (v0)", "|diff| x5 (v0)"]
    fig, axes = plt.subplots(len(rows), len(arms), figsize=(1.72 * len(arms), 8.2))
    fig.subplots_adjust(wspace=0.04, hspace=0.1, left=0.08, right=0.99, top=0.94, bottom=0.03)
    for c, (lab, arm) in enumerate(arms):
        rec = None if arm is None else load_clip(arm)[1]
        img0 = to_img(gt[0, :, t]) if arm is None else to_img(rec[0, :, t])
        img1 = to_img(gt[1, :, t]) if arm is None else to_img(rec[1, :, t])
        xlab0 = None if arm is None else f"{psnr(gt[0, :, t], rec[0, :, t]):.1f} dB"
        xlab1 = None if arm is None else f"{psnr(gt[1, :, t], rec[1, :, t]):.1f} dB"
        show(axes[0, c], img0, title=lab, xlabel=xlab0, boxes=(arm is None))
        show(axes[1, c], img1, xlabel=xlab1)
        show(axes[2, c], crop(img0, BOX_MOUTH))
        show(axes[3, c], crop(img0, BOX_EYE))
        dm = np.zeros((128, 128, 3), np.uint8) if arm is None else diff_img(gt[0, :, t], rec[0, :, t])
        show(axes[4, c], dm)
    for r, lab in enumerate(rows):
        axes[r, 0].set_ylabel(lab, fontsize=7)
    save(fig, "qual_ghosting_v2.pdf")


def fig_capacity_v2():
    arms = [
        ("GT", None),
        ("16-ch", "E1d"),
        ("32-ch", "E11a"),
        ("64-ch", "E11b"),
        ("Best combination", "Ebest"),
    ]
    gt, _ = load_clip("E1d")
    t, v = QUAL_FRAME, 0
    rows = ["view 0", "mouth", "eye", "|diff| x5"]
    fig, axes = plt.subplots(len(rows), len(arms), figsize=(1.72 * len(arms), 6.6))
    fig.subplots_adjust(wspace=0.04, hspace=0.1, left=0.07, right=0.99, top=0.93, bottom=0.03)
    for c, (lab, arm) in enumerate(arms):
        rec = None if arm is None else load_clip(arm)[1]
        img = to_img(gt[v, :, t]) if arm is None else to_img(rec[v, :, t])
        xlab = None if arm is None else f"{psnr(gt[v, :, t], rec[v, :, t]):.1f} dB"
        show(axes[0, c], img, title=lab, xlabel=xlab, boxes=(arm is None))
        show(axes[1, c], crop(img, BOX_MOUTH))
        show(axes[2, c], crop(img, BOX_EYE))
        dm = np.zeros((128, 128, 3), np.uint8) if arm is None else diff_img(gt[v, :, t], rec[v, :, t])
        show(axes[3, c], dm)
    for r, lab in enumerate(rows):
        axes[r, 0].set_ylabel(lab, fontsize=7)
    save(fig, "qual_capacity_v2.pdf")


# Wan causal chunking at T=9: f0 | f1-f4 | f5-f8.
CHUNKS = [(0, 0), (1, 4), (5, 8)]

ARMS_CURVES = [
    ("Fused TC on (16-ch)", "E1d", "#d62728"),
    ("Fused TC on, 32-ch", "E11a", "#ff7f0e"),
    ("Fused TC on, 64-ch", "E11b", "#2ca02c"),
    ("32-ch + diff-loss", "combo", "#9467bd"),
    ("Best combination", "Ebest", "#1f77b4"),
]


def fig_perframe_psnr_v2():
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for lab, arm, color in ARMS_CURVES:
        gt, rec = load_all(arm)
        vals = [psnr(gt[..., t, :, :], rec[..., t, :, :]) for t in range(gt.shape[3])]
        ax.plot(range(len(vals)), vals, marker="o", ms=3, lw=1.3, color=color, label=lab)
    gt, rec = load_all("E1b")
    ref = [psnr(gt[..., t, :, :], rec[..., t, :, :]) for t in range(gt.shape[3])]
    ax.plot(range(len(ref)), ref, ls="--", lw=1.2, color="0.35", label="Per-view TC on (ref)")
    for a, b in CHUNKS[1:]:
        ax.axvline(a - 0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("frame index", fontsize=8)
    ax.set_ylabel("PSNR (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(left=0.1, right=0.68, top=0.97, bottom=0.16)
    save(fig, "perframe_psnr_v2.pdf")


def fig_perframe_motion():
    """Reconstructed motion as a fraction of GT motion per frame transition,
    computed on moving pixels only (GT frame-diff > 2/255, excludes the static
    white background). <100% inside chunks = temporal bleeding."""
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    gt, _ = load_all("E1d")
    gt_d = (gt[:, :, :, 1:] - gt[:, :, :, :-1]).abs()
    mask = (gt_d.mean(2, keepdim=True) > 2 / 255).float()  # [N,V,1,T-1,H,W]
    gt_m = (gt_d * mask).sum(dim=(0, 1, 2, 4, 5)) / mask.sum(dim=(0, 1, 2, 4, 5)).clamp(min=1)
    ax.axhline(100, color="black", lw=1.6, label="GT (100%)")
    for lab, arm, color in [
        ("Per-view TC on", "E1b", "0.35"),
        ("Fused TC on (16-ch)", "E1d", "#d62728"),
        ("Fused TC on, 64-ch", "E11b", "#2ca02c"),
        ("Best combination", "Ebest", "#1f77b4"),
    ]:
        _, rec = load_all(arm)
        rec_d = (rec[:, :, :, 1:] - rec[:, :, :, :-1]).abs()
        rec_m = (rec_d * mask).sum(dim=(0, 1, 2, 4, 5)) / mask.sum(dim=(0, 1, 2, 4, 5)).clamp(min=1)
        ratio = (rec_m / gt_m * 100).numpy()
        ls = "--" if arm == "E1b" else "-"
        ax.plot(range(1, len(ratio) + 1), ratio, marker="o", ms=3, lw=1.3,
                color=color, ls=ls, label=lab)
    for a, b in CHUNKS[1:]:
        ax.axvline(a - 0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("frame transition $k$  (motion $= |x_k - x_{k-1}|$ on moving pixels)", fontsize=8)
    ax.set_ylabel("decoded motion (% of GT)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(left=0.11, right=0.68, top=0.97, bottom=0.17)
    save(fig, "perframe_motion.pdf")


if __name__ == "__main__":
    set_budget(DUMPS_170)   # Table 1 budget
    fig_ghosting_v2()
    set_budget(DUMPS_300)   # Table 3 (capacity) budget
    fig_capacity_v2()
    fig_perframe_psnr_v2()
    fig_perframe_motion()
    print("done")
