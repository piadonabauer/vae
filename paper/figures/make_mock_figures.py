"""
Mock versions of the paper figures, for layout discussions only.

Everything with data in it uses made-up numbers (watermarked MOCK).
The real figures are generated from the rerun results (collect_results.py CSV
+ the eval clip dumps) once the sweep is done.

The thumbnails in assets/ are center-cropped frames from NeRSemble (EMO-1
sequence, frame 100). view1/view2.png are participant 451 through the V=2
camera pair (222200037 / 220700191). p{id}_view{1..4}.png cover several
participants through the four frontal upper-row cameras, left to right
(222200047, 222200037, 220700191, 222200036) -- the same selection
select_upper_middle_cameras() makes at n_views=4. Everything in assets/ is
gitignored on purpose: the dataset license does not allow republishing images,
so don't commit them. If assets/ is missing the script falls back to gray
placeholders.

Run:  python3 make_mock_figures.py   (writes PNGs next to this file)
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = os.path.dirname(os.path.abspath(__file__))

# CVPR template uses Times; STIXGeneral is the closest Times-like font that
# ships with matplotlib.
matplotlib.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# color code used in all schematics
C_FROZEN = "#aecbe8"   # blue: frozen pretrained
C_NEW = "#f5b942"      # orange: new modules, zero-init
C_LORA = "#8fd19e"     # green: LoRA adapters
C_LATENT = "#d9b3e6"   # purple: the latent
C_GT = "#444444"


def load_view(name):
    path = os.path.join(OUT, "assets", f"{name}.png")
    if os.path.exists(path):
        return plt.imread(path)
    return np.full((64, 64, 3), 0.82)  # gray placeholder if assets are missing


def thumb(ax, img, x, y, s, label=None, fontsize=7.5, alpha=1.0):
    """Draw a square image thumbnail with an optional label below it."""
    ax.imshow(img, extent=(x, x + s, y, y + s), zorder=3, alpha=alpha)
    ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec="black", lw=0.8, zorder=4))
    if label:
        ax.text(x + s / 2, y - 0.09, label, ha="center", va="top", fontsize=fontsize)


def box(ax, x, y, w, h, color, text, fontsize=8.5, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=color, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color="black", **kw))


# ------------------------------------------------------- broad overview
def fig_broad_overview():
    """The one-glance version: input -> encoder -> latent -> decoder -> output."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    v1, v2 = load_view("view1"), load_view("view2")

    # input stack: V views (only two drawn), offset like a card stack
    thumb(ax, v2, 0.75, 1.05, 0.95)
    thumb(ax, v1, 0.55, 0.85, 0.95)
    ax.text(1.15, 0.55, "input $x$\n$[V, 3, T, H, W]$\n$V{=}2,\\ T{=}9,\\ 128^2$",
            ha="center", va="top", fontsize=8)

    # encoder trapezoid (wide -> narrow)
    ax.add_patch(Polygon([(2.4, 0.6), (2.4, 2.4), (4.0, 1.9), (4.0, 1.1)],
                         fc=C_FROZEN, ec="black", lw=0.8))
    ax.text(3.2, 1.5, "encoder\n(Wan, frozen\n+ fusion)", ha="center", va="center", fontsize=8.5)
    arrow(ax, 1.85, 1.5, 2.35, 1.5)

    # latent
    box(ax, 4.5, 1.15, 0.85, 0.7, C_LATENT, "$z$")
    ax.text(4.92, 0.95, "$[16, T', \\frac{H}{8}, \\frac{W}{8}]$\n$T' = 1 + \\frac{T-1}{4} = 3$",
            ha="center", va="top", fontsize=8)
    arrow(ax, 4.05, 1.5, 4.45, 1.5)
    ax.text(4.92, 2.15, "one shared latent\nfor all $V$ views", ha="center", va="bottom",
            fontsize=7.5, style="italic")

    # decoder trapezoid (narrow -> wide)
    ax.add_patch(Polygon([(5.9, 1.1), (5.9, 1.9), (7.5, 2.4), (7.5, 0.6)],
                         fc=C_FROZEN, ec="black", lw=0.8))
    ax.text(6.7, 1.5, "decoder\n(Wan, frozen\n+ per-view LoRA)", ha="center", va="center", fontsize=8.5)
    arrow(ax, 5.4, 1.5, 5.85, 1.5)

    # output stack
    thumb(ax, v2, 8.35, 1.05, 0.95)
    thumb(ax, v1, 8.15, 0.85, 0.95)
    ax.text(8.75, 0.55, "output $\\hat{x}$\n$[V, 3, T, H, W]$",
            ha="center", va="top", fontsize=8)
    arrow(ax, 7.55, 1.5, 8.1, 1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_broad_overview.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")

    v1, v2 = load_view("view1"), load_view("view2")

    # inputs: real frames
    thumb(ax, v1, 0.25, 2.35, 0.85, "view 1  $[3,T,H,W]$", fontsize=7)
    thumb(ax, v2, 0.25, 0.75, 0.85, "view 2  $[3,T,H,W]$", fontsize=7)

    # shared frozen encoder stem (drawn once per view for the data flow)
    box(ax, 1.6, 2.35, 1.5, 0.8, C_FROZEN, "Wan encoder stem\n(frozen, shared)")
    box(ax, 1.6, 0.75, 1.5, 0.8, C_FROZEN, "Wan encoder stem\n(frozen, shared)")
    arrow(ax, 1.15, 2.77, 1.6, 2.77)
    arrow(ax, 1.15, 1.17, 1.6, 1.17)

    # fusion
    box(ax, 3.5, 1.5, 1.3, 0.9, C_NEW, "cross-view\nattention\n(zero-init)")
    arrow(ax, 3.1, 2.77, 3.6, 2.3)
    arrow(ax, 3.1, 1.17, 3.6, 1.6)
    box(ax, 5.1, 1.5, 1.0, 0.9, C_NEW, "tree\nmerge")
    arrow(ax, 4.8, 1.95, 5.1, 1.95)

    # bottleneck middle/head (frozen + LoRA) -> latent
    box(ax, 6.4, 1.5, 1.0, 0.9, C_FROZEN, "middle/\nhead")
    ax.add_patch(FancyBboxPatch((6.45, 2.43), 0.9, 0.28, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(6.9, 2.57, "LoRA", ha="center", va="center", fontsize=7)
    arrow(ax, 6.1, 1.95, 6.4, 1.95)
    box(ax, 7.7, 1.6, 0.7, 0.7, C_LATENT, "$z$\n16 ch")
    ax.text(8.05, 1.4, "$[16,T',\\frac{H}{8},\\frac{W}{8}]$", ha="center", va="top", fontsize=7)
    arrow(ax, 7.4, 1.95, 7.7, 1.95)

    # per-view decode
    box(ax, 8.7, 2.35, 1.2, 0.8, C_FROZEN, "Wan decoder\n(frozen)")
    box(ax, 8.7, 0.75, 1.2, 0.8, C_FROZEN, "Wan decoder\n(frozen)")
    ax.add_patch(FancyBboxPatch((8.75, 3.2), 0.65, 0.26, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(9.07, 3.33, "LoRA v=1", ha="center", va="center", fontsize=6.5)
    ax.add_patch(FancyBboxPatch((8.75, 0.38), 0.65, 0.26, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(9.07, 0.51, "LoRA v=2", ha="center", va="center", fontsize=6.5)
    arrow(ax, 8.4, 2.1, 8.7, 2.65)
    arrow(ax, 8.4, 1.8, 8.7, 1.25)

    # reconstructed outputs (same frames, slightly faded = "reconstruction")
    thumb(ax, v1, 10.05, 2.32, 0.85, "$\\hat{x}$ view 1", fontsize=7, alpha=0.85)
    thumb(ax, v2, 10.05, 0.72, 0.85, "$\\hat{x}$ view 2", fontsize=7, alpha=0.85)
    arrow(ax, 9.92, 2.75, 10.03, 2.75)
    arrow(ax, 9.92, 1.15, 10.03, 1.15)

    # legend
    for i, (c, t) in enumerate([(C_FROZEN, "frozen pretrained"), (C_NEW, "new, zero-init"),
                                (C_LORA, "LoRA (rank 32)"), (C_LATENT, "fused latent")]):
        box(ax, 1.7 + i * 2.1, 3.45, 0.25, 0.22, c, "")
        ax.text(2.02 + i * 2.1, 3.56, t, va="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_overview.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fusion mechanisms
def fusion_chain(ax, stages, title):
    """One fusion variant as a vertical pipeline: f1, f2 at the top, then the
    mechanism-specific stages, fused f at the bottom."""
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(title, fontsize=8.5)

    # the two per-view feature maps
    box(ax, 0.45, 5.4, 0.85, 0.55, "white", "$f_1$", fontsize=9)
    box(ax, 1.70, 5.4, 0.85, 0.55, "white", "$f_2$", fontsize=9)

    # stage boxes, evenly spaced between inputs and output
    n = len(stages)
    ys = np.linspace(4.4, 1.3, n)
    h = 0.62
    arrow(ax, 0.88, 5.38, 1.35, ys[0] + h + 0.02)
    arrow(ax, 2.12, 5.38, 1.65, ys[0] + h + 0.02)
    for i, ((label, color), y) in enumerate(zip(stages, ys)):
        box(ax, 0.45, y, 2.1, h, color, label, fontsize=7)
        if i > 0:
            arrow(ax, 1.5, ys[i - 1] - 0.02, 1.5, y + h + 0.02)

    box(ax, 1.15, 0.15, 0.7, 0.55, C_LATENT, "$f$", fontsize=9)
    arrow(ax, 1.5, ys[-1] - 0.02, 1.5, 0.74)


def fig_fusion():
    """The four fusion variants of the ablation. Each panel shows the actual
    stage sequence, so the differences are visible at a glance."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))

    fusion_chain(axes[0], [
        ("self-attention over all\n$V{\\cdot}N$ tokens (RMSNorm)", C_NEW),
        ("zero-init output proj\n(+ residual)", C_NEW),
        ("tree merge: pairwise\nconcat $\\to$ 2 ResBlocks", C_NEW),
    ], "(a) cross-view attention\n+ tree merge (default)")

    fusion_chain(axes[1], [
        ("self-attention over all\n$V{\\cdot}N$ tokens (LayerNorm)", C_NEW),
        ("standard-init MHA\n(+ residual)", C_NEW),
        ("flat concat of all views\n$\\to$ 2 ResBlocks", C_NEW),
    ], "(b) flat attention fusion\n(no zero-init, no tree)")

    fusion_chain(axes[2], [
        ("stack on channel axis\n$[2C, T, H', W']$", "white"),
        ("1x1x1 Conv3d $2C{\\to}C$\n+ GN + SiLU", C_NEW),
        ("2 symmetric (non-causal)\n3D ResBlocks", C_NEW),
    ], "(c) channel-concat\nConv3d")

    fusion_chain(axes[3], [
        ("stack on view axis\n$[C, V, T, H', W']$", "white"),
        ("spatial 3x3 Conv2d", C_NEW),
        ("temporal 3x3x3 Conv3d", C_NEW),
        ("view conv, kernel $(V,3,3)$\ncompresses $V \\to 1$", C_NEW),
    ], "(d) factorized (2+1+1)D\nconvolution")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_fusion.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- chunking
def fig_chunking():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.4, 3.6)
    ax.axis("off")

    chunk_colors = ["#cccccc", "#f9d9a0", "#a0c8f9"]
    chunks = [[0], [1, 2, 3, 4], [5, 6, 7, 8]]

    # encode: input frames -> latent frames
    for ci, frames in enumerate(chunks):
        for f in frames:
            box(ax, f, 2.8, 0.8, 0.5, chunk_colors[ci], str(f), fontsize=8)
        mid = (frames[0] + frames[-1]) / 2 + 0.4
        arrow(ax, mid, 2.75, mid, 2.15)
        box(ax, mid - 0.35, 1.6, 0.7, 0.5, C_LATENT, f"$z_{ci}$", fontsize=9)

    # cache arrows between chunks (encode)
    for x0, x1 in [(0.9, 1.0), (4.9, 5.0)]:
        ax.annotate("", xy=(x1 + 0.1, 3.35), xytext=(x0 - 0.1, 3.35),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.9,
                                    connectionstyle="arc3,rad=-0.3"))
    ax.text(4.5, 3.55, "rolling feature cache (last 2 activations)", ha="center",
            fontsize=7.5, color="gray")

    # decode: latent frames -> output frames
    for ci, frames in enumerate(chunks):
        mid = (frames[0] + frames[-1]) / 2 + 0.4
        arrow(ax, mid, 1.55, mid, 0.95)
        for f in frames:
            box(ax, f, 0.4, 0.8, 0.5, chunk_colors[ci], str(f), fontsize=8)

    ax.text(-0.45, 3.05, "input\nframes", ha="right", va="center", fontsize=8)
    ax.text(-0.45, 1.85, "latent\n(T'=3)", ha="right", va="center", fontsize=8)
    ax.text(-0.45, 0.65, "decoded\nframes", ha="right", va="center", fontsize=8)

    # where bleeding happens
    ax.add_patch(plt.Rectangle((1.0, 0.3), 3.8, 0.7, fill=False, ec="red", lw=1.2,
                               linestyle="--"))
    ax.text(2.9, -0.15, "1 latent frame -> 4 output frames:\nbleeding happens inside this group",
            ha="center", fontsize=7.5, color="red")
    ax.text(0.4, -0.15, "frame 0:\ncold start", ha="center", fontsize=7.5, color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_chunking.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- rate-quality
def fig_rate_quality():
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # made-up numbers, only the geometry matters
    ref_x = [12, 24, 48]
    ref_y = [33.5, 32.0, 30.2]
    ax.plot(ref_x, ref_y, "o--", color=C_GT, label="per-view references (graceful)")
    ax.annotate("Wan zero-shot", (12, 33.5), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.annotate("per-view ft, TC off", (24, 32.0), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.annotate("per-view ft, TC on", (48, 30.2), textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.plot([24], [31.4], "s", color="#2a7fbd", ms=9, label="fused, TC off")
    ax.annotate("fused, TC off", (24, 31.4), textcoords="offset points", xytext=(-72, -12), fontsize=8)

    ax.plot([96], [25.0], "X", color="#d43d3d", ms=11, label="fused + TC (joint)")
    ax.annotate("fused + TC:\nfalls off the curve?", (96, 25.0), textcoords="offset points",
                xytext=(-100, 18), fontsize=8, color="#d43d3d")

    for x, y, lbl in [(48, 27.5, "widen 32ch"), (24, 29.5, "widen 64ch")]:
        ax.plot([x], [y], "^", color="#7a4fb0", ms=9)
        ax.annotate(lbl + "?", (x, y), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color="#7a4fb0")
        ax.annotate("", xy=(x, y + 1.6), xytext=(x, y + 0.3),
                    arrowprops=dict(arrowstyle="->", color="#7a4fb0", lw=1.2, alpha=0.6))

    ax.set_xscale("log")
    ax.set_xticks([12, 24, 48, 96])
    ax.set_xticklabels(["12x", "24x", "48x", "96x"])
    ax.minorticks_off()
    ax.set_xlabel("compression ratio (pixels / latent floats)")
    ax.set_ylabel("val PSNR [dB]")
    ax.set_title("Rate-quality plane (headline figure)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25)

    ax.text(0.5, 0.5, "MOCK DATA", transform=ax.transAxes, fontsize=34, color="gray",
            alpha=0.18, ha="center", va="center", rotation=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_rate_quality.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- per-frame diagnostics
def fig_perframe():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    frames = np.arange(9)

    # left: per-frame PSNR, cold start at frame 0, dips after chunk boundaries
    ref = 32 + 0.15 * np.random.default_rng(0).standard_normal(9)
    joint = np.array([24.5, 27.0, 28.5, 28.8, 28.6, 26.8, 28.3, 28.7, 28.5])
    ax1.plot(frames, ref, "o-", color=C_GT, label="per-view reference")
    ax1.plot(frames, joint, "s-", color="#d43d3d", label="fused + TC")
    for b in [0.5, 4.5]:
        ax1.axvline(b, color="gray", ls=":", lw=1)
    ax1.text(0.15, 25.0, "cold start", fontsize=7.5, color="gray", rotation=90, va="bottom")
    ax1.text(4.65, 29.5, "chunk boundary", fontsize=7.5, color="gray", rotation=90, va="bottom")
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("PSNR [dB]")
    ax1.set_title("per-frame error profile")
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.25)

    # right: inter-frame |delta|, rec is too flat inside chunks
    pairs = np.arange(8)  # pair i = frames (i, i+1)
    gt = np.array([2.1, 2.4, 2.0, 2.6, 2.3, 2.5, 2.2, 2.4])
    rec = np.array([2.0, 0.9, 0.8, 1.0, 2.1, 0.9, 0.8, 1.0])
    ax2.plot(pairs, gt, "o-", color=C_GT, label="ground truth")
    ax2.plot(pairs, rec, "s-", color="#d43d3d", label="fused + TC recon")
    for ci, (lo, hi) in enumerate([(0.5, 3.5), (4.5, 7.5)]):
        ax2.axvspan(lo, hi, color="#f9d9a0" if ci == 0 else "#a0c8f9", alpha=0.35)
    ax2.text(2, 1.45, "within chunk:\nmotion undershoots\n= bleeding", ha="center",
             fontsize=7.5, color="#b06000")
    ax2.set_xlabel("frame pair (i, i+1)")
    ax2.set_ylabel(r"mean $|\Delta$frame$|$")
    ax2.set_title("inter-frame motion, GT vs recon")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.25)

    for ax in (ax1, ax2):
        ax.text(0.5, 0.5, "MOCK", transform=ax.transAxes, fontsize=30, color="gray",
                alpha=0.15, ha="center", va="center", rotation=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_perframe.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- data scale
def fig_datascale():
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = [0, 1, 2]
    labels = ["single\nsequence", "one\nperson", "all people\n(one expr.)"]

    bleed = [0.97, 0.80, 0.55]
    ghost = [0.01, 0.05, 0.12]  # xview similarity minus GT reference

    ax.plot(x, bleed, "o-", color="#b06000", label="bleed ratio (within chunk)")
    ax.axhline(1.0, color="#b06000", ls=":", lw=1)
    ax.text(2.02, 1.0, "faithful motion", fontsize=7, color="#b06000", va="bottom", ha="right")

    ax2 = ax.twinx()
    ax2.plot(x, ghost, "s-", color="#2a7fbd", label="ghosting (xview sim $-$ GT level)")
    ax2.axhline(0.0, color="#2a7fbd", ls=":", lw=1)
    ax2.set_ylabel("ghosting", color="#2a7fbd")
    ax2.tick_params(axis="y", colors="#2a7fbd")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("bleed ratio", color="#b06000")
    ax.tick_params(axis="y", colors="#b06000")
    ax.set_title("failure vs generalization pressure")

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center left")

    ax.text(0.5, 0.5, "MOCK", transform=ax.transAxes, fontsize=30, color="gray",
            alpha=0.15, ha="center", va="center", rotation=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_datascale.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_broad_overview()
    fig_overview()
    fig_fusion()
    fig_chunking()
    fig_rate_quality()
    fig_perframe()
    fig_datascale()
    print("wrote mock figures to", OUT)
