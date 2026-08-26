"""
Mock versions of the paper figures, for layout discussions only.

Everything with data in it uses made-up numbers (watermarked MOCK). The real
figures are generated from the rerun results (collect_results.py CSV + the
eval clip dumps) once the sweep is done.

Run:  python3 make_mock_figures.py   (writes PNGs next to this file)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))

# color code used in all schematics
C_FROZEN = "#aecbe8"   # blue: frozen pretrained
C_NEW = "#f5b942"      # orange: new modules, zero-init
C_LORA = "#8fd19e"     # green: LoRA adapters
C_LATENT = "#d9b3e6"   # purple: the latent
C_GT = "#444444"


def box(ax, x, y, w, h, color, text, fontsize=8.5, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=color, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color="black", **kw))


# ---------------------------------------------------------------- overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    # inputs
    box(ax, 0.2, 2.1, 0.9, 0.8, "white", "view 1\n[3,T,H,W]")
    box(ax, 0.2, 0.5, 0.9, 0.8, "white", "view 2\n[3,T,H,W]")

    # shared frozen encoder stem (drawn once per view for the data flow)
    box(ax, 1.5, 2.1, 1.5, 0.8, C_FROZEN, "Wan encoder stem\n(frozen, shared)")
    box(ax, 1.5, 0.5, 1.5, 0.8, C_FROZEN, "Wan encoder stem\n(frozen, shared)")
    arrow(ax, 1.1, 2.5, 1.5, 2.5)
    arrow(ax, 1.1, 0.9, 1.5, 0.9)

    # fusion
    box(ax, 3.4, 1.25, 1.3, 0.9, C_NEW, "cross-view\nattention\n(zero-init)")
    arrow(ax, 3.0, 2.5, 3.5, 2.0)
    arrow(ax, 3.0, 0.9, 3.5, 1.4)
    box(ax, 5.0, 1.25, 1.0, 0.9, C_NEW, "tree\nmerge")
    arrow(ax, 4.7, 1.7, 5.0, 1.7)

    # bottleneck middle/head (frozen + LoRA) -> latent
    box(ax, 6.3, 1.25, 1.0, 0.9, C_FROZEN, "middle/\nhead")
    ax.add_patch(FancyBboxPatch((6.35, 2.18), 0.9, 0.28, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(6.8, 2.32, "LoRA", ha="center", va="center", fontsize=7)
    arrow(ax, 6.0, 1.7, 6.3, 1.7)
    box(ax, 7.6, 1.35, 0.7, 0.7, C_LATENT, "z\n16ch")
    arrow(ax, 7.3, 1.7, 7.6, 1.7)

    # per-view decode
    box(ax, 8.6, 2.1, 1.2, 0.8, C_FROZEN, "Wan decoder\n(frozen)")
    box(ax, 8.6, 0.5, 1.2, 0.8, C_FROZEN, "Wan decoder\n(frozen)")
    ax.add_patch(FancyBboxPatch((8.65, 2.95), 0.65, 0.26, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(8.97, 3.08, "LoRA v=1", ha="center", va="center", fontsize=6.5)
    ax.add_patch(FancyBboxPatch((8.65, 0.13), 0.65, 0.26, boxstyle="round,pad=0.01",
                                fc=C_LORA, ec="black", lw=0.6))
    ax.text(8.97, 0.26, "LoRA v=2", ha="center", va="center", fontsize=6.5)
    arrow(ax, 8.3, 1.85, 8.6, 2.4)
    arrow(ax, 8.3, 1.55, 8.6, 1.0)
    ax.text(9.2, 1.7, "shared latent,\ndecoded V times", ha="center", va="center",
            fontsize=7, style="italic")

    # legend
    for i, (c, t) in enumerate([(C_FROZEN, "frozen pretrained"), (C_NEW, "new, zero-init"),
                                (C_LORA, "LoRA (rank 32)"), (C_LATENT, "fused latent")]):
        box(ax, 0.3 + i * 2.0, 3.05, 0.25, 0.22, c, "")
        ax.text(0.62 + i * 2.0, 3.16, t, va="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_overview.png"), dpi=180)
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
    ax.text(0.5, 0.5, "MOCK", transform=ax.transAxes, fontsize=30, color="gray",
            alpha=0.15, ha="center", va="center", rotation=20)

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mock_datascale.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_overview()
    fig_chunking()
    fig_rate_quality()
    fig_perframe()
    fig_datascale()
    print("wrote mock figures to", OUT)
