"""
Generate all paper figures that don't require loading .pt dump files.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Colour palette ─────────────────────────────────────────────────────────
C_PVTCF = "#1f77b4"   # per-view TC=off (blue)
C_PVTCT = "#ff7f0e"   # per-view TC=on  (orange)
C_FUSED = "#2ca02c"   # fused TC=off    (green)
C_JOINT = "#d62728"   # fused TC=on     (red)
C_E11A  = "#9467bd"   # widen 32ch      (purple)
C_E11B  = "#8c564b"   # widen 64ch      (brown)
C_CEIL  = "#7f7f7f"   # ceiling         (grey)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.92,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 1 — Rate–Quality scatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_rate_quality():
    """2x2 grouped bars: the super-additivity story, without a crowded scatter."""
    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    x = np.arange(2)
    w = 0.34
    perview = [34.01, 31.50]  # TC=off, TC=on
    fused   = [31.21, 25.31]

    b1 = ax.bar(x - w/2, perview, w, color=C_PVTCF, label="Per-view", edgecolor="white")
    b2 = ax.bar(x + w/2, fused,   w, color=C_JOINT, label="Fused",    edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)

    # Additive expectation for fused+TC: 34.01 - 2.51 - 2.80 = 28.70
    ax.plot([1 + w/2], [28.70], marker="_", markersize=16, color="#555", mew=2, zorder=5)
    ax.annotate("additive\nexpect. 28.7", xy=(1 + w/2, 28.70), xytext=(1.55, 29.6),
                fontsize=7, color="#555", ha="left",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(["TC=off", "TC=on"])
    ax.set_ylim(22.5, 37.0)
    ax.set_title("Joint compression is super-additive")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)

    fig.savefig(os.path.join(OUT, "rate_quality.pdf"))
    plt.close(fig)
    print("Saved rate_quality.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 2 — Per-frame PSNR profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_perframe():
    # Show only the most informative subset; cleaner labels
    perframe = {
        "E1c": {"label": "Fused, TC=off",           "col": C_FUSED, "ls": "-",  "lw": 1.8, "ms": 4,
                "data": [33.39, 33.76, 33.36, 33.17, 33.14, 33.11, 33.35, 33.28, 32.79]},
        "E1d": {"label": "Fused, TC=on (baseline)", "col": C_JOINT, "ls": "-",  "lw": 1.8, "ms": 4,
                "data": [32.48, 30.59, 28.30, 29.56, 28.42, 25.56, 26.60, 28.77, 28.07]},
        "E4h": {"label": "+ temporal diff-loss",    "col": "#17becf","ls": "--", "lw": 1.6, "ms": 4,
                "data": [32.81, 31.30, 29.57, 30.21, 28.93, 27.35, 28.23, 29.82, 28.93]},
        "E11a":{"label": "+ 32-channel latent",     "col": C_E11A,  "ls": ":",  "lw": 2.0, "ms": 4,
                "data": [34.19, 32.25, 29.92, 31.77, 30.04, 28.12, 28.85, 31.57, 30.00]},
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xs = np.arange(9)

    # Chunk shading
    ax.axvspan(-0.5, 0.5, alpha=0.06, color="black")
    ax.axvspan(0.5,  4.5, alpha=0.05, color="steelblue")
    ax.axvspan(4.5,  8.5, alpha=0.05, color="tomato")

    # Chunk labels (below the lines)
    ax.text(0,    23.8, "f₀",            fontsize=7, ha="center", color="#555")
    ax.text(2.5,  23.8, "chunk 1 (f₁–f₄)", fontsize=7, ha="center", color="steelblue", alpha=0.8)
    ax.text(6.5,  23.8, "chunk 2 (f₅–f₈)", fontsize=7, ha="center", color="tomato",    alpha=0.8)
    ax.axvline(0.5, color="#aaa", lw=0.7, ls="--")
    ax.axvline(4.5, color="#aaa", lw=0.7, ls="--")

    for cfg in perframe.values():
        ax.plot(xs, cfg["data"], color=cfg["col"], ls=cfg["ls"], lw=cfg["lw"],
                marker="o", ms=cfg["ms"], markeredgewidth=0, label=cfg["label"])

    ax.set_xlabel("Frame index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-frame PSNR Profile")
    ax.set_xticks(xs)
    ax.set_ylim(23.0, 35.5)
    ax.grid(True, alpha=0.2, lw=0.5)

    # Legend outside below
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True)

    fig.savefig(os.path.join(OUT, "perframe_psnr.pdf"))
    plt.close(fig)
    print("Saved perframe_psnr.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 3 — Temporal interventions (PSNR only; bleed-W as secondary)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_interventions():
    # Side channel dropped: it is a pixel-to-decoder skip and is not usable
    # under unconditional generation (no input video at sample time).
    items = [
        ("Baseline",                   25.31, "#d62728"),
        ("+ Reflection pad",           25.88, "#aec7e8"),
        ("+ Learned cache update",     25.96, "#aec7e8"),
        ("+ Sub-frame pos embedding",  25.89, "#aec7e8"),
        ("+ Diff-loss + cache",        26.93, "#aec7e8"),
        ("+ Temporal diff-loss",       27.14, "#aec7e8"),
        ("+ Diff-loss + 32-ch",        29.56, C_E11A),
    ]
    labels = [x[0] for x in items]
    psnrs  = [x[1] for x in items]
    cols   = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    ys = np.arange(len(labels))
    bars = ax.barh(ys, psnrs, color=cols, edgecolor="white", height=0.60)
    ax.axvline(25.31, color="#d62728", lw=1.0, ls="--", alpha=0.55)

    for bar, psnr in zip(bars, psnrs):
        delta = psnr - 25.31
        suffix = f"  ({'+' if delta>=0 else ''}{delta:.2f})" if delta != 0 else ""
        ax.text(psnr + 0.04, bar.get_y() + bar.get_height()/2,
                f"{psnr:.2f}{suffix}", va="center", fontsize=7.5)

    ax.set_xlabel("PSNR (dB)")
    ax.set_xlim(24.2, 31.2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Temporal Interventions")
    ax.grid(True, axis="x", alpha=0.2, lw=0.5)

    fig.savefig(os.path.join(OUT, "interventions.pdf"))
    plt.close(fig)
    print("Saved interventions.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 4 — Multi-view degradation (2 panels only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_view_count():
    views   = [2, 4, 8]
    psnr_f  = [31.21, 28.00, 25.15]
    psnr_t  = [25.31, 23.23, 20.25]
    bleed_f = [0.974, 0.954, 0.927]
    bleed_t = [0.919, 0.837, 0.755]

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    fig.subplots_adjust(wspace=0.40)

    # PSNR vs V
    ax = axes[0]
    ax.plot(views, psnr_f, "o-", color=C_FUSED, lw=1.8, ms=7, label="TC=off")
    ax.plot(views, psnr_t, "s-", color=C_JOINT, lw=1.8, ms=7, label="TC=on")
    ax.set_xlabel("Number of views (V)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(views)
    ax.set_title("Reconstruction quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)

    # Bleed ratio vs V
    ax = axes[1]
    ax.plot(views, bleed_f, "o-", color=C_FUSED, lw=1.8, ms=7, label="TC=off")
    ax.plot(views, bleed_t, "s-", color=C_JOINT, lw=1.8, ms=7, label="TC=on")
    ax.axhline(1.0, color="#999", lw=0.8, ls="--", alpha=0.5)
    ax.text(8.15, 1.0, "ideal", fontsize=7, va="center", color="#999")
    ax.set_xlabel("Number of views (V)")
    ax.set_ylabel("Bleed ratio (within-chunk) ↑")
    ax.set_xticks(views)
    ax.set_title("Temporal bleeding")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle("Multi-view Scaling", fontsize=11, y=1.02)
    fig.savefig(os.path.join(OUT, "view_count.pdf"))
    plt.close(fig)
    print("Saved view_count.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 5 — Fusion mechanism ablation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_fusion():
    # Default fusion is multi-head self-attention over all view tokens + tree merge.
    names  = ["Self-attention\n(tree merge)",
               "Channel\nConv3D",
               "Factorized\n4D conv"]
    psnrs  = [31.21, 30.78, 27.86]
    cols   = ["#1f77b4", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(names, psnrs, color=cols, width=0.55, edgecolor="white")

    for bar, p in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.08,
                f"{p:.2f}", ha="center", fontsize=8)

    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Fusion Mechanism Ablation")
    ax.set_ylim(25.5, 33.5)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)

    fig.savefig(os.path.join(OUT, "fusion_ablation.pdf"))
    plt.close(fig)
    print("Saved fusion_ablation.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 6 — Latent width ablation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_latent_width():
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.axhline(31.50, color=C_PVTCT, lw=1.3, ls="--", alpha=0.75)
    ax.text(0.98, 31.50, "Per-view, TC=on  (31.5 dB)",
            transform=ax.get_yaxis_transform(), fontsize=7.5,
            color=C_PVTCT, va="bottom", ha="right")

    # Equal spacing: 16→32 and 32→64 are both one doubling.
    xs     = np.arange(3)
    psnrs  = [25.31, 27.71, 27.60]
    cols   = [C_JOINT, C_E11A, C_E11B]
    labels = ["16 ch", "32 ch", "64 ch"]

    ax.plot(xs, psnrs, "o-", color="#aaa", lw=1.0, zorder=2, ms=0)
    for x, p, col in zip(xs, psnrs, cols):
        ax.scatter(x, p, color=col, s=70, zorder=5)

    ax.annotate("", xy=(1, 27.71), xytext=(0, 25.31),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.28, 26.35, "+2.40 dB", fontsize=8, ha="center", color="#333")

    ax.set_xlabel("Latent channels")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Latent Width Ablation  (fused, TC=on)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["16\n(Wan default)", "32", "64"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(24.0, 32.5)
    ax.grid(True, alpha=0.2, lw=0.5)

    fig.savefig(os.path.join(OUT, "latent_width.pdf"))
    plt.close(fig)
    print("Saved latent_width.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 7 — Resolution scaling (PSNR only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_resolution():
    resolutions = [128, 256, 512]
    psnrs       = [25.31, 27.78, 26.64]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    ax.plot(resolutions, psnrs, "o-", color=C_JOINT, lw=1.8, ms=7, zorder=5)
    for r, p in zip(resolutions, psnrs):
        ax.annotate(f"{p:.2f} dB", (r, p), xytext=(0, 9), textcoords="offset points",
                    fontsize=8.5, ha="center")

    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Resolution Scaling  (fused, TC=on)")
    ax.set_xticks(resolutions)
    ax.set_xlim(80, 590)
    ax.set_ylim(23.5, 30.0)
    ax.grid(True, alpha=0.2, lw=0.5)

    fig.savefig(os.path.join(OUT, "resolution_scaling.pdf"))
    plt.close(fig)
    print("Saved resolution_scaling.pdf")


# ── Run all ──────────────────────────────────────────────────────────────────
fig_rate_quality()
fig_perframe()
fig_interventions()
fig_view_count()
fig_fusion()
fig_latent_width()
fig_resolution()
print("All figures done.")
