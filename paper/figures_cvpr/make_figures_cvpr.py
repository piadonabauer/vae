"""
CVPR 2-column figure suite.

Outputs land in this folder (figures_cvpr/). Qualitative panels are rebuilt by
cropping the previous paper PDFs / wandb eval grids (eval dumps were cleaned
from disk). Quantitative plots use the final protocol metrics.

Single-column target width ≈ 3.3 in; full-width ≈ 6.8 in.
"""
from __future__ import annotations

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(OUT), "figures")
OUT_ROOT = "/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs"

# CVPR-ish sizing
W1 = 3.30   # single column
W2 = 6.80   # full width (two columns)

C_PVTCT = "#ff7f0e"
C_FUSED = "#2ca02c"
C_JOINT = "#d62728"
C_E11A  = "#9467bd"
C_E11B  = "#8c564b"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("Saved", name)


def pdf_to_png(pdf_name, dpi=200):
    src = os.path.join(OLD, pdf_name)
    stem = os.path.join(OUT, "_src_" + pdf_name.replace(".pdf", ""))
    subprocess.check_call(["pdftoppm", "-png", "-r", str(dpi), src, stem])
    return Image.open(stem + "-1.png").convert("RGB")


def grid_crop(img, nrows, ncols, margin_l=0.14, margin_r=0.02,
              margin_t=0.08, margin_b=0.10, hpad=0.01, wpad=0.01):
    """Crop an evenly spaced nrows x ncols face grid from a rendered PDF."""
    W, H = img.size
    x0, x1 = int(W * margin_l), int(W * (1 - margin_r))
    y0, y1 = int(H * margin_t), int(H * (1 - margin_b))
    cw = (x1 - x0) / ncols
    ch = (y1 - y0) / nrows
    cells = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            L = int(x0 + c * cw + cw * wpad)
            R = int(x0 + (c + 1) * cw - cw * wpad)
            T = int(y0 + r * ch + ch * hpad)
            B = int(y0 + (r + 1) * ch - ch * hpad)
            # trim residual label strip at bottom of each cell if present
            cell = img.crop((L, T, R, B))
            row.append(cell)
        cells.append(row)
    return cells


def trim_cell(cell, top=0.10, bottom=0.16, left=0.02, right=0.02):
    """Drop baked-in title / PSNR bands that leak from neighboring cells."""
    w, h = cell.size
    return cell.crop((int(w * left), int(h * top),
                      int(w * (1 - right)), int(h * (1 - bottom))))


def trim_bottom_label(cell, frac=0.14):
    """Drop the PSNR text strip under a face cell."""
    return trim_cell(cell, top=0.02, bottom=frac)


def amplify_absdiff(a, b, gain=6.0):
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB"), dtype=np.float32)
    if aa.shape != bb.shape:
        bb = np.asarray(Image.fromarray(bb.astype(np.uint8)).resize(a.size), dtype=np.float32)
    d = np.clip(np.abs(aa - bb) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(d)


def mouth_crop(face, l=0.20, t=0.42, r=0.80, b=0.95):
    w, h = face.size
    return face.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def show_face(ax, img, title=None, ylabel=None, psnr=None, box_color=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if box_color is not None:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(box_color)
            sp.set_linewidth(1.6)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, rotation=90, labelpad=2)
    if psnr is not None:
        ax.set_xlabel(f"{psnr:.1f} dB", fontsize=6.5, labelpad=1)
    else:
        ax.set_xlabel(" ", fontsize=6.5, labelpad=1)


# ═══════════════════════════════════════════════════════════════════════════
# QUALITATIVE — rearrange + visibility aids
# ═══════════════════════════════════════════════════════════════════════════

def fig_qual_ghosting():
    """2 rows (views) × 4 cols (settings). Settings on top, PSNR below.
    Extra row: amplified |rec−GT| to make the collapse visible.
    Extra companion: mouth zoom (view 1) — differences are subtle at full face."""
    img = pdf_to_png("qual_ghosting.pdf", dpi=220)
    # original layout: 4 rows (GT, TCoff, TCon, 32ch) × 2 cols (v0, v1)
    cells = grid_crop(img, 4, 2, margin_l=0.18, margin_r=0.02,
                      margin_t=0.04, margin_b=0.06, hpad=0.02, wpad=0.03)
    # top trim kills previous-row PSNR that leaks into the next cell
    faces = [[trim_cell(cells[r][c], top=0.14 if r > 0 else 0.04, bottom=0.18)
              for c in range(2)] for r in range(4)]
    psnrs = {
        1: (33.0, 31.4),  # TC=off
        2: (28.8, 28.3),  # TC=on
        3: (30.2, 29.3),  # 32-ch
    }
    titles = ["Ground truth", "Fused, TC=off", "Fused, TC=on", "Fused, 32-ch"]

    fig, axes = plt.subplots(3, 4, figsize=(W1, 2.55))
    fig.subplots_adjust(wspace=0.04, hspace=0.28, left=0.10, right=0.99, top=0.90, bottom=0.06)

    for c, title in enumerate(titles):
        show_face(axes[0, c], faces[c][0], title=title,
                  ylabel="View 0" if c == 0 else None,
                  psnr=None if c == 0 else psnrs[c][0])
        show_face(axes[1, c], faces[c][1],
                  ylabel="View 1" if c == 0 else None,
                  psnr=None if c == 0 else psnrs[c][1])
        if c == 0:
            axes[2, c].axis("off")
            axes[2, c].text(0.5, 0.5, "|rec−GT|×6\n(view 0)", ha="center", va="center",
                            fontsize=6.5, transform=axes[2, c].transAxes, color="#555")
        else:
            diff = amplify_absdiff(faces[0][0], faces[c][0], gain=6.0)
            show_face(axes[2, c], diff, ylabel="Error" if c == 1 else None)

    save(fig, "qual_ghosting.pdf")

    # mouth zoom on view 1 — makes TC blur / capacity recovery readable
    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.05))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.72, bottom=0.18)
    for c, title in enumerate(titles):
        show_face(axes[c], mouth_crop(faces[c][1]), title=title,
                  psnr=None if c == 0 else psnrs[c][1])
    fig.suptitle("Mouth crop · view 1", fontsize=8, y=1.05)
    save(fig, "qual_ghosting_mouthzoom.pdf")


def fig_qual_bleeding():
    """Methods as columns, frames as rows (user suggestion). Compact for 1-col.
    Yes — frames on y / methods on x reads better for temporal collapse."""
    img = pdf_to_png("qual_bleeding.pdf", dpi=220)
    # original: 4 rows (GT,16,32,64) × 4 cols (frames 1-4)
    cells = grid_crop(img, 4, 4, margin_l=0.16, margin_r=0.02,
                      margin_t=0.10, margin_b=0.06, hpad=0.02, wpad=0.015)
    faces = [[trim_cell(cells[r][c], top=0.12 if r > 0 else 0.06, bottom=0.18)
              for c in range(4)] for r in range(4)]
    psnrs = [
        [None, None, None, None],
        [32.7, 30.3, 29.7, 27.4],
        [34.9, 32.3, 32.4, 29.0],
        [33.0, 31.4, 32.2, 28.7],
    ]
    col_titles = ["GT", "16-ch", "32-ch", "64-ch"]

    fig, axes = plt.subplots(4, 4, figsize=(W1, 3.15))
    fig.subplots_adjust(wspace=0.03, hspace=0.18, left=0.11, right=0.99, top=0.90, bottom=0.04)
    for f in range(4):
        for m in range(4):
            show_face(axes[f, m], faces[m][f],
                      title=col_titles[m] if f == 0 else None,
                      ylabel=f"f{f + 1}" if m == 0 else None,
                      psnr=psnrs[m][f])
    fig.suptitle("Temporal bleeding (chunk 1)", fontsize=8, y=0.97)
    save(fig, "qual_bleeding.pdf")

    # mouth zoom on frame 4 (most motion)
    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.05))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.72, bottom=0.18)
    for m in range(4):
        show_face(axes[m], mouth_crop(faces[m][3], l=0.22, t=0.45, r=0.78, b=0.92),
                  title=col_titles[m], psnr=psnrs[m][3])
    fig.suptitle("Mouth crop · frame 4", fontsize=8, y=1.05)
    save(fig, "qual_bleeding_mouthzoom.pdf")

    # error maps across frames for 16-ch (worst) — makes bleed readable
    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.05))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.72, bottom=0.08)
    for f in range(4):
        diff = amplify_absdiff(faces[0][f], faces[1][f], gain=5.0)
        show_face(axes[f], diff, title=f"f{f + 1}")
    fig.suptitle("|rec−GT|×5 · 16-ch across chunk 1", fontsize=8, y=1.05)
    save(fig, "qual_bleeding_error.pdf")


def fig_qual_bleeding_9frame():
    """Full T=9 strip from wandb E1d grid, with chunk rectangles on f0 / f1–4 / f5–8."""
    path = os.path.join(OUT, "_wandb_E1d.png")
    if not os.path.exists(path):
        print("skip qual_bleeding_9frame: missing wandb png")
        return
    img = Image.open(path).convert("RGB")
    cells = grid_crop(img, 4, 9, margin_l=0.10, margin_r=0.01,
                      margin_t=0.08, margin_b=0.14, hpad=0.015, wpad=0.008)
    faces = [[trim_cell(cells[r][c], top=0.04, bottom=0.18) for c in range(9)]
             for r in range(4)]

    fig, axes = plt.subplots(2, 9, figsize=(W2, 1.55))
    fig.subplots_adjust(wspace=0.03, hspace=0.08, left=0.06, right=0.99, top=0.78, bottom=0.12)
    chunk_colors = ["#444444"] + ["#4C78A8"] * 4 + ["#E45756"] * 4
    for c in range(9):
        show_face(axes[0, c], faces[0][c],
                  title=f"f{c}",
                  ylabel="GT" if c == 0 else None,
                  box_color=chunk_colors[c])
        show_face(axes[1, c], faces[2][c],
                  ylabel="Rec" if c == 0 else None,
                  box_color=chunk_colors[c])

    fig.text(0.11, 0.92, "f₀", color="#444", fontsize=7, ha="center")
    fig.text(0.33, 0.92, "chunk 1  (f₁–f₄)", color="#4C78A8", fontsize=7, ha="center")
    fig.text(0.68, 0.92, "chunk 2  (f₅–f₈)", color="#E45756", fontsize=7, ha="center")
    fig.suptitle("Joint compression · 9-frame strip (fused, TC=on, 16-ch)", fontsize=8.5, y=1.02)
    save(fig, "qual_temporal_9frame.pdf")


def fig_qual_capacity():
    """Compact latent-width qualitative: faces + mouth zoom + error in one panel."""
    img = pdf_to_png("qual_capacity.pdf", dpi=220)
    cells = grid_crop(img, 2, 4, margin_l=0.12, margin_r=0.02,
                      margin_t=0.14, margin_b=0.08, hpad=0.03, wpad=0.015)
    faces = [[trim_cell(cells[r][c], top=0.16, bottom=0.18) for c in range(4)]
             for r in range(2)]
    titles = ["GT", "16-ch", "32-ch", "64-ch"]
    psnrs = [[None, 27.4, 29.0, 28.7],
             [None, 27.1, 28.1, 28.3]]

    # profile faces sit left in the crop — mouth is lower-left, not center
    def _mouth(f):
        return mouth_crop(f, l=0.02, t=0.48, r=0.58, b=0.98)

    fig, axes = plt.subplots(3, 4, figsize=(W1, 2.45))
    fig.subplots_adjust(wspace=0.03, hspace=0.22, left=0.11, right=0.99, top=0.90, bottom=0.06)
    for c, title in enumerate(titles):
        show_face(axes[0, c], faces[0][c], title=title,
                  ylabel="Motion" if c == 0 else None, psnr=psnrs[0][c])
        show_face(axes[1, c], _mouth(faces[0][c]),
                  ylabel="Mouth" if c == 0 else None)
        if c == 0:
            axes[2, c].axis("off")
            axes[2, c].text(0.5, 0.5, "|rec−GT|×5", ha="center", va="center",
                            fontsize=6.5, color="#555", transform=axes[2, c].transAxes)
        else:
            show_face(axes[2, c], amplify_absdiff(faces[0][0], faces[0][c], gain=5.0),
                      ylabel="Error" if c == 1 else None)
    fig.suptitle("Latent width · high-motion frame", fontsize=8, y=0.97)
    save(fig, "qual_capacity.pdf")

    fig, axes = plt.subplots(2, 4, figsize=(W1, 1.85))
    fig.subplots_adjust(wspace=0.03, hspace=0.22, left=0.11, right=0.99, top=0.88, bottom=0.08)
    for c, title in enumerate(titles):
        show_face(axes[0, c], faces[1][c], title=title,
                  ylabel="Identity" if c == 0 else None, psnr=psnrs[1][c])
        if c == 0:
            axes[1, c].axis("off")
            axes[1, c].text(0.5, 0.5, "|rec−GT|×5", ha="center", va="center",
                            fontsize=6.5, color="#555", transform=axes[1, c].transAxes)
        else:
            show_face(axes[1, c], amplify_absdiff(faces[1][0], faces[1][c], gain=5.0),
                      ylabel="Error" if c == 1 else None)
    fig.suptitle("Latent width · identity", fontsize=8, y=0.98)
    save(fig, "qual_capacity_identity.pdf")

    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.05))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.72, bottom=0.18)
    for c in range(4):
        show_face(axes[c], _mouth(faces[0][c]), title=titles[c], psnr=psnrs[0][c])
    fig.suptitle("Mouth crop · high motion", fontsize=8, y=1.05)
    save(fig, "qual_capacity_mouthzoom.pdf")

    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.05))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.72, bottom=0.12)
    axes[0].axis("off")
    axes[0].text(0.5, 0.5, "|rec−GT|×5", ha="center", va="center", fontsize=7, color="#555")
    for c in range(1, 4):
        show_face(axes[c], amplify_absdiff(faces[0][0], faces[0][c], gain=5.0), title=titles[c])
    fig.suptitle("Where capacity fails (high motion)", fontsize=8, y=1.05)
    save(fig, "qual_capacity_error.pdf")


def fig_latent_width():
    """Compact; zoomed y-axis; label BOTH jumps (+2.40 and −0.11)."""
    fig, ax = plt.subplots(figsize=(W1 * 0.92, 1.45))
    xs = np.arange(3)
    psnrs = [25.31, 27.71, 27.60]
    cols = [C_JOINT, C_E11A, C_E11B]

    ax.plot(xs, psnrs, "-", color="#bbb", lw=1.0, zorder=2)
    for x, p, col in zip(xs, psnrs, cols):
        ax.scatter([x], [p], color=col, s=42, zorder=5)
        ax.text(x, p + 0.22, f"{p:.2f}", ha="center", fontsize=6.5)

    ax.annotate("", xy=(1, 27.71), xytext=(0, 25.31),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(0.42, 26.15, "+2.40", fontsize=7, ha="center", color="#222", fontweight="bold")

    ax.annotate("", xy=(2, 27.60), xytext=(1, 27.71),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0,
                                connectionstyle="arc3,rad=-0.45"))
    ax.text(1.55, 28.05, "−0.11", fontsize=7, ha="center", color="#666")

    # per-view reference as a thin note (don't stretch the axis)
    ax.text(0.98, 0.08, "per-view TC=on: 31.50 dB", transform=ax.transAxes,
            fontsize=5.5, color=C_PVTCT, ha="right", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels(["16 (Wan)", "32", "64"], fontsize=7)
    ax.set_ylabel("PSNR (dB)", fontsize=7)
    ax.set_title("Latent width (fused, TC=on)", fontsize=7.5, pad=3)
    ax.set_ylim(24.7, 28.5)
    ax.set_xlim(-0.35, 2.35)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18)
    save(fig, "latent_width.pdf")


def fig_datascale():
    """Bleed + ghosting + PSNR vs training-set size (final values).
    E8b one-person run incomplete — omit middle point rather than invent it."""
    points = [
        ("Single\nsequence", 0.983, 0.847, 0.845, 34.56),
        ("All people\n(one expr.)", 0.919, 0.920, 0.918, 25.31),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(W1, 1.55))
    fig.subplots_adjust(wspace=0.55, left=0.12, right=0.98, top=0.78, bottom=0.28)
    xs = np.arange(len(points))

    ax = axes[0]
    bleeds = [p[1] for p in points]
    ax.plot(xs, bleeds, "o-", color=C_JOINT, lw=1.5, ms=6)
    for x, b in zip(xs, bleeds):
        ax.text(x, b + 0.010, f"{b:.3f}", ha="center", fontsize=6)
    ax.axhline(1.0, color="#999", lw=0.7, ls="--")
    ax.set_ylabel("Bleed-W ↑", fontsize=7)
    ax.set_title("Temporal fidelity", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points], fontsize=5.5)
    ax.set_ylim(0.88, 1.03)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    ax = axes[1]
    rec = [p[2] for p in points]
    gt  = [p[3] for p in points]
    ax.plot(xs, rec, "o-", color=C_JOINT, lw=1.5, ms=6, label="rec")
    ax.plot(xs, gt,  "s--", color="#888", lw=1.2, ms=5, label="GT")
    ax.set_ylabel("XView cos sim", fontsize=7)
    ax.set_title("Ghosting", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points], fontsize=5.5)
    ax.legend(fontsize=5.5, loc="best")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    ax = axes[2]
    psnrs = [p[4] for p in points]
    ax.plot(xs, psnrs, "o-", color=C_JOINT, lw=1.5, ms=6)
    for x, p in zip(xs, psnrs):
        ax.text(x, p + 0.6, f"{p:.2f}", ha="center", fontsize=6)
    ax.set_ylabel("PSNR (dB)", fontsize=7)
    ax.set_title("Quality", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points], fontsize=5.5)
    ax.set_ylim(23, 37)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("Failure vs generalization pressure", fontsize=8, y=0.98)
    save(fig, "datascale.pdf")


def fig_perframe():
    """Per-frame PSNR with chunk rectangles on f0 / f1–4 / f5–8. Dual: V=2 and V=4."""
    series_v2 = {
        "Fused TC=off": (C_FUSED, "-", [33.42, 33.81, 33.38, 33.20, 33.19, 33.10, 33.35, 33.27, 32.81]),
        "Fused TC=on":  (C_JOINT, "-", [32.48, 30.59, 28.30, 29.56, 28.42, 25.56, 26.60, 28.77, 28.07]),
        "+ diff-loss":  ("#17becf", "--", [32.81, 31.30, 29.57, 30.21, 28.93, 27.35, 28.23, 29.82, 28.93]),
        "+ 32-ch":      (C_E11A, ":", [34.19, 32.25, 29.92, 31.77, 30.04, 28.12, 28.85, 31.57, 30.00]),
        "+ both":       ("#7a3e9d", "-.", [35.44, 33.76, 31.33, 32.60, 31.03, 29.84, 30.28, 32.53, 31.24]),
    }
    series_v4 = {
        "V=4, TC=off": (C_FUSED, "-", [30.62, 30.63, 30.04, 29.77, 29.86, 29.94, 29.91, 30.10, 30.08]),
        "V=4, TC=on":  (C_JOINT, "-", [29.45, 28.03, 26.37, 26.12, 25.38, 23.89, 25.03, 25.89, 25.35]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.15), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.99, top=0.86, bottom=0.18)

    def draw(ax, series, title, ymin, ymax):
        xs = np.arange(9)
        ax.axvspan(-0.5, 0.5, color="#ddd", alpha=0.55, lw=0)
        ax.axvspan(0.5, 4.5, color="#4C78A8", alpha=0.10, lw=0)
        ax.axvspan(4.5, 8.5, color="#E45756", alpha=0.10, lw=0)
        for spine_x in (0.5, 4.5):
            ax.axvline(spine_x, color="#999", lw=0.6, ls="--")
        for lab, (col, ls, data) in series.items():
            ax.plot(xs, data, color=col, ls=ls, lw=1.4, marker="o", ms=3.0,
                    markeredgewidth=0, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"f{i}" for i in range(9)])
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, loc="lower left", framealpha=0.9)

    draw(axes[0], series_v2, "V=2  (main setting)", 24.5, 36.5)
    draw(axes[1], series_v4, "V=4", 22.5, 32.0)
    axes[0].set_ylabel("PSNR (dB)")
    fig.text(0.22, 0.02, "f₀", color="#555", fontsize=7, ha="center")
    fig.text(0.38, 0.02, "chunk 1 (f₁–f₄)", color="#4C78A8", fontsize=7, ha="center")
    fig.text(0.72, 0.02, "chunk 2 (f₅–f₈)", color="#E45756", fontsize=7, ha="center")
    save(fig, "perframe_psnr.pdf")


def fig_resolution():
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 1.55))
    res = [128, 256, 512]
    psnrs = [25.31, 27.78, 26.64]
    lpips = [0.082, 0.102, 0.170]
    ax.plot(res, psnrs, "o-", color=C_JOINT, lw=1.5, ms=6)
    for r, p in zip(res, psnrs):
        ax.annotate(f"{p:.2f}", (r, p), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_xticks(res)
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Resolution scaling (fused, TC=on)", fontsize=8)
    ax.set_ylim(24.5, 29.0)
    ax.grid(True, alpha=0.25, lw=0.5)
    save(fig, "resolution_scaling.pdf")

    fig, ax = plt.subplots(figsize=(W1 * 0.85, 1.45))
    ax.plot(res, lpips, "s-", color="#555", lw=1.5, ms=6)
    for r, p in zip(res, lpips):
        ax.annotate(f"{p:.3f}", (r, p), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_xticks(res)
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("LPIPS ↓")
    ax.set_title("Perceptual error vs resolution", fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.5)
    save(fig, "resolution_lpips.pdf")


def fig_view_count():
    views = [2, 4, 8]
    psnr_f = [31.21, 28.00, 25.15]
    psnr_t = [25.31, 23.23, 20.25]
    bleed_f = [0.974, 0.954, 0.927]
    bleed_t = [0.919, 0.837, 0.755]

    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.55))
    fig.subplots_adjust(wspace=0.40, left=0.14, right=0.98, top=0.84, bottom=0.22)

    ax = axes[0]
    ax.plot(views, psnr_f, "o-", color=C_FUSED, lw=1.4, ms=5, label="TC=off")
    ax.plot(views, psnr_t, "s-", color=C_JOINT, lw=1.4, ms=5, label="TC=on")
    ax.set_xticks(views)
    ax.set_xlabel("# views V")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Quality", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.plot(views, bleed_f, "o-", color=C_FUSED, lw=1.4, ms=5, label="TC=off")
    ax.plot(views, bleed_t, "s-", color=C_JOINT, lw=1.4, ms=5, label="TC=on")
    ax.axhline(1.0, color="#999", lw=0.7, ls="--")
    ax.set_xticks(views)
    ax.set_xlabel("# views V")
    ax.set_ylabel("Bleed-W ↑")
    ax.set_title("Bleeding", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25, lw=0.5)
    save(fig, "view_count.pdf")


def fig_interventions():
    items = [
        ("Baseline", 25.31, C_JOINT),
        ("+ Reflect pad", 25.88, "#aec7e8"),
        ("+ Learned cache", 25.96, "#aec7e8"),
        ("+ Subframe PE", 25.89, "#aec7e8"),
        ("+ Diff+cache", 26.93, "#aec7e8"),
        ("+ Diff-loss", 27.14, "#aec7e8"),
        ("+ Diff+32ch", 29.56, C_E11A),
    ]
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    ys = np.arange(len(items))
    vals = [x[1] for x in items]
    cols = [x[2] for x in items]
    ax.barh(ys, vals, color=cols, edgecolor="white", height=0.62)
    ax.axvline(25.31, color=C_JOINT, lw=0.9, ls="--", alpha=0.5)
    for y, (lab, psnr, _) in enumerate(items):
        d = psnr - 25.31
        s = f"{psnr:.2f}" + (f" ({d:+.2f})" if d else "")
        ax.text(psnr + 0.05, y, s, va="center", fontsize=6.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([x[0] for x in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(24.2, 31.0)
    ax.set_xlabel("PSNR (dB)")
    ax.set_title("Temporal interventions", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    save(fig, "interventions.pdf")


def fig_rate_quality():
    fig, ax = plt.subplots(figsize=(W1, 1.85))
    x = np.arange(2)
    w = 0.34
    perview = [34.01, 31.50]
    fused = [31.21, 25.31]
    b1 = ax.bar(x - w / 2, perview, w, color="#1f77b4", label="Per-view", edgecolor="white")
    b2 = ax.bar(x + w / 2, fused, w, color=C_JOINT, label="Fused", edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{bar.get_height():.2f}", ha="center", fontsize=6.5)
    ax.plot([1 + w / 2], [28.70], marker="_", markersize=14, color="#555", mew=2)
    ax.annotate("additive\n28.7", xy=(1 + w / 2, 28.70), xytext=(1.55, 30.0),
                fontsize=6, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(["TC=off", "TC=on"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(22.5, 36.5)
    ax.set_title("Joint compression is super-additive", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    save(fig, "rate_quality.pdf")


def fig_chunking_schematic():
    """Simple timeline schematic: f0 | chunk1 | chunk2 with bleeding callout."""
    fig, ax = plt.subplots(figsize=(W1, 0.95))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    boxes = [
        (0, 1, "#dddddd", "f₀"),
        (1, 4, "#4C78A8", "chunk 1 → 1 latent"),
        (5, 4, "#E45756", "chunk 2 → 1 latent"),
    ]
    for x, w, col, lab in boxes:
        ax.add_patch(FancyBboxPatch((x + 0.08, 0.28), w - 0.16, 0.42,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=col, edgecolor="none", alpha=0.40))
        ax.text(x + w / 2, 0.49, lab, ha="center", va="center", fontsize=6.5, color="#222")
    for i in range(9):
        ax.text(i + 0.5, 0.12, f"f{i}", ha="center", fontsize=6, color="#444")
    ax.text(4.5, 0.95, "bleeding inside each 4-frame decode",
            ha="center", fontsize=6, color="#555")
    ax.set_title("Wan causal chunking (T=9 → T′=3)", fontsize=7.5, pad=1)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02)
    save(fig, "chunking_schematic.pdf")


def fig_resolution_qual_hint():
    """If wandb 256px grid exists, show a compact GT/Rec pair note — optional."""
    path = os.path.join(OUT, "_wandb_E9a.png")
    if not os.path.exists(path):
        return
    img = Image.open(path).convert("RGB")
    # 256px grid is larger; crop first sample roughly
    # Keep as a small reference strip
    fig, ax = plt.subplots(figsize=(W1, 1.4))
    ax.imshow(img.resize((img.width // 2, img.height // 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("256px eval grid (wandb) · fused TC=on", fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "resolution_wandb_grid.pdf")


if __name__ == "__main__":
    fig_qual_ghosting()
    fig_qual_bleeding()
    fig_qual_bleeding_9frame()
    fig_qual_capacity()
    fig_latent_width()
    fig_perframe()
    fig_resolution()
    fig_datascale()
    fig_view_count()
    fig_interventions()
    fig_rate_quality()
    fig_chunking_schematic()
    fig_resolution_qual_hint()
    print("All CVPR figures written to", OUT)
