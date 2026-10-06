"""
CVPR figure suite, v2.

Qualitative panels are rebuilt from the face tiles EMBEDDED in the original
matplotlib PDFs (paper/figures/qual_*.pdf) via pymupdf, so every face is whole
and at native resolution -- no pixel-margin cropping, no leaked labels.
The 9-frame strip comes from the wandb eval grid (_wandb_E1d.png) with
automatic gutter detection.

Sizing: figures are produced at their FINAL print size, so fonts are exact:
  single column = 3.25 in  (\columnwidth), full width = 6.875 in (\textwidth).
Font: STIXGeneral (Times-compatible, matches the CVPR body font).

No |rec-GT| error maps, no mouth-zoom crops, no in-figure titles
(captions belong to LaTeX).
"""
from __future__ import annotations

import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import pymupdf
except ImportError:  # optional: only needed for PDF-tile qualitative extractors
    pymupdf = None
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(OUT), "figures")

W1 = 3.25    # \columnwidth in inches
W2 = 6.875   # \textwidth in inches

C_PVTCT = "#ff7f0e"   # per-view reference
C_FUSED = "#2ca02c"   # fused, TC off
C_JOINT = "#d62728"   # fused, TC on (baseline)
C_32CH  = "#9467bd"
C_64CH  = "#8c564b"
C_CHUNK1 = "#4C78A8"
C_CHUNK2 = "#E45756"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight" if tight else None, pad_inches=0.01)
    plt.close(fig)
    print("Saved", name)


# ---------------------------------------------------------------------------
# tile sources
# ---------------------------------------------------------------------------

def extract_pdf_tiles(pdf_name, nrows, ncols):
    """Pull the face images embedded in an old matplotlib PDF, in grid order."""
    doc = pymupdf.open(os.path.join(OLD, pdf_name))
    page = doc[0]
    tiles = []
    for info in page.get_image_info(xrefs=True):
        raw = doc.extract_image(info["xref"])
        img = Image.open(io.BytesIO(raw["image"])).convert("RGB")
        x0, y0, _, _ = info["bbox"]
        tiles.append((y0, x0, img))
    assert len(tiles) == nrows * ncols, f"{pdf_name}: found {len(tiles)} tiles"
    tiles.sort(key=lambda t: (round(t[0]), t[1]))
    grid = [[tiles[r * ncols + c][2] for c in range(ncols)] for r in range(nrows)]
    return grid


def detect_grid_tiles(png_path, nrows, ncols, x_min_frac=0.11, thresh=245):
    """Auto-detect face tiles in a wandb eval grid via white-gutter projection."""
    img = np.asarray(Image.open(png_path).convert("RGB"))
    H, W, _ = img.shape
    content = img.min(axis=2) < thresh

    def runs(profile, min_len):
        out, start = [], None
        for i, v in enumerate(profile):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_len:
                    out.append((start, i))
                start = None
        if start is not None and len(profile) - start >= min_len:
            out.append((start, len(profile)))
        return out

    xs0 = int(W * x_min_frac)
    col_profile = content[:, xs0:].any(axis=0)
    # rows first, using only the tile x-region
    row_profile = content[:, xs0:].any(axis=1)
    row_runs = sorted(runs(row_profile, int(0.05 * H)),
                      key=lambda r: r[1] - r[0], reverse=True)[:nrows]
    row_runs.sort()
    col_profile = np.zeros(W - xs0, dtype=bool)
    for r0, r1 in row_runs:
        col_profile |= content[r0:r1, xs0:].any(axis=0)
    col_runs = sorted(runs(col_profile, int(0.03 * W)),
                      key=lambda r: r[1] - r[0], reverse=True)[:ncols]
    col_runs.sort()

    pil = Image.fromarray(img)
    grid = []
    for r0, r1 in row_runs:
        row = []
        for c0, c1 in col_runs:
            row.append(pil.crop((xs0 + c0, r0, xs0 + c1, r1)))
        grid.append(row)
    return grid


def show_face(ax, img, title=None, ylabel=None, psnr=None, box_color=None):
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if box_color is not None:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(box_color)
            sp.set_linewidth(1.2)
    if title:
        ax.set_title(title, fontsize=7, pad=2.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
    if psnr is not None:
        ax.set_xlabel(f"{psnr:.1f} dB", fontsize=6.5, labelpad=1.5)


# ---------------------------------------------------------------------------
# qualitative panels
# ---------------------------------------------------------------------------

def fig_qual_ghosting():
    """Single column: ONE view (view 0), 4 method columns -- same view
    everywhere, only the method varies. Source PDF layout is 2 rows (views)
    x 4 cols (GT / fused TC off / fused TC on / fused 64-ch); per-tile dB are
    the per-clip view-0 values printed in the cluster-generated source."""
    grid = extract_pdf_tiles("qual_ghosting.pdf", 2, 4)  # grid[view][method]
    titles = ["Ground truth", "Fused, TC off", "Fused, TC on", "Fused, 64-ch"]
    psnrs = [None, 32.7, 27.8, 30.9]

    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.15))
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.995,
                        top=0.86, bottom=0.12)
    for c in range(4):
        show_face(axes[c], grid[0][c], title=titles[c], psnr=psnrs[c])
    save(fig, "qual_ghosting.pdf")


def fig_qual_bleeding_strip():
    """Full width: GT vs fused 16-ch reconstruction over all 9 frames (view 2),
    chunk-colored borders. This is the main temporal-bleeding visual.
    Per-frame dB labels: per-clip PSNR computed from the dump-grid tiles
    themselves (PNG is lossless; rec row re-aligned by its constant 1-px
    vertical crop offset; values cross-checked against the dataset per-frame
    curve). A 3-row variant with an amplified |GT-Rec| error row exists at
    commit 2bd1284 if ever needed."""
    grid = detect_grid_tiles(os.path.join(OUT, "_wandb_E1d.png"), 4, 9)
    gt_row, rec_row = grid[1], grid[3]   # view 2 input / view 2 reconstruction
    frame_db = [31.7, 26.1, 25.2, 27.4, 27.1, 25.4, 28.5, 29.5, 27.3]

    fig, axes = plt.subplots(2, 9, figsize=(W2, 1.92))
    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.035, right=0.998,
                        top=0.80, bottom=0.095)
    chunk_colors = ["#555555"] + [C_CHUNK1] * 4 + [C_CHUNK2] * 4
    for c in range(9):
        show_face(axes[0, c], gt_row[c], title=f"$f_{c}$",
                  ylabel="GT" if c == 0 else None, box_color=chunk_colors[c])
        show_face(axes[1, c], rec_row[c],
                  ylabel="Rec." if c == 0 else None, box_color=chunk_colors[c],
                  psnr=frame_db[c])

    # group labels above the tiles, at the true axes centers (no overlap)
    fig.canvas.draw()
    def center(i0, i1):
        b0 = axes[0, i0].get_position()
        b1 = axes[0, i1].get_position()
        return (b0.x0 + b1.x1) / 2
    fig.text(center(0, 0), 0.965, "frame 0", color="#555555",
             fontsize=7, ha="center", va="top")
    fig.text(center(1, 4), 0.965, "chunk 1  ($f_1$–$f_4$ → 1 latent frame)",
             color=C_CHUNK1, fontsize=7, ha="center", va="top")
    fig.text(center(5, 8), 0.965, "chunk 2  ($f_5$–$f_8$ → 1 latent frame)",
             color=C_CHUNK2, fontsize=7, ha="center", va="top")
    save(fig, "qual_bleeding.pdf", tight=False)


def fig_qual_bleeding_grid():
    """Single-column 3x4 companion (supplement): methods x frames, chunk 1.
    Source PDF rows: GT / 16-ch / 32-ch (3 rows); becomes 4 rows with a 64-ch
    row once make_qual_from_dumps.py is regenerated on the cluster (E11b dump).
    Per-tile dB labels are the per-clip, per-frame values printed in the
    cluster-generated source PDF (computed from the dumps on the correct
    [0,1] path -- unchanged by the clamp fix)."""
    doc = pymupdf.open(os.path.join(OLD, "qual_bleeding.pdf"))
    nrows = len(doc[0].get_image_info()) // 4
    doc.close()
    grid = extract_pdf_tiles("qual_bleeding.pdf", nrows, 4)
    row_labels = ["GT", "16-ch", "32-ch", "64-ch"][:nrows]
    psnrs = [None,
             [28.6, 27.3, 28.0, 27.8],
             [31.0, 28.5, 30.7, 30.3],
             None]  # 64-ch: fill from the regenerated source PDF's labels
    fig, axes = plt.subplots(nrows, 4, figsize=(W1, 0.84 * nrows))
    fig.subplots_adjust(wspace=0.05, hspace=0.155, left=0.075, right=0.995,
                        top=1 - 0.14 / nrows, bottom=0.186 / nrows)
    for r in range(nrows):
        for c in range(4):
            show_face(axes[r, c], grid[r][c],
                      title=f"$f_{c+1}$" if r == 0 else None,
                      ylabel=row_labels[r] if c == 0 else None,
                      psnr=None if psnrs[r] is None else psnrs[r][c])
    save(fig, "qual_bleeding_grid.pdf")


def fig_qual_capacity():
    """Single column: 1 example row x 4 width columns, whole faces.
    Source PDF is 1x4 since the 2026-10-02 regen (GT / 16 / 32 / 64 at one
    expressive frame). Per-tile dB labels are the per-clip values printed in
    the cluster-generated source PDF (correct [0,1] path, unchanged by fix)."""
    grid = extract_pdf_tiles("qual_capacity.pdf", 1, 4)
    titles = ["Ground truth", "16-ch", "32-ch", "64-ch"]
    psnrs = [None, 28.0, 29.1, 29.3]

    fig, axes = plt.subplots(1, 4, figsize=(W1, 1.15))
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.995,
                        top=0.86, bottom=0.12)
    for c in range(4):
        show_face(axes[c], grid[0][c], title=titles[c], psnr=psnrs[c])
    save(fig, "qual_capacity.pdf")


# ---------------------------------------------------------------------------
# quantitative plots
# ---------------------------------------------------------------------------

def _chunk_bands(ax, ymax_label=None):
    ax.axvspan(-0.5, 0.5, color="#dddddd", alpha=0.5, lw=0)
    ax.axvspan(0.5, 4.5, color=C_CHUNK1, alpha=0.08, lw=0)
    ax.axvspan(4.5, 8.5, color=C_CHUNK2, alpha=0.08, lw=0)
    for x in (0.5, 4.5):
        ax.axvline(x, color="#999999", lw=0.5, ls="--")
    if ymax_label is not None:
        ax.text(0.0, ymax_label, "$f_0$", color="#666666", fontsize=6, ha="center", va="top")
        ax.text(2.5, ymax_label, "chunk 1", color=C_CHUNK1, fontsize=6, ha="center", va="top")
        ax.text(6.5, ymax_label, "chunk 2", color=C_CHUNK2, fontsize=6, ha="center", va="top")


def fig_perframe():
    """Single-column per-frame PSNR (V=2 main setting). Clean-wave final_eval."""
    series = [
        ("Fused, TC off", C_FUSED, "-",
         [33.60, 33.45, 32.81, 32.38, 32.39, 32.34, 32.52, 32.63, 32.54]),
        ("Fused, TC on", C_JOINT, "-",
         [32.95, 30.82, 28.66, 29.35, 27.88, 25.94, 27.35, 28.60, 28.22]),
        ("+ diff-loss", "#17becf", "--",
         [32.01, 30.20, 28.69, 29.75, 28.38, 26.04, 27.38, 29.15, 28.43]),
        ("+ 32-ch", C_32CH, ":",
         [33.51, 31.76, 29.96, 30.30, 29.38, 27.94, 28.59, 30.23, 29.46]),
        ("+ both", "#7a3e9d", "-.",
         [35.37, 33.57, 31.26, 32.51, 30.95, 29.62, 30.03, 32.37, 31.05]),
    ]
    xs = np.arange(9)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.97, bottom=0.13)
    _chunk_bands(ax, ymax_label=36.6)
    for lab, col, ls, data in series:
        ax.plot(xs, data, color=col, ls=ls, lw=1.2, marker="o", ms=2.6,
                markeredgewidth=0, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$f_{i}$" for i in range(9)])
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(23.6, 36.9)
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.8, loc="lower left", ncol=2, framealpha=0.95,
              borderpad=0.35, handlelength=1.6, columnspacing=0.9)
    save(fig, "perframe_psnr.pdf")

    # V=4 companion (supplement)
    series_v4 = [
        ("V=4, TC off", C_FUSED, "-",
         [30.62, 30.63, 30.04, 29.77, 29.86, 29.94, 29.91, 30.10, 30.08]),
        ("V=4, TC on", C_JOINT, "-",
         [29.45, 28.03, 26.37, 26.12, 25.38, 23.89, 25.03, 25.89, 25.35]),
    ]
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.97, bottom=0.15)
    _chunk_bands(ax, ymax_label=31.25)
    for lab, col, ls, data in series_v4:
        ax.plot(np.arange(9), data, color=col, ls=ls, lw=1.2, marker="o",
                ms=2.6, markeredgewidth=0, label=lab)
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels([f"$f_{i}$" for i in range(9)])
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(23.4, 31.5)
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.95)
    save(fig, "perframe_psnr_v4.pdf")


def fig_latent_width():
    """Single column: widening under joint compression + per-view reference.
    2026-10-06: local re-run wave @300 epochs (capacity-table budget, seed 42)."""
    xs = np.arange(3)
    psnrs = [28.20, 30.61, 31.34]  # E1d, E11a, E11b (local wave @300 ep)
    cols = [C_JOINT, C_32CH, C_64CH]

    fig, ax = plt.subplots(figsize=(W1, 1.75))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.96, bottom=0.16)

    ax.axhline(33.12, color=C_PVTCT, lw=1.0, ls="--")
    ax.text(2.32, 33.12 - 0.25, "per-view, TC on (33.12 dB)", color=C_PVTCT,
            fontsize=6.5, ha="right", va="top")

    ax.plot(xs, psnrs, "-", color="#aaaaaa", lw=1.0, zorder=2)
    for x, p, col in zip(xs, psnrs, cols):
        ax.scatter([x], [p], color=col, s=34, zorder=5)
        ax.annotate(f"{p:.2f}", (x, p), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.5)
    ax.text(0.5, 29.10, "+2.41 dB", fontsize=6.5, ha="center",
            color="#222222", rotation=32)
    ax.text(1.5, 30.70, "+0.73 dB", fontsize=6.5, ha="center", color="#222222",
            rotation=14)

    ax.set_xticks(xs)
    ax.set_xticklabels(["16 (Wan default)", "32", "64"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(26.7, 34.4)
    ax.set_xlabel("Latent channels")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    save(fig, "latent_width.pdf")


def fig_interventions():
    """Single column: horizontal bars, deltas vs the joint baseline.
    2026-10-06: local re-run wave, uniform 170-epoch budget (Table-2 budget,
    seed 42). NOTE the sign flip vs the old cluster wave: diff-loss ALONE now
    lands below the baseline (its old +1.83 came from the biased-PSNR era);
    it only pays off combined with capacity. "+ diff-loss + cache" (E4i)
    dropped: never re-run in the clean wave."""
    base = 28.28
    items = [
        ("Baseline (fused, TC on)", base, C_JOINT),
        ("+ temporal diff-loss", 27.20, "#aec7e8"),
        ("+ 32-ch latent", 30.35, C_32CH),
        ("+ 64-ch latent", 30.20, C_64CH),
        ("+ diff-loss + 32-ch", 30.36, "#7a3e9d"),
        ("+ all tweaks (64-ch)", 31.56, "#1f77b4"),
    ]
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    fig.subplots_adjust(left=0.42, right=0.985, top=0.97, bottom=0.20)
    ys = np.arange(len(items))
    ax.barh(ys, [v for _, v, _ in items], color=[c for _, _, c in items],
            edgecolor="white", height=0.62, lw=0.4)
    ax.axvline(base, color=C_JOINT, lw=0.7, ls="--", alpha=0.55)
    for y, (_, v, _) in enumerate(items):
        d = v - base
        s = f"{v:.2f}" + (f"  ({d:+.2f})" if abs(d) > 0.005 else "")
        ax.text(v + 0.07, y, s, va="center", fontsize=6.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _, _ in items], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(25.9, 34.4)
    ax.set_xlabel("PSNR (dB)")
    ax.grid(True, axis="x", alpha=0.25, lw=0.4)
    save(fig, "interventions.pdf")


def fig_datascale():
    """Single column, 3 panels: failure metrics vs generalization pressure."""
    labels = ["Single\nsequence", "All people\n(one expr.)"]
    xs = np.arange(2)
    bleed = [0.983, 0.919]
    xv_rec = [0.847, 0.920]
    xv_gt = [0.845, 0.918]
    # PSNR panel corrected 2026-10-02: all-people point = corrected E1d of the
    # same (pre-clean) wave, (27.30+27.02)/2. CAUTION: the single-sequence
    # overfit run has no per-view log, so 34.56 is still the PRE-FIX value
    # (biased >=1.7 dB LOW). Dagger removed per request 2026-10-02 -- REFILL
    # this number from the cluster jsonl before camera-ready.
    psnr = [34.56, 27.16]

    fig, axes = plt.subplots(1, 3, figsize=(W1, 1.55))
    fig.subplots_adjust(wspace=0.62, left=0.115, right=0.985, top=0.88, bottom=0.265)

    ax = axes[0]
    ax.plot(xs, bleed, "o-", color=C_JOINT, lw=1.3, ms=4.5)
    ax.axhline(1.0, color="#999999", lw=0.6, ls="--")
    for x, b, dy, va in zip(xs, bleed, (-0.012, 0.012), ("top", "bottom")):
        ax.annotate(f"{b:.3f}", (x, b), xytext=(0, 8 * np.sign(dy)),
                    textcoords="offset points", ha="center", va=va, fontsize=6)
    ax.set_title("Bleeding", fontsize=7.5, pad=3)
    ax.set_ylabel("Bleed-W $\\uparrow$", fontsize=7, labelpad=1)
    ax.set_ylim(0.885, 1.035)

    ax = axes[1]
    ax.plot(xs, xv_rec, "o-", color=C_JOINT, lw=1.3, ms=4.5, label="rec.")
    ax.plot(xs, xv_gt, "s--", color="#888888", lw=1.0, ms=3.8, label="GT")
    ax.set_title("Ghosting", fontsize=7.5, pad=3)
    ax.set_ylabel("XView sim $\\downarrow$", fontsize=7, labelpad=1)
    ax.set_ylim(0.83, 0.945)
    ax.legend(fontsize=5.5, loc="upper left", framealpha=0.9,
              borderpad=0.3, handlelength=1.4)

    ax = axes[2]
    ax.plot(xs, psnr, "o-", color=C_JOINT, lw=1.3, ms=4.5)
    for x, p, va, off in zip(xs, psnr, ("top", "bottom"), (-8, 8)):
        ax.annotate(f"{p:.2f}", (x, p), xytext=(0, off),
                    textcoords="offset points", ha="center", va=va, fontsize=6)
    ax.set_title("Quality", fontsize=7.5, pad=3)
    ax.set_ylabel("PSNR (dB) $\\uparrow$", fontsize=7, labelpad=1)
    ax.set_ylim(22, 38)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    save(fig, "datascale.pdf")


def fig_rate_quality():
    """Single column: the 2x2 rate-quality matrix as grouped bars.
    2026-10-06: local re-run wave, uniform 170-epoch budget (Table-1 budget)."""
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.96, bottom=0.155)
    x = np.arange(2)
    w = 0.32
    perview = [35.85, 33.13]  # E1a, E1b (local wave @170)
    fused = [33.13, 28.28]    # E1c, E1d (local wave @170)
    # additive prediction at TC on: perview_tcoff - TC cost - fusion cost
    add_pred = perview[0] - (perview[0] - perview[1]) - (perview[0] - fused[0])
    b1 = ax.bar(x - w / 2, perview, w, color="#1f77b4", label="Per-view",
                edgecolor="white", lw=0.4)
    b2 = ax.bar(x + w / 2, fused, w, color=C_JOINT, label="Fused",
                edgecolor="white", lw=0.4)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.18,
                    f"{bar.get_height():.2f}", ha="center", fontsize=6.3)
    ax.plot([1 + w / 2 - 0.16, 1 + w / 2 + 0.16], [add_pred, add_pred],
            color="#444444", lw=1.4)
    ax.text(1 + w / 2 + 0.20, add_pred, f"additive\nprediction: {add_pred:.1f}",
            fontsize=5.8, color="#444444", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["TC off", "TC on"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(25.0, 38.4)
    ax.set_xlim(-0.6, 1.85)
    ax.legend(fontsize=6.3, loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    save(fig, "rate_quality.pdf")


def fig_view_count():
    # V=8 pending clean retrain; plot 2 and 4 only from clean wave.
    # PSNRs corrected 2026-10-02 (clamp bug); E6b/E6c = mean of 4 per-view PSNRs.
    views = [2, 4]
    psnr_f, psnr_t = [32.74, 30.18], [28.27, 25.51]
    bleed_f, bleed_t = [0.969, 0.956], [0.913, 0.842]

    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.55))
    fig.subplots_adjust(wspace=0.50, left=0.125, right=0.985, top=0.88, bottom=0.235)
    for ax, (a, b, ylab, title) in zip(axes, [
            (psnr_f, psnr_t, "PSNR (dB)", "Quality"),
            (bleed_f, bleed_t, "Bleed-W $\\uparrow$", "Bleeding")]):
        ax.plot(views, a, "o-", color=C_FUSED, lw=1.2, ms=3.8, label="TC off")
        ax.plot(views, b, "s-", color=C_JOINT, lw=1.2, ms=3.8, label="TC on")
        ax.set_xticks(views)
        ax.set_xlabel("views $V$", fontsize=7, labelpad=1)
        ax.set_ylabel(ylab, fontsize=7, labelpad=1)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25, lw=0.4)
    axes[1].axhline(1.0, color="#999999", lw=0.6, ls="--")
    axes[0].legend(fontsize=5.5, loc="upper right", framealpha=0.9,
                   borderpad=0.3, handlelength=1.4)
    save(fig, "view_count.pdf")


def fig_resolution():
    # Corrected 2026-10-02. E9a/E9b are pre-clean-wave runs (not retrained in the
    # clean wave): corrected = mean of final_eval per-view PSNRs of those runs
    # (E9a 29.48/30.29, E9b 28.71/29.13). 128px point = corrected E1d of the same
    # era (27.30/27.02); the old 25.31 came from a sweep jsonl not recoverable
    # locally, its corrected equivalent would be ~27.6.
    res = [128, 256, 512]
    psnrs = [27.16, 29.88, 28.92]
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 1.55))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.20)
    ax.plot(res, psnrs, "o-", color=C_JOINT, lw=1.3, ms=4.5)
    for r, p, off in zip(res, psnrs, (8, 8, 8)):
        ax.annotate(f"{p:.2f}", (r, p), xytext=(0, off),
                    textcoords="offset points", ha="center", fontsize=6.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(res)
    ax.set_xticklabels([str(r) for r in res])
    ax.minorticks_off()
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(26.3, 31.2)
    ax.grid(True, alpha=0.25, lw=0.4)
    save(fig, "resolution_scaling.pdf")


def fig_zeroshot_paths():
    """Single column: pretrained Wan 2.1, zero shot, on a natural 720p clip --
    native chunked causal decode vs the per-frame (TC off) wrapper, across
    resolutions. Values from paper/audit/zeroshot_local_results.json
    (2026-10-02 audit); the NeRSemble marker is the corrected E1z table value,
    which runs the same TC-off path at 128 px."""
    import json
    res_names = ["natural_704", "natural_256", "natural_128"]
    try:
        with open(os.path.join(os.path.dirname(OUT), "audit",
                               "zeroshot_local_results.json")) as f:
            R = json.load(f)
        native = [R[f"native_TC_chunked/{n}/bf16"]["psnr"] for n in res_names]
        tcoff = [R[f"E1z_TCoff_sampled/{n}/bf16"]["psnr"] for n in res_names]
    except FileNotFoundError:
        native = [36.88, 30.45, 28.14]
        tcoff = [21.87, 21.98, 21.97]

    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.96, bottom=0.155)
    b1 = ax.bar(x - w / 2, native, w, color="#1f77b4",
                label="Native chunked decode", edgecolor="white", lw=0.4)
    b2 = ax.bar(x + w / 2, tcoff, w, color="#d62728",
                label="Per-frame decode (TC off)", edgecolor="white", lw=0.4)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                    f"{bar.get_height():.1f}", ha="center", fontsize=6.3)
    # NeRSemble zero-shot references at 128 px (local re-run 2026-10-04, true
    # values): native chunked path 32.51 dB, TC-off wrapper 23.05 dB.
    ax.plot([2.47, 2.74], [23.05, 23.05], color="#8b1a1a", lw=1.4)
    ax.text(2.605, 23.43, "NeRSemble\n23.1", fontsize=5.6,
            color="#8b1a1a", va="bottom", ha="center")
    ax.plot([2.47, 2.74], [32.51, 32.51], color="#0d3b66", lw=1.4)
    ax.text(2.605, 32.13, "NeRSemble\n32.5", fontsize=5.6,
            color="#0d3b66", va="top", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["704$^2$", "256$^2$", "128$^2$"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(18.5, 39.6)
    ax.set_xlim(-0.6, 2.95)
    ax.legend(fontsize=6.0, loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    save(fig, "zeroshot_paths.pdf")


if __name__ == "__main__":
    fig_qual_ghosting()
    fig_qual_bleeding_strip()
    fig_qual_bleeding_grid()
    fig_qual_capacity()
    fig_perframe()
    fig_latent_width()
    fig_interventions()
    fig_datascale()
    fig_rate_quality()
    fig_view_count()
    fig_resolution()
    fig_zeroshot_paths()
    print("All CVPR figures written to", OUT)
