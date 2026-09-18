"""
Generate qualitative reconstruction grid from best_val_eval_dump.pt files.
Must be run on a cluster node where PyTorch is available.

Usage: python make_qualitative_grid.py
Outputs: qualitative_grid.pdf in the same directory.
"""
import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs"

# Key runs to compare (ordered row by row in the grid)
RUNS = {
    "E1a": ("paper_E1a_perview_tcF",        "Per-view TC=off\n(E1a, 34.01 dB)"),
    "E1c": ("paper_E1c_fused_tcF",          "Fused TC=off\n(E1c, 31.21 dB)"),
    "E1d": ("paper_E1d_fused_tcT",          "Fused TC=on\n(E1d, 25.31 dB)"),
    "E4h": ("paper_E4h_temporal_diff_loss", "Fused + diff-loss\n(E4h, 27.14 dB)"),
    "E11a":("paper_E11a_fused_tcT_widen32", "Fused 32ch TC=on\n(E11a, 27.71 dB)"),
}
# Validation participants to show (first 3 val clips in the dump)
N_SHOW = 3
VIEW_IDX = 0   # which view to show in the grid (0=first, 1=second)
FRAME_IDX = 4  # which frame to show (frame 4 = inside chunk 1, where bleeding is visible)

def find_dump(prefix):
    for d in sorted(os.listdir(OUT_DIR)):
        if d.startswith(prefix) and "overfit" not in d and "__job" in d:
            p = os.path.join(OUT_DIR, d, "best_val_eval_dump.pt")
            if os.path.exists(p):
                return p
    return None

def load_clips(path):
    """Returns (gt, rec) each shape [N_clips, V, T, H, W, 3] float in [0,1].
    Actual dump format: dict with 'clips' key, each clip is dict with
    'gt'/'rec' tensors of shape [V, C, T, H, W] uint8 in [0,255].
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "clips" in data:
        clips = data["clips"]
        gts  = torch.stack([c["gt"]  for c in clips]).float() / 255.0  # [N, V, C, T, H, W]
        recs = torch.stack([c["rec"] for c in clips]).float() / 255.0
        # [N, V, C, T, H, W] -> [N, V, T, H, W, C]
        gts  = gts.permute(0, 1, 3, 4, 5, 2)
        recs = recs.permute(0, 1, 3, 4, 5, 2)
        return gts, recs

    # Fallback for old format: list of dicts or flat dict
    if isinstance(data, list):
        gts  = torch.stack([c["gt"]  for c in data]).float()
        recs = torch.stack([c["rec"] for c in data]).float()
    elif isinstance(data, dict):
        gts  = data["gt"].float()
        recs = data["rec"].float()
    else:
        raise ValueError(f"Unknown dump format: {type(data)}")

    def norm(x):
        x = x.float()
        if x.max() > 2.0:   # uint8
            x = x / 255.0
        elif x.min() < -0.1:  # [-1, 1]
            x = (x + 1) / 2
        return x.clamp(0, 1)
    gts = norm(gts); recs = norm(recs)

    # [N, V, C, T, H, W] -> [N, V, T, H, W, C]
    if gts.ndim == 6 and gts.shape[2] == 3:
        gts  = gts.permute(0, 1, 3, 4, 5, 2)
        recs = recs.permute(0, 1, 3, 4, 5, 2)
    return gts, recs

def to_img(tensor):
    """tensor: [..., H, W, 3] float -> uint8 HWC numpy."""
    return (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)

def diff_map(gt, rec):
    """Absolute difference, amplified 3x, greyscale as RGB."""
    diff = (gt.float() - rec.float()).abs() * 3
    diff = diff.mean(-1, keepdim=True).expand_as(gt)
    return diff.clamp(0, 1)

# ─── Build the figure ────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 7, "savefig.dpi": 300, "figure.dpi": 100,
                      "pdf.fonttype": 42, "savefig.bbox": "tight"})

N_ROWS = 1 + len(RUNS)     # GT row + one row per model
N_COLS = N_SHOW * 3        # 3 sub-columns per participant: view0, view1, diff
FIG_W  = N_COLS * 1.35
FIG_H  = N_ROWS * 1.35

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig, hspace=0.04, wspace=0.02)

def show(ax, img, title=""):
    ax.imshow(img); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=6, pad=2)

# Row 0: GT
try:
    dump_path = find_dump("paper_E1c_fused_tcF")  # any run with a dump; GT is the same
    if dump_path is None: dump_path = find_dump("paper_E1d_fused_tcT")
    gt, _ = load_clips(dump_path)
    for pi in range(N_SHOW):
        for vi, vname in enumerate(["view 0", "view 1"]):
            ax = fig.add_subplot(gs[0, pi * 3 + vi])
            img = to_img(gt[pi, vi, FRAME_IDX])
            title = f"GT {vname} (p{pi})" if pi == 0 else ""
            show(ax, img, title)
        # diff of GT views (should be non-zero)
        ax = fig.add_subplot(gs[0, pi * 3 + 2])
        show(ax, to_img(diff_map(gt[pi, 0, FRAME_IDX], gt[pi, 1, FRAME_IDX])),
             "GT diff" if pi == 0 else "")
    fig.text(-0.01, 1 - 0.5 / N_ROWS, "Ground\ntruth", ha="right", va="center",
             fontsize=7, fontweight="bold")
except Exception as e:
    print(f"Warning: GT row failed: {e}")

# Model rows
for row_idx, (eid, (prefix, row_label)) in enumerate(RUNS.items()):
    dump_path = find_dump(prefix)
    if dump_path is None:
        print(f"No dump for {prefix}, skipping")
        continue
    try:
        gt, rec = load_clips(dump_path)
    except Exception as e:
        print(f"Failed to load {dump_path}: {e}")
        continue

    for pi in range(N_SHOW):
        # view 0
        ax = fig.add_subplot(gs[row_idx + 1, pi * 3])
        show(ax, to_img(rec[pi, 0, FRAME_IDX]))
        # view 1
        ax = fig.add_subplot(gs[row_idx + 1, pi * 3 + 1])
        show(ax, to_img(rec[pi, 1, FRAME_IDX]))
        # diff between the two views (ghosting visible here)
        ax = fig.add_subplot(gs[row_idx + 1, pi * 3 + 2])
        show(ax, to_img(diff_map(rec[pi, 0, FRAME_IDX], rec[pi, 1, FRAME_IDX])))

    fig.text(-0.01, 1 - (row_idx + 1.5) / N_ROWS, row_label, ha="right", va="center",
             fontsize=6.5, fontweight="bold")

# Column headers
for pi in range(N_SHOW):
    fig.text((pi * 3 + 1 + 0.5) / N_COLS, 1.01,
             f"Participant {pi+1}", ha="center", va="bottom", fontsize=7, fontweight="bold")

fig.suptitle(f"Qualitative reconstructions  (frame {FRAME_IDX}, inside chunk 1; columns: view0 / view1 / cross-view diff×3)",
             y=1.04, fontsize=7)

path = os.path.join(OUT, "qualitative_grid.pdf")
fig.savefig(path)
print(f"Saved {path}")
