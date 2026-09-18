"""
Focused qualitative panels. Clips are matched by participant id; a run whose
GT cameras do not match the reference dump is skipped (E4h used a different
camera pair / frames.pt).
"""
import os, re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs"

RUNS = {
    "E1a":  "paper_E1a_perview_tcF",
    "E1c":  "paper_E1c_fused_tcF",
    "E1d":  "paper_E1d_fused_tcT",
    "E4h":  "paper_E4h_temporal_diff_loss",
    "E11a": "paper_E11a_fused_tcT_widen32",
    "E11b": "paper_E11b_fused_tcT_widen64",
}

plt.rcParams.update({
    "font.size": 8, "savefig.dpi": 300, "figure.dpi": 100,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def find_dump(prefix):
    for d in sorted(os.listdir(OUT_DIR)):
        if d.startswith(prefix) and "overfit" not in d and "__job" in d:
            p = os.path.join(OUT_DIR, d, "best_val_eval_dump.pt")
            if os.path.exists(p):
                return p
    return None


def pid_of(path):
    m = re.search(r"/p(\d+)/", path)
    return f"p{m.group(1)}" if m else path


def load_run(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    clips = data["clips"]
    gts  = torch.stack([c["gt"]  for c in clips]).float() / 255.0
    recs = torch.stack([c["rec"] for c in clips]).float() / 255.0
    pids = [pid_of(c["path"]) for c in clips]
    return {"gt": gts, "rec": recs, "pids": pids}


def to_hwc(t):
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    return (t.numpy() * 255).clip(0, 255).astype(np.uint8)


def psnr_chw(rec, gt):
    mse = float(((rec - gt) ** 2).mean())
    if mse < 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def show(ax, img, title="", ylabel="", psnr=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, rotation=90, labelpad=4)
    ax.set_xlabel(f"{psnr:.1f} dB" if psnr is not None else " ", fontsize=7.5, labelpad=1)


loaded = {}
for eid, prefix in RUNS.items():
    p = find_dump(prefix)
    if p is None:
        print(f"MISSING dump for {eid}")
        continue
    loaded[eid] = load_run(p)
    print(f"Loaded {eid}: {tuple(loaded[eid]['gt'].shape)} pids={loaded[eid]['pids']}")

REF = "E1d"
ref = loaded[REF]
FRAME_MID = 4
PID_HAIR   = "p175"
PID_MOTION = "p240"
PID_ID     = "p085"


def aligned(eid, pid, view, frame):
    """(img_hwc, psnr) for this run, matched to REF cameras. (None, None) if mismatch."""
    run = loaded[eid]
    if pid not in run["pids"] or pid not in ref["pids"]:
        return None, None
    i = run["pids"].index(pid)
    i_ref = ref["pids"].index(pid)
    target = ref["gt"][i_ref, view, :, frame]
    best_v, best_mse = None, 1e9
    for v in range(run["gt"].shape[1]):
        mse = float(((run["gt"][i, v, :, frame] - target) ** 2).mean())
        if mse < best_mse:
            best_mse, best_v = mse, v
    if best_mse > 1e-4:
        print(f"  skip {eid} {pid}: cameras differ (GT mse={best_mse:.4f})")
        return None, None
    rec_t = run["rec"][i, best_v, :, frame]
    gt_t  = run["gt"][i, best_v, :, frame]
    return to_hwc(rec_t), psnr_chw(rec_t, gt_t)


def gt_frame(pid, view, frame):
    i = ref["pids"].index(pid)
    return to_hwc(ref["gt"][i, view, :, frame])


def fig_ghosting():
    models = [("Ground truth", None)]
    for name, eid in [("Fused, TC=off", "E1c"),
                      ("Fused, TC=on", "E1d"),
                      ("Fused, 32-ch", "E11a")]:
        if eid in loaded:
            models.append((name, eid))

    pid, frame = PID_HAIR, FRAME_MID
    fig, axes = plt.subplots(len(models), 2, figsize=(3.6, 1.65 * len(models)))
    fig.subplots_adjust(hspace=0.35, wspace=0.05)
    for r, (name, eid) in enumerate(models):
        for v in (0, 1):
            if eid is None:
                img, psnr = gt_frame(pid, v, frame), None
            else:
                img, psnr = aligned(eid, pid, v, frame)
            if img is None:
                axes[r, v].axis("off")
                continue
            show(axes[r, v], img,
                 title=f"View {v}" if r == 0 else "",
                 ylabel=name if v == 0 else "",
                 psnr=psnr)
    fig.savefig(os.path.join(OUT, "qual_ghosting.pdf"))
    plt.close(fig)
    print("Saved qual_ghosting.pdf")


def fig_bleeding():
    chunk_frames = list(range(1, 5))
    rows = [("Ground truth", None)]
    for name, eid in [("Fused, 16-ch", "E1d"),
                      ("Fused, 32-ch", "E11a"),
                      ("Fused, 64-ch", "E11b")]:
        if eid in loaded:
            rows.append((name, eid))

    pid, view = PID_MOTION, 0
    fig, axes = plt.subplots(len(rows), len(chunk_frames),
                             figsize=(1.55 * len(chunk_frames), 1.7 * len(rows)))
    fig.subplots_adjust(hspace=0.45, wspace=0.08)
    for r, (name, eid) in enumerate(rows):
        for c, f in enumerate(chunk_frames):
            if eid is None:
                img, psnr = gt_frame(pid, view, f), None
            else:
                img, psnr = aligned(eid, pid, view, f)
            if img is None:
                axes[r, c].axis("off")
                continue
            show(axes[r, c], img,
                 title=f"Frame {c + 1}" if r == 0 else "",
                 ylabel=name if c == 0 else "",
                 psnr=psnr)
    fig.suptitle("Temporal bleeding", fontsize=10, y=1.01)
    fig.savefig(os.path.join(OUT, "qual_bleeding.pdf"))
    plt.close(fig)
    print("Saved qual_bleeding.pdf")


def fig_capacity():
    cols = [("Ground truth", None)]
    for name, eid in [("16-ch", "E1d"),
                      ("32-ch", "E11a"),
                      ("64-ch", "E11b")]:
        if eid in loaded:
            cols.append((name, eid))

    people = [(PID_MOTION, "High motion"), (PID_ID, "Identity")]
    fig, axes = plt.subplots(len(people), len(cols),
                             figsize=(1.7 * len(cols), 1.95 * len(people)))
    fig.subplots_adjust(hspace=0.40, wspace=0.08)
    for r, (pid, row_name) in enumerate(people):
        for c, (name, eid) in enumerate(cols):
            if eid is None:
                img, psnr = gt_frame(pid, 0, FRAME_MID), None
            else:
                img, psnr = aligned(eid, pid, 0, FRAME_MID)
            if img is None:
                axes[r, c].axis("off")
                continue
            show(axes[r, c], img,
                 title=name if r == 0 else "",
                 ylabel=row_name if c == 0 else "",
                 psnr=psnr)
    fig.suptitle("Latent width", fontsize=10, y=1.02)
    fig.savefig(os.path.join(OUT, "qual_capacity.pdf"))
    plt.close(fig)
    print("Saved qual_capacity.pdf")


def fig_overview():
    row_spec = [
        ("Ground truth", None),
        ("Per-view, TC=off", "E1a"),
        ("Fused, TC=off", "E1c"),
        ("Fused, TC=on", "E1d"),
        ("32-ch latent", "E11a"),
        ("64-ch latent", "E11b"),
    ]
    row_spec = [r for r in row_spec if r[1] is None or r[1] in loaded]
    pids = [PID_HAIR, PID_MOTION, PID_ID]
    n_rows, n_cols = len(row_spec), len(pids) * 2

    fig = plt.figure(figsize=(n_cols * 1.2, n_rows * 1.35))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.28, wspace=0.03)

    for r, (name, eid) in enumerate(row_spec):
        for pi, pid in enumerate(pids):
            for v in (0, 1):
                ax = fig.add_subplot(gs[r, pi * 2 + v])
                if eid is None:
                    img, psnr = gt_frame(pid, v, FRAME_MID), None
                else:
                    img, psnr = aligned(eid, pid, v, FRAME_MID)
                if img is None:
                    ax.axis("off")
                    continue
                show(ax, img, title=f"{pid}  view {v}" if r == 0 else "", psnr=psnr)
        fig.text(-0.01, 1 - (r + 0.5) / n_rows, name, ha="right", va="center",
                 fontsize=7.5, fontweight="bold")

    fig.savefig(os.path.join(OUT, "qualitative_grid.pdf"))
    plt.close(fig)
    print("Saved qualitative_grid.pdf")


fig_ghosting()
fig_bleeding()
fig_capacity()
fig_overview()
print("All qualitative panels done.")
