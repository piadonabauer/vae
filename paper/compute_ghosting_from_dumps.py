#!/usr/bin/env python3
"""Offline ghosting metrics from best_val_eval_dump.pt (no weights needed).

FG mask: threshold near-white background in GT (RVM composited on white).
Reports whole-frame and FG-masked cross-view cosine, plus inter-view LPIPS
(rec views vs each other, and GT views vs each other as calibration).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    import lpips
except ImportError:
    lpips = None

OUT_DIR = Path(__file__).resolve().parent
DUMP_ROOT = Path("/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs")

ARMS = {
    "E1a": "paper_E1a_perview_tcF__job5562190_t1",
    "E1b": "paper_E1b_perview_tcT__job5556206_t2",
    "E1c": "paper_E1c_fused_tcF__job5562191_t3",
    "E1d": "paper_E1d_fused_tcT__job5562192_t4",
    "E11a": "paper_E11a_fused_tcT_widen32__job5562201_t8",
    "E11b": "paper_E11b_fused_tcT_widen64__job5562202_t9",
    "combo": "paper_E_combo_diffLoss_widen32__job5562206_t36",
    "E3d": "paper_E3d_no_emb_no_lora__job5556246_t16",
    "E2b": "paper_E2b_fused_tcF_self_attn__job5562194_t11",
}


def to_float01(x: torch.Tensor) -> torch.Tensor:
    """Dump format is uint8 [0,255] or float."""
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    x = x.float()
    if x.max() > 1.5:
        return x / 255.0
    return x.clamp(0, 1)


def fg_mask_from_white_bg(gt_vchw: torch.Tensor, thresh: float = 0.96) -> torch.Tensor:
    """gt: (V,C,T,H,W) in [0,1]. Mask True where not near-white."""
    # near-white if all channels high
    white = (gt_vchw > thresh).all(dim=1)  # (V,T,H,W)
    # union over views so both views share a comparable support
    fg = ~white.any(dim=0)  # (T,H,W) — pixel is FG if any view has non-white
    # fallback if everything white
    if fg.sum() < 16:
        fg = ~white[0]
    return fg  # (T,H,W)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b) / denom)


def pairwise_xview(frames: torch.Tensor, mask_thw: torch.Tensor | None) -> float:
    """frames: (V,C,T,H,W). Mean pairwise cosine over view pairs, averaged over T."""
    v = frames.shape[0]
    sims = []
    for t in range(frames.shape[2]):
        if mask_thw is not None:
            m = mask_thw[t]  # (H,W)
            if m.sum() < 8:
                continue
            vecs = [frames[i, :, t][:, m].reshape(-1) for i in range(v)]
        else:
            vecs = [frames[i, :, t].reshape(-1) for i in range(v)]
        for i in range(v):
            for j in range(i + 1, v):
                sims.append(cosine_sim(vecs[i], vecs[j]))
    return float(np.mean(sims)) if sims else float("nan")


def mean_lpips_between_views(frames: torch.Tensor, loss_fn) -> float:
    """Mean LPIPS between view pairs over time. frames (V,C,T,H,W) in [0,1]."""
    if frames.shape[0] < 2:
        return float("nan")
    # LPIPS wants [-1,1], NCHW
    vals = []
    for t in range(frames.shape[2]):
        a = frames[0, :, t].unsqueeze(0) * 2 - 1
        b = frames[1, :, t].unsqueeze(0) * 2 - 1
        with torch.no_grad():
            vals.append(float(loss_fn(a, b).item()))
    return float(np.mean(vals))


def eval_arm(dump_dir: Path, loss_fn) -> dict:
    dump = dump_dir / "best_val_eval_dump.pt"
    if not dump.exists():
        dump = dump_dir / "latest_full_eval_dump.pt"
    data = torch.load(dump, map_location="cpu", weights_only=False)
    clips = data["clips"]
    rows = []
    for clip in clips:
        gt = to_float01(clip["gt"])
        rec = to_float01(clip["rec"])
        mask = fg_mask_from_white_bg(gt)
        rows.append(
            {
                "xview_gt_full": pairwise_xview(gt, None),
                "xview_rec_full": pairwise_xview(rec, None),
                "xview_gt_fg": pairwise_xview(gt, mask),
                "xview_rec_fg": pairwise_xview(rec, mask),
                "lpips_gt_views": mean_lpips_between_views(gt, loss_fn) if loss_fn else None,
                "lpips_rec_views": mean_lpips_between_views(rec, loss_fn) if loss_fn else None,
                "fg_frac": float(mask.float().mean()),
            }
        )

    def mean_key(k):
        vals = [r[k] for r in rows if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(vals)) if vals else None

    return {
        "n_clips": len(rows),
        "fg_frac": mean_key("fg_frac"),
        "xview_gt_full": mean_key("xview_gt_full"),
        "xview_rec_full": mean_key("xview_rec_full"),
        "gap_full": mean_key("xview_rec_full") - mean_key("xview_gt_full")
        if mean_key("xview_rec_full") is not None
        else None,
        "xview_gt_fg": mean_key("xview_gt_fg"),
        "xview_rec_fg": mean_key("xview_rec_fg"),
        "gap_fg": mean_key("xview_rec_fg") - mean_key("xview_gt_fg")
        if mean_key("xview_rec_fg") is not None
        else None,
        "lpips_gt_views": mean_key("lpips_gt_views"),
        "lpips_rec_views": mean_key("lpips_rec_views"),
        "lpips_gap": (
            mean_key("lpips_rec_views") - mean_key("lpips_gt_views")
            if mean_key("lpips_rec_views") is not None and mean_key("lpips_gt_views") is not None
            else None
        ),
    }


def main():
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = None
    # LPIPS is slow on CPU; set GHOSTING_LPIPS=1 to enable.
    if lpips is not None and os.environ.get("GHOSTING_LPIPS", "0") == "1":
        loss_fn = lpips.LPIPS(net="alex").to(device).eval().cpu()
        print("LPIPS enabled")
    else:
        print("LPIPS skipped (set GHOSTING_LPIPS=1 to enable)")

    results = {}
    for name, jobdir in ARMS.items():
        d = DUMP_ROOT / jobdir
        if not d.exists():
            # try glob
            matches = list(DUMP_ROOT.glob(jobdir.split("__job")[0] + "__job*/"))
            matches = [m for m in matches if (m / "best_val_eval_dump.pt").exists()]
            if not matches:
                print(f"SKIP {name}: no dump")
                continue
            d = sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
        print(f"eval {name} <- {d.name}")
        results[name] = eval_arm(d, loss_fn)

    (OUT_DIR / "ghosting_metrics.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Ghosting metrics from clean eval dumps",
        "",
        "Computed offline from `best_val_eval_dump.pt` (gt/rec uint8).",
        "FG mask = not near-white in GT (thresh 0.96), unioned across views.",
        "LPIPS = mean Alex-LPIPS between the two views over time (lower = more similar / more ghosting).",
        "",
        "| arm | xRec full | xGT full | gap full | xRec FG | xGT FG | gap FG | LPIPS rec | LPIPS GT | LPIPS gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        def f(x, p=4):
            return f"{x:.{p}f}" if x is not None else "—"

        lines.append(
            f"| {name} | {f(m['xview_rec_full'])} | {f(m['xview_gt_full'])} | {f(m['gap_full'])} | "
            f"{f(m['xview_rec_fg'])} | {f(m['xview_gt_fg'])} | {f(m['gap_fg'])} | "
            f"{f(m['lpips_rec_views'], 3)} | {f(m['lpips_gt_views'], 3)} | {f(m['lpips_gap'], 3)} |"
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "- Whole-frame XView gaps stay tiny (same conclusion as clean jsonl).",
        "- If FG gaps and/or LPIPS(rec views)−LPIPS(GT views) also stay small, drop absolute XView",
        "  claims from the paper and keep ghosting qualitative + Bleed-W.",
        "- If FG gap clearly separates joint (E1d) from single-axis (E1a/E1c), keep a FG-XView column.",
        "",
    ]
    # auto takeaway
    if "E1d" in results and "E1c" in results:
        g_d = results["E1d"].get("gap_fg")
        g_c = results["E1c"].get("gap_fg")
        lp_d = results["E1d"].get("lpips_gap")
        lp_c = results["E1c"].get("lpips_gap")
        lines.append(f"- E1d FG gap={g_d:+.4f} vs E1c FG gap={g_c:+.4f}" if g_d is not None and g_c is not None else "")
        lines.append(
            f"- E1d LPIPS gap={lp_d:+.3f} vs E1c LPIPS gap={lp_c:+.3f}"
            if lp_d is not None and lp_c is not None
            else ""
        )
        if g_d is not None and g_c is not None and abs(g_d - g_c) < 0.005:
            lines.append(
                "- **Decision lean:** FG cosine still does not separate joint vs fused-TC-off; "
                "do not put absolute XView in the main claim. Prefer qualitative grids."
            )
        elif g_d is not None and g_c is not None and g_d > g_c + 0.01:
            lines.append("- **Decision lean:** FG XView gap is informative — consider a FG-XView column.")

    (OUT_DIR / "ghosting_metrics.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT_DIR / "ghosting_metrics.md")


if __name__ == "__main__":
    main()
