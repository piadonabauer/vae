#!/usr/bin/env python3
"""Fast FG / full XView from clean eval dumps (no LPIPS)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

DUMP_ROOT = Path("/home/piado/projects/aip-lindell/piado/vae/Open-Sora/outputs")
OUT = Path("/home/piado/projects/aip-lindell/piado/vae/paper")
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


def to01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x / 255.0) if float(x.max()) > 1.5 else x.clamp(0, 1)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))


def eval_arm(job: str) -> dict:
    t0 = time.time()
    data = torch.load(DUMP_ROOT / job / "best_val_eval_dump.pt", map_location="cpu", weights_only=False)
    print(f"  loaded in {time.time() - t0:.1f}s, {len(data['clips'])} clips", flush=True)
    full_g, full_r, fg_g, fg_r, fracs = [], [], [], [], []
    for clip in data["clips"]:
        gt = to01(clip["gt"])
        rec = to01(clip["rec"])
        white = (gt > 0.96).all(dim=1)  # V,T,H,W
        fg = (~white.any(dim=0)).float()  # T,H,W
        fracs.append(float(fg.mean()))
        m = fg.unsqueeze(0).unsqueeze(0)  # 1,1,T,H,W
        gt_fg = gt * m
        rec_fg = rec * m
        for t in range(gt.shape[2]):
            full_g.append(cos(gt[0, :, t], gt[1, :, t]))
            full_r.append(cos(rec[0, :, t], rec[1, :, t]))
            fg_g.append(cos(gt_fg[0, :, t], gt_fg[1, :, t]))
            fg_r.append(cos(rec_fg[0, :, t], rec_fg[1, :, t]))

    def mean(xs):
        return float(np.mean(xs))

    return {
        "fg_frac": mean(fracs),
        "xview_gt_full": mean(full_g),
        "xview_rec_full": mean(full_r),
        "gap_full": mean(full_r) - mean(full_g),
        "xview_gt_fg": mean(fg_g),
        "xview_rec_fg": mean(fg_r),
        "gap_fg": mean(fg_r) - mean(fg_g),
        "n_clips": len(data["clips"]),
    }


def main():
    results = {}
    for name, job in ARMS.items():
        print("eval", name, flush=True)
        results[name] = eval_arm(job)
        r = results[name]
        print(
            f"  full gap={r['gap_full']:+.4f}  fg gap={r['gap_fg']:+.4f}  fg_frac={r['fg_frac']:.3f}",
            flush=True,
        )

    (OUT / "ghosting_metrics.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Ghosting metrics from clean eval dumps",
        "",
        "FG mask = not near-white in GT (thresh 0.96); cosine on masked tensors (bg zeroed).",
        "",
        "| arm | xRec full | xGT full | gap full | xRec FG | xGT FG | gap FG | fg frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, m in results.items():
        lines.append(
            f"| {n} | {m['xview_rec_full']:.4f} | {m['xview_gt_full']:.4f} | {m['gap_full']:+.4f} | "
            f"{m['xview_rec_fg']:.4f} | {m['xview_gt_fg']:.4f} | {m['gap_fg']:+.4f} | {m['fg_frac']:.3f} |"
        )
    gd, gc = results["E1d"]["gap_fg"], results["E1c"]["gap_fg"]
    lines += [
        "",
        "## Takeaway",
        "",
        f"- E1d FG gap={gd:+.4f} vs E1c FG gap={gc:+.4f}",
        f"- E1d full gap={results['E1d']['gap_full']:+.4f} vs E1a={results['E1a']['gap_full']:+.4f}",
    ]
    if abs(gd - gc) < 0.01:
        lines.append(
            "- **Decision:** FG cosine still does not cleanly separate joint vs fused-TC-off; "
            "drop absolute XView from the main claim; keep qualitative + Bleed-W."
        )
    else:
        lines.append("- **Decision:** FG gap separates settings — consider a FG-XView column.")
    (OUT / "ghosting_metrics.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "ghosting_metrics.md", flush=True)


if __name__ == "__main__":
    main()
