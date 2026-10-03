#!/usr/bin/env python3
"""Recompute corrected metrics from a final_eval dump (CLUSTER-side tool).

Usage:
    python paper/audit/recompute_from_dumps.py <final_eval_dump_val.pt> [more.pt ...]

Each dump (written by train.py evaluate_model with dump_dir set) is a list of
dicts {"gt": uint8 tensor, "rec": uint8 tensor, "path": str}, with gt/rec either
[V,C,T,H,W] or [C,T,H,W]. This prints, per file:

  - psnr        mean of per-(clip,view) PSNRs on [0,1]  <- THE paper convention
  - psnr_pooled pooled-MSE PSNR (reference only, ~0.3 dB lower)
  - per-view PSNR
  - ssim        canonical 11x11 Gaussian windowed SSIM  <- use this in the table
  - ssim_global legacy train.py global-moment SSIM (comparison only)
  - bleed ratios, per-frame PSNR

and appends a row to recomputed_from_dumps.jsonl next to each dump.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_unified import compute_all_metrics  # noqa: E402


def load_dump(path):
    entries = torch.load(path, map_location="cpu", weights_only=False)
    # Real train.py format (seen 2026-10-03 on local final-wave dumps):
    # {"clips": [{"gt": uint8 [V,C,T,H,W], "rec": ..., "path": str}], "format": str}
    if isinstance(entries, dict) and "clips" in entries:
        entries = entries["clips"]
    gts, recs = [], []
    for e in entries:
        gt, rec = e["gt"], e["rec"]
        if gt.dim() == 4:  # [C,T,H,W] -> [1,C,T,H,W]
            gt, rec = gt.unsqueeze(0), rec.unsqueeze(0)
        gts.append(gt)
        recs.append(rec)
    gt = torch.stack(gts).float() / 255.0   # [B,V,C,T,H,W] in [0,1]
    rec = torch.stack(recs).float() / 255.0
    return gt, rec


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        gt, rec = load_dump(path)
        m = compute_all_metrics(gt, rec, value_range="[0,1]")
        row = {
            "dump": os.path.abspath(path),
            "n_samples": int(gt.shape[0]),
            "psnr": round(m["psnr"], 4),
            "psnr_pooled": round(m["psnr_pooled"], 4),
            "psnr_per_view": [round(p, 4) for p in m["psnr_per_view"]],
            "psnr_per_frame": [round(p, 4) for p in m["psnr_per_frame"]],
            "ssim": round(m["ssim"], 6),
            "ssim_global_legacy": round(m["ssim_global"], 6),
            "bleed_ratio_within": round(m["bleed_ratio_within"], 5),
            "bleed_ratio_across": round(m["bleed_ratio_across"], 5),
        }
        print(json.dumps(row, indent=1))
        out = os.path.join(os.path.dirname(os.path.abspath(path)),
                           "recomputed_from_dumps.jsonl")
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("appended ->", out)


if __name__ == "__main__":
    main()
