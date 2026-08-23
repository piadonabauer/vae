"""Compute LPIPS from the eval clip dumps, offline (no training env needed).

Training only logs PSNR/SSIM/MSE + the paper diagnostics; LPIPS is computed
afterwards from the clips that evaluate_model saves to
outputs/<run>/final_eval_dump_val.pt (and best_val_eval_dump.pt). Frames are
scored individually with the standard AlexNet LPIPS and averaged per clip.

Usage:
    pip install lpips
    python scripts/vae/lpips_from_dump.py outputs/paper_*/final_eval_dump_val.pt
    python scripts/vae/lpips_from_dump.py --out lpips.csv outputs/paper_*/best_val_eval_dump.pt
"""

import argparse
import csv
import os

import torch


def clip_lpips(loss_fn, gt_u8, rec_u8, device, batch_frames=16):
    """gt/rec: uint8 [C,T,H,W] or [V,C,T,H,W] -> mean LPIPS over all frames."""

    def frames(x):
        if x.dim() == 5:  # [V,C,T,H,W]
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], x.shape[3], x.shape[4])
        else:  # [C,T,H,W]
            x = x.permute(1, 0, 2, 3)
        return x.float() / 127.5 - 1.0  # LPIPS expects [-1,1]

    a, b = frames(gt_u8), frames(rec_u8)
    vals = []
    with torch.no_grad():
        for i in range(0, a.shape[0], batch_frames):
            va = a[i : i + batch_frames].to(device)
            vb = b[i : i + batch_frames].to(device)
            vals.append(loss_fn(va, vb).flatten().cpu())
    return float(torch.cat(vals).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="*.pt files written by evaluate_model")
    ap.add_argument("--out", default="lpips_results.csv")
    ap.add_argument("--net", default="alex", choices=["alex", "vgg"])
    args = ap.parse_args()

    import lpips  # deferred: not a training dependency

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=args.net).to(device).eval()

    rows = []
    for dump_path in args.dumps:
        data = torch.load(dump_path, map_location="cpu")
        vals = [clip_lpips(loss_fn, c["gt"], c["rec"], device) for c in data["clips"]]
        t = torch.tensor(vals)
        run = os.path.basename(os.path.dirname(dump_path)).split("__job")[0]
        rows.append(
            {
                "run": run,
                "dump": os.path.basename(dump_path),
                "n_clips": len(vals),
                "lpips_mean": float(t.mean()),
                "lpips_std": float(t.std(unbiased=False)),
            }
        )
        print(f"{run:45s} {os.path.basename(dump_path):28s} "
              f"LPIPS {t.mean():.4f} ± {t.std(unbiased=False):.4f}  (n={len(vals)})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
