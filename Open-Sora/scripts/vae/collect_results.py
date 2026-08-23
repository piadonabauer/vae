"""Collect paper results from eval_metrics.jsonl files into one CSV.

Each training run appends full_eval / final_eval records to
outputs/<experiment>/eval_metrics.jsonl. This script picks, per run,
the best-val row (by PSNR) and the last row, and writes a flat CSV
that the paper tables / rate-quality plot are built from.

Usage:
    python scripts/vae/collect_results.py                      # outputs/paper_*
    python scripts/vae/collect_results.py --glob "outputs/*"   # everything
"""

import argparse
import csv
import glob
import json
import os

# Scalar metrics that go into the CSV (per-frame profiles are kept as JSON strings).
SCALAR_KEYS = [
    "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "mse_mean", "mse_std",
    "bleed_ratio_within", "bleed_ratio_across", "xview_sim_rec", "xview_sim_gt",
]


def load_records(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # half-written last line after a crash; skip
                continue
    return records


def pick_rows(records):
    """Return (best_val, last) full/final-eval records, either may be None."""
    evals = [r for r in records if r.get("kind") in ("full_eval", "final_eval")]
    val = [r for r in evals if r.get("split") == "val"]
    pool = val if val else evals
    if not pool:
        return None, None
    best = max(pool, key=lambda r: r.get("metrics", {}).get("psnr_mean", float("-inf")))
    last = pool[-1]
    return best, last


def row_for(run_name, tag, record):
    m = record.get("metrics", {})
    row = {
        "run": run_name,
        "which": tag,  # best_val | last
        "split": record.get("split", ""),
        "step": record.get("actual_update_step", ""),
        "epoch": record.get("epoch", ""),
    }
    for k in SCALAR_KEYS:
        row[k] = m.get(k, "")
    for k in ("psnr_per_frame", "l1_delta_per_frame_gt", "l1_delta_per_frame_rec"):
        row[k] = json.dumps(m[k]) if k in m else ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/paper_*", help="experiment dirs to scan")
    ap.add_argument("--out", default="paper_results.csv")
    args = ap.parse_args()

    rows = []
    for exp_dir in sorted(glob.glob(args.glob)):
        jsonl = os.path.join(exp_dir, "eval_metrics.jsonl")
        if not os.path.isfile(jsonl):
            continue
        # strip the __job<id>_t<task> suffix so resumed jobs collapse to one run name
        run_name = os.path.basename(exp_dir).split("__job")[0]
        best, last = pick_rows(load_records(jsonl))
        if best is None:
            print(f"[skip] no eval records in {jsonl}")
            continue
        rows.append(row_for(run_name, "best_val", best))
        if last is not best:
            rows.append(row_for(run_name, "last", last))

    if not rows:
        print("nothing found — did the runs write eval_metrics.jsonl?")
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len({r['run'] for r in rows})} runs -> {args.out}")


if __name__ == "__main__":
    main()
