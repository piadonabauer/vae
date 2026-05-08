#!/usr/bin/env python3

import os
import json
import re
import argparse
from collections import defaultdict
import numpy as np
import yaml
from tabulate import tabulate

# # --- Extract rank and fusion_mode from config.yaml ---
# def extract_config_yaml(yaml_path):
#     with open(yaml_path, "r") as f:
#         cfg = yaml.safe_load(f)

#     try:
#         e_dict = cfg["_wandb"]["value"]["e"]
#         first_run = list(e_dict.values())[0]
#         args = first_run["args"]

#         lora_rank = None
#         fusion_mode = None
#         for i, v in enumerate(args):
#             if v == "--model.lora_rank":
#                 lora_rank = int(args[i+1].strip('"'))
#             elif v == "--model.fusion_mode":
#                 fusion_mode = args[i+1]
#         return lora_rank, fusion_mode
#     except Exception as e:
#         print(f"Failed to read {yaml_path}: {e}")
#         return None, None

# # --- Extract from old .py file (fallback) ---
# def extract_config_py(config_path):
#     with open(config_path, "r") as f:
#         text = f.read()
#     lora_rank_match = re.search(r"lora_rank\s*=\s*(\d+)", text)
#     fusion_mode_match = re.search(r'fusion_mode\s*=\s*"([^"]+)"', text)
#     lora_rank = int(lora_rank_match.group(1)) if lora_rank_match else None
#     fusion_mode = fusion_mode_match.group(1) if fusion_mode_match else None
#     return lora_rank, fusion_mode

# # --- Load metrics from JSON files ---
# def load_metrics(file_path):
#     with open(file_path, "r") as f:
#         data = json.load(f)
#     columns = data["columns"]
#     values = data["data"]
#     idx = {col: i for i, col in enumerate(columns)}
#     metrics = defaultdict(list)
#     for row in values:
#         metrics["psnr"].append(row[idx["psnr"]])
#         metrics["ssim"].append(row[idx["ssim"]])
#         metrics["mse"].append(row[idx["mse"]])
#     return metrics

# def compute_stats(metric_list):
#     arr = np.array(metric_list)
#     return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

# # --- Detect .py config ---
# def get_config_path(run_path):
#     outputs_dir = os.path.join(run_path, "outputs")
#     if not os.path.exists(outputs_dir):
#         return None
#     subfolders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
#     if len(subfolders) != 1:
#         return None
#     config_path = os.path.join(outputs_dir, subfolders[0], "training_config_snapshot.py")
#     return config_path if os.path.exists(config_path) else None

# # --- Analyze single run ---
# def analyze_run(run_path, steps):
#     yaml_path = os.path.join(run_path, "config.yaml")
#     if os.path.exists(yaml_path):
#         lora_rank, fusion_mode = extract_config_yaml(yaml_path)
#     else:
#         config_path = get_config_path(run_path)
#         if not config_path:
#             return None
#         lora_rank, fusion_mode = extract_config_py(config_path)

#     if lora_rank is None or fusion_mode is None:
#         return None

#     fixed_seq_dir = os.path.join(run_path, "media/table/fixed_seq")
#     if not os.path.exists(fixed_seq_dir):
#         return None

#     all_metrics = defaultdict(list)
#     for step in steps:
#         files = [f for f in os.listdir(fixed_seq_dir)
#                  if f.startswith(f"train_metrics_{step}_") and f.endswith(".json")]
#         for file in files:
#             metrics = load_metrics(os.path.join(fixed_seq_dir, file))
#             for k, v in metrics.items():
#                 all_metrics[k].extend(v)

#     global_stats = {k: compute_stats(v) for k, v in all_metrics.items()}

#     return {
#         "run_name": os.path.basename(run_path),
#         "rank": lora_rank,
#         "fusion_mode": fusion_mode,
#         "global": global_stats
#     }

# # --- Main ---
# def main(base_dir, steps):
#     results = []
#     for run in os.listdir(base_dir):
#         run_path = os.path.join(base_dir, run)
#         if not os.path.isdir(run_path):
#             continue
#         run_stats = analyze_run(run_path, steps)
#         if run_stats:
#             results.append(run_stats)

#     # --- Aggregation by fusion mode ---
#     fusion_metrics = defaultdict(lambda: defaultdict(list))
#     rank_metrics = defaultdict(lambda: defaultdict(list))

#     for r in results:
#         fusion = r["fusion_mode"]
#         rank = r["rank"]
#         for metric, stats in r["global"].items():
#             fusion_metrics[fusion][metric].append(stats["mean"])
#             rank_metrics[rank][metric].append(stats["mean"])

#     # --- Prepare tables ---
#     fusion_table = []
#     for fusion, metrics_dict in fusion_metrics.items():
#         fusion_table.append([
#             fusion,
#             f"{np.mean(metrics_dict['psnr']):.3f}",
#             f"{np.mean(metrics_dict['ssim']):.5f}",
#             f"{np.mean(metrics_dict['mse']):.6f}"
#         ])

#     rank_table = []
#     for rank, metrics_dict in rank_metrics.items():
#         rank_table.append([
#             rank,
#             f"{np.mean(metrics_dict['psnr']):.3f}",
#             f"{np.mean(metrics_dict['ssim']):.5f}",
#             f"{np.mean(metrics_dict['mse']):.6f}"
#         ])

#     # --- Print tables ---
#     print("\n=== Fusion Mode Comparison ===")
#     print(tabulate(fusion_table, headers=["Fusion Mode", "PSNR mean", "SSIM mean", "MSE mean"], tablefmt="grid"))

#     print("\n=== Rank Comparison ===")
#     print(tabulate(rank_table, headers=["Rank", "PSNR mean", "SSIM mean", "MSE mean"], tablefmt="grid"))

#     print("\n=== All Run Metrics ===")
#     run_table = []
#     for r in results:
#         run_table.append([
#             r["run_name"],
#             r["fusion_mode"],
#             r["rank"],
#             f"{r['global']['psnr']['mean']:.3f}",
#             f"{r['global']['ssim']['mean']:.5f}",
#             f"{r['global']['mse']['mean']:.6f}"
#         ])
#     print(tabulate(run_table, headers=["Run", "Fusion", "Rank", "PSNR", "SSIM", "MSE"], tablefmt="grid"))

#     # --- Save outputs ---
#     out_file = os.path.join(base_dir, "wandb_all_runs_summary.json")
#     with open(out_file, "w") as f:
#         json.dump({
#             "fusion_table": fusion_table,
#             "rank_table": rank_table,
#             "all_runs": results
#         }, f, indent=2)
#     print(f"\nAll summary outputs saved to {out_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Analyze WandB sweep with YAML configs")
#     parser.add_argument("--base_dir", type=str, required=True, help="Path to wandb sweep directory")
#     parser.add_argument("--start", type=int, default=7000, help="Start step")
#     parser.add_argument("--end", type=int, default=8000, help="End step")
#     parser.add_argument("--step", type=int, default=200, help="Step interval")
#     args = parser.parse_args()

#     steps = list(range(args.start, args.end + 1, args.step))
#     main(args.base_dir, steps)

#!/usr/bin/env python3

import os
import json
import numpy as np
import argparse
from collections import defaultdict
from tabulate import tabulate

# --- Load metrics from JSON ---
def load_metrics(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    columns = data["columns"]
    values = data["data"]
    idx = {col: i for i, col in enumerate(columns)}

    metrics = defaultdict(list)
    for row in values:
        metrics["psnr"].append(row[idx["psnr"]])
        metrics["ssim"].append(row[idx["ssim"]])
        metrics["mse"].append(row[idx["mse"]])

    return metrics


# --- Collect metrics for a run ---
def collect_run_metrics(run_path, steps):
    fixed_seq_dir = os.path.join(run_path, "media/table/fixed_seq")

    if not os.path.exists(fixed_seq_dir):
        print(f"Missing metrics folder: {run_path}")
        return None

    all_metrics = defaultdict(list)

    for step in steps:
        files = [
            f for f in os.listdir(fixed_seq_dir)
            if f.startswith(f"train_metrics_{step}_") and f.endswith(".json")
        ]

        for file in files:
            metrics = load_metrics(os.path.join(fixed_seq_dir, file))
            for k, v in metrics.items():
                all_metrics[k].extend(v)

    if not all_metrics:
        return None

    return {
        k: float(np.mean(v))
        for k, v in all_metrics.items()
    }


# --- Main ---
def main(run_paths, start, end, step):
    steps = list(range(start, end + 1, step))

    results = []

    for path in run_paths:
        stats = collect_run_metrics(path, steps)
        if stats:
            results.append([
                os.path.basename(path),
                f"{stats['psnr']:.3f}",
                f"{stats['ssim']:.5f}",
                f"{stats['mse']:.6f}"
            ])
        else:
            results.append([os.path.basename(path), "N/A", "N/A", "N/A"])

    print("\n=== 7000–8000 Average Metrics Comparison ===")
    print(tabulate(
        results,
        headers=["Run", "PSNR", "SSIM", "MSE"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 3 runs (7000–8000 avg metrics)")
    parser.add_argument("--runs", nargs=3, required=True, help="Paths to 3 run folders")
    parser.add_argument("--start", type=int, default=7000)
    parser.add_argument("--end", type=int, default=8000)
    parser.add_argument("--step", type=int, default=200)

    args = parser.parse_args()

    main(args.runs, args.start, args.end, args.step)