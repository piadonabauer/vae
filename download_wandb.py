# import wandb
# import pandas as pd

# api = wandb.Api()

# run = api.run("pia-uni/wan_multiview_vae/77625p7k")

# history = run.history(samples=100000)

# fixed_cols = [c for c in history.columns if c.startswith("fixed_seq")]

# df = history[["_step"] + fixed_cols]

# df.to_csv("fixed_seq_metrics.csv", index=False)

# print(df.head())







# import pandas as pd
# import pandas as pd
# import matplotlib.pyplot as plt
# import os


# df = pd.read_csv("fixed_seq_metrics.csv")

# # keep rows where fixed_seq metrics exist
# df = df.dropna(how="all", subset=[c for c in df.columns if c.startswith("fixed_seq")])
# # Drop rows where all fixed_seq metrics are NaN
# fixed_cols = [c for c in df.columns if c.startswith("fixed_seq")]
# df = df.dropna(how="all", subset=fixed_cols)

# output_dir = "/home/piado/projects/aip-lindell/piado/vae/wandb_outputs"
# os.makedirs(output_dir, exist_ok=True)

# # ---------------------------
# # Helper function to plot metrics
# # ---------------------------
# def plot_frames(df, metric_name, split="train"):
#     cols = [c for c in df.columns if f"{split}/{metric_name}_frame" in c]
#     if not cols:
#         return
    
#     # Per-frame plot
#     plt.figure(figsize=(10,6))
#     for col in sorted(cols):
#         frame = col.split("_")[-1]
#         plt.plot(df["_step"], df[col], label=f"frame {frame}")
#     plt.xlabel("Training Step")
#     plt.ylabel(metric_name.upper())
#     plt.title(f"{split.capitalize()} sequence {metric_name.upper()} per frame")
#     plt.legend(ncol=3)
#     plt.savefig(os.path.join(output_dir, f"{split}_{metric_name}_per_frame.png"), dpi=300, bbox_inches="tight")
#     plt.close()

#     # Mean plot
#     plt.figure(figsize=(10,6))
#     df[f"{metric_name}_mean"] = df[cols].mean(axis=1)
#     plt.plot(df["_step"], df[f"{metric_name}_mean"], label=f"{split} mean")
#     plt.xlabel("Training Step")
#     plt.ylabel(metric_name.upper())
#     plt.title(f"{split.capitalize()} sequence mean {metric_name.upper()}")
#     plt.savefig(os.path.join(output_dir, f"{split}_{metric_name}_mean.png"), dpi=300, bbox_inches="tight")
#     plt.close()

# # ---------------------------
# # Metrics to plot
# # ---------------------------
# metrics = ["psnr", "ssim", "mse"]
# splits = ["train", "val"]

# for metric in metrics:
#     for split in splits:
#         plot_frames(df, metric, split)

# print(f"All plots saved to {output_dir}")





# import pandas as pd

# # Load cleaned CSV
# csv_file = "/home/piado/projects/aip-lindell/piado/vae/fixed_seq_metrics.csv"
# df = pd.read_csv(csv_file)

# # Drop rows where all fixed_seq metrics are NaN
# fixed_cols = [c for c in df.columns if c.startswith("fixed_seq")]
# df = df.dropna(how="all", subset=fixed_cols)

# # Function to compute best frames
# def best_frames(df, metric, split, higher_is_better=True):
#     cols = sorted([c for c in df.columns if f"{split}/{metric}_frame" in c])
#     means = df[cols].mean()
    
#     # Sort frames
#     best = means.sort_values(ascending=not higher_is_better)
#     print(f"\nBest {split} frames for {metric.upper()}:")
#     print(best.head(5))   # top 5 frames
#     print("Worst 5 frames:")
#     print(best.tail(5))

# metrics = ["psnr", "ssim", "mse"]
# splits = ["train", "val"]

# for metric in metrics:
#     for split in splits:
#         higher = True if metric in ["psnr", "ssim"] else False
#         best_frames(df, metric, split, higher_is_better=higher)

import wandb
import os

# Your project path
entity = "pia-uni"
project = "wan_multiview_vae"

# List of run IDs
run_ids = [
    # "cwmfsx6b",
    # "gpuc7wf0",
    # "18hdry13",
    # "hi5odsjr",
    # "aym5w393",
    # "thpqjz2x",
    # "secf6qjt",
    # "1813ph7s",
    # "ljg9x9be",
    # "nbvmyxje",
    # "my0tzjp6",
    # "vrb8rwdn"
    "r64oa1jt",
    "8xc811c3"
]

# Initialize API
api = wandb.Api()

# Output directory
output_dir = "/home/piado/projects/aip-lindell/piado/vae/wandb_outputs/cross-att-128"
os.makedirs(output_dir, exist_ok=True)

for run_id in run_ids:
    run_path = f"{entity}/{project}/{run_id}"
    print(f"Downloading run: {run_path}")
    
    run = api.run(run_path)
    
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Download all files from the run
    for file in run.files():
        print(f"  Downloading {file.name}")
        file.download(root=run_dir, replace=True)

print("Done!")