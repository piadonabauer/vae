# Test VAE training with view-residual decoder strategy
# This uses learned per-view residuals to differentiate multi-view reconstructions

import os
from datetime import datetime

# === Training Data ===
data_root = os.path.expanduser("~/projects/aip-lindell/piado/vae/data/preprocessed_initial_experiments")
metadata = None

videos = [
    f"{data_root}/p17_SEN-01-cramp_small_danger/data.pt",
]

num_frames = 13
height = 128
width = 128

num_workers = 8
prefetch_factor = 2

# === Model ===
model = dict(
    type="multiview_wan_video_vae",
    dim=96,
    z_dim=16,
    view_in=2,
    view_compression=1,  # No compression: 2 views -> 2 latent views (keeps view info)
    use_view_embedding=True,
    view_mixing_strategy="residual",  # KEY: Use residual decoder for strong differentiation
    from_pretrained=os.path.expanduser("~/projects/aip-lindell/piado/vae/data/rvm_mobilenetv3.pth"),
)

# === VAE Loss ===
loss = dict(
    type="vae_loss",
    kl_loss_weight=5e-4,
    perceptual_loss_weight=0.5,
    view_consistency_weight=0.01,  # Weak consistency allows views to differentiate
)

# === Optimizer ===
optimizer = dict(type="adamw", lr=5e-4, weight_decay=1e-4)
optim_type = "adamw"

# === Training ===
num_train_steps = 100000
num_accumulated_batches = 1
batch_size = 1
enable_fixed_batches = False
seed = 42

lr_scheduler = dict(
    type="constant",
)

save_model_steps = 5000
log_every = 100

checkpointer = dict(
    save_model_interval=5000,
    save_model_num_total=5,
)

# === Device ===
mixed_precision = "bf16"
enable_sequence_parallelism = False

log_dir = "./outputs/vae_residual_decoder"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"{log_dir}_{timestamp}"

# W&B
use_wandb = True
wandb_project = "multiview-vae-experiments"
wandb_entity = None  # Set to your W&B entity if desired
wandb_run_name = f"vae_residual_{timestamp}"
