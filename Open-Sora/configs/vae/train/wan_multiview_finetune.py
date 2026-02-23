# ============
# Fine-tuning config for Wan 2.1 VAE on NeRSemble multi-view data
# ============
# This config is for overfitting on a single sequence to test the multi-view VAE setup

# ============
# model config 
# ============
model = dict(
    type="multiview_wan_video_vae",  # Registered in opensora/models/vae/__init__.py
    z_dim=16,  # Latent dimension for Wan 2.1 VAE
    view_in=2,  # Number of input views (2 upper cameras)
    view_compression=2,  # Compression ratio: 2 views -> 1 latent (like time dimension)
    use_view_embedding=True,  # Use STRONG view embeddings to differentiate
    view_mixing_strategy="embedding",  # Key: embeddings guide what each view should decode to
    from_scratch=False,
    # Load Wan 2.1 VAE checkpoint
    from_pretrained="/home/piado/scratch/Wan2.1_VAE.pth",
)

# ============
# data config 
# ============
dataset = dict(
    type="pt_video",  # Our custom dataset for .pt files
    data_path="/home/piado/projects/aip-lindell/piado/vae/data/preprocessed_initial_experiments/p17_EXP-1-head/017_EXP-1-head.pt",
    # Alternative: point to a single .pt file or JSON file
    # data_path="/home/piado/projects/aip-lindell/piado/data/preprocessed_initial_experiments/p17_EXP-1-head/017.pt",
    repeat=1000,  # Repeat the single sample many times for overfitting (1k steps / 1 sample = 1000 repeats)
)

# Bucket config: 128x128 resolution, 13 frames (12+1 temporal downsampling)
bucket_config = {
    "128px_ar1:1": {13: (1.0, 1)},  # 13 frames at 128x128 resolution
}

num_bucket_build_workers = 1  # Small dataset, don't need many workers
num_workers = 2  # Small dataset, minimal workers
prefetch_factor = 2

# ============
# train config 
# ============
optim = dict(
    cls="AdamW",  # Use standard PyTorch AdamW (HybridAdam requires CUDA_HOME)
    lr=5e-4,  # INCREASED: With proper loss scaling, we can use higher LR
    eps=1e-8,
    weight_decay=0.0,
    betas=(0.9, 0.98),
)
lr_scheduler = dict(warmup_steps=0)  # No warmup for overfitting

mixed_strategy = None  # Disable mixed strategy - we only have video
mixed_image_ratio = 0.0

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

grad_clip = 1.0
grad_checkpoint = False
pin_memory_cache_pre_alloc_numels = None  # Small dataset, don't need caching

seed = 42
outputs = "outputs"
epochs = 1  # Single epoch, but we'll use steps instead
log_every = 10  # Log every 10 steps
ckpt_every = 0  # Disable intermediate checkpoints (only save final checkpoint)
keep_n_latest = 1  # Keep only the final checkpoint
ema_decay = 0.9999  # High EMA decay for stable fine-tuning

# Wandb configuration - name will be auto-generated from config
wandb = True
wandb_project = "wan_multiview_vae"
# wandb_expr_name will be auto-generated if not set (see train.py)

update_warmup_steps = False  # No warmup needed

# ============
# loss config 
# ============
vae_loss_config = dict(
    perceptual_loss_weight=0.5,
    kl_loss_weight=5e-4,  # Small KL weight for fine-tuning
    view_consistency_weight=0.01,  # REDUCED: View consistency loss was causing identical reconstructions
)

# ============
# Multi-view specific config
# ============
view_flatten_in_loss = False  # CHANGED: Don't flatten views, preserve view dimension for better loss
view_flatten_in_disc = True  # Flatten views for discriminator (if used)

# ============
# Evaluation config
# ============
# Batch-level metrics (lightweight, every N steps)
eval_every = 50  # Compute metrics on current batch every 50 steps

# Full evaluation pass (more expensive, every N steps)
full_eval_every = 200  # Run full evaluation every 200 steps
eval_num_samples = 1  # Only 1 sample, so evaluate all of it
eval_batch_size = 1  # Batch size for evaluation
eval_use_ema = True  # Use EMA model for evaluation if available

# Final evaluation
final_eval = True  # Run final evaluation after training
final_eval_num_samples = 1  # Only 1 sample

# ============
# Performance logging
# ============
log_memory = False  # Set to True to log GPU memory usage (adds small overhead)

# ============
# Training steps (for overfitting)
# ============
# We want 1000 steps total. With 1 sample and batch_size=1, we need:
# - 1 sample * 1000 repeats = 1000 samples in dataset
# - batch_size = 1 means 1000 steps per epoch
# - epochs = 1 means exactly 1000 steps
batch_size = 1  # Single sample per batch for overfitting
