# ============
# Fine-tuning config for Wan 2.1 VAE on NeRSemble multi-view data
# ============
# Multi-sequence (16 samples): use batch_size=1 + lower LR for stability.
# If output stays white and loss is spiky: (1) Use batch_size=1, lr=2e-4, vae_loss_preset="multiview".
# (2) Verify single-sequence still works (one .pt, repeat=1, epochs=100) to confirm checkpoint/range.
# (3) Ensure all .pt files have same shape [V,C,T,H,W] and value range [0,1].
# batch_size must be <= num_samples or steps_per_epoch is 0.

fixed_seq_eval_every_epochs = 5

# ============
# model config 
# ============
model = dict(
    type="multiview_wan_video_vae",  # Registered in opensora/models/vae/__init__.py
    z_dim=16,  # Latent dimension for Wan 2.1 VAE
    view_in=2,  # Number of input views (2 upper cameras)
    view_compression=2,  # Compression ratio (only used when use_view_group_fusion=True)
    use_view_embedding=False,  # Learnable per-view add before fusion; set True for stronger view separation
    use_view_group_fusion=False,  # If False: only latent_fusion + latent_expand (simplest setup)
    view_mixing_strategy="embedding",  # Key: embeddings guide what each view should decode to
    from_scratch=False,
    # Load Wan 2.1 VAE checkpoint
    from_pretrained="/home/piado/scratch/Wan2.1_VAE.pth",
    freeze_temporal=True,
    train_spatial=True,
    # New cross-view encoder with in-encoder fusion + LoRA
    use_crossview_encoder=True,  # use MultiViewVideoVan instead of latent_fusion path
    # Fusion mode options (only used when use_crossview_encoder=True):
    # - "cross_attention": bidirectional cross-attn between views (existing behavior)
    # - "self_attention": self-attn across view axis per token, then mean(V)->1
    # - "conv3d": channel-concat views, 1x1x1 3D conv 768->384 + norm/act + residual blocks
    fusion_mode="conv3d",
    use_lora=True,               # master switch for LoRA modules in crossview path
    use_lora_before=False,       # apply LoRA to newly introduced pre-bottleneck fusion modules
    use_lora_after=True,         # apply LoRA to bottleneck/decoder ("later" part)
    # Replace additive per-view latent embedding in decoder with view-specific latent LoRA adapters.
    use_viewwise_decoder_lora=False,
    lora_rank=64,                # configurable LoRA rank for all LoRA adapters
)

# ============
# data config 
# ============
# Data root for NeRSemble 128-res (pXXX / <sequence> / *.pt).
DATA_ROOT = "/datasets/lindell-proj/neumayr/nersemble_v2/processed/64-res"

# Preset: "single_sequence" | "one_person" | "some_people" | "all_people_one_expression" | None (custom)
# - single_sequence: 1 person, 1 sequence (p018 EXP-1-head). No val set.
# - one_person: p018, all sequences except EMO-4-disgust+happy and SEN-10-port_strong_smokey; those two are val.
# - some_people: train 17,31,32,33,35,36,37; evaluate on 18,30.
# - all_people_one_expression: train on ALL folders in 128-res (all people) EXCEPT 018,030,038,085,097,124,175,226,227,240;
#   only EXP-1-head sequences for training. Val = those excluded participants, EXP-1-head only.
data_preset = "single_sequence"

if data_preset == "single_sequence":
    dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT + "/p018/EMO-1-shout+laugh/EMO-1-shout+laugh.pt",
        #"/p018/EXP-1-head/p018_EXP-1-head.pt",
        repeat=1,
    )
    val_dataset = None
elif data_preset == "one_person":
    dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[18],
        exclude_sequences=["EMO-4-disgust+happy", "SEN-10-port_strong_smokey"],
        repeat=1,
    )
    val_dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[18],
        include_only_sequences=["EMO-4-disgust+happy", "SEN-10-port_strong_smokey"],
        repeat=1,
    )
elif data_preset == "some_people":
    dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[17, 31, 32, 33, 35, 36, 37],
        repeat=1,
    )
    val_dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[18, 30],
        repeat=1,
    )
elif data_preset == "all_people_one_expression":
    # All people = all folders in DATA_ROOT; exclude these for training (use only for val).
    _val_participants = [18, 30, 38, 85, 97, 124, 175, 226, 227, 240]
    dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=None,  # all pXXX in DATA_ROOT
        exclude_participants=_val_participants,
        expression_filter="EXP-1-head",
        repeat=1,
    )
    val_dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=_val_participants,
        expression_filter="EXP-1-head",
        repeat=1,
    )
else:
    # Custom: set dataset (and optionally val_dataset) as needed
    dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[17, 18, 30, 31, 32, 33, 35, 36, 37],
        repeat=1,
    )
    val_dataset = None

# Bucket config for preprocessed 64x64 / 9-frame inputs.
bucket_config = {
    #"128px_ar1:1": {13: (1.0, 1)},  # 13 frames at 128x128 resolution
    #"128px_ar1:1": {9: (1.0, 1)},  # 9 frames selected from 13 at 128x128 resolution
    "64px_ar1:1": {9: (1.0, 1)},  # 9 frames selected from 13 at 64x64 resolution
}

num_bucket_build_workers = 1  # Small dataset, don't need many workers
num_workers = 2  # Small dataset, minimal workers
prefetch_factor = 2

# ============
# train config 
# ============
optim = dict(
    cls="AdamW",
    lr=5e-4,  # Lower LR for multi-sequence (5e-4 can be too high -> spiky loss, no convergence)
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
epochs = 10000  # One epoch = one pass over ALL samples (.pt files). steps_per_epoch = num_samples // batch_size. Total steps = epochs × steps_per_epoch. (9 participants = many samples, not 9.)
log_every = 200  # Log every 10 steps
# Save a checkpoint every N actual update steps (0 = only final save at end of training)
ckpt_every = 500
# Keep this many latest checkpoint dirs (epochX-global_stepY); -1 = keep all
keep_n_latest = 3
ema_decay = 0.9999  # High EMA decay for stable fine-tuning

# ============
# Resume training
# ============
# To resume, pass the checkpoint subdir (contains running_states.json and model/) via command line:
#   python scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py --load /path/to/epoch0-global_step200
# Use the full path to a folder like .../outputs/<exp_name>/epoch0-global_step200 (not the experiment root).

# Wandb configuration - name will be auto-generated from config
wandb = True
wandb_project = "wan_multiview_vae"
# wandb_expr_name will be auto-generated if not set (see train.py)

update_warmup_steps = False  # No warmup needed

# ============
# loss config 
# ============
# vae_loss_preset: 
# "multiview" = perceptual 0.5, 
# view_consistency 0.01 (use this for multi-view; worked for single-sequence)
vae_loss_preset = "default"
vae_loss_config = dict(
    perceptual_loss_weight=0.5,
    kl_loss_weight=5e-4,            # Small KL weight for fine-tuning
    view_consistency_weight=0.01,  # REDUCED: View consistency loss was causing identical reconstructions
)

# ============
# Multi-view specific config
# ============
# Wan VAE decoder outputs [-1, 1]; scale dataset [0, 1] -> [-1, 1] so loss and vis are correct
vae_target_range = "[-1,1]"
view_flatten_in_loss = False  # CHANGED: Don't flatten views, preserve view dimension for better loss
view_flatten_in_disc = True  # Flatten views for discriminator (if used)

# ============
# Evaluation config
# ============
# Batch-level metrics (lightweight, every N steps)
eval_every = 100  # Compute metrics on current batch every 50 steps

# Full evaluation pass (more expensive, every N steps). Increase to reduce overhead (e.g. 500).
full_eval_every = 100
eval_num_samples = 3  # Evaluate on this many samples (e.g. one per sequence)
eval_batch_size = 1  # Batch size for evaluation
eval_use_ema = True  # Use EMA model for evaluation if available

# Final evaluation
final_eval = True  # Run final evaluation after training
final_eval_num_samples = 3  # Number of samples for final eval

# Number of different training samples to show in reconstruction visualization (e.g. 3 = first 3 sequences)
num_reconstruction_vis_samples = 3

# ============
# Performance logging & bottleneck / shape debugging
# ============
log_memory = False  # Set to True to log GPU memory usage (adds small overhead)
# Print timing every N steps (0 = off). Set to 0 to reduce I/O overhead when optimizing for speed.
log_bottleneck_every = 200
# Print latent shapes [B,V,C,T,H,W] every N steps; also enables VAE-internal shape logs (0 = off).
log_latent_shapes_every = 100

# ============
# Training steps and speed
# ============
# Samples = total .pt files (e.g. 9 participants × ~7 sequences each = 63 samples). Steps per epoch = samples // batch_size. Total steps = epochs × steps_per_epoch (e.g. 63 × 125 = 7875).
batch_size = 128 # Use 1 for multi-sequence stability; increase to 2–4 once training converges
