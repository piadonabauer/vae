# ============
# Fine-tuning config for Wan 2.1 VAE on NeRSemble multi-view data
# ============
# Multi-sequence (16 samples): use batch_size=1 + lower LR for stability.
# If output stays white and loss is spiky: (1) Use batch_size=1, lr=2e-4, vae_loss_preset="multiview".
# (2) Verify single-sequence still works (one .pt, repeat=1, epochs=100) to confirm checkpoint/range.
# (3) Ensure all .pt files have same shape [V,C,T,H,W] and value range [0,1].
# batch_size must be <= num_samples or steps_per_epoch is 0.

fixed_seq_eval_every_epochs = 200

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
    # - "cross_attention": bidirectional cross-attn between views
    # - "self_attention": joint self-attn over all tokens from both views (length 2*N), then concat + ResBlocks
    # - "conv3d": concat views in channels, 1×1×1 Conv3d -> GroupNorm+SiLU -> two FusionResidualBlock3d (symmetric Conv3d)
    fusion_mode="conv3d",
    use_lora=True,               # masfter switch for LoRA modules in crossview path
    use_lora_before=False,       # apply LoRA to newly introduced pre-bottleneck fusion modules
    use_lora_after=True,         # apply LoRA to bottleneck/decoder ("later" part)
    # Replace additive per-view latent embedding in decoder with view-specific latent LoRA adapters.
    use_viewwise_decoder_lora=False,
    lora_rank=64,                # configurable LoRA rank for all LoRA adapters
)

# ============
# data config 
# ============
from opensora.utils.nersemble_bucket import resolve_nersemble_bucket

# Optional: parent of ``64-res`` / ``128-res`` (default: NeRSemble v2 processed root).
nersemble_processed_base = None

# ``DATA_ROOT``, ``train_target_hw``, ``train_target_frames`` are derived from ``bucket_config``:
# - ``128px_...`` + 13 frames → load ``.../128-res``, train at 128×128, 13 frames.
# - ``64px_...`` + ≤9 frames → ``.../64-res``; + >9 frames → ``128-res`` + on-the-fly downsample to 64.
bucket_config = {
    #"128px_ar1:1": {13: (1.0, 1)},
    "128px_ar1:1": {9: (1.0, 1)},
    # "64px_ar1:1": {13: (1.0, 1)},  # uses 128-res on disk, downsamples to 64×64
}
_resolved = resolve_nersemble_bucket(bucket_config, processed_base=nersemble_processed_base)
DATA_ROOT = _resolved["data_root"]
train_target_hw = _resolved["train_target_hw"]
train_target_frames = _resolved["train_target_frames"]

# Preset: "single_sequence" | "one_person" | "some_people" | "all_people_one_expression" | None (custom)
# - single_sequence: 1 person, 1 sequence (p018 EXP-1-head). No val set.
# - one_person: p018, all sequences except EMO-4-disgust+happy and SEN-10-port_strong_smokey; those two are val.
# - some_people: train 17,31,32,33,35,36,37; evaluate on 18,30.
# - all_people_one_expression: all participants (minus val list); use expression_sequence for exact folder name.
data_preset = "single_sequence"
# For all_people_one_expression: exact sequence folder per person (e.g. EMO-1-shout+laugh)
all_people_expression_sequence = "EMO-1-shout+laugh"

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
        expression_sequence=all_people_expression_sequence,
        repeat=1,
    )
    val_dataset = dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=_val_participants,
        expression_sequence=all_people_expression_sequence,
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
# Optional: fixed experiment folder name under outputs/ (else auto timestamp + config name)
# experiment_name = "cross_attn_lora_after_16_all_people_9t_2v_64p"
epochs = 10000  # One epoch = one pass over ALL samples (.pt files). steps_per_epoch = num_samples // batch_size. Total steps = epochs × steps_per_epoch. (9 participants = many samples, not 9.)
log_every = 100  # Log every 10 steps
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

# Wandb: charts use optimizer step (wandb.log step=...). Set wandb_expr_name for a distinct run name.
wandb = True
wandb_project = "wan_multiview_vae"
# wandb_expr_name = "conv3d_lora_after_16_all_people_9t_2v_64p"
# Only call wandb.init after this many optimizer steps (avoids empty runs on short tests; resume past step inits immediately)
wandb_min_steps_before_init = 10
log_step_time = True  # Once: print avg wall time over the first 10 steps (set False to disable; uses tqdm.write)

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
# eval_every: cheap metrics on the *current training batch* (PSNR/SSIM on that batch) every N steps.
eval_every = 200
# full_eval_every: separate dataloader over eval_num_samples from val (or train) — mean/std over several clips. Heavier.
full_eval_every = 1000
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
log_latent_shapes_every = 500

# ============
# Training steps and speed
# ============
# Samples = total .pt files (e.g. 9 participants × ~7 sequences each = 63 samples). Steps per epoch = samples // batch_size. Total steps = epochs × steps_per_epoch (e.g. 63 × 125 = 7875).
batch_size = 1 # Use 1 for multi-sequence stability; increase to 2–4 once training converges
