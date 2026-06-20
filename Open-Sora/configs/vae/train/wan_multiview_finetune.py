# ============
# Fine-tuning config for Wan 2.1 VAE on NeRSemble multi-view data
# ============
# Multi-sequence (16 samples): use batch_size=1 + lower LR for stability.
# If output stays white and loss is spiky: (1) Use batch_size=1, lr=2e-4, vae_loss_preset="multiview".
# (2) Verify single-sequence still works (one .pt, repeat=1, epochs=100) to confirm checkpoint/range.
# (3) Ensure all .pt files have same shape [V,C,T,H,W] and value range [0,1].
# batch_size must be <= num_samples or steps_per_epoch is 0.


fixed_seq_eval_every_epochs = 0 # CHANGE to 200

# ============
# model config /home/piado/projects/aip-lindell/piado/vae/Open-Sora/scripts/vae/vae_fusion_mode_sweep.sh
# ============
model = dict(
    type="multiview_wan_video_vae",  # Registered in opensora/models/vae/__init__.py
    z_dim=16,  # Latent dimension for Wan 2.1 VAE
    view_in=0,  # 0 = auto-detect from first dataset sample (train.py reads V from data shape)
    view_compression=2,  # Compression ratio (only used when use_view_group_fusion=True)
    use_view_embedding=False,  # Learnable per-view add before fusion; set True for stronger view separation
    use_view_group_fusion=False,  # If False: only latent_fusion + latent_expand (simplest setup)
    view_mixing_strategy="embedding",  # Key: embeddings guide what each view should decode to
    from_scratch=False,
    from_pretrained="/home/piado/scratch/Wan2.1_VAE.pth",
    freeze_temporal=True,
    train_spatial=True,
    # New cross-view encoder with in-encoder fusion + LoRA
    use_crossview_encoder=True,  # use MultiViewVideoVan instead of latent_fusion path
    # Fusion mode options (only used when use_crossview_encoder=True):
    # - "cross_attention": all-to-all cross-attn between every view and all views
    # - "self_attention": joint self-attn over all tokens from both views (length 2*N), then concat + ResBlocks
    # - "conv3d": concat views in channels, 1×1×1 Conv3d -> GroupNorm+SiLU -> two FusionResidualBlock3d (symmetric Conv3d)
    # conv4d
    fusion_mode="cross_attention",
    use_lora=True,               # masfter switch for LoRA modules in crossview path
    use_lora_before=False,       # LoRA on Encoder3d stem before fusion (encoder.conv1 + downsamples); additive to use_lora_after
    use_lora_after=True,         # apply LoRA to bottleneck/decoder ("later" part)
    # Replace additive per-view latent embedding in decoder with view-specific latent LoRA adapters.
    use_viewwise_decoder_lora=False,
    lora_rank=32,                # configurable LoRA rank for all LoRA adapters
    # Phase 2 (optional): train original decoder conv/attn inside DiffSynth LoRA wrappers (base weights),
    # while keeping view-conditioned decode (view_idx + nn.Embedding in AttentionMultiViewVideoVan).
    # Still set use_viewwise_decoder_lora / use_view_embedding in the legacy path as needed; crossview always has decode embeddings.
    full_finetune_decoder=False,
)

# ============
# data config 
# ============
from opensora.utils.nersemble_bucket import resolve_nersemble_bucket

# Optional: parent of ``64-res`` / ``128-res`` (default: NeRSemble v2 processed root).
nersemble_processed_base = "/datasets/lindell-proj/neumayr/nersemble_v2/processed/8-frames" #None

# ``DATA_ROOT``, ``train_target_hw``, ``train_target_frames`` are derived from ``bucket_config``:
# - ``256px_...`` + T frames → load ``.../256-res``, train at 256×256, T frames (e.g. 9).
# - ``128px_...`` + T frames → load ``.../128-res``, train at 128×128, T frames.
# - ``64px_...`` + ≤9 frames → ``.../64-res``; + >9 frames → ``128-res`` + on-the-fly downsample to 64.
bucket_config = {
    "128px_ar1:1": {9: (1.0, 1)},
    #"256px_ar1:1": {9: (1.0, 1)},  # ``<processed_base>/256-res``, 256×256, 9 frames
    # "64px_ar1:1": {13: (1.0, 1)},  # uses 128-res on disk, downsamples to 64×64
}
_resolved = resolve_nersemble_bucket(bucket_config, processed_base=nersemble_processed_base)
DATA_ROOT = _resolved["data_root"]
train_target_hw = _resolved["train_target_hw"]
train_target_frames = _resolved["train_target_frames"]

# Preset: "single_sequence" | "one_person" | "some_people" | "all_people" | "all_people_one_expression" | None (custom)
# - single_sequence: 1 person, 1 sequence (p018 EXP-1-head). No val set.
# - one_person: p018, all sequences except EMO-4-disgust+happy and SEN-10-port_strong_smokey; those two are val.
# - some_people: train 17,31,32,33,35,36,37; evaluate on 18,30.
# - all_people: all sequences from all participants except val participants.
# - all_people_one_expression: all participants (minus val list); use expression_sequence for exact folder name.
data_preset = "all_people_one_expression" #"single_sequence" #"all_people_one_expression"
# For all_people_one_expression: exact sequence folder per person (e.g. EMO-1-shout+laugh)
all_people_expression_sequence = "EMO-1-shout+laugh"

_val_participants = [97, 175, 226, 18, 30, 38, 85, 124, 227, 240]

dataset_presets = {
    "single_sequence": dict(
        type="pt_video",
        data_path=DATA_ROOT + "/p017/EMO-1-shout+laugh/frames.pt",
        #"/p018/EXP-1-head/p018_EXP-1-head.pt",
        repeat=1,
    ),
    "one_person": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[18],
        exclude_sequences=["EMO-4-disgust+happy", "SEN-10-port_strong_smokey"],
        repeat=1,
    ),
    "some_people": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[17, 31, 32, 33, 35, 36, 37],
        repeat=1,
    ),
    "all_people": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=None,  # all pXXX in DATA_ROOT
        exclude_participants=_val_participants,
        # Skip mismatched samples so V matches model.view_in
        # doesn't crash during visualization/eval.
        expected_views=2,
        skip_mismatched_views=True,
        repeat=1,
    ),
    "all_people_one_expression": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=None,  # all pXXX in DATA_ROOT
        exclude_participants=_val_participants,
        expression_sequence=all_people_expression_sequence,
        # Skip mismatched samples so V matches model.view_in
        # doesn't crash during visualization/eval.
        #expected_views=2,
        skip_mismatched_views=True,
        repeat=1,
    ),
    "__default__": dict(
        # Custom fallback: set dataset (and optionally val_dataset) as needed
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[17, 18, 30, 31, 32, 33, 35, 36, 37],
        repeat=1,
    ),
}

val_dataset_presets = {
    "single_sequence": None,
    "one_person": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[17],
        include_only_sequences=["EMO-4-disgust+happy", "SEN-10-port_strong_smokey"],
        repeat=1,
    ),
    "some_people": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=[18, 30],
        repeat=1,
    ),
    "all_people": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=_val_participants,
        expected_views=2,
        skip_mismatched_views=True,
        repeat=1,
    ),
    "all_people_one_expression": dict(
        type="pt_video",
        data_path=DATA_ROOT,
        scan_subdirs=True,
        participants=_val_participants,
        expression_sequence=all_people_expression_sequence,
        #expected_views=2,
        skip_mismatched_views=True,
        repeat=1,
    ),
    "__default__": None,
}

# Default construction; train.py may rebuild after CLI overrides.
dataset = dataset_presets.get(data_preset, dataset_presets["__default__"])
val_dataset = val_dataset_presets.get(data_preset, val_dataset_presets["__default__"])

num_bucket_build_workers = 1  # Small dataset, don't need many workers
num_workers = 8
prefetch_factor = 4  # per-worker prefetch queue (needs num_workers > 0)
# pin_memory=True is set in train.py. persistent_workers keeps workers alive between epochs (num_workers > 0).
persistent_workers = True

# ============
# train config 
# ============
# Learning rates (main knobs). ``optim["lr"]`` and, when a GAN discriminator is enabled,
# ``optim_discriminator["lr"]`` are filled from these after the discriminator preset runs.
learning_rate = 1e-4  # VAE / generator (AdamW). Try 2e-4 for single-sequence; lower if loss is spiky.
# Discriminator AdamW LR. ``None`` = keep the value from the discriminator preset (StyleGAN2 vs PatchGAN vs Train).
disc_learning_rate = None

optim = dict(
    cls="AdamW",
    lr=learning_rate,
    eps=1e-8,
    weight_decay=0.0,
    betas=(0.9, 0.98),
)
# Exponential decay: no need to know total training length.
# Every decay_steps optimizer steps the LR is multiplied by decay_factor (halved here).
# At ~50 update-steps/epoch: decay_steps=5000 ≈ every ~100 epochs the LR halves.
# Stops decaying once min_lr is reached and stays there indefinitely.
lr_scheduler = dict(
    warmup_steps=100,
    use_exponential_decay=True,
    decay_steps=5000,        # halve every 5000 optimizer steps (~100 epochs)
    decay_factor=0.5,        # × 0.5 per interval
    min_lr=learning_rate * 0.05,  # 5 % floor
)

mixed_strategy = None  # Disable mixed strategy - we only have video
mixed_image_ratio = 0.0

dtype = "bf16"
# Single-GPU: "none" removes ZeRO overhead (gradient bucketing, optimizer sharding, NCCL) that is
# pointless with one GPU. Switch to "zero2" for multi-GPU. Grad clipping uses torch.clip_grad_norm_
# automatically (train.py detects plugin="none" and sets force_manual_global_grad_clip=True).
plugin = "none"
# TorchDynamo backend used by accelerate launch in sweep scripts.
# Typical values: "no" (default, safest), "inductor" (try for speedups).
dynamo_backend = "no" # for deterministic training, otherwise flexible tiling/blocking strategies
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

grad_clip = 2.0
grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None  # Small dataset, don't need caching

seed = 42
# Best-effort reproducibility (cudnn, TF32, cuBLAS workspace, deterministic algos warn_only).
# Set False to restore cudnn benchmark / TF32 for throughput. Multi-GPU ZeRO + BF16 are still noisy.
deterministic = True
outputs = "outputs"
# Optional: fixed experiment folder name under outputs/ (else auto timestamp + config name)
# experiment_name = "cross_attn_lora_after_16_all_people_9t_2v_64p"
epochs = 100000 # 10000  # One epoch = one pass over ALL samples (.pt files). steps_per_epoch = num_samples // batch_size. Total steps = epochs × steps_per_epoch. (9 participants = many samples, not 9.)
log_every = 100 # 1000  # Log every 10 steps # CHANGE to 100
# Master switch: no checkpoint dirs written when False (periodic + final). Set True to save.
save_ckpt = True
# Save a checkpoint at the end of every epoch (if epoch-mean train PSNR is healthy).
# keep_n_latest controls retention.
# Keep this many latest checkpoint dirs (epochX-global_stepY); -1 = keep all
keep_n_latest = 5
ema_decay = 0.9999  # High EMA decay for stable fine-tuning

# ============
# Resume training
# ============
# To resume, pass the checkpoint subdir (contains running_states.json and model/) via command line:
#   python scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py --load /path/to/epoch0-global_step200
# Use the full path to a folder like .../outputs/<exp_name>/epoch0-global_step200 (not the experiment root).

# Wandb: charts use optimizer step (wandb.log step=...). Run names include a loss_signature slug
# (preset, perceptual/kl/view weights, discriminator_choice, gen_disc_weight). See train.py _loss_config_dict.
# If wandb_expr_name is set, the slug is appended after __. Metrics append to exp_dir/eval_metrics.jsonl.
wandb = True
wandb_project = "wan_multiview_vae"
# Optional run name override: keep None for current auto-generated naming logic.
wandb_expr_name = None #"gen_none__perc1p5__k1em6_256px"
# wandb_expr_name = "manual_override"
# Only call wandb.init after this many optimizer steps (avoids empty runs on short tests; resume past step inits immediately)
wandb_min_steps_before_init = 10
log_step_time = True  # Once: print avg wall time over the first 10 steps (set False to disable; uses tqdm.write)

# One-time design + parameter breakdown on the first training step (after ColossalAI booster wrap):
# fusion_mode comparison, LoRA before/after/viewwise vs embeddings, lora_rank notes, trainable %, LoRA buckets.
log_training_design_summary = True
# When log_training_design_summary is True, the shorter build-time overview is skipped unless you force it:
log_training_param_overview = True

update_warmup_steps = False  # No warmup needed

# ============
# loss config 
# ============
# Reconstruction / KL / perceptual weights for ``VAELoss`` (see ``opensora/models/vae/losses.py``).
# NLL = mean( (L1 + perceptual_loss_weight * LPIPS) / exp(logvar) + logvar ); KL = kl_loss_weight * KL(posterior||N(0,1)).
# ``view_consistency_weight`` is also used in ``train.py`` (MSE between consecutive recon views).
# Default ctor values in losses.py (perceptual 1.0, kl 5e-4) follow common LDM/VAE-GAN recipes; this file overrides them.
# They are Open-Sora ``VAELoss`` / training-script wiring—not DiffSynth scaler defaults.
vae_loss_preset = "default"  # Wandb tag only; does not change math unless you wire presets in train.py.
perceptual_loss_weight = 1.5 #default: 1.5 #0.5
kl_loss_weight = 1e-6 #5e-4, default: 1e-6
view_consistency_weight = 0.00#0.01
logvar_init = 0.0  # learned scalar in VAELoss; initial value

vae_loss_config = dict(
    perceptual_loss_weight=perceptual_loss_weight,
    kl_loss_weight=kl_loss_weight,
    view_consistency_weight=view_consistency_weight,
    logvar_init=logvar_init,
    # Resize frames to this fraction of original before feeding to LPIPS/VGG.
    # VGG features are scale-invariant: 0.5 gives equivalent perceptual gradient
    # at 4× lower cost. Use 1.0 at 128px (no benefit), 0.5 at 256px+.
    lpips_scale=0.5,
    # Process LPIPS in chunks of this many frames to avoid OOM at high resolution.
    # At 512px with scale=0.5: effectively 144 frames at 256px; chunk=64 is fine.
    # At 1024px with scale=0.5: 512px effective; keep chunk=32.
    lpips_chunk_size=64,
)

# ---- GAN discriminator (single switch) ----
# Set exactly one of:
#   None     — no adversarial discriminator
#   "Train"  — 3D PatchGAN from scratch (N_Layer_discriminator_3D); with disc_multiview_mode=flatten_batch + view_flatten_in_disc -> disc_multiview_mode=flatten_batch, view_flatten_in_disc=True
#   "TrainMultiview4D" — joint multi-view disc: both views + per-view embeddings, view-axis merge (N_Layer_discriminator_multiview_4d) -> disc_multiview_mode=joint_4d, view_flatten_in_disc=False
#   "TrainMultiviewStack" — stack views as channels [B,6,T,H,W] with standard 3D PatchGAN (disc_multiview_mode=stack_channels) -> disc_multiview_mode=stack_channels, view_flatten_in_disc=False
#   "StyleGAN2" — pretrained 2D StyleGAN (per-frame); needs disc_per_frame_2d (set by preset)
#   "PatchGAN" — LDM-style 2D NLayer PatchGAN, random init (HF sd-vae has no disc weights; trains with your data)
# You can also set ``discriminator`` to a full dict (MMEngine module cfg); preset fills GAN hyperparams + disc_per_frame_2d.
from opensora.utils.vae_discriminator_presets import apply_discriminator_bundle_to_cfg, resolve_vae_discriminator_bundle

# Keep this selector as a string so CLI overrides are type-safe:
#   --discriminator_choice none | Train | TrainMultiview4D | TrainMultiviewStack | StyleGAN2 | PatchGAN
discriminator_choice = None
# Optional override for generator adversarial loss weight (gen_loss_config.disc_weight).
# None -> keep the discriminator preset default; float -> override (e.g., 0.1).
gen_disc_weight = 0.1
# If True, train.py runs two extra torch.autograd.grad on the generator last layer each step (slow).
# Leave False unless debugging GAN gradient balance.
gan_log_adaptive_grad_metrics = False

apply_discriminator_bundle_to_cfg(globals(), resolve_vae_discriminator_bundle(discriminator_choice))

_optim_discriminator = globals().get("optim_discriminator")
if disc_learning_rate is not None and _optim_discriminator is not None:
    optim_discriminator = {**_optim_discriminator, "lr": disc_learning_rate}

_gen_loss_config = globals().get("gen_loss_config")
if gen_disc_weight is not None and _gen_loss_config is not None:
    gen_loss_config = {**_gen_loss_config, "disc_weight": float(gen_disc_weight)}
    # Keep compatibility with train.py's existing top-level post-bundle override path.
    sweep_gen_disc_weight = float(gen_disc_weight)

# ============
# Multi-view specific config
# ============
# Wan VAE decoder outputs [-1, 1]; scale dataset [0, 1] -> [-1, 1] so loss and vis are correct
vae_target_range = "[-1,1]"
view_flatten_in_loss = False  # CHANGED: Don't flatten views, preserve view dimension for better loss
# Discriminator input layout for multi-view (ignored when disc_per_frame_2d):
#   flatten_batch — [B*V,C,T,H,W] (needs view_flatten_in_disc=True) -> simplest; discriminator never sees both views of one clip in one forward; good baseline, weaker “pair” structure.
#   stack_channels — [B,V*C,T,H,W] (use discriminator TrainMultiviewStack or input_nc=V*3) -> both RGBs in channels; one 3D conv sees them together along C, not a dedicated view axis.
#   joint_4d — [B,V,C,T,H,W] into N_Layer_discriminator_multiview_4d (TrainMultiview4D preset) -> explicit view axis + per-view embeddings + first layer that mixes across V; strongest structural bias for “two views of one scene” (at the cost of a custom head).
disc_multiview_mode = "joint_4d"
view_flatten_in_disc = False  # keep 6D for joint_4d / stack_channels; True only for flatten_batch

# Auto-fix discriminator input layout for the single-view 3D PatchGAN preset.
# "Train" expects 5D [B,C,T,H,W], so flatten views into batch.
if str(discriminator_choice).lower() == "train":
    disc_multiview_mode = "flatten_batch"
    view_flatten_in_disc = True

# ============
# Evaluation config
# ============
# eval_every: cheap metrics on the *current training batch* (PSNR/SSIM on that batch) every N
# optimizer update steps. Scale with dataset size: single_sequence (1 step/epoch) → 1 or 5;
# all_people_one_expression (~30 steps/epoch, 1 optimizer update/epoch) → 1 or 10;
# all_people (many steps/epoch) → 100+.
# NOTE: epoch-end checkpoints are skipped until the first PSNR sample is available, so if
# eval_every is larger than the number of optimizer steps so far, no checkpoint is ever saved.
eval_every = 100  # 1 = evaluate on every optimizer update (low overhead for small datasets)
# full_eval_every: separate dataloader over val (or train) — mean/std over clips. Heavier.
full_eval_every = 500
# 0 = score every clip in the val (or train) holdout set (e.g. all _val_participants). N>0 = cap for quick tests.
eval_num_samples = 0
eval_batch_size = 1  # Batch size for evaluation
eval_use_ema = True  # Use EMA model for evaluation if available

# Final evaluation
final_eval = True  # Run final evaluation after training
# 0 = score every clip in each eval set (train + val). N>0 = cap for quick tests.
final_eval_num_samples = 0

# W&B / periodic log only: how many clips to show in reconstruction grids (not how many are scored in eval).
num_reconstruction_vis_samples = 3

# ============
# Performance logging & bottleneck / shape debugging
# ============
log_memory = False  # Set to True to log GPU memory usage (adds small overhead)
# Print timing every N steps (0 = off). Lower overhead when 0.
log_bottleneck_every = 0
# Print latent shapes [B,V,C,T,H,W] every N steps; also enables VAE-internal shape logs (0 = off).
log_latent_shapes_every = 0

# ============
# Debug / collapse monitoring
# ============
# Appends JSONL to outputs/<exp>/train_debug_stats.jsonl (master): loss + grad norms every
# ``debug_stats_every`` optimizer updates from ``debug_stats_start_step`` onward.
# Weight summaries are extra fields every ``debug_stats_weight_every`` updates only (heavier).
debug_stats_start_step = 0
debug_stats_every = 1
debug_stats_weight_every = 50

# Train-batch PSNR guard (same metric as wandb train_batch/psnr): evaluate at epoch level
# and stop after N consecutive epochs below threshold.
train_psnr_guard = False
train_psnr_guard_threshold = 15.0
train_psnr_guard_start_epoch = 0
train_psnr_guard_consecutive = 5
train_psnr_guard_min_epochs = 0

# Checkpoint policy: epoch-end saves are skipped when epoch-mean train PSNR is below threshold.

# ============
# Training steps and speed
# ============
# Samples = total .pt files (e.g. 400 videos − 10 val participants ≈ 390 train files).
# Steps per epoch = train_samples // batch_size (drop_last=True, so floor division).
# Gradient accumulation uses global_step, so it spans epoch boundaries — batch_size can be
# anything regardless of accumulation_steps (no longer constrained to batch_size × accum ≤ dataset).
# With batch_size=16 and ~390 train files: 24 steps/epoch; effective batch = 16 × 8 = 128.
# More speed (no code changes): raise batch_size if VRAM allows; discriminator_choice none;
# lower perceptual_loss_weight; fusion_mode conv3d vs cross_attention; log_bottleneck_every=0;
# eval_every=0; fixed_seq_eval_every_epochs=0; wandb=False for local runs.
batch_size = 64 #16 #16 #32  # raise if VRAM allows; effective batch = batch_size × accumulation_steps
accumulation_steps = 1 #4

# profile = True       # existing schedule-based profiler (TensorBoard trace, can't combine with profile_step)
profile_step = False   # Kineto trace: overwhelming JSON + op table (legacy; use profile_timing instead)
# User-friendly CUDA-synchronized block timing (attention-focused, readable txt + json).
# Runs once at profile_timing_step (after warmup). Disables wandb automatically.
profile_timing = True
profile_timing_step = 50  # global_step to profile (0-indexed loop counter)

# ---------- performance optimizations ----------
# Set optimization=True to enable all four optimizations at once:
#   1. torch.compile — fuses small ops, ~1.3-1.8x speedup after warmup.
#      Expect very slow first ~10 steps (compilation); then much faster. Recompilation
#      warnings are normal if batch shapes change.
#   2. channels_last_3d on model weights — skipped automatically when ema_decay is set
#   3. gradient checkpointing — reduces VRAM, allows larger batches
# "default": safe op-fusion, compatible with channels_last_3d strides.
# "reduce-overhead": uses CUDA graphs — stalls with channels_last_3d (non-contiguous strides).
# "max-autotune": most aggressive, even slower first step.
optimization = True
optimization_compile_mode = "reduce-overhead"