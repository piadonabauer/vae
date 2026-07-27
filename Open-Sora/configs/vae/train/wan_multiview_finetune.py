# ============
# Fine-tuning config for Wan 2.1 VAE on NeRSemble multi-view data
# ============
# Multi-sequence (16 samples): use batch_size=1 + lower LR for stability.
# If output stays white and loss is spiky: (1) Use batch_size=1, lr=2e-4, vae_loss_preset="multiview".
# (2) Verify single-sequence still works (one .pt, repeat=1, epochs=100) to confirm checkpoint/range.
# (3) Ensure all .pt files have same shape [V,C,T,H,W] and value range [0,1].
# batch_size must be <= num_samples or steps_per_epoch is 0.

# ── Speed preset ──────────────────────────────────────────────────────────────
# FAST_MODE = True  → "reduce-overhead" (CUDA graphs) + decoder checkpoint OFF.
#   CUDA graphs capture forward+backward as one static sequence. Activations are
#   static buffers read in-place — only one view's activations live at a time (~8 GB)
#   instead of all 4 views simultaneously (~34 GB). This is how batch=64 fit in 46 GB.
#   Requires dynamic=False (static shapes) and recompile_limit≥32 (decoder has 11+
#   distinct ResidualBlock shapes; default limit=8 caused eager fallback → OOM).
#   First step is very slow (graph capture + 11+ compilations); then ~2× faster than
#   no_compile per the May benchmark (2.89 s/step vs 5.66 s/step at batch=1).
# FAST_MODE = False → "default" Triton fusion + per-frame decoder checkpoint (safe).
#   Use if FAST_MODE=True still has issues (recompile/graph errors on first step).
FAST_MODE = False


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
    from_pretrained="/home/piado/scratch/Wan2.1_VAE.pth", #,h",
    # Crossview path: freeze the ENTIRE pre-fusion encoder (encoder.conv1 +
    # downsamples, spatial AND temporal convs) to reuse the pretrained Wan
    # features. train_spatial=False -> freeze spatial too; freeze_temporal kept
    # True for the (now redundant) temporal-only case. These now actually apply
    # on the crossview path (previously they were only printed).
    freeze_temporal=True,
    train_spatial=False,
    # New cross-view encoder with in-encoder fusion + LoRA
    use_crossview_encoder=True,  # use MultiViewVideoVan instead of latent_fusion path
    # Fusion mode options (only used when use_crossview_encoder=True):
    # - "cross_attention": all-to-all cross-attn between every view and all views
    # - "self_attention": joint self-attn over all tokens from both views (length 2*N), then concat + ResBlocks
    # - "conv3d": concat views in channels, 1×1×1 Conv3d -> GroupNorm+SiLU -> two FusionResidualBlock3d (symmetric Conv3d)
    # conv4d
    fusion_mode="cross_attention",
    # Temporal compression: when True the cross-view encoder/decoder run Wan's
    # chunked feat_cache path so the temporal stride-convs actually fire
    # (4x: 9 frames -> 3 latent frames -> 9 frames out). False = legacy (T kept = 9).
    temporal_compression=True,
    # Real activation checkpointing inside the cross-view encoder/decoder
    # (per-view down-path + per-view decoder). Large memory cut, ~20-30% slower.
    # Leave False first; flip True only if you still OOM at the batch size you want.
    crossview_grad_checkpoint=True,  # REQUIRED: profiling showed encode.downsample_all_views holds +39GB at batch 8 (un-checkpointed per-view encoder, T=9). This is the ONLY checkpoint switch this VAE honors (grad_checkpoint=True is a no-op here); it checkpoints each view's encode + the decode body, freeing forward activations and recomputing in backward.
    # Per-stage override of crossview_grad_checkpoint (None -> follow the master flag above).
    # EMPIRICAL (260629 profiling): turning the DECODER checkpoint OFF to save the ~1.75s
    # backward recompute OOMs even at batch 16 (>94.8 GB) — the full-res upsample-resblock
    # activations across 4 views x 9 frames are simply too large to retain. The decoder MUST
    # stay checkpointed at any useful batch size, so both stages are left checkpointed here.
    # (Plumbing kept so you can flip the decoder off only if you ever drop to batch <= 8.)
    crossview_grad_checkpoint_encoder=False,
    crossview_grad_checkpoint_decoder=None,
    # Heads for the cross-view fusion attention (None -> auto, head_dim ~64 so
    # SDPA stays on Flash/mem-efficient backends; critical at 512px).
    view_attn_num_heads=None,
    use_lora=True,               # masfter switch for LoRA modules in crossview path
    use_lora_before=False,       # LoRA on Encoder3d stem before fusion (encoder.conv1 + downsamples); additive to use_lora_after
    use_lora_after=True,         # apply LoRA to bottleneck/decoder ("later" part)
    # Replace additive per-view latent embedding in decoder with view-specific latent LoRA adapters.
    use_viewwise_decoder_lora=True,
    lora_rank=32,                # configurable LoRA rank for all LoRA adapters
    # Phase 2 (optional): train original decoder conv/attn inside DiffSynth LoRA wrappers (base weights),
    # while keeping view-conditioned decode (view_idx + nn.Embedding in AttentionMultiViewVideoVan).
    # Still set use_viewwise_decoder_lora / use_view_embedding in the legacy path as needed; crossview always has decode embeddings.
    full_finetune_decoder=False,
    # -----------------------------------------------------------------------
    # Temporal-compression quality flags — round 2 (all False = exact baseline)
    # -----------------------------------------------------------------------
    # Idea 1 — Non-causal full-sequence decode: remove the chunk loop entirely.
    #   All T' latent frames are decoded in ONE pass; every decoder CausalConv3d
    #   switches from all-past to symmetric past+future temporal padding (runtime
    #   switch on the padding buffer, pretrained weights untouched). No cold-start
    #   frame 0, no chunk boundaries. Memory profile ≈ temporal_compression=False.
    #   Requires temporal_compression=True.
    use_noncausal_decode=False,
    # Idea 2 — Temporal reflection padding: prepend 4 reflected real frames
    #   [f4,f3,f2,f1] before encoding (keeps T ≡ 1 mod 4 → one extra leading latent
    #   frame), decode, crop the first 4 output frames. Encoder AND decoder caches
    #   warm up on genuine video statistics instead of a cold cache. Zero new
    #   parameters, ~10% extra compute. Requires temporal_compression=True.
    use_temporal_reflection_pad=False,
    # Idea 3 — High-frequency temporal side channel: tiny full-temporal-rate latent
    #   (side_channel_dim ch, spatial /16) from the view-averaged input, injected
    #   into the decoder right after the last temporal upsample via a zero-init
    #   1×1×1 conv. Carries the temporal detail (blinks, fast motion) that 4×
    #   temporal compression discards; the main 16ch @ T' latent stays untouched.
    #   Works with any decode mode (causal chunked, non-causal, tc=False).
    use_temporal_side_channel=False,
    side_channel_dim=4,                 # channels of the side latent
    # Idea 4 — Temporal attention in the decoder bottleneck: temporal counterpart of
    #   the spatial middle AttentionBlock, attending across frames at 384 channels
    #   (decoder feature space, not the 16ch latent). Zero-init projection.
    #   REQUIRES use_noncausal_decode=True (chunked decode shows the middle block
    #   one frame at a time — nothing to attend over).
    use_decoder_temporal_attention=False,
    # Idea 6 — Learned ConvGRU cache updater: replaces the hand-coded feat_cache
    #   update ("overwrite with the last 2 frames of activations") with a gated
    #   learned update cache = GRU(cache, activations) per decoder cache slot.
    #   Near-identity at init (update gate bias -4). Only for the causal chunked
    #   path — mutually exclusive with use_noncausal_decode.
    use_learned_cache_update=False,
    # After --load of a TC ckpt: re-randomize ViewAttention / JointViewAttention
    # (proj stays zero-init identity; QKV random). Used by fusion-adapt sweeps.
    reinit_view_attention_after_load=False,
)

# Idea 5 — Teacher distillation (training-side; no architecture change).
# Path to a temporal_compression=False checkpoint (a .pt/.pth/.safetensors state
# dict, or a checkpoint dir like .../epochX-global_stepY containing model/).
# The teacher is built with the same model config but temporal_compression=False,
# frozen, and an extra loss distill_weight * L1(student_recon, teacher_recon) is
# added — a dense per-frame signal for what good temporal reconstruction looks
# like. None = disabled (baseline).
distill_teacher_ckpt = None
distill_weight = 1.0

# ============
# data config 
# ============
from opensora.utils.nersemble_bucket import resolve_nersemble_bucket

# Optional: parent of ``64-res`` / ``128-res`` (default: NeRSemble v2 processed root).
nersemble_processed_base = "/datasets/lindell-proj/neumayr/nersemble_v2/processed/"
# "/home/coder/nersemble-data/processed/4view"

# ``DATA_ROOT``, ``train_target_hw``, ``train_target_frames`` are derived from ``bucket_config``:
# - ``256px_...`` + T frames → load ``.../256-res``, train at 256×256, T frames (e.g. 9).
# - ``128px_...`` + T frames → load ``.../128-res``, train at 128×128, T frames.
# - ``64px_...`` + ≤9 frames → ``.../64-res``; + >9 frames → ``128-res`` + on-the-fly downsample to 64.
bucket_config = {
    # 512px: loads ``<processed_base>/512-res`` (falls back to 128-res + on-the-fly
    # resize if 512-res is absent). 9 frames -> 3 latent frames with temporal_compression.
    # "512px_ar1:1": {9: (1.0, 1)},   # phase 2: finetune at 512px (batch 2)
    "128px_ar1:1": {9: (1.0, 1)},     # phase 1: train at 128px (batch 32)
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
data_preset = "single_sequence" #"single_sequence" #"all_people_one_expression"
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
learning_rate = 5e-4  # VAE / generator (AdamW). Try 2e-4 for single-sequence; lower if loss is spiky.
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
# LR scheduler DEACTIVATED: warmup_steps=0 + no cosine + no exponential decay makes
# create_lr_scheduler() return None -> constant LR (= learning_rate) for the whole run.
lr_scheduler = dict(
    warmup_steps=0,
    use_exponential_decay=False,
    decay_steps=5000,        # unused while use_exponential_decay=False
    decay_factor=0.5,        # unused
    min_lr=learning_rate * 0.05,  # unused
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
deterministic = False
outputs = "outputs"
# Optional: fixed experiment folder name under outputs/ (else auto timestamp + config name)
# experiment_name = "cross_attn_lora_after_16_all_people_9t_2v_64p"
epochs = 100000 # 10000  # One epoch = one pass over ALL samples (.pt files). steps_per_epoch = num_samples // batch_size. Total steps = epochs × steps_per_epoch. (9 participants = many samples, not 9.)
# Escalating log schedule (handled by should_log_update in train.py): log at these early
# update steps, then every `log_every` updates after the last one. Each update ~= 20s wall
# (accumulation_steps microbatches x ~10s), so [3,9,15] + log_every=21 => roughly
# 1 min, 3 min, 5 min, then every ~7 min. Set log_schedule_steps=None for plain modulo.
log_schedule_steps = [3, 9, 15]
log_every = 21  # steady interval (in update steps) after the last log_schedule_steps point (~7 min)
# Master switch: no checkpoint dirs written when False (periodic + final). Set True to save.
save_ckpt = True
# Save a checkpoint at the end of every Nth epoch (if epoch-mean train PSNR is healthy).
# At ~5 update-steps/epoch every-epoch saving is wasteful churn; raise this.
# 1 = save every epoch (legacy). N = save only when epoch % N == 0 (plus the final save).
save_every_n_epochs = 25
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
wandb_min_steps_before_init = 2  # init wandb early (~40s) so the first scheduled log at update 3 (~1 min) reaches wandb
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
eval_every = 20  # >0 just enables the per-batch PSNR/SSIM JSONL snapshot; its cadence now follows the log schedule (should_log_update), not this modulo. Set to 0 to disable the snapshot.
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
# 512px is ~16x the spatial activations of 128px. Start small and RAMP UP until just
# before OOM (watch nvidia-smi / the log_memory output). 64 was a 128px value and will
# OOM instantly at 512. With temporal_compression + multi-head fusion attention you
# should be able to push this up on the 96GB card; use accumulation_steps for a larger
# effective batch without more memory.
batch_size = 16  # 128px, temporal_compression=False + no crossview ckpt + reduce-overhead CUDA graphs is memory-heavy; 32 OOMs. Ramp up toward max if it fits. Phase 2 (512px): set to 2.
accumulation_steps = 2  # raise for a larger EFFECTIVE batch with no extra memory

# profile = True       # existing schedule-based profiler (TensorBoard trace, can't combine with profile_step)
profile_step = False # CHANGE   # Kineto trace: overwhelming JSON + op table (legacy; use profile_timing instead)
# User-friendly CUDA-synchronized block timing (attention-focused, readable txt + json).
# Runs once at profile_timing_step (after warmup). Disables wandb automatically.
profile_timing = False # CHANGE  # MUST stay False for wandb: train.py force-disables wandb when this is True
profile_timing_step = 50  # global_step to profile (0-indexed loop counter)
# Live per-block GPU memory printing from step 0 (prints "[mem] <block> delta/cur/peak/reserved").
# Use to locate an OOM/crash: the LAST "[mem]" line before the traceback is the offending block.
# Also adds a MEMORY BREAKDOWN table to the profile_timing report (net MB retained per method).
# REQUIRES optimization=False to be meaningful (torch.compile breaks per-block attribution).
# Turn OFF for real training (per-block cuda.synchronize adds overhead).
profile_memory_live = False # CHANGE # NOT COMPATIBLE WITH REDUCE-OVERHEAD

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
#optimization = True
optimization = False
# Bring-up at 512px on "default" first: it isolates real OOM from compile/CUDA-graph
# issues and tolerates the dynamic per-view / per-frame decode loops. Once a run is
# stable and you've found your max batch, try "reduce-overhead" (CUDA graphs) for speed
# — but expect to LOWER batch_size, since the graph memory pool adds pressure and the
# dynamic temporal-chunk shapes (1-frame vs 4-frame) force multiple captured graphs.
optimization_compile_mode = "default"  # "default": Triton op-fusion, no CUDA graphs. Compatible with dynamic temporal loops (range(T'), range(iter_)) and gradient checkpointing. "reduce-overhead" (CUDA graphs) hangs with temporal_compression loops — dynamic shapes cause infinite sympy pow_by_natural loops in Dynamo — and also crashes backward with "accessing tensor output of CUDAGraphs that has been overwritten".
# None -> torch auto-detects (legacy). True -> one shape-flexible graph: avoids the
# per-chunk recompiles from the temporal feat_cache loop and stays compatible with
# gradient checkpointing (no CUDA graphs in "default" mode). Try with mode="default".
optimization_compile_dynamic = True

# ── FAST_MODE overrides (applied last so they win over everything above) ──────
if FAST_MODE:
    # CUDA-graph compile ("reduce-overhead") + decoder checkpoint OFF.
    # WHY reduce-overhead saves memory (not obvious): CUDA graphs capture forward+backward
    # as one static sequence. Activations are static buffers accessed in-place — the graph
    # doesn't need to keep all 4 views × 9 frames simultaneously. Memory ≈ one view's
    # activations (~8 GB) instead of all four (~34 GB). This is how batch=64 fit in 46 GB.
    #
    # Two things broke it since the May benchmark:
    #   1. ResidualBlock.forward has 11 distinct channel shapes in the current decoder but
    #      the dynamo recompile_limit defaults to 8 → shapes 9-11 fall back to eager mode
    #      → those sub-graphs don't get CUDA graph coverage → activations accumulate → OOM.
    #      Fix: raise recompile_limit to 32.
    #   2. dynamic=True + CUDA graphs → pow_by_natural sympy loop in GroupNorm ops.
    #      Fix: use dynamic=False (fixed batch size, no symbolic shapes needed).
    model["temporal_compression"] = False
    model["crossview_grad_checkpoint_decoder"] = False  # compile reduces per-frame activation size enough
    optimization = True
    optimization_compile_mode = "default"           # Triton op-fusion; reduce-overhead (CUDA graphs) OOMs
                                                    # at batch>=16 even during capture (peak=~43GB, no headroom)
    optimization_compile_dynamic = True             # flexible graph across varied decoder upsampling shapes
    profile_memory_live = False                     # profiling overhead not useful during fast training