# Multi-View Wan 2.1 VAE: Architecture and Training

This document describes the 4D (multi-view) extension of the Wan 2.1 3D Video VAE used for overfitting on NeRSemble-style multi-camera head sequences. It covers the architecture, tensor shapes, timing, and all training/loss parameters and their origins.

---

## 1. Architecture and Architectural Changes

### 1.1 Goal

Extend the Wan 3D Video VAE to a **4D representation** that encodes **x, y, time, and view** in a single latent. Input: two clips from two cameras (stereo). Output: **one** latent for the 3D head avatar. Temporal parts of the VAE stay frozen; view compression/expansion is learned.

### 1.2 Base Model: Wan 2.1 (DiffSynth)

- **Source:** DiffSynth-Studio `WanVideoVAE` / `VideoVAE_`.
- **Architecture:** 3D causal VAE with:
  - **Encoder:** `Encoder3d`, dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, temperal_downsample=[False, True, True]. Input RGB video [B, 3, T, H, W]; output latent [B, 16, T', H', W'] with T'<T, H'<H, W'<W (temporal and spatial downsampling).
- **Decoder:** Mirrors encoder (temporal upsampling [False, True, True]).
- **I/O range:** Decoder output is clamped to **[-1, 1]**; the pretrained model expects input and target in **[-1, 1]**.
- **Checkpoint:** Initialized from Wan 2.1 weights (`from_pretrained`). If the checkpoint stores the inner `VideoVAE_` state dict (keys `encoder.*`, `decoder.*`), weights are loaded into `base_vae.model` so the base encoder/decoder are actually pretrained.

### 1.3 Multi-View Wrapper (Open-Sora)

**Class:** `MultiviewWanVideoVAE` (Open-Sora), wrapping DiffSynth’s `WanVideoVAE`.

**Data flow:**

1. **Input:** `x` [B, V, C, T, H, W] with V=2 views, C=3, e.g. [1, 2, 3, 13, 128, 128].
2. **Per-view encode:** Each view is encoded independently with the **shared** base encoder (no view dimension inside the base). Temporal downsampling is unchanged and **frozen** when `freeze_temporal=True`.
   - Per-view latent: [B, Z, T', H', W'] with Z=16; stacked → [B, V, Z, T', H', W'].
3. **View embeddings (optional):** Add learnable per-view vectors to the stacked latent (`view_embedding`: [V, Z]) for view identity.
4. **View group fusion (if view_compression > 1):** For V=2, view_compression=2, the two views are merged into one in latent space via a 1×1×1 Conv3d: channels 2*Z → Z (e.g. 32→16). Initialized as average over the two views.
5. **Latent fusion:** 1×1×1 Conv3d from `view_out * Z` channels to Z (when view_out=1 this is 16→16). Initialized as average (no-op when view_out=1).
6. **Single latent:** z [B, Z, T', H', W'] — **one** 4D latent (x, y, time, view compressed).
7. **Decode:**  
   - **Latent expansion:** 1×1×1 Conv3d Z → V*Z (e.g. 16→32). Initialized so each of the V outputs is a **copy** of the fused latent (so the decoder sees non-zero input from step 0).  
   - Split to per-view latents [B, V, Z, T', H', W'].  
   - Each view is decoded with the **shared** base decoder (temporal upsampling unchanged, frozen when `freeze_temporal=True`).  
   - Output: [B, V, C, T, H, W], e.g. [1, 2, 3, 13, 128, 128].

**New parameters (trainable):**

- `view_group_fusion`: Conv3d(32, 16, 1) when view_compression=2, view_in=2.
- `latent_fusion`: Conv3d(16, 16, 1) (view_out=1).
- `latent_expand`: Conv3d(16, 32, 1).
- `view_embedding`: [2, 16] if `use_view_embedding=True`.

**Frozen (when `freeze_temporal=True`):** All base parameters whose name contains `"temporal"` (temporal downsampling/upsampling in the base VAE). Rest of the base encoder/decoder is trainable when `train_spatial=True`.

**Design choices:**

- **3D Conv for view:** View is compressed/expanded with 1×1×1 Conv3d (no spatial/temporal extent), so view is treated as “channel” dimension in latent space. No 3D conv over view axis; that can be added later if desired.
- **Copy init for expansion:** Expansion is initialized so each view gets a copy of z. Avoids zero input to the decoder at init (which produced constant/white reconstructions).
- **Input/target range:** Data is scaled from [0, 1] to **[-1, 1]** before the model so it matches the Wan decoder range and losses are correct.

---

## 2. Tensor Shapes (Parameter Sizes / Data Flow)

Shapes below match the logged “[encode] / [decode]” and “[step 900]” lines (batch size 1, 2 views, 13 frames, 128×128).

| Stage | Shape | Description |
|--------|--------|-------------|
| **Input** | [B, V, C, T, H, W] = [1, 2, 3, 13, 128, 128] | Two views, RGB, 13 frames, 128² |
| **After per-view encode** | [B, V, Z, T', H', W'] = [1, 2, 16, 4, 16, 16] | Temporal downsampling in base (13→4), spatial 128→16 |
| **View group fusion in** | [1, 1, 32, 4, 16, 16] | V_out=1, 2*Z=32 channels |
| **After view_group_fusion** | [1, 1, 16, 4, 16, 16] | 32→16 channels |
| **Latent fusion in** | [1, 16, 4, 16, 16] | V*Z=16 (V_out=1) |
| **Final latent z** | [B, Z, T', H', W'] = [1, 16, 4, 16, 16] | Single 4D latent |
| **After latent_expand** | [1, 32, 4, 16, 16] | Z→V*Z=32 |
| **Per-view latents (decode)** | [1, 2, 16, 4, 16, 16] | Same as after per-view encode |
| **Output** | [B, V, C, T, H, W] = [1, 2, 3, 13, 128, 128] | Reconstructed two-view video |

Spatial/temporal compression in the base VAE: 128→16 (8×), 13→4 (temporal). So the single latent has spatial size 16×16 and 4 time steps.

---

## 3. Timing (Bottleneck)

From the “[BOTTLENECK]” log (e.g. step 950), per-step times look like:

| Phase | Time (s) | Share | Notes |
|--------|----------|--------|--------|
| data_load | 0.002 | 0% | Negligible; not a bottleneck. |
| forward | 0.551 | 40% | Encode + decode (two views through base VAE twice each). |
| loss_compute | 0.021 | 2% | L1, LPIPS, KL, view loss. |
| backward | 0.717 | 52% | Largest share. |
| optimizer | 0.094 | 7% | Optimizer step. |
| **total_step** | **1.387** | 100% | ~23 min for 1000 steps at this rate. |

So the bottleneck is **backward** (~52%) and **forward** (~40%). To approach ~5 min for 1k steps you’d need to reduce cost (e.g. fewer frames, lower resolution) or add more GPUs.

---

## 4. Training and Loss Parameters

### 4.1 Where Defaults Come From

- **Model structure and base VAE:** DiffSynth Wan 2.1 (`VideoVAE_`: dim=96, z_dim=16, temperal_downsample=[False, True, True]).
- **Training script, loss wiring, optimizer/EMA/checkpoint logic:** Open-Sora VAE training pipeline.
- **Loss class:** Open-Sora `VAELoss` (Open-Sora); defaults below are from that class and from the reference config `video_dc_ae.py` where relevant.

### 4.2 Parameters and Why They Were Set

**Model (config `model`):**

| Parameter | Value | Origin / reason |
|-----------|--------|------------------|
| type | multiview_wan_video_vae | Our wrapper. |
| z_dim | 16 | Wan 2.1 default; must match checkpoint. |
| view_in | 2 | Two cameras. |
| view_compression | 2 | 2 views → 1 latent. |
| use_view_embedding | True | Learnable view identity in latent. |
| from_pretrained | /path/to/Wan2.1_VAE.pth | Initialize base from Wan 2.1. |
| freeze_temporal | True | Freeze temporal down/upsampling in base. |
| train_spatial | True | Train rest of base + new view layers. |

**Optimizer:**

| Parameter | Value | Origin / reason |
|-----------|--------|------------------|
| cls | AdamW | Open-Sora default; HybridAdam needs CUDA_HOME. |
| lr | 5e-4 | Higher than some ref configs for fine-tuning with small data. |
| weight_decay | 0.0 | No weight decay for this overfitting setup. |
| betas | (0.9, 0.98) | Standard. |

**LR scheduler:** `warmup_steps=0` (no warmup for overfitting).

**VAE loss (`vae_loss_config`) and preset:**

| Parameter | Our value (multiview) | Open-Sora VAELoss default | Reason |
|-----------|------------------------|---------------------------|--------|
| perceptual_loss_weight | 0.5 | 1.0 | Balances L1 and LPIPS; same as video_dc_ae. |
| kl_loss_weight | 5e-4 | 5e-4 | Keep small KL. |
| view_consistency_weight | 0.01 | 0.0 | Small weight so views don’t collapse to identical. |

To train with **Open-Sora default** VAE losses (perceptual 1.0, kl 5e-4, view_consistency 0), set in config:
`vae_loss_preset = "default"`. Leave unset or set `vae_loss_preset = "multiview"` to keep the current working weights above.

**How the loss is used:**

- **Reconstruction:** L1 between target and reconstruction (both in [-1, 1]).
- **Perceptual:** LPIPS on flattened frames; weight 0.5.
- **NLL (in loss):** `nll_loss = recon + perceptual_weight * perceptual`, then reweighted by learned logvar: `nll / exp(logvar) + logvar`; then mean. So effective reconstruction term is L1 + 0.5*LPIPS (with the logvar scaling).
- **KL:** `posterior.kl()` averaged, then multiplied by `kl_loss_weight` (5e-4).
- **View consistency (in train script):** MSE between consecutive view reconstructions `x_rec[:, i]` and `x_rec[:, i+1]`, averaged over consecutive pairs; multiplied by `view_consistency_weight` (0.01) and added to the total VAE loss. So total VAE loss = `nll_loss + kl_loss + 0.01 * view_loss`. This is in addition to whatever `VAELoss` returns (we do not add `ret["view_consistency_loss"]` again; the script’s view term is the one used).

**Multi-view / data:**

| Parameter | Value | Reason |
|-----------|--------|--------|
| vae_target_range | "[-1,1]" | Match Wan decoder; scale data from [0,1] to [-1,1]. |
| view_flatten_in_loss | False | Keep [B, V, C, T, H, W]; loss reshapes to [B*V, C, T, H, W] internally. |
| view_flatten_in_disc | True | Discriminator (if used) sees flattened views. |

**Other training:**

| Parameter | Value | Reason |
|-----------|--------|--------|
| dtype | bf16 | Match Open-Sora VAE training. |
| plugin | zero2 | ZeRO-2 for memory. |
| grad_clip | 1.0 | Gradient clipping. |
| ema_decay | 0.9999 | EMA of model for logging; eval uses booster model (not EMA) to avoid sharded-param issues. |
| ckpt_every | 200 (or 500) | Save checkpoints periodically so runs can be resumed. |
| keep_n_latest | 3 | Limit number of checkpoint dirs. |

**Data:** PT dataset, single .pt file, repeat=1500; bucket 128×128, 13 frames; batch_size=1.

---

## 5. Summary

- **Architecture:** Wan 2.1 3D VAE (DiffSynth) wrapped so that two views are encoded independently, fused in latent (view_compression=2 → one latent), then expanded back to two views and decoded. New layers: view_group_fusion, latent_fusion, latent_expand, optional view_embedding. Temporal layers in the base VAE are frozen.
- **Shapes:** Input/output [1, 2, 3, 13, 128, 128]; single latent [1, 16, 4, 16, 16].
- **Time:** ~1.39 s/step, ~23 min for 1k steps; bottleneck is backward then forward.
- **Loss:** L1 + 0.5×LPIPS (via NLL with logvar) + 5e-4×KL + 0.01×view consistency (MSE between reconstructions of consecutive views). Input/target in [-1, 1]; no discriminator in the current config.

All of this is consistent with the code in Open-Sora (`opensora/models/vae/wan_video_vae.py`, `scripts/vae/train.py`) and the config `configs/vae/train/wan_multiview_finetune.py`.
