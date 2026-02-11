## Wan Multi-View VAE Design Notes

This document summarizes how to extend the Wan 3D Video VAE to multi-view
video (time + view), what to change, and the design/ablation space. It is
written to align with the existing Wan VAE code in `diffsynth/models/wan_video_vae.py`
and the Open-Sora VAE training loop in `Open-Sora/scripts/vae/train.py`.

### 1. What Needs To Be Modified

**Goal:** make the VAE operate on synchronized multi-view sequences and produce
a shared 4D latent that compresses time and view redundancy.

Minimum changes needed in the VAE:
- Add a view axis to the public interface: input shape `[B, V, C, T, H, W]`.
- Provide a view-aware latent mixing/compression stage.
- Allow optional view position embeddings (camera index).
- Keep the base 3D VAE blocks intact to reuse pretrained weights.

Training pipeline changes (outside this file):
- Dataset returns multi-view tensors `[B, V, C, T, H, W]`.
- Training loop stays mostly the same; loss is applied per view but can be
  averaged across views.
- Ensure the discriminator (if used) is compatible with multi-view layout
  (either run per-view or reshape into `[B*V, C, T, H, W]`).

### 2. What It Looks Like Today (3D VAE)

Current Wan VAE is a 3D video VAE that treats only time + space:

```
Input (video)
  x: [B, C, T, H, W]
        |
        v
  Encoder3d
  - CausalConv3d
  - ResidualBlock x N
  - (optional) AttentionBlock
  - Resample down (temporal + spatial)
        |
        v
  Conv1 -> split (mu, logvar)
        |
        v
  z: [B, Cz, T', H', W']
        |
        v
  Conv2
        |
        v
  Decoder3d
  - ResidualBlock x N
  - (optional) AttentionBlock
  - Resample up (temporal + spatial)
        |
        v
Output (video)
  x_hat: [B, C, T, H, W]
```

### 3. Multi-View Extension (Minimal Change Path)

The recommended minimal-change design keeps the pretrained 3D core and adds
view mixing in latent space. This is the approach implemented in the new
multi-view wrapper classes in `wan_video_vae.py`.

```
Input (multi-view video)
  x: [B, V, C, T, H, W]
        |
        v
  Reshape -> [B*V, C, T, H, W]
        |
        v
  Shared Encoder3d (pretrained)
        |
        v
  z_per_view: [B, V, Cz, T', H', W']
        |
        +------------------------------+
        | View Positional Embedding    |  (optional)
        +------------------------------+
        |
        v
  ViewCompressor (1x1 across V)
        |
        v
  z_shared: [B, Vc, Cz, T', H', W']
        |
        v
  ViewExpander (1x1 across V)
        |
        v
  z_per_view: [B, V, Cz, T', H', W']
        |
        v
  Shared Decoder3d (pretrained)
        |
        v
Output (multi-view video)
  x_hat: [B, V, C, T, H, W]
```

Why this path:
- Keeps pretrained 3D weights unchanged.
- Adds only view-axis linear layers for compression.
- Simple to ablate and extend with attention later.
- Directly matches the stated goal of a shared 4D latent.

### 4. Other Modification Options (With Pros/Cons)

**Option A: 4D Convolutions Everywhere (full 4D VAE)**
- Replace all 3D convs with 4D convs `(view, time, height, width)`.
- Pros: explicit 4D modeling.
- Cons: heavy compute/memory, harder to reuse pretrained weights, slow to train.
- Best for: full-scale training and when compute is not a constraint.

**Option B: Factorized (2+1+1)D Blocks**
- Split operations into spatial, temporal, and view sub-kernels.
- Pros: cheaper than full 4D conv, view modeling is explicit.
- Cons: larger refactor, more complex to initialize from pretrained weights.
- Best for: medium budget experiments when deeper view modeling is needed.

**Option C: View Mixing Only in Latent (minimal change)**
- Encode per view using shared 3D VAE, mix/compress views in latent with 1x1.
- Pros: smallest change, fast, stable, easy transfer from pretrained.
- Cons: view interactions only happen in latent, not early in encoder.
- Best for: first implementation, fast validation, ablation baseline.

**Option D: Add Cross-View Attention Between Frozen Blocks**
- Insert attention layers between existing residual blocks; init to zero.
- Pros: more expressive than linear mixing, still lightweight.
- Cons: changes runtime and may need careful tuning.
- Best for: second phase after baseline works.

**Recommended starting point:** Option C (latent view mixing). It is the most
stable and fastest to validate. Option D is the next upgrade if quality plateaus.

### 5. What To Choose Now

Given the goals and compute constraints, choose:
- **View compression in latent space** (Option C).
- **View positional embeddings** (learnable, small overhead).
- **View compression ratio**: start with `V=8 -> Vc=2` (4x).

### 6. Ablations To Run Later

Suggested ablations that are informative and low-cost:
- **Number of views**: `V = 2, 4, 8`.
- **View compression factor**: `2x, 4x, 8x`.
- **Placement of view compression**:
  - After encoder (current design).
  - Split: compress earlier vs later (if you add new blocks).
- **View positional embeddings**: on/off.
- **Cross-view attention**: add or omit (Option D).
- **Loss terms**:
  - Reconstruction only.
  - Reconstruction + reprojection consistency.
- **View mixing initialization**:
  - Average init (stable).
  - Random init (test for faster adaptation).

### 7. Training Notes (From Open-Sora Script)

The Open-Sora training loop is already compatible with video tensors; key
adaptations are in the dataset and model:
- Ensure dataset returns multi-view video in one tensor.
- In the training loop, reshape as needed for the discriminator and losses.
  For example, `x` and `x_rec` can be reshaped to `[B*V, C, T, H, W]` before
  applying losses if the loss expects 5D input.
- Keep the VAE loss unchanged; it works on tensors regardless of view axis.

### 8. Practical Sketches of Block Changes

**Current 3D VAE blocks:**

```
Encoder3d:
  CausalConv3d -> [ResBlock, (Attn)] x N -> Downsample (T/S) -> ...

Decoder3d:
  ResBlock -> (Attn) -> Upsample (T/S) -> ...
```

**New multi-view blocks (minimal change):**

```
MultiView wrapper:
  Reshape: [B, V, C, T, H, W] -> [B*V, C, T, H, W]
  Shared Encoder3d (frozen or finetune)
  ViewPosEmbed (optional)
  ViewCompressor (1x1 across V)
  ViewExpander (1x1 across V)
  Shared Decoder3d
```

**Potential future blocks (if Option D):**

```
Encoder3d (frozen) -> View-Attn -> Encoder3d (frozen)
Decoder3d (frozen) -> View-Attn -> Decoder3d (frozen)
```

### 9. Initial Experiment (Supervisor Request)

This is a concrete first experiment setup that is small, fast, and easy to
reason about while still testing cross-view compression.

**Inputs and scale**
- **Views**: start with `V=2`.
- **Resolution**: start with `128` or `256` to keep memory low.
- **Temporal window**: keep Wan defaults; do not change temporal stride yet.

**View compression target**
- **Two-step downsample in view axis**: `2 -> 1` (effectively a 2x view
  compression). This is the smallest non-trivial case and makes ablation easy.

**How to compress views (creative but stable options)**
- **Option 1: 1x1 view projection (baseline)**  
  Use `ViewCompressor` with `Conv1d(V -> Vc)` along view axis. This is stable
  and matches the current minimal-change design.
- **Option 2: Learnable strided view conv**  
  Implement a small `Conv1d` with stride=2 across view axis to mimic downsample,
  followed by a `1x1` to set channels. Slightly more expressive than Option 1.
- **Option 3: "View pooling + residual"**  
  Average-pool across views and add a residual projection of the per-view
  features (a small MLP or 1x1). Helps preserve identity while compressing.

**When to compress (placement)**
- **After Encoder3d (recommended first)**  
  Compress views after the temporal CausalConv3d stack. This isolates view
  modeling in latent space and avoids reworking temporal convolutions.
- **Before temporal CausalConv3d (ablate later)**  
  Compress early to reduce compute, but it can discard view-specific details
  before temporal modeling. Test only after the baseline works.

**Initial choice (recommended)**
- Start with **Option 1** and **post-encoder compression**, since it is the most
  stable and closest to the pretrained model behavior.

### 10. Figure: Initial Experiment Sketch

```
Input (2 views)                        Latent (1 view)
  x: [B, 2, C, T, H, W]                    z: [B, 1, Cz, T', H', W']
        |                                          ^
        | reshape                                  |
        v                                          |
  [B*2, C, T, H, W]                                |
        |                                          |
        v                                          |
  Encoder3d (shared, pretrained)                   |
        |                                          |
        v                                          |
  z_per_view: [B, 2, Cz, T', H', W']                |
        |                                          |
        +---- ViewCompressor (2 -> 1) -------------+
                         |
                         v
                 z_shared: [B, 1, Cz, T', H', W']
                         |
                         v
                 ViewExpander (1 -> 2)
                         |
                         v
                 Decoder3d (shared)
                         |
                         v
               x_hat: [B, 2, C, T, H, W]
```

### 11. Summary

Start with the minimal view-latent mixing design to validate correctness and
reconstruction quality quickly. It is the most compatible with pretrained Wan
weights, and it isolates view modeling to a small, ablatable module. Once
baseline quality is confirmed, add view attention or factorized view kernels
to improve cross-view consistency.
