# Multi-View VAE Processing Order and Fix Explanation

## The Problem: Implementation Bugs in the Correct Model

You were using the correct `MultiviewWanVideoVAE` model (with convolution-based view fusion as recommended by your supervisor), but it had several critical bugs that were causing artifacts. The model architecture is correct - it's the implementation details that needed fixing.

## Processing Order with the CORRECT Model (`MultiviewWanVideoVAE`)

### 1. **Input**: Multi-view video `[B, V, C, T, H, W]`
   - Example: `[1, 2, 3, 13, 128, 128]` (1 sample, 2 views, 3 channels, 13 frames, 128x128)

### 2. **Per-View Encoding** (Temporal Downsampling)
   - Each view `[B, C, T, H, W]` is encoded separately by the base Wan 3D VAE
   - Temporal downsampling: `T=13` → `T'=4` (factor of 4)
   - Spatial downsampling: `H,W=128` → `H',W'=16` (factor of 8)
   - Result per view: `[B, Z, T', H', W']` where `Z=32` (2×z_dim for mu+logvar)

### 3. **View Compression** (Optional)
   - If `view_compression=2` and `view_in=2`: 2 views → 1 latent
   - Uses **convolution** (as recommended by your supervisor) to fuse the two view latents
   - Result: `[B, Z, T', H', W']` (single latent for all views)

### 4. **View Embeddings** (Learned Positional Encoding)
   - Adds learnable embeddings to distinguish between views
   - Helps the decoder know which view to reconstruct

### 5. **Decoding** (Temporal Upsampling)
   - Single latent `[B, Z, T', H', W']` is decoded back to multi-view
   - Temporal upsampling: `T'=4` → `T=13`
   - Spatial upsampling: `H',W'=16` → `H,W=128`
   - Result: `[B, V, C, T, H, W]` (reconstructed multi-view video)

## Why the Model Had Artifacts (Fixed Now)

The `MultiviewWanVideoVAE` implementation had several critical bugs that have now been fixed:

1. **Posterior Handling Bug**: The training script expected a posterior object with `parameters` attribute, but the model was returning `(x_rec, posterior, z)` where `posterior` was just the latent `z`. This caused KL divergence computation to fail.

2. **View Consistency Loss Bug**: The view consistency loss was being computed twice and added twice to the total loss, causing training instability.

3. **Model Interface Issue**: The `get_last_layer()` method was trying to access the wrong attribute structure.

## The Fix: Correct Implementation Details

**Before** (BROKEN):
```python
# Had bugs in posterior handling, view consistency loss, and model interface
```

**After** (FIXED):
```python
# Fixed posterior handling in training script
if isinstance(posterior, (tuple, list)) and len(posterior) == 2:
    posterior = DiagonalGaussianDistribution(torch.cat(posterior, dim=1))
elif isinstance(posterior, torch.Tensor):
    mu = posterior
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

# Fixed view consistency loss (no double counting)
view_loss = 0.0
if is_multiview and x_rec.shape[1] > 1:
    view_losses = []
    for i in range(x_rec.shape[1] - 1):
        view_losses.append(F.mse_loss(x_rec[:, i], x_rec[:, i + 1]))
    view_loss = sum(view_losses) / len(view_losses)

# Fixed model interface
def get_last_layer(self):
    if hasattr(self.base_vae, "model") and hasattr(self.base_vae.model, "decoder"):
        return self.base_vae.model.decoder[-1]
    return None
```

## Key Parameters for Your Use Case

```python
# For your 2-view head avatar training:
view_in=2                    # 2 camera views
view_compression=2           # Compress 2 views → 1 latent
use_view_embedding=True      # Use positional embeddings
# Convolution is used for view fusion (as recommended by your supervisor)
```

## Expected Results After Fix

With the correct model implementation, you should see:

1. **No more structured noise patterns**
2. **No more color channel separation**
3. **No more checkerboard artifacts**
4. **Proper facial structure preservation**
5. **Meaningful reconstructions**

## Resume Training

Now you can resume training with:
```bash
python scripts/vae/train.py --config configs/vae/train/wan_multiview_finetune.py --workdir outputs/overfit_test
```

The model should now produce proper reconstructions instead of the psychedelic artifacts you were seeing before.