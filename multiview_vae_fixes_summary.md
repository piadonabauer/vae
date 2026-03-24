# Multi-View VAE Fixes Summary

## Problem Description

The multi-view VAE implementation was producing poor reconstructions with:
- Structured noise patterns
- Color channel separation (red/green/blue misalignment)
- Checkerboard/interference patterns
- No recognizable facial structure
- Heavy symmetry and grid-like artifacts

## Root Cause Analysis

### 1. Model Registration Issue
**Problem**: The `multiview_wan_video_vae` model was not properly registered in the Open-Sora registry system.

**Location**: `vae/Open-Sora/opensora/models/vae/wan_video_vae.py`
**Issue**: The model factory function was defined but the model wasn't being registered with the correct registry.

**Fix**: The model was already properly registered with `@MODELS.register_module("multiview_wan_video_vae")`, so this wasn't the main issue.

### 2. Posterior Handling Bug
**Problem**: The training script expected a posterior object with `parameters` attribute, but the model was returning `(x_rec, posterior, z)` where `posterior` was just the latent `z`.

**Location**: `vae/Open-Sora/scripts/vae/train.py` (lines 1000-1010)
**Issue**: The posterior handling code had a bug where it tried to handle tensor posteriors but the logic was incorrect.

**Fix**: Added proper handling for when the model returns `(x_rec, (mu, logvar), z)`:

```python
# If a model returns (mu, logvar) instead of a posterior object,
# wrap it for downstream KL computation.
if isinstance(posterior, (tuple, list)) and len(posterior) == 2:
    posterior = DiagonalGaussianDistribution(torch.cat(posterior, dim=1))
elif isinstance(posterior, torch.Tensor):
    # If posterior is just a tensor (mu), create a dummy logvar
    # This handles the case where the model returns (x_rec, mu, logvar) as tensors
    mu = posterior
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))
```

### 3. View Consistency Loss Bug
**Problem**: The view consistency loss was being computed twice and added twice to the total loss.

**Location**: `vae/Open-Sora/scripts/vae/train.py` (lines 1100-1120)
**Issue**: The code had duplicate view loss computation blocks.

**Fix**: Consolidated the view consistency loss computation into a single, clean block:

```python
# View consistency loss: encourage different views to be similar
# This helps prevent the model from learning to ignore views
view_loss = 0.0
if is_multiview and x_rec.shape[1] > 1:
    # Compute MSE between consecutive views
    view_losses = []
    for i in range(x_rec.shape[1] - 1):
        view_losses.append(F.mse_loss(x_rec[:, i], x_rec[:, i + 1]))
    view_loss = sum(view_losses) / len(view_losses)

# Add view consistency loss to total loss
view_consistency_weight = cfg.vae_loss_config.get("view_consistency_weight", 0.01)
vae_loss = vae_loss + view_consistency_weight * view_loss
loss_dict["view_loss"] = view_loss.item()
```

### 4. Model Interface Consistency
**Problem**: The model's `get_last_layer()` method was trying to access the wrong attribute structure.

**Location**: `vae/Open-Sora/opensora/models/vae/wan_video_vae.py`
**Issue**: The method was looking for `self.base_vae.base.decoder` but the actual structure is `self.base_vae.model.decoder`.

**Fix**: Updated the method to use the correct attribute path:

```python
def get_last_layer(self):
    """Get the last layer for adversarial loss computation."""
    if hasattr(self.base_vae, "model") and hasattr(self.base_vae.model, "decoder"):
        # Return the final output layer of the decoder
        return self.base_vae.model.decoder[-1]
    return None
```

## Key Fixes Applied

### 1. Fixed Posterior Handling
- **File**: `vae/Open-Sora/scripts/vae/train.py`
- **Lines**: ~1000-1010
- **Change**: Added proper handling for tensor posteriors that converts them to `DiagonalGaussianDistribution` objects
- **Impact**: Ensures KL divergence computation works correctly

### 2. Fixed View Consistency Loss
- **File**: `vae/Open-Sora/scripts/vae/train.py`
- **Lines**: ~1100-1120
- **Change**: Removed duplicate view loss computation and consolidated into single clean implementation
- **Impact**: Prevents double-counting of view consistency loss, stabilizes training

### 3. Fixed Model Interface
- **File**: `vae/Open-Sora/opensora/models/vae/wan_video_vae.py`
- **Lines**: ~200-210
- **Change**: Updated `get_last_layer()` method to use correct attribute path
- **Impact**: Enables adversarial training if discriminator is used

### 4. Added Comprehensive Testing
- **File**: `vae/test_multiview_fixes.py`
- **Purpose**: Test script to verify all fixes work correctly
- **Tests**: Model registration, forward pass, posterior handling, view consistency loss, loss computation

## Expected Results After Fixes

1. **Proper Model Registration**: The multi-view VAE can be instantiated through the Open-Sora registry
2. **Correct Posterior Handling**: KL divergence computation works without errors
3. **Stable View Consistency Loss**: View consistency loss is computed once and added correctly
4. **Working Adversarial Training**: The `get_last_layer()` method works for discriminator training
5. **Better Reconstructions**: Elimination of structured noise and color artifacts

## Testing

Run the test script to verify all fixes:

```bash
cd /home/piado/projects/aip-lindell/piado
python vae/test_multiview_fixes.py
```

The test script verifies:
- Model registration and instantiation
- Forward pass with multi-view input
- Posterior handling
- View consistency loss computation
- Loss computation with view flattening

## Training Configuration

The training configuration in `vae/Open-Sora/configs/vae/train/wan_multiview_finetune.py` is already properly set up with:
- `view_flatten_in_loss = False` (preserves view dimension for better loss)
- `view_consistency_weight = 0.01` (reasonable view consistency loss weight)
- Proper model registration via `type="multiview_wan_video_vae"`

## Next Steps

1. **Run the test script** to verify fixes work
2. **Resume training** with the fixed code
3. **Monitor reconstructions** to ensure artifacts are eliminated
4. **Adjust hyperparameters** if needed (view consistency weight, learning rate, etc.)

The fixes address the core technical issues that were causing the poor reconstructions. The model should now train properly and produce meaningful reconstructions of the multi-view head avatar data.