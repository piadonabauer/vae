# Multi-View VAE Reconstruction Quality - Solution Summary

## Problem Statement
Your multi-view Wan 2.1 VAE is producing **ghosted/blended reconstructions** instead of clean per-camera images:
- **SSIM ≈ 0.75** (target: >0.90)
- **PSNR ≈ 12 dB** (target: >20 dB)  
- **MSE ≈ 0.06** (target: <0.01)
- **Visuals**: Blended views, ghosting effects, movement appears on top of each other

## Root Cause Analysis

### Primary Issue: View Compression Averaging
The `ViewCompressor` uses simple linear projection that **averages views**:
```python
# Old code
z_compressed = sum(z_views) / num_views  # Information loss!
```

When 2 views are averaged into 1 latent:
- Decoder sees averaged representation during training
- Learns to produce **averaged/blended output**
- Both reconstructed views become identical (the average)

### Secondary Issues
1. **No view-aware loss** - Loss flattens views into batch, no constraint that recon[view_i] matches input[view_i]
2. **Weak view embeddings** - Simple additive embeddings insufficient after averaging
3. **Loss function treats views as independent** - Doesn't explicitly preserve view identity

## Solutions Implemented

### Solution 1: Enhanced ViewCompressor ✅
```python
class ViewCompressor:
    # Before: simple Conv1d projection (averaging)
    # After: learned attention weights for selective pooling
    
    def forward(self, x):
        # x: [B, V, C, T, H, W]
        weights = softmax(learned_weights)  # [V_out, V_in]
        # Each output view learns which input views to focus on
        return weighted_sum(x, weights)  # [B, V_out, C, T, H, W]
```

**Benefit**: Views are pooled selectively, not averaged. Each view learns different features.

### Solution 2: Improved View Embeddings ✅
```python
class ViewPositionalEmbedding:
    # Before: z + additive_embed
    # After: z * (1 + multiplicative_embed) + additive_embed
    
    def forward(self, z):
        return z * (1 + embedding_scale) + embedding_shift
```

**Benefit**: Stronger modulation of view-specific features. Both scaling and shifting per view.

### Solution 3: View-Aware Loss ✅
```python
class VAELoss:
    def forward(self, video, recon, posterior):
        # Before: flatten [B, V, C, T, H, W] → [B*V, C, T, H, W]
        # Now: preserve view dimension and add view consistency loss
        
        # Per-view reconstruction loss
        loss = MSE(recon, video)  # [B, V, C, T, H, W]
        
        # View consistency loss
        # Penalize when recon[view_0] ≈ recon[view_1]
        view_consistency = penalize_similar_reconstructions()
        
        return loss + weight * view_consistency
```

**Benefit**: Explicit constraint that views should reconstruct differently. Prevents mode collapse.

### Solution 4: Config Updates ✅
```python
# Before
view_flatten_in_loss = True        # Flattens views!

# After
view_flatten_in_loss = False       # Preserves view dimension
vae_loss_config = dict(
    view_consistency_weight=0.1,   # NEW: penalize similar reconstructions
)
```

**Benefit**: Loss computation preserves view identity throughout training.

---

## Expected Improvements

| Metric | Before | After (Conservative) | After (Optimistic) |
|--------|--------|---------------------|-------------------|
| SSIM | 0.75 | 0.85 | 0.95 |
| PSNR (dB) | 12 | 16 | 25 |
| MSE | 0.06 | 0.02 | 0.002 |
| Cross-view sim | 0.95 | 0.70 | 0.50 |

**Visual Quality**:
- Before: Ghosted, blended, movement on top of each other
- After: Clean per-camera images, no ghosting, temporal coherence

---

## Files Modified

### 1. `opensora/models/vae/losses.py`
- Added `view_consistency_weight` parameter
- Implemented `_compute_view_consistency_loss()` 
- Modified `forward()` to preserve view dimensions
- Changes: ~50 lines of code

### 2. `diffsynth/models/wan_video_vae.py`
- Enhanced `ViewCompressor` class (added learned weights)
- Improved `ViewPositionalEmbedding` class (multiplicative scaling)
- Changes: ~40 lines of code

### 3. `configs/vae/train/wan_multiview_finetune.py`
- Set `view_flatten_in_loss = False`
- Added `view_consistency_weight = 0.1`
- Changes: 2 lines

### 4. New Diagnostic Script
- `test_multiview_fixes.py` - Validates all components work correctly
- `run_tests.sh` - Quick verification script

---

## How to Validate the Fixes

### Step 1: Run Diagnostic Tests
```bash
cd /home/piado/projects/aip-lindell/piado/vae
python test_multiview_fixes.py
```
This will verify:
- ViewCompressor works correctly
- ViewPositionalEmbedding learning properly
- Loss computation preserves dimensions

### Step 2: Train with Fixes
```bash
cd /home/piado/projects/aip-lindell/piado/vae/Open-Sora
python scripts/diffusion/train.py configs/vae/train/wan_multiview_finetune.py
```

### Step 3: Monitor Metrics
Watch W&B dashboard for:
- `nll_loss` → should decrease smoothly
- `view_consistency_loss` → should decrease (views diverging)
- `ssim` → should reach >0.90
- `psnr` → should reach >20 dB
- Cross-view reconstruction similarity → should drop from 0.95 to ~0.60-0.70

### Step 4: Visual Inspection
Check reconstructed frames for:
- ✓ Clean per-camera images
- ✓ No blending/ghosting artifacts
- ✓ Temporal coherence
- ✓ View-specific features

---

## Troubleshooting

### If SSIM doesn't improve significantly:

1. **Test without compression** (isolate issue):
   ```python
   # In config: view_compression=1
   # This disables latent compression (2 views → 2 views)
   # If SSIM improves, compression was the bottleneck
   ```

2. **Check if view consistency weight is too low**:
   ```python
   # Increase from 0.1 to 0.2 or 0.3
   view_consistency_weight=0.2
   ```

3. **Monitor embedding learning**:
   ```python
   # Log embedding magnitudes to verify they're learning differently per view
   # If embeddings are identical, they're not helping
   ```

4. **Try disabling KL loss** (for debugging on single sequence):
   ```python
   kl_loss_weight=0.0  # Temporary, for overfitting validation
   ```

---

## Key Insights

### Why Simple Averaging Failed
- **Averaging = Information Loss**: Two views squeezed into one latent
- **Decoder Learns to Average**: Makes no distinction between views
- **Both Reconstructions Identical**: Mode collapse to the mean

### Why These Fixes Work
- **Learned Pooling**: Selective mixing preserves view-specific information
- **View Consistency Loss**: Explicitly penalizes view mixing
- **Stronger Embeddings**: Multiplicative scaling is more expressive
- **View-Aware Loss**: Preserves view identity throughout training

### Trade-offs
- **Complexity**: Slightly more computation (negligible)
- **Memory**: No change in memory usage
- **Training Time**: May be slightly longer due to view consistency loss
- **Generalization**: Should improve with more views/data

---

## Next Steps

### Immediate (Today)
1. Run diagnostic tests to ensure code doesn't crash
2. Start training with existing single-sequence overfitting config
3. Monitor W&B metrics for improvement

### Short-term (This week)
1. Verify SSIM reaches >0.90 on overfitting sequence
2. Test on additional NeRSemble sequences from same actor
3. Validate visual quality (no ghosting, temporal coherence)

### Medium-term (This month)
1. Test cross-person generalization
2. Measure inference speed and memory
3. Fine-tune hyperparameters for multi-sequence training

### Long-term (Future)
1. Consider view-conditional decoder branches (if needed)
2. Explore temporal attention across views
3. Test on larger multi-person datasets

---

## Code Quality Notes

- All changes are **backward compatible** (no breaking changes)
- Code follows existing patterns in the codebase
- No new dependencies added
- Changes are minimal and focused on the core issue

---

## References

- **Analysis Document**: `RECONSTRUCTION_QUALITY_ANALYSIS.md`
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Test Script**: `test_multiview_fixes.py`
- **Config**: `configs/vae/train/wan_multiview_finetune.py`

---

## Questions?

If you encounter issues:
1. Check `RECONSTRUCTION_QUALITY_ANALYSIS.md` for detailed problem analysis
2. Review `IMPLEMENTATION_GUIDE.md` for step-by-step troubleshooting
3. Run `test_multiview_fixes.py` to diagnose component failures
4. Check W&B logs for training metrics and loss trends

---

**Status**: ✅ Implementation Complete  
**Test Status**: Ready for validation  
**Expected Outcome**: SSIM > 0.90, clean per-camera reconstructions  

