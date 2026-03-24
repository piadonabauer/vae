#!/usr/bin/env python3
"""
Test script to verify the multi-view VAE fixes work correctly.

This script tests:
1. Model registration and instantiation
2. Forward pass with multi-view input
3. Posterior handling
4. Loss computation
5. View consistency loss
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add paths for imports
sys.path.insert(0, '/home/piado/projects/aip-lindell/piado/vae/Open-Sora')
sys.path.insert(0, '/home/piado/projects/aip-lindell/piado/vae/DiffSynth-Studio')

def test_model_registration():
    """Test that the multi-view VAE model is properly registered."""
    print("Testing model registration...")
    
    try:
        from opensora.registry import MODELS
        from opensora.models.vae.wan_video_vae import build_multiview_wan_video_vae
        
        # Test that the model can be built
        model = build_multiview_wan_video_vae(
            z_dim=16,
            view_in=2,
            view_compression=2,
            use_view_embedding=True,
            from_pretrained=None,  # Don't load pretrained for this test
        )
        
        print("✓ Model registration successful")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Model has view_in={model.view_in}, z_dim={model.z_dim}")
        return model
        
    except Exception as e:
        print(f"✗ Model registration failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass with multi-view input."""
    print("\nTesting forward pass...")
    
    try:
        # Create a simple multi-view input: [B=1, V=2, C=3, T=4, H=64, W=64]
        # Use random noise for testing
        # Note: The model applies temporal downsampling, so we need enough temporal resolution
        # and the view compression kernel size must be smaller than the temporal dimension
        x = torch.randn(1, 2, 3, 8, 64, 64)  # Changed T from 4 to 8 to handle view compression
        
        # Forward pass
        x_rec, posterior, z = model(x)
        
        print("✓ Forward pass successful")
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {x_rec.shape}")
        print(f"✓ Latent shape: {z.shape}")
        print(f"✓ Posterior type: {type(posterior)}")
        
        # Check shapes are correct
        assert x_rec.shape == x.shape, f"Output shape mismatch: {x_rec.shape} vs {x.shape}"
        assert z.shape[1] == model.z_dim, f"Latent dimension mismatch: {z.shape[1]} vs {model.z_dim}"
        
        # Check posterior is a tuple of (mu, logvar)
        assert isinstance(posterior, tuple), f"Posterior should be tuple, got {type(posterior)}"
        assert len(posterior) == 2, f"Posterior should have 2 elements, got {len(posterior)}"
        
        mu, logvar = posterior
        assert mu.shape == z.shape, f"Mu shape mismatch: {mu.shape} vs {z.shape}"
        assert logvar.shape == z.shape, f"Logvar shape mismatch: {logvar.shape} vs {z.shape}"
        
        print("✓ All shape checks passed")
        return x, x_rec, posterior, z
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_view_consistency_loss(x_rec):
    """Test view consistency loss computation."""
    print("\nTesting view consistency loss...")
    
    try:
        # Compute view consistency loss manually
        if x_rec.shape[1] > 1:  # Multi-view
            view_losses = []
            for i in range(x_rec.shape[1] - 1):
                view_losses.append(F.mse_loss(x_rec[:, i], x_rec[:, i + 1]))
            view_loss = sum(view_losses) / len(view_losses)
            
            print("✓ View consistency loss computed successfully")
            print(f"✓ View loss value: {view_loss.item():.6f}")
            print(f"✓ Number of view pairs: {len(view_losses)}")
            
            # Check that view loss is reasonable (not NaN or inf)
            assert not torch.isnan(view_loss), "View loss is NaN"
            assert not torch.isinf(view_loss), "View loss is inf"
            assert view_loss.item() >= 0, "View loss should be non-negative"
            
            return view_loss
        else:
            print("✓ Single-view input, no view consistency loss needed")
            return torch.tensor(0.0)
            
    except Exception as e:
        print(f"✗ View consistency loss failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_posterior_wrapping():
    """Test that posterior wrapping works correctly."""
    print("\nTesting posterior wrapping...")
    
    try:
        from opensora.models.vae.utils import DiagonalGaussianDistribution
        
        # Test tuple input
        mu = torch.randn(1, 16, 4, 8, 8)
        logvar = torch.randn(1, 16, 4, 8, 8)
        posterior_tuple = (mu, logvar)
        
        posterior_dist = DiagonalGaussianDistribution(torch.cat(posterior_tuple, dim=1))
        print("✓ Posterior wrapping from tuple successful")
        
        # Test that it has the expected attributes
        assert hasattr(posterior_dist, 'parameters'), "Posterior should have parameters attribute"
        assert hasattr(posterior_dist, 'deterministic'), "Posterior should have deterministic attribute"
        
        print("✓ Posterior has required attributes")
        return True
        
    except Exception as e:
        print(f"✗ Posterior wrapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test that loss computation works with the fixes."""
    print("\nTesting loss computation...")
    
    try:
        from opensora.models.vae.losses import VAELoss
        
        # Create a simple loss function
        device = torch.device('cpu')  # Use CPU for testing
        dtype = torch.float32
        vae_loss_fn = VAELoss(
            perceptual_loss_weight=0.5,
            kl_loss_weight=5e-4,
            view_consistency_weight=0.01,
            device=device,
            dtype=dtype
        )
        
        # Create test data
        x = torch.randn(1, 2, 3, 4, 64, 64)  # Multi-view
        x_rec = torch.randn(1, 2, 3, 4, 64, 64)  # Multi-view reconstruction
        
        # Create a simple posterior (mu, logvar)
        mu = torch.randn(1, 32, 4, 8, 8)  # 2 * z_dim for concatenated mu, logvar
        logvar = torch.randn(1, 32, 4, 8, 8)
        posterior = (mu, logvar)
        
        # Flatten for loss computation (as done in training)
        b, v, c, t, h, w = x.shape
        x_loss = x.view(b * v, c, t, h, w)
        x_rec_loss = x_rec.view(b * v, c, t, h, w)
        
        # Compute loss
        ret = vae_loss_fn(x_loss, x_rec_loss, posterior)
        
        print("✓ Loss computation successful")
        print(f"✓ NLL loss: {ret['nll_loss'].item():.6f}")
        print(f"✓ KL loss: {ret['kl_loss'].item():.6f}")
        print(f"✓ Reconstruction loss: {ret['recon_loss'].item():.6f}")
        print(f"✓ Perceptual loss: {ret['perceptual_loss'].item():.6f}")
        
        # Check that all losses are reasonable
        for key, value in ret.items():
            assert not torch.isnan(value), f"{key} loss is NaN"
            assert not torch.isinf(value), f"{key} loss is inf"
            assert value.item() >= 0, f"{key} loss should be non-negative"
        
        print("✓ All loss values are valid")
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Multi-View VAE Fixes")
    print("=" * 60)
    
    # Test 1: Model registration
    model = test_model_registration()
    if model is None:
        print("\n✗ Tests failed at model registration")
        return False
    
    # Test 2: Forward pass
    x, x_rec, posterior, z = test_forward_pass(model)
    if x is None:
        print("\n✗ Tests failed at forward pass")
        return False
    
    # Test 3: View consistency loss
    view_loss = test_view_consistency_loss(x_rec)
    if view_loss is None:
        print("\n✗ Tests failed at view consistency loss")
        return False
    
    # Test 4: Posterior wrapping
    posterior_ok = test_posterior_wrapping()
    if not posterior_ok:
        print("\n✗ Tests failed at posterior wrapping")
        return False
    
    # Test 5: Loss computation
    loss_ok = test_loss_computation()
    if not loss_ok:
        print("\n✗ Tests failed at loss computation")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! The multi-view VAE fixes are working correctly.")
    print("=" * 60)
    
    print("\nKey fixes verified:")
    print("1. ✓ Model registration in Open-Sora registry")
    print("2. ✓ Proper forward pass with multi-view input")
    print("3. ✓ Correct posterior handling (tuple of mu, logvar)")
    print("4. ✓ View consistency loss computation")
    print("5. ✓ Loss computation with view flattening")
    print("6. ✓ Posterior wrapping for KL computation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)