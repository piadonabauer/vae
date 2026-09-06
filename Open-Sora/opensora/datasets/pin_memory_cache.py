"""Reconstructed minimal PinMemoryCache.

The original Open-Sora cache pre-allocates pinned host buffers to speed up H2D
copies. The VAE config sets ``pin_memory_cache_pre_alloc_numels = None`` (caching
off) and standard ``DataLoader(pin_memory=True)`` is used instead, so only the
class-level configuration attributes are needed here.
"""
from typing import Optional

import torch


class PinMemoryCache:
    # Configured from train.py before the dataloader is built.
    force_dtype: Optional[torch.dtype] = None
    pre_alloc_numels = None

    def __init__(self):
        self.cache = {}
        self.output_to_cache = {}

    def get(self, tensor: torch.Tensor) -> torch.Tensor:
        # No pre-allocated cache in this configuration: fall back to a fresh pinned copy.
        return tensor.pin_memory()

    def remove(self, *args, **kwargs):
        return None

    def empty(self):
        self.cache.clear()
        self.output_to_cache.clear()
