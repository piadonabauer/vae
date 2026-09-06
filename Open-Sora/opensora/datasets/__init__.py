"""Reconstructed minimal opensora.datasets package (NeRSemble multi-view pt_video).

The full Open-Sora datasets package is gitignored and absent from this checkout;
this provides exactly what the VAE trainer imports: the registered ``pt_video``
dataset, ``prepare_dataloader``, and ``PinMemoryCache``.

NOTE: pt_video.py and pt_video_dataset.py both register the name "pt_video"
(the latter is an older per-machine copy kept for reference). Import exactly ONE
of them here -- importing both crashes with a duplicate registry entry.
pt_video.PtVideoDataset handles both on-disk conventions (frames.pt / <seq>.pt,
raw tensor or dict payload, [V,T,C,H,W] or [V,C,T,H,W]).
"""
from .dataloader import prepare_dataloader
from .pin_memory_cache import PinMemoryCache
from .pt_video import PtVideoDataset

__all__ = ["prepare_dataloader", "PinMemoryCache", "PtVideoDataset"]
