"""Load NVlabs StyleGAN2-ADA discriminators from Hugging Face state dict checkpoints.

Optional environment (third_party StyleGAN2 ops):

- ``TORCH_EXTENSIONS_DIR``: writable directory for JIT-built CUDA extensions (required on many clusters
  for a successful compile). PyTorch defaults often work on workstations but not on shared filesystems.

- ``STYLEGAN2_DISABLE_CUDA_PLUGINS=1``: do not attempt JIT compile; use PyTorch reference ops only
  (same math as the fused kernels, slower). No failed-build noise.

- ``STYLEGAN2_CUSTOM_OPS_VERBOSITY``: ``none`` (default), ``brief``, or ``full`` for NVlabs compile logs.

If JIT fails, a single ``UserWarning`` explains the situation; training still uses numerically equivalent
reference implementations—not a different discriminator "feature space."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleGAN2ADADiscriminatorWrapper(nn.Module):
    """Wraps NVlabs `Discriminator` so a single image tensor matches `train.py` call sites."""

    def __init__(self, core: nn.Module, img_resolution: int):
        super().__init__()
        self.core = core
        self.img_resolution = int(img_resolution)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        if h != self.img_resolution or w != self.img_resolution:
            img = F.interpolate(
                img,
                size=(self.img_resolution, self.img_resolution),
                mode="bilinear",
                align_corners=False,
            )
        return self.core(img, None)


def _default_stylegan2_ada_root() -> Path:
    # opensora/models/vae/this_file.py -> Open-Sora repo root
    return Path(__file__).resolve().parents[3] / "third_party" / "stylegan2-ada-pytorch"


def _resolve_stylegan2_ada_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("STYLEGAN2_ADA_PYTORCH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _default_stylegan2_ada_root()


def build_stylegan2_ada_discriminator_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    img_resolution: int = 512,
    img_channels: int = 3,
    stylegan2_ada_root: Optional[Union[str, Path]] = None,
) -> nn.Module:
    root = _resolve_stylegan2_ada_root(stylegan2_ada_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"StyleGAN2-ADA-PyTorch not found at {root}. "
            "Set discriminator.stylegan2_ada_root or env STYLEGAN2_ADA_PYTORCH to a clone of "
            "https://github.com/NVlabs/stylegan2-ada-pytorch (or use the copy shipped under "
            "Open-Sora/third_party/stylegan2-ada-pytorch)."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from training.networks import Discriminator  # type: ignore[import-not-found]

    # mbstd_group_size=None: per-sample stats so batch size need not divide 4 (flattened disc batches are often odd).
    core = Discriminator(
        c_dim=0,
        img_resolution=img_resolution,
        img_channels=img_channels,
        epilogue_kwargs=dict(mbstd_group_size=None),
    )
    missing, unexpected = core.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing!r} unexpected={unexpected!r}")
    core = core.to(device)
    return StyleGAN2ADADiscriminatorWrapper(core, img_resolution=img_resolution)
