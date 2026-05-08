"""
LDM / VQGAN-style 2D PatchGAN discriminator (taming-transformers NLayerDiscriminator).

Can **load** weights when a checkpoint contains ``discriminator.*`` or
``loss.discriminator.*`` (many public HF VAE repos such as ``stabilityai/sd-vae-ft-mse``
ship **encoder/decoder only** — no discriminator). For that case use
``random_init_only=True`` or ``fallback_random_init=True`` (see builder).

Inputs should be in **[-1, 1]** (same convention as SD VAE training), which matches
Wan multiview training when ``vae_target_range = "[-1,1]"``.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from opensora.registry import MODELS


def _load_raw_checkpoint(path: str) -> Dict[str, Any]:
    """Load a ``.bin`` / ``.pt`` / ``.safetensors`` checkpoint as a flat str->tensor map."""
    path_lower = path.lower()
    if path_lower.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "Loading .safetensors requires `safetensors`. Install with: pip install safetensors"
            ) from e
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state dict at {path}, got {type(obj)}")
    return obj


def extract_ldm_discriminator_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Collect discriminator weights from a VAE / LDM checkpoint.

    Supports:
    - ``discriminator.<subkey>`` (common in some diffusers exports)
    - ``loss.discriminator.<subkey>`` (CompVis LDM / Lightning checkpoints)
    """
    out: Dict[str, torch.Tensor] = {}
    prefixes = ("discriminator.", "loss.discriminator.")
    for k, v in ckpt.items():
        if not isinstance(v, torch.Tensor):
            continue
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p) :]] = v
                break
    return out


class NLayerDiscriminatorLDM(nn.Module):
    """
    PatchGAN discriminator as in taming-transformers / LDM (Pix2Pix-style).

    Module layout: ``self.main`` = ``nn.Sequential(...)`` so state dict keys are
    ``main.0.weight``, ``main.1.weight``, … matching published LDM checkpoints.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(
                "use_actnorm=True requires ActNorm from taming-transformers; use use_actnorm=False for SD VAE."
            )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def _freeze_first_main_children(disc: NLayerDiscriminatorLDM, num_children: int) -> None:
    if num_children <= 0:
        return
    for i, m in enumerate(disc.main.children()):
        if i < num_children:
            m.requires_grad_(False)


def _weights_init_nlayer_disc(module: nn.Module) -> None:
    """Pix2Pix / LDM-style init (taming ``weights_init``)."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


def _build_nlayer_random(
    input_nc: int,
    ndf: int,
    n_layers: int,
    use_actnorm: bool,
    freeze_layers: int,
) -> NLayerDiscriminatorLDM:
    model = NLayerDiscriminatorLDM(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        use_actnorm=use_actnorm,
    )
    model.apply(_weights_init_nlayer_disc)
    _freeze_first_main_children(model, int(freeze_layers or 0))
    return model


def build_pretrained_sd_vae_nlayer_discriminator(
    from_pretrained: Optional[str] = None,
    repo_id: str = "stabilityai/sd-vae-ft-mse",
    filename: str = "diffusion_pytorch_model.bin",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    input_nc: int = 3,
    ndf: int = 64,
    n_layers: int = 3,
    use_actnorm: bool = False,
    freeze_layers: int = 0,
    strict: bool = True,
    random_init_only: bool = False,
    fallback_random_init: bool = False,
    **kwargs: Any,
) -> NLayerDiscriminatorLDM:
    """
    Build :class:`NLayerDiscriminatorLDM` and optionally load from a checkpoint.

    - ``random_init_only``: skip all checkpoint I/O; train the PatchGAN from scratch (default for
      preset ``PatchGAN`` because ``sd-vae-ft-mse`` has no ``discriminator.*`` tensors).
    - ``fallback_random_init``: if the file has no discriminator keys, init randomly instead of error.

    ``kwargs`` absorbs unused keys from MMEngine config (e.g. ``type``).
    """
    kwargs.pop("type", None)
    kwargs.pop("random_init_only", None)
    kwargs.pop("fallback_random_init", None)
    kwargs.clear()

    if random_init_only:
        return _build_nlayer_random(input_nc, ndf, n_layers, use_actnorm, freeze_layers)

    if from_pretrained is not None and str(from_pretrained).strip():
        ckpt_path = os.path.expanduser(str(from_pretrained))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"from_pretrained path not found: {ckpt_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download SD VAE weights. "
                "Install with: pip install huggingface_hub filelock"
            ) from e
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    raw = _load_raw_checkpoint(ckpt_path)
    disc_sd = extract_ldm_discriminator_state_dict(raw)
    if not disc_sd:
        if fallback_random_init:
            warnings.warn(
                f"No discriminator.* tensors in {ckpt_path}; using random init for NLayerDiscriminatorLDM. "
                "Public sd-vae-ft-mse/ema repos often omit the GAN head; set random_init_only=True to skip download.",
                UserWarning,
                stacklevel=2,
            )
            return _build_nlayer_random(input_nc, ndf, n_layers, use_actnorm, freeze_layers)
        raise RuntimeError(
            f"No discriminator tensors found in {ckpt_path}. "
            "Expected keys prefixed with 'discriminator.' or 'loss.discriminator.'. "
            "The usual Hugging Face VAE weights (e.g. stabilityai/sd-vae-ft-mse) do **not** include them. "
            "Fix: use discriminator config random_init_only=True (train PatchGAN from scratch), "
            "or fallback_random_init=True, or point from_pretrained to an LDM checkpoint that still has the disc, "
            "or use type='pretrained_stylegan2_discriminator'."
        )

    model = NLayerDiscriminatorLDM(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        use_actnorm=use_actnorm,
    )
    missing, unexpected = model.load_state_dict(disc_sd, strict=strict)
    if missing or unexpected:
        print(
            f"[pretrained_sd_vae_nlayer_discriminator] load_state_dict(strict={strict}): "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
        )
        if missing and len(missing) <= 20:
            print(f"  missing: {missing}")
        if unexpected and len(unexpected) <= 20:
            print(f"  unexpected: {unexpected}")

    _freeze_first_main_children(model, int(freeze_layers or 0))
    return model


@MODELS.register_module("pretrained_sd_vae_nlayer_discriminator")
def PRETRAINED_SD_VAE_NLAYER_DISCRIMINATOR(**kwargs: Any) -> NLayerDiscriminatorLDM:
    """Registry entry for MMEngine ``build_module`` (same builder as manual use)."""
    return build_pretrained_sd_vae_nlayer_discriminator(**kwargs)
