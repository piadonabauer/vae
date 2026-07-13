"""
Presets for VAE GAN discriminators: expand None | short string | dict into train.py fields.

Used by ``wan_multiview_finetune.py`` and by ``scripts/vae/train.py`` when the merged config
still has a string ``discriminator`` (e.g. ``--cfg-options discriminator StyleGAN2``).
"""

from __future__ import annotations

from typing import Any, Dict, Union

Choice = Union[None, str, Dict[str, Any]]


def resolve_vae_discriminator_bundle(choice: Choice) -> Dict[str, Any]:
    """
    Map user choice to kwargs for the training script.

    choice:
      - ``None`` or ``\"None\"`` (case-insensitive): no discriminator.
      - ``\"Train\"``: 3D PatchGAN from scratch (ndf=64, n_layers=5, ~18M params).
      - ``\"TrainLight\"``: lightweight 3D PatchGAN (ndf=32, n_layers=3, ~4M params, grad-ckpt). ~8x cheaper than Train.
      - ``\"TrainMultiview4D\"`` / ``\"TrainMv4d\"``: joint multiview 3D disc with view-axis merge (ndf=64, n_layers=5). num_views auto-detected.
      - ``\"TrainLight4D\"``: lightweight multiview 4D disc (ndf=32, n_layers=3, grad-ckpt). Best default for limited VRAM.
      - ``\"TrainPerFrame\"``: cheapest — 2D PatchGAN applied per-frame independently (no temporal signal).
      - ``\"TrainMultiviewStack\"``: 6-channel stacked-view 3D PatchGAN; use ``disc_multiview_mode=\"stack_channels\"``.
      - ``\"StyleGAN2\"``: pretrained NVlabs StyleGAN2-ADA 2D disc, per-frame.
      - ``\"PatchGAN\"``: LDM-style ``NLayerDiscriminator`` with **random init**, per-frame (HF VAE repos omit disc weights).
      - ``dict``: used as ``cfg.discriminator``; ``disc_per_frame_2d`` is set True when
        ``type`` is a 2D pretrained path, else False; GAN hyperparams get sane defaults.

    Returns keys: ``discriminator``, ``disc_per_frame_2d``, and when discriminator is set:
    ``disc_lr_scheduler``, ``gen_loss_config``, ``disc_loss_config``, ``optim_discriminator``.
    """
    if isinstance(choice, dict):
        dct = dict(choice)
        t = dct.get("type", "")
        per_frame = t in ("pretrained_stylegan2_discriminator", "pretrained_sd_vae_nlayer_discriminator")
        return dict(
            discriminator=dct,
            disc_per_frame_2d=per_frame,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=0, disc_weight=0.05),
            disc_loss_config=dict(disc_start=0, disc_loss_type="hinge"),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if choice is None:
        sn = None
    else:
        sn = str(choice).strip().lower()
    if sn in (None, "", "none"):
        return dict(discriminator=None, disc_per_frame_2d=False)

    if sn == "train":
        return dict(
            discriminator=dict(
                type="N_Layer_discriminator_3D",
                from_pretrained=None,
                input_nc=3,
                n_layers=5,
                conv_cls="conv3d",
            ),
            disc_per_frame_2d=False,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("train_multiview_4d", "train_mv4d", "trainmultiview4d"):
        # Joint multi-view 3D disc: sees both views + per-view embeddings (use with disc_multiview_mode=joint_4d).
        # num_views is intentionally omitted here; train.py probes the dataset at startup and injects the
        # correct value automatically, so the disc works with any number of views.
        return dict(
            discriminator=dict(
                type="N_Layer_discriminator_multiview_4d",
                from_pretrained=None,
                rgb_channels=3,
                n_layers=5,
                ndf=64,
                view_embed_dim=8,
            ),
            disc_per_frame_2d=False,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("train_light", "trainlight"):
        # Lightweight 3D PatchGAN: ndf=32, n_layers=3 + gradient checkpointing.
        # ~4M params, ~8x cheaper activations than Train.  Good first choice when Train OOMs.
        return dict(
            discriminator=dict(
                type="N_Layer_discriminator_3D",
                from_pretrained=None,
                input_nc=3,
                ndf=32,
                n_layers=3,
                conv_cls="conv3d",
                gradient_checkpointing=True,
            ),
            disc_per_frame_2d=False,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("train_light_4d", "trainlight4d", "train_light_mv4d"):
        # Lightweight multiview 4D disc: ndf=32, n_layers=3 + gradient checkpointing.
        # num_views auto-detected by train.py from the dataset.
        return dict(
            discriminator=dict(
                type="N_Layer_discriminator_multiview_4d",
                from_pretrained=None,
                rgb_channels=3,
                n_layers=3,
                ndf=32,
                view_embed_dim=8,
                gradient_checkpointing=True,
            ),
            disc_per_frame_2d=False,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("train_per_frame", "trainperframe", "per_frame"):
        # Cheapest option: 2D PatchGAN applied per-frame independently (no temporal signal).
        # Uses the LDM-style head (ndf=64, n_layers=3, 2D convs).  Very low memory cost.
        return dict(
            discriminator=dict(
                type="pretrained_sd_vae_nlayer_discriminator",
                random_init_only=True,
                input_nc=3,
                ndf=64,
                n_layers=3,
                freeze_layers=0,
            ),
            disc_per_frame_2d=True,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("train_multiview_stack", "train_mv_stack"):
        # Stack views in channel dim [B, V*3, T, H, W]; standard 3D PatchGAN (disc_multiview_mode=stack_channels).
        return dict(
            discriminator=dict(
                type="N_Layer_discriminator_3D",
                from_pretrained=None,
                input_nc=6,
                n_layers=5,
                conv_cls="conv3d",
            ),
            disc_per_frame_2d=False,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    if sn in ("stylegan2", "stylegan"):
        return dict(
            discriminator=dict(
                type="pretrained_stylegan2_discriminator",
                pretrained="stylegan2-ffhq-256",
                repo_id="mukhbiir/StyleGAN2_Discriminator",
                filename="stylegan2_discriminator.pth",
                freeze_layers=6,
            ),
            disc_per_frame_2d=True,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge", disc_factor=0.05),
            optim_discriminator=dict(
                cls="AdamW",
                lr=5e-6,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.0, 0.99),
            ),
        )

    if sn in ("patchgan", "sd_vae", "ldm"):
        # LDM-style 2D PatchGAN. HF ``sd-vae-ft-mse`` / ``ema`` checkpoints do **not** ship
        # ``discriminator.*`` weights (encoder/decoder only) — train this head from scratch.
        return dict(
            discriminator=dict(
                type="pretrained_sd_vae_nlayer_discriminator",
                random_init_only=True,
                input_nc=3,
                ndf=64,
                n_layers=3,
                freeze_layers=0,
            ),
            disc_per_frame_2d=True,
            disc_lr_scheduler=dict(warmup_steps=0),
            gen_loss_config=dict(gen_start=3000, disc_weight=0.05),
            disc_loss_config=dict(disc_start=3000, disc_loss_type="hinge"),
            optim_discriminator=dict(
                cls="AdamW",
                lr=1e-4,
                eps=1e-8,
                weight_decay=0.0,
                betas=(0.9, 0.98),
            ),
        )

    raise ValueError(
        f"Invalid discriminator preset {choice!r}. "
        "Use None, 'Train', 'TrainLight', 'TrainMultiview4D', 'TrainLight4D', "
        "'TrainPerFrame', 'TrainMultiviewStack', 'StyleGAN2', 'PatchGAN', "
        "or a full discriminator dict."
    )


def apply_discriminator_bundle_to_cfg(cfg: Any, bundle: Dict[str, Any]) -> None:
    """Write bundle keys onto an MMEngine Config or plain namespace."""
    for key, val in bundle.items():
        cfg[key] = val
