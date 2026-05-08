import os

import torch
import torch.nn as nn

from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_conv(m):
    if hasattr(m, "conv"):
        m = m.conv
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""

    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=5,
        norm_layer=nn.BatchNorm3d,
        conv_cls="conv3d",
        dropout=0.30,
    ):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        assert conv_cls == "conv3d"
        use_bias = False

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=2 if n == 1 else (1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.main(x)


@MODELS.register_module("N_Layer_discriminator_3D")
def N_LAYER_DISCRIMINATOR_3D(from_pretrained=None, force_huggingface=None, **kwargs):
    model = NLayerDiscriminator3D(**kwargs).apply(weights_init)
    if from_pretrained is not None:
        if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
            raise NotImplementedError
        else:
            load_checkpoint(model, from_pretrained)
        print(f"loaded model from: {from_pretrained}")
    return model


class NLayerDiscriminatorMultiview4D(nn.Module):
    """
    Multi-view 3D PatchGAN that processes both views jointly.

    Expects ``[B, V, C, T, H, W]`` (e.g. V=2, C=3). Per-view learnable embeddings are
    concatenated to RGB, then a Conv3d with kernel depth ``V`` merges the view axis so
    the first layer sees both views at once. Remaining blocks are standard 3D PatchGAN
    on ``[B, ndf, T, H', W']``.
    """

    def __init__(
        self,
        num_views=2,
        rgb_channels=3,
        ndf=64,
        n_layers=5,
        norm_layer=nn.BatchNorm3d,
        dropout=0.30,
        view_embed_dim=8,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.rgb_channels = int(rgb_channels)
        self.ndf = ndf
        if view_embed_dim and view_embed_dim > 0:
            self.view_embed_dim = int(view_embed_dim)
            self.view_emb = nn.Embedding(self.num_views, self.view_embed_dim)
        else:
            self.view_embed_dim = 0
            self.view_emb = None

        in_after_emb = self.rgb_channels + (self.view_embed_dim if self.view_emb is not None else 0)
        kw = 3
        padw = 1
        # Merge view axis V with spatial (H, W): input [N, C, V, H, W]
        self.view_merge = nn.Conv3d(
            in_after_emb,
            ndf,
            kernel_size=(self.num_views, kw, kw),
            stride=(1, 2, 2),
            padding=(0, padw, padw),
        )
        use_bias = False
        sequence = [nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=2 if n == 1 else (1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.tail = nn.Sequential(*sequence)

    def forward(self, x):
        if x.dim() != 6:
            raise ValueError(
                f"NLayerDiscriminatorMultiview4D expects [B,V,C,T,H,W], got shape {tuple(x.shape)}"
            )
        b, v, c, t, h, w = x.shape
        if v != self.num_views:
            raise ValueError(f"Expected V={self.num_views}, got {v}")
        if c != self.rgb_channels:
            raise ValueError(f"Expected C={self.rgb_channels}, got {c}")

        if self.view_emb is not None:
            emb = self.view_emb.weight.view(1, v, self.view_embed_dim, 1, 1, 1).expand(
                b, v, self.view_embed_dim, t, h, w
            )
            x = torch.cat([x, emb], dim=2)

        c2 = x.shape[2]
        # [B, V, C2, T, H, W] -> [B, T, C2, V, H, W] -> [B*T, C2, V, H, W]
        x = x.permute(0, 3, 2, 1, 4, 5).contiguous().view(b * t, c2, v, h, w)
        y = self.view_merge(x)
        if y.shape[2] != 1:
            raise RuntimeError(
                f"view_merge should collapse the view axis to 1, got D={y.shape[2]} "
                f"(num_views={self.num_views})"
            )
        y = y.squeeze(2)
        _, c3, h1, w1 = y.shape
        y = y.view(b, t, c3, h1, w1).permute(0, 2, 1, 3, 4).contiguous()
        return self.tail(y)


@MODELS.register_module("N_Layer_discriminator_multiview_4d")
def N_LAYER_DISCRIMINATOR_MULTIVIEW_4D(from_pretrained=None, force_huggingface=None, **kwargs):
    model = NLayerDiscriminatorMultiview4D(**kwargs).apply(weights_init)
    if from_pretrained is not None:
        if force_huggingface or not os.path.exists(from_pretrained):
            raise NotImplementedError
        load_checkpoint(model, from_pretrained)
        print(f"loaded model from: {from_pretrained}")
    return model


# Side effect: registers ``pretrained_sd_vae_nlayer_discriminator`` for ``build_module``.
import opensora.models.vae.sd_ldm_discriminator  # noqa: E402, F401
