import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from opensora.models.vae.lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def wgan_gp_loss(logits_real, logits_fake):
    d_loss = 0.5 * (-logits_real.mean() + logits_fake.mean())
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


def batch_mean(x):
    return torch.sum(x) / x.shape[0]


def sigmoid_cross_entropy_with_logits(labels, logits):
    # The final formulation is: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(nn.ReLU()(real_pred - ema_fake_pred), 2))
    lecam_loss += torch.mean(torch.pow(nn.ReLU()(ema_real_pred - fake_pred), 2))
    return lecam_loss


def gradient_penalty_fn(images, output):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class VAELoss(nn.Module):
    def __init__(
        self,
        logvar_init=0.0,
        perceptual_loss_weight=1.0,
        kl_loss_weight=5e-4,
        device="cpu",
        dtype="bf16",
        view_consistency_weight=0.0,
        lpips_chunk_size=None,
        lpips_scale=1.0,
    ):
        super().__init__()

        if type(dtype) == str:
            if dtype == "bf16":
                dtype = torch.bfloat16
            elif dtype == "fp16":
                dtype = torch.float16
            elif dtype == "fp32":
                dtype = torch.float32
            else:
                raise NotImplementedError(f"dtype: {dtype}")

        # KL Loss
        self.kl_weight = kl_loss_weight
        # Perceptual Loss
        self.perceptual_loss_fn = LPIPS().eval().to(device, dtype)
        self.perceptual_loss_fn.requires_grad_(False)
        self.perceptual_loss_weight = perceptual_loss_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # Multi-view loss weight
        self.view_consistency_weight = view_consistency_weight
        # LPIPS chunk size: process this many frames at a time through VGG to avoid OOM
        # at high resolutions. None = no chunking (all frames in one call).
        self.lpips_chunk_size = lpips_chunk_size
        # LPIPS scale: resize frames to this fraction of original before VGG.
        # VGG features are scale-invariant so 0.5 gives equivalent gradient signal
        # at 4× lower cost. Set to 1.0 to disable downsampling.
        self.lpips_scale = float(lpips_scale)

    def forward(
        self,
        video,
        recon_video,
        posterior,
    ) -> dict:
        video.size(0)
        
        # Check input format and handle different shapes
        # Expected formats: [B, C, T, H, W] or [B, T, C, H, W] or [B, V, C, T, H, W]
        is_multiview = False
        b, v = None, None
        if video.dim() == 6:
            # Multi-view: [B, V, C, T, H, W] - preserve view dimension initially
            is_multiview = True
            b, v, c, t, h, w = video.shape
            # Some sequences can be shorter/longer in T than others (e.g. 9 vs 10/13 frames).
            # Align reconstruction and target along the temporal dimension before reshaping
            # to avoid invalid .view() calls when T differs.
            _, _, _, t_rec, _, _ = recon_video.shape
            if t_rec != t:
                new_t = min(t, t_rec)
                video = video[:, :, :, :new_t, :, :].contiguous()
                recon_video = recon_video[:, :, :, :new_t, :, :].contiguous()
                t = new_t
            # Reshape for temporal batching: [B*V, C, T, H, W]
            video_reshaped = video.view(b * v, c, t, h, w)
            recon_video_reshaped = recon_video.view(b * v, c, t, h, w)
        elif video.dim() == 5:
            # Check if it's [B, T, C, H, W] or [B, C, T, H, W]
            # If T dimension (dim=1) is larger than C dimension (dim=2), it's likely [B, T, C, H, W]
            if video.shape[1] > video.shape[2] and video.shape[2] <= 4:
                # It's [B, T, C, H, W], permute to [B, C, T, H, W]
                video = video.permute(0, 2, 1, 3, 4).contiguous()
                recon_video = recon_video.permute(0, 2, 1, 3, 4).contiguous()
            video_reshaped = video
            recon_video_reshaped = recon_video
        else:
            video_reshaped = video
            recon_video_reshaped = recon_video
        
        # Flatten temporal dimension for loss computation: [B*V*T, C, H, W]
        video_flat = rearrange(video_reshaped, "b c t h w -> (b t) c h w").contiguous()
        recon_video_flat = rearrange(recon_video_reshaped, "b c t h w -> (b t) c h w").contiguous()

        # reconstruction loss
        recon_loss = l1(video_flat, recon_video_flat)

        # perceptual loss — optionally downsampled + chunked to avoid OOM.
        # LPIPS measures perceptual similarity via VGG features (texture/structure),
        # which are scale-invariant. Running at half resolution gives an equivalent
        # gradient signal at 4× lower VGG cost and is standard practice (SD uses
        # this too). Chunking along the N (frames) dimension handles any remaining
        # memory pressure. LPIPS returns [N, 1, 1, 1].
        _vf = video_flat
        _rf = recon_video_flat
        if self.lpips_scale < 1.0:
            _vf = F.interpolate(_vf, scale_factor=self.lpips_scale, mode="bilinear", align_corners=False)
            _rf = F.interpolate(_rf, scale_factor=self.lpips_scale, mode="bilinear", align_corners=False)
        if self.lpips_chunk_size is not None and _vf.shape[0] > self.lpips_chunk_size:
            _lpips_parts = []
            for _i in range(0, _vf.shape[0], self.lpips_chunk_size):
                _lpips_parts.append(
                    self.perceptual_loss_fn(
                        _vf[_i : _i + self.lpips_chunk_size],
                        _rf[_i : _i + self.lpips_chunk_size],
                    )
                )
            perceptual_loss = torch.cat(_lpips_parts, dim=0)
        else:
            perceptual_loss = self.perceptual_loss_fn(_vf, _rf)
        
        # nll loss (from reconstruction loss and perceptual loss)
        nll_loss = recon_loss + perceptual_loss * self.perceptual_loss_weight
        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar

        # Compute means properly - average over ALL dimensions
        # For per-pixel losses, we need to average over all elements, not just batch
        nll_loss = torch.mean(nll_loss)  # Average over all elements
        recon_loss = torch.mean(recon_loss)  # Average over all elements
        # The perceptual loss is already normalized by the LPIPS network
        perceptual_loss = torch.mean(perceptual_loss)  # Average over all elements

        # KL Loss
        if posterior is None:
            kl_loss = torch.tensor(0.0).to(video_flat.device, video_flat.dtype)
        else:
            kl_loss = posterior.kl()
            kl_loss = torch.mean(kl_loss)  # Average over all elements, not just batch
        weighted_kl_loss = kl_loss * self.kl_weight

        # View consistency loss (only for multi-view)
        view_consistency_loss = torch.tensor(0.0).to(video.device, video.dtype)
        if is_multiview and self.view_consistency_weight > 0:
            # Penalize if different views produce similar reconstructions
            # This encourages the decoder to produce view-specific outputs
            view_consistency_loss = self._compute_view_consistency_loss(recon_video, video)
            view_consistency_loss = view_consistency_loss * self.view_consistency_weight

        loss_dict = {
            "nll_loss": nll_loss,
            "kl_loss": weighted_kl_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": perceptual_loss,
        }
        
        if is_multiview and self.view_consistency_weight > 0:
            loss_dict["view_consistency_loss"] = view_consistency_loss

        return loss_dict

    def _compute_view_consistency_loss(self, recon_video, video):
        """
        Compute view consistency loss to encourage each view to have unique reconstruction.
        
        Args:
            recon_video: [B, V, C, T, H, W]
            video: [B, V, C, T, H, W]
        
        Returns:
            Scalar loss value
        """
        b, v, c, t, h, w = recon_video.shape
        if v < 2:
            return torch.tensor(0.0, device=recon_video.device, dtype=recon_video.dtype)
        
        # Compute cross-view reconstruction similarity
        # For each pair of views, compute L1 distance between their reconstructions
        # Higher cross-view distance = better (each view has unique features)
        # We want to MINIMIZE reconstruction similarity across views
        
        # Flatten spatial-temporal dims for easier comparison
        recon_flat = rearrange(recon_video, "b v c t h w -> b v (c t h w)")
        
        consistency_loss = 0.0
        num_pairs = 0
        
        # Compare all pairs of views
        for i in range(v):
            for j in range(i + 1, v):
                # L2 distance between view i and view j reconstructions
                # High distance = less consistent = good
                # But we want them to be consistent with THEIR OWN inputs, not with each other
                # So we penalize when reconstruction[i] ≈ reconstruction[j]
                pair_similarity = torch.nn.functional.cosine_similarity(
                    recon_flat[:, i:i+1], 
                    recon_flat[:, j:j+1],
                    dim=2
                ).mean()
                consistency_loss += pair_similarity
                num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss


class GeneratorLoss(nn.Module):
    def __init__(self, gen_start=2001, disc_factor=1.0, disc_weight=0.5):
        super().__init__()
        self.disc_factor = disc_factor
        self.gen_start = gen_start
        self.disc_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(
        self,
        logits_fake,
        nll_loss,
        last_layer,
        global_step,
        is_training=True,
    ):
        g_loss = -torch.mean(logits_fake)

        if self.disc_factor is not None and self.disc_factor > 0.0:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
        else:
            d_weight = torch.tensor(1.0)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.gen_start)
        weighted_gen_loss = d_weight * disc_factor * g_loss

        return weighted_gen_loss, g_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_start=2001, disc_factor=1.0, disc_loss_type="hinge"):
        super().__init__()

        assert disc_loss_type in ["hinge", "vanilla", "wgan-gp"]
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.disc_loss_type = disc_loss_type

        if self.disc_loss_type == "hinge":
            self.loss_fn = hinge_d_loss
        elif self.disc_loss_type == "vanilla":
            self.loss_fn = vanilla_d_loss
        elif self.disc_loss_type == "wgan-gp":
            self.loss_fn = wgan_gp_loss
        else:
            raise ValueError(f"Unknown GAN loss '{self.disc_loss_type}'.")

    def forward(
        self,
        real_logits,
        fake_logits,
        global_step,
    ):
        if self.disc_factor is not None and self.disc_factor > 0.0:
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            disc_loss = self.loss_fn(real_logits, fake_logits)
            weighted_discriminator_loss = disc_factor * disc_loss
        else:
            weighted_discriminator_loss = 0

        return weighted_discriminator_loss