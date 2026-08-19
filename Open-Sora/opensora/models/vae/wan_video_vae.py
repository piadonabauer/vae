"""
Wrapper for Wan 2.1 Video VAE with multi-view support.

This module wraps the DiffSynth Wan 2.1 VAE and registers it with the opensora model registry.
The multi-view VAE compresses multiple camera views into a single latent representation
while preserving view-specific information through learned pooling and view embeddings.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the multi-view VAE implementation from DiffSynth
# We need to add DiffSynth to the path
diffsynth_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "DiffSynth-Studio"
if str(diffsynth_root) not in sys.path:
    sys.path.insert(0, str(diffsynth_root))

from diffsynth.models.wan_video_vae import (
    AttentionMultiViewVideoVan,
    LoRAAttentionBlock,
    LoRAConv2d,
    LoRAConv3d,
    ProfileTimer,
    WanVideoVAE,
)

from opensora.registry import MODELS


def _unfreeze_lora_wrapped_bases(module: nn.Module) -> None:
    """Enable gradients on frozen base convs/attn inside DiffSynth LoRA wrappers (decoder full finetune)."""
    for m in module.modules():
        if isinstance(m, LoRAConv3d):
            for p in m.base_conv.parameters():
                p.requires_grad = True
        elif isinstance(m, LoRAConv2d):
            for p in m.base_conv.parameters():
                p.requires_grad = True
        elif isinstance(m, LoRAAttentionBlock):
            for p in m.to_qkv.parameters():
                p.requires_grad = True
            for p in m.proj.parameters():
                p.requires_grad = True


# class MultiviewWanVideoVAE(nn.Module):
#     """
#     Multi-view wrapper for Wan 2.1 3D Video VAE.
    
#     This model:
#     1. Encodes multi-view videos [B, V, C, T, H, W] into a compressed latent space [B, C, T, H, W]
#     2. Uses learned view pooling to preserve view-specific information
#     3. Applies view positional embeddings to help reconstruction distinguish between views
#     4. Decodes back to multi-view reconstructions
    
#     Args:
#         dim: Base channel dimension of the VAE (default: 96 for Wan 2.1)
#         z_dim: Latent dimension (default: 16 for Wan 2.1)
#         view_in: Number of input views (e.g., 2 for stereo)
#         view_compression: Compression ratio (e.g., 2 means 2 views -> 1 latent view)
#         use_view_embedding: Whether to use view positional embeddings
#         view_mixing_strategy: Strategy for mixing/decoding views:
#             - "embedding": Use view embeddings to modulate latent (lightweight)
#             - "residual": Use view-specific residual decoder (stronger differentiation)
#             - "both": Combine embeddings and residuals (strongest)
#         from_pretrained: Path to pretrained Wan 2.1 checkpoint to load
#         device_map: Device to place the model on (e.g., "cuda")
#         torch_dtype: Data type for the model (e.g., torch.bfloat16)
#     """
    
#     def __init__(
#         self,
#         dim=96,
#         z_dim=16,
#         view_in=2,
#         view_compression=2,
#         use_view_embedding=True,
#         view_mixing_strategy="embedding",
#         from_pretrained=None,
#         device_map=None,
#         torch_dtype=None,
#         **kwargs,
#     ):
#         super().__init__()
        
#         # Store config for reference
#         self.z_dim = z_dim
#         self.view_in = view_in
#         self.view_compression = view_compression
#         self.use_view_embedding = use_view_embedding
#         self.view_mixing_strategy = view_mixing_strategy
#         self.from_pretrained = from_pretrained
        
#         # Calculate output views after compression
#         view_out = max(1, view_in // view_compression)
        
#         # Initialize the multi-view VAE with default Wan 2.1 architecture
#         self.vae = MultiViewVideoVAE_(
#             dim=dim,
#             z_dim=z_dim,
#             dim_mult=[1, 2, 4, 4],  # Wan 2.1 architecture
#             num_res_blocks=2,        # Wan 2.1 architecture
#             attn_scales=[],          # Wan 2.1 architecture
#             temperal_downsample=[False, True, True],  # Wan 2.1 architecture
#             dropout=0.0,
#             view_in=view_in,
#             view_out=view_out,
#             use_view_embedding=use_view_embedding,
#             view_init="avg",
#         )
        
#         # Add view-conditional decoder if requested
#         # This injects view embeddings throughout the decode process, not just post-hoc
#         if view_mixing_strategy in ["conditional", "both"]:
#             # Import here to avoid circular imports
#             from diffsynth.models.wan_video_vae import ViewConditionalDecoder
#             self.view_conditional_decoder = ViewConditionalDecoder(
#                 num_views=view_in,
#                 base_channels=dim
#             )
#         else:
#             self.view_conditional_decoder = None
        
#         # Load pretrained weights if provided
#         if from_pretrained is not None:
#             try:
#                 checkpoint = torch.load(from_pretrained, map_location="cpu")
#                 # Handle both direct state dict and wrapped checkpoint formats
#                 if "model" in checkpoint:
#                     state_dict = checkpoint["model"]
#                 elif "state_dict" in checkpoint:
#                     state_dict = checkpoint["state_dict"]
#                 else:
#                     state_dict = checkpoint
                
#                 # Load base VAE weights (these should be compatible)
#                 self.vae.base.load_state_dict(state_dict, strict=False)
#                 print(f"Loaded pretrained Wan 2.1 VAE from {from_pretrained}")
#             except Exception as e:
#                 print(f"Warning: Failed to load pretrained weights from {from_pretrained}: {e}")
        
#         # Move to device and set dtype if specified
#         if device_map is not None:
#             self.vae = self.vae.to(device_map)
        
#         if torch_dtype is not None:
#             self.vae = self.vae.to(torch_dtype)
    
#     def forward(self, x, scale=None, **kwargs):
#         """
#         Forward pass through the VAE.
        
#         Args:
#             x: Input tensor. Can be:
#                - [B, V, C, T, H, W] for multi-view data
#                - [B, C, T, H, W] for single-view data (will be reshaped to [B, 1, C, T, H, W])
#             scale: Scaling factors for latent space. Can be:
#                - None (defaults to zeros and ones for no scaling)
#                - [scale_0, scale_1] where z_scaled = (z - scale_0) * scale_1
            
#         Returns:
#             x_rec: Reconstructed tensor with same shape as input
#             posterior: Posterior distribution (mu, logvar) as tuple [2, B, C_latent, T, H, W]
#             z: Latent code [B, C_latent, T, H, W]
#         """
#         # Handle shape: if 5D, reshape to 6D (add view dimension)
#         if x.dim() == 5:
#             # Single-view: [B, C, T, H, W] -> [B, 1, C, T, H, W]
#             x = x.unsqueeze(1)
        
#         assert x.dim() == 6, f"Expected 5D or 6D input, got {x.dim()}D with shape {x.shape}"
#         assert x.shape[1] == self.view_in, f"Expected {self.view_in} views, got {x.shape[1]}"
        
#         # Handle scale parameter - needs to be [z_dim] shaped tensors
#         if scale is None:
#             # Default: no scaling (scale by 1, shift by 0)
#             # Create as tensors with shape [z_dim] to match VAE expectations
#             scale = [torch.zeros(self.z_dim, dtype=x.dtype, device=x.device),
#                     torch.ones(self.z_dim, dtype=x.dtype, device=x.device)]
#         elif isinstance(scale, (int, float)):
#             # Convert single value - treat as shift, scale by 1
#             scale = [torch.full((self.z_dim,), float(scale), dtype=x.dtype, device=x.device),
#                     torch.ones(self.z_dim, dtype=x.dtype, device=x.device)]
#         elif isinstance(scale, (list, tuple)) and len(scale) == 2:
#             # Convert list elements to tensors if needed
#             scale = [
#                 torch.as_tensor(scale[0], dtype=x.dtype, device=x.device) if not isinstance(scale[0], torch.Tensor) else scale[0].to(dtype=x.dtype, device=x.device),
#                 torch.as_tensor(scale[1], dtype=x.dtype, device=x.device) if not isinstance(scale[1], torch.Tensor) else scale[1].to(dtype=x.dtype, device=x.device)
#             ]
#             # Ensure they have shape [z_dim]
#             if scale[0].dim() == 0:
#                 scale[0] = scale[0].unsqueeze(0).expand(self.z_dim)
#             if scale[1].dim() == 0:
#                 scale[1] = scale[1].unsqueeze(0).expand(self.z_dim)
#         # Otherwise assume it's already in correct format
        
#         # Encode to latent space
#         z = self.vae.encode(x, scale)
        
#         # Sample from posterior (using mean for deterministic reconstruction)
#         # The posterior is already returned from encode as [2, ...] format (mu, logvar)
#         mu = z  # For VAE, z is the latent code
#         logvar = torch.zeros_like(mu)
#         posterior = (mu, logvar)
        
#         # Decode from latent space
#         x_rec = self.vae.decode(z, scale)
        
#         # NOTE: View-conditional decoding would require wrapping the base VAE decoder itself,
#         # which is complex due to the VAE's internal architecture. The ViewConditionalDecoder
#         # is available for future integration directly into MultiViewVideoVAE_ decode loop.
#         # For now, view differentiation relies on strong view embeddings applied in forward pass.
        
#         return x_rec, posterior, z
    
#     def encode(self, x, scale=0):
#         """Encode input to latent space."""
#         return self.vae.encode(x, scale)
    
#     def decode(self, z, scale=0):
#         """Decode latent code to reconstruction."""
#         return self.vae.decode(z, scale)
    
#     def get_last_layer(self):
#         """Get the last layer for adversarial loss computation."""
#         if hasattr(self.vae, "base") and hasattr(self.vae.base, "decoder"):
#             # Return the final output layer of the decoder
#             return self.vae.base.decoder[-1]
#         return None


class MultiviewWanVideoVAE(nn.Module):
    """
    Wan 2.1 extended to 4D (view dimension).

    - Per-view encoding with shared Wan 3D encoder
    - Latent fusion to ONE latent
    - Latent expansion back to per-view
    - Temporal downsampling layers frozen
    - Learnable view embeddings
    """

    def __init__(
        self,
        dim=96,
        z_dim=16,
        view_in=2,
        view_compression=1,
        from_pretrained=None,
        freeze_temporal=True,
        train_spatial=True,
        use_view_embedding=False,
        use_view_group_fusion=True,
        debug_shapes=False,
        device_map=None,
        torch_dtype=None,
        use_crossview_encoder: bool = True,
        use_lora: bool = True,
        fusion_mode: str = "cross_attention",
        use_lora_before: bool = False,
        use_lora_after: bool = True,
        use_viewwise_decoder_lora: bool = False,
        lora_rank: int = 16,
        full_finetune_decoder: bool = False,
        # Per-view reference for the paper: views are folded into the batch and each
        # one goes through plain Wan (+LoRA) with no fusion and no view conditioning.
        # Same trainer/losses/eval as the fused model, so the comparison is clean.
        independent_views: bool = False,
        temporal_compression: bool = True,
        crossview_grad_checkpoint: bool = False,
        crossview_grad_checkpoint_encoder: bool = None,
        crossview_grad_checkpoint_decoder: bool = None,
        view_attn_num_heads: int = None,
        use_noncausal_decode: bool = False,
        use_temporal_reflection_pad: bool = False,
        use_temporal_side_channel: bool = False,
        side_channel_dim: int = 4,
        use_decoder_temporal_attention: bool = False,
        use_learned_cache_update: bool = False,
        use_subframe_position_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.view_in = view_in
        self.z_dim = z_dim
        self.view_compression = view_compression
        self.use_view_embedding = use_view_embedding
        self.use_view_group_fusion = use_view_group_fusion
        self.debug_shapes = debug_shapes
        self.use_crossview_encoder = use_crossview_encoder
        self.independent_views = independent_views
        if independent_views:
            if not use_crossview_encoder:
                raise ValueError("independent_views=True requires use_crossview_encoder=True")
            # Build the inner model single-view with no fusion; the real view count
            # (self.view_in) only matters for the fold/unfold in forward().
            fusion_mode = "none"
            use_viewwise_decoder_lora = False

        # Views going into latent_fusion: after group fusion (if used) or all views
        self.view_out = max(1, view_in // view_compression) if use_view_group_fusion else view_in

        # --------------------------------------------------
        # 1️⃣ Choose backend VAE
        # --------------------------------------------------
        if self.use_crossview_encoder:
            # New encoder: AttentionMultiViewVideoVan with internal cross-view fusion and LoRA
            self.crossview_vae = AttentionMultiViewVideoVan(
                dim=dim,
                z_dim=z_dim,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0,
                use_lora=use_lora,
                lora_rank=lora_rank,
                fusion_mode=fusion_mode,
                use_lora_before=use_lora_before,
                use_lora_after=use_lora_after,
                use_viewwise_decoder_lora=use_viewwise_decoder_lora,
                num_views=1 if independent_views else view_in,
                temporal_compression=temporal_compression,
                grad_checkpoint=crossview_grad_checkpoint,
                grad_checkpoint_encoder=crossview_grad_checkpoint_encoder,
                grad_checkpoint_decoder=crossview_grad_checkpoint_decoder,
                view_attn_num_heads=view_attn_num_heads,
                use_noncausal_decode=use_noncausal_decode,
                use_temporal_reflection_pad=use_temporal_reflection_pad,
                use_temporal_side_channel=use_temporal_side_channel,
                side_channel_dim=side_channel_dim,
                use_decoder_temporal_attention=use_decoder_temporal_attention,
                use_learned_cache_update=use_learned_cache_update,
                use_subframe_position_embedding=use_subframe_position_embedding,
            )

            # Optionally load Wan 2.1 weights into the internal encoder/decoder
            if from_pretrained is not None:
                checkpoint = torch.load(from_pretrained, map_location="cpu")
                inner = checkpoint.get("model", checkpoint)
                try:
                    result = self.crossview_vae.load_state_dict(inner, strict=False)
                    n_missing, n_unexp = len(result.missing_keys), len(result.unexpected_keys)
                    n_ckpt = len(inner)
                    print(f"[MultiviewWanVideoVAE] Loaded Wan 2.1 into crossview_vae: {from_pretrained}")
                    print(f"  -> {n_ckpt} ckpt keys; {n_missing} missing, {n_unexp} unexpected")
                except Exception as e:
                    print(f"Warning: failed to load pretrained weights into crossview_vae: {e}")

            # Train full decoder conv/attn weights inside LoRA wrappers (still uses view_idx + view embeddings).
            if full_finetune_decoder:
                _unfreeze_lora_wrapped_bases(self.crossview_vae.decoder)
                print(
                    "[MultiviewWanVideoVAE] full_finetune_decoder=True: unfrozen base weights in "
                    "crossview_vae.decoder LoRA wrappers (LoRA deltas remain trainable)."
                )

            # --------------------------------------------------
            # Freeze the PRE-FUSION encoder (encoder.conv1 + downsamples) so we
            # reuse the pretrained Wan features. On the crossview path these flags
            # used to be no-ops (only printed in the design summary); here we make
            # them actually take effect. NOTE: encoder.middle / encoder.head are
            # POST-fusion bottleneck layers (LoRA-after wrapped) and are left
            # untouched; fusion blocks, view embeddings/LoRA, and the decoder are
            # also untouched.
            prefusion = [self.crossview_vae.encoder.conv1, self.crossview_vae.encoder.downsamples]
            n_frozen = 0
            if not train_spatial:
                # Freeze the entire pre-fusion encoder (spatial AND temporal convs).
                for mod in prefusion:
                    for p in mod.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            n_frozen += p.numel()
                print(
                    f"[MultiviewWanVideoVAE] train_spatial=False: froze entire pre-fusion encoder "
                    f"(conv1 + downsamples), {n_frozen/1e6:.2f}M params."
                )
            elif freeze_temporal:
                # Freeze only the temporal convs in the pre-fusion encoder.
                for mod in prefusion:
                    for name, p in mod.named_parameters():
                        if p.requires_grad and ("temporal" in name.lower() or "time_conv" in name.lower()):
                            p.requires_grad = False
                            n_frozen += p.numel()
                print(
                    f"[MultiviewWanVideoVAE] freeze_temporal=True (train_spatial=True): froze temporal "
                    f"convs in pre-fusion encoder, {n_frozen/1e6:.2f}M params."
                )

            # For the cross-view encoder path, we do NOT use latent_fusion / latent_expand / group_fusion;
            # all multi-view fusion happens inside AttentionMultiViewVideoVan.
            self.latent_fusion = None
            self.latent_expand = None
            self.view_group_fusion = None
            self.view_embedding = None
        else:
            # Original base VAE path: WanVideoVAE + latent fusion/expansion
            # The current WanVideoVAE constructor only accepts `z_dim`;
            # all other architecture hyperparams are baked into the
            # pretrained 2.1 checkpoint and cannot be overridden, so we
            # ignore `dim`, `dim_mult`, etc.  Keep them in the signature for
            # backwards compatibility, but pass only z_dim here.
            self.base_vae = WanVideoVAE(z_dim=z_dim)

            if from_pretrained is not None:
                # WanVideoVAE wraps a VideoVAE_ (base_vae.model). Checkpoints may be:
                # - full state_dict with "model" key -> inner has keys encoder.xxx, decoder.xxx
                # - or top-level keys model.encoder.xxx (then load into base_vae)
                # If we load inner dict into base_vae, no keys match -> decoder stays random -> white output.
                checkpoint = torch.load(from_pretrained, map_location="cpu")
                if "model" in checkpoint:
                    inner = checkpoint["model"]
                else:
                    inner = checkpoint
                try:
                    result = self.base_vae.load_state_dict(inner, strict=False)
                    n_missing, n_unexp = len(result.missing_keys), len(result.unexpected_keys)
                    n_ckpt = len(inner)
                    if n_missing > 80:
                        # Checkpoint likely has inner keys (encoder.xxx); load into base_vae.model
                        result = self.base_vae.model.load_state_dict(inner, strict=False)
                        n_missing, n_unexp = len(result.missing_keys), len(result.unexpected_keys)
                        print(f"[MultiviewWanVideoVAE] Loaded Wan 2.1 into base_vae.model: {from_pretrained}")
                    else:
                        print(f"[MultiviewWanVideoVAE] Loaded Wan 2.1 into base_vae: {from_pretrained}")
                    print(f"  -> {n_ckpt} ckpt keys; {n_missing} missing, {n_unexp} unexpected")
                    if n_missing > 80:
                        print(f"  -> WARNING: base VAE may not have loaded; reconstruction may be white.")
                    if result.missing_keys and n_missing <= 20:
                        print(f"  -> missing: {result.missing_keys}")
                    if result.unexpected_keys and n_unexp <= 10:
                        print(f"  -> unexpected: {result.unexpected_keys}")
                except Exception as e:
                    print(f"Warning: failed to load pretrained weights: {e}")

            # --------------------------------------------------
            # 2️⃣ Latent Fusion (V_out → 1)
            # --------------------------------------------------
            # use `view_out` (views after compression) as input channels multiplier
            self.latent_fusion = nn.Conv3d(
                in_channels=self.view_out * z_dim,
                out_channels=z_dim,
                kernel_size=1,
                bias=False,
            )

            # --------------------------------------------------
            # 3️⃣ Latent Expansion (1 → V_in)
            # --------------------------------------------------
            # expansion always needs to produce the original number of views
            self.latent_expand = nn.Conv3d(
                in_channels=z_dim,
                out_channels=view_in * z_dim,
                kernel_size=1,
                bias=False,
            )

            # --------------------------------------------------
            # 4️⃣ Learnable view embeddings
            # --------------------------------------------------
            if use_view_embedding:
                self.view_embedding = nn.Parameter(
                    torch.randn(view_in, z_dim) * 0.02
                )
            else:
                self.view_embedding = None

            # Optional: group-wise fusion (view_compression views -> 1 per group). If disabled, only latent_fusion is used.
            if use_view_group_fusion and self.view_compression > 1:
                k = self.view_compression
                self.view_group_fusion = nn.Conv3d(
                    in_channels=k * z_dim,
                    out_channels=z_dim,
                    kernel_size=1,
                    bias=False,
                )
            else:
                self.view_group_fusion = None

            # --------------------------------------------------
            # 5️⃣ Freeze Temporal Layers
            # --------------------------------------------------
            # Base VAE (DiffSynth WanVideoVAE) is created with .requires_grad_(False).
            # Unfreeze base so we can selectively freeze only temporal when train_spatial.
            if train_spatial:
                for param in self.base_vae.parameters():
                    param.requires_grad = True
            if freeze_temporal:
                for name, param in self.base_vae.named_parameters():
                    if "temporal" in name.lower():
                        param.requires_grad = False

            # Optionally freeze everything in base except our fusion/expand
            if not train_spatial:
                for param in self.base_vae.parameters():
                    param.requires_grad = False

        # Device + dtype
        if device_map:
            self.to(device_map)
        if torch_dtype:
            self.to(torch_dtype)

        if not self.use_crossview_encoder:
            self.init_multiview_layers()

    def init_multiview_layers(self):
        """Initializes multi-view layers for training stability."""
        print("Initializing multi-view layers...")
        C = self.z_dim
        V_in = self.view_in
        V_out = self.view_out

        # 1a. When view_compression > 1, view_group_fusion does 2 views -> 1; init as average
        if self.view_group_fusion is not None:
            with torch.no_grad():
                k = self.view_compression
                self.view_group_fusion.weight.zero_()
                for i in range(C):
                    for v in range(k):
                        self.view_group_fusion.weight[i, i + v * C, 0, 0, 0] = 1.0 / k
                if self.view_group_fusion.bias is not None:
                    self.view_group_fusion.bias.zero_()

        # 1b. Latent fusion (V_out channels -> z_dim): init as average over views
        if hasattr(self, 'latent_fusion') and isinstance(self.latent_fusion, nn.Conv3d):
            with torch.no_grad():
                V = V_out  # number of view channels going into latent_fusion
                self.latent_fusion.weight.zero_()
                for i in range(C):
                    for v in range(V):
                        self.latent_fusion.weight[i, i + v * C, 0, 0, 0] = 1.0 / V
                if self.latent_fusion.bias is not None:
                    self.latent_fusion.bias.zero_()

        # 2. Initialize Expansion Layer so each view gets a COPY of the fused latent
        # (Zero init caused decoder to receive zeros -> colored squares / no head.)
        if hasattr(self, 'latent_expand') and isinstance(self.latent_expand, nn.Conv3d):
            with torch.no_grad():
                self.latent_expand.weight.zero_()
                for v in range(V_in):
                    for i in range(C):
                        self.latent_expand.weight[v * C + i, i, 0, 0, 0] = 1.0
                if self.latent_expand.bias is not None:
                    self.latent_expand.bias.zero_()

        # 3. Initialize View Embeddings (attribute is view_embedding, not view_embeddings)
        if hasattr(self, 'view_embedding') and self.view_embedding is not None:
            nn.init.trunc_normal_(self.view_embedding, std=0.02)

    # --------------------------------------------------
    # ENCODE
    # --------------------------------------------------
    def encode(self, x):
        # x: [B, V, C, T, H, W]
        if self.use_crossview_encoder:
            raise NotImplementedError("encode() is not used in crossview mode; call forward(x) instead.")

        B, V, C, T, H, W = x.shape
        latents = []

        # encode each view using single_encode (temporal downsampling inside base VAE)
        for v in range(V):
            z_v = self.base_vae.single_encode(x[:, v], x.device)
            latents.append(z_v)
        # stack → [B, V, Z, T', H', W']
        z_stack = torch.stack(latents, dim=1)

        # optionally add view embeddings
        if self.use_view_embedding and self.view_embedding is not None:
            emb = self.view_embedding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            z_stack = z_stack + emb  # broadcasting over batch and spatial/temporal dims

        # Optional group-wise fusion (only when view_group_fusion is used)
        if self.view_group_fusion is not None and V > 1:
            k = self.view_compression
            assert V % k == 0, "view_in must be divisible by view_compression"
            V_out = V // k
            B2, V2, Z2, T2, H2, W2 = z_stack.shape
            assert B2 == B and V2 == V and Z2 == self.z_dim
            # reshape to groups: [B, V_out, k, Z, T, H, W]
            z_stack = z_stack.view(B, V_out, k, Z2, T2, H2, W2)
            # merge k and Z into channel dimension: [B, V_out, k*Z, T, H, W]
            z_stack = z_stack.permute(0, 1, 3, 2, 4, 5, 6).reshape(B, V_out, k * Z2, T2, H2, W2)
            # fuse each group with the learned conv: merge batch and V_out dims
            z_stack = z_stack.view(B * V_out, k * Z2, T2, H2, W2)
            z_stack = self.view_group_fusion(z_stack)
            # result [B*V_out, Z, T, H, W] -> reshape back
            z_stack = z_stack.view(B, V_out, Z2, T2, H2, W2)
            V = V_out

        # reshape for fusion: [B, V*Z, T', H', W']
        B, V, Z, T2, H2, W2 = z_stack.shape
        z_stack = z_stack.reshape(B, V * Z, T2, H2, W2)

        # fuse
        z = self.latent_fusion(z_stack)
        return z

    # --------------------------------------------------
    # DECODE
    # --------------------------------------------------
    def decode(self, z):
        if self.use_crossview_encoder:
            raise NotImplementedError("decode() is not used in crossview mode; call forward(z) via forward(x) instead.")

        # z: [B, Z, T', H', W']
        B, Z, T2, H2, W2 = z.shape

        # expand latent
        z_expand = self.latent_expand(z)
        # reshape to per-view
        z_expand = z_expand.view(B, self.view_in, Z, T2, H2, W2)

        # decode each view using single_decode (temporal upsampling inside base VAE)
        recons = [self.base_vae.single_decode(z_expand[:, v], z_expand.device) for v in range(self.view_in)]

        # stack → [B, V, C, T, H, W]
        x_rec = torch.stack(recons, dim=1)
        return x_rec

    # --------------------------------------------------
    # FORWARD
    # --------------------------------------------------
    def forward(self, x):
        if self.use_crossview_encoder and self.independent_views:
            # Per-view reference: fold views into batch, run each through the
            # single-view model, unfold. One latent PER VIEW (V x the latent rate
            # of the fused model) -- that is the point of this reference.
            B, V, C, T, H, W = x.shape
            x_flat = x.reshape(B * V, 1, C, T, H, W)
            scale = [
                torch.zeros(self.z_dim, dtype=x.dtype, device=x.device),
                torch.ones(self.z_dim, dtype=x.dtype, device=x.device),
            ]
            mu, logvar = self.crossview_vae.encode(x_flat, scale)
            z = self.crossview_vae.reparameterize(mu, logvar)
            rec = self.crossview_vae.decode(z, scale, view_idx=0)  # [B*V, C, T, H, W]
            x_rec = rec.reshape(B, V, C, T, H, W)
            return x_rec, (mu, logvar), z

        if self.use_crossview_encoder:
            # x: [B, V, C, T, H, W] with V == view_in (typically 2)
            B, V, C, T, H, W = x.shape
            assert V == self.view_in, f"Expected {self.view_in} views, got {V}"

            # Build default scale tensors if needed: [z_dim] each
            scale = [
                torch.zeros(self.z_dim, dtype=x.dtype, device=x.device),
                torch.ones(self.z_dim, dtype=x.dtype, device=x.device),
            ]

            # Encode once with the cross-view encoder
            mu, logvar = self.crossview_vae.encode(x, scale)
            z = self.crossview_vae.reparameterize(mu, logvar)

            # Decode once per view using view-conditioned embeddings
            # VIEW_EMBEDDINGS 3: loop over views
            recons = []
            with ProfileTimer.block("decode.all_views"):
                for v_idx in range(V):
                    with ProfileTimer.block(f"decode.view_{v_idx}"):
                        rec_v = self.crossview_vae.decode(z, scale, view_idx=v_idx)
                    recons.append(rec_v)

            # Stack → [B, V, C, T, H, W]
            x_rec = torch.stack(recons, dim=1)
            posterior = (mu, logvar)
            return x_rec, posterior, z

        # Original path
        z = self.encode(x)
        mu = z
        logvar = torch.zeros_like(mu)
        posterior = (mu, logvar)
        x_rec = self.decode(z)
        return x_rec, posterior, z
    
    def load_state_dict(self, state_dict, strict: bool = False, **kwargs):
        """
        Backward-compat helper.

        Some ColossalAI/ZeRO checkpoints store certain tensors flattened (1D) even
        when the target module expects multi-dim weights (e.g. Conv3d weights or
        RMSNorm gamma broadcast shapes). When the flattened tensor has the same
        numel as the target parameter, reshape it before loading.
        """
        if isinstance(state_dict, dict):
            expected = self.state_dict()
            reshaped = {}
            for k, v in state_dict.items():
                if k in expected:
                    tgt = expected[k]
                    # Reshape only when shapes are different but total size matches,
                    # and the checkpoint tensor looks like a flattened view.
                    if (
                        hasattr(v, "ndim")
                        and hasattr(tgt, "ndim")
                        and v.ndim == 1
                        and tgt.ndim > 1
                        and v.numel() == tgt.numel()
                    ):
                        reshaped[k] = v.view_as(tgt)
                    else:
                        reshaped[k] = v
                else:
                    reshaped[k] = v
            state_dict = reshaped
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
    
    def get_last_layer(self):
        """Get the last layer for adversarial loss computation."""
        def _pick_tensor_from_layer(layer: nn.Module):
            """
            GeneratorLoss expects `last_layer` to be a Tensor/Parameter (or iterable of Tensors),
            because it calls `torch.autograd.grad(nll_loss, last_layer, ...)`.
            """
            # Common case: Conv/Linear with a `weight` tensor.
            if hasattr(layer, "weight") and isinstance(getattr(layer, "weight"), torch.Tensor):
                return getattr(layer, "weight")

            # LoRAConv3d wrapper in DiffSynth has `lora_up` / `lora_down`; use the final trainable weight.
            if hasattr(layer, "lora_up") and hasattr(layer.lora_up, "weight"):
                return layer.lora_up.weight

            # Otherwise, pick any trainable parameter tensor.
            trainable_params = [p for p in layer.parameters() if p.requires_grad]
            if trainable_params:
                return trainable_params[-1]

            # Fall back to the first parameter tensor (may be frozen, but prevents crashes).
            params = list(layer.parameters())
            return params[-1] if params else None

        # Original (non-crossview) path.
        if hasattr(self, "base_vae") and hasattr(self.base_vae, "model") and hasattr(self.base_vae.model, "decoder"):
            layer = self.base_vae.model.decoder[-1]
            return _pick_tensor_from_layer(layer)

        # Crossview encoder path (use_crossview_encoder=True) does not define `base_vae`.
        if hasattr(self, "crossview_vae") and hasattr(self.crossview_vae, "decoder"):
            dec = self.crossview_vae.decoder
            if hasattr(dec, "head") and isinstance(dec.head, nn.Sequential) and len(dec.head) > 0:
                layer = dec.head[-1]
                return _pick_tensor_from_layer(layer)
            # Fallback: use decoder itself.
            return _pick_tensor_from_layer(dec)

        return None

@MODELS.register_module("multiview_wan_video_vae")
def build_multiview_wan_video_vae(
    dim=96,
    z_dim=16,
    view_in=2,
    view_compression=2,
    use_view_embedding=True,
    use_view_group_fusion=True,
    from_pretrained=None,
    device_map=None,
    torch_dtype=None,
    use_crossview_encoder: bool = True,
    use_lora: bool = True,
    fusion_mode: str = "cross_attention",
    use_lora_before: bool = False,
    use_lora_after: bool = True,
    use_viewwise_decoder_lora: bool = False,
    lora_rank: int = 16,
    full_finetune_decoder: bool = False,
    temporal_compression: bool = True,
    crossview_grad_checkpoint: bool = False,
    crossview_grad_checkpoint_encoder: bool = None,
    crossview_grad_checkpoint_decoder: bool = None,
    view_attn_num_heads: int = None,
    use_noncausal_decode: bool = False,
    use_temporal_reflection_pad: bool = False,
    use_temporal_side_channel: bool = False,
    side_channel_dim: int = 4,
    use_decoder_temporal_attention: bool = False,
    use_learned_cache_update: bool = False,
    use_subframe_position_embedding: bool = False,
    **kwargs,
):
    """
    Factory function to build a MultiviewWanVideoVAE model.
    
    This function is registered with the opensora model registry and can be called via:
        model = build_module(cfg.model, MODELS)
    
    where cfg.model contains the above parameters.
    
    This uses the MultiviewWanVideoVAE implementation which:
    - Uses convolution for proper view fusion (as recommended by your supervisor)
    - Has correct posterior handling
    - Uses learned view embeddings for view differentiation
    - Follows the Wan 2.1 architecture properly
    
    The view_in parameter specifies the number of input views (e.g., 2 for stereo)
    The view_compression parameter controls how many views are compressed into one latent
    Convolution is used for view fusion as this is the proper approach for combining
    separate encoded views into a single latent representation.
    """
    
    # Use the MultiviewWanVideoVAE implementation with convolution-based view fusion
    # This is the correct approach as recommended by your supervisor
    model = MultiviewWanVideoVAE(
        dim=dim,
        z_dim=z_dim,
        view_in=view_in,
        view_compression=view_compression,
        use_view_embedding=use_view_embedding,
        use_view_group_fusion=use_view_group_fusion,
        from_pretrained=from_pretrained,
        device_map=device_map,
        torch_dtype=torch_dtype,
        use_crossview_encoder=use_crossview_encoder,
        use_lora=use_lora,
        fusion_mode=fusion_mode,
        use_lora_before=use_lora_before,
        use_lora_after=use_lora_after,
        use_viewwise_decoder_lora=use_viewwise_decoder_lora,
        lora_rank=lora_rank,
        full_finetune_decoder=full_finetune_decoder,
        temporal_compression=temporal_compression,
        crossview_grad_checkpoint=crossview_grad_checkpoint,
        crossview_grad_checkpoint_encoder=crossview_grad_checkpoint_encoder,
        crossview_grad_checkpoint_decoder=crossview_grad_checkpoint_decoder,
        view_attn_num_heads=view_attn_num_heads,
        use_noncausal_decode=use_noncausal_decode,
        use_temporal_reflection_pad=use_temporal_reflection_pad,
        use_temporal_side_channel=use_temporal_side_channel,
        side_channel_dim=side_channel_dim,
        use_decoder_temporal_attention=use_decoder_temporal_attention,
        use_learned_cache_update=use_learned_cache_update,
        use_subframe_position_embedding=use_subframe_position_embedding,
        **kwargs,
    )

    return model
