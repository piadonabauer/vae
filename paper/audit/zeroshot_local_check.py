#!/usr/bin/env python3
"""Local zero-shot baseline wiring checks (2026-10-02 audit, Part 2).

Runs the PRETRAINED Wan 2.1 VAE (Wan2.1_VAE.pth from HF Wan-AI/Wan2.1-T2V-1.3B)
on a natural 720p clip (jellyfish, real footage) through TWO paths:

  A. "native"  - the stock WanVideoVAE_ chunked causal path (1+4+4 temporal
     chunks, feat_cache, deterministic mu encode). This is what "zero-shot
     Wan 2.1" should mean.
  B. "E1z"     - the repo's zero-shot arm configuration: AttentionMultiView-
     VideoVan with independent_views (num_views=1, fusion_mode='none'),
     use_lora=True (zero-init), temporal_compression=False -> the temporal
     stride convs are SKIPPED on encode and every latent frame is decoded
     INDEPENDENTLY (no cache), plus z is SAMPLED from the posterior
     (reparameterize) exactly as evaluate_model's forward does.

Also: resolution ladder (native res / 256 / 128), z=mu vs z~posterior,
temporal misalignment probe (rec_t vs gt_{t-1}, gt_t, gt_{t+1}), and
bf16 vs fp32. All metrics via metrics_unified (declared-range API).

NeRSemble clips are NOT available on this machine (dataset lives on the
cluster), so the NeRSemble resolution/matting ablations must be run there;
this script is written so it can be pointed at a NeRSemble .pt via
--nersemble_pt when run on the cluster.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics_unified import compute_all_metrics  # noqa: E402

WAN_VAE_FILE = "/home/coder/vae/DiffSynth-Studio/diffsynth/models/wan_video_vae.py"
WEIGHTS = os.path.join(HERE, "zeroshot", "Wan2.1_VAE.pth")
VIDEO = os.path.join(HERE, "zeroshot", "jellyfish_720p.mp4")


def load_wan_module():
    """Load the DiffSynth fork's wan_video_vae.py WITHOUT importing the
    diffsynth package (its __init__ chain needs unavailable deps)."""
    spec = importlib.util.spec_from_file_location("wan_vae_mod", WAN_VAE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def remap_wan_keys_for_lora(ckpt: dict, model_sd: dict) -> dict:
    """Copy of Open-Sora/opensora/models/vae/wan_video_vae.py::_remap_wan_keys_for_lora
    (that module imports opensora.registry, unavailable here)."""
    remapped = {}
    for k, v in ckpt.items():
        if k not in model_sd:
            stem, _, leaf = k.rpartition(".")
            candidate = f"{stem}.base_conv.{leaf}"
            if candidate in model_sd:
                remapped[candidate] = v
                continue
        remapped[k] = v
    return remapped


def read_clip(path: str, n_frames: int = 9, stride: int = 3) -> torch.Tensor:
    """Read n_frames RGB frames -> [1, 3, T, H, W] float in [0,1].
    imageio/pyav give RGB (not BGR). stride>1 adds visible motion."""
    import imageio.v3 as iio

    frames = []
    for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
        if i % stride == 0:
            frames.append(torch.from_numpy(fr.copy()))
        if len(frames) == n_frames:
            break
    x = torch.stack(frames)  # [T, H, W, 3] uint8 RGB
    x = x.permute(3, 0, 1, 2).float() / 255.0  # [3, T, H, W]
    return x.unsqueeze(0)


def resize_clip(x01: torch.Tensor, size: int | None) -> torch.Tensor:
    """Center-crop to square and antialiased-bilinear resize [1,3,T,H,W]."""
    _, c, t, h, w = x01.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    x01 = x01[:, :, :, y0 : y0 + s, x0 : x0 + s]
    if size is None:
        return x01
    flat = x01.squeeze(0).permute(1, 0, 2, 3)  # [T, 3, S, S]
    flat = torch.nn.functional.interpolate(
        flat, size=(size, size), mode="bilinear", antialias=True, align_corners=False
    )
    return flat.permute(1, 0, 2, 3).unsqueeze(0)


@torch.no_grad()
def run_native(mod, sd, x01, device, dtype):
    """Stock chunked causal Wan path, deterministic mu encode."""
    vae = mod.WanVideoVAE(z_dim=16)
    missing, unexpected = vae.model.load_state_dict(sd, strict=False)
    assert not unexpected, f"native load: unexpected keys {unexpected[:5]}"
    vae = vae.eval().to(device, dtype)
    xm1 = (x01 * 2 - 1).to(device, dtype)  # [1,3,T,H,W] in [-1,1]
    z = vae.model.encode(xm1, vae.scale)          # mu, chunked 1+4+4, cache reset inside
    rec = vae.model.decode(z, vae.scale)          # chunked, cache reset inside
    rec = rec.clamp(-1, 1)
    return xm1.float().cpu(), rec.float().cpu()


@torch.no_grad()
def build_e1z(mod, sd, device, dtype, temporal_compression=False):
    """The repo zero-shot arm model: independent views, LoRA on, TC flag as given."""
    vae = mod.AttentionMultiViewVideoVan(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
        attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0,
        use_lora=True, lora_rank=16, fusion_mode="none",
        use_lora_before=False, use_lora_after=True,
        use_viewwise_decoder_lora=False, num_views=1,
        temporal_compression=temporal_compression,
    )
    remapped = remap_wan_keys_for_lora(sd, vae.state_dict())
    res = vae.load_state_dict(remapped, strict=False)
    dropped = [k for k in res.unexpected_keys if k.startswith(("encoder.", "decoder.", "conv1", "conv2"))]
    assert not dropped, f"E1z load dropped pretrained keys: {dropped[:5]}"
    return vae.eval().to(device, dtype)


@torch.no_grad()
def run_e1z(vae, x01, device, dtype, sample_posterior):
    xm1 = (x01 * 2 - 1).to(device, dtype).unsqueeze(1)  # [1,1,3,T,H,W] (V=1)
    scale = [torch.zeros(16, dtype=xm1.dtype, device=device),
             torch.ones(16, dtype=xm1.dtype, device=device)]
    mu, logvar = vae.encode(xm1, scale)
    z = vae.reparameterize(mu, logvar) if sample_posterior else mu
    rec = vae.decode(z, scale, view_idx=0).clamp(-1, 1)  # [1,3,T,H,W]
    stats = {
        "posterior_std_mean": float(torch.exp(0.5 * logvar).mean().float().cpu()),
        "latent_T": int(mu.shape[2]),
    }
    return xm1.squeeze(1).float().cpu(), rec.float().cpu(), stats


def psnr_shifted(gt_m1, rec_m1, shift):
    """PSNR of rec frame t vs gt frame t+shift (both [-1,1], [1,3,T,H,W])."""
    t = gt_m1.shape[2]
    if shift >= 0:
        g, r = gt_m1[:, :, shift:], rec_m1[:, :, : t - shift]
    else:
        g, r = gt_m1[:, :, : t + shift], rec_m1[:, :, -shift:]
    m = compute_all_metrics(g, r, value_range="[-1,1]")
    return m["psnr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nersemble_pt", default=None,
                    help="optional NeRSemble .pt clip [V,C,T,H,W] in [0,1] (cluster only)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    mod = load_wan_module()
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    print(f"weights: {len(sd)} keys")

    results = {}

    def record(name, gt_m1, rec_m1, extra=None):
        m = compute_all_metrics(gt_m1, rec_m1, value_range="[-1,1]")
        row = {"psnr": round(m["psnr"], 3), "ssim": round(m["ssim"], 5),
               "bleed_within": round(m.get("bleed_ratio_within", float("nan")), 4),
               "bleed_across": round(m.get("bleed_ratio_across", float("nan")), 4),
               "psnr_per_frame": [round(p, 2) for p in m["psnr_per_frame"]]}
        if extra:
            row.update(extra)
        results[name] = row
        print(f"{name:52s} PSNR {row['psnr']:7.3f}  SSIM {row['ssim']:.5f}  "
              f"bleedW {row['bleed_within']}  bleedA {row['bleed_across']}")
        return row

    clips = {}
    full = read_clip(VIDEO)
    print("clip:", tuple(full.shape))
    clips["natural_704"] = resize_clip(full, 704)   # ~native (720 center square)
    clips["natural_256"] = resize_clip(full, 256)
    clips["natural_128"] = resize_clip(full, 128)

    if args.nersemble_pt:
        ner = torch.load(args.nersemble_pt, map_location="cpu")  # [V,C,T,H,W] [0,1]
        v0 = ner[0:1] if ner.dim() == 5 else ner
        for s in (128, 256, 512):
            clips[f"nersemble_{s}"] = resize_clip(v0.unsqueeze(0) if v0.dim() == 4 else v0, s)

    # ---- A. native chunked Wan, bf16 (matches training dtype) -----------
    for name, clip in clips.items():
        gt, rec = run_native(mod, sd, clip, device, torch.bfloat16)
        record(f"native_TC_chunked/{name}/bf16", gt, rec)

    # fp32 vs bf16 on one config
    gt, rec = run_native(mod, sd, clips["natural_128"], device, torch.float32)
    record("native_TC_chunked/natural_128/fp32", gt, rec)

    # misalignment probe on native path, 128
    gt, rec = run_native(mod, sd, clips["natural_128"], device, torch.bfloat16)
    for sh in (-1, 0, 1):
        print(f"  native 128 shift {sh:+d}: PSNR {psnr_shifted(gt, rec, sh):.3f}")
    results["native_misalign_128"] = {str(sh): round(psnr_shifted(gt, rec, sh), 3) for sh in (-1, 0, 1)}

    # ---- B. repo E1z path (TC=False, per-frame decode) -------------------
    e1z = build_e1z(mod, sd, device, torch.bfloat16, temporal_compression=False)
    for name in ("natural_704", "natural_256", "natural_128"):
        gt, rec, st = run_e1z(e1z, clips[name], device, torch.bfloat16, sample_posterior=True)
        record(f"E1z_TCoff_sampled/{name}/bf16", gt, rec, extra=st)
    # z = mu (no posterior sampling) to isolate the reparameterize noise
    gt, rec, st = run_e1z(e1z, clips["natural_128"], device, torch.bfloat16, sample_posterior=False)
    record("E1z_TCoff_mu/natural_128/bf16", gt, rec, extra=st)
    gt, rec, st = run_e1z(e1z, clips["natural_256"], device, torch.bfloat16, sample_posterior=False)
    record("E1z_TCoff_mu/natural_256/bf16", gt, rec, extra=st)

    # misalignment probe on the E1z path (per-frame decode)
    gt, rec, _ = run_e1z(e1z, clips["natural_128"], device, torch.bfloat16, sample_posterior=True)
    for sh in (-1, 0, 1):
        print(f"  E1z 128 shift {sh:+d}: PSNR {psnr_shifted(gt, rec, sh):.3f}")
    results["e1z_misalign_128"] = {str(sh): round(psnr_shifted(gt, rec, sh), 3) for sh in (-1, 0, 1)}
    del e1z

    # ---- C. same class but TC=True (chunked path inside the repo model) --
    e1z_tc = build_e1z(mod, sd, device, torch.bfloat16, temporal_compression=True)
    for name in ("natural_256", "natural_128"):
        gt, rec, st = run_e1z(e1z_tc, clips[name], device, torch.bfloat16, sample_posterior=True)
        record(f"E1z_TCon_sampled/{name}/bf16", gt, rec, extra=st)
    gt, rec, st = run_e1z(e1z_tc, clips["natural_128"], device, torch.bfloat16, sample_posterior=False)
    record("E1z_TCon_mu/natural_128/bf16", gt, rec, extra=st)

    if args.nersemble_pt:
        for s in (128, 256, 512):
            gt, rec, st = run_e1z(e1z_tc, clips[f"nersemble_{s}"], device, torch.bfloat16, True)
            record(f"E1z_TCon_sampled/nersemble_{s}/bf16", gt, rec, extra=st)

    out = os.path.join(HERE, "zeroshot_local_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
