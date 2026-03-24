#!/usr/bin/env python3
"""Quick forward-pass debug for multiview VAE.

Loads one .pt sample, runs model forward, prints stats, and saves one original/recon pair.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import os
from PIL import Image

# Prefer absolute paths to ensure imports work in different execution contexts
sys.path.insert(0, '/home/piado/projects/aip-lindell/piado/vae/Open-Sora')
sys.path.insert(0, '/home/piado/projects/aip-lindell/piado/vae/DiffSynth-Studio')

# Imports
from opensora.datasets.pt_video_dataset import PTVideoDataset
from opensora.models.vae.wan_video_vae import MultiviewWanVideoVAE


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Use the example dataset file used in config
    pt_path = '/datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res/p018/EXP-1-head/p018_EXP-1-head.pt'
    if not Path(pt_path).exists():
        print('ERROR: pt file not found:', pt_path)
        return 2

    ds = PTVideoDataset(pt_path, repeat=1)
    sample = ds[0]
    video = sample['video']  # [V, C, T, H, W]
    print('Loaded video shape (V,C,T,H,W):', video.shape)
    print('Video dtype:', video.dtype)
    print('Video min/max:', float(video.min()), float(video.max()), 'mean:', float(video.mean()))

    # Add batch dim: [1, V, C, T, H, W]
    x = video.unsqueeze(0).to(device)

    # Build model
    ckpt_path = Path('/home/piado/scratch/Wan2.1_VAE.pth')
    model = MultiviewWanVideoVAE(z_dim=16, view_in=x.shape[1], view_compression=1, from_pretrained=None)
    model = model.to(device)

    # If checkpoint exists, load into base_vae and report matching keys
    if ckpt_path.exists():
        print('Loading checkpoint:', ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        base_sd = model.base_vae.state_dict()
        match_shapes = 0
        for k, v in base_sd.items():
            if k in sd and sd[k].shape == v.shape:
                match_shapes += 1
        print(f'Base VAE params: {len(base_sd)}, matching-shape keys in ckpt: {match_shapes}')
        # Load with strict=False and report missing/unexpected
        res = model.base_vae.load_state_dict(sd, strict=False)
        try:
            missing = len(res.missing_keys)
            unexpected = len(res.unexpected_keys)
            print(f'Loaded checkpoint: missing keys={missing}, unexpected keys={unexpected}')
        except Exception:
            print('Loaded checkpoint (no missing/unexpected info)')

    model.eval()

    # Forward
    with torch.no_grad():
        out = model(x)
    print('Model forward returned length:', len(out) if isinstance(out, (tuple, list)) else 'single')
    if isinstance(out, (tuple, list)):
        x_rec, posterior, z = out
    else:
        x_rec = out
        posterior = None
        z = None

    print('x_rec shape:', x_rec.shape if x_rec is not None else None)
    if z is not None:
        print('z shape:', z.shape)
        print('z min/max/mean:', float(z.min()), float(z.max()), float(z.mean()))
    if posterior is not None:
        if isinstance(posterior, (tuple, list)):
            print('posterior tuple shapes:', [p.shape for p in posterior])
        else:
            print('posterior shape:', getattr(posterior, 'shape', str(type(posterior))))

    print('x_rec min/max/mean:', float(x_rec.min()), float(x_rec.max()), float(x_rec.mean()))

    # --- Additional checks: encode/decode each view directly with base VAE ---
    base = model.base_vae
    per_view_recons = []
    for v in range(x.shape[1]):
        vid = x[0, v].unsqueeze(0)  # [1, C, T, H, W]
        h = base.single_encode(vid, device)
        rec = base.single_decode(h.unsqueeze(0) if h.dim()==5 else h, device) if False else base.single_decode(h, device)
        # rec shape [1, C, T, H, W]
        per_view_recons.append(rec.squeeze(0).cpu())
        # compute simple mse between original and base recon
        orig = vid.squeeze(0).cpu()
        mse = torch.mean((orig - rec.squeeze(0).cpu())**2)
        print(f'Base VAE recon view {v} mse:', float(mse))

    # Compare wrapper output per-view MSE
    for v in range(x_rec.shape[1]):
        orig = x[0, v].cpu()
        rec = x_rec[0, v].cpu()
        mse = torch.mean((orig - rec)**2)
        print(f'Wrapper recon view {v} mse:', float(mse))

    # Save one frame original and recon for view 0, middle frame
    out_dir = Path('/home/piado/tmp_multiview_debug')
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = x[0, 0]  # [C, T, H, W]
    rec = x_rec[0, 0]
    # select middle temporal frame if available
    t = orig.shape[1]
    fi = t // 2
    orig_frame = orig[:, fi].cpu().numpy()  # C,H,W
    rec_frame = rec[:, fi].cpu().numpy()

    # Save frames as PNGs (handle ranges automatically)
    def save_frame_png(arr, path, assume_range=None):
        # arr: numpy array C,H,W
        a = arr.copy()
        # detect range if not provided
        mn = float(a.min())
        mx = float(a.max())
        if assume_range is None:
            # if values go below -0.1 assume [-1,1], else [0,1]
            if mn < -0.1:
                mn, mx = -1.0, 1.0
            else:
                mn, mx = 0.0, 1.0
        else:
            mn, mx = assume_range
        a = (a - mn) / (mx - mn + 1e-8)
        a = (a * 255.0).clip(0, 255).astype('uint8')
        # transpose to H,W,C
        img = a.transpose(1, 2, 0)
        Image.fromarray(img).save(path)

    save_frame_png(orig_frame, out_dir / 'orig_frame.png')
    save_frame_png(rec_frame, out_dir / 'rec_frame.png')
    print('Saved frames to', out_dir)
    # Also save a simple summary text
    with open(out_dir / 'debug.txt', 'w') as f:
        f.write(f'video_shape={tuple(video.shape)}\n')
        f.write(f'video_min={float(video.min())}\n')
        f.write(f'video_max={float(video.max())}\n')
        f.write(f'rec_min={float(x_rec.min())}\n')
        f.write(f'rec_max={float(x_rec.max())}\n')
        # add per-view mse info if computed
        try:
            for v in range(x.shape[1]):
                orig = x[0, v].cpu()
                rec = x_rec[0, v].cpu()
                mse = float(torch.mean((orig - rec)**2))
                f.write(f'view{v}_wrapper_mse={mse}\n')
        except Exception:
            pass
    print('Done')
    return 0

if __name__ == '__main__':
    sys.exit(main())
