#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import sys
import types

import numpy as np
import torch
import matplotlib.pyplot as plt


def pick_upper_cameras(
    root: Path,
    participant_id: int,
    sequence_name: str,
    data_manager,
    top_k: int = 8,
    axis: int = 1,
):
    """
    Select upper cameras by sorting camera centers along a world-axis.
    Assumption: world axis 1 (Y) is "up" for this dataset.
    """
    images_dir = root / f"{participant_id:03d}/sequences/{sequence_name}/images"  # video folder
    if images_dir.exists():
        available = [p.stem.split("_")[1] for p in images_dir.glob("cam_*.mp4")]
    else:
        available = data_manager.list_cameras(sequence_name)

    cam_json = root / f"{participant_id:03d}/calibration/camera_params.json"  # calibration file
    world_2_cam = json.loads(cam_json.read_text())["world_2_cam"]

    centers = {}
    for serial in available:
        w2c = np.array(world_2_cam[serial], dtype=np.float32)  # world-to-camera
        c2w = np.linalg.inv(w2c)
        centers[serial] = c2w[:3, 3]

    ordered = sorted(available, key=lambda s: centers[s][axis], reverse=True)  # highest first
    return ordered[:top_k], centers


def image_to_video_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC float [0,1] -> C,T,H,W float [-1,1] with T=1."""
    tensor = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(1)  # HWC->CTHW
    return tensor * 2.0 - 1.0


def load_wan_video_vae_module(module_path: Path):
    """
    Load the module; if it fails due to commented blocks, strip top-level
    triple-quote delimiters and reload from source.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("wan_video_vae_local", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print("Standard import failed, applying source patch:", exc)
        src = module_path.read_text()
        # Remove top-level triple-quote lines that comment out class blocks.
        src = re.sub(r'(?m)^"""\s*$', "", src)
        patched = types.ModuleType("wan_video_vae_patched")
        exec(compile(src, str(module_path), "exec"), patched.__dict__)
        return patched


def build_arg_parser():
    parser = argparse.ArgumentParser(  # CLI entrypoint
        description="NeRSemble -> WanVideoVAE encode/decode demo"
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/home/piado/projects/aip-lindell/piado"),
    )
    parser.add_argument(
        "--nersemble-root",
        type=Path,
        default=Path("/scratch/piado/data/nersemble"),
    )
    parser.add_argument("--participant", type=int, default=17)
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale factor for images; 0 disables downscaling.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="/home/piado/scratch/Wan2.1_VAE.pth",
        help="Path to pretrained WanVideoVAE checkpoint (ckpt/pt/safetensors).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save output images (default: <workspace>/outputs).",
    )
    parser.add_argument(
        "--no-color-correction",
        action="store_true",
        help="Disable NeRSemble color correction.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()  # parse CLI args
    workspace_root = args.workspace_root  # repo root
    nersemble_root = args.nersemble_root  # dataset root
    diffsynth_root = workspace_root / "vae/DiffSynth-Studio"  # DiffSynth code
    nersemble_pkg_root = workspace_root / "nersemble-data" / "src"  # data utils

    # Local imports for the downloaded dataset.
    sys.path.insert(0, str(nersemble_pkg_root))  # import local nersemble_data
    sys.path.insert(0, str(diffsynth_root))  # import local DiffSynth

    from nersemble_data.data.nersemble_data import (
        NeRSembleDataManager,
        NeRSembleParticipantDataManager,
    )

    # nersemble_data path : /home/piado/projects/aip-lindell/piado/vae/DiffSynth-Studio/diffsynth/core/data/nersemble-data/src/nersemble_data/data/nersemble_data.py

    data_folder = NeRSembleDataManager(str(nersemble_root))  # scan participants
    participants = sorted(data_folder.list_participants())
    if not participants:
        raise RuntimeError("No participants found in the NeRSemble folder.")

    # Take given participant if available, otherwise use the first
    participant_id = args.participant if args.participant in participants else participants[0]  # pick subject
    data_manager = NeRSembleParticipantDataManager(str(nersemble_root), participant_id)

    sequences = data_manager.list_sequences()  # list sequences

    # Prefer given sequence, otherwise use the first non-background
    if args.sequence and args.sequence in sequences:
        sequence_name = args.sequence
    else:
        sequence_name = next((s for s in sequences if s != "BACKGROUND"), sequences[0])

    print("Participant:", participant_id)
    print("Sequence:", sequence_name)
    print("Available sequences:", sequences)

    # Load camera calibration via the data manager
    camera_calibration = data_manager.load_camera_calibration()  # intrinsics/extrinsics
    world_2_cam_poses = camera_calibration.world_2_cam
    intrinsics = camera_calibration.intrinsics

    print("Intrinsics:\n", intrinsics)
    print("Number of cameras in calibration:", len(world_2_cam_poses))

    upper_serials, _camera_centers = pick_upper_cameras(
        nersemble_root,
        participant_id,
        sequence_name,
        data_manager,
        top_k=8,
        axis=1,
    )

    print("Upper-view cameras (8):", upper_serials)

    timestep = args.timestep  # frame index
    downscale_factor = None if args.downscale == 0 else args.downscale  # speed/quality
    apply_color_correction = not args.no_color_correction  # dataset correction

    output_dir = args.output_dir or (workspace_root / "outputs")  # output folder
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for serial in upper_serials:
        img = data_manager.load_image(  # read frame
            sequence_name,
            serial,
            timestep,
            apply_color_correction=apply_color_correction,
            downscale_factor=downscale_factor,
        )
        images.append(img)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 8-view grid
    for ax, serial, img in zip(axes.flat, upper_serials, images):
        ax.imshow(img)
        ax.set_title(f"cam {serial}")
        ax.axis("off")
    plt.tight_layout()
    originals_path = output_dir / f"nersemble_{participant_id:03d}_{sequence_name}_upper8.png"  # save path
    fig.savefig(originals_path, dpi=150)
    plt.close(fig)
    print("Saved original grid to:", originals_path)

    videos = [image_to_video_tensor(img) for img in images]  # to model input
    print("Single-view tensor shape:", videos[0].shape)



    # Load the VAE implementation from DiffSynth-Studio

    vae_module = load_wan_video_vae_module(  # dynamic import
        diffsynth_root / "diffsynth/models/wan_video_vae.py"
    )
    WanVideoVAE = getattr(vae_module, "WanVideoVAE", None)
    #MultiViewWanVideoVAE = getattr(vae_module, "MultiViewWanVideoVAE", None)

    print("WanVideoVAE available:", WanVideoVAE is not None)
    #print("MultiViewWanVideoVAE available:", MultiViewWanVideoVAE is not None)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # choose device

    #use_multiview = MultiViewWanVideoVAE is not None
    #if use_multiview:
    #    vae = MultiViewWanVideoVAE(view_in=len(upper_serials)).to(device)  # multiview
    #else:
    if WanVideoVAE is None:
        raise RuntimeError("WanVideoVAE is not available in the module.")
    vae = WanVideoVAE().to(device)  # single-view

    # Optional: load a checkpoint if you have one.
    checkpoint_path = args.checkpoint
    if checkpoint_path is not None:
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")  # load weights
            converter = vae.state_dict_converter()
            vae.load_state_dict(converter.from_civitai(state), strict=False)
            print("Loaded checkpoint:", checkpoint_path)
        else:
            print("Checkpoint not found:", checkpoint_path)
    else:
        print("Warning: no checkpoint provided; reconstruction will be random.")

    vae.eval()  # inference mode
    print("Using device:", device)
    #print("Multi-view mode:", use_multiview)
    use_multiview = False

    # Main VAE operations: encode and decode
    with torch.no_grad():  # no gradients
        if use_multiview:
            # [B, V, C, T, H, W]
            video_batch = torch.stack(videos).unsqueeze(0)  # [B,V,C,T,H,W]
            latents = vae.encode(video_batch, device=device)
            recon = vae.decode(latents, device=device)
            recon_videos = recon.squeeze(0)
        else:
            latents = vae.encode(videos, device=device)  # encode views
            recon_videos = vae.decode(latents, device=device)

    recon_images = []  # postprocess
    for v in recon_videos:
        img = (v[:, 0].clamp(-1, 1) + 1.0) / 2.0
        recon_images.append(img.permute(1, 2, 0).cpu().numpy())

    print("Latent shape:", latents.shape)
    print("Recon tensor shape:", recon_videos.shape)

    fig, axes = plt.subplots(  # original vs recon
        len(upper_serials), 2, figsize=(8, 3 * len(upper_serials))
    )
    for i, serial in enumerate(upper_serials):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Original cam {serial}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon_images[i])
        axes[i, 1].set_title(f"Reconstruction cam {serial}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    recon_path = output_dir / f"nersemble_{participant_id:03d}_{sequence_name}_upper8_recon.png"  # save path
    fig.savefig(recon_path, dpi=150)
    plt.close(fig)
    print("Saved reconstruction grid to:", recon_path)


if __name__ == "__main__":
    main()
