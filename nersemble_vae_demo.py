#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import json
import re
import sys
import tarfile
import tempfile
import types
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2


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
    images_dir = data_manager.get_sequence_images_dir(sequence_name)
    if images_dir.exists():
        available = [p.stem.split("_")[1] for p in images_dir.glob("cam_*.mp4")]
    else:
        available = data_manager.list_cameras(sequence_name)

    cam_json = root / data_manager.participant_subdir / "calibration" / "camera_params.json"
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


def compute_mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return float(np.mean((x - y) ** 2))


def compute_psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(x, y)
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10((data_range**2) / mse))


def compute_ssim_rgb(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Simple SSIM implementation using OpenCV Gaussian blur.
    Assumes x/y are float images in [0, 1], shape [H, W, C].
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape:
        raise ValueError(f"SSIM expects [H,W,C] with matching shapes, got {x.shape} vs {y.shape}")

    h, w, c = x.shape
    if c not in (1, 3, 4):
        raise ValueError(f"Unexpected channel count: {c}")

    # Ensure odd window size and not larger than the image.
    window = int(window_size)
    window = window if window % 2 == 1 else window + 1
    window = min(window, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
    window = max(window, 3)

    sigma = 1.5
    L = 1.0
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    def ssim_channel(xc: np.ndarray, yc: np.ndarray) -> float:
        mu_x = cv2.GaussianBlur(xc, (window, window), sigmaX=sigma, sigmaY=sigma)
        mu_y = cv2.GaussianBlur(yc, (window, window), sigmaX=sigma, sigmaY=sigma)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(xc * xc, (window, window), sigmaX=sigma, sigmaY=sigma) - mu_x2
        sigma_y2 = cv2.GaussianBlur(yc * yc, (window, window), sigmaX=sigma, sigmaY=sigma) - mu_y2
        sigma_xy = cv2.GaussianBlur(xc * yc, (window, window), sigmaX=sigma, sigmaY=sigma) - mu_xy

        num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / np.maximum(den, 1e-12)
        return float(np.mean(ssim_map))

    # Compute per-channel and average.
    if c == 1:
        return ssim_channel(x[..., 0], y[..., 0])
    # For RGBA, ignore alpha if present.
    channels = 3 if c >= 3 else c
    scores = [ssim_channel(x[..., i], y[..., i]) for i in range(channels)]
    return float(np.mean(scores))


def save_rgb_float_image(path: Path, img: np.ndarray) -> None:
    """
    Save float RGB image in [0,1] to disk as PNG/JPG using OpenCV.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Expected HWC RGB-like image, got shape {img.shape}")
    img_uint8 = (np.clip(img[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV expects BGR.
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


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


def prepare_nersemble_root_from_tar(
    tar_path: Path,
    participant_id: int,
) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Extract a NeRSemble tar archive into a temporary directory and return a
    `nersemble_root` compatible with the data manager.

    Expected archive layout (based on your description):
      - top-level contains `calibration/` and `sequences/` directly
      - tar filename contains the participant id (e.g. 017.tar)
    """
    tmp = tempfile.TemporaryDirectory(prefix="nersemble_tar_")
    tmp_root = Path(tmp.name)

    with tarfile.open(tar_path, "r:*") as tf:
        members = [m.name for m in tf.getmembers() if m.name and not m.name.endswith("/")]
        top_levels = set()
        for m in members:
            top_levels.add(m.split("/", 1)[0])

        # If the archive contains calibration/sequences at the top level, extract into p{ID}/.
        if "calibration" in top_levels or "sequences" in top_levels:
            expected_p_dir = f"p{participant_id:03d}"
            participant_dir = tmp_root / expected_p_dir
            participant_dir.mkdir(parents=True, exist_ok=True)
            tf.extractall(str(participant_dir))
        else:
            # Otherwise assume there is already a participant-like wrapper folder.
            tf.extractall(str(tmp_root))

    return tmp_root, tmp


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
        default=Path("/datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res"),
        # /datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res
    )
    parser.add_argument(
        "--nersemble-tar",
        type=Path,
        default=Path("/datasets/lindell-proj/neumayr/nersemble_v2/017.tar"),
        help=(
            "Optional tar archive containing NeRSemble processed data. "
            "If set and exists, it is extracted to a temp directory and used instead of --nersemble-root."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional path to a single MP4 video. If set, NeRSemble dataset loading is skipped.",
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
        "--downsample-size",
        type=int,
        default=None,
        help=(
            "Override --downscale with a target max image dimension (pixels). "
            "We resize the input so max(H, W) == this value."
        ),
    )
    parser.add_argument(
        "--downsample-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of --downsample-size values to try sequentially "
            '(e.g. "32,64,128").'
        ),
    )
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help=(
            "Which camera serial to reconstruct (e.g. '220700191'). "
            "If omitted, uses the first camera in the calibration."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=5,
        help="Number of frames in the reconstructed clip.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=4,
        help="Number of frames between samples (gap=4 => indices step by 5).",
    )
    parser.add_argument(
        "--vis-cams",
        type=int,
        default=5,
        help="Number of camera views (frames) to visualize/reconstruct.",
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
        "--reconstruction-dir",
        type=Path,
        default=Path("/home/piado/projects/aip-lindell/piado/vae/test_reconstruction"),
        help="Where to save reconstruction inputs/outputs + quality metrics.",
    )
    parser.add_argument(
        "--no-color-correction",
        action="store_true",
        help="Disable NeRSemble color correction.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()  # parse CLI args
    # Treat the repo root as the directory *above* this script, so it works
    # both on the local workstation and on the cluster.
    # Example: /.../piado/vae/nersemble_vae_demo.py -> workspace_root = /.../piado
    #args.workspace_root can still override this if needed in future.
    workspace_root = Path(__file__).resolve().parent.parent
    nersemble_root = args.nersemble_root  # dataset root
    tmp_tar: Optional[tempfile.TemporaryDirectory] = None
    if args.video is None:
        if args.nersemble_tar is not None and args.nersemble_tar.exists():
            print("Using nersemble tar:", args.nersemble_tar)
            nersemble_root, tmp_tar = prepare_nersemble_root_from_tar(
                args.nersemble_tar, participant_id=args.participant
            )
        else:
            if args.nersemble_tar is not None:
                print("Warning: nersemble tar not found, using --nersemble-root:", nersemble_root)
    # DiffSynth code lives at <workspace_root>/vae/DiffSynth-Studio in this repo.
    diffsynth_root = workspace_root / "vae" / "DiffSynth-Studio"

    # NeRSemble data helper package is vendored inside DiffSynth-Studio:
    # <workspace_root>/vae/DiffSynth-Studio/diffsynth/core/data/nersemble-data/src
    nersemble_pkg_root = (
        diffsynth_root / "diffsynth" / "core" / "data" / "nersemble-data" / "src"
    )
    sys.path.insert(0, str(nersemble_pkg_root))  # import local nersemble_data
    sys.path.insert(0, str(diffsynth_root))  # import local DiffSynth

    timestep = args.timestep  # frame index
    downsample_size = args.downsample_size
    downscale_factor = None if args.downscale == 0 else args.downscale  # speed/quality
    if downsample_size is not None:
        # Make it explicit: if a target size is specified, we ignore the factor.
        downscale_factor = None

    if args.video is not None:
        # Direct MP4 mode: skip NeRSemble dataset helpers and use a single video.
        video_path = args.video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print("Using direct video:", video_path)

        # Load frames from the video. If timestep > 0, start there; otherwise from 0.
        reader = imageio.get_reader(str(video_path))
        frames = []
        try:
            start_idx = max(timestep, 0)
            for idx, frame in enumerate(reader):
                if idx < start_idx:
                    continue
                frames.append(frame.astype(np.float32) / 255.0)  # [0,1] HWC
        finally:
            reader.close()

        if not frames:
            raise RuntimeError(f"No frames read from video {video_path}")

        # Stack into [T, H, W, C] then convert to [C, T, H, W].
        video_np = np.stack(frames, axis=0)
        video_tensor = (
            torch.from_numpy(video_np)
            .to(torch.float32)
            .permute(3, 0, 1, 2)  # THWC -> CTHW
        )

        # For plotting, keep just the first frame as an "image".
        img = frames[0]

        participant_id = 0
        sequence_name = video_path.stem
        upper_serials = ["cam"]
        images = [img]
    else:
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
        participant_id = (
            args.participant if args.participant in participants else participants[0]
        )  # pick subject
        data_manager = NeRSembleParticipantDataManager(
            str(nersemble_root), participant_id
        )

        sequences = data_manager.list_sequences()  # list sequences
        if not sequences:
            raise RuntimeError(
                "No sequences found under the participant folder. "
                "Expected per-sequence folders containing cam_*.mp4, either under "
                ".../sequences/<name>/images/ (official layout), or .../<name>/ (flat/processed). "
                f"Participant root: {(nersemble_root / data_manager.participant_subdir)!s}."
            )

        # Prefer given sequence, otherwise use the first non-background
        if args.sequence and args.sequence in sequences:
            sequence_name = args.sequence
        else:
            sequence_name = next(
                (s for s in sequences if s != "BACKGROUND"), sequences[0]
            )

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
            top_k=max(int(args.vis_cams), 1),
            axis=1,
        )

        print(f"Upper-view cameras ({len(upper_serials)}):", upper_serials)

        apply_color_correction = not args.no_color_correction  # dataset correction

        images = []
        for serial in upper_serials:
            img = data_manager.load_image(  # read frame
                sequence_name,
                serial,
                timestep,
                apply_color_correction=apply_color_correction,
                downscale_factor=downscale_factor,
                downsample_size=downsample_size,
            )
            images.append(img)

    output_dir = args.reconstruction_dir  # output folder (requested)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save input frames (one per selected camera view).
    for serial, img in zip(upper_serials, images):
        in_path = output_dir / f"input_{sequence_name}_cam{serial}.png"
        save_rgb_float_image(in_path, img)
    print(f"Saved {len(images)} input images to: {output_dir}")

    # Build input videos for the VAE.
    if args.video is not None:
        # Direct video mode: we already built a full [C,T,H,W] tensor.
        videos = [video_tensor]
    else:
        # NeRSemble mode: each "image" is a single frame; wrap as T=1.
        videos = [image_to_video_tensor(img) for img in images]  # to model input

    print("Single-view tensor shape:", videos[0].shape)



    # Load the VAE implementation from DiffSynth-Studio
    print(f"diffsynth_root: {diffsynth_root}")
    print("Loading WanVideoVAE from:", diffsynth_root / "diffsynth/models/wan_video_vae.py")

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

    # Enable detailed shape logging for single-view WanVideoVAE (DiffSynth implementation).
    if not use_multiview and hasattr(vae, "model") and hasattr(vae.model, "debug_shapes"):
        vae.model.debug_shapes = True
        print("Enabled debug_shapes on inner VideoVAE_ model.")

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

    # Reconstruction quality metrics + visualization.
    n = len(upper_serials)
    if n == 0:
        raise RuntimeError("No frames to reconstruct.")

    in_h, in_w = images[0].shape[:2]
    resolution_str = f"{in_w}x{in_h}"

    per_view = []
    psnrs = []
    ssim_scores = []
    mses = []

    # Ensure axes is always 2D for consistent indexing.
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))
    axes = np.atleast_2d(axes)

    for i, serial in enumerate(upper_serials):
        inp = images[i]
        rec = recon_images[i]

        if rec.shape[:2] != inp.shape[:2]:
            rec = cv2.resize(rec.astype(np.float32), (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_LINEAR)

        mse = compute_mse(inp, rec)
        psnr = compute_psnr(inp, rec, data_range=1.0)
        ssim = compute_ssim_rgb(inp, rec)

        per_view.append({"cam": serial, "psnr_db": psnr, "ssim": ssim, "mse": mse})
        psnrs.append(psnr)
        ssim_scores.append(ssim)
        mses.append(mse)

        # Save individual images.
        #save_rgb_float_image(output_dir / f"input_{sequence_name}_cam{serial}.png", inp)
       # save_rgb_float_image(output_dir / f"recon_{sequence_name}_cam{serial}.png", rec)

        # Grid with metric annotations.
        axes[i, 0].imshow(inp)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Input cam {serial}")

        axes[i, 1].imshow(rec)
        axes[i, 1].axis("off")
        psnr_txt = f"{psnr:.2f}" if math.isfinite(psnr) else "inf"
        axes[i, 1].set_title(
            "Recon cam "
            f"{serial}\n"
            f"PSNR {psnr_txt} dB\n"
            f"SSIM {ssim:.4f}\n"
            f"MSE {mse:.6g}"
        )

    # Averages.
    avg_psnr = float(np.mean([p for p in psnrs if math.isfinite(p)]) if any(math.isfinite(p) for p in psnrs) else float("inf"))
    avg_ssim = float(np.mean(ssim_scores))
    avg_mse = float(np.mean(mses))

    title_bits = [f"WanVideoVAE reconstruction quality", f"Resolution: {resolution_str}"]
    if downsample_size is not None:
        title_bits.append(f"downsample-size={downsample_size}")
    elif downscale_factor is not None:
        title_bits.append(f"downscale-factor={downscale_factor}")
    title_bits.append(f"num-cams={n}")
    fig.suptitle("\n".join(title_bits), fontsize=12)

    plt.tight_layout()
    grid_path = output_dir / f"reconstruction_quality_grid_{sequence_name}_{resolution_str}.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    # Save metrics summary.
    summary_path = output_dir / f"metrics_summary_{sequence_name}_{resolution_str}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Sequence: {sequence_name}\n")
        f.write(f"Resolution: {resolution_str}\n")
        f.write(
            "Downsampling: "
            + (
                f"downsample-size={downsample_size}"
                if downsample_size is not None
                else f"downscale-factor={downscale_factor}"
                if downscale_factor is not None
                else "none"
            )
            + "\n"
        )
        if "participant_id" in locals():
            try:
                f.write(f"Participant ID: {participant_id:03d}\n")
            except Exception:
                f.write(f"Participant ID: {participant_id}\n")
        f.write("\n")
        f.write(f"Average PSNR (dB): {avg_psnr}\n")
        f.write(f"Average SSIM: {avg_ssim}\n")
        f.write(f"Average MSE: {avg_mse}\n")
        f.write("\nPer-view metrics:\n")
        for row in per_view:
            psnr_row = row["psnr_db"]
            psnr_txt = f"{psnr_row:.6g}" if math.isfinite(psnr_row) else "inf"
            f.write(
                f"  cam {row['cam']}: PSNR={psnr_txt} dB, SSIM={row['ssim']:.6g}, MSE={row['mse']:.6g}\n"
            )

    print("Saved reconstruction grid:", grid_path)
    print("Saved metrics summary:", summary_path)

    if tmp_tar is not None:
        tmp_tar.cleanup()


def _parse_csv_ints(s: str) -> list[int]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out: list[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _resize_float_image_by_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    max_hw = max(h, w)
    if max_hw <= 0:
        raise ValueError(f"Invalid image shape for resize: {img.shape}")
    scale = float(max_dim) / float(max_hw)
    new_w = max(1, int(w * scale + 0.5))
    new_h = max(1, int(h * scale + 0.5))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=interp)


def _resize_float_image_by_downscale_factor(img: np.ndarray, downscale_factor: int) -> np.ndarray:
    if downscale_factor is None or downscale_factor == 0:
        return img
    scale = 1.0 / float(downscale_factor)
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale + 0.5))
    new_h = max(1, int(h * scale + 0.5))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=interp)


def main_serial_clip():
    """
    Reconstruct a short video clip from a single camera serial:
    - pick 1 camera serial
    - take N frames starting at --timestep, with step (frame_gap+1)
    - reconstruct as a video clip with WanVideoVAE
    - compute per-frame PSNR/SSIM/MSE and save input/recon frames + summary
    - optionally try multiple --downsample-sizes sequentially
    """
    args = build_arg_parser().parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    nersemble_root = args.nersemble_root
    tmp_tar: Optional[tempfile.TemporaryDirectory] = None

    if args.video is None and args.nersemble_tar is not None and args.nersemble_tar.exists():
        nersemble_root, tmp_tar = prepare_nersemble_root_from_tar(
            args.nersemble_tar, participant_id=args.participant
        )
        print("Using extracted nersemble tar:", args.nersemble_tar)
    elif args.video is None and args.nersemble_tar is not None and not args.nersemble_tar.exists():
        print("Warning: --nersemble-tar not found; using --nersemble-root:", nersemble_root)

    diffsynth_root = workspace_root / "vae" / "DiffSynth-Studio"
    nersemble_pkg_root = (
        diffsynth_root / "diffsynth" / "core" / "data" / "nersemble-data" / "src"
    )
    sys.path.insert(0, str(nersemble_pkg_root))
    sys.path.insert(0, str(diffsynth_root))

    n_frames = max(int(args.n_frames), 1)
    frame_gap = max(int(args.frame_gap), 0)
    start_idx = max(int(args.timestep), 0)
    frame_indices = [start_idx + i * (frame_gap + 1) for i in range(n_frames)]

    downsample_sizes = _parse_csv_ints(args.downsample_sizes) if args.downsample_sizes else []
    if downsample_sizes:
        run_specs = [{"downsample_size": s, "downscale_factor": None} for s in downsample_sizes]
    else:
        if args.downsample_size is not None:
            run_specs = [{"downsample_size": args.downsample_size, "downscale_factor": None}]
        else:
            df = None if args.downscale == 0 else int(args.downscale)
            run_specs = [{"downsample_size": None, "downscale_factor": df}]

    # Load WanVideoVAE once.
    print(f"diffsynth_root: {diffsynth_root}")
    vae_module = load_wan_video_vae_module(
        diffsynth_root / "diffsynth/models/wan_video_vae.py"
    )
    WanVideoVAE = getattr(vae_module, "WanVideoVAE", None)
    if WanVideoVAE is None:
        raise RuntimeError("WanVideoVAE is not available in wan_video_vae.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = WanVideoVAE().to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        converter = vae.state_dict_converter()
        vae.load_state_dict(converter.from_civitai(state), strict=False)
        print("Loaded checkpoint:", checkpoint_path)
    else:
        print("Warning: checkpoint not found; reconstruction will be random:", checkpoint_path)

    vae.eval()
    use_multiview = False
    if not use_multiview and hasattr(vae, "model") and hasattr(vae.model, "debug_shapes"):
        vae.model.debug_shapes = True

    output_dir = args.reconstruction_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-side setup (sequence and camera).
    if args.video is None:
        from nersemble_data.data.nersemble_data import (
            NeRSembleDataManager,
            NeRSembleParticipantDataManager,
        )

        data_folder = NeRSembleDataManager(str(nersemble_root))
        participants = sorted(data_folder.list_participants())
        if not participants:
            raise RuntimeError("No participants found in the NeRSemble root.")

        participant_id = args.participant if args.participant in participants else participants[0]
        data_manager = NeRSembleParticipantDataManager(str(nersemble_root), participant_id)

        sequences = data_manager.list_sequences()
        if not sequences:
            raise RuntimeError(
                "No sequences found under participant folder. "
                f"Participant root: {(nersemble_root / data_manager.participant_subdir)!s}"
            )

        if args.sequence and args.sequence in sequences:
            sequence_name = args.sequence
        else:
            sequence_name = next((s for s in sequences if s != "BACKGROUND"), sequences[0])

        camera_calibration = data_manager.load_camera_calibration()
        world_2_cam = camera_calibration.world_2_cam

        camera_serial = args.camera_serial if args.camera_serial is not None else next(iter(world_2_cam.keys()))
        camera_serial = str(camera_serial)
        if camera_serial not in world_2_cam:
            raise RuntimeError(
                f"Camera serial {camera_serial!r} not found in calibration. "
                f"Example keys: {list(world_2_cam.keys())[:5]}"
            )

        apply_color_correction = not args.no_color_correction

        print("Participant:", participant_id)
        print("Sequence:", sequence_name)
        print("Camera serial:", camera_serial)
    else:
        video_path = args.video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        sequence_name = video_path.stem
        camera_serial = args.camera_serial if args.camera_serial is not None else "video"
        apply_color_correction = False

    for spec in run_specs:
        cur_downsample_size = spec["downsample_size"]
        cur_downscale_factor = spec["downscale_factor"]

        print(
            f"\n=== Reconstructing clip: downsample_size={cur_downsample_size}, "
            f"downscale_factor={cur_downscale_factor} ==="
        )

        input_frames: list[np.ndarray] = []
        if args.video is None:
            for idx in frame_indices:
                img = data_manager.load_image(
                    sequence_name,
                    camera_serial,
                    idx,
                    apply_color_correction=apply_color_correction,
                    downscale_factor=cur_downscale_factor,
                    downsample_size=cur_downsample_size,
                )
                input_frames.append(img)
        else:
            reader = imageio.get_reader(str(video_path))
            targets = set(frame_indices)
            idx_to_frame: dict[int, np.ndarray] = {}
            try:
                for idx, frame in enumerate(reader):
                    if idx in targets:
                        img = frame.astype(np.float32) / 255.0
                        if cur_downsample_size is not None:
                            img = _resize_float_image_by_max_dim(img, cur_downsample_size)
                        elif cur_downscale_factor is not None:
                            img = _resize_float_image_by_downscale_factor(img, cur_downscale_factor)
                        idx_to_frame[idx] = img
                        if len(idx_to_frame) >= n_frames:
                            break
                    if idx > max(frame_indices):
                        break
            finally:
                reader.close()

            for idx in frame_indices:
                if idx not in idx_to_frame:
                    raise RuntimeError(f"Could not read frame idx={idx} from {video_path}")
                input_frames.append(idx_to_frame[idx])

        if not input_frames:
            raise RuntimeError("No input frames loaded.")

        in_h, in_w = input_frames[0].shape[:2]
        resolution_str = f"{in_w}x{in_h}"

        video_tensor = torch.cat([image_to_video_tensor(f) for f in input_frames], dim=1)  # [C,T,H,W]
        videos = [video_tensor]

        with torch.no_grad():
            latents = vae.encode(videos, device=device)
            recon_videos = vae.decode(latents, device=device)

        recon_tensor = recon_videos[0]  # [C,T,H,W]
        recon_frames: list[np.ndarray] = []
        for t in range(n_frames):
            rec = ((recon_tensor[:, t].clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
            inp = input_frames[t]
            if rec.shape[:2] != inp.shape[:2]:
                rec = cv2.resize(rec.astype(np.float32), (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_LINEAR)
            recon_frames.append(rec)

        per_frame = []
        psnrs = []
        ssim_scores = []
        mses = []

        for t in range(n_frames):
            inp = input_frames[t]
            rec = recon_frames[t]
            mse = compute_mse(inp, rec)
            psnr = compute_psnr(inp, rec, data_range=1.0)
            ssim = compute_ssim_rgb(inp, rec)
            row = {"t": t, "frame_idx": frame_indices[t], "psnr_db": psnr, "ssim": ssim, "mse": mse}
            per_frame.append(row)
            psnrs.append(psnr)
            ssim_scores.append(ssim)
            mses.append(mse)

        avg_psnr = float(np.mean([p for p in psnrs if math.isfinite(p)]) if any(math.isfinite(p) for p in psnrs) else float("inf"))
        avg_ssim = float(np.mean(ssim_scores))
        avg_mse = float(np.mean(mses))

        ds_tag = f"ds{cur_downsample_size}" if cur_downsample_size is not None else f"down{cur_downscale_factor}"
        for t in range(n_frames):
            inp = input_frames[t]
            rec = recon_frames[t]
            idx = frame_indices[t]
            save_rgb_float_image(
                output_dir / f"input_{sequence_name}_cam{camera_serial}_t{t}_idx{idx}_{resolution_str}_{ds_tag}.png",
                inp,
            )
            save_rgb_float_image(
                output_dir / f"recon_{sequence_name}_cam{camera_serial}_t{t}_idx{idx}_{resolution_str}_{ds_tag}.png",
                rec,
            )

        fig, axes = plt.subplots(n_frames, 2, figsize=(10, 2.8 * n_frames))
        axes = np.atleast_2d(axes)
        for t in range(n_frames):
            inp = input_frames[t]
            rec = recon_frames[t]
            row = per_frame[t]

            axes[t, 0].imshow(inp)
            axes[t, 0].axis("off")
            axes[t, 0].set_title(f"Input t={t} (idx {row['frame_idx']})")

            axes[t, 1].imshow(rec)
            axes[t, 1].axis("off")
            psnr_txt = f"{row['psnr_db']:.2f}" if math.isfinite(row["psnr_db"]) else "inf"
            axes[t, 1].set_title(
                f"Recon t={t}\nPSNR {psnr_txt} dB\nSSIM {row['ssim']:.4f}\nMSE {row['mse']:.6g}"
            )

        suptitle = (
            f"WanVideoVAE reconstruction\n"
            f"Resolution {resolution_str} | {ds_tag}\n"
            f"Avg PSNR {avg_psnr:.2f} dB | Avg SSIM {avg_ssim:.4f} | Avg MSE {avg_mse:.6g}"
        )
        fig.suptitle(suptitle, fontsize=12)
        plt.tight_layout()

        grid_path = output_dir / f"reconstruction_quality_grid_{sequence_name}_cam{camera_serial}_{resolution_str}_{ds_tag}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)

        summary_path = output_dir / f"metrics_summary_{sequence_name}_cam{camera_serial}_{resolution_str}_{ds_tag}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Sequence: {sequence_name}\n")
            f.write(f"Camera serial: {camera_serial}\n")
            f.write(f"Resolution: {resolution_str}\n")
            f.write(f"Downsampling: {ds_tag}\n")
            f.write(f"Frame indices: {frame_indices}\n\n")
            f.write(f"Average PSNR (dB): {avg_psnr}\n")
            f.write(f"Average SSIM: {avg_ssim}\n")
            f.write(f"Average MSE: {avg_mse}\n\n")
            f.write("Per-frame metrics:\n")
            for row in per_frame:
                psnr_row = row["psnr_db"]
                psnr_txt = f"{psnr_row:.6g}" if math.isfinite(psnr_row) else "inf"
                f.write(
                    f"  t={row['t']} idx={row['frame_idx']}: "
                    f"PSNR={psnr_txt} dB, SSIM={row['ssim']:.6g}, MSE={row['mse']:.6g}\n"
                )

        print("Saved:", grid_path)
        print("Saved:", summary_path)

    if tmp_tar is not None:
        tmp_tar.cleanup()


if __name__ == "__main__":
    main_serial_clip()
