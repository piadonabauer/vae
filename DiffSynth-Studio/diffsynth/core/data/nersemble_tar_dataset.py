import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nersemble-data" / "src"))
from nersemble_data.util.color_correction import correct_color

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .operators import ImageCropAndResize


class NeRSembleTarDataset(torch.utils.data.Dataset):
    """
    Load NeRSemble clips from tar files with deterministic train/test split.

    Expected tar contents: frames or videos per camera, with camera serials
    embedded in filenames (e.g. cam_222200049.mp4).
    """

    def __init__(
        self,
        tar_paths: Sequence[str] | str | Path,
        nersemble_root: str | Path,
        participant_id: int,
        sequence_name: str,
        num_frames: int = 81,
        image_size: int = 512,
        split: str = "train",
        test_ratio: float = 0.05,
        split_seed: int = 0,
        upper_views: int = 8,
        time_center: bool = True,
        shared_time_window: bool = True,
        allow_missing_views: bool = False,
        apply_color_correction: bool = False,
    ):
        self.nersemble_root = Path(nersemble_root)  # dataset root
        self.participant_id = int(participant_id)  # subject id
        self.sequence_name = str(sequence_name)  # sequence id
        self.num_frames = int(num_frames)  # frames per clip
        self.image_size = int(image_size)  # square output size
        self.split = split  # "train", "test", or "all"
        self.test_ratio = float(test_ratio)  # test split fraction
        self.split_seed = int(split_seed)  # deterministic seed
        self.upper_views = int(upper_views)  # number of upper cameras
        self.time_center = bool(time_center)  # use center window in time
        self.shared_time_window = bool(shared_time_window)  # sync views in time
        self.allow_missing_views = bool(allow_missing_views)  # tolerate missing cams
        self.apply_color_correction = bool(apply_color_correction)  # color calibration

        self.tar_paths = self._expand_tar_paths(tar_paths)  # list of tar files
        self.upper_serials = self._get_upper_serials()  # top cameras by height
        self.cropper = ImageCropAndResize(  # center crop + resize
            height=self.image_size, width=self.image_size, max_pixels=None
        )
        self.color_calibration = (  # optional color calibration
            self._load_color_calibration() if self.apply_color_correction else None
        )

        self.samples = [p for p in self.tar_paths if self._in_split(p)]  # split filter
        if not self.samples:
            raise RuntimeError("No tar samples matched the requested split.")

    @staticmethod
    def _expand_tar_paths(tar_paths: Sequence[str] | str | Path) -> List[str]:
        """Normalize input into a list of tar files."""
        if isinstance(tar_paths, (str, Path)):
            path = Path(tar_paths)
            if path.is_dir():
                return sorted(str(p) for p in path.glob("*.tar*"))  # all tar files
            if path.is_file() and path.suffix == ".txt":
                return [line.strip() for line in path.read_text().splitlines() if line.strip()]
            return [str(path)]
        return [str(p) for p in tar_paths]

    def _get_upper_serials(self) -> List[str]:
        """Choose top cameras by world Y coordinate."""
        calib_path = self.nersemble_root / f"{self.participant_id:03d}/calibration/camera_params.json"
        camera_params = json.loads(calib_path.read_text())  # calibration JSON
        world_2_cam = camera_params["world_2_cam"]
        centers = {}
        for serial, w2c in world_2_cam.items():
            w2c = np.array(w2c, dtype=np.float32)
            c2w = np.linalg.inv(w2c)  # invert to get camera-to-world
            centers[serial] = c2w[:3, 3]  # camera center in world
        ordered = sorted(centers.keys(), key=lambda s: centers[s][1], reverse=True)  # sort by height
        return ordered[: self.upper_views]

    def _load_color_calibration(self) -> Dict[str, np.ndarray]:
        """Load per-camera color correction matrices."""
        calib_path = self.nersemble_root / f"{self.participant_id:03d}/calibration/color_calibration.json"
        if not calib_path.exists():
            raise RuntimeError(f"Color calibration not found: {calib_path}")
        color_calibration = json.loads(calib_path.read_text())
        return {serial: np.array(ccm) for serial, ccm in color_calibration.items()}

    def _in_split(self, tar_path: str) -> bool:
        """Deterministic hash split by tar stem name."""
        if self.split in ("all", None):
            return True
        key = Path(tar_path).stem  # stable sample id
        digest = hashlib.sha1(f"{self.split_seed}:{key}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF  # map to [0,1)
        is_test = bucket < self.test_ratio
        return is_test if self.split == "test" else not is_test

    @staticmethod
    def _extract_serial(name: str) -> Optional[str]:
        """Find 9-digit camera serial in a filename."""
        match = re.search(r"(?:cam[_-])?(\d{9})", name)  # NeRSemble serials
        return match.group(1) if match else None

    @staticmethod
    def _extract_frame_index(name: str) -> Optional[int]:
        """Parse frame index from the filename stem."""
        match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(name))
        return int(match.group(1)) if match else None

    def _select_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Select a clip from a single view, center or from start."""
        if not frames:
            return []
        if len(frames) >= self.num_frames:
            if self.time_center:
                start = (len(frames) - self.num_frames) // 2  # center window
                return frames[start : start + self.num_frames]
            return frames[: self.num_frames]  # take earliest frames
        pad = [frames[-1]] * (self.num_frames - len(frames))  # repeat last frame
        return frames + pad

    def _select_frames_shared(self, frames_by_serial: Dict[str, List[Image.Image]]) -> Dict[str, List[Image.Image]]:
        """Select a synchronized temporal window across views."""
        available = [s for s in self.upper_serials if s in frames_by_serial and frames_by_serial[s]]
        if not available:
            return {}
        min_len = min(len(frames_by_serial[s]) for s in available)  # sync to shortest
        if min_len >= self.num_frames:
            if self.time_center:
                start = (min_len - self.num_frames) // 2  # center window
            else:
                start = 0  # start of sequence
            end = start + self.num_frames
            return {s: frames_by_serial[s][start:end] for s in available}
        pad_count = self.num_frames - min_len
        selected = {}
        for s in available:
            frames = frames_by_serial[s][:min_len]
            selected[s] = frames + [frames[-1]] * pad_count  # pad with last frame
        return selected

    def _process_frame(self, frame: Image.Image, serial: str) -> Image.Image:
        """Apply optional color correction then crop/resize."""
        if self.apply_color_correction:
            ccm = self.color_calibration.get(serial)
            if ccm is None:
                raise RuntimeError(f"Missing color calibration for camera {serial}")
            rgb = np.asarray(frame).astype(np.float32) / 255.0  # normalize
            rgb = correct_color(rgb, ccm)  # apply CCM
            frame = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))  # back to uint8
        return self.cropper(frame)  # center crop + resize

    def _load_frames_from_images(self, tar: tarfile.TarFile) -> Dict[str, List[Image.Image]]:
        """Load image frames from tar into per-serial lists."""
        frames_by_serial: Dict[str, List[Tuple[int, Image.Image]]] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            serial = self._extract_serial(member.name)
            if serial is None or serial not in self.upper_serials:
                continue
            frame_index = self._extract_frame_index(member.name)
            with tar.extractfile(member) as fp:
                if fp is None:
                    continue
                img = Image.open(io.BytesIO(fp.read())).convert("RGB")  # decode image
            frames_by_serial.setdefault(serial, []).append((frame_index or 0, img))
        output: Dict[str, List[Image.Image]] = {}
        for serial, items in frames_by_serial.items():
            items.sort(key=lambda x: x[0])  # sort by frame index
            output[serial] = [img for _, img in items]
        return output

    def _load_frames_from_videos(self, tar: tarfile.TarFile) -> Dict[str, List[Image.Image]]:
        """Load video frames from tar into per-serial lists."""
        import imageio

        frames_by_serial: Dict[str, List[Image.Image]] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                continue
            serial = self._extract_serial(member.name)
            if serial is None or serial not in self.upper_serials:
                continue
            with tar.extractfile(member) as fp:
                if fp is None:
                    continue
                data = fp.read()  # load file bytes
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(data)  # materialize for imageio
                tmp.flush()
                reader = imageio.get_reader(tmp.name)
                frames = []
                for frame in reader:
                    frames.append(Image.fromarray(frame).convert("RGB"))  # decode frame
                reader.close()
            frames_by_serial[serial] = frames
        return frames_by_serial

    def __len__(self) -> int:
        """Number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[str] | str]:
        """Return one sample: [V,T,C,H,W] tensor + metadata."""
        tar_path = self.samples[idx]  # input tar
        sample_id = Path(tar_path).stem  # stable name
        with tarfile.open(tar_path, "r:*") as tar:
            frames_by_serial = self._load_frames_from_images(tar)
            if not frames_by_serial:
                frames_by_serial = self._load_frames_from_videos(tar)  # fallback to videos

        if self.shared_time_window:
            frames_by_serial = self._select_frames_shared(frames_by_serial)  # sync time

        views = []
        used_serials = []
        for serial in self.upper_serials:
            frames = frames_by_serial.get(serial, [])  # per-camera frames
            if not frames:
                if self.allow_missing_views:
                    continue
                raise RuntimeError(f"Missing frames for camera {serial} in {tar_path}")
            if not self.shared_time_window:
                frames = self._select_frames(frames)  # independent per view
            frames = [self._process_frame(frame, serial) for frame in frames]  # color + crop
            tensor = torch.stack([TF.to_tensor(f) for f in frames], dim=0)  # T,C,H,W
            views.append(tensor)
            used_serials.append(serial)

        if not views:
            raise RuntimeError(f"No usable frames found in {tar_path}")

        video = torch.stack(views, dim=0)  # V,T,C,H,W
        return {
            "video": video,
            "serials": used_serials,
            "path": str(tar_path),
            "sample_id": sample_id,
        }
