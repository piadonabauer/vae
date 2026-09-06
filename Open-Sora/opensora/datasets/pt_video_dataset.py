"""PTVideoDataset — load pre-processed .pt video tensors for multiview VAE training.

Each .pt file stores a tensor of shape [V, C, T, H, W] (float32, values in [0, 1])
or a dict with a ``"video"`` key holding the same tensor.

Registered in ``opensora.registry.DATASETS`` as ``"pt_video"`` so that training
configs can refer to it via ``type="pt_video"``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from opensora.registry import DATASETS


@DATASETS.register_module("pt_video")
class PTVideoDataset(Dataset):
    """Dataset that loads pre-processed ``.pt`` video files.

    Parameters
    ----------
    data_path:
        Either a single ``.pt`` file path **or** a directory root.  When a
        directory is given, set ``scan_subdirs=True`` to recurse into it.
    scan_subdirs:
        If *True* and ``data_path`` is a directory, recursively collect every
        ``.pt`` file according to the filtering options below.
    participants:
        Whitelist of integer participant IDs (e.g. ``[18]``).  The scanner
        expects sub-directories named ``pXXX`` (zero-padded to 3 digits).
        ``None`` → include all.
    exclude_participants:
        Blacklist of integer participant IDs.  Applied after ``participants``.
    expression_sequence:
        If set (e.g. ``"EMO-1-shout+laugh"``), only load .pt files found
        under a sub-directory whose name **starts with** this string.
    exclude_sequences:
        Sequence folder names to skip (substring match against the folder
        containing the .pt file).
    include_only_sequences:
        If set, only include files whose parent folder name is in this list.
    repeat:
        Virtual dataset length multiplier (useful for single-clip overfitting
        runs to avoid short epochs).
    expected_views:
        If set, skip or reject samples whose V dimension does not match.
    skip_mismatched_views:
        If *True*, silently skip mismatched samples and try the next one
        (up to ``max_skip_attempts`` tries) rather than raising.
    max_skip_attempts:
        Maximum number of consecutive files to try when skipping mismatched
        views.
    target_frames:
        If set, temporally subsample every loaded clip to exactly this many
        frames using ``torch.linspace`` indices (uniform spacing).  Applied
        in ``__getitem__`` before returning, so DataLoader collation always
        sees the same T.
    """

    def __init__(
        self,
        data_path: str,
        scan_subdirs: bool = False,
        participants: Optional[List[int]] = None,
        exclude_participants: Optional[List[int]] = None,
        expression_sequence: Optional[str] = None,
        exclude_sequences: Optional[List[str]] = None,
        include_only_sequences: Optional[List[str]] = None,
        repeat: int = 1,
        # Optional sanity checks for multi-view VAEs.
        expected_views: Optional[int] = None,
        skip_mismatched_views: bool = False,
        max_skip_attempts: int = 20,
        # Subsample temporal axis to a fixed length in __getitem__ so DataLoader
        # collation never mixes T (e.g. one EMO-1 clip with T=9 among T=13).
        # train.py also applies train_target_frames after the batch; if both are
        # set to the same value the second pass is a no-op.
        target_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Pop any unused kwargs to avoid silent errors from registry builder
        kwargs.pop("type", None)

        self.expected_views = expected_views
        self.skip_mismatched_views = skip_mismatched_views
        self.max_skip_attempts = max_skip_attempts

        if target_frames is not None and int(target_frames) < 1:
            raise ValueError(f"target_frames must be >= 1, got {target_frames}")
        self.target_frames = int(target_frames) if target_frames is not None else None

        self.repeat = max(1, int(repeat))

        participants_set = set(participants) if participants is not None else None
        exclude_set = set(exclude_participants) if exclude_participants is not None else set()

        data_path = Path(data_path)

        if not scan_subdirs:
            # Single file mode
            if data_path.is_file():
                self.pt_files: List[Path] = [data_path]
            else:
                raise FileNotFoundError(
                    f"PTVideoDataset: data_path is not a file and scan_subdirs=False: {data_path}"
                )
        else:
            # Directory scan mode
            if not data_path.is_dir():
                raise FileNotFoundError(f"PTVideoDataset: data_path directory not found: {data_path}")

            self.pt_files = self._scan(
                data_path,
                participants_set=participants_set,
                exclude_set=exclude_set,
                expression_sequence=expression_sequence,
                exclude_sequences=exclude_sequences,
                include_only_sequences=include_only_sequences,
            )

        if len(self.pt_files) == 0:
            raise RuntimeError(
                f"PTVideoDataset: no .pt files found under {data_path} "
                f"(participants={participants}, exclude={exclude_participants}, "
                f"expression_sequence={expression_sequence})"
            )

    @staticmethod
    def _scan(
        root: Path,
        participants_set,
        exclude_set,
        expression_sequence,
        exclude_sequences,
        include_only_sequences,
    ) -> List[Path]:
        """Collect .pt files from a nersemble-style directory tree.

        Expected layout::

            root/
              pXXX/
                EXPRESSION-NAME/
                  EXPRESSION-NAME.pt   ← or any *.pt

        """
        files: List[Path] = []

        # Regex to extract participant ID from folder name "pXXX"
        pid_re = re.compile(r"^p(\d+)$", re.IGNORECASE)

        for person_dir in sorted(root.iterdir()):
            if not person_dir.is_dir():
                continue
            m = pid_re.match(person_dir.name)
            if m is None:
                continue  # not a pXXX directory
            pid = int(m.group(1))

            if participants_set is not None and pid not in participants_set:
                continue
            if pid in exclude_set:
                continue

            # Iterate expression subdirectories
            for expr_dir in sorted(person_dir.iterdir()):
                if not expr_dir.is_dir():
                    continue
                expr_name = expr_dir.name

                if expression_sequence is not None and not expr_name.startswith(expression_sequence):
                    continue
                if exclude_sequences is not None and any(s in expr_name for s in exclude_sequences):
                    continue
                if include_only_sequences is not None and expr_name not in include_only_sequences:
                    continue

                for pt in sorted(expr_dir.glob("*.pt")):
                    files.append(pt)

        return files

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pt_files) * self.repeat

    def __getitem__(self, index: int) -> torch.Tensor:
        # Handle repetition for overfitting
        base_file_idx = index % len(self.pt_files)

        for attempt in range(self.max_skip_attempts):
            file_idx = (base_file_idx + attempt) % len(self.pt_files)
            pt_file = self.pt_files[file_idx]

            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
            except Exception as exc:
                if attempt < self.max_skip_attempts - 1:
                    continue
                raise RuntimeError(f"Failed to load {pt_file}") from exc

            if isinstance(data, dict):
                video = data.get("video", None)
                if video is None:
                    # fall back to first tensor value
                    for v in data.values():
                        if isinstance(v, torch.Tensor):
                            video = v
                            break
                if video is None:
                    if attempt < self.max_skip_attempts - 1:
                        continue
                    raise RuntimeError(f"No tensor found in dict at {pt_file}")
            else:
                video = data

            if not isinstance(video, torch.Tensor):
                if attempt < self.max_skip_attempts - 1:
                    continue
                raise RuntimeError(f"Expected tensor, got {type(video)} from {pt_file}")

            # Some older files were saved with an extra batch dim [1, V, C, T, H, W]
            if video.dim() == 6:
                video = video[0]
                # Fix layout if stored as [V, T, C, H, W] instead of [V, C, T, H, W]
                if video.dim() == 5 and video.shape[1] > video.shape[2]:
                    video = video.permute(0, 2, 1, 3, 4)

            # View count validation
            if self.expected_views is not None and video.dim() == 5:
                v = int(video.shape[0])
                if v != self.expected_views:
                    if self.skip_mismatched_views and attempt < self.max_skip_attempts - 1:
                        continue
                    if not self.skip_mismatched_views:
                        raise RuntimeError(
                            f"Expected {self.expected_views} views, got {v} in {pt_file}"
                        )
                    # exhausted attempts below
                    break

            video = torch.clamp(video, 0.0, 1.0)
            if self.target_frames is not None and video.dim() == 5:
                # video is [V, C, T, H, W]
                t = int(video.shape[2])
                if t != self.target_frames:
                    if t < 1:
                        raise ValueError(f"Empty temporal axis in {pt_file}")
                    idx = torch.linspace(0, t - 1, self.target_frames).long()
                    video = video.index_select(2, idx)
            return {"video": video, "path": str(pt_file)}

        raise RuntimeError(
            f"PTVideoDataset: could not load a valid sample after {self.max_skip_attempts} "
            f"attempts starting from {self.pt_files[base_file_idx]}"
        )

