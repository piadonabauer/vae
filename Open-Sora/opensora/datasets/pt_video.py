"""Reconstructed ``pt_video`` dataset for NeRSemble multi-view ``frames.pt`` tensors.

Disk layout (see data/processing/preprocess_nersemble.py):
    <DATA_ROOT>/p<id>/<sequence>/frames.pt   # tensor [V, T, C, H, W] in [0, 1]

This dataset yields one item per ``frames.pt`` as:
    {"video": FloatTensor[V, C, T, H, W] in [0, 1], "path": str, "index": int}

The trainer batches these to [B, V, C, T, H, W], scales to [-1, 1], and feeds the
multi-view VAE. (Stored layout is [V, T, C, H, W]; we permute T<->C here.)
"""
import os
from glob import glob

import torch
from torch.utils.data import Dataset

from opensora.registry import DATASETS


def _participant_id(dirname: str):
    """'p017' -> 17, else None."""
    if len(dirname) >= 2 and dirname[0] == "p" and dirname[1:].isdigit():
        return int(dirname[1:])
    return None


def _read_num_views(path: str) -> int:
    """Cheaply read V = shape[0] via mmap (no full tensor load)."""
    try:
        t = torch.load(path, map_location="cpu", mmap=True)
    except Exception:
        t = torch.load(path, map_location="cpu")
    t = _extract_video_tensor(t, path)
    return int(t.shape[0])


def _extract_video_tensor(payload, path):
    """Accept both storage conventions: a raw tensor or a dict with a 'video' key."""
    if isinstance(payload, dict):
        video = payload.get("video")
        if video is None:
            for v in payload.values():
                if isinstance(v, torch.Tensor):
                    video = v
                    break
        if video is None:
            raise ValueError(f"pt_video: no tensor found in dict payload at {path}")
        return video
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"pt_video: expected tensor or dict, got {type(payload)} at {path}")
    return payload


@DATASETS.register_module("pt_video")
class PtVideoDataset(Dataset):
    def __init__(
        self,
        data_path,
        scan_subdirs: bool = False,
        participants=None,
        exclude_participants=None,
        expression_sequence=None,
        exclude_sequences=None,
        include_only_sequences=None,
        expected_views=None,
        skip_mismatched_views: bool = False,
        repeat: int = 1,
        **kwargs,
    ):
        self.data_path = data_path
        self.scan_subdirs = scan_subdirs
        # Stored so train.py's getattr(dataset, "participants", None) reflects the filter.
        self.participants = participants
        self.exclude_participants = set(exclude_participants) if exclude_participants else set()
        self.expression_sequence = expression_sequence
        self.exclude_sequences = set(exclude_sequences) if exclude_sequences else set()
        self.include_only_sequences = set(include_only_sequences) if include_only_sequences else set()
        self.expected_views = expected_views
        self.skip_mismatched_views = skip_mismatched_views
        self.repeat = max(1, int(repeat))
        # Optional temporal subsampling at load time (uniform indices), so the
        # DataLoader never collates mixed clip lengths (e.g. T=13 files with T=9).
        self.target_frames = int(kwargs["target_frames"]) if kwargs.get("target_frames") else None

        files = self._discover_files()
        files = self._filter_views(files)
        if len(files) == 0:
            raise RuntimeError(
                f"pt_video: no .pt files found under {data_path} with the given filters "
                f"(participants={participants}, expression_sequence={expression_sequence})."
            )
        # Unique file list; train.py's fixed-sequence eval reads dataset.pt_files.
        self.pt_files = list(files)
        self.samples = files * self.repeat

    # -- file discovery --
    def _discover_files(self):
        if isinstance(self.data_path, str) and self.data_path.endswith(".pt"):
            return [self.data_path]

        if not self.scan_subdirs:
            # Treat data_path as a dir of frames.pt or *.pt files directly inside.
            cand = sorted(glob(os.path.join(self.data_path, "*.pt")))
            return cand

        participant_filter = set(self.participants) if self.participants is not None else None
        files = []
        for pdir in sorted(os.listdir(self.data_path)):
            ppath = os.path.join(self.data_path, pdir)
            if not os.path.isdir(ppath):
                continue
            pid = _participant_id(pdir)
            if pid is None:
                continue
            if participant_filter is not None and pid not in participant_filter:
                continue
            if pid in self.exclude_participants:
                continue
            for seq in sorted(os.listdir(ppath)):
                seqpath = os.path.join(ppath, seq)
                if not os.path.isdir(seqpath):
                    continue
                if self.expression_sequence is not None and seq != self.expression_sequence:
                    continue
                if self.include_only_sequences and seq not in self.include_only_sequences:
                    continue
                if seq in self.exclude_sequences:
                    continue
                # Two on-disk conventions exist: '<seq>/frames.pt' (original
                # preprocessing) and '<seq>/<seq>.pt' (newer runs). Prefer
                # frames.pt, otherwise take any .pt in the sequence folder.
                fp = os.path.join(seqpath, "frames.pt")
                if os.path.isfile(fp):
                    files.append(fp)
                else:
                    others = sorted(glob(os.path.join(seqpath, "*.pt")))
                    if others:
                        files.append(others[0])
        return files

    # -- view-count consistency filter --
    def _filter_views(self, files):
        if not files:
            return files
        if not self.skip_mismatched_views and self.expected_views is None:
            return files
        target = self.expected_views
        if target is None:
            target = _read_num_views(files[0])
        kept, dropped = [], 0
        for fp in files:
            try:
                v = _read_num_views(fp)
            except Exception:
                dropped += 1
                continue
            if v == target:
                kept.append(fp)
            else:
                dropped += 1
        if dropped:
            print(f"[pt_video] skipped {dropped} sample(s) with V != {target}; kept {len(kept)}.")
        return kept

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        t = _extract_video_tensor(torch.load(path, map_location="cpu"), path)
        if t.dim() == 6:  # stray batch dim from some older files
            t = t[0]
        if t.dim() != 5:
            raise ValueError(f"pt_video: expected 5D tensor, got {tuple(t.shape)} at {path}")
        # Files are stored either as [V, T, C, H, W] (original preprocessing) or
        # [V, C, T, H, W] (newer runs). RGB makes the channel axis unambiguous.
        if t.shape[2] == 3 and t.shape[1] != 3:
            t = t.permute(0, 2, 1, 3, 4)  # [V,T,C,H,W] -> [V,C,T,H,W]
        t = t.contiguous().float().clamp(0.0, 1.0)
        if self.target_frames is not None and t.shape[2] != self.target_frames:
            idx = torch.linspace(0, t.shape[2] - 1, self.target_frames).long()
            t = t.index_select(2, idx)
        return {"video": t, "path": path, "index": index}
