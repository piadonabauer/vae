"""
Pack two (or more → first two) MP4s per sequence into one ``.pt`` for multiview training.

Expected layout (example)::

    <root>/p069/EMO-1-shout+laugh/cam_*_processed.mp4   (≥2 files)
    <root>/p069/EMO-1-shout+laugh/EMO-1-shout+laugh.pt  (optional; if present, skip)

Output: ``<sequence_folder>/<sequence_folder>.pt`` with ``{"video": tensor [V,C,T,H,W]}`` in [0,1].

Example::

    python save_multiview.py /datasets/lindell-proj/neumayr/nersemble_v2/processed/128-res
"""

import sys
from pathlib import Path

import cv2
import torch


def process_folder(folder_path: str | Path) -> None:
    folder = Path(folder_path).resolve()

    if not folder.exists():
        raise RuntimeError(f"Folder does not exist: {folder}")

    participants = sorted(f for f in folder.iterdir() if f.is_dir())
    if not participants:
        raise RuntimeError(f"No participant folders found in {folder}")

    for participant in participants:
        expressions = sorted(f for f in participant.iterdir() if f.is_dir())

        for expr in expressions:
            output_file = expr / f"{expr.name}.pt"

            if output_file.exists():
                print(f"Skipping (exists): {output_file}")
                continue

            mp4s = sorted(expr.glob("*.mp4"))
            if len(mp4s) < 2:
                print(f"Warning: only {len(mp4s)} mp4 in {expr}, need 2 — skip")
                continue
            if len(mp4s) > 2:
                print(f"Warning: {len(mp4s)} mp4 in {expr}, using first 2 (sorted): {mp4s[0].name}, {mp4s[1].name}")
                mp4s = mp4s[:2]

            videos: list[torch.Tensor] = []
            for p in mp4s:
                print(f"  Reading: {p.name}")
                cap = cv2.VideoCapture(str(p))
                frames: list[torch.Tensor] = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(t)
                cap.release()
                if frames:
                    videos.append(torch.stack(frames, dim=1))  # C,T,H,W

            if not videos:
                print(f"  No frames in {expr}")
                continue

            # Same T across views (trim to shortest clip)
            min_t = min(v.shape[1] for v in videos)
            if not all(v.shape[1] == min_t for v in videos):
                print(f"  Aligning T to min={min_t} across views")
            videos = [v[:, :min_t].contiguous() for v in videos]

            video_tensor = torch.stack(videos, dim=0)  # V,C,T,H,W
            print(f"  Tensor: {video_tensor.shape} -> {output_file}")
            torch.save({"video": video_tensor}, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_multiview.py <root_dir>")
        print("  e.g. python save_multiview.py /datasets/.../processed/128-res")
        sys.exit(1)

    process_folder(sys.argv[1])
