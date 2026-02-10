# save as, e.g., data/processing/check_pt_sample.py
from pathlib import Path
import torch
from PIL import Image
import numpy as np

pt_path = Path("/home/piado/projects/aip-lindell/piado/data/preprocessed_initial_experiments/p17_EMO-1-shout+laugh/017_EMO-1-shout+laugh.pt")
out_dir = Path("check_pt_debug")
out_dir.mkdir(exist_ok=True)

data = torch.load(pt_path, map_location="cpu")
video = data["video"]    # [V, T, C, H, W]
serials = data.get("serials", [])
print("video shape:", video.shape)
print("serials:", serials)
print("min/max/mean:", float(video.min()), float(video.max()), float(video.mean()))

V, T, C, H, W = video.shape
for v in range(min(2, V)):
    for t in range(min(3, T)):
        frame = video[v, t]            # [C,H,W]
        img = (frame.clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"v{v}_t{t}.png")
print("Saved frames to", out_dir)

"""
from pathlib import Path

from preprocess_nersemble import NeRSembleTarDataset

nersemble_root = Path("/scratch/piado/data/nersemble")
tar_paths = ["/datasets/lindell-proj/neumayr/nersemble_v2/017.tar"]  # adjust to one tar from tasks.json
dataset = NeRSembleTarDataset(
    tar_paths=tar_paths,
    nersemble_root=nersemble_root,
    participant_id=17,
    sequence_name="EMO-1-shout+laugh",  # same as in your tasks.json
    num_frames=12,
    image_size=128,
    split="all",
    upper_views=2,
    time_center=True,
    shared_time_window=True,
    allow_missing_views=False,
    apply_color_correction=True,
)

sample = dataset[0]
video = sample["video"]  # [V,T,C,H,W]
from PIL import Image
import numpy as np

out = Path("nersemble_dataset_raw_debug")
out.mkdir(exist_ok=True)
for v in range(min(2, video.shape[0])):
    for t in range(min(2, video.shape[1])):
        arr = (video[v, t].clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(arr).save(out / f"v{v}_t{t}_DATASET.png")
"""