# save as: data/processing/debug_rvm_direct.py
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths
workspace_root = Path("/home/piado/projects/aip-lindell/piado")
rvm_root = workspace_root / "RobustVideoMatting"
checkpoint_path = workspace_root / "data/rvm_mobilenetv3.pth"
png_path = workspace_root / "data/processing/check_pt_debug/v0_t0.png"

assert png_path.exists(), png_path
assert checkpoint_path.exists(), checkpoint_path

# Import MattingNetwork
sys.path.insert(0, str(rvm_root))
from model import MattingNetwork  # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
model = MattingNetwork(variant="mobilenetv3").eval().to(device)
state = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state)
print("Loaded RVM checkpoint")

# Load image -> tensor [1,3,H,W] in [0,1]
img = Image.open(png_path).convert("RGB")
img_np = np.asarray(img).astype(np.float32) / 255.0
src = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

# Recurrent states
rec = [None] * 4

with torch.no_grad():
    fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)

# fgr, pha: [1,3,H,W], [1,1,H,W] in [0,1]
fgr = fgr[0].clamp(0, 1)
pha = pha[0].clamp(0, 1)

# Compose onto green background like README example
bgr = torch.tensor([0.47, 1.0, 0.6], device=device).view(3, 1, 1)
comp = fgr * pha + bgr * (1 - pha)  # [3,H,W]

# To numpy images
orig_show = img_np  # original in [0,1] HWC
comp_show = comp.permute(1, 2, 0).cpu().numpy()

# Save side-by-side
out_dir = png_path.parent / "rvm_direct_debug"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "v0_t0_rvm_direct.png"

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(orig_show)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(comp_show)
axes[1].set_title("RVM foreground on green")
axes[1].axis("off")

plt.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print("Saved:", out_path)