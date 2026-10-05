#!/usr/bin/env python3
"""Identity-embedding metrics from eval dumps (eval-only, CPU).

For every arm, using FaceNet (InceptionResnetV1, VGGFace2) embeddings:
  - id_sim:    cos( emb(rec[v,t]), emb(gt[v,t]) )          -> identity preservation
  - xview_gt:  cos( emb(gt[0,t]),  emb(gt[1,t]) )          -> GT cross-view baseline
  - xview_rec: cos( emb(rec[0,t]), emb(rec[1,t]) )         -> decoded cross-view consistency
  - xview_gap: xview_gt - xview_rec (positive = decoding makes the two views
               drift apart as identities more than the real camera change does)

Faces are detected with MTCNN on the GT frame only, and the same box is applied
to the reconstruction, so the crop is identical for GT and rec (fair comparison;
also robust when the reconstruction is too blurry to detect).

Run with the faceid venv (NOT the training venv):
  /home/coder/venvs/faceid/bin/python paper/audit/identity_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

DUMP_ROOT = Path("/home/coder/vae/Open-Sora/outputs")
OUT_JSON = Path(__file__).resolve().parent / "identity_metrics.json"
OUT_MD = Path(__file__).resolve().parent / "identity_metrics.md"

# Table-1 arms + capacity arms, uniform 170-epoch dumps (update dirs after @300).
ARMS = {
    "zero-shot per-view (TC on)": "paper_E1z_perview_zeroshot_tcON__job1791026450_t50",
    "per-view TC on": "paper_E1b_perview_tcT__job1791027540_t2",
    "per-view TC off": "paper_E1a_perview_tcF__job1791135751_t1",
    "fused TC off": "paper_E1c_fused_tcF__job1791042863_t3",
    "fused TC on (16-ch)": "paper_E1d_fused_tcT__job1791058355_t4",
    "fused TC on, 32-ch": "paper_E11a_fused_tcT_widen32__job1791073606_t8",
    "fused TC on, 64-ch": "paper_E11b_fused_tcT_widen64__job1791088848_t9",
    "32-ch + diff-loss": "paper_E_combo_diffLoss_widen32__job1791104094_t36",
    "all tweaks": "paper_Ebest_allcombined__job1791119530_t52",
}

device = torch.device("cpu")
mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def detect_box(img_hwc_uint8):
    boxes, _ = mtcnn.detect(img_hwc_uint8)
    if boxes is None or len(boxes) == 0:
        return None
    x0, y0, x1, y1 = boxes[0]
    # square box with margin, clipped
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * 1.1 / 2
    h, w = img_hwc_uint8.shape[:2]
    y0, y1 = int(max(0, cy - s)), int(min(h, cy + s))
    x0, x1 = int(max(0, cx - s)), int(min(w, cx + s))
    return y0, y1, x0, x1


@torch.no_grad()
def embed(img_hwc_uint8, box):
    y0, y1, x0, x1 = box
    crop = img_hwc_uint8[y0:y1, x0:x1]
    t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=(160, 160), mode="bilinear", align_corners=False)
    t = (t - 127.5) / 128.0  # facenet fixed_image_standardization
    return resnet(t.to(device))[0]


def cos(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def main():
    results = {}
    gt_boxes = {}  # (clip, v, t) -> box, shared across arms (GT identical)
    for name, d in ARMS.items():
        data = torch.load(DUMP_ROOT / d / "final_eval_dump_val.pt",
                          map_location="cpu", weights_only=False)
        id_sims, xview_gts, xview_recs = [], [], []
        skipped = 0
        for ci, clip in enumerate(data["clips"]):
            gt = clip["gt"].numpy()   # [V,C,T,H,W] uint8
            rec = clip["rec"].numpy()
            T = gt.shape[2]
            for t in range(T):
                embs = {}
                for v in range(2):
                    g = np.transpose(gt[v, :, t], (1, 2, 0))
                    r = np.transpose(rec[v, :, t], (1, 2, 0))
                    key = (ci, v, t)
                    if key not in gt_boxes:
                        gt_boxes[key] = detect_box(g)
                    box = gt_boxes[key]
                    if box is None:
                        skipped += 1
                        break
                    embs[v] = (embed(g, box), embed(r, box))
                if len(embs) < 2:
                    continue
                for v in range(2):
                    id_sims.append(cos(embs[v][0], embs[v][1]))
                xview_gts.append(cos(embs[0][0], embs[1][0]))
                xview_recs.append(cos(embs[0][1], embs[1][1]))
        results[name] = {
            "id_sim_mean": float(np.mean(id_sims)),
            "id_sim_std": float(np.std(id_sims)),
            "xview_gt_mean": float(np.mean(xview_gts)),
            "xview_rec_mean": float(np.mean(xview_recs)),
            "xview_gap": float(np.mean(xview_gts) - np.mean(xview_recs)),
            "n_frames": len(xview_gts),
            "n_skipped_no_face": skipped,
        }
        r = results[name]
        print(f"{name:32s} id_sim={r['id_sim_mean']:.4f}  "
              f"xview rec={r['xview_rec_mean']:.4f} (gt {r['xview_gt_mean']:.4f}, "
              f"gap {r['xview_gap']:+.4f})  n={r['n_frames']} skip={skipped}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    lines = [
        "# Identity-embedding metrics (FaceNet VGGFace2, eval-only, from 170-ep final dumps)",
        "",
        "Crops detected on GT, identical crop applied to rec. 10 val clips x 9 frames.",
        "",
        "| arm | id sim (rec vs GT) | cross-view sim (rec) | cross-view sim (GT) | gap |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['id_sim_mean']:.4f} | {r['xview_rec_mean']:.4f} "
                     f"| {r['xview_gt_mean']:.4f} | {r['xview_gap']:+.4f} |")
    lines += [
        "",
        "id sim: cosine similarity of face embeddings, reconstruction vs GT (1 = identity",
        "perfectly preserved). cross-view gap: how much more the two decoded views disagree",
        "about identity than the two real camera views do (positive = identity drift added",
        "by the codec).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_JSON, "and", OUT_MD)


if __name__ == "__main__":
    main()
