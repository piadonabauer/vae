# Figure Plan

Ordered by importance. For each: what it shows, layout, and how to produce it from this repo.
(F1/F2/F3 are the ones every reader will see — invest there.)

## F1 — Teaser (page 1, full width)
**Message:** multi-view facial video is 4D-redundant; we compress it into one latent; the paper
measures how much fits.
**Layout:** left — a V x T grid of NeRSemble frames (2-4 views x 5 frames) with two labeled
redundancy arrows: horizontal "time" and vertical "view". Middle — a small latent cuboid
labeled `[16, T/4, H/8, W/8]` with input/output arrows ("encode" / "decode per view"). Right —
reconstructed grid; inset crops showing the two failure modes side by side (ghosted view,
temporally bled frame) with red boxes, captioned "what happens when both axes are compressed".
**Production:** frames from any preprocessed `frames.pt` (use `preview_one_sequence.sh` /
`save_multiview.py`); recon crops from E1-c vs E1-d checkpoints. Assemble in
matplotlib/Illustrator.

## F2 — Architecture overview (Method, full width)
**Message:** the adaptation recipe at a glance; color = training regime.
**Layout:** left to right: V input streams -> per-view Wan encoder stems (BLUE = frozen,
optional striped = LoRA) -> fusion block at the 384-ch bottleneck (GREEN = new, zero-init,
with a small "x4 variants" tag) -> binary tree merge (V-1 merge nodes) -> shared latent
`[16, T', h, w]` -> ONE shared frozen decoder drawn once with V parallel arrows through it,
each arrow tagged with its per-view latent LoRA (GREEN) + view embedding -> V reconstructions.
Legend: frozen / LoRA / new zero-init. Add a thin "feat_cache" ribbon along the temporal axis
of encoder+decoder to foreshadow F4.
**Production:** draw (TikZ/Figma). This replaces your current `vae_overview.pdf` conceptual figure.

## F3 — Headline: the rate–quality curve (Results)
**Message:** quality vs latent rate; single-axis compression degrades gracefully, joint
compression falls off a cliff. (This IS the paper's framing — no baseline bar chart.)
**Layout:** PSNR (y) vs total compression ratio (x, log scale) with one point per config:
zero-shot reference, finetuned per-view (12x/view), fused 2-view (24x), per-view TC (48x),
fused+TC (96x), and V=4 points if available (192x). Connect the single-axis points with a
trend line; the joint point sits visibly below it — annotate the gap as "super-additive
degradation". Second small panel: SSIM, same x-axis. Optionally shade the region covered by
published video VAEs (48x @ 16ch) for context.
**Production:** from the E1/E6 results CSV; matplotlib. Rates from
rate = (V·3·T·H·W)/(V'·16·T'·(H/8)·(W/8)).

## F4 — Temporal chunking mechanics + bleeding (Method or Results)
**Message:** where temporal compression happens and where it fails.
**Layout:** top — timeline schematic: frame 0 alone, then 4-frame chunks; arrows showing the
rolling feat_cache (last 2 activations) passed between chunks; cold-cache marker on frame 0.
Bottom — two aligned strips over frame index: (i) per-frame PSNR profile (E1-b vs E1-a),
showing the frame-0 dip and chunk-boundary dips; (ii) |Δ frame| curves for GT vs recon — the
flat recon segments inside chunks ARE the bleeding, visually.
**Production:** top drawn; bottom from eval logs (per-frame metrics exist in fixed_seq logging).

## F5 — Qualitative grid (Results, full width)
**Message:** what ghosting and bleeding actually look like.
**Layout:** rows = methods (GT, E1-a, E1-b, E1-c, E1-d, best E4 intervention); columns = view 0
/ view 1 at frame t, plus frames t..t+3 of view 0 (so both axes visible in one grid). Zoom
insets on mouth/eyes (where facial motion makes bleeding obvious). Add a difference-image row
(|view0 - view1| of the recon vs of GT) — ghosting shows up as a near-black difference image.
**Production:** decode fixed val clips from each checkpoint (`nersemble_vae_demo.py` /
`save_multiview.py`); assemble with a small matplotlib script.

## F6 — Fusion-mode ablation (Results, half width)
Bar chart: PSNR/SSIM for cross_attention / self_attention / conv3d / conv4d (E2), with
parameter counts as secondary axis or labels. Optionally small qualitative strip underneath.

## F7 — Diagnostics figure (Results, half width)
**Panel a:** cross-view cosine similarity over training steps for E3-b/c vs E3-d (negative
control) with the GT similarity as dashed reference — shows view separation being learned (or
not). **Panel b:** bleed_ratio_within and _across over training for E1-b/d and the best E4 arm;
healthy=1 line. This figure sells the diagnostics as a contribution.
**Production:** wandb export (`download_wandb.py` / `compute_wandb.py` already exist).

## F8 — Intervention sweep summary (Results, half width)
Horizontal bar chart: Δ bleed_ratio_within (and Δ PSNR) vs E4 baseline for each of the 8
interventions, sorted; color = hypothesis family (cold-start / missing-signal / loss-side).
Instantly shows which hypothesis family was right.

## F9 — Data-scale plot (Discussion, small)
bleed_ratio_within (y) vs training-set size (x: single sequence -> one person -> all people),
E8. The "compression only breaks when it must generalize" argument in one panel.

## F10 — Preprocessing pipeline (supplement)
Camera rig sketch with the 4/8 selected serials highlighted -> raw frame -> center square crop
-> RVM matte -> white composite -> multi-resolution stack -> `[V,T,C,H,W]` tensor. Screenshot-
style, from actual intermediate outputs of `data/processing/preprocess_nersemble.py`.

## F11 — Assets to reuse from the intermediate presentation
- The slide-7-11 architecture build-up (Wan base -> +view modules at [B,384,T,H,W] -> +LoRAs)
  is a good skeleton for F2 — redraw once with the frozen/LoRA/zero-init color coding.
- The slide-4 research-gap table goes into Related Work as Table 1 (already in 01_draft.md).
- The discriminator qualitative comparisons (none vs 3D vs 4D, skin-tone shift; slides 30-34)
  and the "missing earring / nuanced appearance" crops (slides 36-37) are ready-made panels
  for a loss-ablation figure and for the capacity discussion — re-export at print resolution.
- The latent-shape example ([1,3,13,128,128] -> [1,16,4,16,16], slide 5) is a nice concrete
  annotation for F1/F2.

---

### Practical notes
- F3/F6/F7/F8/F9 are pure matplotlib from the results CSV + wandb exports — write one
  `paper/make_figures.py` so they regenerate when reruns land.
- F5/F1 need checkpoints decoded on a GPU box; pick 2-3 FIXED val clips now (e.g. one val
  participant, EMO-1) and use them for every qualitative figure and the supplement video.
- A supplementary video (GT vs recon, all methods, side by side) is cheap once F5's decode
  script exists and is very persuasive for temporal artifacts — bleeding is far more visible
  in motion than in stills.
