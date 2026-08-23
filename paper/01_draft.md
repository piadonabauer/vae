# Paper Draft — Multi-View Video VAE ("4D latent compression")

Working titles (pick a flavor):
- *Multi-View Facial Video Compression via a Joint 4D Latent Representation* (your existing
  title from the intermediate presentation — fine, but it promises success; with the capacity
  finding as the headline, consider one of the variants below)
- *How Much Fits in a Video Latent? An Empirical Study of Joint View–Temporal Compression in Pretrained Video VAEs*
- *Towards 4D Latents: Extending a Pretrained Video VAE to Multi-View Facial Video*
- *One Latent, Many Cameras: On the Capacity Limits of Multi-View Video Autoencoding*

`[FROM PRESENTATION]` below marks facts/results taken from your intermediate presentation
(15/04/2026); these predate the temporal-compression bleeding arc (July), i.e. they cover the
"view axis works" half of the story.

Framing: this is an **empirical study / systems paper**. The contribution is (a) a minimally
invasive adaptation recipe for turning a pretrained 3D video VAE into a multi-view (4D) VAE,
(b) diagnostics that localize where information is lost, and (c) the finding that the fixed
16-channel Wan latent can absorb *either* 4x temporal compression *or* the view axis, but not
both — a capacity limit, not a training failure.

All `\cite{...}` keys below exist in `references.bib` in this folder.
`[VERIFY]` marks numbers/claims you must confirm against your runs before submission.

---

## 1. Introduction

Paragraph 1 — application pull (keep the spirit of your draft, sharpened):

> Photo-realistic head avatars are built from synchronized multi-view facial video captured
> with dense camera rigs \cite{lombardi2018deep, lombardi2021mixture, cao2022authentic,
> kirschstein2023nersemble, qian2024gaussianavatars}. A single NeRSemble capture produces 16
> camera streams at 73 fps and 3208x2200 resolution \cite{kirschstein2023nersemble} — hours of
> capture yield tens of terabytes of raw video. This data is extraordinarily redundant along
> two axes: *time*, where adjacent frames differ only by smooth facial motion, and *view*,
> where neighboring cameras observe the same face under slowly varying pose
> \cite{shah2024mv2mae, taubner2025mvp4d}.

Paragraph 2 — the latent-space argument (this is the motivation your supervisor likely means —
it is not just about storage; it is about *where modern generative modeling happens*):

> Modern generative pipelines do not operate on pixels: latent diffusion models generate in
> the compressed latent space of a VAE \cite{rombach2022high}, and every state-of-the-art
> video generator — Wan \cite{wan2025wan}, CogVideoX \cite{yang2024cogvideox}, HunyuanVideo
> \cite{kong2024hunyuanvideo}, Cosmos \cite{agarwal2025cosmos} — rests on a causal 3D video
> VAE that compresses space by 8x and time by 4x. The tokenizer therefore *defines* what the
> generator can express \cite{yu2024language}. If we want to generate, edit, or interpolate
> multi-view facial performances with diffusion models, we first need a latent representation
> of multi-view video. Today no such tokenizer exists: multi-view streams are encoded
> independently, which (i) ignores the massive inter-view redundancy and (ii) provides no
> architectural prior for cross-view consistency — each view's latent can drift independently
> under generation or editing.

Paragraph 3 — the question and the honest answer:

> This motivates a single, jointly learned **4D latent spanning space, time, and view**. Rather
> than designing a 4D architecture from scratch — which would forfeit the enormous pretraining
> investment of existing video VAEs — we ask: *can a pretrained 3D video VAE be minimally
> extended to absorb the view axis?* We extend the Wan 2.1 VAE \cite{wan2025wan} with
> zero-initialized cross-view fusion in the encoder and view-conditioned decoding via low-rank
> adapters \cite{hu2022lora, zhang2023adding}, keeping the pretrained backbone frozen so the
> latent remains compatible with unconditional sampling. Through an extensive experimental
> study on NeRSemble we find a consistent pattern: the model reconstructs well when *either*
> the temporal axis is compressed (4x, the native Wan setting) *or* the view axis is fused
> into a shared latent — but combining both degrades reconstructions with two characteristic
> artifacts: *cross-view ghosting* (views collapse toward their mean) and *intra-chunk
> temporal bleeding* (frames within a 4x-compressed chunk blur into each other). We trace
> both to the same cause: the 16-channel latent, sized for single-view video, has no spare
> rate for a second compressed axis. This is consistent with the broader tokenizer literature,
> where higher compression is only achieved by widening the latent
> \cite{dai2023emu, chen2024deep, hacohen2024ltx, yao2025vavae}.

Contributions (bullet list):
1. **A minimally invasive multi-view extension of a pretrained video VAE**: per-view frozen
   Wan encoder stems, zero-initialized cross-view attention fusion at the bottleneck, a
   tree-structured merge to a shared latent, and view-conditioned decoding through per-view
   latent LoRA adapters — preserving the pretrained latent distribution and hence
   compatibility with latent diffusion.
2. **Diagnostics that localize information loss**: a cross-view reconstruction-similarity
   metric (detects view ghosting) and an intra-chunk bleed ratio (detects temporal
   mean-regression inside 4-frame chunks), plus per-frame-index error profiles that expose
   cold-cache first-frame and chunk-boundary artifacts.
3. **A systematic study** over fusion mechanisms (cross-attention, joint self-attention,
   channel-concat Conv3d, factorized 4D convolution), adapter placement, discriminator
   designs, and seven targeted temporal-quality interventions (non-causal decoding, reflection
   padding, side channels, learned cache updates, sub-frame embeddings, temporal-difference
   loss, teacher distillation).
4. **A capacity finding with design implications**: either axis alone fits the 16-channel Wan
   latent; both together do not. We argue future 4D tokenizers must scale latent rate with the
   number of compressed axes, or keep view identity out of the latent entirely.

---

## 2. Related Work

**Video tokenizers / video VAEs.** Latent generative modeling was established for images by
VQGAN and latent diffusion \cite{esser2021taming, rombach2022high} and extended to video by
inflating autoencoders with temporal layers \cite{blattmann2023align} or training causal 3D
tokenizers \cite{yu2024language}. Current video foundation models (Wan \cite{wan2025wan},
CogVideoX \cite{yang2024cogvideox}, HunyuanVideo \cite{kong2024hunyuanvideo}, Cosmos
\cite{agarwal2025cosmos}, Open-Sora \cite{zheng2024opensora, lin2024opensoraplan}) share the
same recipe: causal 3D convolutions, 8x spatial and 4x temporal compression, 16 latent
channels, trained with the L1+LPIPS+KL+GAN objective of \cite{esser2021taming}. A parallel
line disentangles structure and dynamics or pushes compression rates: VidTwin
\cite{wang2025vidtwin}, VideoVAE+ \cite{xing2024large}, Hi-VAE \cite{liu2025hi}, LeanVAE
\cite{cheng2025leanvae}, CV-VAE \cite{zhao2024cvvae}, LTX-Video \cite{hacohen2024ltx}. All of
these are single-view; multi-view data is encoded stream-by-stream.
*(Keep your three bullet observations from \cite{xing2024large}; add a fourth: joint
view+temporal compression is bounded by latent capacity — ours.)*

**Latent capacity.** A consistent empirical law across tokenizers: reconstruction quality at a
given compression is governed by latent rate. LDM ablates downsampling factor against channel
count \cite{rombach2022high}; Emu shows that widening the image latent from 4 to 16 channels
is what makes fine detail reconstructable \cite{dai2023emu}, a change adopted by SDXL-class
and video models \cite{podell2023sdxl}; DC-AE shows aggressive spatial compression is only
viable with proportionally wider latents \cite{chen2024deep}; LTX-Video buys 1:192 compression
with 128 channels \cite{hacohen2024ltx}; and VA-VAE formalizes the reconstruction–generation
trade-off of latent dimensionality \cite{yao2025vavae}. Our study is, to our knowledge, the
first to probe this budget along a *view* axis on a *pretrained, fixed-width* latent.

**Multi-view and 4D generative models.** Cross-view attention is the standard mechanism for
consistency in multi-view diffusion \cite{shi2023mvdream, gao2024cat3d, voleti2024sv3d,
zuo2024videomv}; MV2MAE reconstructs held-out views from synchronized inputs — masked
autoencoding for cross-view *reconstruction*, not learned *compression* \cite{shah2024mv2mae}.
4D generation methods extend this to space-time-view but rely on iterative sampling or
explicit geometry \cite{zhang20244diffusion, xie2024sv4d, shao2024human4dit, jiang2026mesh4d}
and do not yield a compact latent code. Head-avatar pipelines \cite{lombardi2018deep,
lombardi2021mixture, ma2021pixel, cao2022authentic, qian2024gaussianavatars, taubner2025mvp4d}
consume exactly the multi-view facial video we target but operate on pixels or per-frame
codes — none provides a reusable 4D latent space.

**Geometry-based representations.** `[FROM PRESENTATION]` — keep your third category as its
own paragraph: explicit 3D structure (meshes, radiance fields, Gaussians) guarantees view
consistency by construction \cite{jiang2026mesh4d, kirschstein2023nersemble,
qian2024gaussianavatars}, but the geometry becomes the bottleneck: modeling, animating, and
editing the explicit structure constrains downstream tasks, and there is no compact latent a
generative model can sample. This motivates a *latent* rather than *geometric* route to view
consistency.

**Research-gap table** `[FROM PRESENTATION]` — your slide-4 table works well as Table 1 of the
paper; extend it with a "Compression" column to sharpen the gap:

| Approach | Time | Views | Joint latent | Learned compression |
|---|---|---|---|---|
| Image VAEs \cite{rombach2022high} | ✗ | ✗ | ✓ | ✓ |
| Video VAEs \cite{wan2025wan, yang2024cogvideox} | ✓ | ✗ | ✓ | ✓ |
| MV masked autoencoding \cite{shah2024mv2mae} | ✓ | ✓ | ✗ | ✗ |
| Geometry priors \cite{jiang2026mesh4d} | (✓) | ✓ | ✗ | ✗ |
| **Ours** | ✓ | ✓ | ✓ | ✓ |

Closing line of the section (from slide 4, keep it): *4D generative models are the intended
downstream consumers of such a compact latent.*

**Parameter-efficient adaptation.** We follow the adapt-don't-retrain philosophy: LoRA
\cite{hu2022lora} for frozen backbones, zero-initialized new pathways so the pretrained
function is exactly preserved at step 0 \cite{zhang2023adding}, and dimensional inflation as
the classical alternative for adding an axis \cite{carreira2017quo}. Our design combines the
first two; we discuss why inflation is inapplicable to a frozen, fixed-width latent in Sec. 5.

---

## 3. Method

### 3.1 Preliminaries: the Wan 2.1 video VAE

- Causal 3D CNN, 16-channel latent, 8x8 spatial and 4x temporal compression
  (T' = 1 + (T-1)/4) \cite{wan2025wan}.
- **The chunked cache mechanism matters for the whole paper — explain it carefully.**
  Temporal compression only fires through `feat_cache` chunked processing: frame 0 is
  encoded/decoded *alone*, then 4-frame chunks follow, each strided `time_conv` consuming a
  rolling cache (last 2 activations) from the previous chunk. Decoding mirrors this: one
  latent frame in, four frames out, with a persistent cache. Consequences: a cold-start
  asymmetry for frame 0, chunk-boundary seams, and — crucial constraint — the strided
  temporal convolutions admit no LoRA path (a 1x1x1 low-rank branch cannot match the strided
  output shape), so the compression bottleneck itself is frozen in all LoRA configurations.

### 3.2 Multi-view extension (the actual architecture — key design decisions)

Input `[B, V, C, T, H, W]`. Four decisions define the design; present each as
decision -> alternatives -> rationale:

**D1. Adapt a pretrained 3D VAE rather than train a 4D VAE from scratch.**
Rationale: transfer of the pretraining investment \cite{carreira2017quo, blattmann2023align},
data efficiency on a rig-scale dataset, and — decisive — keeping the latent distribution close
to the pretrained one preserves compatibility with the Wan latent-diffusion ecosystem and
unconditional sampling. All new pathways are zero-initialized so the model is *exactly* the
pretrained per-view VAE at step 0 \cite{zhang2023adding}.

**D2. Fuse views at the encoder bottleneck, before `encoder.middle`/`head`.**
The fusion operates at 384 channels on an 8x-downsampled grid — late enough that tokens are
cheap for attention, early enough that the fused information shapes the latent (unlike
post-hoc latent mixing, which we show fails — see D4/history note below).

**D3. Fusion mechanism — a design space of four, all reducing V views to one shared latent:**
- *Cross-view attention* (`ViewAttention`): multi-head SDPA over all V·N tokens per (t),
  RMSNorm QKV, **zero-init output projection** (identity at init), followed by a
  **binary tree merge**: V-1 pairwise merges, each `concat -> ResBlock(2C->C) -> ResBlock(C->C)`.
  Analogous to the cross-view attention of multi-view diffusion \cite{shi2023mvdream, gao2024cat3d}.
- *Joint self-attention* (`JointViewAttention`): one attention over the concatenated token
  sequence of all views, then concat + 2 ResBlocks.
- *Channel-concat Conv3d*: views stacked on channels, 1x1x1 Conv3d + GN/SiLU + two symmetric
  (non-causal) 3D residual blocks.
- *Factorized 4D convolution*: spatial 3x3 Conv2d, temporal 3x3x3 Conv3d, then a view-axis
  Conv3d with kernel (V,3,3) compressing V->1 — a (2+1+1)D factorization in the spirit of
  \cite{carreira2017quo}. (This is the realized version of the "4D convolution" idea from the
  proposal — say so.)

**D4. Decode distinct views from one shared latent — view-conditioned decoding.**
The single fused latent `[16, T', H/8, W/8]` is decoded V times by the *shared frozen*
decoder; view identity is injected as (i) a learned per-view latent embedding
(`nn.Embedding(V, 16)` added to z), and/or (ii) **per-view latent LoRA adapters**
(`Conv3d z->rank->z`, zero-init up-projection) — the low-rank analogue of per-view decoders at
~0.1% of the cost. This is the load-bearing design choice for view separation.
*History note worth one paragraph:* an earlier variant compressed views in latent space by
(learned) averaging (`ViewCompressor`) with per-view embeddings for recovery; it produced
ghosted, near-identical views (old diagnostics suggested SSIM ~0.75-0.84; if you want a number
here, rerun this variant once as a negative-control arm, or describe it qualitatively) —
averaging destroys view identity
before any embedding can recover it. This motivated moving fusion *into* the encoder and view
identity *into* the decoder. Keep this as a documented negative design iteration; it is one of
the paper's lessons.

**D5. What is trainable — LoRA placement.**
Pre-fusion per-view encoder stem: frozen (optionally LoRA, `use_lora_before`).
Bottleneck + full decoder: LoRA rank 32 (`use_lora_after`), zero-init up-projections
\cite{hu2022lora}. New fusion modules: fully trained. Strided temporal convs: structurally
frozen (Sec. 3.1) — flag as a limitation and an ablation axis (full unfreeze vs LoRA).

**D6. Training discipline — overfit first, then generalize; staged initialization.**
Every architectural change is validated in two stages: first overfit a single sequence
(reconstruction must be near-perfect — verifies the architecture can represent the signal at
all), only then train for generalization on the full dataset (where the latent must encode
rather than memorize). Joint view+temporal models additionally warm-start from the converged
temporal-compression checkpoint by default (temporal first, view second), rather than
learning both axes at once from Wan weights (evaluated as an ablation, Sec. 4.5).

### 3.3 Temporal compression: interventions as hypotheses

Present the seven flags as competing hypotheses about *where* temporal information is lost
(each 1-2 sentences; all are opt-in, zero-init, baseline-preserving):
- **Cold-start / boundary hypotheses:** non-causal full-sequence decode (drop the chunk loop;
  symmetric temporal padding at inference-time switch), temporal reflection padding (warm the
  caches on 4 reflected real frames, crop after), learned ConvGRU cache update (replace the
  hand-coded "keep last 2 activations" rule with a gated learned update, identity at init).
- **Missing-signal hypotheses:** sub-frame position embedding (a (2,dim) zero-init bias telling
  the temporal upsampler which of its two output frames it is producing), decoder temporal
  attention at the bottleneck, a tiny full-frame-rate side channel (4 channels at /16 spatial)
  carrying high-frequency temporal detail past the 4x bottleneck.
- **Loss-side hypotheses:** temporal-difference loss L1(Δgt, Δrec) directly penalizing the
  bleeding symptom; distillation from a temporal_compression=False teacher.

### 3.4 Objective and diagnostics

- Loss: `nll = (L1 + 1.5·LPIPS)/exp(σ) + σ` with learned scalar σ, plus `1e-6·KL` — the
  standard tokenizer recipe \cite{esser2021taming, rombach2022high, zheng2024opensora},
  LPIPS \cite{zhang2018unreasonable}. Optional adversarial term with three multi-view
  discriminator layouts (PatchGAN-style 3D \cite{isola2017image}): flatten views into batch,
  stack views on channels, or an explicit 4D discriminator with a view axis and per-view
  embeddings.
- **Diagnostics (contribution — give equations):**
  - *Cross-view similarity*: mean pairwise cosine similarity of reconstructed views —
    detects ghosting (GT gives the reference level).
  - *Bleed ratio*: mean |Δrec| / mean |Δgt| over consecutive-frame pairs, computed separately
    *within* 4-frame chunks and *across* chunk boundaries; ≈1 is healthy, ≪1 within-chunk is
    the bleeding signature.
  - *Per-frame-index error profile*: PSNR as a function of frame index, exposing the frame-0
    cold-cache dip and chunk-boundary seams.

### 3.5 Data and preprocessing

NeRSemble \cite{kirschstein2023nersemble}: 16 synchronized cameras, 73 fps, 3208x2200. Our
pipeline: select 4 or 8 frontal/upper-row cameras (list serials in supplement); temporal
subsampling to 24 fps then uniform selection of T=9 frames (chosen so the compressed latent
T'=3 contains frame 0 plus two 4-frame chunks — i.e. at least one chunk boundary, making both
within-chunk bleeding and boundary artifacts measurable); per-camera color-correction
(Cheung et al. CCM) [VERIFY if used in final runs]; center square crop; background removal
with RobustVideoMatting \cite{lin2022robust} composited on white; bilinear resize to
{128, 256, 512}²; stored as `[V, T, C, H, W]` in [0,1], trained as `[B, V, C, T, H, W]`
rescaled to [-1,1]. Data scales: single sequence -> one person -> all participants with one
expression -> all participants; 10 held-out validation identities. *(State the actual
operating point honestly: main experiments at 128², T=9, 2-4 views; 512² as scaling check.
The 8-view/81-frame/full-resolution setting from the proposal moves to future work.)*

---

## 4. Experiments

*(This is the paper section; the internal run plan with exact flags lives in
`02_experiments.md`. Written CVPR-style below — prose skeleton with `[...]` placeholders for
numbers/tables. LaTeX headers as you'd paste them.)*

\subsection{Experimental Setup}

**Dataset and protocol.** We train and evaluate on NeRSemble \cite{kirschstein2023nersemble}
(Sec. 3.5): V=2 synchronized frontal views, T=9 frames at 128x128, one expression sequence
per participant. We hold out 10 participants entirely for validation; all reported metrics
are computed on these unseen identities. Unless stated otherwise, every experiment uses the
identical recipe: AdamW (lr 5e-4, constant), bf16, effective batch 64, 170 epochs, LoRA rank
32, EMA 0.9999, no discriminator; a single L40S GPU per run. Because the effective batch is
fixed, the epoch budget corresponds to an identical number of optimizer updates and samples
seen for every model variant, and all evaluations fire at the same update steps (every 50
updates) — variants are therefore compared at strictly equal training budget throughout.
Qualitative results always show the same fixed participants (chosen by dataset order, held
identical across all variants), so reconstruction grids are directly comparable between
models. No
run is early-stopped or extended individually: when a variant plateaus below the target
quality within the budget, we report the plateau — under a fixed rate and budget, that *is*
the measurement.

**Two-stage protocol: overfit, then generalize.** Every configuration passes two gates.
*Stage 1 (overfit):* train on a single sequence; reconstruction must be near-perfect
[PSNR >= 35; runs stop once this holds for three consecutive epochs, with a fixed cap]. This
verifies that the architecture *can represent* the signal — a failure here
is an implementation or model-capacity problem and disqualifies the configuration before any
expensive run. *Stage 2 (generalize):* train on all participants (one expression) and
evaluate on held-out identities — here the latent must *encode* rather than memorize, and
this is where compression artifacts appear. The contrast between the two stages is itself a
result we use throughout: temporal bleeding and view ghosting are absent in Stage 1 and
emerge only in Stage 2 (Sec. 4.3), which identifies them as generalization failures of a
rate-limited representation rather than optimization failures.

**Initialization (staged by default).** Joint models (fusion + temporal compression) are
initialized from the converged temporal-compression checkpoint of the corresponding per-view
run, with the view-attention re-randomized — i.e., the model learns the temporal axis first
and the view axis second. The alternative, training from Wan-pretrained weights only (all
new modules zero-initialized), is evaluated as an ablation (Sec. 4.5).

**Metrics.** PSNR, SSIM, and LPIPS \cite{zhang2018unreasonable} over full validation clips.
In addition we report three diagnostics designed to localize *where* information is lost:
(i) *cross-view similarity* — mean pairwise cosine similarity between reconstructed views,
with the similarity of the ground-truth views as reference (values above the reference
indicate view ghosting); (ii) *bleed ratio* — the ratio of mean absolute inter-frame
differences of the reconstruction to those of the ground truth, computed separately within
4-frame temporal chunks and across chunk boundaries (a value of 1 is faithful motion; values
well below 1 indicate temporal bleeding); (iii) the *per-frame-index error profile*, which
exposes cold-start and chunk-boundary artifacts.

**Protocol design rationale** *(bullets to weave into 4.1 prose or a supplementary
"experimental design" paragraph — each pre-empts a likely reviewer question)*:

- **Why V=2.** The minimal view count is the *strongest* setting for a capacity claim: two
  nearby frontal views are barely more information than one, so if the joint latent already
  saturates at V=2, it fails a fortiori for more views — whereas a failure at V=8 could be
  dismissed as an unreasonably aggressive 8-to-1 ratio rather than a property of the
  representation. V=2 also makes the rate accounting a clean factor of two against the
  per-view references. Larger view counts are studied as an explicit ablation axis (E4),
  turning the view count into a measured result instead of a defended choice.
- **Why T=9.** The latent then has T'=3: frame 0 (encoded alone by the causal chunking) plus
  two 4-frame chunks — the shortest clip in which both *within-chunk* bleeding and
  *cross-chunk-boundary* artifacts are measurable. Shorter windows (e.g. T=5) contain no
  chunk boundary at all and silently hide the boundary failure mode.
- **Why 128x128 for the matrix.** The contribution is a *controlled comparison*, not a
  scaling record: the low resolution is what makes the full arm matrix affordable at a fixed
  budget on single GPUs; resolution is scaled separately as its own check.
- **Comparability.** The training budget is defined in optimizer updates, not epochs or wall
  clock: the effective batch is pinned (micro-batch x accumulation = 64, re-balanced
  automatically on memory limits), so every variant sees the identical number of updates
  *and* samples. All evaluations fire at the same update steps; no variant is early-stopped
  or extended individually — a variant that plateaus below the target within the budget is
  reported at its plateau, which under fixed rate and budget is the measurement itself.
- **No adversarial loss in the main protocol.** GAN terms introduce run-to-run variance and
  a second optimization that can mask or mimic capacity effects; the main matrix is purely
  L1 + LPIPS + KL, and discriminators are studied in a dedicated ablation (E7).
- **Held-out identities, not held-out frames.** The val split holds out 10 *participants*
  entirely, so generalization means encoding unseen faces — the regime where a rate-limited
  latent must compress rather than memorize.
- **Deterministic qualitative panels.** Visualized samples are selected by sorted dataset
  order (never from the shuffled batch), so every figure shows the same people for every
  variant — qualitative panels are directly comparable across models and training stages.
- **Overfit gate before every expensive run.** Each configuration must first reconstruct a
  single sequence near-perfectly (train PSNR >= 35 held for 3 epochs); this separates
  implementation/architecture failures from generalization behavior before GPU-weeks are
  spent, and the overfit-vs-generalize contrast itself localizes bleeding/ghosting as
  generalization failures.

**Reference points.** No existing method produces a joint latent for synchronized multi-view
video, so there is no like-for-like baseline. We instead report two per-view *reference
points* that bound our setting from above in latent rate: the pretrained Wan 2.1 VAE applied
to each view independently (zero-shot), and the same model LoRA-finetuned on our data
(per-view, no fusion). Both allocate V times our latent budget and enforce no cross-view
consistency; we therefore compare all models as points in the rate–quality plane rather than
in a single-rate ranking. Total compression ratio is
\(r = \frac{V \cdot 3 \cdot T \cdot H \cdot W}{V' \cdot 16 \cdot T' \cdot (H/8)(W/8)}\);
our configurations span \(r = [12\times]\) (per-view, no temporal compression) to
\(r = [96\times]\) (fused views + 4x temporal compression). [Recompute exact values.]

\subsection{Main Results: The Rate--Quality Trade-off}

Table [X] and Figure [F3] present all configurations on the rate–quality plane. Three
findings structure the results.

**A shared latent can hold two views.** With cross-view fusion and view-conditioned decoding
(temporal compression off), the fused model reaches [PSNR/SSIM] on held-out identities,
[Δ] below the finetuned per-view reference — at half its latent rate. Cross-view similarity
stays at the ground-truth reference level ([x] vs [y]), i.e., view identity is preserved;
the per-view difference maps in Figure [F5] confirm the two decoded views are genuinely
distinct. Without view conditioning, the model collapses to near-identical views
(similarity [z]), confirming that decoder-side conditioning, not the fusion itself, carries
view identity.

**Temporal compression alone degrades gracefully.** Enabling Wan's native 4x temporal
compression in the per-view setting costs [Δ PSNR] relative to the uncompressed reference.
The loss is not uniform over time: the per-frame profile (Figure [F4]) shows the
characteristic cold-start dip at frame 0 and elevated error inside 4-frame chunks, with a
within-chunk bleed ratio of [x] (vs [y] across boundaries).

**Joint compression collapses.** Activating both axes yields [PSNR], [Δ] below the fused
TC-off model and [Δ] below what the sum of the two individual degradations would predict —
the drop is super-additive. Qualitatively, both artifact classes intensify simultaneously:
reconstructed views converge toward each other *and* frames inside each chunk blur together
(Figure [F5]). No new failure mode appears; the existing ones amplify, consistent with a
shared cause. We analyze this in Sec. 4.3 and argue in Sec. 5 that the cause is the fixed
16-channel latent rate.

\subsection{Analysis: Where Does Joint Compression Fail?}

\subsubsection{Temporal bleeding is a generalization failure}
Overfitting a single sequence with temporal compression reproduces the input cleanly
(bleed ratio [≈1]); the artifact appears only when training spans many identities
([bleed ratio] on the full training set, Figure [F9]). The compressed representation can
memorize temporal detail but cannot encode it for unseen content — a capacity, not an
optimization, signature.

\subsubsection{Isolating the chunk mechanism}
Decoding all latent frames in a single non-causal pass (symmetric temporal padding, no
chunk loop) removes the frame-0 dip and boundary seams and recovers [Δ PSNR], at the cost of
the causal/streaming property. This oracle bounds how much of the degradation the chunked
decoding mechanism itself causes: [fraction]; the remainder is attributable to the
compression bottleneck.

\subsubsection{Targeted interventions}
Table [Y] evaluates seven baseline-preserving interventions (Sec. 3.3), grouped by the
hypothesis they test: cold-start/boundary fixes (temporal reflection padding, learned cache
update), missing-signal fixes (sub-frame position embedding, decoder temporal attention,
high-frequency side channel), and loss-side fixes (temporal-difference loss, teacher
distillation). [Report which moved bleed ratio / PSNR and which did not.] The pattern —
[e.g., "signal-path interventions help marginally; post-hoc and loss-side ones do not"] —
supports the information-flow argument of Sec. 5.

\subsection{Ablations}

\subsubsection{Fusion mechanism}
We compare four fusion operators at the encoder bottleneck (all reducing V views to one
shared latent; temporal compression off, all else fixed): (i) *cross-view attention* with
tree merge (default); (ii) *joint self-attention* — one attention over the concatenated
token sequence of all views, followed by concat + two residual blocks; (iii) *channel-concat
Conv3d* — views stacked on channels, 1x1x1 Conv3d + GN/SiLU + two symmetric (non-causal) 3D
residual blocks; (iv) *factorized 4D convolution* — spatial 3x3 Conv2d, temporal 3x3x3
Conv3d, then a view-axis Conv3d with kernel (V,3,3) compressing V->1, a (2+1+1)D
factorization in the spirit of \cite{carreira2017quo}. [Expected: near-equivalent — which is
itself evidence that the bottleneck is latent capacity, not fusion mechanism.] Table [Z].

\subsubsection{View-conditioned decoding}
How should the shared decoder be told *which* view to produce? We compare (i) per-view
latent LoRA adapters (default), (ii) a learned per-view latent embedding
(`nn.Embedding(V, 16)` added to z), (iii) both, (iv) neither (negative control), and (v) a
fully finetuned decoder as upper bound. [Expected: (i)≈(ii)≈(iii) ≫ (iv); (v) marginally
better.] The negative control quantifies how much view identity the fused latent itself
carries: [cross-view similarity numbers].

\subsubsection{Pre-fusion encoder LoRA}
Our default freezes the per-view encoder stem entirely and adapts only the bottleneck and
decoder. Adding LoRA to the pre-fusion stem (`use_lora_before`) tests whether per-view
feature extraction needs domain adaptation before fusion. [Result + one-sentence take.]

\subsubsection{Unfreezing the temporal convolutions}
The strided temporal convolutions — the compression bottleneck itself — admit no LoRA path
(Sec. 3.1) and are frozen in all LoRA configurations. We unfreeze them (and optionally the
full encoder / full decoder) to test whether the joint-compression failure is
adaptation-constrained rather than fundamental. [If full unfreezing does not close the gap,
the capacity interpretation is strengthened — this is the key supporting ablation for the
paper's claim.]

\subsubsection{Initialization strategy}
Our default is staged: the joint model warm-starts from the converged temporal-compression
checkpoint (temporal axis first, view axis second; view-attention re-randomized). The
ablation trains from Wan-pretrained weights only, with all new modules zero-initialized
\cite{zhang2023adding}. [Result: does the curriculum over the two axes help, or does joint
training from scratch reach the same point?]

\subsubsection{Number of views}
V=2 vs V=4 [vs V=8] at fixed rate-per-view. The capacity argument predicts monotonic
degradation of the fused model with V even without temporal compression. [Result.]

\subsubsection{Training-set size}
Single sequence -> one person -> all participants (Figure [F9]): reconstruction under
compression is nearly perfect in the overfit regime and deteriorates with dataset size,
while the uncompressed configurations stay flat. [Result.] LoRA rank and discriminator
variants are deferred to the appendix [ranks 8/32/128 comparable; no discriminator ≈ 4D
multi-view discriminator > 3D PatchGAN, which shifts colors].

### Expected findings `[FROM PRESENTATION — PRELIMINARY, ALL NUMBERS MUST BE REGENERATED]`

None of the numbers below are citeable — every run gets redone under the fixed protocol in
`02_experiments.md`. Treat these April findings as *hypotheses the reruns should confirm*
(and if a rerun contradicts one, the rerun wins and the text changes):
- **Fusion mechanism barely matters** (Experiment 1): cross-attention, joint self-attention,
  and factorized 4D conv performed *almost equivalently*. This is itself a result worth
  stating: the fusion mechanism is not the bottleneck — supports the capacity framing
  (if information doesn't fit, no fusion operator recovers it).
- **View-conditioning mechanism barely matters** (Experiment 2): learnable view embeddings vs
  two per-view LoRAs — almost equivalent (overfit setting). Same interpretation.
- **Loss configuration matters a lot** (Experiment 3, 81-run sweep): quantitatively best with
  *no discriminator or the 4D multi-view discriminator*; the standard 3D PatchGAN (views
  flattened into batch) gave worse colors and less sharp outputs — evidence that if a
  discriminator is used on multi-view data, it should see the view axis jointly. Lower
  perceptual and KL weights improved pixel fidelity ("stronger regularization harms
  reconstruction").
- **LoRA rank** (ablation, overfit): rank 128 best PSNR, rank 32 best SSIM & MSE, 64 in
  between -> rank 32 chosen as default.
- **Training dynamics**: PSNR > 30 after ~13k steps (~33 epochs) on the overfit setting;
  longer training (21k steps) mainly improved *color fidelity*; remaining failures are
  fine details (nuanced facial appearance, a missing earring) — exactly the detail classes
  that latent-capacity work predicts to go first \cite{dai2023emu}.
- Working configuration from the presentation: **cross-attention fusion + rank-32 LoRA + view
  embeddings, no discriminator (or 4D discriminator), lower perceptual/KL weights.**
  Use this as the fixed backbone config for the reruns — but its *numbers* come only from
  the reruns.

Qualitative slide assets (skin-tone shifts, earring crops) may still be shown as figures if
the reruns reproduce the effect — regenerate them from the new checkpoints rather than reusing
slide exports, so figures and tables come from the same runs.

---

## 5. Discussion

**The capacity argument, made quantitative (centerpiece).**
A Wan latent stores 16 channels per 8x8x4 pixel block: 3·8·8·4/16 = **48x** compression
(ignoring float width). Fusing V=2 views into one latent doubles this to **96x**; V=4 to
**192x**. For comparison, image latents needed a widening from 4 to 16 channels just to hold
fine detail at 48x-equivalent rates \cite{dai2023emu}, DC-AE scales channels proportionally
with spatial compression \cite{chen2024deep}, and LTX-Video's 192x compression uses 128
channels \cite{hacohen2024ltx}. Our setting demands ~2-4x the information density of the
pretrained latent *at fixed width* — and the observed failure is exactly what rate exhaustion
predicts: the decoder regresses to the conditional mean along whichever axis was compressed,
manifesting as ghosting (view mean) and bleeding (temporal chunk mean). Two further
observations fit the same account `[FROM PRESENTATION]`: (i) the *choice* of fusion mechanism
and view-conditioning mechanism barely moved quality — if the latent cannot hold the
information, no operator recovers it; (ii) what fails first under joint compression is
high-frequency identity detail (nuanced facial appearance, small accessories such as
earrings) — precisely the detail class that widening image latents from 4 to 16 channels was
needed to preserve \cite{dai2023emu}. Frame it as: *we did
not fail to train the model; we measured the budget.*

**Localizing the bottleneck: encoder-side latent, not the decoder.** A sharp way to state the
finding — the decode side *works*: per-view LoRA + embeddings reliably disambiguate views from
a shared latent (the negative control without them ghosts [VERIFY with rerun E3-d]), and the
insensitivity to which conditioning mechanism is used says the decoder is not starved of
mechanism. What it is starved of is *information*: the failure appears exactly when the
encoder-side latent must hold more than its rate allows. This cleanly separates "can a shared
decoder emit distinct views?" (yes) from "can a fixed-width latent carry them?" (no, under
joint compression).

**Why post-hoc fixes cannot work.** Embeddings, residual decoders, and auxiliary losses cannot
recover information destroyed upstream by averaging or striding — a data-processing-inequality
argument. Every intervention that operated after the bottleneck (stronger embeddings,
sinusoidal codes, output residuals) failed; the interventions that helped operated on the
information path itself [VERIFY which: e.g., unfreezing the encoder, sub-frame embeddings].
Parallel finding in the literature: naive 3D extensions of image VAEs blur motion
\cite{xing2024large}; multi-view diffusion needs attention *inside* the network, not output
alignment \cite{shi2023mvdream, gao2024cat3d}.

**The frozen-bottleneck confound (be upfront).** The strided temporal convs admit no LoRA, so
part of the joint-compression failure could be adaptation-constrained rather than fundamental.
Two mitigating observations: (i) the unfreeze ablation (train_spatial/freeze_temporal=False)
[VERIFY: did full unfreezing close the gap? If not, that *strengthens* the capacity claim];
(ii) chunk artifacts (cold start, boundary seams) are reported for causal chunked video VAEs
generally \cite{zhao2024cvvae, lin2024opensoraplan, yang2024cogvideox}, i.e. they are a
property of the mechanism, not our adaptation.

**Design lessons for a working 4D tokenizer.**
1. Scale latent width with the number of compressed axes (\cite{chen2024deep, hacohen2024ltx}
   suggest roughly proportional scaling) — i.e., a 4D VAE wants 32-64 channels, which requires
   pretraining or heavy finetuning, not adapters.
2. Keep view identity out of the latent: compress only shared content; carry view as decoder
   conditioning (our per-view LoRA is the adapter-scale version; a real system would train it).
3. Asymmetric designs are the pragmatic middle ground: compress time natively, keep views
   uncompressed but *consistent* via cross-view attention — matching what 4D diffusion systems
   converged to \cite{xie2024sv4d, shao2024human4dit}.
4. Zero-init everything: it made every ablation baseline-preserving and cheap to try
   \cite{zhang2023adding}.

**Engineering reality (short paragraph or appendix — reviewers of empirical papers value
this).** Everything ran on a single L40S (48 GB) via SLURM. Making V-view training feasible
required per-view activation checkpointing of the encoder down-path and the decode body
(un-checkpointed, the per-view encoder alone held +39 GB at batch 8); at 512² the decoder
must stay checkpointed at any useful batch size, and OOM-fallback batch ladders keep the
effective batch fixed. One correctness fix worth a sentence: the PatchGAN discriminators used
in-place LeakyReLU activations, which break autograd under activation checkpointing; they were
switched to out-of-place (`paper-snapshot-aug2026`) — discriminator results predating this fix
should not be reported. (A parallel session also claimed chunked-LPIPS and dataloader fixes;
these do NOT exist in any branch — do not mention them.)

**Limitations.** 9-frame clips (rate pressure grows with T — longer clips could shift
conclusions in either direction); faces on white background only; 2-4 views of a frontal arc,
not the full rig; one backbone family (Wan 2.1); the latent-diffusion generation stage
(Stage 2 of the original plan) was not run — the study is about the representation;
compute-constrained hyperparameter coverage.

---

## 6. Conclusion

Three sentences of substance:
1. We extended a pretrained 3D video VAE to synchronized multi-view video with zero-initialized
   cross-view fusion and per-view low-rank decoding, preserving the pretrained latent
   distribution and unconditional sampling.
2. A systematic study on multi-view facial video shows the fixed 16-channel latent absorbs
   either 4x temporal compression or the fused view axis, but not both: joint compression
   exhausts the latent's rate and the decoder regresses to means — cross-view ghosting and
   intra-chunk temporal bleeding, which our diagnostics quantify.
3. Future 4D tokenizers should scale latent capacity with the number of compressed axes or
   keep view identity out of the latent; our adaptation recipe, diagnostics, and negative
   results chart the design space for that next attempt.

---

## Appendix candidates
- Camera serials + rig geometry sketch; preprocessing details (RVM settings, CCM).
- Full hyperparameters; LoRA coverage table (which modules, ranks, param counts, trainable %).
- The chunked feat_cache algorithm as pseudocode.
- Extended sweep tables (loss-weight sweep: 81 runs; discriminator grid).
- The compile/memory engineering notes (CUDA graphs, activation checkpointing) — one
  paragraph; reviewers of empirical papers value reproducibility detail.

## The "most important design decisions" list (for your supervisor)
1. Adapt a pretrained 3D VAE with zero-init pathways instead of training 4D from scratch (D1).
2. Fuse views inside the encoder at the bottleneck — not in latent space (the averaging
   failure is the evidence) (D2, D4-history).
3. Cross-view attention + tree merge as the default fusion; conv alternatives as ablations (D3).
4. View identity via per-view latent LoRA adapters on a shared frozen decoder — the decisive
   mechanism for view separation (D4).
5. LoRA-after (bottleneck+decoder) as the training regime; the structurally frozen strided
   time convs as the known constraint (D5).
6. Use the native chunked causal path for temporal compression (weight compatibility) and
   treat its artifacts with baseline-preserving interventions (3.3).
7. Keep the standard tokenizer loss; add targeted diagnostics rather than new loss terms (3.4).
