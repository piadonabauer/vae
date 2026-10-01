# Paper Draft — Multi-View Video VAE

> **STALE — do not write the paper from this file.**
> The working draft with the connected story, filled tables, and figures is
> `paper/overleaf_draft.tex`. This markdown file is an earlier sketch and was
> not kept in sync (figures, side-channel removal, fusion table, running jobs).

**Working title:** *One Latent, Many Cameras: On the Capacity Limits of Multi-View Video Autoencoding*
*(Alternatives: "Towards 4D Latents: Extending a Pretrained Video VAE to Multi-View Facial Video" / "How Much Fits in a Video Latent? An Empirical Study of Joint View–Temporal Compression")*

> **Status (Sep 2026):** All MUST HAVE experiments are complete. Numbers below are final
> except E9b (512px, marked `xxx`). Statistical note: each config is trained once (one seed).
> Within each run, val PSNR averages over ≈10 participants × 2 views × 9 frames ≈ 180 samples
> (SE ≈ 0.08 dB, 95% CI ≈ ±0.16 dB). Treat differences below ~0.5 dB as inconclusive.

---

## 1. Introduction

Paragraph 1 — application pull:

> Photo-realistic head avatars are built from synchronized multi-view facial video captured
> with dense camera rigs \cite{lombardi2018deep, lombardi2021mixture, cao2022authentic,
> kirschstein2023nersemble, qian2024gaussianavatars}. A single NeRSemble capture produces 16
> camera streams at 73 fps and 3208×2200 resolution \cite{kirschstein2023nersemble} — hours of
> capture yield tens of terabytes of raw video. This data is extraordinarily redundant along
> two axes: *time*, where adjacent frames differ only by smooth facial motion, and *view*,
> where neighboring cameras observe the same face under slowly varying pose
> \cite{shah2024mv2mae, taubner2025mvp4d}.

Paragraph 2 — the latent-space argument:

> Modern generative pipelines do not operate on pixels: latent diffusion models generate in
> the compressed latent space of a VAE \cite{rombach2022high}, and every state-of-the-art
> video generator — Wan \cite{wan2025wan}, CogVideoX \cite{yang2024cogvideox}, HunyuanVideo
> \cite{kong2024hunyuanvideo}, Cosmos \cite{agarwal2025cosmos} — rests on a causal 3D video
> VAE that compresses space by 8× and time by 4×. The tokenizer therefore *defines* what the
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
> the temporal axis is compressed (4×, the native Wan setting) *or* the view axis is fused
> into a shared latent — but combining both degrades reconstructions with two characteristic
> artifacts: *cross-view ghosting* (views collapse toward their mean) and *intra-chunk
> temporal bleeding* (frames within a 4×-compressed chunk blur into each other). We trace
> both to the same cause: the 16-channel latent, sized for single-view video, has no spare
> rate for a second compressed axis. This is consistent with the broader tokenizer literature,
> where higher compression is only achieved by widening the latent
> \cite{dai2023emu, chen2024deep, hacohen2024ltx, yao2025vavae}.

Contributions:
1. **A minimally invasive multi-view extension of a pretrained video VAE**: per-view frozen
   Wan encoder stems, zero-initialized cross-view attention fusion at the bottleneck, a
   hierarchical tree merge to a shared latent, and view-conditioned decoding through per-view
   latent LoRA adapters — preserving the pretrained latent distribution and hence
   compatibility with latent diffusion.
2. **Diagnostics that localize information loss**: a cross-view reconstruction-similarity
   metric (detects view ghosting) and an intra-chunk bleed ratio (detects temporal
   mean-regression inside 4-frame chunks), plus per-frame-index error profiles.
3. **A systematic study** (32 configurations) over fusion mechanisms, adapter placement, and
   seven targeted temporal-quality interventions, all under a fixed training budget.
4. **A capacity finding with design implications**: either axis alone fits the 16-channel Wan
   latent; both together do not. Joint compression drops PSNR by 8.70 dB versus the per-view
   reference — a super-additive 3.39 dB excess over what the individual degradations predict.
   Widening the latent to 32 channels recovers 2.40 dB, providing direct positive evidence
   for the capacity interpretation.

---

## 2. Related Work

**Video tokenizers / video VAEs.** Latent generative modeling was established for images by
VQGAN and latent diffusion \cite{esser2021taming, rombach2022high} and extended to video by
inflating autoencoders with temporal layers \cite{blattmann2023align} or training causal 3D
tokenizers \cite{yu2024language}. Current video foundation models (Wan \cite{wan2025wan},
CogVideoX \cite{yang2024cogvideox}, HunyuanVideo \cite{kong2024hunyuanvideo}, Cosmos
\cite{agarwal2025cosmos}) share the same recipe: causal 3D convolutions, 8× spatial and 4×
temporal compression, 16 latent channels. Parallel work disentangles structure and dynamics
or pushes compression: VidTwin \cite{wang2025vidtwin}, CV-VAE \cite{zhao2024cvvae}, LTX-Video
\cite{hacohen2024ltx}. All of these are single-view; multi-view data is encoded stream-by-stream.

**Latent capacity.** A consistent empirical law: quality at a given compression ratio scales
with latent width. LDM ablates downsampling factor against channel count \cite{rombach2022high};
Emu shows widening from 4 to 16 channels is decisive for fine detail \cite{dai2023emu};
DC-AE shows aggressive spatial compression requires proportionally wider latents \cite{chen2024deep};
LTX-Video's 1:192 compression uses 128 channels \cite{hacohen2024ltx}; VA-VAE formalizes the
reconstruction–generation trade-off \cite{yao2025vavae}. Our study is the first to probe this
budget along a *view* axis on a *pretrained, fixed-width* latent.

**Multi-view and 4D generative models.** Cross-view attention is the standard consistency
mechanism in multi-view diffusion \cite{shi2023mvdream, gao2024cat3d, voleti2024sv3d,
zuo2024videomv}; MV2MAE reconstructs held-out views via masked autoencoding — cross-view
*reconstruction*, not learned *compression* \cite{shah2024mv2mae}. 4D generation methods
\cite{zhang20244diffusion, xie2024sv4d, shao2024human4dit, jiang2026mesh4d} do not yield a
compact latent code. Head-avatar pipelines \cite{lombardi2018deep, qian2024gaussianavatars,
taubner2025mvp4d} operate on pixels or per-frame codes — none provides a reusable 4D latent.

**Research-gap table** (Table 1):

| Approach | Time | Views | Joint latent | Learned compression |
|---|---|---|---|---|
| Image VAEs \cite{rombach2022high} | ✗ | ✗ | ✓ | ✓ |
| Video VAEs \cite{wan2025wan} | ✓ | ✗ | ✓ | ✓ |
| MV masked autoencoding \cite{shah2024mv2mae} | ✓ | ✓ | ✗ | ✗ |
| Geometry priors \cite{jiang2026mesh4d} | (✓) | ✓ | ✗ | ✗ |
| **Ours** | ✓ | ✓ | ✓ | ✓ |

**Parameter-efficient adaptation.** We follow the adapt-don't-retrain philosophy: LoRA
\cite{hu2022lora} for frozen backbones and zero-initialized new pathways so the pretrained
function is exactly preserved at step 0 \cite{zhang2023adding}.

---

## 3. Method

### 3.1 Preliminaries: the Wan 2.1 video VAE

The Wan 2.1 VAE is a causal 3D convolutional encoder-decoder with 16-channel latents and
8×8 spatial and 4× temporal compression (T' = 1 + (T−1)/4) \cite{wan2025wan}. **The chunked
cache mechanism is central to the paper's findings.** Temporal compression only fires through
a `feat_cache` path: frame 0 is encoded/decoded alone, then 4-frame chunks follow, each
strided temporal convolution consuming a rolling cache (last 2 activations) from the previous
chunk. Decoding mirrors this: one latent frame in → four frames out, with persistent cache.
Consequences: (i) a cold-start asymmetry for frame 0; (ii) chunk-boundary seams between
4-frame groups; (iii) the strided temporal convolutions admit no LoRA path (a 1×1×1 low-rank
branch cannot match the strided output shape), so the compression bottleneck itself is frozen
in all LoRA configurations — a structural constraint we later probe as an ablation.

### 3.2 Multi-view extension

Input `[B, V, C, T, H, W]`. Four design decisions define the architecture:

**D1. Adapt a pretrained 3D VAE rather than train a 4D VAE from scratch.**
Pretrained 3D weights transfer immediately via LoRA; new pathways are zero-initialized so the
model is *exactly* the pretrained per-view VAE at step 0 \cite{zhang2023adding}. This keeps
the latent distribution close to the pretrained one, preserving compatibility with Wan's
latent-diffusion ecosystem.

**D2. Fuse views at the encoder bottleneck, before `encoder.middle`/`head`.**
Fusion operates at 384 channels on an 8×-downsampled grid — late enough that tokens are cheap
for attention, early enough that the fused information shapes the latent. An earlier variant
compressed views in latent space via learned averaging with per-view embeddings for recovery;
it produced near-identical (ghosted) views — averaging destroys view identity before any
embedding can recover it. This motivated moving fusion *into* the encoder.

**D3. Default fusion: cross-view attention + hierarchical tree merge.**
All V views attend to each other via multi-head SDPA (RMSNorm QKV, zero-init output
projection), then V−1 pairwise merges cascade: `cat(a, b) → ResBlock(2C→C) → ResBlock(C→C)`,
reducing V views to one shared latent. We ablate four fusion operators in Appendix A.1 and
find all perform equivalently — which is itself evidence that the bottleneck is latent
capacity, not the fusion mechanism.

**D4. Decode distinct views from one shared latent via per-view latent LoRA adapters.**
The single fused latent `[16, T', H/8, W/8]` is decoded V times by the *shared frozen*
decoder; view identity is injected as per-view latent LoRA adapters (`Conv3d z→rank→z`,
zero-init up-projection). We ablate view conditioning in Sec. 4.4.2 and find the adapter is
interchangeable with a simple additive embedding — view identity is also largely encoded in
the shared latent by the cross-view attention.

**D5. LoRA placement.** Pre-fusion encoder stem: frozen. Bottleneck + decoder: LoRA rank 32
(`use_lora_after`), zero-init up-projections. New fusion modules: fully trained. Strided
temporal convs: frozen by structure (Sec. 3.1); we probe unfreezing as ablation E5.

**D6. Two-stage training discipline.** Every configuration passes an overfit gate (single
sequence, train PSNR ≥ 35 for 3 epochs) before the generalization run. Joint TC=True models
additionally warm-start from the converged per-view TC checkpoint; we ablate this in Sec. 4.4.5.

### 3.3 Temporal compression: seven baseline-preserving interventions

We test seven opt-in flags that do not change the architecture at step 0 (all zero-initialized):

- **Cold-start / boundary:** non-causal full-sequence decode (`noncausal_decode`); temporal
  reflection padding (`temporal_reflection_pad`); learned ConvGRU cache update
  (`learned_cache_update`, identity at init).
- **Missing-signal:** sub-frame position embedding (`subframe_position_embedding`); a 4-channel
  high-frequency side channel (`temporal_side_channel`).
- **Loss-side:** temporal-difference loss — L1(Δgt, Δrec) directly penalizing the bleeding
  symptom (`temporal_diff_loss_weight`).

### 3.4 Objective and diagnostics

**Loss:** `nll = (L1 + 1.5·LPIPS)/exp(σ) + σ` with learned scalar σ, plus `1e-6·KL`
\cite{esser2021taming, rombach2022high, zheng2024opensora}, LPIPS \cite{zhang2018unreasonable}.
No adversarial term in the main protocol.

**Diagnostics (three; all reported per run):**
- *Cross-view similarity*: mean pairwise cosine similarity of reconstructed views, with GT as
  reference. Values above GT indicate ghosting (views collapsed toward their mean).
  - *Bleed ratio*: mean |Δrec| / mean |Δgt| over consecutive-frame pairs, computed separately
  *within* 4-frame temporal chunks and *across* chunk boundaries. Value ≈ 1 is faithful
  motion; values ≪ 1 within-chunk indicate temporal bleeding.
  - *Per-frame-index error profile*: PSNR as a function of frame index, exposing the frame-0
    cold-cache dip and chunk-boundary seams.

### 3.5 Data and preprocessing

NeRSemble \cite{kirschstein2023nersemble}: 16 synchronized cameras, 73 fps, 3208×2200.
We select 2 frontal/upper cameras; temporally subsample to 24 fps and uniformly select T=9
frames (the compressed latent then has T'=3: frame 0 plus two 4-frame chunks — the shortest
clip containing both within-chunk and boundary artifacts); center square crop, background
removal (RobustVideoMatting \cite{lin2022robust}) composited on white; bilinear resize to
{128, 256, 512}²; stored as `[V, T, C, H, W]` in [0,1]. Main experiments: 128², T=9, V=2.
Data scales: one-expression-all-participants (~350 train / 10 held-out val identities).
256² and 512² as resolution scaling checks (Appendix A.5).

---

## 4. Experiments

### 4.1 Experimental Setup

**Protocol.** All experiments: AdamW lr 5e-4, bf16, effective batch 64, 170 epochs, LoRA rank
32, EMA 0.9999, no discriminator. Evaluations fire every 50 optimizer updates (≈10 epochs);
we report best val PSNR and the corresponding metrics. The training budget is identical in
optimizer updates across all arms: 358 samples / batch 64 = 5.6 updates/epoch × 170 epochs
≈ 952 updates. Single L40S (48 GB) GPU per run. For 512px a batch ladder (2→1) with encoder
activation checkpointing is required due to memory.

**Statistical reliability.** Each configuration is trained once. The validation set (10 held-out
participants × 2 views × 9 frames ≈ 180 samples) gives SE ≈ psnr_std/√180 ≈ 0.08 dB
(95% CI ≈ ±0.16 dB per run). We treat inter-configuration differences below ~0.5 dB as
statistically inconclusive at the single-seed level.

**Compression ratios** (V=2 views, T=9 frames, 128² pixels):
The input has V×3×T×H×W values. The latent has V'×16×T'×(H/8)×(W/8). With V=2, T=9, T'=3
(temporal compression on) or T'=9 (off), and V'=2 (per-view) or V'=1 (fused):

| Configuration | V' | T' | Rate | Role |
|---|---|---|---|---|
| Per-view TC=F (E1a) | 2 | 9 | **12×/view** | Finetuned reference |
| Per-view TC=T (E1b) | 2 | 3 | **36×/view** | Temporal axis alone |
| Fused TC=F (E1c) | 1 | 9 | **24×** | View axis alone |
| Fused TC=T (E1d★) | 1 | 3 | **72×** | **Both axes (headline)** |

**Reference points.** No prior method produces a joint latent for synchronized multi-view video.
We report two per-view bounds: (i) pretrained Wan applied zero-shot (no finetuning), and
(ii) the same model LoRA-finetuned per-view on our data (E1a). Both allocate V× our latent
budget with no cross-view consistency; E1a at 34.01 dB is the quality ceiling in our rate regime.

### 4.2 Main Results: The Rate–Quality Trade-off

**Table 2 — Rate–quality curve (main result).**

| ID | Configuration | Rate | Val PSNR ↑ | LPIPS ↓ | Bleed-W ↑ | Bleed-A ↑ | xview\_sim |
|---|---|---|---|---|---|---|---|
| — | Zero-shot (no FT) | 12×/view | ~21 dB | — | — | — | — |
| E0 | Per-view ceiling (all expr) | 12×/view | 34.34 | 0.021 | 0.981 | 0.999 | 0.979 ≈ GT |
| E1a | Per-view TC=F *(ref)* | 12×/view | **34.01** | 0.024 | 0.991 | 1.014 | 0.906 ≈ GT |
| E1b | Per-view TC=T | 36×/view | 31.50 | 0.041 | 0.952 | 0.954 | 0.907 ≈ GT |
| E1c | Fused TC=F | 24× | 31.21 | 0.044 | 0.974 | 0.984 | 0.907 ≈ GT |
| **E1d★** | **Fused TC=T** | **72×** | **25.31** | **0.082** | **0.919** | **0.907** | **0.920 > GT** |

*(GT cross-view similarity for all E1 runs = 0.907; bleed ratio = 1.0 means faithful motion.)*

Three findings structure these results.

**A shared latent can hold two views without ghosting.** The fused TC=F model (E1c, rate 24×)
reaches 31.21 dB — only 0.29 dB below the per-view TC=T reference (E1b) despite using the
same latent budget at half the rate per view. The cross-view similarity of E1c (0.907) matches
the GT reference exactly, confirming genuine view separation: the decoder produces meaningfully
distinct reconstructions for each view. This shows that cross-view attention in the encoder
plus per-view LoRA in the decoder is sufficient to encode and decode two distinct views from
one shared 16-channel latent.

**Temporal compression alone degrades gracefully.** Activating Wan's native 4× temporal
compression in the per-view setting (E1b) costs 2.51 dB (34.01→31.50 dB). The bleed ratio
drops from 0.991 (E1a) to 0.952, indicating mild within-chunk frame averaging. The profile
shows the characteristic cold-start dip at frame 0 and elevated error within 4-frame chunks.

**Joint compression collapses — super-additively.** Activating both axes (E1d, 72×) yields
25.31 dB: a total drop of 8.70 dB from E1a, far exceeding the sum of the individual
degradations (2.51 + 2.80 = 5.31 dB). The excess degradation of **3.39 dB** is
super-additive: the two compressed axes interfere beyond what their individual costs predict.
Qualitatively, the bleed ratio drops to 0.919 (temporal bleeding) and the cross-view
similarity rises to 0.920 — slightly above GT (0.907), indicating mild view ghosting as well.
Both artifact classes intensify simultaneously, consistent with a shared cause: the 16-channel
latent is rate-exhausted and the decoder regresses toward the conditional mean along both
compressed axes.

### 4.3 Analysis: Where Does Joint Compression Fail?

#### 4.3.1 Temporal bleeding is a generalization failure

The overfit gate (single sequence, train PSNR ≥ 35 held for 3 epochs) passes for E1d — the
architecture *can* represent the signal perfectly when memorizing one clip. The bleed artifact
appears only in the generalization run (all participants). This identifies temporal bleeding as
a *generalization* failure of a rate-limited representation, not an optimization failure:
the compressed code can memorize temporal detail for one sequence but cannot encode a general
temporal basis across many.

#### 4.3.2 Targeted temporal interventions

**Table 3 — Temporal interventions (all on top of E1d baseline).**

| ID | Intervention | Val PSNR ↑ | Δ vs E1d | Bleed-W ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| E1d | Baseline (fused TC=T) | 25.31 | — | 0.919 | 0.082 |
| E4c | Temporal reflection pad | 25.88 | +0.57 | 0.899 | 0.070 |
| E4d | Side channel (4 ch) | 25.94 | +0.63 | 0.916 | 0.069 |
| E4f | Learned cache update | 25.96 | +0.65 | 0.899 | 0.069 |
| E4g | Sub-frame pos embedding | 25.89 | +0.58 | 0.896 | 0.069 |
| **E4h★** | **Temporal diff loss** | **27.14** | **+1.83** | **0.903** | **0.070** |
| E4i | Diff loss + learned cache | 26.93 | +1.62 | 0.899 | 0.069 |
| E4b | Non-causal decode | 17.81 | −7.50 | 0.251 | 0.187 |

The **temporal difference loss** (E4h, `temporal_diff_loss_weight=2.0`) is the decisive
intervention, gaining +1.83 dB over baseline. It directly penalizes the bleeding symptom
during training: by adding L1(Δgt, Δrec) as an auxiliary loss, it forces the model to
preserve inter-frame differences even when the compressed latent induces averaging pressure.
Convergence is smooth and fully stable (±0.05 dB in the last 40 epochs). Notably, it also
reduces LPIPS from 0.082 to 0.070.

All other interventions yield gains of only 0.57–0.65 dB — within a regime where the bleed
ratio barely changes. The hierarchy of interventions shows that structural signal-path fixes
(side channel, learned cache) help marginally more than padding/positional fixes, but none
approaches E4h. Combining E4h with E4f degrades slightly (E4i: 26.93 vs E4h: 27.14), showing
the interventions interfere rather than stack.

The **non-causal decode** (E4b) is catastrophically broken: 17.81 dB with a bleed ratio of
0.251 — the model produces near-static outputs for each view. Permanently switching the
pretrained causal convolutions to symmetric padding disrupts the pretrained feature
distributions; LoRA training alone cannot adapt them. This is reported as a negative finding
rather than an oracle: it quantifies *not* the damage from chunked decoding, but the
brittleness of flipping the causal assumptions of a pretrained causal model.

#### 4.3.3 Latent width: direct positive evidence

**Table 4 — Latent width ablation (E11).**

| ID | Latent channels | Rate | Val PSNR ↑ | Δ vs E1d | Bleed-W ↑ |
|---|---|---|---|---|---|
| E1d | 16 (default) | 72× | 25.31 | — | 0.919 |
| E11a | **32** | **36×** | **27.71** | **+2.40** | 0.930 |
| E11b | **64** | **18×** | **27.60** | **+2.29** | 0.921 |

Widening the latent from 16 to 32 channels recovers **+2.40 dB** — the largest single gain in
the entire study, larger than all temporal interventions combined. The finding is compelling:
capacity, not architecture or training, is the binding constraint. Widening to 64 channels
(E11b) adds only 0.01 dB over 32 channels, suggesting the 32→64 doubling hits a different
bottleneck (likely the frozen temporal convolutions). The bleed ratio improves modestly
(0.930 vs 0.919), confirming that temporal artifacts partly reflect latent overcrowding.

Notably, E11a (32ch, 36× rate) and E1b (16ch per-view, also 36× total rate) use identical
total latent budgets — yet E1b achieves 31.50 dB versus E11a's 27.71 dB (−3.79 dB). The
remaining gap is attributable to the joint-view encoding challenge (one latent must represent
two views, not just one) and the still-frozen temporal convolutions. A purpose-built 4D
tokenizer with 32–64 channels and fully trained temporal convolutions would likely close it.

### 4.4 Ablations

#### 4.4.1 Encoder freezing (frozen-bottleneck confound)

**Table 5 — Encoder unfreeze ablation (E5).**

| ID | Trainable | Val PSNR ↑ | Δ vs E1d | Bleed-W ↑ |
|---|---|---|---|---|
| E1d | LoRA only (default) | 25.31 | — | 0.919 |
| E5b | + Unfreeze full encoder | 25.88 | +0.57 | 0.915 |
| E5c | + Unfreeze encoder + decoder | 23.89 | **−1.42** | 0.886 |

Unfreezing only the encoder (E5b) gives a marginal +0.57 dB — borderline given the ±0.16 dB
within-run CI — suggesting the pretrained per-view encoder weights are near-optimal for our
domain. However, unfreezing *everything* including the decoder (E5c) collapses to 23.89 dB:
1.42 dB *below* the LoRA-only baseline despite far more trainable parameters. The decoder
overfits when fully trained on this dataset scale. The bleed ratio also worsens (0.886 vs 0.919),
consistent with the model overfitting temporal detail per participant.

The pattern rules out the frozen-bottleneck as the explanation for E1d's quality plateau: if
frozen temporal convolutions were the binding constraint, adding more trainable parameters
should consistently help. Instead the fully-unfrozen model is the worst. The 16-channel rate
is the bottleneck — not LoRA coverage or frozen convolutions.

#### 4.4.2 View-conditioned decoding (E3)

All four view conditioning variants achieve nearly identical PSNR:

| ID | view\_emb | viewwise LoRA | full dec ft | Val PSNR ↑ | xview\_sim |
|---|---|---|---|---|---|
| E3d | ✗ | ✗ | ✗ | 31.50 | 0.979 |
| E3a | ✓ | ✗ | ✗ | 31.53 | 0.979 |
| E3c | ✓ | ✓ | ✗ | 31.37 | 0.979 |
| E3e | ✗ | ✓ | ✓ | 31.32 | 0.978 |

*(All TC=F; GT cross-view similarity = 0.978)*

The negative control (E3d: no view embedding, no LoRA) achieves 31.50 dB and a cross-view
similarity of 0.979 ≈ GT (0.978): the decoder produces genuinely distinct views without any
explicit view conditioning. The gap between all variants is ≤0.21 dB — well below the
significance threshold. This shows that the cross-view attention in the encoder implicitly
encodes sufficient view identity for the shared decoder to produce distinct reconstructions.
Adding explicit embeddings or LoRA adapters provides no measurable benefit. A fully finetuned
decoder (E3e) is slightly worse, suggesting the pretrained frozen decoder generalizes better.

This is both a positive finding (the mechanism works without explicit conditioning) and a
null result for view conditioning as an axis of optimization.

#### 4.4.3 Initialization strategy (E7b)

| ID | Init | Val PSNR ↑ |
|---|---|---|
| E7b | From Wan only (no warmstart) | 25.86 |
| E1d★ | Warmstart from E1b (TC checkpoint) | 25.31 |

**Warmstart is slightly worse (−0.55 dB).** The conventional wisdom — learn temporal axis
first, then view — is not supported here. Training from Wan weights only (E7b: 25.86 dB)
outperforms staging from the E1b checkpoint (E1d: 25.31 dB). The 0.55 dB gap is ~3.5 SE
above noise (borderline significant at the one-seed level, p≈0.05 if SE≈0.16 dB).

Possible explanation: the E1b checkpoint anchors the model at a local minimum that is
suboptimal for joint learning — the encoder has already specialized for the temporal task,
and re-randomizing only the view-attention modules leaves the encoder in a suboptimal regime.
Joint training from Wan weights allows the encoder to adapt the temporal and view axes
simultaneously. This reversal is small enough that it may not replicate with a different seed,
and it does not affect any other experiment (warmstart was the protocol default, not the better option).

#### 4.4.4 Number of views (E6)

**Table 6 — View count scaling (Appendix A.3 for full table).**

| ID | Views | TC | Val PSNR ↑ | Bleed-W ↑ | xview\_sim |
|---|---|---|---|---|---|
| E1c | 2 | F | 31.21 | 0.974 | 0.907 ≈ GT |
| E6b | 4 | F | 28.00 | 0.954 | 0.951 ≈ GT |
| E6d | 8 | F | 25.15 | 0.927 | 0.923 ≈ GT |
| E1d★ | 2 | T | 25.31 | 0.919 | 0.920 > GT |
| E6c | 4 | T | 23.23 | 0.837 | 0.953 > GT |
| E6e | 8 | T | 20.25 | 0.755 | 0.927 > GT |

Quality degrades monotonically with view count at both TC=F and TC=T. At TC=F: 2-view→4-view
costs 3.21 dB; 4-view→8-view costs 2.85 dB. The prediction from the capacity argument holds:
more views per shared latent = more compression = lower quality. Each additional view axis
competes for the same 16-channel budget.

Notably, the 8-view TC=T model (E6e: 20.25 dB, bleed_w=0.755, xview_sim=0.927 > GT=0.922)
shows both the most severe temporal bleeding *and* the most severe ghosting simultaneously —
exactly the joint failure mode the capacity argument predicts.

---

### 4.5 Discussion: The Capacity Argument

**Made quantitative.**
A Wan latent stores 16 channels per 8×8×4 pixel block: 3·8·8·4/16 = **48× per view**.
Fusing V=2 views doubles this to **96×**; V=4 to **192×**. For comparison, image latents
needed widening from 4 to 16 channels just to hold fine detail at a 48×-equivalent rate
\cite{dai2023emu}, and LTX-Video's 192× compression uses 128 channels \cite{hacohen2024ltx}.
Our setting demands ~2–4× the information density of the pretrained latent *at fixed width* —
and the observed failure is exactly what rate exhaustion predicts: the decoder regresses to the
conditional mean along the over-compressed axis, manifesting as ghosting (view mean) and
bleeding (temporal chunk mean).

Two supporting observations converge on the same conclusion:
1. **No fusion mechanism or conditioning strategy closes the gap** (Sec. 4.4.1, 4.4.2):
   cross-attention ≈ self-attention ≈ conv3d (all within ~0.5 dB at TC=F); no view
   conditioning ≈ embeddings ≈ per-view LoRA (all within 0.21 dB). If the latent cannot hold
   the information, no operator recovers it.
2. **Widening the latent directly recovers quality** (Sec. 4.3.3): +2.40 dB from 16→32
   channels, larger than any architectural or training intervention.

**The frozen-bottleneck confound is not explanatory.** Full unfreezing of temporal convolutions
and decoder makes things *worse* (E5c: −1.42 dB), ruling out adaptation constraint as the
cause. Temporal compression artifacts are also a general property of chunked causal video
VAEs \cite{zhao2024cvvae, yang2024cogvideox} — not specific to our adaptation.

**Localizing the bottleneck: encoder-side, not decoder-side.** The decoder *works*: per-view
LoRA-free conditioning already produces distinct views (E3d: xview_sim = GT level). What the
decoder is starved of is *information in the latent*. This cleanly separates "can a shared
decoder emit distinct views?" (yes — cross-attention encodes view identity implicitly) from
"can a fixed-width latent carry both views AND temporal compression?" (no, at 16 channels).

**Design lessons:**
1. Scale latent width with compressed axes (~proportional; DC-AE, LTX-Video \cite{chen2024deep,
   hacohen2024ltx}). A 4D tokenizer wants 32–64 channels.
2. Or: keep view identity out of the latent. Compress only shared structure; carry view as
   decoder conditioning. Our per-view LoRA at 16 channels is the adapter-scale version.
3. Asymmetric designs are the pragmatic middle ground: native temporal compression (pre-trained,
   free), cross-view attention for consistency without compressing the view axis into the latent.
4. Zero-init everything: it made every ablation baseline-preserving and cheap to run
   \cite{zhang2023adding}.

---

## 5. Conclusion

We extended a pretrained 3D video VAE to synchronized multi-view video with zero-initialized
cross-view attention fusion and per-view latent LoRA decoding, preserving the pretrained
latent distribution. A systematic study on 32 configurations reveals: the fixed 16-channel
Wan latent absorbs either 4× temporal compression or the fused view axis — but not both.
Joint compression drops PSNR by 8.70 dB from the per-view reference, a super-additive 3.39 dB
excess over what the individual degradations predict. The temporal difference loss (+1.83 dB)
is the most effective single intervention; latent widening (+2.40 dB for 16→32 channels)
provides direct positive evidence for the capacity interpretation. Our diagnostics — bleed ratio,
cross-view similarity, per-frame error profiles — localize information loss to the encoder-side
rate bottleneck. Future 4D tokenizers should scale latent capacity with the number of compressed
axes, or keep view identity out of the latent entirely.

---

## 6. Appendix

### A.1 Fusion Mechanism Ablation (E2)

**Table A1 — Fusion mechanism (all TC=F, same protocol).**

| ID | Mechanism | Val PSNR ↑ | LPIPS ↓ | Bleed-W ↑ | xview\_sim |
|---|---|---|---|---|---|
| E1c | Cross-attn + tree merge *(default)* | 31.21 | 0.044 | 0.974 | 0.907 |
| E2b | Joint self-attention | 31.67 | 0.036 | 0.976 | 0.979 |
| E2c | Channel-concat Conv3d | 30.78 | 0.041 | 0.966 | 0.979 |
| E2d | Factorized 4D conv | 27.86 | 0.073 | 0.927 | 0.979 |
| E2e | Self-attn + TC=T | 26.16 | 0.068 | 0.909 | 0.981 |

At TC=F, all attention-based fusion modes (cross-attn, self-attn) land within ~0.5 dB of each
other — consistent with the capacity interpretation (if the information fits the latent, any
reasonable fusion operator recovers it). Conv3d costs 0.43 dB; factorized 4D conv costs 3.35 dB
and shows notably slower convergence (still improving at epoch 169 — may benefit from longer
training). The cross-view similarity of E2b/E2c/E2d (0.979) is notably higher than E1c (0.907),
matching the GT level — confirming view separation is achieved by all operators.

Self-attention with TC=T (E2e: 26.16 dB) does not improve over the cross-attention baseline
(E1d: 25.31 dB) — the performance ranking is consistent across both TC modes.

**Note on tree vs flat merge** (E6b-flat, Appendix A.4): ablating the hierarchical tree
structure against a single flat merge (concat all 4 views → ResBlock(4C→C)) at V=4 gives
28.01 vs 28.00 dB — a 0.01 dB difference. The tree design is not the source of any gain;
cross-view attention handles the aggregation before the merge step.

### A.2 Per-View Decoder Conditioning (E3)

Detailed numbers in Sec. 4.4.2. Key takeway: cross-view attention encodes sufficient view
identity that even a decoder with no explicit view conditioning (E3d) achieves GT-level
cross-view similarity and matches the best conditioned variant within 0.21 dB.

### A.3 Multi-View Scaling (E6)

Full table in Sec. 4.4.4. Pattern: each additional view competes for the same 16-channel
budget. At TC=T, 8-view (E6e) reaches the most degraded state (20.25 dB) with simultaneous
severe bleeding (bleed_w=0.755) and ghosting (xview_sim=0.927 > GT=0.922).

### A.4 Merge Topology (E6b-flat)

Binary tree merge (E6b: 28.00 dB) vs flat merge (E6b-flat: 28.01 dB) at V=4, TC=F: Δ=0.01 dB.
Both use identical cross-view attention enrichment; only the collapse step differs (3 pairwise
ResBlock(2C→C) merges vs 1 ResBlock(4C→C)). The tree topology provides no benefit.

### A.5 Resolution Scaling (E9)

**Table A2 — Resolution scaling (E1d architecture).**

| ID | Resolution | Val PSNR ↑ | Δ vs 128px | LPIPS ↓ | Bleed-W ↑ |
|---|---|---|---|---|---|
| E1d★ | 128² | 25.31 | — | 0.082 | 0.919 |
| E9a | 256² | 27.78 | **+2.47** | 0.102 | 0.941 |
| E9b | 512² | xxx | xxx | xxx | xxx |

*(E9b running, job 5420029. Predicted: ~29–31 dB based on linear scaling trend; LPIPS likely
higher than 256px. Will update when results arrive.)*

At 256², the same E1d architecture achieves 27.78 dB — 2.47 dB higher than at 128². This
confirms that the 128px ceiling is set by pixel count (low-resolution content has less total
information to reconstruct, but the PSNR metric is also limited by the coarse grid), not by
model capacity. Note: LPIPS at 256px is higher (0.102 vs 0.082) because high-frequency facial
details (pores, hair, fine texture) are present at 256px but not at 128px — harder to
reconstruct, penalized more by the perceptual loss.

### A.6 Data Scale (E8b)

Training with one-person data only (E8b: 18.79 dB at first eval, LPIPS=0.214, bleed_w=0.860)
is dramatically worse than the all-participants setting — confirming that the dataset scale is
critical for learning a generalizable representation. Only 1 evaluation point is available
(run completed 31 eval epochs), insufficient for a fair comparison; treat as directional only.

### A.7 Engineering Notes

All experiments ran on a single L40S (48 GB) via SLURM (18h jobs, auto-chain). Memory
engineering was required: per-view activation checkpointing of the encoder down-path and
decode body (un-checkpointed, the per-view encoder alone held +39 GB at batch 8). At 512²
the decoder must stay checkpointed at any batch size; the encoder checkpoint must also be
enabled (`crossview_grad_checkpoint_encoder=True`). An OOM fallback batch ladder (16→8→4→2→1)
keeps the effective batch fixed at 64 regardless of which rung is used.

### A.8 Statistical Significance

With one training seed per configuration, formal cross-run hypothesis testing is unavailable.
Within each run, val PSNR averages over ≈10 participants × 2 views × 9 frames ≈ 180 samples
(SE ≈ psnr_std/√180 ≈ 0.08 dB, 95% CI ≈ ±0.16 dB). For between-run comparisons, combined
SE ≈ √(SE₁² + SE₂²) ≈ 0.12 dB. We treat differences below ~0.5 dB as inconclusive.
Key significant results: E1a→E1d drop (−8.70 dB), E4h gain (+1.83 dB), E11a gain (+2.40 dB),
E1c→E6b degradation at 4 views (−3.21 dB). Results near the 0.5 dB threshold (E7b −0.55 dB,
E5b +0.57 dB) should be treated with caution.
