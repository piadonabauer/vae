# Evaluation audit — 2026-10-02

Audit of all eval metrics after the PSNR/SSIM clamp bug (fixed in commit `40295bb`),
plus a full zero-shot baseline audit. Repo at branch `paper-experiments`
(HEAD `511529d`). All file:line references are to that commit. Companion artifacts:

- `paper/clean_retrain_metrics_corrected.md` — corrected table, old/new side by side
- `paper/audit/metrics_unified.py` — single documented metric module
- `paper/audit/test_metrics_unified.py` — unit tests (9/9 pass)
- `paper/audit/build_corrected_table.py` — reproducible table generation from wandb
- `paper/audit/zeroshot_local_check.py` + `paper/audit/zeroshot_local_results.json` —
  local zero-shot experiments (pretrained Wan 2.1 on a natural 720p clip)
- `paper/figures_cvpr/*.pdf` — regenerated from corrected numbers

---

## Part 1 — metric audit and recomputation

### 1(a) Assumed vs received value range, per metric

All code in `Open-Sora/scripts/vae/train.py`. The eval loader yields clips in
`[-1,1]` (`vae_target_range="[-1,1]"`), and the model output is `clamp(-1,1)`.

| metric | function (line) | range it ASSUMES | range it RECEIVED (pre-fix) | verdict |
|---|---|---|---|---|
| PSNR (`psnr_mean`) | `compute_psnr`:546 via `compute_metrics`:639 | `[0,1]` (MAX=1, clamps at :653) | **`[-1,1]`** | **BUGGED** — negative half clamped to 0; under-reports 1.6–3.4 dB (arm-dependent) |
| SSIM (`ssim_mean`) | `compute_ssim`:572 via `compute_metrics`:639 | `[0,1]` (data_range=1) | **`[-1,1]`** clamped to `[0,1]` | **BUGGED** — same clamp; biased low |
| PSNR per view | `evaluate_model`:1372–1384 | `[0,1]` | `x01 = (x+1)/2` — correct | OK (always) |
| LPIPS | `evaluate_model`:1388–1404 | `[-1,1]` (LPIPS net convention) | `x01*2-1` = `[-1,1]` | OK — fed exactly what it expects |
| Bleed-W / Bleed-A | `compute_intra_chunk_bleed_metrics`:708 | ratio of temporal-diff energies — **range-free** (scale cancels) | `x01` | OK |
| XView (cross-view cosine) | `compute_cross_view_similarity`:774 | cosine — scale-invariant but NOT shift-invariant; assumes consistent range | `x01` for GT and rec alike | OK |
| Per-frame PSNR | `compute_metrics_per_frame`:881 | `[0,1]` | `x01` | OK |
| Qual dumps | `evaluate_model`:1432–1442 | `[0,1]`→uint8 | `x01` | OK |

So **only the headline `psnr_mean`/`ssim_mean` were biased**; every diagnostic
(per-view, per-frame, LPIPS, bleed, xview, dumps) was always computed on the
correctly remapped `x01` path. This is also what makes recovery possible: the
corrected PSNR equals the mean of the per-view PSNRs that were logged all along.

### 1(b) Unified metric module + unit tests

`paper/audit/metrics_unified.py` exposes one entry point:

```python
compute_all_metrics(gt, rec, *, value_range, chunk_size=4, lpips_fn=None)
```

`value_range` is **mandatory** (`"[0,1]"` or `"[-1,1]"`); tensors are converted
to `[0,1]` exactly once (`to_unit_range`) and everything downstream assumes unit
range. SSIM is the canonical 11×11 Gaussian-window Wang SSIM with *valid*
convolution (the train.py global-moment version is kept as `ssim_global` for
comparison only). Unit tests (`test_metrics_unified.py`, 9/9 pass, run with
`/home/coder/venvs/inpaint-test/bin/python -m pytest paper/audit/test_metrics_unified.py`):

1. identical inputs → PSNR = ∞, SSIM = 1, LPIPS ≈ 0
2. Gaussian noise σ=0.05/0.1 on [0,1] → PSNR within 0.1 dB of −20·log10(σ)
3. white GT vs black rec → PSNR 0 dB, SSIM = C1/(1+C1) (analytic)
4. same data given as [−1,1] and as [0,1] → bitwise-identical metrics
5. Bleed-W pair indices for T=9, chunk=4 → within {1-2,2-3,3-4},{5-6,6-7,7-8};
   across {0-1},{4-5}; frame 0 in no within-pair
6. clamp-bug reproduction: feeding [−1,1] into a [0,1]-assuming PSNR
   reproduces the biased numbers
7. aggregation: mean-of-per-sample vs pooled-MSE PSNR diverge (~0.3 dB) on
   heterogeneous batches — guards the convention choice

### 1(c) PSNR aggregation convention (the ONE convention)

**PSNR = mean over (clip, view) samples of per-sample PSNR, computed on [0,1].**

- This equals the corrected `psnr_mean` and the mean of `psnr_view*`, so logged
  per-view numbers, the corrected table, and the figures are all mutually consistent.
- Pooled-MSE PSNR (`psnr_pooled` in the module) is ~0.3 dB lower on our data and
  is reported nowhere except as a reference field.

### 1(d) Corrected table

See `paper/clean_retrain_metrics_corrected.md`. Method: the corrected PSNR per arm
is the mean of `final_eval/val/psnr_view*` from the clean-wave wandb run
(entity `pia-uni`, project `wan_multiview_vae_paper`), matched to the table row by
the biased `psnr_mean` (±0.02); E4h recovered from `paper/clean_peraxis_metrics.json`
(cluster jsonl). 29 of 30 arms corrected. Headlines: E0 34.30→**36.02**,
E1a 34.20→**35.89**, E1d 25.96→**28.27**, combo 29.38→**31.39**,
zero-shot 19.71→**23.07**. Bias is arm-dependent (+1.59 to +3.4 dB), largest for
weak reconstructions.

**Cannot be recomputed without the cluster** (dumps/jsonl live on Vector, no SSH
from this machine):

- **E5b** corrected PSNR (its final_eval wandb summary lacks per-view fields) —
  refill from its `final_eval_metrics.jsonl` or dump.
- **Corrected SSIM for every arm** — per-view SSIM was never logged; recompute
  from `final_eval_dump_{label}.pt` with `metrics_unified.compute_all_metrics`.
  (Marked `dump` in the table.)
- **Per-frame PSNR** exists for only the 15 arms in `clean_peraxis_metrics.json`;
  the rest need their jsonl.
- Exact per-sample refill may shift wandb-derived values by <~0.05 dB
  (batch-weighting of the per-view means).

### 1(e) Recomputed deltas — sign changes and <0.5 dB flags

Full table in `clean_retrain_metrics_corrected.md`. Summary:

- Per-view TC-on (E1b 33.19) vs fused TC-on (E1d 28.27): **+4.92 dB** (was +5.41) — holds.
- Channel widening 16→32→64: +1.52 dB then +1.14 dB (was +1.56/+1.28) — holds.
- Temporal-diff loss: @16ch **+0.17 dB** (E4h−E1d, was +0.11 — below 0.5 dB
  before and after, flag stands); with 32-ch latent **+1.60 dB** (combo−E11a,
  was +1.86) — holds.
- Additive prediction: old 34.20−2.83−3.45 = 27.92 vs actual 25.96 (excess +1.96);
  corrected 35.89−2.70−3.15 = **30.04 vs actual 28.27 (excess +1.77)** —
  super-additivity conclusion survives.
- **ONE SIGN FLIP**: E2b−E1d (self-attn fusion TC-off vs joint baseline)
  −0.43 → **+0.24**. Any paper sentence ordering E2b below E1d must be revised.
- **Six pairwise ranking flips** among arms (enumerated in the corrected md):
  E1d/E2b, E2b/E2e, E2b/E4h, E2b/E7_p3.0, E2b/E7_kl1e7, E3c/E3d. All involve
  arms within ~0.5 dB of each other; E2b is the main mover (largest bias, +2.98 dB).

### 1(f) Figures

All 12 PDFs in `paper/figures_cvpr/` regenerated from corrected numbers via
`make_figures_cvpr.py` (rate–quality, latent-width, interventions, data-scale,
view-count, resolution, qualitative grids with PSNR labels updated to corrected
values). The single-sequence data-scale point is marked † (not correctable from
logs; needs cluster). Qualitative tiles themselves are pixel dumps and were never
affected by the bug.

### 1(g) Did the clamp touch training? — LOUD section

**Losses: clean.** L1, LPIPS, KL, and discriminator losses in
`Open-Sora/opensora/models/vae/losses.py` contain no pixel-range clamp (the only
clamp is the standard `d_weight` clamp at losses.py:266). Gradients were never
affected. The corrected numbers re-score existing checkpoints; nothing needs
retraining for the bias itself.

**However, three non-loss training-time consumers DID use the biased PSNR —
stating this loudly as requested:**

1. **`train_psnr_guard`** (train.py:3720–3765, threshold 15 dB, on by default)
   consumed the biased train-batch PSNR (train.py:2798–2824). Verified harmless:
   every clean-wave run completed its full 170 epochs (E0: 20 by design) and all
   biased train PSNRs were far above 15, so the guard never fired. No run was
   killed by the bug.
2. **Overfit gate** (`--stop_at_train_psnr 35`, OVERFIT=1 smoke runs only)
   compared the biased PSNR against 35 → effectively a stricter ~37 dB true
   threshold. These runs are PASS/FAIL smoke tests and none feed the paper table.
3. **Best-val dump selection** (train.py:3638–3646) picked the "best" epoch by
   biased `psnr_mean`. Within a run the bias is approximately constant across
   epochs, so the argmax is very unlikely to change; and the paper's qualitative
   figures use the **final_eval** dumps (train.py:3951–4042), which are taken at
   the end regardless of this selection. Residual risk: negligible, but if any
   figure were switched to best-val dumps, re-select on the cluster.

**Checkpoint/early-stopping selection for the reported numbers was NOT affected**:
all reported metrics come from `final_eval` at end of training, not from any
PSNR-based selection.

---

## Part 2 — zero-shot baseline audit

Reported zero-shot (E1z): 19.71 dB / LPIPS 0.155 / Bleed-W 1.09 at 128².
After the clamp correction alone this is already **23.07 dB** (bias −3.36 dB,
largest of any arm). The rest of this section explains the remaining gap to
"well above 30 dB".

### Key architectural finding first

The zero-shot arm (TASK=7 in `run_paper_sweep.sh`) does **not** run the native
Wan 2.1 VAE. It instantiates `AttentionMultiViewVideoVan` with
`independent_views=True` and **`temporal_compression=False`**, which

- skips the pretrained temporal-stride convolutions on encode
  (`_encode_one_view`, DiffSynth `wan_video_vae.py:2663` →
  `_run_down_path(x, None, [0])`), and
- decodes **every frame independently with no feat_cache**
  (`_decode_body`:2914–2958),
- and `evaluate_model`'s forward decodes a **sampled** latent
  (`reparameterize`), not `mu` (opensora wrapper `wan_video_vae.py:780–832`).

This is a legacy per-frame configuration, not "pretrained Wan zero-shot".

### 2(a) Natural ≥480p clip at native resolution

`paper/audit/zeroshot_local_check.py`, run on the local L40S with
`Wan2.1_VAE.pth` (HF `Wan-AI/Wan2.1-T2V-1.3B`) and a real 720p jellyfish clip
(9 frames, stride 3), inputs normalized to [−1,1], metrics via the unified module:

| config | res | PSNR | SSIM | Bleed-W |
|---|---|---:|---:|---:|
| **native chunked Wan (1+4+4, mu)** | 704² | **36.88** | 0.939 | 0.97 |
| native chunked Wan | 256² | 30.45 | 0.908 | 0.93 |
| native chunked Wan | 128² | 28.14 | 0.891 | 0.90 |
| repo E1z path (TC=off, sampled) | 704² | 21.87 | 0.865 | **1.23** |
| repo E1z path (TC=off, sampled) | 256² | 21.98 | 0.843 | 1.07 |
| repo E1z path (TC=off, sampled) | 128² | 21.97 | 0.818 | 0.96 |
| repo class, **TC=on**, sampled | 128² | **28.14** | 0.891 | 0.90 |
| repo class, TC=on, 256² | 256² | 30.45 | 0.908 | 0.93 |

**Yes — the pretrained VAE at native resolution is well above 30 dB (36.9 dB).**
Weights, normalization, and pre/post-processing are correct.

Robustness check on a second, very different clip (Big Buck Bunny, rendered
animation; `zeroshot_bbb_results.json`): native 32.7 / 30.8 / 30.0 dB at
704/256/128 — content moves the native numbers by a couple of dB, as expected —
while the E1z TC=off path lands at **22.1 / 21.6 dB**, flat again. Together with
NeRSemble's 23.07 dB through the same path, the TC=off decode pins PSNR to a
~22–23 dB band *independent of content and resolution*: the finding does not
depend on the choice of clip.

### 2(b) Same clip at 128² and 256²

Native path: 30.45 dB @256², 28.14 dB @128² — resolution alone costs ~8.7 dB from
704²→128² on natural video (small objects lose high-frequency detail the VAE was
trained to reproduce at scale). The **E1z TC=off path is flat at ~22 dB at every
resolution** — its per-frame no-cache decode is the binding constraint, costing
**6.2 dB at 128² and 15.0 dB at 704²** relative to the true Wan path.

The repo model with TC=on matches the native implementation to 0.01 dB →
checkpoint loading (`_remap_wan_keys_for_lora`), zero-init LoRA, and
`fusion_mode="none"` are all exactly identity. The entire anomaly is the
`temporal_compression=False` flag.

### 2(c) NeRSemble clip at 128/256/512 ± matting

**Not runnable on this machine** — the NeRSemble data lives on the Vector cluster.
`zeroshot_local_check.py --nersemble_pt <clip.pt>` implements it; run on the
cluster. What we know without it: corrected E1z on NeRSemble @128² is 23.07 dB vs
21.97 dB for the same TC=off path on the natural clip @128² — i.e. **white-matted
NeRSemble heads are, if anything, slightly easier than natural video** at this
resolution/path. The measured domain gap at 128² is ≈ −1 dB (favoring NeRSemble),
not the +10 dB implied by the old table.

### 2(d) Temporal misalignment check

PSNR of reconstruction frame t vs GT frame t−1 / t / t+1 (natural clip, 128²):

| path | t−1 | **t** | t+1 |
|---|---:|---:|---:|
| native chunked | 23.79 | **28.14** | 23.82 |
| E1z TC=off | 20.49 | **21.97** | 20.36 |

Aligned frame wins by 1.5–4.3 dB on both paths → **no off-by-one frame shift
anywhere in the pipeline.** The Bleed-W > 1 of the zero-shot arm (1.09 reported;
0.96–1.23 reproduced here on the TC=off path only) comes from independent
per-frame reconstruction noise inflating temporal differences, not misalignment —
the native path has Bleed-W < 1 on the same clip.

### 2(e) Wiring checklist (yes/no + evidence)

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | [−1,1] normalization on encoder input AND metric GT | **yes** | loader yields [−1,1] (`vae_target_range`); model consumes it directly; metrics remap via `x01=(x+1)/2` at train.py:1372–1384 (post-fix; pre-fix headline PSNR/SSIM got [−1,1] — that was the bug) |
| 2 | RGB channel order throughout | **yes** | preprocessing is PIL-RGB end-to-end; the one cv2 decode path converts explicitly (`cv2.COLOR_BGR2RGB`, `data/processing/preprocess_nersemble.py:878`); eval/dump path never touches cv2 |
| 3 | 1+4+4 chunking + cache reset between clips | **yes for TC-on** (`_encode_with_stats` and `decode` both call `clear_cache()` per clip, DiffSynth wan_video_vae.py:1515/1576); **N/A for TC-off arms** — no cache exists, every frame decoded independently (that IS the zero-shot problem) |
| 4 | Wan latent mean/std applied exactly once | **applied ZERO times, symmetrically** — the wrapper passes `scale=[zeros, ones]` (opensora wrapper :780–832), so encode scales by identity and decode un-scales by the same identity. Benign for reconstruction (verified: TC-on repo path == native path which uses real constants, 28.14 vs 28.14); would matter only if latents were consumed externally |
| 5 | LoRA / view-fusion absent or identity in zero-shot | **yes** — LoRA zero-init, `fusion_mode="none"`; empirically identity: repo TC-on reproduces native to 0.01 dB. **But TC=False is NOT identity** (−6.2 dB @128²) |
| 6 | Resize antialiasing; same interpolation for GT and input | **same tensor for input and GT — yes; antialiasing — NO.** Eval calls `downsample_video_tensor` (train.py:501–518, `F.interpolate(mode="bilinear")` with no `antialias=True`) before both the model and the metrics (evaluate_model :1354–1366). For native 128/256 buckets it is a spatial no-op; when it does resize, GT and input are still the same post-resize tensor, so metrics are self-consistent — the missing antialias only degrades GT fidelity vs the source, it cannot create a model-vs-GT mismatch. Offline preprocessing itself is PIL center-crop + resize |
| 7 | Same dtype zero-shot vs finetuned | **yes, and immaterial** — both bf16; measured bf16 vs fp32 difference on native @128²: 28.141 vs 28.158 dB (0.02 dB) |
| 8 | Dump is decoder output, not intermediate | **yes** — dumps built from `x01` of the final clamped decode (train.py:1432–1442); final_eval dumps at :3951–4042 |
| 9 | (extra) deterministic vs sampled latent | sampled in eval (`reparameterize` has no deterministic/`self.training` branch, DiffSynth wan_video_vae.py:2891–2894; wrapper :805) — flagged as suspicious by the independent wiring audit, but **measured to be a no-op** for the pretrained model: mu vs sampled identical to 0.001 dB (posterior std is tiny). Ruled out as a PSNR-floor contributor |
| 10 | (extra) output clamp | decode output clamped to [−1,1] before metrics — correct, matches GT range |

An independent subagent audit of the same checklist (TASK=7/49 exact args,
effective model kwargs, frame-count accounting 9-in/9-out on the TC-off path,
deterministic linspace frame sampling with no random eval offset, LoRA rank 32
zero-init = identity) reached the same verdicts on all items; its two flagged
suspects — stochastic decode and TC=False — are respectively ruled out
empirically (item 9) and confirmed as the dominant cause (see 2(a)/2(b)). It
also confirms Bleed-W's "chunk" labels are architecturally meaningless under
TC=False, consistent with the per-frame-noise explanation in 2(d).

### 2(f) Verdict

**The 19.71 dB zero-shot number is neither a pure domain gap nor a metric-GT
mismatch — it is two stacked artifacts on top of a modest real resolution cost.**
First, the clamp bug suppressed it by 3.36 dB (true value 23.07 dB). Second, the
zero-shot arm never ran the actual pretrained Wan VAE: `temporal_compression=False`
skips the pretrained temporal convolutions and decodes frames independently
without the causal cache, which we measured to cost 6.2 dB at 128² (and 15 dB at
native resolution) on in-domain natural video. The genuine resolution cost of
128² for the native VAE is real but moderate (36.9 → 28.1 dB on natural video),
and the NeRSemble domain itself appears roughly as easy as natural video at 128²
(23.07 vs 21.97 through the identical path). No wiring faults were found:
normalization, channel order, chunking/cache, dtype, LoRA-identity, and frame
alignment all check out, and the Bleed-W > 1 is fully explained by per-frame
decode noise. **Affected arms: only the zero-shot row E1z** (and the pending
re-runs 5845272/73, which use the same TASK=7/49 config and will therefore land
near 23 dB again — correct numbers for the *wrong* baseline). Finetuned arms are
unaffected: each trains through its own declared architecture, and TC-off vs
TC-on arms are compared as architecture ablations, which remains valid. For the
paper, either (i) relabel the current row as "pretrained weights through the
per-frame (TC-off) wrapper", or (ii) add a true zero-shot row by running TASK=7
with `temporal_compression=True` on the cluster — expected ≈28–30 dB at 128²
based on the local evidence, which also makes the fine-tuning gains story cleaner
(fine-tuning closes a real but smaller gap).

---

## Pending jobs (5845272–5845277)

As of this audit (2026-10-02), **no new wandb runs exist after 2026-09-20** —
all six jobs are still queued. Per instruction they were not touched. When they
land:

- 5845272/73 (zero-shot 128/256, fixed eval): paste `final_eval/val/psnr` directly
  into the corrected table (no correction needed — eval is fixed); expect ≈23 dB
  @128². Remember these are TC-off numbers (see verdict above).
- 5845274–77 (seed-43 E1c/E1d/E11a/combo, SAVE_CKPT=True): add as seed-2 columns;
  `paper/audit/build_corrected_table.py` can be re-run to pick them up, or read
  `final_eval/val/psnr` from wandb directly (already unbiased).
- Also retrieve from the cluster: `final_eval_dump_*.pt` for corrected SSIM
  (all arms), E5b jsonl, and the per-frame PSNRs missing from
  `clean_peraxis_metrics.json`.

## Rerun / refill list (consolidated)

> **Actionable version with exact sbatch/python calls and a CVPR-completeness
> checklist (rFVD, error bars, rate accounting, data statement, ...):
> `paper/rerun_plan_2026-10-02.md`** — written to be executed on the cluster.

| item | where | why |
|---|---|---|
| E5b corrected PSNR | cluster jsonl/dump | wandb summary lacks per-view fields |
| corrected SSIM, all arms | cluster dumps + `metrics_unified.py` | per-view SSIM never logged |
| per-frame PSNR, arms outside peraxis json | cluster jsonl | not logged to wandb |
| NeRSemble 128/256/512 ± matting zero-shot | cluster, `zeroshot_local_check.py --nersemble_pt` | no local data |
| (recommended) true zero-shot TC-on arm | cluster, TASK=7 with TC=True | current E1z is not native Wan |
| data-scale single-seq figure point | cluster | marked † in figure |
