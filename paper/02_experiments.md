# Clean Experiments to (Re-)Run for Reportable Numbers

Goal: one fixed protocol, every number in the paper comes from this protocol. Your historical
sweeps mixed batch sizes, epochs, data presets and eval cadences — do NOT cite those directly;
rerun the arms below and report from the standardized eval.

**All old numbers (presentation + historical sweeps) are considered unreliable — every value
in the paper comes from the runs below.** The April presentation still earns its keep as a
set of *predictions*: fusion modes ≈ equivalent (E2), embeddings ≈ per-view LoRA (E3), no disc
or 4D disc best / 3D disc hurts colors (E7), ranks 32-128 similar (E10). Where a rerun
confirms a prediction you can report it with confidence; where it contradicts, the rerun wins.
Priority stays: E1 (the 2x2) and E5 (full unfreeze) first — they are the runs with no prior
answer at all — then E11 (latent width, the positive capacity test), then E4, then the
confirmation reruns E2/E3/E7/E10.

**Why the old numbers are DEFINITELY unreliable (found Aug 2026, fixed on this branch):**
every historical run with `use_lora=True` silently trained with a partially random-init
backbone. `_enable_lora()` renames the wrapped convs to `*.base_conv.weight`, but the raw
Wan checkpoint has plain conv names, so `load_state_dict(strict=False)` dropped ~90 keys —
the pretrained conv weights of `encoder.middle`, `encoder.head`, and the ENTIRE decoder
never loaded (visible in old logs: "194 ckpt keys; 263 missing, 90 unexpected"). Those convs
then sat frozen at random init, and the rank-32 LoRA paths had to compensate. The loader now
remaps the keys and hard-fails if any pretrained encoder/decoder key goes unloaded. Expect
the rerun numbers (and possibly some qualitative conclusions) to differ from the old sweeps.

## Fixed protocol (applies to every run unless the experiment varies it)

| Item | Value |
|---|---|
| Config | `Open-Sora/configs/vae/train/wan_multiview_finetune.py` + CLI overrides only |
| Data | NeRSemble `128-res`, `data_preset=all_people_one_expression` (EMO-1-shout+laugh) |
| Views / frames | V=2, **T=9 everywhere, no exceptions** (V=4/8 only in the view-count experiment). Rationale: T=9 -> latent T'=3 = frame 0 + two 4-frame chunks, so both bleeding *within* chunks and artifacts *across* a chunk boundary are measurable; T=5 (used in some old scripts) has no chunk boundary at all |
| Val split | the 10 `_val_participants` (held-out identities), full val set (`eval_num_samples=0`) |
| Optimizer | AdamW, lr 5e-4 constant (no scheduler), betas (0.9, 0.98), wd 0, grad_clip 2.0 |
| Batch | effective 64 (e.g. bs16 x accum4 at 128px; keep effective batch fixed if OOM forces bs down) |
| Duration | 170 epochs (same as bleed sweep) — fix ONE number and never vary it |
| Precision / EMA / seed | bf16, ema_decay 0.9999 (eval with EMA), seed 42 |
| Loss | perceptual 1.5, kl 1e-6, view_consistency 0, discriminator none (except E7). NOTE: the presentation found lower perceptual/KL better — if your best-config numbers used e.g. kl 1e-7, adopt THAT here and state it once; whatever you pick, freeze it for every run |
| LoRA | use_lora=True, use_lora_after=True, use_lora_before=False, rank 32, viewwise decoder LoRA on |
| Eval | `full_eval_every=50` updates (~10 epochs; matches `run_paper_sweep.sh`); report **best val** and **final-EMA val** |
| Logging | wandb REQUIRED for every run (the sweep script refuses to start without a wandb login). Scalars: early schedule [1,2,3,5,8,12,20,30,50,75,100,150,200], then every 20 updates. Reconstruction grids: same early schedule, then geometric backoff (x1.5, cap 2000). Fixed-sequence per-frame eval every 10 epochs |
| Qualitative vis | Always the same people, in every run and arm: samples come from sorted dataset order, not the shuffled batch. Train grids = first 3 distinct train participants (p017 + next two non-val); val grids = first 3 val clips = **p018, p030, p038**; fixed-seq eval = first train/val file with the shared sequence name |

**Two-stage discipline (applies to EVERY arm):**
- Stage 1 — overfit gate: run the arm on `single_sequence` first (cheap: 1 sample). PASS =
  near-perfect reconstruction (train PSNR >= 35). FAIL = implementation/architecture problem;
  do not launch Stage 2. (`OVERFIT=1` in `run_paper_sweep.sh` switches any arm to this mode.)
  If an arm plateaus just below 35 but reconstructions look pixel-perfect, judge by eye —
  the gate catches breakage, it is not a benchmark.
- Stage 2 — generalization: the actual protocol run on `all_people_one_expression` with the
  held-out identities. All reported numbers come from Stage 2; Stage 1 results appear in the
  paper only as the overfit-vs-generalize contrast (bleeding/ghosting exist only in Stage 2).

**Run length & comparability (non-negotiable):**
- The unit of training budget is the **optimizer update**, never the epoch number by itself
  and never wall clock. With effective batch pinned at 64 (bs x accum, the OOM ladder only
  reshuffles the product) and the same dataset, *170 epochs = the same number of updates and
  the same samples seen for every arm* — that is the only reason epochs are usable as a label.
- Eval points align automatically: `full_eval_every=50` is counted in optimizer updates
  (train.py checks `actual_update_step % full_eval_every`), so every arm is evaluated at
  update 50, 100, 150, ... (~17 val evals over the 850-update budget: 358 samples /
  effective batch 64 = 5 updates per epoch, x 170 epochs). Comparisons are only valid at
  equal `actual_update_step`; the
  jsonl records carry it, `collect_results.py` exports it — check the column before citing
  two numbers side by side.
- Stage-1 gate stopping rule: the gate run stops itself once epoch-mean train PSNR >= 35
  holds for 3 consecutive epochs (`stop_at_train_psnr`), hard cap 2000 epochs. If the cap is
  hit below 35, the arm FAILED the gate. Gates are pass/fail — never compare gate PSNRs
  between arms (they stop at different step counts by construction).
- Stage-2 stopping rule: there is NONE, not even a divergence guard — every arm runs the
  full fixed budget, even if it plateaus early or never reaches PSNR 30 (a TC arm plateauing
  at e.g. 27 dB IS the result; the fixed budget is what makes that a statement about the
  model and not about training time). The old train-PSNR divergence guard is neutralized in
  the sweep (`train_psnr_guard_threshold 0`) for two reasons: a slow-learning generalization
  arm must never be killed by a heuristic, and the guard's activation epoch depended on the
  micro-batch size (steps-per-epoch), so arms on different OOM-ladder rungs would have had
  different guard behavior — a protocol violation. If a run truly diverges you will see it
  on wandb; let it finish or relaunch it, but never let a heuristic stop it.
- If compute forces trimming the budget, change `TRAIN_EPOCHS` for ALL arms and rerun the
  affected comparisons — never shorten a single arm.

**Wall clock / cluster reality:** jobs request 18h (<20h schedules much faster than 40h).
`run_paper_sweep.sh` chains automatically: each job queues an `afterany` successor before
training, resumes from the newest epoch checkpoint, and writes `outputs/<run>.DONE` when the
arm completes so leftover successors no-op. Up to 4 x 18h per arm by default (`CHAIN_LEFT=3`).
Nothing about chaining affects comparability — a resumed run continues the same update count.

**Initialization (default = staged warm-start):** joint arms (E1-d, E5b/c) initialize from
the converged per-view TC checkpoint (E1-b) with `reinit_view_attention_after_load`; set
`INIT_CKPT=<path to E1-b ckpt>` when launching. Training from Wan weights only (new modules
zero-init) is the ablation E7b. This creates a dependency: **E1-b must finish before the
joint arms start.**

**Report for every run:** val PSNR / SSIM / MSE / LPIPS (mean ± std over clips),
bleed_ratio_within, bleed_ratio_across, cross-view recon cosine similarity (+ GT reference
level), per-frame-index PSNR profile (at least frame0 vs mean of frames 1-8), per-frame-pair
|Δframe| profiles for GT and rec (the raw bleeding curves for figure F4). Log all to wandb
with a fixed naming scheme, e.g. `paper/E1a_perview_tcF`. LPIPS is now computed directly in
`full_eval`/`final_eval` (VGG backbone, fp32, per-clip mean over views and frames; logged as
`eval/val/lpips_mean`). `scripts/vae/lpips_from_dump.py` remains as an offline fallback for
old dumps.

**Clip dumps (raw material for all qualitative figures):** every full/final eval saves the
evaluated clips (GT + reconstruction, uint8, ~20 MB for the 10-clip val set) into the run dir:
`latest_full_eval_dump.pt` (rolling), `best_val_eval_dump.pt` (kept at the best-val step —
matches the "best val" table row), `final_eval_dump_{train,val}.pt`. F1 insets, the F5
qualitative grid, difference maps, LPIPS, and the supplement video are all offline CPU
scripts over these files — no checkpoint re-decoding needed.

Note: all of these are wired into `full_eval`/`final_eval` on this branch (train.py computes
bleed ratios, cross-view similarity rec+gt, per-frame PSNR profile, and LPIPS; everything
lands in `outputs/<run>/eval_metrics.jsonl` and wandb).

---

## E1 — The rate–quality curve: view fusion x temporal compression  [MUST HAVE, RQ1+RQ3]

Framing: these are NOT baselines to beat — per-view models use V× the latent rate and produce
no joint code, so they are *reference points* on the rate–quality curve. Report each run with
its compression ratio so the headline figure (quality vs rate) can be drawn directly.

| ID | use_crossview_encoder | temporal_compression | Rate (V=2, T=9) | Role |
|---|---|---|---|---|
| E1-0 | — | — | 12x/view | zero-shot reference (eval only, no finetuning) |
| E1-a | False (per-view, LoRA ft) | False | 12x/view | finetuned reference / quality ceiling |
| E1-b | False (per-view, LoRA ft) | True | 48x/view | temporal axis alone (RQ2 anchor) |
| E1-c | True (fused latent) | False | 24x | view axis alone (RQ1) |
| E1-d | True (fused latent) | True | 96x | **joint — the headline point (RQ3)** |

Core claim = quality drop E1-a->E1-d is super-additive vs the E1-b and E1-c drops.
5 runs (E1-0 is eval-only). Optional 6th: single-view Wan (V=1) TC=True — isolates whether
bleeding needs multi-view at all. [Recompute the rate numbers with the exact formula
rate = (V·3·T·H·W)/(V'·16·T'·(H/8)·(W/8)) and your T'=3.]

## E2 — Fusion mechanism ablation (at TC=False, then best mode also at TC=True)  [MUST HAVE]

Fixed: everything per protocol; vary `fusion_mode`:
E2-a `cross_attention` (= E1-c, reuse) | E2-b `self_attention` | E2-c `conv3d` | E2-d `conv4d`.
Then E2-e: best mode with TC=True (sanity that fusion ranking holds under TC). 4 new runs.

## E3 — View-conditioned decoding (how does one latent become two views?)  [MUST HAVE]

Fixed: crossview encoder, TC=False. Vary decoder view conditioning:
| ID | view embedding | viewwise decoder LoRA | full_finetune_decoder |
|---|---|---|---|
| E3-a | on | off | off |
| E3-b | off | on | off |
| E3-c | on | on | off (= E1-c if identical, reuse) |
| E3-d | off | off | off — **must ghost**: negative control |
| E3-e | off | on | on (upper bound: full decoder finetune) |

Report cross-view similarity prominently. ~4 new runs.
Optional E3-f (negative control for the paper's history paragraph): the legacy latent-averaging
path (`use_crossview_encoder=False`, ViewCompressor + embeddings) — one run, expected to ghost;
gives you a real number for the "averaging destroys view identity" claim instead of an old log.

## E4 — Temporal interventions (TC=True, per-view OR fused — pick E1-d config)  [MUST HAVE]

One arm per flag, all else fixed (this is your `run_tc_quality_sweep` + bleed-sweep arms, redone
under the fixed protocol):
E4-a baseline (= E1-d, reuse) | E4-b `use_noncausal_decode` — frame as the **oracle upper
bound** (removes chunking and its artifacts, at the cost of causal/streaming decode; the gap
E4-b vs E4-a isolates the chunk mechanism's damage) | E4-c `use_temporal_reflection_pad`
| E4-d `use_temporal_side_channel` (dim 4) | E4-e noncausal + `use_decoder_temporal_attention`
| E4-f `use_learned_cache_update` | E4-g `use_subframe_position_embedding`
| E4-h `temporal_diff_loss_weight=2.0` | E4-i best combo of whatever wins.
Primary metric: bleed_ratio_within + per-frame PSNR profile. ~8 runs.

## E5 — Encoder freezing (the frozen-bottleneck confound)  [MUST HAVE — supports the capacity claim]

Fixed: E1-d config. Vary:
E5-a frozen encoder (default, reuse E1-d) | E5-b `freeze_temporal=False, train_spatial=True`
(full encoder unfreeze) | E5-c unfreeze + `full_finetune_decoder=True` (nothing frozen anywhere).
If E5-c still bleeds/ghosts → capacity, not adaptation. 2 new runs.

## E6 — Number of views  [SHOULD HAVE]

E6-a V=2 (reuse E1-c/d) | E6-b V=4, TC=False | E6-c V=4, TC=True | E6-d V=8, TC=False
(only if the 8-view .pt data is preprocessed — the 8-camera preprocessing script exists).
Prediction from the capacity argument: fused-view quality degrades monotonically with V even
at TC=False. 2-3 runs.

## E6b — Merge topology: binary tree vs flat V→1  [SHOULD HAVE at V=4]

Fixed: E6-b config (V=4, cross-attention, TC=False). Compare the hierarchical binary tree
merge (V-1 pairwise merge blocks, default) against a single flat merge (concat all V at once
-> ResBlock(V·C→C)). Justifies the tree-merge design choice — currently it is asserted, not
ablated. 1-2 runs (flat merge needs a small code addition).

## E7 — Discriminator layouts  [SKIP unless E1-E4 suggest a GAN term helps]

The April sweep suggested no discriminator (or 4D) is best and 3D hurts colors — and a second
opinion concurs the paper's story doesn't need GAN ablations. Only run if Tier-1 results look
GAN-limited: E7-a none (reuse) | E7-b `Train` (3D PatchGAN, flatten views) |
E7-c `TrainMultiview4D` | E7-d `TrainMultiviewStack`. 3 runs. Same for LR/KL/EMA sweeps:
exploratory, keep out of the paper.
IMPORTANT if you do run these: use the `paper-snapshot-aug2026` discriminator code — it fixes
in-place LeakyReLU activations that can corrupt gradients under activation checkpointing. All
pre-fix discriminator results (including the April sweep's) are additionally suspect for this
reason, which is one more argument for the rerun-everything policy.

## E7b — Initialization ablation  [SHOULD HAVE]

DECIDED: staged warm-start is now the **default protocol** (see top of file), so this
ablation is just ONE extra run:
- E7b: the E1-d config trained from Wan-pretrained weights only (no `INIT_CKPT`; all new
  modules zero-init) — compared against E1-d itself, which warm-starts from the E1-b
  checkpoint by default.
Answers "does staged training (temporal first, then view) beat joint training from scratch?".
T=9 like everything else; the old T=5 A/B (`run_tc_all_xattn_fft_init_ab.sh`) is superseded.
1 run.

## E8 — Data scale / generalization axis  [SHOULD HAVE — explains "overfit works"]

Fixed: E1-d config. Vary `data_preset`:
E8-a `single_sequence` | E8-b `one_person` | E8-c `all_people_one_expression` (reuse E1-d)
| E8-d `all_people` (if data ready). Show bleed_ratio_within vs dataset size — the "compression
only breaks when it must generalize" plot. 2-3 runs.

## E9 — Resolution scaling  [NICE TO HAVE]

Best config from E1-E4 at 256² and 512² (batch ladder per your 512 scripts, keep effective
batch). 2 runs. Frames the capacity argument in bits-per-pixel terms across resolutions.

## E11 — Latent width (the POSITIVE capacity test)  [MUST HAVE — supervisor-requested]

The capacity claim is currently supported only by what FAILS. This experiment tests it
directly: widen the latent and see whether joint view+temporal compression recovers.
Fixed: E1-d config (fused, TC on), no warm start (boundary shapes differ from E1-b). Vary
`latent_widen_to`:

| ID | latent channels | rate (V=2, T=9, 128px) | sweep arm |
|---|---|---|---|
| E11-0 | 16 (= E1-d, reuse) | 96x | TASK=4 |
| E11-a | 32 | 48x | TASK=8 |
| E11-b | 64 | 24x | TASK=9 |

Implementation (`latent_widen_to` in the model config): only four boundary convs touch the
latent width (encoder head, the two 1x1 latent convs, decoder conv_in). Pretrained weights
are expanded with zero/identity surgery — new mu rows zero, new logvar rows zero (unit
variance), identity diagonal on the 1x1s, zero decoder input columns — so at step 0 the model
is EXACTLY the pretrained 16-ch VAE. The four boundary convs are unfrozen (the new capacity
must be trainable; ~2M params, still adapter-scale). VAE-side only: retraining the diffusion
model on a wider latent is explicitly out of scope (state this in the paper; VA-VAE says
wider latents are harder to sample from — that trade-off is future work, not our claim).

Predictions: if E11-a/b recover most of the E1-c→E1-d drop, the bottleneck is the rate and
the paper's central claim gets positive evidence (DC-AE/LTX-style "width buys compression"
extended to the view axis). If they do NOT recover, the failure is architectural/optimization
— equally important to know before writing the discussion. 2 runs.

## E10 — LoRA rank / placement  [NICE TO HAVE — appendix]

Rank {8, 32, 128} at E1-c; `use_lora_before` on/off. 3-4 runs. You already have single-sequence
rank sweeps in `wandb_outputs`; redo only if you want val-set numbers in the main paper,
otherwise report as appendix with the overfit caveat.

---

## Budget summary

- MUST HAVE: E1 (5) + E2 (4) + E3 (4) + E4 (8) + E5 (2) + E11 (2) ≈ **25 runs** at
  128px/T=9/V=2 — each ≈ bleed-sweep cost; batch them with your existing OOM-ladder scripts.
- SHOULD HAVE: E6 (2-3) + E6b (1-2) + E8 (3) ≈ 6-8 runs.
- NICE TO HAVE / SKIP-UNLESS: E7 (3) + E9 (2) + E10 (4) ≈ 9 runs.

Practical order (both this plan and the parallel session's tiering converge on it):
1. E1-a (finetuned per-view reference) + E1-c (fused, TC off) + E1-0 (zero-shot eval) — the
   RQ1 rate–quality comparison: how small is the quality gap at half the latent rate?
2. E1-b + E1-d — completes the rate–quality curve and isolates the TC effect.
3. E11 (latent width) — the positive capacity test; can start as soon as E1-d exists to
   compare against (no warm-start dependency).
4. E5 (full unfreeze) — kills the frozen-bottleneck objection.
5. E4 (temporal interventions, with E4-b as oracle upper bound), then E2/E3 confirmations,
   then E6/E6b/E8.
Run Tier 1-2 at the fixed small setting (V=2, 128px, T=9, all_people_one_expression); scale
resolution only for the final main-result run if compute allows.

Before launching: (1) expose the three diagnostics as always-on eval metrics; (2) fix the
run-naming scheme; (3) write a tiny results-collection script that pulls
best-val/final-EMA rows from each run's `eval_metrics.jsonl` into one CSV — that CSV becomes
your paper tables.
