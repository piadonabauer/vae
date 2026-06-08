# VAE decoder profiling — findings summary

CUDA-synchronized block timers (`ProfileTimer` / `profile_timing=True`) on the Wan multiview cross-attention + LoRA path. One profiled optimizer step per run (typically `global_step=50`).

---

## Where the results live

| Location | Contents |
|----------|----------|
| `outputs/view_profile_benchmark/` | **May 24** view-count sweep (2 / 4 / 8 views). `summary.json` + `views_*v_profile_step50.{txt,json}` |
| `outputs/benchmark_no_compile/` | **May 29** detailed decoder breakdown (best source for upsample stages) |
| `outputs/benchmark_compile_*/` | May 29 compile benchmark profiles — **internal decoder blocks missing** when `optimization=True` |
| `outputs/decode_profile_all_people_1expr/` | **May 29** `all_people_one_expression` run — top-level timings only (compile on) |
| `outputs/views_{2,4,8}v/` | Per-run experiment dirs from view benchmark (copies also in `view_profile_benchmark/`) |
| `slurm_logs/view_profile_*.out` | May 24 view benchmark logs |
| `slurm_logs/compile_bench_3804478_*.out` | May 29 compile + profile logs |

Re-read a report:

```bash
cat outputs/view_profile_benchmark/summary.json
cat outputs/benchmark_no_compile/profile_timing_step50.txt
```

---

## Profile runs at a glance

| Run | Date | Job | Data preset | Views | Resolution | Batch | Clip / dataset | Internal decode detail? |
|-----|------|-----|-------------|-------|------------|-------|----------------|-------------------------|
| **View benchmark** | 2026-05-24 | 3738627 (+ earlier attempts) | `single_sequence` | **2, 4, 8** (one clip each) | **128×128**, T=9 | 1 | One `.pt` per view count (see below) | Yes — coarse blocks |
| **Compile bench (no_compile)** | 2026-05-29 | 3804478 task 1 | `single_sequence` | **8** (auto-detected) | **128×128**, T=9 | 1 | `8-frames/128-res/p017/EMO-1-shout+laugh/frames.pt` | **Yes — full DECODER DETAIL** |
| **all_people profile** | 2026-05-29 | (manual) | `all_people_one_expression` | 8 (typical in 8-frames tree) | **128×128**, T=9 | **64** | ~390 train clips, EMO-1-shout+laugh | **No** — compile swallowed timers |
| Compile bench (compiled) | 2026-05-29 | 3804478 tasks 2–5 | `single_sequence` | 8 | 128×128 | 1 | same clip | No — `optimization=True` |

### View benchmark clips (May 24)

| Views | Data path |
|-------|-----------|
| 2 | `processed/128-res/p017/EMO-2-surprise+fear/EMO-2-surprise+fear.pt` |
| 4 | `processed/4-frames/128-res/p017/EMO-1-shout+laugh/frames.pt` |
| 8 | `processed/8-frames/128-res/p017/EMO-1-shout+laugh/frames.pt` |

Script: `vae/run_view_profile_benchmark.sh`.

### Shared model / training settings (all meaningful profiles)

- Cross-view encoder, `fusion_mode=cross_attention`, LoRA after bottleneck/decoder
- `freeze_temporal=True`, `train_spatial=True`
- BF16, single GPU, `deterministic=True`
- No discriminator

---

## Headline: decoder is the forward bottleneck

Across all runs with working internal timers:

1. **`decode.temporal_loop` ≈ 91–93% of `train.forward`** — not encoder, not view-fusion attention.
2. **Inside the temporal loop, `decode.body.upsamples` ≈ 69–70% of forward** — upsample + residual stack dominates.
3. **`decode.body.middle` ≈ 17–18% of forward** — bottleneck residual + one attention block.
4. **Encoder (all stages) ≈ 6–8% of forward.**
5. **View fusion attention (`attention.view.*`) &lt; 2 ms** — negligible vs decode.
6. **`decode.temporal.cat` (frame stitching) ≈ 0.3% of decode** — not the problem.
7. **Backward ≈ 55–65% of total step** — separate from forward decode cost; scales similarly with views.

Decode cost scales **~linearly with view count** (each view decoded separately with view embedding):

| Views | Forward (ms) | Decode loop total (ms) | Step total (ms) |
|-------|-------------|------------------------|-----------------|
| 2 | 304 | 279 | 774 |
| 4 | 581 | 535 | 1478 |
| 8 | 1175 | 1084 | 2971 |
| 8 (May 29 detailed) | 1280 | 1189 | 3101 |

Source: `view_profile_benchmark/summary.json` (2/4/8) and `benchmark_no_compile/profile_timing_step50.txt` (8).

---

## Detailed decoder breakdown (May 29 — primary rerun)

**Run:** `outputs/benchmark_no_compile/` · step 50 · **8 views** · **128×128** · **single_sequence** (one 8-view clip) · batch 1 · **no torch.compile**

### Top-level step (ms)

| Block | ms | % of step |
|-------|-----|-----------|
| train.backward | 1746 | 56.3% |
| train.forward | 1280 | 41.3% |
| train.optimizer | 48 | 1.6% |
| train.loss | 27 | 0.9% |

### Forward split

| Block | % of forward |
|-------|----------------|
| decode.temporal_loop | **92.9%** |
| encode.downsample_all_views | 5.5% |
| encode.fusion.tree_merge | 0.8% |
| everything else (incl. cross-attn) | &lt; 1% |

### Inside decode (`decode.all_views` = 1193 ms)

| Block | total ms | % of decode |
|-------|----------|-------------|
| decode.temporal.decoder (72 calls) | 1181 | **99.0%** |
| decode.temporal.cat | 4 | 0.3% |
| Per-view decode (8×) | ~146–154 each | ~12.5–12.9% each (even split) |

**72 decoder calls** = 8 views × 9 latent time steps (one `Decoder3d` forward per latent frame per view).

### Inside `Decoder3d` body (aggregated over 72 calls)

| Block | total ms | % of forward |
|-------|----------|----------------|
| decode.body.upsamples | 896 | **70.0%** |
| decode.body.middle | 223 | 17.4% |
| decode.body.head | 34 | 2.6% |
| decode.body.conv1 | 22 | 1.7% |

### Upsample stages (L0 = coarsest latent grid → L3 finest before head)

Each stage has 3 residual blocks + resample (except L3: resblocks only).

| Stage | Component | total ms | % of decode |
|-------|-----------|----------|-------------|
| L0 | resblock | 191 | 16.0% |
| L0 | resample | 32 | 2.7% |
| L1 | resblock | 208 | 17.5% |
| L1 | resample | 32 | 2.7% |
| L2 | resblock | 188 | 15.8% |
| L2 | resample | 32 | 2.7% |
| L3 | resblock | 189 | 15.8% |

**Takeaway:** Upsample **residual blocks** (~65% of decode) matter more than **resample ops** (~8%). Stages L0–L3 are roughly balanced; no single stage dominates.

### Middle bottleneck

| Block | total ms | % of decode |
|-------|----------|-------------|
| decode.body.middle.resblocks | 218 | 18.3% |
| decode.body.middle.attn | (inside middle total 223 ms) | small fraction |

---

## Coarse profiles (May 24 view benchmark)

Same resolution (**128×128**, T=9), **single_sequence**, batch 1. No per-stage upsample split (instrumentation added later).

**2 views** — `% of forward`:

- decode.temporal_loop: 91.6%
- decode.body.upsamples: 68.6%
- decode.body.middle: 17.5%

**4 views:** 92.0% / 68.8% / 17.6%  
**8 views:** 92.3% / 69.1% / 17.6%

Full reports: `outputs/view_profile_benchmark/views_{2,4,8}v_profile_step50.txt`.

---

## Runs that did *not* capture decoder detail

### `decode_profile_all_people_1expr` (May 29)

- **Preset:** `all_people_one_expression` (EMO-1-shout+laugh, ~390 participants)
- **Resolution:** 128×128, 8-frames tree
- **Batch:** 64
- **`optimization=True`** (`reduce-overhead` compile + CUDA graphs)

Profile file only has `train.forward` / `train.backward` / `train.loss` / `train.optimizer`. **No `decode.*` blocks** — torch.compile fuses/skips the Python-level `ProfileTimer` hooks inside the VAE.

Step 50 totals: forward 303 ms, backward 700 ms, step ~1076 ms (batch 64, real dataloader — not comparable to batch-1 view benchmarks).

### Compile benchmark tasks 2–5 (May 29)

Same issue: internal decode blocks empty; only top-level step timings usable for compile speed comparison (`outputs/compile_benchmark/summary.json`).

---

## How to reproduce / extend

**View scaling (2 / 4 / 8 views, single clip):**

```bash
sbatch vae/run_view_profile_benchmark.sh
```

**Detailed decoder stages (must disable compile):**

```bash
cd vae/Open-Sora
python scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py \
  --experiment_name decode_profile_test \
  --data_preset single_sequence \
  --batch_size 1 \
  --profile_timing True \
  --profile_timing_step 50 \
  --optimization False \
  --epochs 55 \
  --wandb False
```

For **`all_people_one_expression`** with internal decode detail: use `--optimization False`, `--batch_size 1` (or accept top-level timings only with compile on).

Config knobs: `profile_timing`, `profile_timing_step` in `configs/vae/train/wan_multiview_finetune.py`.

---

## Likely optimization levers (from profiling)

1. **Reduce decoder invocations** — temporal loop calls full `Decoder3d` per latent frame × per view; largest structural cost.
2. **Upsample residual blocks** — main target inside the decoder (~70% of forward).
3. **View count** — linear multiplier on decode; 8 views ≈ 4× cost of 2 views.
4. **Not worth chasing first:** view fusion attention, `torch.cat` between frames, encode path.
5. **When profiling:** always `--optimization False`; compile hides internal blocks.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-24 | Initial view benchmark + coarse decode blocks |
| 2026-05-29 | Added `DECODER DETAIL` (per-view, temporal.decoder/cat, ups L0–L3); captured in `benchmark_no_compile` |
| 2026-05-29 | `decode_profile_all_people_1expr` — real dataset step timing; decoder detail blocked by compile |
