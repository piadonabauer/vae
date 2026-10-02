# Eval / architecture checks (2026-10-02)

## 1. PSNR mean vs per-view — BUG, not aggregation convention

**Cause:** with `vae_target_range="[-1,1]"`, `evaluate()` called `compute_metrics(x, x_rec)` on tensors still in `[-1,1]`. Inside `compute_metrics`, `.clamp(0,1)` crushed the negative half of the signal. Meanwhile `psnr_per_view` was computed on correctly remapped `x01 = (x+1)/2`.

**Evidence (E1a dump):**

| convention | PSNR |
|---|---:|
| reported `psnr_mean` (jsonl) | 34.20 |
| simulate wrong clamp on dump | 34.22 |
| correct mean over clip×view on dump | 35.92 |
| reported / dump `psnr_per_view` | ≈35.89 / 35.92 |
| PSNR(pooled MSE) vs mean-of-PSNRs | only −0.28 dB (not the 1.7 gap) |

So the ~1.7 dB gap is **not** pooled-MSE vs mean-of-PSNRs. Per-view numbers were the honest ones; table `psnr_mean` was biased low.

**Bias is not constant** (so deltas from reported means are slightly distorted):

| arm | jsonl mean | dump-correct mean | bias | v0−v1 |
|---|---:|---:|---:|---:|
| E1a | 34.20 | 35.92 | −1.72 | +0.01 |
| E1b | 31.37 | 33.21 | −1.84 | +0.04 |
| E1c | 30.75 | 33.00 | −2.25 | **+1.35** |
| E1d | 25.96 | 28.35 | −2.39 | −0.49 |
| E11a | 27.52 | 29.76 | −2.24 | **+1.93** |

**Fix:** `train.py` now runs `compute_metrics` on `x01`/`xr01` (same path as per-view). Seed-2 / zeroshot jobs that have not started yet will pick this up. Clean-wave table numbers should be **refilled from dumps** (or from `psnr_per_view` mean), not left as biased `psnr_mean`.

**Paper convention going forward:** mean of per-(clip, view) PSNRs on `[0,1]` — matches corrected `psnr_mean` and mean of `psnr_per_view`.

## 2. Default fusion vs methods Eq. — methods are right; config name is misleading

Default arm (`fusion_mode=cross_attention`, E1c/E1d):

1. `ViewAttention`: **does** attend over all `V·N` spatial tokens per time step (`s = v * n` in forward). Zero-init output proj. This matches Eq.~\ref{eq:fusion_attn}.
2. Then **binary tree merge** (`tree_resblocks`) → Eq.~\ref{eq:tree_merge}.

Ablation E2b (`fusion_mode=self_attention`):

1. `JointViewAttention`: also joint `V·N` MHA (LayerNorm + `nn.MultiheadAttention`).
2. Then **flat** channel-concat + two ResBlocks — **no tree merge**.

So: methods text describes the **default** attention correctly. The confusion is the config string `cross_attention` (really joint view-token self-attn + tree merge). E2b is “similar attention family, different implementation + flat merge,” not “the thing Eq. fusion_attn describes.”

For V=2, tree merge is a single pairwise 2C→C merge, so E2b’s collapse to ~25.5 dB is unlikely to be “missing the tree” alone — more likely the JointViewAttention / init path.

## 3. View asymmetry — real, not a metric artifact

Confirmed on dumps (correct `[0,1]` PSNR):

- Per-view refs E1a/E1b: symmetric (~0 dB).
- Fused TC-off E1c: **+1.35 dB** (view 0 better).
- Widen E11a: **+1.93 dB**.
- Fused TC-on E1d: mild reverse (−0.5 dB).

Not explained by the PSNR bug. Plausible sources: per-view LoRA init, camera order in the 2-view dataset, or attention/merge treating views asymmetrically. Worth a short note in the paper or a swap-views sanity check; not blocking tables if you report both views.

## 4. Pending jobs (still Priority as of check)

| job | arm | est. start (EDT) |
|---|---|---|
| 5845272/73 | zeroshot 128/256 | ~16:50 |
| 5845274–77 | seed-2 E1c/E1d/E11a/combo | ~17:00–17:30 |
| 5844331/32 | E6d/E6e | ~18:10 |

Eval fix above is in the script those jobs will bake at start.
