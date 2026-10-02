# Clean retrain metrics — CORRECTED (2026-10-02)

Correction for the eval PSNR/SSIM clamp bug (commit 40295bb): the old table's
`psnr_mean`/`ssim_mean` were computed on [-1,1] tensors clamped to [0,1].
**Convention:** PSNR = mean of per-(clip, view) PSNRs on [0,1] (NOT PSNR of pooled MSE;
the two differ by ~0.3 dB). Corrected PSNR = mean of `final_eval` per-view PSNRs,
which train.py always computed on the correct [0,1] path; exact per-sample refill from
the cluster dumps may shift values by <~0.05 dB (batch-weighting).

- **PSNR old → new**: old = biased table value; new = corrected.
- **SSIM**: old value is biased the same way and **cannot be corrected from logs** —
  recompute from `final_eval_dump_val.pt` on the cluster (column marked `dump`).
- **LPIPS / Bleed-W / XView / per-frame PSNR**: unchanged — these were always computed
  on the correctly remapped tensors (`x01` path in `evaluate_model`; LPIPS consumes
  `x01*2-1` which is the [-1,1] the LPIPS net expects).

| arm | PSNR old | PSNR new | Δbias | SSIM old | SSIM new | LPIPS | Bleed-W | PSNR v0 | PSNR v1 | v2 | v3 | source |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| E0 | 34.30 | **36.02** | +1.72 | 0.9990 | dump | 0.021 | 0.988 | 36.00 | 36.03 | — | — | wandb final_eval |
| E1a | 34.20 | **35.89** | +1.69 | 0.9990 | dump | 0.022 | 0.994 | 35.89 | 35.88 | — | — | wandb final_eval |
| E1b | 31.37 | **33.19** | +1.82 | 0.9981 | dump | 0.037 | 0.947 | 33.20 | 33.17 | — | — | wandb final_eval |
| E1c | 30.75 | **32.74** | +1.99 | 0.9977 | dump | 0.038 | 0.969 | 33.33 | 32.14 | — | — | wandb final_eval |
| E1d | 25.96 | **28.27** | +2.31 | 0.9933 | dump | 0.068 | 0.913 | 28.03 | 28.50 | — | — | wandb final_eval |
| E2b | 25.53 | **28.51** | +2.98 | 0.9926 | dump | 0.041 | 0.972 | 28.70 | 28.31 | — | — | wandb final_eval |
| E2c | 28.99 | **31.00** | +2.01 | 0.9966 | dump | 0.045 | 0.940 | 30.02 | 31.97 | — | — | wandb final_eval |
| E2d | 28.52 | **30.36** | +1.84 | 0.9962 | dump | 0.057 | 0.939 | 30.46 | 30.26 | — | — | wandb final_eval |
| E2e | 26.15 | **28.46** | +2.31 | 0.9935 | dump | 0.068 | 0.902 | 28.08 | 28.85 | — | — | wandb final_eval |
| E3a | 30.24 | **32.28** | +2.04 | 0.9974 | dump | 0.038 | 0.975 | 32.96 | 31.60 | — | — | wandb final_eval |
| E3c | 31.28 | **33.10** | +1.82 | 0.9979 | dump | 0.037 | 0.980 | 34.37 | 31.83 | — | — | wandb final_eval |
| E3d | 31.33 | **33.05** | +1.72 | 0.9980 | dump | 0.037 | 0.989 | 32.42 | 33.68 | — | — | wandb final_eval |
| E3e | 31.08 | **32.83** | +1.75 | 0.9979 | dump | 0.040 | 0.951 | 32.77 | 32.90 | — | — | wandb final_eval |
| E4b | 17.98 | **20.74** | +2.76 | 0.9505 | dump | 0.161 | 0.273 | 20.84 | 20.65 | — | — | wandb final_eval |
| E4h | 26.07 | **28.43** | +2.36 | 0.9935 | dump | 0.071 | 0.903 | 28.22 | 28.64 | — | — | cluster jsonl (via clean_peraxis_metrics.json) |
| E4i | 26.65 | **28.89** | +2.24 | 0.9942 | dump | 0.069 | 0.910 | 28.79 | 29.00 | — | — | wandb final_eval |
| E5b | 26.01 | — | — | 0.9933 | dump | 0.072 | 0.927 | — | — | — | — | no corrected source |
| E5c | 23.91 | **25.83** | +1.92 | 0.9892 | dump | 0.118 | 0.856 | 25.75 | 25.91 | — | — | wandb final_eval |
| E6b | 28.00 | **30.18** | +2.18 | 0.9957 | dump | 0.061 | 0.956 | 29.51 | 29.69 | 32.14 | 29.38 | wandb final_eval |
| E6c | 22.99 | **25.51** | +2.52 | 0.9863 | dump | 0.100 | 0.842 | 24.64 | 25.56 | 26.93 | 24.91 | wandb final_eval |
| E11a | 27.52 | **29.78** | +2.26 | 0.9952 | dump | 0.059 | 0.911 | 30.66 | 28.90 | — | — | wandb final_eval |
| E11b | 28.80 | **30.92** | +2.12 | 0.9965 | dump | 0.050 | 0.930 | 30.96 | 30.89 | — | — | wandb final_eval |
| combo | 29.38 | **31.39** | +2.01 | 0.9970 | dump | 0.050 | 0.946 | 31.50 | 31.27 | — | — | wandb final_eval |
| E10_r8 | 31.22 | **32.97** | +1.75 | 0.9979 | dump | 0.038 | 0.978 | 34.32 | 31.61 | — | — | wandb final_eval |
| E10_r16 | 31.33 | **33.15** | +1.82 | 0.9979 | dump | 0.037 | 0.980 | 34.59 | 31.71 | — | — | wandb final_eval |
| E10_r64 | 31.94 | **33.60** | +1.66 | 0.9983 | dump | 0.034 | 0.973 | 33.08 | 34.12 | — | — | wandb final_eval |
| E10_r128 | 32.24 | **33.83** | +1.59 | 0.9984 | dump | 0.032 | 0.978 | 33.28 | 34.38 | — | — | wandb final_eval |
| E7_p0.5 | 26.61 | **28.88** | +2.27 | 0.9942 | dump | 0.069 | 0.895 | 28.67 | 29.10 | — | — | wandb final_eval |
| E7_p3.0 | 25.71 | **28.06** | +2.35 | 0.9928 | dump | 0.068 | 0.902 | 28.09 | 28.03 | — | — | wandb final_eval |
| E7_kl1e7 | 26.04 | **28.31** | +2.27 | 0.9932 | dump | 0.068 | 0.903 | 27.93 | 28.70 | — | — | wandb final_eval |

## Zero-shot floor (E1z, pretrained Wan, eval only — 2026-09-06 re-eval run)

| split | PSNR old | PSNR new | SSIM old | LPIPS |
|---|---:|---:|---:|---:|
| val | 19.71 | **23.07** (v0 23.06 / v1 23.07) | 0.9707 (biased) | 0.155 |
| train | 19.52 | **23.12** (v0 23.23 / v1 23.01) | 0.9686 (biased) | 0.151 |

The zero-shot bias (−3.4 dB) is much larger than for finetuned arms (−1.7 to −2.4 dB):
the clamp bias grows with reconstruction error mass in the dark half of the range.
Jobs 5845272/73 re-run this with the fixed eval and should land near 23 dB (128px).

## Headline deltas, old vs corrected

| claim | old | corrected | verdict |
|---|---:|---:|---|
| TC cost per-view (E1a−E1b) | +2.83 | +2.70 | ok |
| Fusion cost TC-off (E1a−E1c) | +3.45 | +3.15 | ok |
| Joint drop (E1a−E1d) | +8.24 | +7.62 | ok |
| Per-view TC-on vs fused TC-on (E1b−E1d) | +5.41 | +4.92 | ok |
| Widen 16→32 (E11a−E1d) | +1.56 | +1.52 | ok |
| Widen 32→64 (E11b−E11a) | +1.28 | +1.14 | ok |
| Temp-diff loss @16ch (E4h−E1d) | +0.11 | +0.17 | below 0.5 dB (was already) |
| Diff-loss+cache (E4i−E1d) | +0.69 | +0.63 | ok |
| Temp-diff loss @32ch (combo−E11a) | +1.86 | +1.60 | ok |
| Combo vs baseline (combo−E1d) | +3.42 | +3.12 | ok |
| Unfreeze enc (E5b−E1d) | +0.05 | n/a (needs refill) | — |
| Unfreeze all (E5c−E1d) | -2.05 | -2.43 | ok |
| Self-attn fusion TC-off vs default (E2b−E1c) | -5.22 | -4.23 | ok |
| Self-attn fusion TC-on vs default (E2e−E1d) | +0.19 | +0.20 | below 0.5 dB (was already) |
| Self-attn TC-off vs joint baseline (E2b−E1d) | -0.43 | +0.24 | **SIGN FLIP** |
| Additive prediction vs actual E1d | pred 27.92 vs 25.96 (excess +1.96) | pred 30.04 vs 28.27 (excess +1.77) | super-additivity holds |

### Ranking changes between arms (bias is not constant: +1.59 to +2.98 dB)

- **E1d vs E2b**: old 25.96 vs 25.53 (>) → corrected 28.27 vs 28.51 (<)
- **E2b vs E2e**: old 25.53 vs 26.15 (<) → corrected 28.51 vs 28.46 (>)
- **E2b vs E4h**: old 25.53 vs 26.07 (<) → corrected 28.51 vs 28.43 (>)
- **E2b vs E7_p3.0**: old 25.53 vs 25.71 (<) → corrected 28.51 vs 28.06 (>)
- **E2b vs E7_kl1e7**: old 25.53 vs 26.04 (<) → corrected 28.51 vs 28.31 (>)
- **E3c vs E3d**: old 31.28 vs 31.33 (<) → corrected 33.10 vs 33.05 (>)

Any prose that orders these pairs (e.g. "E2b collapses below even the joint
baseline E1d") must be re-checked against the corrected column.

## Arms that cannot be recomputed locally

- **E6d / E6e (8-view)**: never produced a final_eval (clean-wave TIMEOUT; both
  2026-09-20 attempts failed/crashed before final eval). Resubmitted 2026-10-01 as
  jobs 5844331 / 5844332 with the walltime fix; those jobs started after the eval fix
  landed only if they baked the script post-40295bb — verify on completion, else the
  psnr_mean they report is biased and per-view must be used.
- **E5b**: wandb crashed at final eval and the arm is not in
  clean_peraxis_metrics.json. The exact corrected number is in the cluster
  `eval_metrics.jsonl` (`final_eval` → `psnr_per_view`). Last mid-training full_eval
  per-view was 28.06/28.42 (biased mean then 25.87 vs final 26.01), so the corrected
  final is ≈28.4–28.6 dB — refill from cluster, do not print the estimate.
- **SSIM (all arms)**: corrected SSIM requires the eval dumps
  (`final_eval_dump_val.pt` per job dir on the cluster); logs only contain the biased value.
- **Seed-2 arms + zero-shot refresh (jobs 5845272–5845277)**: still queued as of
  2026-10-02 14:00 UTC+2 (no wandb runs created after 2026-09-20); they will report
  corrected psnr_mean directly once they run (script baked at job start includes 40295bb).

