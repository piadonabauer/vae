# Final wave results (local L40S, started 2026-10-03)

Living results file for the local final wave (plan: `rerun_plan_2026-10-02.md`
§L). PSNR convention: mean of per-(clip,view) PSNRs on [0,1] val reconstructions
(`paper/audit/metrics_unified.py`). Raw numbers per run live in
`Open-Sora/outputs/<run>/eval_metrics.jsonl` (`final` entry, `psnr_per_view`)
and on wandb; full tensors in `final_eval_dump_{train,val}.pt` (feeds SSIM
recompute + qualitative figures).

## M0 — zero-shot evals (DONE 2026-10-03, no training)

| arm | res | PSNR (val) | LPIPS | SSIM (win) | Bleed-W | Bleed-A | wandb |
|---|---|---|---|---|---|---|---|
| **TRUE zero-shot, pretrained Wan, TC=on** | 128² | **32.51** | 0.043 | 0.931 | 0.977 | 1.006 | [1jep01kb](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/1jep01kb) |
| TRUE zero-shot, pretrained Wan, TC=on | 256² | **34.15** | 0.060 | 0.931 | 0.985 | 0.996 | [f2nalxpn](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/f2nalxpn) |
| zero-shot TC=off (per-frame wrapper, = old E1z) | 128² | 23.05 | 0.146 | 0.887 | 1.137 | 1.367 | [w95ecfdw](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/w95ecfdw) |
| zero-shot TC=off (per-frame wrapper) | 256² | 22.74 | 0.183 | — | — | — | [00mnihtl](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/00mnihtl) |

LPIPS = `final_eval/val/lpips_mean` (same VGG eval as all table rows). SSIM =
canonical 11×11 windowed, bleed ratios per `paper/audit/metrics_unified.py`,
both recomputed from the local `final_eval_dump_val.pt` via
`recompute_from_dumps.py` (loader fixed 2026-10-03 for the real train.py dump
format; dump-PSNR matches wandb to 0.01 dB). Main-table zero-shot row (filled):
`Wan VAE zero-shot & on & 36x & 32.51 & 0.043 & 0.977`.

NOTE for the main table (CORRECTED 2026-10-03, verified against pre-fix code
`40295bb^` lines 1368–1405): the clamp bug biased ONLY `compute_metrics`
outputs — `psnr_mean`, `ssim_mean`, `mse`. **LPIPS and bleed ratios were
always computed from the correctly remapped `x01` tensors and are valid in
every old table (Table 2, appendix included)** — matching the audit's
per-metric verdict; an earlier note here claiming old LPIPS was biased was
wrong. Empirical cross-check: old E1z LPIPS 0.155 ≈ fresh TC-off 0.146 while
its PSNR moved 19.71→23.05. The column that DOES need fresh dumps is **SSIM**
(windowed, corrected) — available only for arms in this wave; KEEP-arm SSIMs
remain unrecoverable and should be omitted or footnoted.

Takeaways:

- The genuine Wan zero-shot floor on NeRSemble is **>30 dB without any
  training**; the paper's old 19.71 (pre-fix) / 23.07 (corrected) zero-shot row
  was the TC=off per-frame wrapper, not Wan — label it as such.
- Local-pipeline validation: TC=off @128² reproduces the cluster's corrected
  value to 0.02 dB (23.05 vs 23.07) on independently re-downloaded and
  re-preprocessed data.
- TC=off degrades *with* resolution (23.05 → 22.74) while TC=on improves
  (32.51 → 34.13) — more evidence the wrapper path is the bottleneck.

## M1 — core arms @170 epochs, seed 42 (updated 2026-10-04 17:30)

Queue order: E1b → E1c → E1d → E11a → E11b → combo → E_best → E1a → E4h
(`tmux attach -t paperwave`; logs in `Open-Sora/local_logs/`).

| arm | status | PSNR (val) | LPIPS | old corrected (cluster) | Δ |
|---|---|---|---|---|---|
| E1b per-view TC=on | **done** | **33.13** | 0.0370 | 33.19 | −0.06 |
| E1c fused TC=off | **done** | **33.13** | 0.0374 | 32.74 | +0.39 |
| E1d fused TC=on | **done** | **28.28** | 0.0692 | 28.27 | +0.01 |
| E11a widen-32 | **done** | **30.35** | 0.0540 | 29.78 | +0.57 |
| E11b widen-64 | **done** | **30.20** | 0.0528 | 30.92 | −0.72 |
| combo | **done** | **30.36** | 0.0526 | 31.39 | −1.03 |
| E_best all-combined (new) | **done** | **31.56** | 0.0505 | — (new arm) | |
| E1a per-view TC=off | **done** | **35.85** | 0.0218 | 35.89 | −0.04 |
| E4h diff-loss | **done** | **27.20** | 0.0749 | — | |

**M1 COMPLETE** (all 9 arms, 2026-10-04 19:40 PT). E1a wandb: n7flxf2v,
E4h wandb: 47modqhy. E1a's −0.04 is the third near-exact reproduction
(with E1b −0.06, E1d +0.01) — comparability to the old wave is solid.
E4h (diff-loss alone on the fused TC-on base) lands BELOW the E1d
baseline on PSNR (27.20 vs 28.28): the temporal-diff term trades pixel
PSNR for temporal consistency; its value shows up combined with capacity
(combo 30.36, E_best 31.56), not alone.

E_best note: best TC-on fused arm already at 170 ep (beats combo by +1.2 and
E11b by +1.4), trained FROM SCRATCH (rank-128 LoRA can't load the rank-32 E1b
warm start — disclose in the caption). Gap to the per-view TC-on reference
(E1b): 1.57 dB at 170 ep; the 300-ep extension will narrow or confirm it.

Reading:

- **Comparability validated**: E1b (−0.06) and E1d (+0.01) reproduce the old
  values on independently re-created data — the protocol is sound.
- E1c +0.39 and E11a +0.57: the uniform warm-start/budget helped (old E11a was
  from scratch).
- **E11b −0.72 and combo −1.03**: both were from-scratch in the old wave and
  are warm-started now; at 170 epochs the warm start has NOT caught up for the
  widened arms — consistent with the convergence audit (E11b still climbing).
  **Do not conclude capacity ordering from the 170-ep numbers.** The M2
  extensions to 300 epochs (auto-chained in tmux session `paperwave2`,
  tasks 9/36/52) are the decisive numbers for the capacity table.

## Qualitative figures (regenerated 2026-10-04 from local final dumps)

`paper/figures_cvpr/make_qual_from_dumps.py` now reads the local
`final_eval_dump_val.pt` of each finished arm (images therefore match the
reported final_eval/val numbers exactly; same clip 0 / frame 5 as before).
Regenerated: `qual_ghosting`, `qual_bleeding`, `qual_capacity` (paper figs,
same composition as before), plus two new working visuals (PDF+PNG each):

- `qual_overview_finalwave` — all 8 arms + GT on one clip/frame, both views,
  with a ×5 error heatmap row. E1d's error map is visibly the worst (mouth,
  eyes, hair); E_best's is close to E1b/zero-shot despite 72×→18× rate.
- `qual_best_temporal` — GT vs E1d vs E_best across all 9 frames: E1d's
  chunk-interior frames (f2–f5) show the mouth ghosting/blur; E_best keeps
  the mouth shape and identity sharp throughout (+3–7 dB per frame).

Caveat: frame 5 is an odd index, so these frames may differ by ±1 source
frame from the old-wave PDFs — never compare old vs new figures per-frame.
Re-run the script after M2 to refresh with 300-ep reconstructions, and add
E1a to `DUMPS` once its final dump exists.

## M2+ — extensions to 300 ep / references / seed 43

2026-10-05 01:39 PT: the paperwave2 auto-chain never fired — its
`tmux has-session -t paperwave` check matched its OWN session by prefix
("paperwave" matches "paperwave2"), so it waited forever and the GPU sat
idle 19:40–01:39. Killed the watcher and started M2 directly in tmux
`paperwave` (`TRAIN_EPOCHS=300 QUEUE="9 36 52"`, log
`local_logs/queue_m2.log`); the @reboot cron now relaunches this M2
command. E11b verified resuming at epoch 170 from its checkpoint
(~3.5 s/it → ~3.5–4 h per arm). ETA: all three 300-ep arms (E11b, combo,
E_best) done ~Mon evening. M4/seed-43 decisions after that.

### M2 results @300 epochs, seed 42

COMPLETE 2026-10-06 (M2+M3 all drained overnight):

| arm | rate | PSNR @170 | PSNR @300 | Δ ext | LPIPS @300 | wandb |
|---|---|---|---|---|---|---|
| E1d 16-ch | 72× | 28.28 | 28.20 | −0.08 | 0.0709 | 4agg9o4d |
| E11a 32-ch | 36× | 30.35 | 30.61 | +0.26 | 0.0505 | bxa9fpmp |
| E11b 64-ch | 18× | 30.20 | 31.34 | +1.14 | 0.0465 | (prev) |
| combo w32+diff | 36× | 30.36 | **31.71** | +1.35 | 0.0475 | 5j9rgid2 |
| E_best all tweaks | 18× | 31.56 | 31.65 | +0.09 | 0.0468 | k84i5082 |
| E1b per-view ref | 36× | 33.13 | 33.12 | −0.01 | 0.0368 | ng1hy27r |

Findings @300 (the capacity-table numbers):

- E1b reference and E1d baseline are CONVERGED (−0.01 / −0.08) → captions
  can say "300 epochs"; tweak delta vs no-tweaks is +3.45 dB (28.20→31.65).
- **combo (31.71, 36×) now beats E_best (31.65, 18×)** at twice the
  compression — diff-loss + 32-ch is the better recipe than all-tweaks-in;
  E_best barely moved with the extension (+0.09, from-scratch arm).
- Capacity ordering @300: 16ch 28.20 < 32ch 30.61 < 64ch 31.34 <
  combo 31.71 ≈ E_best 31.65. Combo @36× > 64-ch @18×: the loss tweak
  outperforms pure widening even with half the latent budget.
- Gap of best fused arm to per-view reference: 1.41 dB (33.12 − 31.71).
- E11b @300 exceeds its old cluster value (30.92 → 31.34, +0.42) —
  confirms the convergence audit: wide arms were undertrained at 170 ep.

### M3 — complete the 300-ep capacity table (queued 2026-10-05)

After M2 drains, tmux `extchain` runs `run_extension_pipeline.sh` →
`TRAIN_EPOCHS=300 QUEUE="4 8 2"`: E1d (16-ch) → E11a (32-ch) → E1b
(reference insurance), each resuming from its epoch-169 checkpoint.
The @reboot cron now launches the same pipeline (idempotent: .DONE
markers skip finished arms; `.M3_STARTED` sentinel guards the one-time
marker clearing). ETA: M3 done ~Tue morning.

### Identity metrics + draft figure variants (2026-10-05, eval-only)

`paper/audit/identity_metrics.{py,json,md}` — FaceNet (VGGFace2) embeddings
from the 170-ep final dumps, GT-detected crops applied identically to rec.
Headlines (id sim rec-vs-GT / cross-view gap, GT xview baseline 0.964):

- per-view TC off 0.971 / +0.025 — fused TC off 0.924 / +0.049
- per-view TC on 0.893 / +0.061 — zero-shot 0.889 / +0.064
- fused TC on 16-ch **0.756 / +0.176** — 32-ch 0.821 — 64-ch 0.830
- combo 0.846 / +0.111 — all tweaks 0.849 / +0.101

Two usable findings: (1) joint view+temporal compression hurts identity far
more than either axis alone (gap 0.176 vs ~0.05), and tweaks only halve it;
(2) TC hurts identity MORE than view fusion at equal PSNR cost — per-view
TC-on (33.1 dB) has lower id-sim than fused TC-off (33.1 dB), 0.893 vs 0.924.

Draft figure variants (separate files, originals untouched; from 170-ep
dumps, rerun `make_qual_variants.py` after M3 for @300): `qual_ghosting_v2`,
`qual_capacity_v2` (insets: mouth/hair + diff maps), `perframe_psnr_v2`
(legend outside, per-view ref line), `perframe_motion` (decoded motion as %
of GT motion on moving pixels — the direct bleeding evidence: fused 16-ch
reproduces only ~79–81% of true motion at early transitions).

Still pending (need GPU/checkpoints; run after the matched-width controls):
latent statistics (sampleability proxy), view-swap eval of fused TC-off.
NOT queued (decision needed): E5b refill, V=8 (needs new data download),
512² (no data), seed-43.

### M4 — matched-width per-view controls (launched 2026-10-06)

Tasks 53/54 (`paper_Ectrl_perview_tcT_widen{32,64}`): per-view TC-on with
latent widened to 32/64 ch — the control for "is the fused gap just
capacity?". Protocol parity with the fused widen arms: warm start from the
SAME E1b epoch-169 checkpoint (INIT_CKPT passed explicitly; the auto-finder
would now pick the extended @300 E1b), TRAIN_EPOCHS=300, seed 42. Task 53
OOM'd at batch 16 and runs at batch 8 / accum 8 (effective 64 unchanged).
ETA ~9–10 h each → both done ~Wed morning. Log: `local_logs/queue_m4.log`.

### Reporting rule (budgets per table)

- Keep @170 and @300 side by side in this file for every arm that has
  both; never mix budgets within one paper table.
- Table 1 (main matrix) and Table 2 (ablations): @170 throughout.
- Table 3 (capacity): @300 throughout — 16-ch, 32-ch, 64-ch, combo,
  E_best all at 300 ep once M3 lands.
- E1b @300 is insurance: if it stays ~33.1, the reference is converged
  and captions can say "300 epochs"; if it moves, use the @300 value.
- State the two budgets once, in Setup.
- Note: E1d @300 may raise the no-tweaks baseline and shrink E_best's
  "+3.28 dB from tweaks" delta — report the @300 delta, whatever it is.
