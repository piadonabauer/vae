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

NOTE for the main table: the old LPIPS values of E0/E1a/E1b/E1c/E1d went
through the clamp-biased eval and could NOT be recovered (dumps gone) — REDO
arms get fresh LPIPS from this wave; the E0 (all-data, KEEP) row's LPIPS/Bleed
must be footnoted as pre-fix or dropped.

Takeaways:

- The genuine Wan zero-shot floor on NeRSemble is **>30 dB without any
  training**; the paper's old 19.71 (pre-fix) / 23.07 (corrected) zero-shot row
  was the TC=off per-frame wrapper, not Wan — label it as such.
- Local-pipeline validation: TC=off @128² reproduces the cluster's corrected
  value to 0.02 dB (23.05 vs 23.07) on independently re-downloaded and
  re-preprocessed data.
- TC=off degrades *with* resolution (23.05 → 22.74) while TC=on improves
  (32.51 → 34.13) — more evidence the wrapper path is the bottleneck.

## M1 — core arms @170 epochs, seed 42 (RUNNING)

Queue order: E1b → E1c → E1d → E11a → E11b → combo → E_best
(`tmux attach -t paperwave`; logs in `Open-Sora/local_logs/`).

| arm | status | PSNR (val) | old corrected (cluster) |
|---|---|---|---|
| E1b per-view TC=on | training | — | 33.19 |
| E1c fused TC=off | queued | — | 32.74 |
| E1d fused TC=on | queued | — | 28.27 |
| E11a widen-32 | queued | — | 29.78 |
| E11b widen-64 | queued | — | 30.92 (was still climbing) |
| combo | queued | — | 31.39 |
| E_best all-combined (new) | queued | — | — |

## M2+ — extensions to 300 ep / references / seed 43

Pending M1; see plan §L3.
