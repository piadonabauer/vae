# Final wave results (local L40S, started 2026-10-03)

Living results file for the local final wave (plan: `rerun_plan_2026-10-02.md`
§L). PSNR convention: mean of per-(clip,view) PSNRs on [0,1] val reconstructions
(`paper/audit/metrics_unified.py`). Raw numbers per run live in
`Open-Sora/outputs/<run>/eval_metrics.jsonl` (`final` entry, `psnr_per_view`)
and on wandb; full tensors in `final_eval_dump_{train,val}.pt` (feeds SSIM
recompute + qualitative figures).

## M0 — zero-shot evals (DONE 2026-10-03, no training)

| arm | res | PSNR (val) | view0 / view1 | wandb |
|---|---|---|---|---|
| **TRUE zero-shot, pretrained Wan, TC=on** | 128² | **32.51** | 32.56 / 32.45 | [1jep01kb](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/1jep01kb) |
| TRUE zero-shot, pretrained Wan, TC=on | 256² | **34.13** | 34.17 / 34.09 | [f2nalxpn](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/f2nalxpn) |
| zero-shot TC=off (per-frame wrapper, = old E1z) | 128² | 23.05 | 23.18 / 22.92 | [w95ecfdw](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/w95ecfdw) |
| zero-shot TC=off (per-frame wrapper) | 256² | 22.74 | 22.85 / 22.62 | [00mnihtl](https://wandb.ai/pia-uni/wan_multiview_vae_paper/runs/00mnihtl) |

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
