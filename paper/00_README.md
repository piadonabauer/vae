# Paper branch — what is here and how to run it

Everything needed for the paper lives on this branch (`paper-experiments`):
the writing material in `paper/`, the (minimal) code changes for the reruns,
and the scripts to launch the experiments and collect the numbers.

## Layout

| Where | What |
|---|---|
| `paper/01_draft.md` | The paper draft (intro, related work, method, CVPR-style experiments section, discussion). All numbers are `[placeholders]` until the reruns finish. |
| `paper/02_experiments.md` | The internal run plan: fixed protocol, every experiment arm (E1–E10), priorities, and which old results are superseded. Read this before launching anything. |
| `paper/03_figures.md` | Figure plan (teaser, architecture, rate–quality curve, bleed/ghosting visualizations). |
| `paper/references.bib` | BibTeX for every `\cite` key in the draft. |
| `run_paper_sweep.sh` | Slurm sweep for the core arms (E1a–d, E5b/c, E1z). One fixed protocol, only the model config differs per arm. |
| `Open-Sora/scripts/vae/collect_results.py` | Collects `eval_metrics.jsonl` from all `outputs/paper_*` runs into one CSV for the tables/plots. |

Code changes vs. the old branches (all in commit `52f6333`):
- `fusion_mode="none"` + `--model.independent_views True`: per-view reference
  mode (views folded into batch, plain Wan + LoRA, no fusion, no view conditioning).
- Eval diagnostics in `train.py`: bleed_ratio_within / bleed_ratio_across,
  cross-view reconstruction similarity (rec and gt), per-frame PSNR profile.
  All logged to wandb and appended to `outputs/<run>/eval_metrics.jsonl`.

## How to run

Fixed protocol (details in `02_experiments.md`): V=2, T=9, 128px,
`all_people_one_expression`, effective batch 64, 170 epochs, lr 5e-4, LoRA
rank 32, no discriminator, EMA eval on 10 held-out identities.

Two-stage discipline — every arm first has to pass the overfit gate:

```bash
# Stage 1: overfit gate (single sequence, batch 1; expect near-perfect PSNR)
sbatch --export=ALL,TASK=4,OVERFIT=1 ./run_paper_sweep.sh

# Stage 2: the real generalization run
sbatch --export=ALL,TASK=4 ./run_paper_sweep.sh
```

Order matters because of the staged init: arms 4–6 (fused + temporal
compression) warm-start from the arm-2 checkpoint by default.

1. arms 1, 2, 3, 7 — no dependencies, start immediately
2. arms 4, 5, 6 — after arm 2 finished (script auto-finds the E1b checkpoint,
   or pass `INIT_CKPT=/path/to/epochN-...`; `INIT_CKPT=none` = the E7b
   init ablation, i.e. Wan-only weights)

The script handles OOM (retries smaller batch at same effective batch) and
auto-resumes from the newest checkpoint of a previous job for the same arm.

## Collecting results

```bash
cd Open-Sora
python scripts/vae/collect_results.py            # scans outputs/paper_*
```

Writes `paper_results.csv` with best-val and last rows per run (PSNR/SSIM/MSE,
bleed ratios, cross-view similarity, per-frame PSNR profile). The rate–quality
plot and all tables in the draft are built from this file.

## What is NOT rerun by the sweep script

The remaining ablations (fusion modes E2, view conditioning E3, view count E4,
init E7b, data scale E8, ...) are specified in `02_experiments.md` with exact
flags but launched by hand / small variations of the sweep script.
