# Rerun plan + CVPR completeness checklist (2026-10-02)

Everything below is ready to copy-paste **on the cluster** (Vector, user
`piado`), from the repo root on branch `paper-experiments` after `git pull`.
Background: `paper/eval_audit_2026-10-02.md`. Nothing here retrains existing
arms — the clamp bug never touched losses or checkpoints; these are the gaps
the audit could not fill locally plus two crashed arms.

Prerequisite everywhere:

```bash
cd /home/piado/projects/aip-lindell/piado/vae && git pull   # must include TASK 50/51
```

---

## A. Jobs to submit (sbatch)

### A1. TRUE zero-shot baseline, TC=on (REQUIRED for the paper)

The current zero-shot row (TASK=7, 19.71 → corrected 23.07 dB) runs
`temporal_compression=False`: per-frame decode, no feat_cache, pretrained
temporal convs skipped. That costs ~6 dB at 128² (measured, see audit §2).
TASK=50/51 are new eval-only tasks (epochs=0, ~minutes of GPU) that run the
native Wan chunked path. Expected ≈28–30 dB at 128².

```bash
sbatch --parsable --partition=gpubase_l40s_b3 --time=0-03:00:00 \
  --export=ALL,TASK=50,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh    # true zero-shot @128, run paper_E1z_perview_zeroshot_tcON

sbatch --parsable --partition=gpubase_l40s_b3 --time=0-03:00:00 \
  --export=ALL,TASK=51,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh    # true zero-shot @256, run paper_E1z_perview_zeroshot_tcON_256
```

Table use: report BOTH rows — "pretrained, per-frame wrapper (TC off)" (the
old row, now 23.07) and "pretrained, native chunked (TC on)" (new). The TC-on
row is the honest floor a reviewer would reproduce with stock Wan.

### A2. E6d / E6e (V=8) — both crashed on 2026-09-20

Only needed if the paper keeps the V=8 point in the view-count figure
(currently plotted for V=2,4 only). Check the previous crash logs first
(`Open-Sora/slurm_logs/`, runs `paper_E6d_*`/`paper_E6e_*` — likely OOM at
V=8; if so add `--mem=96G` or drop to the "1:64" batch rung).

```bash
sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=31,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh    # E6d V=8 TC=off

sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=32,OVERFIT=0,CHAIN_LEFT=0,SAVE_CKPT=False,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh    # E6e V=8 TC=on
```

### A3. Already queued — do NOT resubmit

Jobs 5845272–5845277 (`run_seed2_and_zeroshot.sh`: zero-shot 128/256 with the
fixed eval + seed-43 E1c/E1d/E11a/combo). As of 2026-10-02 they had not started
(no wandb runs after 2026-09-20). When they land, their `final_eval/val/psnr`
is already unbiased — paste directly into the corrected table, and report
seed-42/seed-43 as mean ± half-range for the headline arms.

---

## B. Post-processing on the cluster (no training, CPU/single-GPU, minutes)

### B1. Corrected SSIM for every arm + E5b corrected PSNR + missing per-frame PSNRs

One command per arm, pointed at its final_eval dump (saved in each run's
output/exp dir as `final_eval_dump_val.pt` / `final_eval_dump_train.pt`):

```bash
# example for one arm; repeat for every run dir in paper/clean_retrain_jobs.txt
python paper/audit/recompute_from_dumps.py \
  Open-Sora/outputs/<run_dir>/final_eval_dump_val.pt
```

Prints and appends to `recomputed_from_dumps.jsonl` next to the dump:
corrected PSNR (paper convention: mean of per-(clip,view) PSNRs), per-view,
per-frame, canonical windowed SSIM (use THIS as the table SSIM), legacy
global-moment SSIM, bleed ratios. Bulk version:

```bash
find Open-Sora/outputs -name 'final_eval_dump_val.pt' \
  -exec python paper/audit/recompute_from_dumps.py {} +
```

Then fill into `paper/clean_retrain_metrics_corrected.md`: the `SSIM new`
column (currently `dump`), the E5b PSNR row (currently `n/a`), and per-frame
PSNRs for arms missing from `paper/clean_peraxis_metrics.json`.

### B2. Data-scale figure refill (single-sequence point)

`paper/figures_cvpr/make_figures_cvpr.py::fig_datascale` still plots the
PRE-FIX 34.56 dB for the single-sequence arm (biased >=1.7 dB LOW; dagger was
removed on request, so this is now silent — do not forget it). Fix: run B1 on
the E8b/one_person run's dump (or its `final_eval_metrics.jsonl` per-view
PSNRs) and replace `psnr = [34.56, 27.16]` with the corrected value. The
plotted trend only gets stronger.

### B3. NeRSemble zero-shot resolution/matting ladder (audit §2c, cluster-only)

```bash
# point at one preprocessed clip tensor [V,C,T,H,W] in [0,1]; runs native and
# TC-off paths at 128/256/512 and writes zeroshot_local_results.json
python paper/audit/zeroshot_local_check.py \
  --nersemble_pt /datasets/lindell-proj/neumayr/nersemble_v2/processed/512-res/<some_clip>.pt
```

For the with/without-matting comparison, pass a clip preprocessed without the
white-matting step (re-run `data/processing/preprocess_nersemble.py` on one
sequence with matting disabled) — only one clip is needed, this is a
diagnostic, not a table row.

---

## C. Things CVPR reviewers will expect (completeness checklist)

1. **True zero-shot baseline (A1)** — the single most attackable number right
   now; a reviewer rerunning stock Wan gets ~28 dB at 128², not 23.
2. **rFVD (reconstruction FVD)** — standard for video VAE papers; PSNR/SSIM/
   LPIPS alone is light. Compute from the final_eval dumps (gt vs rec, I3D
   features); the dumps already contain everything needed.
3. **Multi-seed error bars** — seed-43 jobs (A3) cover the headline arms
   (E1c/E1d/E11a/combo); report mean ± half-range in the main table, state
   single-seed for the rest.
4. **Compression-rate accounting per arm** — the 32/64-ch and TC-off arms
   change the rate: report latent floats per pixel next to PSNR
   (e.g. 128²×9 frames → TC-on 16ch: 3×16×16×16 latents; TC-off: 9×...;
   widen-32 doubles it). The rate–quality figure must say this explicitly or a
   reviewer will call the widen gains "just more bits".
5. **Trainable-parameter counts** — LoRA-rank sweep (E10) begs for a
   params-vs-PSNR statement; one column in the supplement table.
6. **Data statement for NeRSemble** — identifiable faces: cite the NeRSemble
   license/consent terms and the train/val participant split (no val
   participant in training). CVPR requires this in the ethics section.
7. **Resolution figure provenance** — the 128/256/512 points mix pre-clean-wave
   runs (E9a/E9b) with a clean-wave 128 point; either rerun E9a/E9b in the
   clean config (TASK=34/35) or footnote the provenance.
8. **SSIM definition** — after B1, the table SSIM switches from the
   global-moment approximation to canonical windowed SSIM; state the window
   (11×11 Gaussian, σ=1.5) in the supplement. Metric conventions (PSNR
   aggregation, bleed pair indices) are documented in
   `paper/audit/metrics_unified.py`.
9. **Eval-bug disclosure** — one supplement sentence: headline PSNR/SSIM were
   recomputed after fixing a range bug in eval (commit 40295bb); diagnostics
   (LPIPS, bleed, per-view/per-frame) were unaffected. Full audit:
   `paper/eval_audit_2026-10-02.md`.

---

## D. Where everything lives

| artifact | path |
|---|---|
| full audit (metrics + zero-shot, all evidence) | `paper/eval_audit_2026-10-02.md` |
| corrected table (old/new per arm) | `paper/clean_retrain_metrics_corrected.md` |
| unified metric module + tests | `paper/audit/metrics_unified.py`, `paper/audit/test_metrics_unified.py` |
| table builder (wandb → corrected md) | `paper/audit/build_corrected_table.py` |
| dump → corrected metrics tool (B1) | `paper/audit/recompute_from_dumps.py` |
| zero-shot local/cluster experiment script | `paper/audit/zeroshot_local_check.py` |
| local zero-shot results (jellyfish / BBB) | `paper/audit/zeroshot_local_results.json`, `paper/audit/zeroshot_bbb_results.json` |
| figures (regenerated, corrected) | `paper/figures_cvpr/*.pdf` via `make_figures_cvpr.py` |
| this plan | `paper/rerun_plan_2026-10-02.md` |
