# Rerun plan + CVPR completeness checklist (2026-10-02, FINAL)

Everything below is ready to copy-paste **on the cluster** (Vector, user
`piado`), from the repo root on branch `paper-experiments` after `git pull`.
Background: `paper/eval_audit_2026-10-02.md`.

Prerequisite everywhere:

```bash
cd /home/piado/projects/aip-lindell/piado/vae && git pull   # must include TASK 50/51
```

---

## THE DEFINITIVE REDO / KEEP LIST (read this first)

Decided 2026-10-02 after the convergence + comparability audit
(`paper/audit/convergence_curves.json`; wandb configs). Reasons that force a
retrain of the main table: (i) E11b still climbing at epoch 170 (+0.9 dB over
the last ~45 epochs), E11a/combo borderline; (ii) warm-start was NOT uniform —
E1d/E4h warm-started from (two different!) E1b checkpoints while E11a/E11b/
combo trained from scratch; (iii) no checkpoints were saved (SAVE_CKPT=False),
so extending is impossible and only a restart keeps arms budget-comparable.

| item | verdict | where |
|---|---|---|
| Main-table arms E1a/E1b/E1c/E1d/E4h/E4i/E11a/E11b/combo | **REDO** — final wave, 300 epochs, uniform staged init, ckpts, 2 seeds for headline | §A0 |
| E_best "all improvements combined" (widen64+diff+perc0.5+rank128, TASK=52) | **RUN (new)** — one arm, answers "did you tune?" | §A0 |
| True zero-shot TC=on @128/@256 | **RUN (new)** — eval-only, minutes | §A1 |
| Zero-shot TC=off rows (queued 5845272/73) | keep — label as "per-frame wrapper", not Wan floor | §A3 |
| Queued seed-43 trainings (5845274–77) | **CANCEL** — superseded by §A0 (old budget + mixed init; E11a/combo inits don't even match their seed-42 partners) | §A3 |
| E6d/E6e (V=8, crashed) | REDO **only if** the paper keeps V=8; else drop the point | §A2 |
| E9a/E9b (256/512 resolution) | REDO at final-wave budget **if** the resolution figure stays (its 128 px partner becomes the new E1d) | §A0b |
| E8b (single-person data-scale arm) | REDO at final-wave budget **if** the data-scale figure stays (same reason) | §A0b |
| Supplement families E2*, E3*, E7*, E10*, E6b/c, E5b/c | **KEEP** — converged (flat val curves), internally consistent, only compared within family | — |
| E0 ceiling | KEEP — converged at 36.0, by-design different budget/data | — |
| Corrected SSIM / E5b PSNR / per-frame refills from dumps | DO — still needed for every KEEP arm | §B1 |
| Data-scale single-seq 34.56 figure value | superseded if E8b is redone; else refill from dump | §B2 |
| NeRSemble 128/256/512 ± matting zero-shot ladder | DO — diagnostic, one clip | §B3 |
| rFVD, rate accounting, param counts, data statement | DO — paper text/supplement work | §C |

---

## A0. FINAL WAVE — main-table retrain (the "last time" run)

**Design (uniform for all arms):** `TRAIN_EPOCHS=300` (converged arms will just
confirm their plateau; wide arms get their missing tail), `SAVE_CKPT=True`
(never again be unable to extend), `CHAIN_LEFT=1` (300 epochs ≈ 32 h = 2
chained 18 h jobs; the chain auto-resumes from the saved ckpt), seed 42 for
all + seed 43 for the headline four (E1c, E1d, E11a, combo). Init policy:
staged — per-view arms and TC-off fused from pretrained Wan only
(`INIT_CKPT=none`); every TC-on fused arm warm-starts from the **final-wave
E1b of the same seed** (this fixes both old blemishes: mixed E1b sources and
scratch-vs-warm asymmetry). Budget: 14 runs ≈ 450 GPU-h ≈ 4–5 days at 4
concurrent.

**Phase 1 — no-warm-start arms (submit immediately, ~32 h each):**

```bash
for S in 42 43; do EXTRA=""; [[ $S == 43 ]] && EXTRA=",SEED=43"
  sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
    --export=ALL,TASK=2,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT=none$EXTRA \
    --array=1 ./run_paper_sweep.sh   # E1b (warm-start source for phase 2)
  sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
    --export=ALL,TASK=3,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT=none$EXTRA \
    --array=1 ./run_paper_sweep.sh   # E1c
done
sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=1,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh     # E1a (seed 42 only)
```

**Phase 2 — TC-on fused arms (submit when the matching-seed E1b finishes).**
Locate the new checkpoints (seed-43 run dirs end in `_s43`):

```bash
ls -d Open-Sora/outputs/paper_E1b_perview_tcT*/epoch299-* 2>/dev/null
E1B42=<paste seed-42 epoch299 ckpt path>
E1B43=<paste seed-43 epoch299 ckpt path>
```

```bash
# seed 42: E1d, E4h, E4i, E11a, E11b, combo
for T in 4 24 28 8 9 36; do
  sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
    --export=ALL,TASK=$T,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT="$E1B42" \
    --array=1 ./run_paper_sweep.sh
done
# seed 43: E1d, E11a, combo (headline error bars)
for T in 4 8 36; do
  sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
    --export=ALL,TASK=$T,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,SEED=43,INIT_CKPT="$E1B43" \
    --array=1 ./run_paper_sweep.sh
done
# E_best "all improvements combined" (TASK=52, new): widen64 + diff-loss +
# perc0.5 + rank128 under TC-on fused -- last row of the capacity table,
# answers "did you tune?". Rank-128 LoRA shapes cannot load the rank-32 E1b
# warm start, so this arm runs from scratch (disclose in the table footnote);
# if train.py's loader skips mismatched keys cleanly you may try "$E1B42" and
# keep it only if the load log shows no dropped non-LoRA keys.
sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
  --export=ALL,TASK=52,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT=none \
  --array=1 ./run_paper_sweep.sh
```

**Pre-flight checks (do BEFORE submitting phase 2, 5 minutes):**

1. Widen-aware warm-start: the seed scripts used `INIT_CKPT=<E1b>` for the
   32-ch arm ("widen init handles 16→32"), but 16→64 (TASK=9, E11b) has never
   been exercised. Verify with `DRY_RUN=1 TASK=9 INIT_CKPT=<E1b ckpt>
   ./run_paper_sweep.sh` and check the printed load/widen log; if the loader
   rejects 16→64, run E11b with `INIT_CKPT=none` and **say so in the paper**
   (one asymmetric arm, disclosed, beats a silent one).
2. Confirm each phase-2 run's wandb config shows `load=<the new E1b>` and
   `epochs=300` in its first minutes.
3. After ~1 day, confirm on wandb that the flat arms (E1a/E1b/E1c) match their
   170-epoch values (±0.1 dB) — that is the sanity check that the final wave
   is consistent with the audited corrected table.

**When done:** rebuild the table (`paper/audit/build_corrected_table.py`
matches runs by name; final-wave run names are identical with `_s43` suffixes
for seed 43), report headline arms as mean ± half-range over seeds, and run
§B1 on the new dumps for SSIM. Expected movement vs the corrected table:
E1a/E1b/E1c/E1d ±0.1; E4h/E4i +0.0–0.2; E11a +0.1–0.3; E11b +0.3–0.8;
combo +0.0–0.5. All paper claims only strengthen.

## A0b. Figure-support arms at the final-wave budget (optional, same pattern)

Only if the respective figure stays in the paper — their comparison partner
(the 128 px / all-people point) becomes the NEW E1d, so budgets must match:

```bash
# E9a 256px, E9b 512px (resolution figure), E8b one-person (data-scale figure)
for T in 34 35 26; do
  sbatch --parsable --partition=gpubase_l40s_b3 --time=0-18:00:00 \
    --export=ALL,TASK=$T,OVERFIT=0,CHAIN_LEFT=1,TRAIN_EPOCHS=300,SAVE_CKPT=True,INIT_CKPT="$E1B42" \
    --array=1 ./run_paper_sweep.sh
done
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

### A3. Already queued jobs 5845272–5845277 — keep two, CANCEL four

As of 2026-10-02 none had started (no wandb runs after 2026-09-20).

- **Keep 5845272/73** (zero-shot TC-off @128/@256, fixed eval): their numbers
  go in the table as the "pretrained, per-frame wrapper" row (~23 dB expected);
  the Wan floor row comes from §A1.
- **Cancel 5845274–77** (seed-43 E1c/E1d/E11a/combo): superseded by the §A0
  final wave. They would train at the old 170-epoch budget AND with
  inconsistent inits — `run_seed2_and_zeroshot.sh` warm-starts seed-43
  E11a/combo from a seed-42 E1b checkpoint, while the seed-42 clean-wave
  E11a/combo trained from scratch, so they would not even be valid seed
  repeats of their partners. `scancel 5845274 5845275 5845276 5845277`.

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
removed on request, so this is now silent — do not forget it). If §A0b reruns
E8b, both figure points come from the final wave and this refill is moot.
Otherwise: run B1 on the E8b/one_person run's dump (or its
`final_eval_metrics.jsonl` per-view PSNRs) and replace `psnr = [34.56, 27.16]`
with the corrected value. The plotted trend only gets stronger.

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

### B4. Add the 64-ch row to the qualitative bleeding grid

`qual_bleeding_grid.pdf` currently shows GT/16-ch/32-ch only because the
cluster-generated source (`paper/figures/qual_bleeding.pdf`, built by
`paper/figures_cvpr/make_qual_from_dumps.py`) never included E11b. The script
now has the 64-ch row; on the cluster run:

```bash
python paper/figures_cvpr/make_qual_from_dumps.py   # needs the E11b best_val dump
```

commit the regenerated `paper/figures/qual_bleeding.pdf`, then locally rerun
`make_figures_cvpr.py` (the extractor auto-detects 3 vs 4 rows) and copy the
64-ch per-frame dB printed in the new source into the `psnrs` list.
(After the §A0 final wave, point `DUMPS` in make_qual_from_dumps.py at the
final-wave run dirs instead.)

**Same identity + same frame across the qualitative pair (supervisor request):**
the ghosting (failure) and capacity (repair) figures must show the SAME clip
and frame so a reader can line them up as one story. This is now enforced in
`make_qual_from_dumps.py` via `QUAL_CLIP = 0` / `QUAL_FRAME = 5` (f5 = the
hardest frame: chunk-2 interior, per-frame PSNR minimum), used by both
figures; a built-in assertion verifies every arm's dump holds the same clip at
that index (deterministic eval order). The bleeding grid intentionally shows
frames f1–f4 of the same clip — its subject is within-chunk dynamics, not the
failure/repair pairing. After regenerating, eyeball f5 once: if the expression
at f5 is unremarkable for this clip, pick another frame by changing
`QUAL_FRAME` in ONE place (both figures follow automatically). Diversity
across identities belongs in a supplement grid, not in these two figures.

---

## C. Things CVPR reviewers will expect (completeness checklist)

1. **True zero-shot baseline (A1)** — the single most attackable number right
   now; a reviewer rerunning stock Wan gets ~28 dB at 128², not 23.
2. **rFVD (reconstruction FVD)** — standard for video VAE papers; PSNR/SSIM/
   LPIPS alone is light. Compute from the final_eval dumps (gt vs rec, I3D
   features); the dumps already contain everything needed.
3. **Multi-seed error bars** — covered by the §A0 final wave (seed 42+43 for
   E1c/E1d/E11a/combo with per-seed warm-start sources); report mean ±
   half-range in the main table, state single-seed for the rest.
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
   runs (E9a/E9b) with a clean-wave 128 point; §A0b fixes this at the
   final-wave budget (or footnote the provenance if the figure is dropped).
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
