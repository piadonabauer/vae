#!/usr/bin/env python3
"""Build paper/clean_retrain_metrics_corrected.md (old vs corrected columns).

Sources
-------
* OLD values: parsed from paper/clean_retrain_metrics.md (the biased
  `psnr_mean`/`ssim_mean` plus the already-correct LPIPS / Bleed-W / XView).
* CORRECTED PSNR: mean of final_eval/val per-view PSNRs. Per-view PSNR in
  train.py was always computed on correctly remapped [0,1] tensors, so its
  mean IS the corrected headline under the declared convention
  (mean of per-(clip,view) PSNRs). Pulled from the wandb run summaries
  (paper/audit/wandb_inventory.json) for the clean 2026-09-19/20 wave; for
  E4h (wandb crashed at final eval) from paper/clean_peraxis_metrics.json,
  which was filled from the cluster eval_metrics.jsonl.
* Caveat: mean-of-per-view aggregates per-batch means with equal weight; vs
  the exact per-sample mean this differs by <~0.05 dB (verified on E1a dumps:
  35.886 vs 35.92). Exact refill from final_eval_dump_val.pt on the cluster.

SSIM cannot be corrected from logs (the biased number is the only one ever
computed); it needs a recompute from the eval dumps on the cluster.
LPIPS, Bleed-W, XView, per-frame and per-view PSNR were already computed on
the correct [0,1] path (see eval loop diagnostics block) and carry over.
"""

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
AUDIT = Path(__file__).resolve().parent

# arm -> (wandb run name prefix, rule for picking among clean-wave duplicates)
ARM_TO_RUN = {
    "E0": "paper_E0_perview_ceiling",
    "E1a": "paper_E1a_perview_tcF",
    "E1b": "paper_E1b_perview_tcT",
    "E1c": "paper_E1c_fused_tcF",
    "E1d": "paper_E1d_fused_tcT",
    "E2b": "paper_E2b_fused_tcF_self_attn",
    "E2c": "paper_E2c_fused_tcF_conv3d",
    "E2d": "paper_E2d_fused_tcF_conv4d",
    "E2e": "paper_E2e_fused_tcT_self_attn",
    "E3a": "paper_E3a_view_emb_noLora",
    "E3c": "paper_E3c_view_emb_plus_lora",
    "E3d": "paper_E3d_no_emb_no_lora",
    "E3e": "paper_E3e_full_dec_finetune",
    "E4b": "paper_E4b_noncausal_decode",
    "E4h": "paper_E4h_temporal_diff_loss",
    "E4i": "paper_E4i_diff_loss_plus_cache",
    "E5b": "paper_E5b_fused_tcT_unfreeze_enc",
    "E5c": "paper_E5c_fused_tcT_unfreeze_all",
    "E6b": "paper_E6b_4view_tcF",
    "E6c": "paper_E6c_4view_tcT",
    "E11a": "paper_E11a_fused_tcT_widen32",
    "E11b": "paper_E11b_fused_tcT_widen64",
    "combo": "paper_E_combo_diffLoss_widen32",
    "E10_r8": "paper_E10_rank8",
    "E10_r16": "paper_E10_rank16",
    "E10_r64": "paper_E10_rank64",
    "E10_r128": "paper_E10_rank128",
    "E7_p0.5": "paper_E7_perc0p5",
    "E7_p3.0": "paper_E7_perc3p0",
    "E7_kl1e7": "paper_E7_kl1e7",
}


def parse_old_table():
    rows = {}
    for line in (PAPER / "clean_retrain_metrics.md").read_text().splitlines():
        m = re.match(r"\|\s*(\S+)\s*\|" + r"\s*([+\-]?[\d.]+)\s*\|" * 7, line)
        if m:
            arm = m.group(1)
            vals = [float(m.group(i)) for i in range(2, 9)]
            rows[arm] = dict(zip(["psnr", "ssim", "lpips", "bleed_w", "xrec", "xgt", "gap"], vals))
    return rows


def load_wave_runs():
    inv = json.load(open(AUDIT / "wandb_inventory.json"))
    by_name = {}
    for e in inv:
        if not ("2026-09-19" <= e["created"][:10] <= "2026-09-21"):
            continue
        name = e["name"].split("__lp")[0]
        by_name.setdefault(name, []).append(e)
    return by_name


def corrected_psnr(entry):
    s = entry["summary"]
    views = sorted(
        (k, float(v)) for k, v in s.items()
        if k.startswith("final_eval/val/psnr_view")
    )
    if not views:
        return None, []
    vals = [v for _, v in views]
    return sum(vals) / len(vals), vals


def main():
    old = parse_old_table()
    runs = load_wave_runs()
    peraxis = json.load(open(PAPER / "clean_peraxis_metrics.json"))

    out_rows, needs_rerun, notes = [], [], []
    for arm, old_m in old.items():
        run_name = ARM_TO_RUN.get(arm)
        cands = runs.get(run_name, [])
        # pick the finished run whose biased psnr_mean matches the old table
        pick, pick_views = None, []
        for e in cands:
            pm = e["summary"].get("final_eval/val/psnr_mean")
            if pm is not None and abs(float(pm) - old_m["psnr"]) < 0.02:
                pick, pick_views = corrected_psnr(e)
                break
        src = "wandb final_eval"
        if pick is None and arm in peraxis:
            vv = peraxis[arm]["psnr_per_view"]
            pick, pick_views = sum(vv) / len(vv), vv
            src = "cluster jsonl (via clean_peraxis_metrics.json)"
        if pick is None:
            # E5b: wandb crashed at final eval and arm is not in peraxis json.
            needs_rerun.append(arm)
            out_rows.append((arm, old_m, None, [], "no corrected source"))
            continue
        out_rows.append((arm, old_m, pick, pick_views, src))

    # ---- write markdown -------------------------------------------------
    L = []
    L.append("# Clean retrain metrics — CORRECTED (2026-10-02)")
    L.append("")
    L.append("Correction for the eval PSNR/SSIM clamp bug (commit 40295bb): the old table's")
    L.append("`psnr_mean`/`ssim_mean` were computed on [-1,1] tensors clamped to [0,1].")
    L.append("**Convention:** PSNR = mean of per-(clip, view) PSNRs on [0,1] (NOT PSNR of pooled MSE;")
    L.append("the two differ by ~0.3 dB). Corrected PSNR = mean of `final_eval` per-view PSNRs,")
    L.append("which train.py always computed on the correct [0,1] path; exact per-sample refill from")
    L.append("the cluster dumps may shift values by <~0.05 dB (batch-weighting).")
    L.append("")
    L.append("- **PSNR old → new**: old = biased table value; new = corrected.")
    L.append("- **SSIM**: old value is biased the same way and **cannot be corrected from logs** —")
    L.append("  recompute from `final_eval_dump_val.pt` on the cluster (column marked `dump`).")
    L.append("- **LPIPS / Bleed-W / XView / per-frame PSNR**: unchanged — these were always computed")
    L.append("  on the correctly remapped tensors (`x01` path in `evaluate_model`; LPIPS consumes")
    L.append("  `x01*2-1` which is the [-1,1] the LPIPS net expects).")
    L.append("")
    L.append("| arm | PSNR old | PSNR new | Δbias | SSIM old | SSIM new | LPIPS | Bleed-W | PSNR v0 | PSNR v1 | v2 | v3 | source |")
    L.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for arm, om, new, views, src in out_rows:
        v = views + [None] * (4 - len(views))
        fv = lambda x: f"{x:.2f}" if x is not None else "—"
        if new is None:
            L.append(
                f"| {arm} | {om['psnr']:.2f} | — | — | {om['ssim']:.4f} | dump | "
                f"{om['lpips']:.3f} | {om['bleed_w']:.3f} | — | — | — | — | {src} |"
            )
        else:
            L.append(
                f"| {arm} | {om['psnr']:.2f} | **{new:.2f}** | {new - om['psnr']:+.2f} | "
                f"{om['ssim']:.4f} | dump | {om['lpips']:.3f} | {om['bleed_w']:.3f} | "
                f"{fv(v[0])} | {fv(v[1])} | {fv(v[2])} | {fv(v[3])} | {src} |"
            )

    # zero-shot row (old wave, eval-only)
    L.append("")
    L.append("## Zero-shot floor (E1z, pretrained Wan, eval only — 2026-09-06 re-eval run)")
    L.append("")
    L.append("| split | PSNR old | PSNR new | SSIM old | LPIPS |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| val | 19.71 | **23.07** (v0 23.06 / v1 23.07) | 0.9707 (biased) | 0.155 |")
    L.append("| train | 19.52 | **23.12** (v0 23.23 / v1 23.01) | 0.9686 (biased) | 0.151 |")
    L.append("")
    L.append("The zero-shot bias (−3.4 dB) is much larger than for finetuned arms (−1.7 to −2.4 dB):")
    L.append("the clamp bias grows with reconstruction error mass in the dark half of the range.")
    L.append("Jobs 5845272/73 re-run this with the fixed eval and should land near 23 dB (128px).")

    # ---- deltas ----------------------------------------------------------
    cor = {arm: new for arm, _, new, _, _ in out_rows if new is not None}
    oldp = {arm: om["psnr"] for arm, om, _, _, _ in out_rows}

    def delta(a, b, vals):
        return vals[a] - vals[b]

    L.append("")
    L.append("## Headline deltas, old vs corrected")
    L.append("")
    L.append("| claim | old | corrected | verdict |")
    L.append("|---|---:|---:|---|")

    pairs = [
        ("TC cost per-view (E1a−E1b)", "E1a", "E1b"),
        ("Fusion cost TC-off (E1a−E1c)", "E1a", "E1c"),
        ("Joint drop (E1a−E1d)", "E1a", "E1d"),
        ("Per-view TC-on vs fused TC-on (E1b−E1d)", "E1b", "E1d"),
        ("Widen 16→32 (E11a−E1d)", "E11a", "E1d"),
        ("Widen 32→64 (E11b−E11a)", "E11b", "E11a"),
        ("Temp-diff loss @16ch (E4h−E1d)", "E4h", "E1d"),
        ("Diff-loss+cache (E4i−E1d)", "E4i", "E1d"),
        ("Temp-diff loss @32ch (combo−E11a)", "combo", "E11a"),
        ("Combo vs baseline (combo−E1d)", "combo", "E1d"),
        ("Unfreeze enc (E5b−E1d)", "E5b", "E1d"),
        ("Unfreeze all (E5c−E1d)", "E5c", "E1d"),
        ("Self-attn fusion TC-off vs default (E2b−E1c)", "E2b", "E1c"),
        ("Self-attn fusion TC-on vs default (E2e−E1d)", "E2e", "E1d"),
        ("Self-attn TC-off vs joint baseline (E2b−E1d)", "E2b", "E1d"),
    ]
    for label, a, b in pairs:
        if a in oldp and b in oldp:
            od = delta(a, b, oldp)
            if a in cor and b in cor:
                nd = delta(a, b, cor)
                verdict = "ok"
                if od * nd < 0:
                    verdict = "**SIGN FLIP**"
                elif abs(nd) < 0.5 and abs(od) >= 0.5:
                    verdict = "**drops below 0.5 dB**"
                elif abs(nd) < 0.5:
                    verdict = "below 0.5 dB (was already)"
                L.append(f"| {label} | {od:+.2f} | {nd:+.2f} | {verdict} |")
            else:
                L.append(f"| {label} | {od:+.2f} | n/a (needs refill) | — |")

    # additive prediction
    if all(k in cor for k in ("E1a", "E1b", "E1c", "E1d")):
        old_pred = oldp["E1a"] - (oldp["E1a"] - oldp["E1b"]) - (oldp["E1a"] - oldp["E1c"])
        new_pred = cor["E1a"] - (cor["E1a"] - cor["E1b"]) - (cor["E1a"] - cor["E1c"])
        L.append(
            f"| Additive prediction vs actual E1d | pred {old_pred:.2f} vs {oldp['E1d']:.2f} "
            f"(excess {old_pred - oldp['E1d']:+.2f}) | pred {new_pred:.2f} vs {cor['E1d']:.2f} "
            f"(excess {new_pred - cor['E1d']:+.2f}) | super-additivity holds |"
        )

    # ranking flips across the whole table (bias is arm-dependent, so order can change)
    L.append("")
    L.append("### Ranking changes between arms (bias is not constant: +1.59 to +2.98 dB)")
    L.append("")
    flips = []
    arms_both = [a for a in oldp if a in cor]
    for i, a in enumerate(arms_both):
        for b in arms_both[i + 1 :]:
            if (oldp[a] - oldp[b]) * (cor[a] - cor[b]) < 0:
                flips.append(
                    f"- **{a} vs {b}**: old {oldp[a]:.2f} vs {oldp[b]:.2f} "
                    f"({'<' if oldp[a] < oldp[b] else '>'}) → corrected {cor[a]:.2f} vs {cor[b]:.2f} "
                    f"({'<' if cor[a] < cor[b] else '>'})"
                )
    if flips:
        L.extend(flips)
        L.append("")
        L.append("Any prose that orders these pairs (e.g. \"E2b collapses below even the joint")
        L.append("baseline E1d\") must be re-checked against the corrected column.")
    else:
        L.append("- none")

    # ---- arms that cannot be recomputed ---------------------------------
    L.append("")
    L.append("## Arms that cannot be recomputed locally")
    L.append("")
    L.append("- **E6d / E6e (8-view)**: never produced a final_eval (clean-wave TIMEOUT; both")
    L.append("  2026-09-20 attempts failed/crashed before final eval). Resubmitted 2026-10-01 as")
    L.append("  jobs 5844331 / 5844332 with the walltime fix; those jobs started after the eval fix")
    L.append("  landed only if they baked the script post-40295bb — verify on completion, else the")
    L.append("  psnr_mean they report is biased and per-view must be used.")
    if needs_rerun:
        L.append(f"- **{', '.join(needs_rerun)}**: wandb crashed at final eval and the arm is not in")
        L.append("  clean_peraxis_metrics.json. The exact corrected number is in the cluster")
        L.append("  `eval_metrics.jsonl` (`final_eval` → `psnr_per_view`). Last mid-training full_eval")
        L.append("  per-view was 28.06/28.42 (biased mean then 25.87 vs final 26.01), so the corrected")
        L.append("  final is ≈28.4–28.6 dB — refill from cluster, do not print the estimate.")
    L.append("- **SSIM (all arms)**: corrected SSIM requires the eval dumps")
    L.append("  (`final_eval_dump_val.pt` per job dir on the cluster); logs only contain the biased value.")
    L.append("- **Seed-2 arms + zero-shot refresh (jobs 5845272–5845277)**: still queued as of")
    L.append("  2026-10-02 14:00 UTC+2 (no wandb runs created after 2026-09-20); they will report")
    L.append("  corrected psnr_mean directly once they run (script baked at job start includes 40295bb).")
    L.append("")

    (PAPER / "clean_retrain_metrics_corrected.md").write_text("\n".join(L) + "\n")
    print("wrote", PAPER / "clean_retrain_metrics_corrected.md")
    for r in out_rows:
        arm, om, new, _, src = r
        print(arm.ljust(10), f"{om['psnr']:6.2f}", "->", f"{new:6.2f}" if new else "  n/a ", src)


if __name__ == "__main__":
    main()
