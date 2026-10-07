#!/usr/bin/env python3
"""Extract Bleed-W / Bleed-A for the final-wave arms from eval_metrics.jsonl.

train.py logs bleed_ratio_within / bleed_ratio_across at every val eval; the
LAST val record of a finished run corresponds to the final evaluation. No
recompute needed — this just collects what the runs already measured.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DUMP_ROOT = Path("/home/coder/vae/Open-Sora/outputs")

ARMS = {
    "zero-shot TC on (@0)": "paper_E1z_perview_zeroshot_tcON__job1791026450_t50",
    "E1a per-view TC off (@170)": "paper_E1a_perview_tcF__job1791135751_t1",
    "E1b per-view TC on (@170)": "paper_E1b_perview_tcT__job1791027540_t2",
    "E1c fused TC off (@170)": "paper_E1c_fused_tcF__job1791042863_t3",
    "E1d fused TC on (@170)": "paper_E1d_fused_tcT__job1791058355_t4",
    "E4h diff-loss (@170)": "paper_E4h_temporal_diff_loss__job1791152780_t24",
    "E5b unfreeze-enc (@170)": "paper_E5b_fused_tcT_unfreeze_enc__job1791325522_t5",
    "E1d fused TC on (@300)": "paper_E1d_fused_tcT__job1791225734_t4",
    "E11a 32-ch (@300)": "paper_E11a_fused_tcT_widen32__job1791237549_t8",
    "E11b 64-ch (@300)": "paper_E11b_fused_tcT_widen64__job1791189563_t9",
    "combo w32+diff (@300)": "paper_E_combo_diffLoss_widen32__job1791201374_t36",
    "E_best all tweaks (@300)": "paper_Ebest_allcombined__job1791213193_t52",
    "ctrl per-view 32-ch (@300)": "paper_Ectrl_perview_tcT_widen32__job1791271134_t53",
    "ctrl per-view 64-ch (@300)": "paper_Ectrl_perview_tcT_widen64__job1791298425_t54",
    "E1b per-view TC on (@300)": "paper_E1b_perview_tcT__job1791249364_t2",
}


def last_val_bleed(jobdir: Path):
    f = jobdir / "eval_metrics.jsonl"
    if not f.exists():
        return None
    last = None
    with open(f) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = rec.get("metrics", rec)
            if "bleed_ratio_within" not in json.dumps(rec):
                continue
            split = rec.get("split", rec.get("kind", ""))
            if "train" in str(split) and "val" not in str(split):
                continue
            last = rec
    if last is None:
        return None

    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None

    return {
        "bleed_within": find(last, "bleed_ratio_within"),
        "bleed_across": find(last, "bleed_ratio_across"),
        "kind": last.get("kind"),
        "epoch": last.get("epoch"),
    }


def main():
    results = {}
    lines = [
        "# Bleed ratios, final wave (extracted from eval_metrics.jsonl, last val eval)",
        "",
        "Bleed-W = bleed_ratio_within (motion within chunks, 1.0 = GT motion);",
        "Bleed-A = bleed_ratio_across (across chunk boundaries).",
        "",
        "| arm | Bleed-W | Bleed-A | from epoch |",
        "|---|---:|---:|---:|",
    ]
    for name, d in ARMS.items():
        r = last_val_bleed(DUMP_ROOT / d)
        results[name] = r
        if r and r["bleed_within"] is not None:
            lines.append(f"| {name} | {r['bleed_within']:.3f} | "
                         f"{(r['bleed_across'] if r['bleed_across'] is not None else float('nan')):.3f} "
                         f"| {r.get('epoch', '—')} |")
            print(f"{name:34s} W={r['bleed_within']:.3f} "
                  f"A={r['bleed_across'] if r['bleed_across'] is not None else float('nan'):.3f} "
                  f"(epoch {r.get('epoch')}, kind {r.get('kind')})")
        else:
            lines.append(f"| {name} | — | — | — |")
            print(f"{name:34s} NO BLEED RECORD")
    (OUT_DIR / "bleed_finalwave.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "bleed_finalwave.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT_DIR / "bleed_finalwave.md")


if __name__ == "__main__":
    main()
