# Ghosting metrics from local final-wave eval dumps (2026-10-07)

Computed offline from `final_eval_dump_val.pt` (gt/rec uint8).
@170 = Table-1 budget, @300 = capacity-table budget (never mix).
FG mask = not near-white in GT (thresh 0.96), unioned across views.
LPIPS = mean Alex-LPIPS between the two views over time (lower = more similar / more ghosting).

| arm | xRec full | xGT full | gap full | xRec FG | xGT FG | gap FG | LPIPS rec | LPIPS GT | LPIPS gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1a@170 | 0.9777 | 0.9774 | 0.0003 | 0.9490 | 0.9481 | 0.0009 | 0.096 | 0.097 | -0.001 |
| E1b@170 | 0.9780 | 0.9774 | 0.0006 | 0.9496 | 0.9481 | 0.0015 | 0.096 | 0.097 | -0.001 |
| E1c@170 | 0.9779 | 0.9774 | 0.0005 | 0.9498 | 0.9481 | 0.0017 | 0.093 | 0.097 | -0.004 |
| E1d@170 | 0.9796 | 0.9774 | 0.0022 | 0.9538 | 0.9481 | 0.0057 | 0.090 | 0.097 | -0.007 |
| zeroshot_tcON@0 | 0.9775 | 0.9774 | 0.0001 | 0.9479 | 0.9481 | -0.0002 | 0.099 | 0.097 | 0.002 |
| E1d@300 | 0.9795 | 0.9774 | 0.0021 | 0.9535 | 0.9481 | 0.0055 | 0.092 | 0.097 | -0.005 |
| E11a@300 | 0.9783 | 0.9774 | 0.0010 | 0.9497 | 0.9481 | 0.0016 | 0.094 | 0.097 | -0.003 |
| E11b@300 | 0.9781 | 0.9774 | 0.0007 | 0.9499 | 0.9481 | 0.0018 | 0.095 | 0.097 | -0.002 |
| combo@300 | 0.9781 | 0.9774 | 0.0008 | 0.9499 | 0.9481 | 0.0018 | 0.095 | 0.097 | -0.002 |
| Ebest@300 | 0.9787 | 0.9774 | 0.0013 | 0.9511 | 0.9481 | 0.0030 | 0.096 | 0.097 | -0.001 |
| ctrl_pv32@300 | 0.9778 | 0.9774 | 0.0005 | 0.9493 | 0.9481 | 0.0013 | 0.096 | 0.097 | -0.001 |
| ctrl_pv64@300 | 0.9779 | 0.9774 | 0.0006 | 0.9495 | 0.9481 | 0.0014 | 0.096 | 0.097 | -0.001 |

## Takeaway

- Whole-frame XView gaps stay tiny (same conclusion as clean jsonl).
- If FG gaps and/or LPIPS(rec views)−LPIPS(GT views) also stay small, drop absolute XView
  claims from the paper and keep ghosting qualitative + Bleed-W.
- If FG gap clearly separates joint (E1d) from single-axis (E1a/E1c), keep a FG-XView column.

- E1d FG gap=+0.0057 vs E1c FG gap=+0.0017
- E1d LPIPS gap=-0.007 vs E1c LPIPS gap=-0.004
- **Decision lean:** FG cosine still does not separate joint vs fused-TC-off; do not put absolute XView in the main claim. Prefer qualitative grids.
