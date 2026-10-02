# Ghosting metrics from clean eval dumps

FG mask = not near-white in GT (thresh 0.96); cosine on masked tensors (bg zeroed).

| arm | xRec full | xGT full | gap full | xRec FG | xGT FG | gap FG | fg frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1a | 0.9782 | 0.9779 | +0.0003 | 0.9493 | 0.9480 | +0.0012 | 0.621 |
| E1b | 0.9784 | 0.9779 | +0.0005 | 0.9493 | 0.9480 | +0.0012 | 0.621 |
| E1c | 0.9791 | 0.9779 | +0.0012 | 0.9514 | 0.9480 | +0.0034 | 0.621 |
| E1d | 0.9804 | 0.9779 | +0.0025 | 0.9543 | 0.9480 | +0.0062 | 0.621 |
| E11a | 0.9792 | 0.9779 | +0.0013 | 0.9522 | 0.9480 | +0.0041 | 0.621 |
| E11b | 0.9793 | 0.9779 | +0.0014 | 0.9517 | 0.9480 | +0.0036 | 0.621 |
| combo | 0.9790 | 0.9779 | +0.0011 | 0.9512 | 0.9480 | +0.0031 | 0.621 |
| E3d | 0.9785 | 0.9779 | +0.0006 | 0.9499 | 0.9480 | +0.0019 | 0.621 |
| E2b | 0.9785 | 0.9779 | +0.0006 | 0.9497 | 0.9480 | +0.0017 | 0.621 |

## Takeaway

- E1d FG gap=+0.0062 vs E1c FG gap=+0.0034
- E1d full gap=+0.0025 vs E1a=+0.0003
- **Decision:** FG cosine still does not cleanly separate joint vs fused-TC-off; drop absolute XView from the main claim; keep qualitative + Bleed-W.
