# Bleed ratios, final wave (extracted from eval_metrics.jsonl, last val eval)

Bleed-W = bleed_ratio_within (motion within chunks, 1.0 = GT motion);
Bleed-A = bleed_ratio_across (across chunk boundaries).

| arm | Bleed-W | Bleed-A | from epoch |
|---|---:|---:|---:|
| zero-shot TC on (@0) | 0.969 | 1.006 | None |
| E1a per-view TC off (@170) | 0.992 | 1.010 | None |
| E1b per-view TC on (@170) | 0.942 | 0.959 | None |
| E1c fused TC off (@170) | 0.972 | 0.987 | None |
| E1d fused TC on (@170) | 0.910 | 0.934 | None |
| E4h diff-loss (@170) | 0.950 | 0.937 | None |
| E5b unfreeze-enc (@170) | 0.895 | 0.903 | None |
| E1d fused TC on (@300) | 0.920 | 0.943 | None |
| E11a 32-ch (@300) | 0.941 | 0.949 | None |
| E11b 64-ch (@300) | 0.947 | 0.952 | None |
| combo w32+diff (@300) | 0.945 | 0.958 | None |
| E_best all tweaks (@300) | 0.928 | 0.908 | None |
| ctrl per-view 32-ch (@300) | 0.948 | 0.964 | None |
| ctrl per-view 64-ch (@300) | 0.946 | 0.963 | None |
| E1b per-view TC on (@300) | 0.947 | 0.963 | None |
