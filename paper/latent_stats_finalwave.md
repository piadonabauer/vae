# Latent statistics (sampleability proxy), final wave @300

Encoded the 10 fixed val clips; posterior stats per arm (EMA weights).
KL/dim = mean per-dim KL to N(0,1). Active ch = fraction of channels
with std(mu) > 0.1 (collapse proxy). Healthy-for-sampling: mu-std near 1,
sigma neither ~0 nor >1, no dead channels.

| arm | ch | KL/dim | std(mu) | mean sigma | active ch | SNR>2 ch | weights |
|---|---:|---:|---:|---:|---:|---:|---|
| pretrained zero-shot (per-view, TC on) | 16 | 9.521 | 1.841 | 0.000 | 100% | 100% | pretrained |
| per-view 16-ch (@300) | 16 | 9.509 | 1.840 | 0.000 | 100% | 100% | ema |
| per-view 32-ch ctrl (@300) | 32 | 4.757 | 1.308 | 0.490 | 50% | 50% | ema |
| per-view 64-ch ctrl (@300) | 64 | 2.379 | 0.927 | 0.736 | 25% | 25% | ema |
| fused 16-ch (@300) | 16 | 9.625 | 0.870 | 0.002 | 100% | 100% | ema |
| fused 32-ch (@300) | 32 | 5.535 | 0.581 | 0.459 | 50% | 50% | ema |
| fused 64-ch (@300) | 64 | 2.499 | 0.383 | 0.719 | 25% | 25% | ema |
| fused 32-ch + diff-loss (@300) | 32 | 5.537 | 0.581 | 0.457 | 50% | 50% | ema |
| fused all tweaks (@300) | 64 | 2.508 | 0.379 | 0.729 | 25% | 25% | ema |
