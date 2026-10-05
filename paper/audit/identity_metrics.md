# Identity-embedding metrics (FaceNet VGGFace2, eval-only, from 170-ep final dumps)

Crops detected on GT, identical crop applied to rec. 10 val clips x 9 frames.

| arm | id sim (rec vs GT) | cross-view sim (rec) | cross-view sim (GT) | gap |
|---|---|---|---|---|
| zero-shot per-view (TC on) | 0.8894 | 0.9000 | 0.9643 | +0.0642 |
| per-view TC on | 0.8928 | 0.9038 | 0.9643 | +0.0605 |
| per-view TC off | 0.9708 | 0.9393 | 0.9643 | +0.0249 |
| fused TC off | 0.9235 | 0.9156 | 0.9643 | +0.0487 |
| fused TC on (16-ch) | 0.7557 | 0.7885 | 0.9643 | +0.1757 |
| fused TC on, 32-ch | 0.8210 | 0.8322 | 0.9643 | +0.1321 |
| fused TC on, 64-ch | 0.8301 | 0.8289 | 0.9643 | +0.1354 |
| 32-ch + diff-loss | 0.8462 | 0.8532 | 0.9643 | +0.1110 |
| all tweaks | 0.8488 | 0.8629 | 0.9643 | +0.1014 |

id sim: cosine similarity of face embeddings, reconstruction vs GT (1 = identity
perfectly preserved). cross-view gap: how much more the two decoded views disagree
about identity than the two real camera views do (positive = identity drift added
by the codec).
