# Identity-embedding metrics (FaceNet VGGFace2, eval-only, from 300-ep final dumps (capacity-table budget))

Crops detected on GT, identical crop applied to rec. 10 val clips x 9 frames.

| arm | id sim (rec vs GT) | cross-view sim (rec) | cross-view sim (GT) | gap |
|---|---|---|---|---|
| per-view TC on (16-ch) | 0.8923 | 0.8994 | 0.9643 | +0.0648 |
| per-view TC on, 32-ch (ctrl) | 0.8918 | 0.8933 | 0.9643 | +0.0709 |
| per-view TC on, 64-ch (ctrl) | 0.8904 | 0.8944 | 0.9643 | +0.0699 |
| fused TC on (16-ch) | 0.7465 | 0.7773 | 0.9643 | +0.1870 |
| fused TC on, 32-ch | 0.8364 | 0.8393 | 0.9643 | +0.1250 |
| fused TC on, 64-ch | 0.8561 | 0.8555 | 0.9643 | +0.1088 |
| 32-ch + diff-loss | 0.8599 | 0.8622 | 0.9643 | +0.1021 |
| all tweaks | 0.8653 | 0.8797 | 0.9643 | +0.0846 |

id sim: cosine similarity of face embeddings, reconstruction vs GT (1 = identity
perfectly preserved). cross-view gap: how much more the two decoded views disagree
about identity than the two real camera views do (positive = identity drift added
by the codec).
