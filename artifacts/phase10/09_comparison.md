# Phase 9 vs Phase 10 — Side-by-side comparison

## PF#1 — Per-collection balanced AUC

| Collection | Phase 9 AUC | Phase 10 AUC | Δ |
|---|---|---|---|
| c=212 | 0.9428 | 0.9600 | +0.0172 |
| c=70 | 0.5480 | 0.8668 | +0.3188 |
| c=249 | 0.7776 | 0.8652 | +0.0876 |

## PF#2 — Counterfactual ΔP per shortcut

| Shortcut | Phase 9 mean ΔP | Phase 10 mean ΔP | Δ-of-Δ |
|---|---|---|---|
| vignette | +0.0070 | -0.0081 | -0.0150 |
| ruler | +0.0030 | -0.0007 | -0.0036 |
| hair | -0.0662 | -0.0089 | +0.0573 |
| colorcast | +0.1039 | +0.0144 | -0.0895 |

## PF#3 — Original vs all-removed AUC per collection

| Collection | P9 orig | P9 all-rm | P10 orig | P10 all-rm |
|---|---|---|---|---|
| c=212 | 0.964 | 0.871 | 0.962 | 0.910 |
| c=70 | 0.530 | 0.723 | 0.841 | 0.780 |
| c=249 | 0.770 | 0.706 | 0.850 | 0.809 |

## H6-H10 outcomes

- **H6**: SUPPORTED  ({'supported': True, 'post_auc': 0.9368798187373264, 'delta': 0.13517439011003796})
- **H7**: SUPPORTED  ({'supported': True, 'p10_mean': 0.01435458977376261, 'p9_mean': 0.10386518064516129, 'reduction': 0.8617959388834766})
- **H8**: SUPPORTED  ({'supported': True, 'p10_auc': 0.8668, 'lo': 0.7854385817664031, 'hi': 0.9358974358974359})
- **H9**: SUPPORTED  ({'supported': True, 'gap': 0.09319999999999995})
- **H10**: SUPPORTED  ({'supported': True, 'recall': 0.71})
