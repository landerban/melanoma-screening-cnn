# Phase 13 — Lesion-Aware Jitter + LACN-Consistency Fine-Tune (Design)

Post-defense follow-up. Designs a single training-time intervention
that addresses both limitations identified in PRs #4 and #5:

- **Aggregate AUC trade-off** (Phase 7a 0.9513 → Phase 10 0.9369,
  -0.014). Diagnosis: Phase 10's global ColorJitter(hue=0.10) damages
  lesion-internal color signal alongside the background acquisition
  cast. IG attribution (PR #5) supports this — Phase 10 reduces bg/fg
  attribution ratio but the median is unchanged, consistent with
  global jitter applying everywhere.
- **OOD AUC ceiling at 0.85** (Phase 12 PR #4, every LACN variant).
  Diagnosis: the Phase 6/10 base was not trained to be LACN-invariant,
  so applying LACN at test time shifts probabilities into a region the
  model was never optimized over.

The design here combines two interventions in a single 5-epoch fine-
tune from the Phase 6 base ckpt, allowing a head-to-head comparison
against Phase 10 (same epochs, same lr, same loss, same sampler).

---

## 1. Design

### 1.1 D1 — Lesion-Aware Jitter (LAJ)

Replace Phase 10's global `ColorJitter(hue=0.10)` with a *spatially-
selective* jitter:

- Per-image Otsu lesion mask (re-uses the Phase 9 / Step 3i pipeline).
- Sample a random hue offset ∈ [-0.15 × 180, +0.15 × 180] on the
  OpenCV Hue scale (≈ ±54° on the 0-360° hue circle).
- Apply the offset only to background pixels (outside the mask).
  Lesion-interior hue is *preserved*.

Rationale: ABCDE's C-axis (color diversity inside the lesion) is the
single most discriminative dermoscopy cue (7-point checklist gives
"atypical pigment network" and "blue-gray structures" the highest
weights). Phase 10's global jitter dilutes this signal alongside the
background cast. LAJ keeps the dilution where the shortcut lives
(background) and the signal where the diagnosis lives (lesion).

### 1.2 D2 — LACN-consistency regularization

For every training sample, generate a partner image via Phase 12's
background-only Lab-mean shift (LACN). Both copies share the same
geometric augmentation, so the only difference between them is the
*color-space* transformation. Add an MSE consistency term to the
loss:

```
L_total =      L_focal(x_train)               # LAJ-augmented original
        + 0.5 *L_focal(x_lacn)               # LACN-applied partner
        + 0.5 *MSE(sigmoid(f(x_train)),
                   sigmoid(f(x_lacn)))       # consistency term
```

Rationale: the consistency term penalises any output difference
between the original and the LACN-applied version of the same image.
The model is trained to be *invariant* to the exact test-time
intervention that Phase 12 (PR #4) showed otherwise drops OOD AUC by
0.05. Combined with the L_focal on the LACN sample, both copies
contribute to the discriminative loss, so the model is asked to be
correct *and* invariant.

### 1.3 Why combined (rather than D1 or D2 alone)

| Property | Phase 10 (hue) | D1 only (LAJ) | D2 only (consistency) | **D1+D2** |
|---|---|---|---|---|
| Aggregate AUC restored | partial | yes (lesion color preserved) | yes (no jitter to lesion) | **yes** |
| c=70 SIIM held at 0.87 | yes | yes | yes | **yes** |
| Color-cast ΔP suppressed | yes (86%) | yes (background only) | yes (LACN-invariant) | **yes** |
| OOD AUC ceiling broken | no | no | **yes** | **yes** |
| Loss term count | 1 | 1 | 2 (+1 reg) | 2 (+1 reg) |
| Forward passes / step | 1× | 1× | 2× | 2× |

The combination is the only configuration that addresses *both*
identified limitations without adding hyperparameters beyond what
D2 already requires.

## 2. Training configuration

| Field | Value |
|---|---|
| Base ckpt | `best_model.pth` (Phase 6, in-dist Test AUC 0.9513) |
| Epochs | 5 (matches Phase 10) |
| Optimizer | AdamW, lr = 1e-5, weight_decay from CFG |
| Loss | L_focal + 0.5·L_focal_LACN + 0.5·MSE_consistency |
| Sampler | `WeightedRandomSampler` (Phase 6/10 reuse) |
| Augmentation order | Geometric (random crop / flip / rot) → LAJ + LACN → ToTensor + Normalize |
| `bg_hue_strength` | 0.15 (≈ ±27° on OpenCV scale) |
| `lambda_consistency` | 0.5 |
| `lacn_focal_weight` | 0.5 |
| Hardware (intended) | Colab A100 (batch_size 96) |
| Hardware (fallback) | Local M3 MPS (batch_size 32) |
| Seed | 42 |

## 3. Pre-registered evaluation (run via `02_validate.py`)

Identical to Phase 10's H6-H10 protocol so results are directly
comparable.

| Hypothesis | Threshold | What it tests |
|---|---|---|
| H_aggr | Test AUC ∈ [0.945, 0.955] | Aggregate AUC restored above Phase 10's 0.9369 |
| H_cf  | Color-cast ΔP magnitude ≤ Phase 10's +0.014 | Shortcut still removed |
| H_c70 | c=70 balanced AUC ≥ Phase 10's 0.867 | Phase 10's signature result preserved |
| H_ood | OOD AUC (no LACN) ≥ 0.85 | OOD performance maintained |
| H_ood_lacn | **OOD AUC (with LACN applied) ≥ 0.85** | **D2 hypothesis: model is now LACN-invariant on OOD** |

Falsification: if H_ood_lacn fails, the consistency term did not
generalize across the in-dist/OOD shift; Phase 13 would be a
documented null result for OOD AUC and a confirmatory result for
the other four hypotheses.

## 4. Implementation locations

- Training: `scripts/phase13/01_finetune_laj_consistency.py`
- Validation: `scripts/phase13/02_validate.py`
- Colab pipeline: `scripts/phase13/colab_phase13.ipynb` (Drive backup
  + keepalive)

## 5. Why Colab, not local

Local M3 MPS smoke + 2-epoch trend on a 5000-image subset confirmed
the code paths work end-to-end (loss components in the expected
range, val AUC moving). However the *full* training set requires
the c=212 / c=70 / c=249 ISIC image downloads (~50 GB) that we did
not pull locally for Phase 9-12 (those phases only needed the 300-
image analytical sample). Re-using Phase 10's Colab workflow is
significantly cheaper than a local download + train cycle.

## 6. Colab session safety

The Phase 10 first run lost its checkpoint to a session timeout.
Phase 13 mitigates this:

- `--save-every-epoch` writes a checkpoint after every epoch
  containing both the current model state and the best-so-far state.
- `--drive-backup-dir` additionally copies the epoch checkpoint and
  the training log to Drive after every save.
- The notebook starts a daemon `keepalive` thread that performs a
  small computation every 60 seconds; this prevents Colab's 90-min
  idle disconnect.
- The notebook caches the training data on Drive (`rsync -a`) so a
  subsequent session does not re-download.

If the session dies between epochs, the latest epoch ckpt is on
Drive, and notebook §13 documents how to resume.

## 7. Acceptance for PR merge

- All five hypotheses' values are reported in `02_validate.log`.
- A comparison table against Phase 6 and Phase 10 is included in the
  PR body and committed to `docs/phase13/02_results.md` (populated
  after the Colab run completes).
- The Phase 13 checkpoint is small enough to commit
  (`best_model_laj_consistency.pth`, ~18 MB).
