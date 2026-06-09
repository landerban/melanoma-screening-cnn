# Phase 12 / Part 2 — LACN OOD Deployment Evaluation

Companion to `01_preregistration.md`. Reports honest, leakage-controlled
evaluation of LACN as a test-time intervention on the PAD-UFES-20 OOD
cohort, including two extensions designed to recover the cost
identified by the baseline LACN run.

---

## Abstract

We evaluated LACN (Lesion-Aware Color Normalization, Phase 12) as a
deployment-ready test-time intervention on the PAD-UFES-20 out-of-
distribution cohort (N=200, 50% malignant). A stratified
calibration/test split (100/100, seed=42) was used to avoid
threshold-fitting leakage. Three variants were tested: full Lab-mean
shift (baseline LACN), channel-selective Hue-only shift (motivated by
Phase 11's 93% Hue attribution), and partial-strength alpha
interpolation between original and full-Lab-LACN images.

All three variants produced AUC strictly below the no-LACN reference
(0.850). Lab-LACN preserved recall (+0.020) at the cost of specificity
(-0.080) and AUC (-0.052). Hue-only LACN was strictly dominated by
Lab-LACN on every metric, indicating the in-distribution channel
attribution does not transfer cleanly to OOD without an OOD-aware
target. The alpha sweep showed AUC monotonically decreasing in
intervention strength with an F1 sweet spot at alpha=0.75 marginally
above the no-LACN baseline.

The conclusion that LACN is a *promising candidate, not a validated
intervention* — drawn during the defense from in-distribution evidence
alone — is now empirically confirmed on OOD. LACN remains a viable
*recall-first* operating-point knob; it is not an AUC improvement.

---

## 1. Setup

- **OOD cohort**: PAD-UFES-20, N=200 (100 malignant, 100 benign;
  Phase 9 `01_select_sample.csv`).
- **Model**: Phase 6 best ckpt (`best_model.pth`, Test AUC 0.9513
  in-distribution).
- **Lesion masks**: Otsu segmentation on the L channel of Lab; 172/200
  pass the reliability heuristic (0.01 <= area_frac <= 0.60). Images
  with unreliable masks fall back to the original-image prediction.
- **Target reference**:
  - For Lab and alpha-sweep variants: Phase 9 analytical-sample mean
    Lab = `[155.44, 142.81, 135.46]`.
  - For Hue-only: circular mean Hue over the Phase 9 analytical sample
    = 359.08 deg.
- **Calibration / test split**: stratified, seed=42, 100/100, balanced
  positive/negative within each split. *Identical split is reused
  across all three variants* so head-to-head metrics are on the same
  test patients.

`TAU_BASELINE = 0.661` is the operating threshold reported by the
team's Phase 7c README and reused throughout Phase 9-12.
`TARGET_RECALL = 0.79` matches the Phase 9 OOD baseline recall
measured in `artifacts/phase9/11_ood_h4.log`.

## 2. Variants and protocol

### 2.1 Baseline LACN (Lab-mean shift, full strength)

Background pixels (outside Otsu mask) have their Lab-channel mean
shifted to the Phase 9 reference. Lesion-interior pixels are
preserved.

### 2.2 Hue-only LACN

Background pixels have their *circular* Hue mean shifted to the
Phase 9 reference; Saturation and Value are preserved. The shift
direction is chosen as the *shorter* arc on the 360-deg circle.

### 2.3 Partial-strength alpha sweep

For each alpha in {0.00, 0.25, 0.50, 0.75, 1.00}, the served image is
a per-pixel BGR blend:

    img_alpha = (1 - alpha) * img_orig + alpha * img_lacn_full

alpha=0 reduces to the Phase 9 baseline; alpha=1 reduces to §2.1.

### 2.4 Threshold recalibration

For each variant, the operating threshold is chosen on the *calibration
half* as the largest tau satisfying `recall(tau) >= 0.79`. The choice
is then *frozen* and evaluated on the held-out test half. This avoids
the leakage that would arise from sweeping tau on the test set.

## 3. Results

### 3.1 Lab-LACN baseline (Step 2, `02_lacn_ood_recalibration.py`)

`tau_LACN_Lab = 0.670` selected on calibration (recall 0.82, spec
0.62). On the held-out test (N=100):

| Mode                          | recall | spec  | F1    | AUC   |
|-------------------------------|--------|-------|-------|-------|
| (A) No LACN + tau=0.661       | 0.820  | 0.720 | 0.781 | 0.850 |
| (B) LACN + tau=0.661          | 0.860  | 0.600 | 0.761 | 0.798 |
| (C) LACN + tau_LACN_Lab       | 0.840  | 0.640 | 0.764 | 0.798 |

Recalibration moves the operating point but does not move the ROC
curve: AUC is fixed at 0.798 regardless of tau. **Partial win**:
recall preserved (+0.020), specificity loses 0.080, AUC loses 0.052.

### 3.2 Hue-only LACN (Step 3, `03_lacn_hue_only.py`)

`tau_LACN_Hue = 0.650` selected on calibration. On the same test
patients:

| Mode                          | recall | spec  | F1    | AUC   |
|-------------------------------|--------|-------|-------|-------|
| (A) No LACN  + tau=0.661      | 0.820  | 0.720 | 0.781 | 0.850 |
| (B) Hue LACN + tau=0.661      | 0.800  | 0.520 | 0.702 | 0.738 |
| (C) Hue LACN + tau_LACN_Hue   | 0.820  | 0.520 | 0.713 | 0.738 |

Hue-only LACN is **strictly dominated** by Lab-LACN on every metric
(recall -0.020, spec -0.120, AUC -0.060 vs §3.1 C). The Phase 11
"93% Hue" decomposition does *not* transfer to PAD-UFES-20.

The plausible mechanism: the target circular Hue (359 deg ~ red) is
anchored to dermoscopy contact-illumination backgrounds (dark, low
saturation, hue near red). PAD-UFES-20 backgrounds (skin around the
lesion, clothing, walls) live elsewhere on the hue circle. Forcing
the background hue toward red while preserving saturation and value
produces a visually implausible color boundary at the lesion edge
that the classifier seems to interpret as an abnormality cue.

### 3.3 Alpha sweep (Step 4, `04_lacn_alpha_sweep.py`)

Per-alpha test-set metrics (recalibrated tau per alpha):

| alpha | AUC   | tau_alpha | recall | spec  | F1    |
|-------|-------|-----------|--------|-------|-------|
| 0.00  | 0.850 | 0.640     | 0.840  | 0.680 | 0.778 |
| 0.25  | 0.831 | 0.650     | 0.820  | 0.700 | 0.774 |
| 0.50  | 0.818 | 0.670     | 0.760  | 0.720 | 0.745 |
| 0.75  | 0.806 | 0.680     | 0.840  | 0.700 | **0.785** |
| 1.00  | 0.798 | 0.670     | 0.840  | 0.640 | 0.764 |

Two findings:

1. **AUC monotonically decreases in alpha.** Discriminative quality
   is strictly worse at every alpha > 0. No partial-strength sweet
   spot exists for AUC.
2. **F1 sweet spot at alpha=0.75** (0.785), marginally above the
   no-LACN baseline (0.778). This is the *recall-first* operating
   point: same recall as alpha=1 (0.840) with 0.060 recovered
   specificity, at a cost of 0.044 AUC.

### 3.4 Head-to-head summary

| Variant + tau                  | recall | spec  | F1    | AUC   |
|--------------------------------|--------|-------|-------|-------|
| No LACN (Phase 9 baseline)     | 0.820  | 0.720 | 0.781 | 0.850 |
| Lab LACN + tau_LACN_Lab        | 0.840  | 0.640 | 0.764 | 0.798 |
| Hue LACN + tau_LACN_Hue        | 0.820  | 0.520 | 0.713 | 0.738 |
| alpha=0.75 Lab + tau_alpha     | 0.840  | 0.700 | **0.785** | 0.806 |

The alpha=0.75 partial Lab-LACN is the only variant that beats the
no-LACN baseline on F1 while keeping recall >= 0.79.

## 4. Discussion

### 4.1 LACN is a knob, not a fix

The headline finding from Step 2 — recall preserved, specificity lost,
AUC lost — was hypothesized to be a *strength* artifact that
intermediate alpha could resolve. The sweep falsifies that hypothesis:
AUC is strictly decreasing in alpha. LACN reshapes the ROC operating
point without improving the ROC.

This is consistent with the mechanism: background Lab-mean shift
distorts legitimate OOD color cues alongside the shortcut, and that
distortion is paid in AUC regardless of strength.

### 4.2 In-distribution channel attribution does not transfer to OOD

Phase 11 measured Hue carrying 93% of the in-distribution color cast.
If that decomposition were a property of the *model*, shifting only
Hue would recover specificity and AUC. It does the opposite. The
Phase 11 attribution is therefore a property of the in-distribution
*data*, not a transferable model invariant.

Practical consequence for shortcut interventions: channel attribution
measured on in-distribution counterfactuals is necessary but not
sufficient evidence for OOD deployment design. An OOD-aware target
distribution would be required.

### 4.3 Where LACN remains useful

The alpha=0.75 operating point still has clinical value: under a
recall-first protocol (cancer screening triage where false negatives
are the dominant cost), it raises F1 by +0.007 vs no-LACN, raises
recall by +0.020, while costing 0.044 AUC. Whether 0.044 AUC is an
acceptable trade for +0.020 recall depends on the cost matrix; that
is a deployment decision, not a research result.

The framing on the final defense slides — "LACN: promising candidate,
needs threshold recalibration before deployment" — is now empirically
supported.

### 4.4 Limitations

- Single OOD cohort. Generalization claims to other skin-photograph
  OOD cohorts (FitzPatrick17k, dermatoscopy-vs-phone splits) require
  separate evaluation.
- Otsu mask quality on PAD-UFES-20: 86% reliable. The 14% failure
  cohort is excluded; their LACN effect is unknown.
- Single seed for the cal/test split. Variance over splits not
  measured.
- The alpha grid is coarse (5 points). A finer grid around
  alpha in [0.6, 0.9] could refine the F1 sweet spot.

## 5. Conclusion

The Phase 12 LACN intervention, evaluated honestly on PAD-UFES-20
with leakage-controlled threshold recalibration, is a *test-time
operating-point knob*, not an OOD AUC improvement. Three variants
were tested; none lifts AUC above the no-LACN baseline. Hue-only
LACN underperforms Lab-LACN on every metric, demonstrating that
in-distribution channel attribution does not directly transfer to
OOD without an OOD-aware target distribution.

The result corrects the defense-time uncertainty about LACN status:
recalibration is feasible, the trade-off shape is now characterized,
and LACN is appropriate for a recall-first deployment at alpha=0.75
with tau=0.680.

## 6. Reproducibility

- Scripts: `scripts/phase12/02_lacn_ood_recalibration.py`,
  `scripts/phase12/03_lacn_hue_only.py`,
  `scripts/phase12/04_lacn_alpha_sweep.py`.
- Outputs: `artifacts/phase12/02_lacn_ood.*`, `03_lacn_hue_ood.*`,
  `04_lacn_alpha.*`.
- Figures: `artifacts/phase12/figures/fig18_*.png`,
  `fig19_*.png`, `fig20_*.png`.
- Seed=42, bootstrap B=5000 (where applicable), PyTorch 2.6 (local
  M3 MPS), Phase 6 ckpt unchanged.
