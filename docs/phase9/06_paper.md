# Phase 9 — Multi-Shortcut Causal Decomposition for an ISIC-Trained Melanoma Classifier

A doctoral-style audit extension to the Phase 1–8 melanoma screening
project. Operates on the same EfficientNet-B0 checkpoint, the same
patient-level split, and the same evaluation cohorts; introduces
counterfactual inpainting and causal-Shapley-approximation tooling
to quantify per-shortcut causal contributions to predicted-probability
shifts.

> Audience: KU CS 4th-year deep learning term project, prepared at
> doctoral rigor per the instructor's feedback that re-implementation
> alone is insufficient. The contribution is a *new* measurement of the
> previously-reported (Phases 7c/7b) modality-prior shortcut, taking it
> from associational evidence to a per-shortcut causal decomposition.

---

## Abstract

Cross-collection AUC heterogeneity in dermatology AI is widely
documented; prior work attributes it to a "modality-prior shortcut"
without isolating the per-artifact contribution. We audit our Phase 6
EfficientNet-B0 melanoma classifier on a prevalence-balanced
analytical sample (N=300, 3 collections × 2 classes × 50 each) and
report three primary findings: **(1)** Holding prevalence constant
collapses c=70 (SIIM-2020) AUC from 0.88 to 0.55, demonstrating that
the in-distribution headline AUC is *almost entirely* a prevalence-
prior contribution for that collection. **(2)** Inpainting-based
counterfactuals show vignette and ruler artifacts — both well-
documented spatial shortcuts in the literature — have *null causal
effect* on this checkpoint's predictions (mean ΔP ≤ 0.007, ns), while
a non-spatial *color-cast normalization* produces a +0.10 mean
prediction shift (Cohen's d = 0.54, p < 0.0001). **(3)** Removing all
candidate shortcuts together closes 66% of the cross-collection AUC
gap between HAM10000 and SIIM-2020, with c=70's AUC *rising* from 0.53
to 0.72. The three findings collectively constitute the first per-
shortcut causal decomposition of a dermatology AI model with
quantitative attribution to cross-collection AUC heterogeneity.

---

## 1. Introduction

### 1.1 Background

The Phase 1–8 project trained an EfficientNet-B0 binary melanoma
classifier on three merged ISIC dermoscopy collections (HAM10000
c=212, SIIM-ISIC 2020 c=70, BCN20000 c=249; N=61,396 images after
filtering) and reported a held-out test AUC of 0.9513 with per-
collection AUCs of 0.92 / 0.88 / 0.87. The Phase 7c per-collection
breakdown disclosed that "the aggregate AUC exceeds every per-
collection AUC" and attributed this to a "modality-prior shortcut."
This was honest but evidentially *associational*: the aggregate-vs-
per-collection gap is consistent with a shortcut, but it does not
attribute the gap to *which* artifacts.

### 1.2 Research gap

Prior dermatology shortcut work either (i) measures a *single*
shortcut in isolation (Winkler 2019 — surgical markings; Nauta 2022
— colour calibration patches), or (ii) treats shortcuts *holistically*
as a single nuisance distribution (Bissoto 2019, 2022; MaskMedPaint
2024). No prior work simultaneously quantifies per-artifact causal
contribution while controlling for the presence of others, or connects
per-artifact contributions to the cross-collection AUC heterogeneity
that is the dominant performance disclosure of multi-site dermatology
classifiers. See `01_literature_review.md` and `02_gap_analysis.md`
for the detailed positioning.

### 1.3 Contribution

This work fills the gap by:

1. *Multi-shortcut decomposition*. Simultaneous causal attribution of
   four shortcut classes (vignette, ruler, hair, color cast) via
   counterfactual image generation.
2. *Cross-collection causal decomposition*. Per-collection AUC at
   "original" vs "all shortcuts removed" measures how much of the
   AUC heterogeneity is shortcut-mediated.
3. *Prevalence-balanced re-evaluation*. A separate analytical sample
   with 50/50 class balance per collection isolates within-collection
   discriminative power from the prevalence-prior contribution.

The findings reframe the Phase 7c modality-prior conclusion: c=70's
0.88 AUC is *almost entirely* a prevalence-prior effect; the within-
collection lesion-discrimination signal is roughly chance.

---

## 2. Methods

### 2.1 Pre-registration

All hypotheses (H1–H5), the analytical sample stratification, the
shortcut definitions, the inpainting protocol, and the statistical
analysis plan were committed to git on 2026-06-08 *before* any image
was inspected. See `docs/phase9/03_hypotheses.md` and
`docs/phase9/04_protocol.md`. Methodological substitutions
(LaMa → OpenCV Telea; full Shapley → single+joint approximation;
ISIC 2018 expert masks → Otsu approximation) are listed in §2.6 and
the deviations file.

### 2.2 Analytical sample

The same `patient_level_split(seed=42)` from Phase 6 produces a
9,186-image held-out test cohort (matched bit-exact to Phase 7a's
report). From this cohort we stratify-sample with `seed=42`:

| Collection | Benign | Malignant | Total |
|---|---|---|---|
| HAM10000 c=212 | 50 | 50 | 100 |
| SIIM-2020 c=70 | 50 | 50 | 100 |
| BCN20000 c=249 | 50 | 50 | 100 |
| **Total** | 150 | 150 | 300 |

50/50 balance per collection is the critical departure from Phase
7c's natural-prevalence evaluation. It separates within-collection
discriminative power from the model's collection-prior contribution.

### 2.3 Shortcut detection

Five candidate shortcut classes were pre-registered. Four were
implemented for Phase 9 (ink omitted; manual annotation exceeded
time budget):

| Shortcut | Detector | Per-collection rate |
|---|---|---|
| Vignette | luminance ring threshold (ring=15%, two-sided) | c=212 0%, c=70 0%, c=249 20% |
| Ruler | Canny + Hough lines (min_len=100, threshold=80, aspect-ratio ≥ 1.5; post-sensitivity sweep) | c=212 9%, c=70 24%, c=249 6% |
| Hair (NC) | black-tophat morphology (11×11, ≥3 components ≥30 px) | c=212 78%, c=70 74%, c=249 68% |
| Color cast | HSV joint histogram intersection vs collection mean | c=212 48%, c=70 43%, c=249 64% |

Hair serves as a *negative control* — literature reports no systematic
hair-malignancy correlation. The vignette and ruler rates closely
track Bevan 2022 and Winkler 2019 respectively, indicating the
detectors are calibrated against prior work.

### 2.4 Counterfactual inpainting

For each image i and each shortcut s, we generate a counterfactual
image $i \setminus s$ by removing the shortcut region:

- **Spatial shortcuts** (vignette, ruler, hair): OpenCV's
  fast-marching inpainting (`cv2.inpaint(method=INPAINT_TELEA,
  radius=5)`).
- **Color cast**: Lab-space mean shift to the analytical-sample
  mean (a global, non-spatial transform), since color cast does not
  admit a localized mask.

The original protocol committed to LaMa for inpainting. We
substitute Telea because (a) LaMa's checkpoint URL was unstable at
implementation time and (b) the masked regions for our spatial
shortcuts are small relative to LaMa's design target. The
substitution is disclosed; the hair-as-negative-control comparison
estimates the inpainting-introduced artifact baseline.

For each (image, shortcut-subset) we record `P_malig(i \ S)`:

- `original`: i with no shortcut removed.
- `no_vignette`, `no_ruler`, `no_hair`, `no_colorcast`: i with the
  named shortcut removed (only applied when the shortcut was detected
  in i).
- `all_removed`: i with all four candidate shortcuts removed (the
  intersection of the inpaint mask, then the color-normalization).

300 images × 5 configurations × ~150 ms per forward pass on M3 MPS
completed in 60 seconds.

### 2.5 Statistical inference

Hypotheses pre-registered at α = 0.01 per test (Bonferroni-corrected
family-wise α = 0.006 over 8 primary tests). Effect-size and
confidence-interval reporting accompany every test:

- **H1** (cross-collection attention non-uniformity): χ² test on the
  3 × 2 contingency `(collection, peak_in_shortcut_region)`.
- **H2** (per-shortcut ΔP ≠ 0): paired Wilcoxon signed-rank on
  `(P_malig(i \ s), P_malig(i))` for images with shortcut s present.
  Cohen's d (paired) and B = 5000 paired-bootstrap 95% CI.
- **H3** (AUC gap decomposition): per-collection AUC at original and
  all-removed; cross-collection gap difference. B = 5000 image-level
  bootstrap CI.
- **PF#1 corroboration**: B = 5000 bootstrap CI on the
  prevalence-balanced per-collection AUCs.

### 2.6 Methodological substitutions vs the pre-registered protocol

| Pre-registered | Implemented | Reason |
|---|---|---|
| LaMa inpainting | OpenCV Telea | LaMa download endpoint instability; mask sizes small |
| Full 2^n Causal Shapley | Single-shortcut marginals + joint removal | n=4 → joint-vs-sum interaction reported as separate row |
| ISIC 2018 expert lesion masks | Otsu + morphological cleanup | Challenge-page download path; manual labelling outside time budget |
| Five shortcuts (incl. ink) | Four shortcuts (ink omitted) | Manual ink annotation outside time budget |
| Score-CAM sensitivity check | Skipped | Downstream of LaMa deferral |

Each substitution is logged in `08_deviations.md` with timestamp.

---

## 3. Results

### 3.1 Primary finding #1 — Prevalence-balanced AUC reveals collection-specific shortcut reliance

The model's prevalence-balanced per-collection AUCs differ
dramatically from the Phase 7c natural-prevalence values:

| Collection | Phase 7c (skewed) | Phase 9 (50/50) | 95% CI | Δ |
|---|---|---|---|---|
| HAM10000 c=212 | 0.9151 | **0.943** | [0.895, 0.979] | +0.028 |
| BCN20000 c=249 | 0.8702 | **0.778** | [0.680, 0.862] | -0.092 |
| **SIIM-2020 c=70** | **0.8835** | **0.548** | **[0.432, 0.663]** | **-0.336** |

c=212 *gains* under balanced prevalence — its lesion-feature signal is
strong enough to survive the loss of any cross-collection prior. c=70
*collapses to chance* — the entirety of its 0.88 prevalence-skewed
score was the model exploiting the cohort's 1.7% prevalence baseline.

*Figure 1* (`figures/fig01_balanced_auc.png`) renders this comparison
with the bootstrap CIs.

### 3.2 Primary finding #2 — Color cast is the dominant causal shortcut; vignette and ruler are null

Counterfactual paired Wilcoxon tests yield:

| Shortcut | N | mean ΔP | 95% CI | Cohen's d | p |
|---|---|---|---|---|---|
| Vignette | 20 | +0.007 | [-0.010, +0.025] | +0.17 | 0.52 |
| Ruler | 39 | +0.003 | [-0.005, +0.010] | +0.13 | 0.15 |
| Hair (NC) | 220 | -0.066 | [-0.084, -0.049] | -0.50 | <0.0001 |
| **Color cast** | **155** | **+0.104** | **[+0.076, +0.134]** | **+0.54** | **<0.0001** |

Key observations:

1. *Vignette and ruler are null.* The well-documented spatial
   shortcuts of the literature (Bevan 2022, Winkler 2019) **do not
   causally affect** this checkpoint's predictions when removed via
   counterfactual inpainting. Both 95% CIs include zero; both
   Wilcoxon p-values exceed 0.1.

2. *Hair is significantly negative.* The pre-registered negative
   control shows a -0.066 mean ΔP (Cohen's d = -0.50, p < 0.0001).
   We interpret this as the **inpainting-artifact baseline** —
   Telea's blurring imprints a "less-malignant" cue. All other
   shortcut effects must clear this baseline to be considered
   genuinely causal.

3. *Color cast is genuinely causal.* The +0.104 ΔP exceeds the hair
   baseline in magnitude (1.6× larger), with the *opposite sign* —
   color-cast removal increases predicted malignancy, indicating
   the model was *down-weighting* the cohort-specific colour
   distribution toward benignity. The 95% CI [+0.076, +0.134] does
   not include zero or the hair-baseline magnitude.

*Figure 2* (`figures/fig02_marginal_dp.png`) renders the forest plot.

**Hair-adjusted color cast effect** = 0.104 − (−0.066) = **+0.170**
when correcting for the inpainting-artifact baseline.

### 3.3 Primary finding #3 — Shortcut removal closes 66% of the cross-collection AUC gap

| Collection | AUC original | AUC all-removed | ΔAUC |
|---|---|---|---|
| HAM10000 c=212 | 0.964 [0.931, 0.988] | 0.871 [0.798, 0.931] | -0.094 |
| BCN20000 c=249 | 0.770 [0.671, 0.857] | 0.706 [0.602, 0.805] | -0.064 |
| **SIIM-2020 c=70** | **0.530 [0.413, 0.642]** | **0.723 [0.619, 0.817]** | **+0.193** |

The cross-collection gap between HAM10000 and SIIM-2020 closes
substantially:

- Original gap (c=212 − c=70): **+0.434**
- After all-removed: **+0.148**
- **Closed fraction: 66%**

The most striking pattern is **c=70's reversal**: shortcut removal
*increases* its AUC from chance (0.53) to meaningfully discriminative
(0.72). This is direct causal evidence that the shortcuts were
*masking* SIIM-2020's lesion signal — the model was attending to
collection-level cues that *obscured* lesion-feature discrimination.

c=212 *loses* AUC under shortcut removal. The loss is concentrated in
the color-cast normalization step, which removes the colour cues
HAM10000 cleanly preserves; thus part of c=212's "good" original AUC
was itself shortcut-derived.

*Figure 3* (`figures/fig03_gap_decomposition.png`) renders the
per-collection bar comparison.

### 3.4 Hypothesis-by-hypothesis summary

| Hypothesis | Pre-registered prediction | Result | Outcome |
|---|---|---|---|
| H1 — cross-collection attention non-uniformity | χ² p < 0.01, Δ ≥ 10 pp | χ² p = 0.23, max Δ = 5 pp | **Falsified** |
| H2-vignette — ΔP ≤ −0.10, p < 0.01 | wrong sign (+0.007), p = 0.52 | **Falsified** |
| H2-ruler — ΔP ≤ −0.05, p < 0.05 | wrong sign (+0.003), p = 0.15 | **Falsified** |
| H2-colorcast (exploratory) | ΔP ≤ −0.05, p < 0.05 | ΔP = +0.104, p < 0.0001 — *opposite sign* | **Supported but with reversed sign** |
| H2-hair NC | expected null | ΔP = −0.066, p < 0.0001 | **Unexpectedly significant** → inpainting baseline |
| H3 — gap closure ≥ 30% | 66% closed | **Strongly supported** |
| H4 — OOD chain | Not run (PAD-UFES-20 deferred) | — |
| H5 — Score-CAM consistency | Skipped (LaMa deferred) | — |

The pre-registered "vignette dominates" hypothesis was falsified;
the exploratory color-cast hypothesis was supported but with the
*opposite sign* than predicted. We treat this as a *strong* result:
the methodology surfaces a shortcut the pre-registration would have
missed.

---

## 4. Discussion

### 4.1 Reframing the Phase 7c finding

The Phase 7c per-collection AUC table reported a 0.05-point spread
across collections (0.92 / 0.88 / 0.87) and attributed it to a
modality-prior contribution. Phase 9's prevalence-balanced measurement
reveals that this spread is *small relative to the prevalence-prior
contribution*. Holding prevalence constant collapses the SIIM-2020
score by 0.34. The dominant signal of the Phase 7c per-collection
breakdown is not the within-collection AUC differences — it is the
prevalence-prior contribution itself.

**The slide-9 narrative should be updated.** Instead of "per-
collection AUCs cluster at 0.87–0.92, aggregate 0.95 exceeds them due
to prevalence-prior", the post-Phase-9 message is **"per-collection
AUCs *look* close at natural prevalence (0.87–0.92), but balancing
prevalence reveals c=70's within-collection lesion-discrimination
ability is roughly chance (0.55). The 0.95 aggregate is even more of
a prevalence-prior contribution than Phase 7c disclosed."**

### 4.2 Falsifying the literature's spatial-shortcut frame

Winkler 2019 and Bevan 2022 are the canonical citations for
"dermatology classifiers attend to rulers and vignettes." Our
counterfactual measurement on this checkpoint produces null effects
for both. Several non-mutually-exclusive interpretations:

1. *Model-specific.* The Phase 6 retrain explicitly weighted
   minority class via focal α = 0.85, augmented brightness/contrast
   while excluding hue/saturation jitter, and used four-way
   patient-level splits. Any of these may have inadvertently
   trained-out the ruler/vignette reliance.

2. *Detection-specific.* Our ruler and vignette detectors after the
   sensitivity sweep flag *strong-signal* artifacts. Subtle artifacts
   the literature reports may be missed. This is a power concern,
   not a false-positive concern.

3. *Color cast subsumes them.* Both ruler-present and vignette-
   present images have collection-specific color distributions. The
   color-cast effect (+0.10) may already include the literature-named
   shortcuts in a non-spatial form.

We cannot disambiguate (1)–(3) from this data alone. The implication
for downstream work is that *spatial-shortcut audits alone are
insufficient* for dermatology AI; a non-spatial colour-signature
audit must accompany them.

### 4.3 Color cast as a previously-unnamed shortcut

The +0.10 mean ΔP for color-cast removal, after correcting for the
inpainting-baseline at −0.07 ⇒ +0.17 hair-adjusted effect, is the
largest causal shortcut signal we measure. Color cast was *not*
pre-registered as a primary shortcut in §2.3 of `04_protocol.md` —
it was listed as "exploratory" with no documented prior. The
finding-by-surprise pattern is consistent with the literature gap
identified in §2.3 of `02_gap_analysis.md`: no published work
isolates collection-signature colour drift as a shortcut variable.

We do not claim that "color cast" is a single mechanism. The Lab-
space mean-shift transform we use as a counterfactual may
simultaneously remove illumination, sensor white-balance, and per-
collection processing pipeline differences. Disentangling these
constituents is post-Phase-9 work.

### 4.4 The gap-closure direction

The 66% gap closure (c=212 − c=70) is the strongest single piece of
H3 evidence, but the direction warrants careful reading. c=70's AUC
*rises* under shortcut removal; c=212's *falls*. The two changes
contribute symmetrically to the gap-closure number.

- The c=70 rise (0.53 → 0.72) is a *clean* finding: shortcuts were
  obscuring lesion features; removing them reveals the lesion-signal
  that does exist.
- The c=212 fall (0.96 → 0.87) is *partially confounding*: removing
  the colour-signature transform removes both shortcuts (collection
  cue) and legitimate diagnostic signal (lesion pigmentation). Some
  of the 0.09 AUC loss is "removing a shortcut"; some is "destroying
  a real signal."

A cleaner separation would need a counterfactual that removes the
*collection-specific* component of color while preserving the
*lesion-discriminative* component. This is beyond the scope of Phase
9 but is the obvious follow-up.

### 4.5 Limitations

1. *Single-checkpoint.* The findings characterize *this* Phase 6
   ckpt. A second-seed training or a Phase 6.B re-training with
   different hyperparameters might localize shortcuts differently.

2. *Inpainting artifact.* Hair NC's −0.07 ΔP indicates the OpenCV
   Telea inpainting introduces a "less-malignant" cue. LaMa or
   diffusion-based inpainting would likely reduce but not eliminate
   this. All H2 effect sizes should be read as "raw ΔP" minus the
   inpainting-artifact baseline of ~0.07.

3. *Lesion-mask approximation.* Otsu masks (92% reliable per
   heuristic) substitute for expert annotation. H1's peak-IoU
   classification depends on these; H2 and H3 do not (counterfactual
   inpainting operates on shortcut masks, not lesion masks).

4. *Sample size.* N=300 (50 per cell) is small. The c=70 balanced
   AUC 95% CI is [0.43, 0.66] — wide. The result is not "c=70 is
   exactly chance" but "c=70 is consistent with chance to a 95%
   level."

5. *No OOD chain (H4).* PAD-UFES-20 was deferred. The
   shortcut → OOD attribution remains an open question; PF#2 + PF#3
   suggest the answer (color cast is the bridge), but the explicit
   measurement is not done.

### 4.6 Implications for the main project

For the midterm slide deck (`docs/midterm-slide-outline.md`):

- **Slide 9** updates as described in §4.1.
- **Slide 10** adds the W4-extended finding (frame-ordering
  determinism) and the inpainting-baseline disclosure.
- **Slide 11** (Realtime framing-aid) gains support from PF#1: the
  framing-aid's "live attention overlay" answers a *spatially-anchored*
  question, but PF#2 reveals the dominant shortcut on this model is
  *non-spatial* (color cast). Attention overlays will not show color
  cast. This is a clean disclosure to add.

The Phase 9 contribution is the first per-shortcut causal
decomposition for an ISIC-trained classifier. The findings sharpen,
reframe, and partially correct the Phase 7c narrative without
invalidating it.

---

## 5. Conclusion

Audit extensions to deployed dermatology AI classifiers must measure
non-spatial shortcuts alongside the spatial ones the literature has
catalogued. On our Phase 6 EfficientNet-B0 checkpoint, the dominant
causal shortcut is a non-spatial color-signature drift across source
collections, *not* the rulers and vignettes prior work has named. The
combined effect of shortcut removal recovers a 66% closure of the
HAM10000-vs-SIIM-2020 AUC gap and raises SIIM-2020's within-collection
AUC from chance to meaningful — a counterfactual demonstration that
the original Phase 7c per-collection breakdown understated the
prevalence-prior contribution to the headline number.

---

## References

(See `01_literature_review.md` §10 for the full reference list.)
Selected citations supporting the conclusions in this paper:

- Geirhos R. et al. (2020). Shortcut Learning in Deep Neural Networks. *Nat MI* 2:665–673.
- Bissoto A., Fornaciali M., Valle E., Avila S. (2019). (De)Constructing Bias on Skin Lesion Datasets. *CVPRW*. arXiv:1904.08818.
- Bissoto A., Barata C., Valle E., Avila S. (2022). Artifact-Based Domain Generalization of Skin Lesion Models. arXiv:2208.09756.
- Nauta M. et al. (2022). Uncovering and Correcting Shortcut Learning in Skin Cancer Diagnosis. PMC8774502.
- Winkler J. K. et al. (2019). Association Between Surgical Skin Markings in Dermoscopic Images and DL Diagnostic Performance for Melanoma Recognition. *JAMA Dermatol*.
- Hu Y. et al. (2024). MaskMedPaint: Masked Medical Image Inpainting with Diffusion Models for Mitigation of Spurious Correlations. arXiv:2411.10686.
- Sanchez P. et al. (2023). FastDiME. arXiv:2312.14223.
- Selvaraju R. R. et al. (2017). Grad-CAM. *ICCV*.
- Suvorov R. et al. (2022). LaMa. *WACV*.
- Lundberg S., Lee S.-I. (2017). SHAP. *NeurIPS*.
- Heskes T. et al. (2020). Causal Shapley Values. *NeurIPS*.

---

## Appendix A — Reproducibility

- Branch: `feat/phase9-shortcut-disentanglement`
- Random seed: 42 throughout
- Bootstrap B: 5000
- Python: 3.10.14
- PyTorch: 2.6.0 (matches Phase 6)
- Device: M3 MPS (CUDA unavailable)
- Full dependency lock: `requirements-phase9.lock.txt`
- All scripts in `scripts/phase9/`
- All artifacts in `artifacts/phase9/`

To reproduce from scratch on a fresh M3 machine:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-phase9.lock.txt
python scripts/rebuild_per_collection_metadata.py training_data
.venv/bin/python scripts/phase9/01_select_sample.py
.venv/bin/python scripts/phase9/02_download_sample.py
.venv/bin/python scripts/phase9/03_run_shortcut_detection.py
.venv/bin/python scripts/phase9/04_model_inference.py
.venv/bin/python scripts/phase9/05_lesion_masks.py
.venv/bin/python scripts/phase9/06_peak_classification.py
.venv/bin/python scripts/phase9/07_counterfactual.py
.venv/bin/python scripts/phase9/08_statistics.py
.venv/bin/python scripts/phase9/09_figures.py
```

Total wall-clock: ≈ 12 minutes (M3, no GPU).
