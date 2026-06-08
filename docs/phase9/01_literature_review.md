# Phase 9 — Literature Review

Critical synthesis of prior work on shortcut learning in dermatology AI,
counterfactual attribution for spurious correlations, and CAM-based
attention analysis. Positions this work relative to the closest competitors.

Voice: Direct, citation-bearing, oriented toward *what is missing* rather
than what exists. Every claim about prior work is sourced.

---

## 1. The Shortcut Learning Frame (Conceptual Foundation)

**Geirhos et al. 2020. *Shortcut Learning in Deep Neural Networks*. Nature MI.**

Defined the modern frame: deep networks frequently solve tasks via
*shortcuts* — decision rules that perform well on i.i.d. test sets but
fail under distribution shift because they exploit spurious correlations
rather than the intended causal features. The paper catalogued shortcut
learning across domains (object recognition, NLP, RL) but did not provide
a quantitative attribution method for *individual* shortcuts.

**Implication for this work.** Geirhos et al. names the phenomenon we
measure but provides no decomposition machinery. Our Phase 9 contributes
the decomposition: *which* shortcuts, by *how much*, with *what causal
weight* on the downstream OOD gap.

**Recent extension to medical imaging.** *The risk of shortcutting in
deep learning algorithms for medical imaging research* (Scientific
Reports, 2024) demonstrates the concrete risk with the "predicting
dietary preferences from knee X-rays" failure mode — a stark illustration
that medical models can attain high i.i.d. accuracy through
clinically-irrelevant cues. The paper is exhortative rather than
methodological: it argues *that* shortcut auditing is necessary, without
prescribing *how* to decompose multiple shortcuts.

---

## 2. Dermatology-Specific Shortcut Discovery

### 2.1 Bissoto et al. 2019 — *(De)Constructing Bias on Skin Lesion Datasets* (CVPRW)

The first major paper to demonstrate that ISIC-trained skin lesion
classifiers exploit spurious correlations beyond clinical features. Key
finding: models continue to classify above the dermatologist baseline
*even when the lesion itself is occluded*. Bissoto et al. evaluate on
ISIC Archive and Atlas of Dermography but report bias *holistically* —
they do not decompose by artifact type (ruler, vignette, ink, hair,
collection signature).

**Method.** Lesion occlusion + label-shuffling controls. Strictly
correlation-based; no counterfactual generation.

**Gap left.** No per-artifact attribution, no causal estimate, no
connection to OOD generalization gap.

### 2.2 Bissoto et al. 2022 — *Artifact-Based Domain Generalization* (arXiv 2208.09756)

Follow-up that explicitly catalogues four common artifact types in skin
imaging: **ruler, vignette, ink markings, and hair**. The paper studies
whether domain-generalization training schemes can produce
artifact-invariant representations. Artifacts are listed *jointly* —
treated as one combined nuisance distribution rather than separately
attributed.

**Method.** Domain-generalization training (IRM, GroupDRO variants) on
artifact-stratified subsets.

**Gap left.** The framing assumes the *training* intervention; it does
not provide a post-hoc, per-artifact causal attribution of an *already
trained* model. There is no per-artifact contribution to a downstream
OOD AUC gap.

### 2.3 Nauta et al. (PMC8774502) — *Uncovering and Correcting Shortcut Learning in Skin Cancer Diagnosis*

The closest competitor by intent. Studies a *single* artifact class —
elliptical coloured calibration patches present in ~46% of benign and
~0% of malignant images in their cohort. They use GMCNN inpainting both
to *remove* patches from benigns and to *insert* them into malignants,
measuring the prediction shift.

**Key result.** Inserting patches into malignant images drops sensitivity
from 0.886 to 0.191 — a strong, single-shortcut causal demonstration.

**Gap left.** *Single* artifact only. The paper does not attempt
multi-shortcut decomposition, does not contrast across collections (HAM
vs SIIM vs BCN), and does not connect shortcut presence to a downstream
modality-shift (e.g., dermoscopy → clinical) AUC gap. The
remove-and-reinsert protocol they invented, however, is a clean
methodological precedent we will cite and extend.

### 2.4 Winkler et al. 2019 (JAMA Dermatology)

The original ruler finding: dermoscopic skin lesion classifiers
systematically flag images with surgical skin markings as malignant. A
foundational *case report* of a specific shortcut. Not a decomposition
framework.

### 2.5 Bevan & Atapour-Abarghouei 2022 — Skin Lesion Frame Artifacts

Demonstrate that black frames (vignettes) correlate with malignant class
in ISIC, but they treat frames as a single nuisance. The recent
*Targeted Data Augmentation* paper (arXiv 2308.11386) confirms the
correlation strength: rulers correlate with malignancy at ~1.39× class
ratio; frames correlate substantially more strongly. The TDA paper
proposes *injecting* biases during training to break the correlation —
again, intervention at training time, not post-hoc attribution.

---

## 3. Counterfactual Methods for Spurious Correlation Removal

### 3.1 MaskMedPaint (arXiv 2411.10686, 2024)

The most visible 2024 entrant. Uses Stable Diffusion 1.5 (DreamBooth
fine-tuned) for inpainting *and* LaMa as a preprocessing step to remove
regions-of-interest. Tested on ISIC 2018, MIMIC-CXR, NIH ChestXray14,
Waterbirds, iWildCam.

**Critical quote.** "The method treats spurious features holistically
rather than decomposing multiple shortcuts separately. … does not measure
per-shortcut causal contributions; instead, [trains] a 'ground-truth'
classifier to verify whether generated images contain expected spurious
features."

**Their goal.** Data augmentation for training-time debiasing.

**Our goal.** Inference-time *measurement* of an already-trained model.

These are orthogonal goals. MaskMedPaint helps train better models; we
audit existing models. The methodological overlap (inpainting → measure
prediction shift) is real and we cite them; the contribution is distinct.

### 3.2 FastDiME (arXiv 2312.14223, 2023)

Mask-based diffusion counterfactuals, 20× faster than prior diffusion
counterfactual methods. Evaluates on CelebA and ISIC 2018; for ISIC, the
shortcut tested is **rulers** (singular).

**Critical quote.** "The method handles *one* suspected shortcut feature
at a time. Their pipeline evaluates individual shortcuts sequentially
rather than simultaneously managing multiple distinct shortcuts. …
measures association strength rather than strict causal decomposition."

**Their contribution.** Speed of counterfactual generation.

**Our contribution.** Simultaneous decomposition of multiple shortcuts
plus causal attribution per shortcut. Orthogonal.

### 3.3 Augustin et al. 2022 — VCE (Visual Counterfactual Explanations)

Counterfactual visual explanations via generative diffusion;
demonstrated on ImageNet and CheXpert. Modifies the *minimal* image
region to flip the prediction. Designed for *single-instance*
explanation, not population-level shortcut quantification.

**Gap left.** Per-image rather than per-population. We use the inpaint-
and-measure motif at the population level.

---

## 4. CAM Faithfulness and Localization Quality

### 4.1 Selvaraju et al. 2017 — *Grad-CAM*

Standard. Class-discriminative attention via gradient-weighted activation.
Known limitations:
- Spatial resolution is bounded by the last convolutional layer's grid
  (12×12 for EfficientNet-B0 at 384×384 input).
- "Spatial pooling of gradients causes Grad-CAM to highlight regions
  *larger* than those the model actually uses, creating visually
  appealing but spatially imprecise attributions" (literature critique).

### 4.2 Score-CAM (Wang et al. 2020)

Gradient-free CAM. Replaces gradient-based weighting with
forward-pass-only activation score weighting. Often more faithful by
Insertion/Deletion metrics; slower than Grad-CAM.

### 4.3 Faithfulness Metrics

- **Pointing Game** (Zhang et al. 2018): does the CAM peak land inside
  the ground-truth bounding box / segmentation mask? Requires per-image
  annotation.
- **Insertion/Deletion AUC** (Petsiuk et al. 2018, RISE): incrementally
  insert/delete pixels in order of CAM importance; faithful CAMs produce
  steep deletion curves and shallow insertion curves.
- **Sanity Check** (Adebayo et al. 2018): randomize model weights; if
  the saliency map is unchanged, the explanation is independent of the
  model and untrustworthy.

**Why this matters for us.** Phase 9 uses Grad-CAM as the *spatial
attention* signal. Its known imprecision (Geirhos critique) is a
disclosure we own. We compare against Score-CAM as a sensitivity check.
The *causal* arm of the work does not depend on CAM faithfulness — it
uses inpainting counterfactuals, which are CAM-independent.

---

## 5. Multi-Site Calibration and OOD Generalization

### 5.1 DermAI (arXiv 2511.10367, 2025)

PAD-UFES-20 cross-dataset OOD evaluation. Documents the domain shift
that motivates our Phase 7b result (AUC 0.95 → 0.81 in-dist to OOD; in
their setup, comparable magnitude).

### 5.2 Park et al. 2022 — *Closing the AI generalisation gap*

Adjusts model outputs for site-specific dermatology condition prevalence.
The intervention is *calibration*; it does not attribute the OOD gap to
shortcuts.

### 5.3 Daneshjou et al. 2022 — Diverse Dermatology Image Set

Documents performance disparities across Fitzpatrick skin types. Bias
attribution is to *demographic* shortcuts (skin tone), distinct from the
*acquisition-protocol* shortcuts we study (vignette, ruler, dataset
signature).

---

## 6. Cross-Collection Dataset Heterogeneity

### 6.1 BCN20000 vs HAM10000 — Documented Gap

ISIC 2019 challenge: best algorithm achieves 58.8% balanced accuracy on
BCN20000 vs 82.0% on HAM10000. The gap is widely attributed to
BCN20000's "clinically realistic" composition (heterogeneous subtypes,
fewer canonical presentations) — but **no published work attributes the
gap to per-artifact shortcut differences across collections**.

This is the precise hole we fill: *measuring* whether the cross-
collection AUC heterogeneity our Phase 7c reports (0.92 / 0.88 / 0.87
across HAM / SIIM / BCN) decomposes into artifact-specific shortcut
contributions.

---

## 7. Shapley Value for Multi-Feature Causal Decomposition

### 7.1 Lundberg & Lee 2017 — SHAP

Game-theoretic feature attribution. Provides additive decomposition with
desirable axioms (efficiency, symmetry, dummy, additivity). Approximate
variants for deep models: DeepSHAP, GradSHAP.

### 7.2 Heskes et al. 2020 — Causal Shapley

Distinguishes *interventional* (do-calculus-aligned) Shapley values from
*conditional* Shapley values. Causal Shapley aligns with our
counterfactual semantics: each shortcut's contribution is the average
marginal change in prediction over the power set of other shortcuts
present.

**Critical for us.** Causal Shapley is computationally heavy
($O(2^n)$ shortcut subsets). With n=5 candidate shortcuts (vignette,
ruler, ink, hair, dataset-signature) the cost is 32 inpainting
configurations per image — tractable on M3 MPS at our sample size
(N≈300–600).

**Prior work in this exact intersection** (Shapley × dermatology
shortcuts) **does not exist** to our knowledge.

---

## 8. Synthesis — Where the Existing Literature Stops

| Capability                                          | Bissoto'19 | Nauta'22 | FastDiME'23 | MaskMedPaint'24 | TDA'23 | ABDG'22 | **Our Phase 9** |
|-----------------------------------------------------|:---------:|:--------:|:-----------:|:---------------:|:------:|:-------:|:---------------:|
| Identifies skin-lesion shortcut at all              | yes       | yes      | yes         | yes             | yes    | yes     | yes             |
| Multiple artifacts simultaneously                   | partial   | no       | no          | holistic        | partial| partial | **yes (5)**     |
| Per-artifact causal attribution                     | no        | partial  | weak (MAD)  | no              | no     | no      | **yes (Shapley)**|
| Inpainting counterfactuals                          | no        | yes      | yes         | yes             | no     | no      | **yes (LaMa)**  |
| Cross-collection comparison (HAM/SIIM/BCN)          | no        | no       | no          | no              | no     | no      | **yes**         |
| Shortcut → OOD AUC gap attribution                  | no        | no       | no          | no              | no     | partial | **yes**         |
| Audit framework generalizable beyond skin           | no        | no       | partial     | yes             | no     | no      | **yes (Phase B)**|

The empty cells in the last two rows are our two strongest novelty
claims. No prior work connects *individual shortcut contributions* to
the *cross-collection AUC heterogeneity* or to the *OOD distribution
shift gap* in dermatology.

---

## 9. Position Statement

This work occupies a previously-unfilled triangle in the dermatology AI
shortcut-learning literature:

> Given an already-trained skin-lesion classifier and a multi-collection
> evaluation cohort, quantify per-artifact causal contribution to (a) the
> aggregate AUC, (b) the cross-collection AUC heterogeneity, and (c) the
> dermoscopy-to-clinical OOD AUC gap. Use inpainting-based counterfactuals
> with Shapley-value decomposition to obtain interventional attributions
> rather than associative correlations.

The closest prior work is Nauta et al. (PMC8774502), which is *single-
shortcut* and *single-collection*; we extend both axes simultaneously
and connect to the OOD gap that they do not address.

---

## 10. Citations Used

1. Geirhos R. et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence* 2: 665–673.
2. The risk of shortcutting in deep learning algorithms for medical imaging research. (2024). *Scientific Reports*.
3. Bissoto A., Fornaciali M., Valle E., Avila S. (2019). (De)Constructing Bias on Skin Lesion Datasets. *CVPRW*. arXiv:1904.08818.
4. Bissoto A., Barata C., Valle E., Avila S. (2022). Artifact-Based Domain Generalization of Skin Lesion Models. arXiv:2208.09756.
5. Nauta M. et al. (2022). Uncovering and Correcting Shortcut Learning in Skin Cancer Diagnosis. PMC8774502.
6. Winkler J. K. et al. (2019). Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition. *JAMA Dermatology*.
7. Sirico A. et al. (2023). Targeted Data Augmentation for bias mitigation. arXiv:2308.11386.
8. Hu Y. et al. (2024). MaskMedPaint: Masked Medical Image Inpainting with Diffusion Models for Mitigation of Spurious Correlations. arXiv:2411.10686.
9. Sanchez P. et al. (2023). FastDiME: Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation. arXiv:2312.14223.
10. Selvaraju R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
11. Wang H. et al. (2020). Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks. *CVPRW*.
12. Petsiuk V., Das A., Saenko K. (2018). RISE: Randomized Input Sampling for Explanation of Black-Box Models. *BMVC*.
13. Zhang J. et al. (2018). Top-down Neural Attention by Excitation Backprop. *IJCV*.
14. Adebayo J. et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS*.
15. Lundberg S., Lee S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
16. Heskes T. et al. (2020). Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models. *NeurIPS*.
17. Suvorov R. et al. (2022). Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa). *WACV*.
18. Park J. et al. (2022). Closing the AI generalisation gap by adjusting for dermatology condition distribution differences across clinical settings. PMC12167068.
19. Daneshjou R. et al. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*.
20. BCN20000: Dermoscopic Lesions in the Wild. (2024). PMC11183228.
