# Phase 9 — Experimental Protocol (Pre-registration)

Locked *before* image collection. Defines data, shortcut definitions,
attention extraction, inpainting pipeline, causal decomposition, and
statistical analysis. Departures from this protocol are logged in
`08_deviations.md` with timestamp.

Companion documents: `01_literature_review.md`, `02_gap_analysis.md`,
`03_hypotheses.md`.

---

## 1. Data

### 1.1 Source cohort

The Phase 7a held-out test cohort (N=9,186 dermoscopy images, 1,767
malignant). Patient-level disjoint from train/val/cal. Source columns
already tagged with `source_collection` ∈ {212 (HAM10000), 70 (SIIM
2020), 249 (BCN20000)}.

### 1.2 Stratified analytical sample

Random patient-level sampling, seed = 42, stratified by 3 × 2 cells
(collection × {benign, malignant}). Target N per cell:

| Collection | Benign | Malignant | Cell N |
|---|---|---|---|
| c=212 HAM10000 | 50 | 50 | 100 |
| c=70 SIIM 2020 | 50 | 50 | 100 |
| c=249 BCN20000 | 50 | 50 | 100 |
| **Total** | **150** | **150** | **300** |

Equal benign/malignant sub-cell counts give the H2 test maximum power
on the *benign* sub-cohort (where shortcut→malignant-shift is the
prediction). The malignant sub-cell is used for H3 (cross-collection)
and as a negative control for H2 (inpainting on already-malignant images
should produce smaller |ΔP|).

A *secondary* analytical sample of N=600 (200 per collection, same
stratification) will be drawn *if and only if* the primary N=300 sample
produces a primary-hypothesis p-value in [0.005, 0.02] — i.e., the
inconclusive band. The size of the secondary sample is pre-registered to
prevent p-hacking via post-hoc N inflation.

### 1.3 Image acquisition

The Phase 6 retrain ran on a remote A100X server (Elice Cloud). The
training data are not currently mirrored to the local M3 machine. Phase
9 will download *only the analytical sample* via `isic-cli`, targeting
the ~300 selected `isic_id` strings. Estimated disk: 300 × ~10 MB =
3 GB.

OOD cohort (PAD-UFES-20, ISIC c=406) is held out for Phase 9 H4 only,
sample N=200 by same stratification.

---

## 2. Shortcut Definitions

The five candidate shortcuts. Each is defined by (a) a verbal
specification, (b) an automatic or hybrid detection rule, (c) a
*known* literature prior on its prevalence/correlation in skin imaging.

### 2.1 Vignette (S₁)

**Definition.** A radial darkening band along the image periphery,
typically the outer 10–20% of the image width. Caused by the dermoscope
optical cone.

**Detection.** Pixel-value thresholding on the outer ring:
```
ring_mask = outer 15% of image radius
vignette_present = mean(luminance | ring_mask) < 0.20  (on [0,1])
                   AND mean(luminance | center) > 0.40
```
The two-sided rule prevents false positives on uniformly dark images.

**Mask for inpainting.** The ring region itself (outer 15%, with a
2-pixel Gaussian feather).

**Literature prior.** Bevan & Atapour-Abarghouei 2022; Bissoto 2022
(ABDG); Sirico 2023 (TDA). Frame/vignette is the *strongest* known
artifact-class correlation with malignancy in ISIC.

### 2.2 Ruler / surgical markings (S₂)

**Definition.** Linear or scale-bar imaging of a measurement device or
surgical marker pen ink. Distinct from the periphery vignette.

**Detection.** Hybrid. First-pass: morphological line detection (Hough
transform on Canny edges, line length ≥ 60 px). Second-pass: manual
verification on the first-pass positives. Threshold: ≥ 1 line longer
than 60 px = ruler-present.

**Mask for inpainting.** The detected line region(s) dilated by 5 px.

**Literature prior.** Winkler 2019; FastDiME 2023. Documented
correlation with malignancy ~1.39 class ratio (Sirico 2023).

### 2.3 Ink markings (S₃)

**Definition.** Skin-marker pen ink on the periphery of the lesion (used
by dermatologists to track follow-up).

**Detection.** Manual labelling only — automatic detection is unreliable.
A single annotator labels the analytical sample's images for ink
presence/absence and (when present) draws a coarse rectangular bounding
box. Annotation reliability is checked by re-labelling 30 images two
weeks apart; intra-annotator agreement target κ ≥ 0.7.

**Mask for inpainting.** The bounding box dilated by 10 px.

**Literature prior.** Bissoto 2022 (ABDG). Less correlated than ruler;
expected weak effect.

### 2.4 Hair (S₄) — *negative control*

**Definition.** Body hair occluding part of the lesion or surrounding
skin.

**Detection.** Automatic. Black-tophat morphological filter with a
disk-shaped kernel (radius 5 px); threshold the response and check
connectivity (≥ 3 connected components of length > 30 px).

**Mask for inpainting.** The detected hair pixels dilated by 3 px.

**Literature prior.** Hair has no documented systematic correlation
with malignancy. Pre-registered as a negative control for the inpainting
pipeline's validity (§3 in `03_hypotheses.md`).

### 2.5 Collection-signature colour cast (S₅)

**Definition.** A diffuse colour temperature / saturation cast specific
to a source collection's acquisition device. *Not* spatially localized;
manifests as a global colour distribution shift.

**Detection.** Per-image colour histogram is computed; a per-collection
"signature" cast is the mean histogram. Membership-strength score is
the histogram intersection between the image and its collection's
signature.

**Mask for inpainting.** Not maskable in the local-region sense (S₅ is
global). For Shapley decomposition, S₅ is *operationalized* differently:
we apply a colour-normalization transform to the image (CLAHE + per-
channel mean shift to the *target* collection's signature) rather than
LaMa inpainting.

**Literature prior.** No published per-collection colour-signature
analysis on ISIC. This is the *most exploratory* shortcut in our set.

---

## 3. Attention Analysis (H1)

### 3.1 CAM extraction

Grad-CAM via the existing `app.py` implementation. Target layer: the
last convolutional layer of `EfficientNet-B0.features` (the 1280-channel
12×12 output). Class index = 1 (malignant).

Per image:
1. Forward pass to obtain `logit, activations[1280×12×12]`.
2. Backward pass on `sigmoid(logit) → 1` to obtain
   `grads[1280×12×12]`.
3. Channel weights `α_k = mean(grads_k over spatial)`.
4. CAM = ReLU(Σ_k α_k · activations_k), upsampled to 384×384 bilinearly.
5. Normalize CAM to [0,1].

### 3.2 Peak region classification

The CAM "peak" is the centroid of the *top-5% mass* pixels (a more
robust choice than the absolute argmax for noisy CAMs).

For each image, the peak centroid is classified into one of six
categories by intersection-over-presence with the shortcut masks
defined in §2:

```
peak ∈ {S1_vignette, S2_ruler, S3_ink, S4_hair, lesion, normal_skin}
```

Lesion mask: from ISIC 2018 Task 1 segmentation masks (publicly available
for HAM10000 subset). For c=70 and c=249 — where ISIC 2018 masks are
not available — a sub-sampled subset (N=50 per collection) is manually
masked by the annotator.

For images with no shortcut present, the peak is forced into {lesion,
normal_skin} via lesion-mask intersection.

### 3.3 Score-CAM sensitivity check (H5)

Replicate §3.1–§3.2 with Score-CAM. Score-CAM weights each channel by
the *forward-pass score* contribution of that channel's activation map
as a soft mask, rather than by gradient. Implementation: `torchcam` or
hand-coded.

### 3.4 Output artifact

`artifacts/phase9/01_attention_analysis.csv`. Columns:
`isic_id, source_collection, label, cam_method, peak_x, peak_y,
peak_category, peak_iou_lesion, peak_iou_vignette, peak_iou_ruler,
peak_iou_ink, peak_iou_hair`.

---

## 4. Counterfactual Inpainting (H2)

### 4.1 LaMa setup

Pre-trained LaMa checkpoint (the official `big-lama` model). Inference-
only on M3 MPS. Verified throughput on a 384×384 image: ~1.0–1.5 sec
per image per shortcut subset (16,000 inpainting passes ≈ 6.5 hours,
acceptable batch).

### 4.2 Inpainting procedure

For each image i and each shortcut subset S ⊆ {S₁, S₂, S₃, S₄, S₅}:

1. Construct combined mask M_S = union of shortcut masks in S.
   (S₅ has no mask; treat S₅ as a global colour-normalization
   transform applied *after* LaMa runs on M_{S \ {S₅}}.)
2. Run LaMa on (image, M_S) to obtain `inpainted_i_S`.
3. Forward through the classifier to obtain `P_malig(inpainted_i_S)`.
4. Record (i, S, P_malig, plausibility_score).

`plausibility_score` is a manual visual QA flag (0 = inpainting
introduced obvious artifact; 1 = inpainting is plausible). Score < 1
images are dropped from the causal estimate but retained in the
sensitivity-disclosure pool.

### 4.3 Output artifact

`artifacts/phase9/02_inpainting_predictions.csv`. Columns:
`isic_id, shortcut_subset_S, P_malig_original, P_malig_inpainted,
plausibility_score, inpaint_runtime_ms`.

---

## 5. Causal Decomposition (H3)

### 5.1 Causal Shapley value

For each image i and each shortcut s, the Causal Shapley value is:

```
φ_i(s) = sum over T ⊆ S \ {s} of
         |T|!(n-|T|-1)!/n! · [P_malig(i, T ∪ {s}) - P_malig(i, T)]
```

where:
- S is the set of shortcuts *present* in image i.
- P_malig(i, T) is the model's predicted malignant probability on image
  i after LaMa-inpainting away the shortcuts *not* in T (the "with
  shortcuts T retained" counterfactual).
- The full enumeration is 2^|S| inpainting configurations per image.
  With max(|S|) ≤ 5, the cost is ≤ 32 configurations per image.

### 5.2 Aggregate per-shortcut attribution

```
Φ(s) = mean over images i where s is present of φ_i(s)
```

Bootstrap (B = 1000, seed = 42) over images for 95% CI.

### 5.3 Cross-collection gap decomposition

Define:

```
ΔAUC(c1, c2) = AUC on c1 − AUC on c2
```

The *shortcut-explained* gap is:

```
ΔAUC_shortcut(c1, c2) = AUC(c1, all shortcuts inpainted)
                      − AUC(c2, all shortcuts inpainted)
```

The *closed fraction* is:

```
closed_frac = (ΔAUC(c1,c2) − ΔAUC_shortcut(c1,c2)) / ΔAUC(c1,c2)
```

Bootstrap 95% CI as above.

### 5.4 Output artifact

`artifacts/phase9/03_causal_attribution.csv`,
`artifacts/phase9/04_gap_decomposition.json` (aggregate + CI).

---

## 6. OOD Chain Attribution (H4)

### 6.1 PAD-UFES-20 baseline

Re-run Phase 7b to obtain per-image P_malig on the OOD cohort. *No
inpainting* on OOD images (the artifacts are different in clinical
phone photos).

### 6.2 Bridge

For each in-distribution image i in the analytical sample:
- Sum of |φ_i(s)| over s = total causal shortcut magnitude.
- For each OOD image j, the OOD prediction deviation from in-dist
  baseline = |P_malig(j) − mean_in_dist_baseline_for_same_class|.

Compute Spearman ρ over (per-collection mean) of:
- in-dist total causal shortcut magnitude (one value per collection)
- OOD per-collection AUC drop (one value per collection)

Three pairs (HAM, SIIM, BCN) → ρ computed on n=3. This is
under-powered for a per-collection-level analysis; the H4 test is
therefore re-specified to use *per-image* analogues:

For each in-dist image i:
- shortcut-magnitude_i = sum_s |φ_i(s)|

For each OOD image j:
- deviation_j = |P_malig_OOD(j) − P_malig_OOD_collection_mean|

The H4 test correlates the *distribution-level* in-dist
shortcut-magnitude statistics with the *distribution-level* OOD
deviation statistics. Spearman ρ over the bootstrap-resampled
collection-level pairs (B = 10000 resamples to inflate the n=3 base).

### 6.3 Output artifact

`artifacts/phase9/05_ood_chain.json`.

---

## 7. Statistical Analysis Plan

### 7.1 Multiple-comparison correction

H1, H2 (5 shortcuts), H3, H4 → 8 tests in the primary analysis.
Bonferroni-corrected α = 0.05 / 8 ≈ 0.006. Each individual hypothesis
is pre-registered at α = 0.01, more stringent than the corrected
threshold.

### 7.2 Effect-size reporting

Every significance test is accompanied by:
- Point estimate (median or mean as pre-specified).
- 95% bootstrap CI (B = 5000 unless otherwise noted).
- Cohen's *d* (for paired tests) or proportion difference (for χ²).

### 7.3 Open-science deposit

All result CSVs, all per-image inpainted PNGs (under
`artifacts/phase9/inpainted/`), the LaMa weights pointer, the random
seeds, and the git hash of `feat/phase9-shortcut-disentanglement` at the
time of the analysis run are committed to the branch. A future reader
can re-run from the saved seeds and obtain bit-exact results (extending
the Phase 5 A3 reproducibility regime to Phase 9).

---

## 8. Compute Budget

Local M3 MPS, no GPU rental.

| Stage | Compute | Estimate |
|---|---|---|
| Sample download (3 GB) | network | 1 hr |
| Lesion segmentation acquisition (ISIC 2018 masks) | network | 30 min |
| Grad-CAM × 300 images × 2 methods | inference | 1 hr |
| Manual annotation (ink masks, lesion masks for c=70/249) | human | 4 hr |
| Shortcut detection automated runs | CPU | 30 min |
| LaMa inpainting 300 × 32 subsets | MPS | 8 hr |
| Forward-pass on inpainted images | MPS | 2 hr |
| Shapley + bootstrap | CPU | 1 hr |
| OOD bridge | MPS | 1 hr |
| **Total** | | **≈ 19 hr** (across ≈ 5 calendar days) |

No GPU rental needed. The LaMa inpainting batch is the longest run; can
be backgrounded overnight.

---

## 9. Pre-registered Risk Register

Items that could threaten validity, with the mitigation already
committed to:

| Risk | Mitigation |
|---|---|
| LaMa inpainting introduces a counterfactual-image artifact that the classifier reacts to spuriously | Hair negative control (H2-hair); plausibility-score filtering; visual QA on every shortcut subset |
| Manual ink-mask annotation is unreliable | Intra-annotator κ check; secondary annotator on 50 images for inter-rater κ |
| Lesion masks unavailable for c=70 / c=249 | Hand-segment N=50 per collection; report uncertainty interval for those collections |
| Insufficient images with shortcut S present | Pre-register the *secondary sample* (N=600) trigger and run only if power < 80% |
| CAM-method dependence | H5 Score-CAM replication |
| Per-collection N=100 too small for AUC bootstrap | Bootstrap CIs reported on every per-collection metric |
| OOD H4 has n=3 collection-level pairs | Re-specified to per-image bootstrap (§6.2) |

---

## 10. Authorship and Scope Tag

All Phase 9 work is committed under
`feat/phase9-shortcut-disentanglement`. Final merge to `main` only after
the paper-format writeup (`07_paper.md`) and slide-integration draft
(`09_slide_integration.md`) are complete. The branch is the canonical
location for the audit-rewrite extension; no Phase-9 commits to other
branches.
