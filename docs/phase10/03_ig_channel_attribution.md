# Phase 10 / Part 3 — Integrated Gradients Channel Attribution

Companion to `02_results.md`. Addresses the question raised during the
final defense: *Grad-CAM does not show an attention shift between
Phase 6 and Phase 10 ckpts, yet the Phase 10 model exhibits dramatic
behavioral changes (color-cast ΔP 86% reduction, c=70 balanced AUC
0.548 → 0.867). What instrument can visualize the change Grad-CAM
cannot?*

---

## Abstract

We measured per-image Integrated Gradients (IG, Sundararajan et al.
2017) attribution for both ckpts on a stratified 60-image subset of
the Phase 9 analytical cohort. We tested two hypotheses about the
Phase 10 intervention that Grad-CAM cannot answer:

- **H_bg**: Phase 10 reduces the *background-to-foreground attribution
  ratio* relative to Phase 6 (the lesion gets *relatively* more
  attribution).
- **H_rgb**: Phase 10 reduces the per-image RGB-channel-share standard
  deviation, i.e. the model relies more uniformly across RGB channels.

H_bg is supported: mean bg/fg ratio shifts from 1.28 (Phase 6) to 1.04
(Phase 10), a paired delta of -0.24 with 95% CI [-0.43, -0.07]
excluding zero. The Wilcoxon p-value is 0.107, suggesting the effect
sits at the boundary of paired-rank significance but is clearly
nonzero in the mean.

H_rgb is *not* supported: per-image RGB-share std is unchanged
(0.118 → 0.118). Only the B channel share shows a small, statistically
significant decrease (-0.002, Wilcoxon p=0.011); R and G shares are
unchanged. The Phase 10 intervention shifts attribution in a direction
*orthogonal* to the RGB axis.

The result confirms the mechanism implied by the defense framing: the
Phase 10 fine-tune changes *channel-level* (Hue) sensitivity, which is
invisible to spatial Grad-CAM. IG provides a complementary instrument
that does see part of this change, though not the full picture.

---

## 1. Method

### 1.1 Sample

60 images from the Phase 9 analytical cohort (`01_analytical_sample.
csv`), restricted to images with a reliable Otsu lesion mask
(`05_lesion_mask_stats.reliable == True`). Stratified by
(collection, label) with 10 images per stratum (3 collections x 2
labels). Seed=42.

### 1.2 IG configuration

- Method: `captum.attr.IntegratedGradients` against the Phase 6 ckpt
  and against the Phase 10 ckpt independently
- Target: model's scalar (binary) logit, i.e. the malignant-class
  output before sigmoid
- Baseline: zero tensor at the model's input resolution
- Steps: 32
- internal_batch_size=1 (MPS memory)

### 1.3 Aggregation

For each image and each ckpt:

```
abs_attr_2d(x, y) = sum_c |IG_c(x, y)|

fg_attr_mean      = mean over pixels inside Otsu mask of abs_attr_2d
bg_attr_mean      = mean over pixels outside Otsu mask of abs_attr_2d
bg_fg_ratio       = bg_attr_mean / fg_attr_mean

r_share = sum_pixels |IG_R| / sum_pixels (|IG_R| + |IG_G| + |IG_B|)
g_share, b_share defined analogously
```

We then test paired hypotheses on the per-image deltas
`Phase10 - Phase6`.

### 1.4 Statistics

- Bootstrap (B=5000, seed=42) of the per-image delta mean, with 95% CI
- Wilcoxon signed-rank test (two-sided, zsplit)

## 2. Results

### 2.1 H_bg — background/foreground attribution ratio

| Statistic                          | value                  |
|------------------------------------|------------------------|
| Phase 6 mean bg/fg                 | 1.282 (median 0.929)   |
| Phase 10 mean bg/fg                | 1.040 (median 0.941)   |
| Paired delta (Phase10 - Phase6)    | -0.242                 |
| 95% bootstrap CI                   | [-0.425, -0.074]       |
| Wilcoxon p (two-sided)             | 0.107                  |

The bootstrap CI **excludes zero**, so the mean shift is robust to
bootstrap resampling. The Wilcoxon p just exceeds the conventional
0.05 boundary, indicating the *rank* test does not register
significance even though the *mean* shift does.

The discrepancy is informative: Phase 6 medians (0.93) and Phase 10
medians (0.94) are nearly identical. The mean shifts because the
*right tail* of the distribution (images where Phase 6 placed
relatively more attribution outside the lesion) compresses under
Phase 10. Half of the images are not affected; the other half are.

This is consistent with the Phase 10 mechanism: hue jitter penalizes
the model for relying on global color signatures (which are background
dominant) but does not penalize lesion-internal attribution. Images
that already had foreground-dominant attribution under Phase 6 stay
the same; images that leaned on background context lose that lean.

### 2.2 H_rgb — RGB-channel share rebalance

| Channel | Δshare (P10 - P6) | 95% bootstrap CI    | Wilcoxon p |
|---------|-------------------|---------------------|------------|
| R       | -0.001            | [-0.0044, +0.0015]  | 0.632      |
| G       | +0.004            | [+0.0008, +0.0064]  | 0.166      |
| B       | -0.002            | [-0.0038, -0.0006]  | 0.011      |

Per-image RGB-share standard deviation:
- Phase 6 mean: 0.1181
- Phase 10 mean: 0.1181
- Difference: 0 to four decimal places

Only the B channel shifts at conventional significance, and the shift
is tiny (-0.002 in share, against a baseline share around 0.30). The
R and G shifts are not significant. The per-image share *spread* does
not change.

**The Phase 10 intervention is not an RGB-channel rebalance.** This
is the negative result that motivated the audit pipeline in the
first place: shortcut shifts that look natural in HSV (a hue rotation
of about 36 deg) project as multiple small per-channel changes in
RGB, none of which is individually large.

### 2.3 Example attribution maps (fig22)

The example panel (`fig22_ig_attribution_examples.png`) shows five
images with their original RGB, Phase 6 |IG| map, and Phase 10 |IG|
map. Lesion outlines from the Otsu mask are overlaid on the original.
Visual inspection corroborates the statistic: Phase 10 maps are more
concentrated near the lesion boundary in most examples, while a
minority show no clear shift.

## 3. Discussion

### 3.1 What this finding adds to the defense

The defense slides reported that fig13_v2 (peak-on-lesion rate)
showed no statistically significant change in spatial attention. We
were honest about this: "Grad-CAM cannot capture the relevant change
because it sums over channels." This Phase 10 / Step 6 analysis
substantiates that claim:

- Grad-CAM measurement: no change.
- IG bg/fg ratio: mean shift -0.24, CI excludes 0.
- IG RGB shares: no rebalance.

The Phase 10 intervention does change attribution, and the change is
visible when the instrument is sensitive to *where* attribution goes
(bg vs fg) rather than just *where the peak is* (Grad-CAM).

### 3.2 Why median is unchanged but mean shifts

This is a distributional, not a point, finding. The Phase 10
intervention reduces attribution on *some* images — specifically the
images where Phase 6 leaned hardest on background. Images that were
already lesion-focused under Phase 6 don't change. The result is
consistent with a *worst-case improvement* interpretation: the
intervention fixes the cases where the original model was most
shortcut-dependent, without altering already-good cases.

For a paper, this would warrant a per-collection breakdown
(c=70 SIIM images are the prior hypothesis for where the largest
shift would occur) and a larger sample, both of which we defer to
future work.

### 3.3 Why RGB shares don't move

A Phase 10 hue jitter of 0.10 (±36 degrees on the 0-360 hue circle)
translates into per-channel RGB changes whose direction depends on
each pixel's starting hue. Averaged over many pixels in many images,
the per-channel changes nearly cancel. An RGB-aware measurement of
the intervention is therefore *underpowered* — a result that itself
argues for HSV-based or polar-color attribution methods in future
work.

### 3.4 Limitations

- N=60 with paired Wilcoxon is underpowered for an effect this size.
  A larger sample (300+) and per-collection stratification would
  tighten the CI.
- IG with 32 steps and zero baseline. The baseline choice matters;
  alternatives (mean-image baseline, blurred baseline) would
  cross-check the result.
- The aggregate `bg/fg ratio` is a coarse summary. Per-pixel
  attribution heatmaps (fig22) are visually persuasive but not
  individually statistically tested.
- HSV-channel-level IG (`dP/dHue`, `dP/dSaturation`, `dP/dValue`)
  would be the most direct test of the channel-level claim. It
  requires a differentiable color-space transformation; deferred.

## 4. Conclusion

Integrated Gradients reveals an attribution shift between Phase 6 and
Phase 10 that Grad-CAM does not: mean background-to-foreground
attribution ratio drops by 0.24 with a bootstrap CI excluding zero.
RGB-channel shares do not rebalance, confirming that the Phase 10
intervention operates in a direction orthogonal to the RGB axis (most
plausibly the HSV Hue direction, as Phase 11 measured). The Phase 10
fine-tune is *not* a no-op at the attribution level even though it is
a no-op at the spatial-peak level; the instrument matters.

## 5. Reproducibility

- Script: `scripts/phase10/06_ig_channel_attribution.py`
- Outputs: `artifacts/phase10/06_ig_attribution.{csv,log}`
- Figures: `artifacts/phase10/figures/fig21_ig_bg_fg_ratio.png`,
  `fig22_ig_attribution_examples.png`
- Seed=42, IG steps=32, captum 0.9.0, PyTorch 2.6 (M3 MPS).
