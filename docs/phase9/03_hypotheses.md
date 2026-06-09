# Phase 9 — Pre-registered Hypotheses

Locked *before* any new data is collected or evaluated. Each hypothesis
carries (1) the precise prediction, (2) the statistical test, (3) the
effect-size threshold below which we treat the result as null, (4) the
falsification criterion. Drift after this point — modifying H to fit
the data — is recorded in `08_deviations.md` if it occurs.

Notation. *In-dist* = held-out test cohort from Phase 7a (N=9,186, 3
collections). *OOD* = PAD-UFES-20 from Phase 7b (N=1,568, smartphone
clinical). *Model* = `best_model.pth` from Phase 6.

---

## H1 — Attention is non-uniform across collections

**Claim.** Model attention (Grad-CAM peak location) lands on candidate
shortcut regions at *significantly different* rates across the three
training collections.

**Prediction.** Across the three collections {HAM10000, SIIM 2020,
BCN20000}, the proportion of test images whose Grad-CAM peak (top-5%
mass centroid) falls inside a candidate shortcut region (vignette OR
ruler OR ink OR hair) differs by at least 10 percentage points between
the highest and lowest collection.

**Concrete numeric prediction.**
- BCN20000 shortcut-attention rate ≥ 20%
- HAM10000 shortcut-attention rate ≤ 10%
- Δ(BCN − HAM) ≥ 10 percentage points

**Statistical test.** Three-way proportion comparison via
χ²-test of independence on the 3 × 2 contingency table (collection ×
{peak-on-shortcut, peak-on-lesion-or-skin}). Significance threshold
α = 0.01 (Bonferroni-corrected across H1–H3).

**Falsification.** Δ < 5 pp between any two collections *and* χ² p > 0.05
→ H1 falsified. Implication: aggregate AUC gap in Phase 7c is *not*
mediated by attention pattern differences; another mechanism explains
the gap.

**Why this prediction.** Phase 7c reports AUC gap of 4.5 percentage
points between HAM (0.92) and BCN (0.87). If attention is uniform
across collections, the gap must be explained by *something other than
shortcut focus* (e.g., lesion morphology heterogeneity in BCN). A Δ ≥
10pp shortcut-attention gap is the minimum that makes shortcut focus a
*plausible* mediator of the AUC gap.

---

## H2 — Counterfactual inpainting causally lowers malignant probability for benigns

**Claim.** Removing a candidate shortcut from a benign image (where the
shortcut is present) causally lowers the model's malignant probability.

**Prediction (per shortcut s ∈ {vignette, ruler, ink, hair, color-cast}).**
On the sub-cohort of *benign* images containing shortcut s, inpainting s
out using LaMa produces a paired prediction change ΔP(malig) with:

- **Vignette** (BCN): mean ΔP ≤ −0.10, paired Wilcoxon p < 0.01
- **Ruler** (HAM): mean ΔP ≤ −0.05, paired Wilcoxon p < 0.05
- **Ink** (any): mean ΔP ≤ −0.05, paired Wilcoxon p < 0.05
- **Hair** (any): expected null. Pre-registered as a negative control.
  If hair shows ΔP ≤ −0.05 we treat H2-hair as unexpectedly supported
  and flag it as a finding rather than a noise result.

**Statistical test.** Paired Wilcoxon signed-rank on (original, inpainted)
prediction pairs. Effect size reported as median ΔP and Cohen's *d* for
paired samples. Bootstrap 95% CI on the median (B = 5000 resamples,
seed = 42).

**Falsification (per shortcut).** |median ΔP| < 0.02 AND Wilcoxon p > 0.1
→ H2-{shortcut} falsified. Implication: this artifact is not causally
load-bearing for the model's prediction; its correlation with malignancy
in the training set was not internalized as a feature dependence.

**Negative control rationale.** Hair appears uniformly across benign and
malignant lesions (no training-set correlation in literature). It serves
as a method-validity check: if the inpainting pipeline produces a non-
null effect on hair removal, the result casts doubt on the validity of
the positive shortcuts' effects too (inpainting-introduced artifact).

---

## H3 — Per-shortcut contributions decompose the cross-collection AUC gap

**Claim.** The cross-collection AUC heterogeneity (Phase 7c: HAM 0.92,
SIIM 0.88, BCN 0.87) is *measurably mediated* by per-shortcut differences
in attention rate and per-shortcut causal weights estimated under H2.

**Operationalization.** Compute the Causal-Shapley-style attribution
*A_c(s)* of shortcut *s* to the predicted-probability shift in
collection *c*:

```
A_c(s) = average over images i in collection c of
         (ΔP_i if shortcut s present and removed,
          0    otherwise)
```

Define the *shortcut-explained AUC gap* between two collections as
the AUC gap that *would close* if both collections' images were
counterfactually shortcut-removed.

**Prediction.** At least 30% of the (HAM − BCN) AUC gap of 4.5 pp closes
under joint shortcut removal:

- Closed gap ≥ 1.35 percentage points → H3 supported.
- Closed gap < 1.35 pp → H3 falsified.

**Statistical test.** Bootstrap (B = 1000) the closed gap estimate by
resampling images within collections; report 95% CI. Significance via
the bootstrap CI not crossing zero from below.

**Falsification implication.** If H3 is falsified while H1 and H2 are
supported, the cross-collection AUC gap is *not primarily* due to
shortcut differences — it is due to lesion-feature-distribution
heterogeneity (BCN truly contains harder cases). This is also a
publishable finding, just a different one.

---

## H4 — Shortcut removal partially recovers OOD AUC

**Claim.** Inpainting-removed in-distribution training-style images,
when used as a *re-calibration* probe for the OOD operating point,
predict a recoverable fraction of the dermoscopy → smartphone OOD gap.

**Important methodological caveat.** We *cannot* re-train the model in
Phase 9 (out of scope). H4 is therefore an *attribution* claim, not a
*correction* claim. The prediction is:

> The total causal magnitude of shortcut contributions in the
> in-distribution test set (sum over shortcuts of expected |ΔP|)
> correlates positively with the per-image OOD prediction *deviation*
> from in-distribution baseline.

**Prediction.** Spearman correlation ρ between in-dist shortcut-induced
|ΔP| magnitude and OOD per-collection AUC drop ≥ 0.4. CI excludes 0.

**Statistical test.** Permutation test on Spearman ρ (10,000 permutations).
Bootstrap 95% CI on ρ.

**Falsification.** ρ < 0.2 OR CI crosses zero → H4 falsified. Implication:
the dermoscopy → smartphone OOD gap is *not* substantially mediated by
the shortcuts identified in H1–H2; it is mediated by other factors
(lighting, focus, polarization, framing, lesion-size-prior). This is
again a publishable finding — it would *negatively* attribute the OOD
gap, narrowing the causal hypothesis space.

---

## H5 — Score-CAM sensitivity check

**Claim (auxiliary).** The H1 result (attention non-uniformity across
collections) replicates qualitatively when Grad-CAM is replaced with
Score-CAM.

**Prediction.** The collection-ranking of shortcut-attention rates
(highest → lowest) is preserved under Score-CAM. Rank correlation
(Spearman) between Grad-CAM and Score-CAM per-collection shortcut rates
≥ 0.8.

**Falsification.** Rank disagreement (Spearman ρ < 0.5) → the H1 result
is CAM-method-dependent and should be disclosed as such. We do *not*
treat this as falsifying H1 itself; we report Score-CAM as a sensitivity
disclosure.

---

## Joint Decision Rules

- **All 4 primary hypotheses (H1, H2-vignette, H3, H4) supported** →
  full chain attribution; primary slide deck claim.
- **H1 + H2 supported, H3 falsified** → "attention differs, causal
  weights differ, but cross-collection AUC gap is not shortcut-mediated."
  Re-frame as a *negative* contribution to a shortcut-only explanation.
- **H1 supported, H2 falsified** → "attention is non-uniform but does
  not cause prediction shifts." Implies attention is *epiphenomenal* to
  the model's decisions on this task. Strong methodological finding;
  reframes slide 9 of the main deck.
- **H1 falsified** → cross-collection AUC gap has nothing to do with
  shortcut-pattern differences. We then ablate H2/H3/H4 as designed and
  report the negative finding as the primary result.

**Pre-committed disclosure.** Any post-hoc redefinition of H1–H5 after
data inspection is logged in `08_deviations.md` with timestamp and
justification. This is the discipline that separates a confirmatory
analysis from an exploratory one.

---

## Power analysis (informal)

For H2-vignette (the most concrete prediction):
- Target effect size: median ΔP = 0.10 (anchored on Nauta 2022's
  sensitivity-drop magnitude scaled down for benign→benign-inpainted
  rather than benign→malignant-with-patch-inserted).
- Standard deviation of ΔP under H0: estimated from prior Phase-7 logit
  distribution variance ≈ 0.15.
- Cohen's *d* ≈ 0.67 (medium-large effect).
- For 80% power at α = 0.01 (two-sided) with paired Wilcoxon,
  required N ≈ 35 paired observations.

We will collect N ≈ 60–100 benign-with-vignette images per collection
to comfortably exceed power for H2 and to allow stratified sub-analyses
for H3.

For H1 (3×2 χ²):
- At target rates 25% (BCN) vs 8% (HAM) with N=150 per collection, power
  at α=0.01 is ~99%. The sample is N=300 total across both relevant
  collections; per-collection N=100–150 in the analytical step.
