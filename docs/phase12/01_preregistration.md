# Phase 12 — Lesion-Aware Color Normalization (LACN)

**Pre-registered before any Phase 12 measurement runs.**

## Position

Phase 9 / Discussion §4.3 noted a confounding interpretation of the
color-cast shortcut: the Lab-mean shift normalizes *all* of an
image's colour, including the *lesion-interior* colour that
encodes legitimate diagnostic signal (Phase 6 explicitly excluded
hue/sat jitter for this reason). So a +0.10 ΔP on color-cast
removal mixes:

1. *Removing the collection-specific background colour signature*
   (what we want, a true shortcut)
2. *Destroying the lesion-interior pigmentation/colour cues*
   (an unwanted side-effect)

Phase 11 narrowed the cue to hue but did not separate (1) and (2).
Phase 12 separates them spatially: normalize *only* the
background region (outside the Otsu lesion mask), leaving the lesion
interior untouched.

## Novelty

The LACN intervention is novel relative to:
- **Phase 9's Lab-mean shift** (which normalizes everything jointly).
- **Phase 11's HSV-channel decomposition** (which separates channels but
  not space).
- Prior dermatology shortcut work (Bissoto, Nauta, FastDiME,
  MaskMedPaint) treat colour transforms as global. We are the first to
  apply *spatially-aware* colour normalization to a per-image
  counterfactual on a dermatology classifier.

LACN is also a *test-time intervention* — no re-training needed. It
complements Phase 10's training-time intervention.

## Hypotheses

### H14 — LACN (background-only) reduces ΔP magnitude
Prediction: LACN's mean ΔP is smaller in magnitude than Phase 9's
full Lab shift, AND retains the *direction* (positive — color cast
removal lifts P_malig). Specifically:

- 0.03 ≤ mean ΔP_LACN ≤ 0.08 (vs Phase 9's +0.104)
- 95% CI does not cross zero

Falsification: LACN ΔP outside [0.03, 0.08] band OR CI crosses zero.

### H15 — Lesion-only normalization is *harmful*
Prediction: replacing the lesion-interior colour with sample-mean
colour decreases P_malig substantially on *malignant* images (the
lesion colour was carrying real diagnostic information).

- For malignant images: mean ΔP_lesion-only ≤ -0.05.

Falsification: |ΔP| < 0.02 for malignant images.

### H16 — Per-collection LACN profile matches PF#3 direction
Prediction: c=70 (SIIM) shows the largest LACN ΔP, consistent with
PF#3's finding that c=70's shortcut reliance is highest. Specifically,
c=70's LACN ΔP > 1.5 × c=212's.

## Method

For each image i in the analytical sample with a *reliable* lesion mask
(Phase 9's lesion-mask reliability heuristic; ~92% of N=300):

1. Compute target Lab-mean (= analytical-sample-mean Lab, same anchor
   as Phase 9).
2. Generate 4 counterfactuals:
   - `full_norm`: full Lab-mean shift (= Phase 9's no_colorcast,
     reference)
   - `LACN_bg_only`: shift only the background region (outside the
     Otsu lesion mask). Lesion-interior colour preserved.
   - `lesion_only_norm`: shift only the lesion region. Background
     preserved.
   - `original`: untouched (reference)
3. Forward through Phase 6 ckpt; record P_malig.
4. Compute ΔP for each variant.

Per-collection bootstrap CIs (B=5000). Wilcoxon on (LACN, original)
paired.

## Output

- `artifacts/phase12/01_lacn.csv`
- `artifacts/phase12/01_lacn.log`
- Figure: per-collection LACN bar chart (`artifacts/phase12/figures/fig12.png`)
- Companion analysis combining Phase 11 and Phase 12 into the *paper-
  publishable* "color cast = hue cast in background only" framing.

## Decision tree

- **H14 supported** → LACN works as a *test-time fix*; reduces shortcut
  effect while preserving diagnostic signal. Highly publishable.
- **H14 falsified, H15 supported** → the color cast effect *requires*
  destroying lesion colour to manifest, so "color cast" is mostly a
  lesion-colour effect (problem for Phase 9's interpretation). Re-frame.
- **H15 falsified** → lesion colour is not load-bearing for malignancy
  prediction, contradicting standard dermatology AI lore. Strong result
  if true.

The pre-registration commits the falsification logic; post-hoc
re-interpretation is forbidden.
