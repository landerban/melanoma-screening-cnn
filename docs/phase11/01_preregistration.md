# Phase 11 — HSV Channel Decomposition of the Color-Cast Shortcut

**Pre-registered before any Phase 11 measurement runs.**

## Position

Phase 9 identified non-spatial color cast as the dominant causal
shortcut (PF#2: mean ΔP = +0.104, p < 0.0001) but operationalized it
as a *single* Lab-space mean shift. The "color cast" finding is
mechanistically opaque — does the model attend to hue (the
collection-specific colour temperature), saturation (the optical
profile), or value (the overall lighting)?

Phase 11 decomposes the shortcut by channel. Each candidate channel
(H, S, V) is normalized in isolation; the resulting ΔP measures that
channel's marginal contribution.

## Novelty

No prior dermatology shortcut work (Bissoto 2019/2022, Nauta 2022,
FastDiME 2023, MaskMedPaint 2024) decomposes a color-cast shortcut
by HSV channel. The closest is Bevan & Atapour-Abarghouei 2022's
analysis of overall colour distribution differences. Phase 11 is
finer-grained per-channel attribution.

## Hypotheses

### H11 — One channel dominates
At least one of {H, S, V} captures ≥ 60% of the total color-cast ΔP
(Phase 9 measured the full transform at +0.104). Prediction: hue is
the dominant channel — Bevan 2022 documents collection-specific hue
casts.

Falsification: max(|ΔP_H|, |ΔP_S|, |ΔP_V|) / |ΔP_all| < 0.50.

### H12 — Additivity holds approximately
Sum of single-channel |ΔP|s falls within ±20% of the joint |ΔP|.

Specifically: |Σ_c ΔP_c − ΔP_all| / |ΔP_all| ≤ 0.20.

Falsification: discrepancy > 0.30 → significant interaction between
channels.

### H13 — Channel-specific cross-collection profile
Predicts that the *dominant* channel differs per collection because
each collection's acquisition pipeline emphasizes different cues.
E.g.: c=249 (BCN20000) has stronger hue-cast than c=212 (HAM10000).

Statistical test: per-collection mean ΔP per channel; one-way ANOVA
across (channel, collection).

## Method

For each of the 300 analytical-sample images:

1. Compute the analytical-sample-mean Lab-image-mean (same anchor as
   Phase 9).
2. For each channel c ∈ {H, S, V}:
   - Convert image to HSV.
   - Replace channel c with target value (= sample-mean of channel c).
   - Leave other two channels unchanged.
   - Convert back to BGR.
   - Forward through Phase 6 ckpt → P_malig_c.
3. Compare to:
   - `original` (Phase 9's baseline).
   - `no_colorcast_all` (Phase 9's full Lab-mean shift; replicated as
     reference).
4. Compute per-image (ΔP_H, ΔP_S, ΔP_V, ΔP_all).

Statistical tests: paired Wilcoxon for each channel; per-collection
bootstrap 95% CIs (B=5000).

## Output

- `artifacts/phase11/01_hsv_decomposition.csv` (per-image P_malig per
  channel-normalization)
- `artifacts/phase11/01_hsv_decomposition.log` (per-channel ΔP +
  per-collection breakdown + H11-H13 outcomes)
- Figure: per-channel ΔP forest plot (`artifacts/phase11/figures/fig11.png`)
