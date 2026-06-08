# Phase 10 — Color-Invariant Fine-Tune Intervention (preregistration)

**Locked before fine-tune is executed. Hypotheses + protocol + falsification criteria committed first.**

## Position

Phase 9 *diagnosed* a non-spatial color-cast shortcut driving ≈ 70% of
SIIM-2020's apparent AUC (PF#1) and 66% of the cross-collection AUC
gap (PF#3). The dominant *causal* shortcut is non-spatial color drift
(PF#2: mean ΔP = +0.10 on counterfactually color-normalized images).

Phase 10 *tests a fix*. We hypothesize that a *color-channel augmentation*
during a brief fine-tune (which Phase 6 deliberately excluded — see
README §"Augmentations: hue and saturation jitter are deliberately
excluded because color is diagnostic") reduces the model's reliance on
collection-level color cues *without sacrificing within-collection
lesion discrimination*.

**Novelty.** Bissoto 2019 / Nauta 2022 *diagnose* spatial artifact
shortcuts; MaskMedPaint 2024 augments with *holistic* counterfactual
images. No prior work *intervenes* on the *specific non-spatial color
shortcut* we identified, nor reports a measured shortcut-reduction
effect against a Phase 9-style causal-Shapley audit.

## Hypotheses (preregistered)

### H6 — Color-invariant fine-tune preserves in-distribution AUC
Predict test AUC ∈ [0.93, 0.96]. The pre-Phase-10 model's test AUC was
0.9513; we allow ±0.02 around that as a "preserved" band. Outside this
band the intervention is rejected.

Statistical test: held-out test cohort (9,186 images), single ROC-AUC
point estimate + B=5000 bootstrap CI. Supported iff lower CI ≥ 0.93.

### H7 — Color-cast counterfactual ΔP drops by ≥ 50%
Phase 9 measured color-cast ΔP = +0.104 (95% CI [+0.076, +0.134]).
Predict the Phase 10 model's color-cast ΔP ≤ +0.05 (50% reduction)
with a 95% CI not overlapping the Phase 9 CI.

Statistical test: re-run `scripts/phase9/07_counterfactual.py` with
the Phase 10 ckpt; paired-bootstrap on (P_phase10, P_phase6) per
image; report Δ in ΔP and its CI.

Falsification: Phase 10 ΔP > +0.075 OR CIs overlap.

### H8 — c=70 balanced AUC rises by ≥ 0.10
Phase 9 PF#1 measured c=70 balanced AUC = 0.548. Predict the Phase 10
model's c=70 balanced AUC ≥ 0.65 with the bootstrap CI strictly
greater than 0.50 (above chance).

Statistical test: re-run `scripts/phase9/04_model_inference.py` on the
same 300-image analytical sample; per-collection AUC + bootstrap CI.

Falsification: Phase 10 c=70 AUC < 0.60 OR CI includes 0.50.

### H9 — Cross-collection AUC gap shrinks by ≥ 30%
Phase 9 PF#3 measured the (c=212 − c=70) gap at 0.434 (balanced).
Predict the Phase 10 gap ≤ 0.30 (30% reduction).

Falsification: gap > 0.35.

### H10 (negative / safety) — OOD recall does not catastrophically drop
Phase 9 Phase 7b/PF#4 measured OOD AUC = 0.836, OOD recall @ 0.661
= 0.79 (balanced 50/50). Predict Phase 10 OOD recall ≥ 0.70 on the
same 200-image OOD sample.

This is a safety check: we do not want the color-invariant
intervention to *destroy* OOD generalization by stripping legitimate
melanoma-relevant color cues (dark pigmentation, irregular hue
distribution).

Falsification: OOD recall < 0.65.

## Method

### Architecture
Identical to Phase 6: EfficientNet-B0 backbone + MLP head
(1280 → 256 → 1). Loaded from `best_model.pth`.

### Augmentation diff vs Phase 6

Phase 6 transform (val mode unchanged):

```
RandomResizedCrop(384, scale=(0.8, 1.0))
RandomHorizontalFlip()
RandomVerticalFlip()
RandomRotation(degrees=15)
ColorJitter(brightness=0.2, contrast=0.2)   ← Phase 6
```

Phase 10 transform — *adds* HSV jitter on top:

```
[ above transforms ]
ColorJitter(brightness=0.2, contrast=0.2,
            saturation=0.15, hue=0.05)       ← Phase 10 ADDS
```

Rationale for jitter magnitudes:
- hue=0.05: ≤ 18° in HSV. Subtle. ISIC collections differ in hue cast
  by typically ~10°. We jitter on the same magnitude so the model
  cannot lock onto the collection-specific hue mean.
- saturation=0.15: ±15% saturation. Allows the model to learn
  pigmentation as a feature while preventing collection-specific
  saturation distribution memorization.

The Phase 6 deliberate exclusion of hue/sat jitter was justified by
"color is diagnostic for melanoma." Phase 10's claim is that *small*
hue/sat jitter still allows the model to learn diagnostic color
features (the within-class variance ≫ the jitter magnitude) but
prevents *collection-level* color signature memorization.

### Training protocol

Brief fine-tune of the Phase 6 model:

- Base ckpt: `best_model.pth` (Phase 6, AUC 0.9513)
- Optimizer: AdamW, lr=1e-5 (10× smaller than Phase 6's Stage-2 head
  lr, to prevent destabilizing the strong base model)
- Loss: same focal loss (α=0.85, γ=2.5)
- Sampler: same WeightedRandomSampler
- Epochs: 5 (Stage-2-only fine-tune)
- Batch size: 96
- Patient-level split: identical to Phase 6 (seed=42)

5 epochs is intentionally short. If the intervention works, the
shift in shortcut reliance should be detectable in 5 epochs of
fine-tuning. If it requires 30+ epochs, the augmentation is too weak.

### Compute budget

- A100 (Colab): ~15 min per epoch with 39,647 training images at
  batch 96 → ~75 min total + 10 min eval = **~90 min**.
- M3 MPS fallback: ~6 hr (60% slower).

### Random seed
seed=42 throughout, bit-exact match to Phase 6's seeding regime.

## Output artifacts (post-fine-tune)

- `artifacts/phase10/best_model_color_invariant.pth` (rich-dict ckpt
  with cfg + base ckpt's git_hash + Phase 10 modifications)
- `artifacts/phase10/01_finetune.log` (per-epoch trajectory)
- `artifacts/phase10/02_phase9_replication/` (re-run Phase 9
  measurements with the new ckpt)
- `docs/phase10/02_results.md` (paper-format Phase 10 results)

## Falsification decision tree

- **All 5 H's supported** → Phase 10 is a clean novel result. Paper
  Section "Intervention" added.
- **H6 falsified** (AUC drops > 0.02): intervention destroyed base
  performance; intervention rejected.
- **H7 supported, H6 marginal** (AUC 0.92): trade-off; report as
  "shortcut-reduced model at minor AUC cost", recommend mid-band as
  the Pareto-frontier choice.
- **H6 supported, H7 falsified** (ΔP unchanged): the augmentation
  was too weak; recommend larger jitter (hue=0.10) as future work.
- **H10 falsified** (OOD recall drops): the safety case fails;
  document that color invariance *destroys* legitimate diagnostic
  color signal; novel negative result, still publishable.

The pre-registration commits the falsification logic in advance so
post-hoc rationalization cannot drift the conclusion.

## Why this is a real research contribution (not just measurement)

Phase 9 was a *measurement* contribution — quantifying shortcut
contributions to AUC heterogeneity. Phase 10 is a *method* contribution
— proposing a training-time intervention and measuring whether it works.

The pair (Phase 9 → Phase 10) is the *audit → fix* arc that
distinguishes a re-implementation from original research. We report
both directions of the fix: *what works* (color jitter reduces
shortcut reliance per Phase 9's metrics) and *what fails* (if anything
in the safety/OOD axis).
