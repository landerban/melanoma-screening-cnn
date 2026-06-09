# Phase 9 — Research Gap Analysis

Companion to `01_literature_review.md`. Identifies the precise gap our
work fills, explains why filling it matters scientifically and clinically,
and previews how Phase 9 fills it. This document is *the answer to* "why
are you doing this?" — the response that has to land in front of an
adversarial reviewer.

---

## 1. The Three Holes in the Existing Literature

The literature review surfaced three orthogonal capabilities that *no
single prior work* combines:

```
            ┌─────────────────────────────┐
            │   Multi-shortcut decomp.    │
            │   (≥ 4 distinct artifacts)  │
            └─────────────┬───────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
              │           │           │
   ┌──────────▼─┐  ┌──────▼──────┐  ┌─▼────────────┐
   │ Per-shortcut│  │ Cross-      │  │ Shortcut → OOD│
   │ causal      │  │ collection  │  │ gap attrib.   │
   │ attribution │  │ comparison  │  │               │
   └─────────────┘  └─────────────┘  └───────────────┘
```

**Hole 1 — Multi-shortcut causal decomposition.** Existing dermatology
shortcut work tests *one* artifact at a time (Nauta 2022: colour patches;
FastDiME 2023: rulers; Winkler 2019: skin markings) or treats artifacts
*holistically* (MaskMedPaint 2024; Bissoto 2019). No prior work
*simultaneously* estimates per-artifact contributions while controlling
for the presence of the others. This is the Shapley-style decomposition
problem and it is unanswered for skin.

**Hole 2 — Cross-collection shortcut differential.** ISIC 2019 documents
a 23-point balanced-accuracy gap between HAM10000 (82.0%) and BCN20000
(58.8%). The gap is attributed in the literature to "BCN20000 is more
clinically realistic." This is descriptive, not mechanistic — no paper
attributes the gap to per-artifact differences in shortcut prevalence
across the source collections.

**Hole 3 — Shortcut → OOD gap attribution.** Multiple papers report the
dermoscopy → smartphone-clinical OOD gap (our Phase 7b: AUC 0.95 →
0.81). Multiple papers report shortcut presence in dermoscopy. *Nobody*
connects them: how much of the 0.14-AUC OOD drop is explained by
shortcuts that disappear on phone-camera input vs. genuine lesion-feature
distribution shift? Without this attribution, the dermatology AI field
cannot prescribe *which* shortcuts to fix to recover OOD performance.

---

## 2. Why Filling Each Hole Matters

### Hole 1 — Scientific value

The single-shortcut paradigm forces a *categorical* claim: "the model
uses rulers." A multi-shortcut decomposition yields a *graded* claim:
"the model's decisions are 18% attributable to vignette, 12% to ruler,
3% to ink, and 67% to lesion features." The graded claim:

- Guides *prioritization*: which shortcut is the highest-leverage fix?
- Enables *additivity checks*: do the per-shortcut contributions sum to
  the total shortcut effect, or do interactions exist?
- Provides *interventional* attribution (Causal Shapley), which is the
  do-calculus-aligned semantics rather than an associative correlation.

### Hole 2 — Clinical value

Cross-collection shortcut differential matters for *deployment*. If a
hospital's imaging protocol resembles BCN20000 (heavier vignette,
specific colour cast), and we know vignette contributes 18% of the
model's decision in BCN-distributed test data, then the deployment site
needs *vignette-specific* re-calibration — not a generic "watch out for
OOD." The current literature gives no such per-site, per-artifact
prescription.

### Hole 3 — Methodological value

The shortcut → OOD chain is the *bridge claim*. Shortcut-learning
papers measure the cause; OOD papers measure the symptom; nobody connects
them quantitatively. A rigorous chain attribution turns shortcut auditing
into a *predictive* tool: "this model will lose X recall when deployed
on a population that lacks artifact Y." That is the deliverable a
regulatory pre-market submission would want.

---

## 3. Why the Gap Has Persisted

Three structural reasons explain why no one has done this yet:

**(i) Compute & engineering cost.** Multi-shortcut Shapley requires
$O(2^n)$ inpainting configurations per image. With n=5 artifacts and
N=500 test images, that is 16,000 inpainting passes. This was infeasible
before LaMa (Suvorov et al. 2022) made high-quality dermoscopy inpainting
cheap.

**(ii) Annotation cost.** Per-artifact masks (vignette region, ruler
region, ink region) require expert labelling. We solve this with a
*hybrid* protocol: automatic detection for vignette (low-pixel
thresholding) and hair (Otsu + morphology); manual labelling for ruler
and ink on a smaller sub-sample.

**(iii) Disciplinary partition.** Shortcut-learning researchers (Geirhos,
Bissoto) live in a different community than OOD-generalization
researchers (domain generalization, calibration). The two literatures
cite each other sparsely. Our chain-attribution work straddles them.

---

## 4. The Phase 9 Position — One Paragraph

> Phase 9 audits the *already-trained* EfficientNet-B0 melanoma
> classifier from Phases 4–6 by (i) localizing per-image attention
> across five candidate shortcut classes (vignette, ruler, ink markings,
> hair, collection-signature colour cast) via Grad-CAM with Score-CAM
> sensitivity checks, (ii) generating causal counterfactuals via LaMa
> inpainting for each shortcut class, separately and in combinations to
> support a Causal Shapley decomposition, and (iii) regressing the
> per-shortcut attribution scores against the cross-collection AUC
> heterogeneity and the dermoscopy → smartphone OOD AUC gap. The output
> is a quantitative chain: *per-artifact contribution → cross-collection
> heterogeneity → OOD performance gap*, with bootstrap confidence
> intervals at each link.

---

## 5. Scope Boundaries — What We Do *Not* Claim

To pre-empt reviewer pushback:

- **We do not claim a new training method.** This is an audit, not a
  debiasing intervention. The model is fixed. (MaskMedPaint claims a
  training method; we do not compete on that axis.)

- **We do not claim a new CAM method.** Grad-CAM is the inherited tool;
  Score-CAM is used only as a faithfulness sensitivity check. Our
  contribution is the *use* of attention + counterfactuals together in
  a decomposition framework, not a new attention algorithm.

- **We do not claim that LaMa inpainting is artifact-free.** Inpainting
  itself introduces a counterfactual-image artifact. We treat this as a
  *measurement uncertainty* with explicit visual QA and a
  *plausibility-stratified* analysis (drop inpainting failures from the
  causal estimate).

- **We do not claim generalization beyond dermoscopy.** The audit
  framework (Phase 9 Part B) is *positioned* for transfer to other
  multi-site medical imaging tasks, but we run it only on skin.

- **We do not claim subgroup fairness analysis.** Skin-tone disparities
  (Daneshjou 2022) are a separate axis; the artifact axis we study is
  *acquisition-protocol-driven*, not demographic.

---

## 6. The Critical Question We Must Answer

If asked by an adversarial reviewer:

> *"Bissoto et al. already showed in 2019 that ISIC models exploit
> non-lesion features. Nauta et al. already showed in 2022 that you can
> inpaint a single shortcut and measure the effect. What is genuinely
> new here?"*

Our one-sentence answer:

> Bissoto demonstrated the *existence* of shortcuts holistically;
> Nauta demonstrated the *causal effect* of one shortcut in isolation;
> Phase 9 demonstrates the *joint causal decomposition* of five
> shortcuts simultaneously, connects each to a cross-collection AUC
> differential, and attributes a measured fraction of an OOD performance
> gap to specific shortcuts — a chain of inference no prior work has
> closed.

This is the claim we are accountable to. Phase 2 (hypotheses + protocol)
operationalizes each link of the chain into a falsifiable prediction.

---

## 7. Hand-Off to Phase 2

Phase 2 must commit, *in advance of any data being looked at*, to:

- The exact list of candidate shortcuts to test (n=5 as outlined; rule
  out hair if data inspection shows hair is rare).
- The per-shortcut detection/masking protocol (automatic vs manual vs
  hybrid).
- The Shapley sampling scheme (full enumeration with n≤5 is feasible;
  document the choice).
- The statistical test for each hypothesis (paired t-test,
  Wilcoxon signed-rank, bootstrap CI).
- The pre-registered *prediction* under each hypothesis — including the
  pattern of results that would *falsify* each claim.

This pre-registration discipline is the difference between an
exploratory analysis (low evidentiary weight) and a confirmatory analysis
(high evidentiary weight). It is what makes the Phase 9 contribution
defensible at a doctoral level.
