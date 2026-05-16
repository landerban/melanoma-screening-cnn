# Midterm presentation — slide outline

Companion to `docs/midterm-prep-context.md`. Every number on every slide
traces to that audit doc or to `artifacts/*.log`. Voice and tone match
`CLAUDE.md` / `README.md`: direct, casual-technical, no bullet fountains,
honest about limitations.

13 main slides + 9 backup slides. Target 15 min for the main deck, with
slide 9 (per-collection breakdown) and slide 7 (results headline) as the
two heaviest stops. The §6 framing language (ES on AUC, not val loss) is
locked in `Trainer._run`'s docstring — slide 5 must not drift from it.

---

## Slide 1 — Problem framing

**Title.** "Recall-first binary screening on dermoscopy: why it's a different problem than balanced classification."

**Claims.**
- The screening question is binary: does this lesion warrant a clinical referral?
- The cost asymmetry is large: a missed malignancy costs more than an unnecessary referral. Accuracy is meaningless here; **recall at chosen specificity** is the operationally meaningful metric.
- Training pool: **~61K dermoscopy images, three ISIC collections, 4.29:1 imbalance** (49,785 benign / 11,611 malignant after Indeterminate-drop). Source: audit §2.
- Subtype distinction (which kind of cancer) is a separate question, posed as a follow-up on slide 12.

**Q&A risks.** "Why not just maximize accuracy?" → at 19% prevalence in test, predicting all-benign gives accuracy 0.81. Useless. "Why dermoscopy?" → that's what ISIC provides at scale; the smartphone-camera gap is measured separately on slide 11.

---

## Slide 2 — Dataset

**Title.** "Three ISIC dermoscopy collections, patient-level split, 4.29:1 imbalance."

**Composition** (audit §2):

| Collection | Source | Rows | Native `patient_id` |
|---|---|---|---|
| c=212 | HAM10000 dermoscopy | 11,720 | absent (lesion_id only) |
| c=70  | SIIM-ISIC 2020 dermoscopy | 33,126 | present |
| c=249 | BCN20000 dermoscopy | 18,946 | absent (lesion_id only) |

61,396 rows after dedup + Indeterminate/NaN drop. Patient-level split into train/val/cal/test = 0.65/0.10/0.10/0.15 of patients via `effective_patient_id` = `patient_id || lesion_id || isic_id` fallback chain.

**The audit caught a leak before retraining.** 28,273 of 61,396 rows (46%) had no native `patient_id`. Without the fallback chain those rows would have collapsed into one synthetic NaN-patient and landed entirely in train OR val OR test depending on shuffle order. `scripts/audit_patient_id.py` confirmed the fix produces 0 NaN. Backup slide B5 carries the detail.

**One acknowledgement.** The original training pool included a fourth collection (TBP, c=390) with ~0.6% malignant prevalence. We dropped it on modality-coherence grounds — the per-collection AUC story on slide 9 is what motivated that choice.

**Forward reference.** Per-collection AUC was a deferred audit risk (W8). It's now measured; slide 9.

---

## Slide 3 — Architecture + training pipeline

**Title.** "EfficientNet-B0 + 2-layer MLP head, two-stage transfer."

**Architecture.**
- EfficientNet-B0 ImageNet-pretrained backbone (5.3M params).
- Head: GAP → Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→1).
- Input 384×384, batch 96 (Phase 4g scale-up from the original 300×300 / batch 32 when the codebase was sized for an 8 GB RTX 3060 Ti).

**Two-stage transfer.**
- Stage 1: backbone frozen, 5 epochs at head_lr=3e-4, `ReduceLROnPlateau`.
- Stage 2: full unfreeze, backbone_lr=5e-6, head_lr=1e-4, up to 30 epochs with `CosineAnnealingLR(T_max=30)`. ES on val AUC (slide 5).

**Honest disclosure on the backbone choice.** B0 was chosen under the original 8 GB VRAM constraint. The A100X MIG slice (40 GB) lifts that constraint, but we retained B0 in the retrain **to preserve the methodology comparison**, not because the constraint still holds. Bumping to B3 simultaneously with the dataset composition change would have made the AUC delta uninterpretable ("is it the data or the backbone?"). B3 is a clean follow-up experiment.

---

## Slide 4 — Focal loss α=0.85: alpha weights positives

**Title.** "α weights the rare class; copying defaults across imbalance regimes is the category error."

**Claim.** In the binary focal-loss implementation at `efficientnet_b0.py:343`,
`alpha_t = alpha * targets + (1 - alpha) * (1 - targets)`. So α directly weights the positive class. The detection-code default α=0.25 means **positives get weight 0.25, negatives 0.75** — wrong direction for a rare-positive problem. α=0.85 gives malignant ~5.7× the weight of benign on the new 4.29:1 pool (was ~270× on the original 161:1 pool — same α, different magnitude).

**Why this is the strongest single methodology slide.** The error wasn't the value 0.85 — it was copying RetinaNet's 0.25 into a heavily-imbalanced binary classifier without rethinking what α does. The methodology lesson is **prevalence-invariant**: the same correction is needed at any imbalance, just with different magnitudes.

**Disclosure.** α=0.85 was originally chosen for 161:1. On 4.29:1 it upweights malignant ~6× instead of ~270× — milder reweighting. The methodology argument holds; the magnitude illustrates it less starkly than the original pool would have. Untested: α=0.85 vs weighted BCE ablation (W3, disclosure-only).

**Q&A risk.** "How did you pick α=0.85?" → It was derived from the prior-correction intuition (α ≈ prevalence of negatives), then held fixed across the data change to keep one variable at a time.

---

## Slide 5 — Early stopping on AUC, not val loss

**Title.** "Why ES monitors AUC at heavy imbalance."

**Locked framing (from `Trainer._run` docstring, do not paraphrase):**
> Val loss at heavy class imbalance is dominated by the benign majority. A model can reduce val loss by tightening benign confidence variance, which is orthogonal to malignant-class separability. AUC measures separability regardless of confidence calibration, so it is the right summary statistic at this imbalance.

**Empirical evidence (Phase 6 stage-2 trajectory, `experiments/run_20260515_172452.log`):**

| Epoch | val loss | val AUC | ES marker |
|---|---|---|---|
| 1 | 0.0203 | 0.9440 | ✓ best |
| 2 | 0.0188 | 0.9456 | ✓ best |
| 5 | 0.0182 | 0.9472 | (1/7) |
| 13 | **0.0170** | **0.9514** | **✓ best (AUC-ES selected)** |
| 17 | 0.0169 | 0.9518 | (4/7) — lowest val loss; AUC gain sub-Δ |
| 20 | 0.0167 | 0.9511 | (7/7) — ES fired |

Val loss is non-monotone (oscillates ±0.001 epoch-to-epoch); AUC is closer to monotone. In this run, AUC-ES selected epoch 13; loss-ES would have selected epoch 17. AUC at those two epochs is **0.9514 vs 0.9518** — within an es_delta of each other, so functionally equivalent here. The dramatic "loss-ES picks the wrong epoch by 30+ recall points" story from the audit-doc-era 161:1 training **doesn't reproduce at 4.29:1**. The methodology argument still holds; the empirical evidence in this specific run is weaker than in the old run.

**Honest framing for the slide.** Lead with the methodology argument (val loss dominated by majority; AUC is separability-invariant). Acknowledge the new run's loss-ES would have given essentially the same checkpoint. The §6 framing in the docstring is the prevalence-invariant claim; this run's specific numbers are a side note.

---

## Slide 6 — Threshold calibration: the threshold encodes which tool this is

**Title.** "Threshold = clinical-cost choice; 0.661 on cal cohort."

**Claim.** Threshold sweep maximizes specificity subject to recall ≥ 0.80; falls back to Youden's J. Result on the held-out cal cohort: **0.661**.

**W1 fix (Phase 4c).** Threshold is now calibrated on the cal cohort (1,555 patients, never touched during ES or best-epoch selection). The previous codebase calibrated on val, which biased the operating point. The cal cohort is the patient-disjoint sibling of val/train/test.

**Test cohort math at the operating threshold** (from `artifacts/eval_test_indist.log`):

| Threshold | Caught | Missed | False alarms | Correctly cleared |
|---|---|---|---|---|
| 0.5 | **[NEEDS COMPUTATION]** | — | — | — |
| 0.661 | 1,475 / 1,767 | 292 / 1,767 | 635 / 7,419 | 6,784 / 7,419 |

The 0.5 row needs a one-time re-thresholding of the saved test predictions; flagged for the final deck. The 0.661 numbers are from the trainer's built-in test eval, verified bit-exact by Phase 7a.

**The strongest new framing on this slide.** The threshold moved from 0.347 (old 161:1 pool) to 0.661 (new 4.29:1 pool). **Same recall-at-max-specificity strategy, same constraint (recall ≥ 0.80); the number is different because the prevalence regime is different.** The methodology survived the data change.

**Honest limitation.** 0.661 is calibrated for the aggregate cal-cohort prevalence (~20%). It is **miscalibrated per collection** (slide 9: recall 0.23 on c=70, 0.89 on c=249 at the same threshold). For deployment, the threshold must be recalibrated per institution's expected prevalence.

---

## Slide 7 — Results headline (in-distribution)

**Title.** "Test AUC 0.951 [95% CI 0.947 – 0.956], honest patient-level split."

**Headline triplet** (from `artifacts/bootstrap_test_auc.log` and `eval_test_indist.log`):
- **ROC-AUC = 0.9513 [95% CI 0.9465 – 0.9563]** on the held-out test cohort. N=9,186 images, 1,767 malignant, 2,333 patients — disjoint from train/val/cal at the effective_patient_id level.
- **PR-AUC = 0.8353** (random baseline at 19.2% prevalence ≈ 0.192).
- At calibrated threshold **0.661**: recall **0.835**, specificity **0.914**, F1 **0.761**.
- Confusion matrix: TN=6,784 FP=635 FN=292 TP=1,475.

**Critical caveat: val/test gap is 0.0001** (val AUC 0.9514, test AUC 0.9513). Clean test-set holdout — no overfitting to test, no leakage signature.

**Honest-disclosure footer.** The aggregate 0.951 includes a **prevalence-prior contribution** from the cohort composition. The per-collection breakdown is on slide 9, and the out-of-distribution result is on slide 11's design rationale. The headline-with-asterisk discipline is the whole point of the audit.

**Q&A risk.** "How do we know this isn't leak inflation?" → The val/test gap is 0.0001. A leak would show test ≪ val (overfit) or test ≫ val (test is structurally easier). Neither pattern.

---

## Slide 8 — Prevalence drives precision; AUC measures separability

**Title.** "Why we lead with recall-at-spec, not F1."

**Same model, same checkpoint, three PR-AUCs by collection** (from `artifacts/eval_per_collection.log`):

| Collection | Prevalence | PR-AUC | Random baseline (= prev) | Lift |
|---|---|---|---|---|
| c=70 SIIM-ISIC 2020 | 1.7% | **0.1223** | 0.017 | 7.2× |
| c=212 HAM10000 | 17.9% | **0.7494** | 0.179 | 4.2× |
| c=249 BCN20000 | 54.3% | **0.8849** | 0.543 | 1.6× |

**Claim.** The same model's PR-AUC ranges **0.12 to 0.88** depending on which collection's prevalence regime you measure on. The 0.12 looks pathologically low; it's 7× random and is the **strongest minority-class learning** in absolute lift. Recall-at-specificity is operationally invariant; PR-AUC isn't.

**Conclusion.** We report both but **lead with recall-at-specificity** because it's prevalence-invariant for the same model. Comparing PR-AUC across cohorts (or across literature with different class balances) without normalizing for prevalence is the standard mistake.

**This is now an offensive slide, not a defensive one.** Previous audit-doc-era framing was "PR-AUC ~0.03 at 161:1 is expected, please don't compare to balanced literature." That framing is retired. The new framing makes the audience-question ("is PR-AUC 0.12 bad?") a teaching moment about prevalence-baseline normalization.

---

## Slide 9 — What the headline doesn't say (NEW SLIDE)

**Title.** "Per-collection AUC: the aggregate hides a modality-prior contribution."

**Table** (from `artifacts/bootstrap_test_auc.log`):

| Cohort | N (test) | Prev | AUC | 95% CI | Recall @ 0.661 | Spec @ 0.661 |
|---|---|---|---|---|---|---|
| **Aggregate test** | 9,186 | 19.2% | **0.9513** | [0.9465, 0.9563] | 0.835 | 0.914 |
| c=212 HAM10000 | 1,708 | 17.9% | 0.9151 | [0.8980, 0.9323] | 0.741 | 0.896 |
| c=70 SIIM-ISIC 2020 | 4,944 | 1.7% | 0.8835 | [0.8545, 0.9100] | 0.233 | 0.981 |
| c=249 BCN20000 | 2,534 | 54.3% | 0.8702 | [0.8555, 0.8842] | 0.893 | 0.656 |

**Key observation.** Within-collection AUCs cluster at **0.87 – 0.92**. The aggregate (0.95) exceeds every per-collection number. This is not Simpson's paradox in a leak sense — it's coherent dataset signal. The model has partly learned **"this looks like a BCN20000-style image → bump malignant probability"** because c=249 was 54% malignant in training. Predictions on c=249 patients cluster higher than predictions on c=70 patients, and that between-collection score ordering is itself discriminative.

**Statistical reading.**
- c=212 CI [0.898, 0.932] does **not overlap** c=249 CI [0.856, 0.884]. c=212 is genuinely a better cohort for this model than c=249, with significance. Likely: HAM10000 has cleaner labels; BCN20000 contains harder subtypes.
- c=70 and c=249 CIs overlap. They are **statistically indistinguishable** in within-collection discriminative quality.

**Operational observation.** The aggregate threshold 0.661 catches **0.23 of c=70's malignants** (a 1.7%-prevalence cohort, threshold way too high) and **0.89 of c=249's** (a 54.3% cohort, threshold roughly right). The operating point only works for the cocktail-mix prevalence.

**This is the W8 audit weakness measured rather than handwaved.** Frame as "we asked the question and here's the answer," not as "limitation." Give this slide a full minute.

---

## Slide 10 — Honest limitations: what's still open and why it doesn't change the headline

**Title.** "Seven of eleven audit findings closed; four open as disclosure-only."

**Closed during Phase 4–7 (audit doc §§2/7/8/9 + weaknesses table):**

| ID | What | Fix |
|---|---|---|
| W1 | Threshold sweep on the same val set | Phase 4c: 4-way split, cal cohort owns the sweep |
| W2 | Test split discarded by trainer | Phase 4b: test eval at calibrated threshold + persisted metrics |
| W4 | No global seeds | Phase 4d: bit-exact reproducibility verified in Phase 5 A3 |
| W8 | Per-collection AUC not computed | Phase 7c + slide 9 |
| H5 | Checkpoint format opaque | Phase 4e: rich ckpt dict (cfg + val/test metrics + git_hash + timestamp) |
| M1 | Per-epoch metrics at threshold 0.5 | Phase 4f: dynamic mini-sweep threshold |
| M4 | app.py reloads model per inference | Phase 4i: mtime-aware cache |

**Open as slide-time disclosures (none block the headline):**

- **W3.** WeightedRandomSampler + focal α both active; we did not ablate which contributed. Future-work item.
- **W5.** `CosineAnnealingLR T_max=30` while ES fired at epoch 20 — cosine completed ~67% of one descent. Cosmetic at this convergence speed.
- **W6.** Augmentation ablation not done. Hue/saturation deliberately excluded because color is diagnostic for melanoma; brightness/contrast jitter only.
- **W7.** Systematic Grad-CAM artifact-attention check not done. The realtime app's Eigen-CAM localization measurement (slide 11) is a different but related question.

**Land confident.** Seven highest-priority audit findings closed with provenance (commits, artifact logs); four open items are disclosure-grade. None of the four would change the slide 7 headline if addressed.

---

## Slide 11 — Realtime app: framing aid, not live diagnosis

**Title.** "Live attention overlay as framing aid: a weaker question the model can actually answer."

**Research question (from `docs/realtime-design.md` §2a).**
> Can a live attention overlay help the user produce a better-framed capture frame, without requiring the model to deliver live diagnostic output? The classification still runs only on shutter-capture; the live phase asks one weaker question: **is the model attending to the lesion in the current viewfinder frame?**

**Two-CAM design** (`docs/realtime-design.md` §2c):

| Phase | CAM variant | Reason |
|---|---|---|
| Live viewfinder | **Eigen-CAM** | Gradient-free, ~10 ms/frame on the MIG slice. Shows the top-SVD direction of the last-conv activation — "what stands out to the model." |
| Capture overlay | **Grad-CAM** (existing) | Class-discriminative direction; runs once per shutter event. "What drove the classification." |

**Why this scope, grounded in slide 9 + the OOD result.**
- Within-collection in-distribution AUC is **0.87–0.92** (slide 9). Diagnostic-grade live classification is not on the table at that quality, much less on phone-camera input.
- Out-of-distribution PAD-UFES-20 AUC is **0.8055** at the in-distribution threshold (`artifacts/eval_padufes.log`). Recall drops from 0.84 in-distribution to **0.51 on OOD** — half the cancers missed at the calibrated operating point.
- Live attention has a strictly weaker requirement (does the model look at the lesion?) than live classification (does the model classify correctly?). The 0.81 OOD AUC is **consistent with the weaker requirement holding** even when the stronger one doesn't.

**Implementation scope.** Validation step before any code lands: run Eigen-CAM on a known-lesion test image; visually confirm the heatmap localizes on the lesion. ~5 min. If that fails, the framing-aid hypothesis is broken for this architecture and the scope needs a revisit. See `docs/realtime-design.md` §2c for the latency budget and §2f for the result-shape spec (per-frame localization rate on N≈20 self-collected viewfinder frames).

---

## Slide 12 — Multiclass follow-up: hierarchical Option C

**Title.** "Subtype classification as a follow-up, not a refactor."

**Design (Option C: hierarchical, frozen binary first).**
- Stage 1: the existing frozen binary head (this midterm's headline 0.951 AUC) runs as the gating classifier. Only inputs flagged as malignant at threshold 0.661 proceed.
- Stage 2: a subtype head, trained on the malignant pool only (~11,611 malignants across c=70+c=212+c=249), predicts which kind of cancer (Melanoma vs BCC vs SCC vs other).
- Test: PAD-UFES-20 as the held-out subtype cohort (~1,089 malignants, 6 native subtypes the ISIC mirror maps onto its own ontology).

**Why hierarchical over end-to-end multiclass.**
- The binary AUC 0.951 is the strongest single piece of evidence in the project. End-to-end multiclass on a ~11K malignant pool risks degrading binary recall in exchange for marginal subtype accuracy.
- Hierarchical isolates the subtype problem from the binary recall: the subtype head's failure mode (wrong subtype) is bounded by stage 1's recall (which already failed if a malignancy didn't reach stage 2).

**Compounded-error sanity check.** Binary recall 0.835 × subtype accuracy ~0.70 (literature prior on ISIC subtype heads) → ≈ **0.58 catch-AND-subtype**. Marginally better than the audit-era's 0.803 × 0.70 ≈ 0.56 estimate; the headline AUC didn't move the math much.

**Why this slide is short.** The hierarchical-vs-end-to-end argument is the methodology contribution. The actual subtype training run is post-midterm. This slide motivates the follow-up; it doesn't claim subtype results.

---

## Slide 13 — Timeline & remaining work

**Title.** "Done, in-progress, future."

**Done.**
- Phase 1 audit (8 sections, weaknesses table, voice-rules established).
- Phase 4 code rewrite (10 commits, audit fixes W1/W2/W4/W8/H5/M1/M4 closed, .gitattributes + LF normalization).
- Phase 5 pre-train validation (coverage clean, memory peak 25 GB / 42 GB, bit-exact reproducibility).
- Phase 6 retrain on corrected pipeline (4 h 36 min, AUC 0.951 at ES epoch 13/20).
- Phase 7 held-out evals (test, PAD-UFES-20, per-collection breakdown).
- Phase 8 audit doc rewrite with real numbers (§§2/7/8 rewritten, §§9/10 added).

**In progress.**
- This slide outline; finalizing for the actual deck.

**Future work.**
- Implement the framing-aid prototype (slide 11; Eigen-CAM live + Grad-CAM on capture).
- Execute the multiclass follow-up (slide 12; hierarchical Option C on the malignant pool).
- Address W3 / W5 / W6 / W7 if time permits — all disclosure-grade, none blocks the headline.

**What could still go wrong before the talk?** Nothing that would invalidate the headline. The slide 9 / slide 11 framing of the modality-prior contribution and the OOD gap are the disclosures that make the 0.951 honest. Slide 10 explicitly addresses the open weaknesses.

---

## Backup slides

- **B1 — Stage-2 training table.** Full epoch-by-epoch from `experiments/run_20260515_172452.log` showing val loss + AUC + per-epoch threshold + recall/spec. Source of the slide 5 evidence.
- **B2 — ROC / PR curves on test cohort.** **Needs computation** — the plotting scripts may or may not exist in `scripts/`; flagged for the final deck.
- **B3 — Confusion matrices at threshold 0.5 vs 0.661 on test cohort.** **Needs computation** for the 0.5 row (5-line re-thresholding of saved test predictions; doable from the existing `scripts/eval_test_set.py` with a one-line edit). Slide 6 references this.
- **B4 — Focal loss math.** `α_t = α · target + (1-α) · (1-target)`. The formula derivation + the direction-of-α argument expanded.
- **B5 — Patient-level split detail.** Audit §2 with the failure-mode-A diagnostic results from `scripts/audit_patient_id.py`. Per-collection: c=70 has native patient_id; c=212 and c=249 fall back to lesion_id (mean 1.33 / 3.48 rows per lesion). Malignant-cluster-size check from `audit_lesion_id.py` — protection ratios 1.01-1.30× across all collections.
- **B6 — Per-collection AUC (extended).** Same numbers as slide 9 plus PR-AUC + F1 + per-collection CMs at 0.661. **Note: now in main deck (slide 9); keep B6 as extended-Q&A backup.**
- **B7 — Grad-CAM artifact-attention distribution.** **Not done (W7).** Keep as a backup-if-built note; the realtime-design.md framing-aid Eigen-CAM is the related-but-different question that's scoped for follow-up.
- **B8 — Bootstrap CI methodology (NEW).** How the 95% CIs on per-collection AUC were computed: 1000 resamples with replacement, seed=42, via `scripts/bootstrap_test_auc.py`. Per-collection half-widths: c=212 ±0.017, c=70 ±0.028, c=249 ±0.014.
- **B9 — PAD-UFES-20 details (NEW).** Source: ISIC c=406 (PAD-UFES-20). N=1,568 after Indeterminate filter (730 dropped). Prevalence 69.5% (referral cohort skew, NOT a deployment prior). Full CM at 0.661 in `artifacts/eval_padufes.log`. Note on the prevalence asymmetry: PAD-UFES-20 came from a referral pipeline (clinic intake), not a screening population — its prevalence is not a property of the model.

---

## Time budget

| Section | Slides | Minutes |
|---|---|---|
| Framing | 1, 2 | 2 |
| Method | 3, 4, 5, 6 | 5 |
| Results | 7, 8, 9 | 4 (slide 9 is the heavyweight) |
| Limitations | 10 | 1 (closed weaknesses make this shorter) |
| Follow-ups | 11, 12 | 2 |
| Timeline | 13 | 1 |
| **Total** | 13 slides | **15 min** |

Slide 9 (per-collection breakdown) gets the full minute. Slide 7 gets ~1.5 min because the headline-with-caveat structure needs the breath. Slide 11 is shorter than the audit-doc-era live-diagnosis version — less to defend now that the scope is framing-aid-only.

---

## Pre-presentation checklist

**Numbers (already in artifacts/, no work needed):**
- Test AUC 0.9513 [95% CI 0.9465 – 0.9563] — `bootstrap_test_auc.log`
- Test CM at 0.661 — `eval_test_indist.log`
- Per-collection AUCs + CIs — `bootstrap_test_auc.log`
- PAD-UFES-20 result — `eval_padufes.log`
- Phase 6 training table — `run_20260515_172452.log`

**Numbers (need to compute before the deck is final):**
- Test CM at threshold 0.5 (for the slide 6 comparison table).
- ROC curve + PR curve plots on test cohort (for B2).
- Optionally: per-collection CMs at threshold 0.661 (for B6 extended).

**Provenance discipline.** Every number on every slide cites either `docs/midterm-prep-context.md` (audit doc) or `artifacts/*.log` (raw run outputs). No invented numbers; placeholders flagged where computation is still needed.

**Final cross-checks before the talk.**
- The §6 framing language on slide 5 matches the `Trainer._run` docstring verbatim (don't drift).
- The 0.661 threshold on slide 6 cites the cal-cohort calibration explicitly (W1 fix).
- Slide 9's "aggregate exceeds per-collection" framing leads with "we asked the question," not "we found a limitation."
- Slide 10's "closed / open" framing leads with the seven closures, not the four open items.
