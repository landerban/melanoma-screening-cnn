# Midterm presentation prep — working context

For codebase conventions (hyperparams, hard constraints, file map, hard-coded "don't touch" decisions), see `CLAUDE.md` at the repo root.
This file is the presentation-prep state on top of that — what's done, the audit findings (now post-retrain), and the voice conventions established during the Cowork session.

## Professor's brief (verbatim)

> The goal of this project is not necessarily to achieve a new SOTA performance. What matters is clearly defining the problem, setting an appropriate direction to address it, and explaining why your chosen approach is appropriate and reasonable... it is more important to logically explain the purpose of each experiment, what the results indicate, and how those results support your project direction, methodological choices, or subsequent plans... running a baseline model or an existing method that can serve as a comparison point and analyzing what limitations or problems appear from the results can also be considered a meaningful experiment. Even if the experimental results are different from what you expected, they can still be meaningful for the presentation if you can convincingly explain what the results indicate and how they helped you decide the next direction.

Three things follow:

1. The audit was an evidence-gathering pass, not a bug-fixing pass. Each design choice was examined for whether it can be defended in front of the class. Phase 4 then implemented the corrections the audit identified.
2. Negative or unexpected results count. The per-collection AUC breakdown (§9) revealed that the aggregate 0.95 includes a modality-prior contribution — that finding is itself the W8 audit weakness being measured rather than handwaved.
3. The realtime app (§§2/7 framing aid) and multiclass retraining are framed as **follow-up experiments**, each motivated by a specific question the binary model raised — not as feature scope.

## Current status

| Phase | What | Status |
|---|---|---|
| 0 | Orient on code | Done |
| 1 | Evidence audit + honest weaknesses | Done; §§2/7/8 rewritten post-retrain (this pass); §§9/10 added |
| 2 | Realtime app design doc | Done — see `docs/realtime-design.md` (rewritten as framing-aid, not live diagnosis) |
| 3 | Multiclass plan | Pending — unblocked now that Phase 7 is complete |
| 4 | Slide-by-slide outline | Pending — depends on Phase 3 |

## Diagnostics (was: Pending diagnostic runs)

All diagnostic scripts ran. Findings consolidated in `docs/diagnostic_findings.md` and folded into §§2/7/8/9/10 below.

- `scripts/audit_patient_id.py` — failure mode A fired on c=212 and c=249 (no `patient_id` column; 28,273/61,396 rows = 46.05%). Phase 4a's `effective_patient_id` fallback chain (patient_id || lesion_id || isic_id) handles this; the run reports 0 NaN post-fix.
- `scripts/bootstrap_auc_ci.py` — pre-retrain floor estimate on the old checkpoint: AUC 0.6243 [0.5561, 0.6866] on a c=70-only val (the broken split shoved c=212 and c=249 into train; see `docs/diagnostic_findings.md`'s "ghost evaluation" section). Used as the Phase 6 floor estimate; the retrained model cleared it by ~0.33 AUC points.
- `scripts/bootstrap_test_auc.py` — post-retrain test bootstrap: aggregate 0.9513 [0.9465, 0.9563] + per-collection (see §9).
- `scripts/eval_test_set.py` — save/load sanity (Phase 7a). Reproduces test_metrics in the rich checkpoint to bit-exact precision.
- `scripts/eval_per_collection.py` — Phase 7c. Per-collection breakdown; the W8 measurement.
- `scripts/eval_external.py` — Phase 7b. PAD-UFES-20 OOD eval; the distribution-shift gap measurement.

---

## Phase 1 — Evidence audit (post-retrain, final)

Each block: **Choice** / **Rationale** / **Evidence in the repo** / **Honest limitation**. §§2, 7, 8 reflect post-Phase-6 rewrites — those are the canonical versions for the slides.

### §1. Problem framing: binary, recall-first, dermoscopy

**Choice.** Single-output binary classifier (benign vs. malignant), primary metric = recall at usable specificity. Trained on dermoscopy from three ISIC collections rather than clinical or smartphone photographs.

**Rationale.** (a) Binary first because the screening question is binary; subtype is downstream. (b) Recall-first because the clinical cost of a missed malignancy is much larger than the cost of an unnecessary referral, and accuracy at this prevalence (~19% in test) is meaningfully better than at the old ~0.6%. (c) Dermoscopy-only because the c=390 TBP modality was retired during the Phase 2 data pool refresh; the modality-shortcut risk against the dermoscopy collections was unmeasurable without per-collection AUC (now resolved in §9).

**Evidence.** Single-logit head (`efficientnet_b0.py:283`). Threshold sweep on the held-out cal cohort maximizes specificity *subject to* recall ≥ 0.80 (`trainer.py:451`, Phase 4c). Training set assembled in `load_and_merge_metadata` (`trainer.py:129`+).

**Honest limitation.** Dermoscopy is not what an end-user will photograph. The realtime app redesign (Phase 2 follow-up, `docs/realtime-design.md`) explicitly does not claim to close that gap — it measures whether live attention overlay can function as a framing aid, a strictly weaker claim than diagnostic generalization. PAD-UFES-20 eval (§10) measures the actual distribution-shift gap.

### §2. Dataset assembly (POST-RETRAIN REWRITE)

**Choice.** Three ISIC collections, merged at `trainer.py:150`+:

| Collection | Source | Rows | Native `patient_id` | Native `lesion_id` |
|---|---|---|---|---|
| c=212 | HAM10000 dermoscopy | 11,720 | absent | present |
| c=70  | SIIM-ISIC 2020 dermoscopy | 33,126 | present | present |
| c=249 | BCN20000 dermoscopy | 18,946 | absent | present |

Dedup by `isic_id`. Drop `Indeterminate` and NaN diagnoses (2,396 rows). Split at patient level via `effective_patient_id` (Phase 4a). 4-way: train/val/cal/test = 0.65/0.10/0.10/0.15 of patients (Phase 4c).

**Rationale.** Three dermoscopy collections to get enough malignant examples (11,611 in the merged set) and modality-consistent input. Patient-level split via `effective_patient_id` = patient_id || lesion_id || isic_id fallback chain — c=212 and c=249 lack a native `patient_id` column, so the fallback to `lesion_id` is what makes the split honestly patient-level for those rows. The W1 fix (cal cohort) puts threshold calibration on its own patient pool, independent of val.

**Evidence.** `load_and_merge_metadata` (`trainer.py:129`+). `source_collection` tagging at `trainer.py:154`. `effective_patient_id` construction at `trainer.py:166-184`. `patient_level_split` with cal_frac added (`efficientnet_b0.py:188`+). Trainer asserts `effective_patient_id` is non-NaN before split (`trainer.py:283`+). `audit_patient_id.py` confirms 0 NaN post-fix.

**Honest limitations.**

- **Lesion-ID grouping is uneven across collections.** c=212 lesion_id is a weak grouping (mean 1.33 rows/lesion; ~25% of c=212 lesions have ≥1 sibling image). c=249 is genuine grouping (mean 3.48, max 31). c=70 uses native patient_id (mean 16.1 images per patient). The c=212 protection is meaningful but the slide should disclose that "patient-level" for c=212 is in fact "lesion-level with weak grouping." Numbers in `docs/diagnostic_findings.md`.
- **Malignant-cluster-size protection is uniform across classes.** Per `audit_lesion_id.py`, the mean cluster size for malignant groups vs benign is within 1.01-1.30× ratio in every collection. Patient-level protection is not benign-skewed — the leak the slide claims to prevent (same malignant lesion across splits) is protected as well as the analogous benign leak.
- **First-write-wins dedup** at `trainer.py:158` is in place. `audit_merge_consistency.py` checked: 0 cross-collection isic_id duplicates and 0 conflicting `diagnosis_1` between CSVs. No-op for the current composition.
- **Class balance changed.** New training pool is **4.29:1 (49,785 benign / 11,611 malignant)**, not the audit-doc-era 161:1. 76% of malignants come from c=249. The methodology stories in §§4/6 carry over; the imbalance gymnastics framing in §§7/8 (next two sections) had to be rewritten.

### §3. Architecture: EfficientNet-B0 + 2-layer MLP head

**Choice.** EfficientNet-B0 ImageNet-pretrained (5.3M params), GAP, Dropout(0.4), Linear(1280→256), ReLU, Dropout(0.2), Linear(256→1).

**Rationale.** B0 fits in 8GB VRAM at the original 300×300 batch 32. Phase 4g bumped to 384×384 batch 96 for the A100X MIG 3g.40gb (40 GB); B0 is intentionally **kept** at the new compute rather than bumped to B3 — the data composition changed at the same time, and an architecture change would have made the AUC delta uninterpretable ("is it the data or the backbone?"). B3 is a clean follow-up experiment.

**Evidence.** Backbone at `efficientnet_b0.py:285`+, head at `efficientnet_b0.py:299-308`. CFG bump rationale documented in the CFG comment block (`efficientnet_b0.py:38`+).

**Honest limitation.** Hardware-context-driven choice. There is no ablation against ResNet-50 or EfficientNet-B3; the choice is asserted from VRAM + reasonable defaults + the "one variable at a time" discipline from Phase 4.

### §4. Loss function: focal with α=0.85, γ=2.5

**Choice.** Focal loss with α=0.85, γ=2.5. Weighted BCE is implemented as an alternative (`build_loss`, `efficientnet_b0.py:368`+) but capped at pos_weight=80 to avoid gradient blowup.

**Rationale.** In binary focal loss as implemented at `efficientnet_b0.py:343`, α directly weights the positive class: `alpha_t = alpha * targets + (1 - alpha) * (1 - targets)`. The default α=0.25 commonly seen in detection code means *positives get weight 0.25 and negatives get 0.75* — the wrong direction whenever positives are the rare class. α=0.85 gives malignant 5.7× the weight of benign, matching the intent. The methodology story is **"copying defaults across imbalance regimes is a category error"** — that story is prevalence-invariant and survives the new 4.29:1 dataset unchanged.

**Evidence.** `FocalLoss.forward` at `efficientnet_b0.py:338-345`. CFG comment block at `efficientnet_b0.py:53-58` flags the inversion. Empirical: Phase 6 stage-1 epoch 1 reached val AUC 0.932 → 0.943 at epoch 5; stage-2 best val AUC 0.9514 at epoch 13.

**Honest limitation.** α=0.85 vs. weighted BCE was not directly ablated (W3). At the new 4.29:1 imbalance, α=0.85 is still upweighting the minority class, just less aggressively *relative to* the imbalance than it was at 161:1. The specific α value isn't load-bearing on the slide; the methodology argument is.

### §5. Two-stage transfer (unchanged)

**Choice.** Stage 1: backbone frozen, head only, 5 epochs at lr=3e-4 with `ReduceLROnPlateau`. Stage 2: full unfreeze, backbone lr=5e-6, head lr=1e-4, up to 30 epochs with `CosineAnnealingLR(T_max=30)`.

**Rationale.** Same as before — stage 1 prevents head-gradient destruction of pretrained features; stage 2 uses differential LRs because the backbone needs delicate nudges while the head still learns. Cosine in stage 2 instead of plateau-based because plateau-on-loss is unreliable when val loss is dominated by the benign majority.

**Evidence.** Stage 1 at `trainer.py:316`+. Stage 2 at `trainer.py:347`+. Cosine at `trainer.py:363`+. Empirical: stage 1 ends at AUC 0.943 (epoch 5), stage 2 best epoch (13) hits 0.9514.

**Honest limitation.** Cosine annealing with `T_max=30` is mostly inert — ES fired at epoch 20 in Phase 6, so the cosine completed about two-thirds of one descent. Defensible but undertested (W5). At 4.29:1 imbalance, stage 1 also converges meaningfully faster than at 161:1 — the head doesn't have as much work separating the easier minority class.

### §6. Early stopping on AUC, not val loss (unchanged methodology, framing locked in `Trainer._run` docstring)

**Choice.** In stage 2, ES monitors validation AUC and saves the checkpoint that maximizes it (`trainer.py:380`+). Patience 7, delta 5e-4.

**Rationale (the correct §6 vs §8 framing, also embedded in `Trainer._run` docstring).** Val loss at heavy class imbalance is dominated by the benign majority. A model can reduce val loss by tightening benign confidence variance, which is orthogonal to malignant-class separability. AUC measures separability regardless of confidence calibration, so it's the right ES signal. The "recall@0.5 drops over epochs" curve that prior versions of this audit used as evidence was a confidence-calibration artifact, not separation degradation. Phase 4f's dynamic per-epoch threshold replaces threshold=0.5 with a recall≥0.80 mini-sweep, fixing the misleading plot.

**Evidence.** ES logic at `trainer.py:438`+. The comparison `if cur_auc > best_auc + cfg["es_delta"]` is the key line. Comment block above it lays out the §6 framing.

**Honest limitation.** None for the choice itself. The dead `EarlyStopping` class in `efficientnet_b0.py:390` (loss-based) is still present but unused (legacy `train_pipeline`); deleting it is a future cleanup.

### §7. Threshold calibration (POST-RETRAIN REWRITE)

**Choice.** Sweep ROC on the held-out CAL cohort (5,929 images, 1,555 patients, never seen during training or ES). Find threshold maximizing specificity subject to recall ≥ 0.80, fall back to Youden's J. Result: **0.661**.

**Rationale.** Audit weakness W1 (closed by Phase 4c): previously, threshold was selected on the same val cohort that drove ES + best-epoch selection, biasing the calibrated number toward overfitting. The new 4-way split puts cal on its own patient pool; the threshold is genuinely independent of the optimization loop.

The threshold shifted from the old 0.347 to 0.661 because the new model's score distribution is shifted up — at 4.29:1 prevalence and a stronger model, malignant probabilities cluster higher. Both thresholds are the "recall ≥ 0.80, max spec" operating point; they differ because the underlying score distribution differs, not because the methodology changed.

**Evidence.** `_sweep_threshold` at `trainer.py:711`+. `_run` passes `cal_loader` explicitly (`trainer.py:507`). On cal: recall 0.801, spec 0.902. On held-out test at threshold 0.661: recall 0.835, spec 0.914.

**Honest limitation.** The threshold 0.661 is calibrated for the aggregate cal cohort prevalence (~20%). It is **miscalibrated per collection**: at this threshold, recall on c=70 is 0.23 (huge miss rate on a 1.7%-prevalence sub-cohort) and recall on c=249 is 0.89 (above target). The operating point only holds for the cocktail-mix prevalence. For deployment, the threshold must be recalibrated per institution's expected prevalence — and this is the kind of thing a screening-tool clinician should be told.

### §8. Evaluation metrics (POST-RETRAIN REWRITE)

**Choice.** Headline = held-out test ROC-AUC + recall at the calibrated threshold from cal. PR-AUC, F1, specificity, per-collection breakdown all reported.

**Rationale.** At 4.29:1 prevalence, F1 lands ~0.76 at the operating threshold — a defensible secondary metric. AUC remains the primary headline because it's prevalence-invariant. The previous audit-era apologetic ("F1 ~0.03 at this prevalence is expected") and ("PR-AUC ~0.03-0.05 is the imbalance ceiling") are both **retired** — the new dataset is no longer catastrophically imbalanced and the metrics are no longer pathologically small.

**Evidence (headline numbers, slide 7 candidates).**

| Metric | Value | Source |
|---|---|---|
| Test ROC-AUC | **0.9513** [95% CI 0.9465 – 0.9563] | `scripts/bootstrap_test_auc.py`, 1000 resamples |
| Test PR-AUC | 0.8353 | same |
| Recall @ 0.661 | 0.835 (1475/1767) | Phase 7a, ckpt round-trip verified bit-exact |
| Specificity @ 0.661 | 0.914 (6784/7419) | same |
| F1 @ 0.661 | 0.761 | same |
| Test CM @ 0.661 | TN=6784 FP=635 FN=292 TP=1475 | same |

**Honest limitation.** The aggregate AUC is **prevalence-weighted by cohort composition**. See §9 — within-collection AUCs cluster at 0.87-0.92. The 0.95 figure includes both within-class separation AND cross-collection prevalence priors (the model has partly learned "this looks like a BCN20000-style image → bump malignant probability"). Honest within-collection discriminative quality is **0.87-0.92**. The aggregate goes on the slide *with* the per-collection breakdown, not as a clean standalone headline.

### §9. Per-collection AUC breakdown — W8 audit resolution (NEW)

Phase 7c via `scripts/eval_per_collection.py` + bootstrap CIs from `scripts/bootstrap_test_auc.py`:

| Cohort | N (test) | Prev | AUC | 95% CI | PR-AUC | Recall @0.661 | Spec @0.661 |
|---|---|---|---|---|---|---|---|
| **Aggregate** | 9,186 | 19.2% | **0.9513** | [0.9465, 0.9563] | 0.8353 | 0.835 | 0.914 |
| c=212 HAM10000 | 1,708 | 17.9% | 0.9151 | [0.8980, 0.9323] | 0.7494 | 0.741 | 0.896 |
| c=249 BCN20000 | 2,534 | 54.3% | 0.8702 | [0.8555, 0.8842] | 0.8849 | 0.893 | 0.656 |
| c=70 SIIM-ISIC 2020 | 4,944 | 1.7% | 0.8835 | [0.8545, 0.9100] | 0.1223 | 0.233 | 0.981 |

**The aggregate AUC exceeds every per-collection AUC.** That is not Simpson's paradox in the leak sense — it's coherent dataset-composition signal. Predictions on c=249 patients cluster higher than predictions on c=70 patients (because c=249 is 54% malignant in training, c=70 is 1.7%), so the *between-collection* score ordering is itself discriminative. The aggregate AUC captures both within-collection separation AND between-collection prevalence-prior separation.

**Statistical reading of the per-collection CIs.**

- c=212 [0.898, 0.932] does **not** overlap c=249 [0.856, 0.884]. c=212 is genuinely a better cohort for this model than c=249, with statistical significance. Likely explanation: c=212 (HAM10000) has cleaner labels and lower lesion ambiguity than c=249 (BCN20000), which contains more difficult cases including subtypes the model wasn't explicitly tuned for.
- c=70 CI [0.855, 0.910] **overlaps** both c=212 and c=249 CIs. c=70 and c=249 are statistically indistinguishable in within-collection discriminative quality.

**Slide framing.** Lead with the per-collection range (0.87-0.92) as the honest discriminative-quality claim. Quote the aggregate (0.95) as the prevalence-weighted headline with the per-collection breakdown immediately below it. The W8 finding in one sentence: **the model exploits modality-prior shortcuts in addition to lesion-discriminative features, and the magnitude of that shortcut contribution is now measured**.

### §10. PAD-UFES-20 OOD eval — distribution-shift gap (NEW)

The Phase 2 follow-up's motivation slide rests on this measurement. Phase 7b via `scripts/eval_external.py`.

| | |
|---|---|
| Source | ISIC archive c=406 (PAD-UFES-20, smartphone clinical close-ups) |
| N (post-Indeterminate filter) | 1,568 (1,089 malignant / 479 benign; 730 Indeterminate dropped) |
| Prevalence | 69.5% — **referral cohort skew, NOT a deployment-population prior** |
| **ROC-AUC** | **0.8055** |
| PR-AUC | 0.8953 (1.29× random baseline at 69.5%) |
| Recall @ 0.661 | 0.514 (560 of 1,089 cancers caught) |
| Specificity @ 0.661 | 0.887 |
| F1 @ 0.661 | 0.658 |

**Interpretation.** The dermoscopy-trained model partly transfers to clinical close-ups. AUC 0.81 is meaningfully above chance (random = 0.5) — some discriminative features (lesion shape, color heterogeneity, border irregularity) generalize across imaging modality. But the operating threshold 0.661 is severely miscalibrated for this distribution: **recall drops from 0.84 in-distribution to 0.51 OOD**. Half the cancers are missed at the in-distribution operating point.

**The 0.95 → 0.81 gap** is consistent with the §9 prevalence-prior reading. The model relies partly on collection-specific features that disappear on phone-camera input; what remains is the cross-modality lesion signal (~0.81 AUC).

**For the realtime app slide.** This number bounds what live-classification could achieve and grounds the framing-aid redesign in `docs/realtime-design.md`. Live attention overlay (Eigen-CAM, gradient-free) only needs the model to be looking at the lesion — a strictly weaker requirement than 0.95 diagnostic AUC. The 0.81 result is consistent with that weaker condition holding while the stronger one (clean classification at the calibrated threshold) doesn't.

---

## Phase 1 — Honest weaknesses (post-Phase-4 status)

| ID | What | Status after Phase 4–7 |
|---|---|---|
| W1 | Threshold sweep on the same val set used for ES | **CLOSED** (Phase 4c). 4-way split; cal cohort owns the sweep. |
| W2 | Test split discarded by `trainer.py` | **CLOSED** (Phase 4b). Test eval at calibrated threshold; `test_metrics` persisted to checkpoint + sidecar JSON. |
| W3 | Double balancing (sampler + focal) not ablated | Open. Slide-time disclosure: "we used both; we didn't ablate which contributed more." |
| W4 | No global seeds; results not reproducible | **CLOSED** (Phase 4d). Phase 5 A3 verified bit-exact reproducibility across runs. |
| W5 | CosineAnnealingLR `T_max=30` while ES fires around ep 10 | Open. Phase 6 ES fired at ep 20, so cosine completed ~67% of one descent. Less inert than at the old 161:1 imbalance. Slide-time disclosure. |
| W6 | Augmentations not ablated | Open. Slide-time disclosure: hue/saturation deliberately excluded because color is diagnostic for melanoma; jitter on brightness/contrast only. |
| W7 | Grad-CAM is a UI feature, not a systematic check | Partially addressed via `docs/realtime-design.md` framing-aid redesign, which scopes Eigen-CAM localization measurement as a post-midterm experiment. |
| W8 | Per-collection performance breakdown not computed | **CLOSED** (Phase 7c + §9). Within-collection AUCs 0.87-0.92; aggregate 0.95 includes prevalence-prior contribution. Modality-shortcut concern is now measured rather than handwaved. |
| H5 | Checkpoint format opaque | **CLOSED** (Phase 4e). Rich ckpt dict with cfg, val/test metrics, git_hash, timestamp; `app.py` loads both new and legacy formats. |
| M1 | Per-epoch metrics at threshold 0.5 | **CLOSED** (Phase 4f). `_mini_threshold_sweep` per epoch; the per-epoch plot no longer misrepresents recall as "declining over epochs." |
| M4 | `app.py` reloads model on every inference | **CLOSED** (Phase 4i). Mtime-aware cache; cache hit ~0 ms, miss ~140 ms. |

**Closed**: W1, W2, W4, W8, H5, M1, M4 (7 of 11). **Open as slide-time disclosures**: W3, W5, W6, W7. None block presenting the headline numbers.

---

## Voice and tone

Match `CLAUDE.md` / `README.md`:

- Direct, casual-technical, no hedging.
- Cite line numbers when referencing code.
- No bullet fountains. Prose where prose works, tables where they work, code blocks where they work.
- Honest about limitations is the rubric — "we measured this, we didn't measure that, here's why" is the form. Hand-waving isn't.
- Don't invent numbers. Every number on the slide has a `scripts/*.py` + `artifacts/*.log` provenance.
- Don't propose fixes that violate hard constraints in `CLAUDE.md` (e.g., focal α=0.25 for binary, image-level random split) without explicitly arguing why the constraint should be lifted.
