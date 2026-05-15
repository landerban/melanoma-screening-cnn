# Midterm presentation prep — working context

For codebase conventions (hyperparams, hard constraints, file map, hard-coded "don't touch" decisions), see `CLAUDE.md` at the repo root.
This file is the presentation-prep state on top of that — what's done, what's pending, the audit findings, and the voice conventions established during the Cowork session.

## Professor's brief (verbatim)

> The goal of this project is not necessarily to achieve a new SOTA performance. What matters is clearly defining the problem, setting an appropriate direction to address it, and explaining why your chosen approach is appropriate and reasonable... it is more important to logically explain the purpose of each experiment, what the results indicate, and how those results support your project direction, methodological choices, or subsequent plans... running a baseline model or an existing method that can serve as a comparison point and analyzing what limitations or problems appear from the results can also be considered a meaningful experiment. Even if the experimental results are different from what you expected, they can still be meaningful for the presentation if you can convincingly explain what the results indicate and how they helped you decide the next direction.

Three things follow:

1. The audit is an evidence-gathering pass, not a bug-fixing pass. Each design choice is examined for whether it can be defended in front of the class, what's an honest disclosed weakness, and what motivates a next step.
2. Negative or unexpected results count. PR-AUC ≈ 0.03 at this prevalence isn't an embarrassment — it's a finding about the imbalance ceiling. The α=0.25 → α=0.85 correction is itself a presentable story.
3. The realtime app and multiclass retraining are framed as **follow-up experiments**, each motivated by a specific question the binary model raised — not as feature scope.

## Current status

> **Post-Phase-3 note (DO NOT REMOVE until Phase 8 runs).** §§2, 7, and 8
> of the Phase 1 audit are anchored to the c=390+70+212 / 412K-image /
> 161:1 dataset. After the Phase 6 retrain on c=70+212+249 the cohort
> shape, prevalence, and threshold-calibration math will all shift. These
> sections need narrative rewrites in Phase 8, not just number swaps. The
> methodology stories in §4 (focal α=0.85), §6 (ES on AUC), and §7
> (threshold encodes which tool) are prevalence-invariant and survive
> as-is.

| Phase | What | Status |
|---|---|---|
| 0 | Orient on code | Done |
| 1 | Evidence audit + honest weaknesses | Done; §4, §7, §8 corrected after review (see below) |
| 2 | Realtime app design doc | Done — see `docs/realtime-design.md` |
| 3 | Multiclass plan | Pending — blocked on diagnostic outputs (below) |
| 4 | Slide-by-slide outline | Pending — depends on Phase 3 |

## Pending diagnostic runs (final inputs to the audit)

Two scripts live under `scripts/`. Both need to run on the local machine where the ~100GB dataset lives — neither has been run yet.

- `scripts/audit_patient_id.py` — answers whether the patient-level-split guarantee actually holds. Watch the line *"N rows would be bucketed into ONE synthetic NaN-patient"* in the POST-MERGE REPORT. If N=0, the patient-level split is honest. If N is large, the patient-level claim has a hole and the corrected-split test-set AUC becomes the honest headline.
- `scripts/bootstrap_auc_ci.py` — bootstrap 95% CI on the headline AUC plus recall/specificity at threshold 0.347 on the actual val cohort. Two sanity checks: (a) CI half-width should be ~±0.04 (Hanley-McNeil); much wider implies an effective-N reduction, likely from the H2 patient leak. (b) Recall/specificity at 0.347 should reproduce the CLAUDE.md numbers (~0.803 / ~0.876); if not, the checkpoint loaded isn't the one CLAUDE.md describes.

After both have run, fold the numbers into §2 (patient-level split) and §8 (headline AUC) below, then unblock Phase 3.

---

## Phase 1 — Evidence audit (final, post-corrections)

Each block: **Choice** / **Rationale** / **Evidence in the repo** / **Honest limitation**. §4, §7, §8 reflect post-review corrections — those are the canonical versions for the slides.

### §1. Problem framing: binary, recall-first, dermoscopy + TBP

**Choice.** Single-output binary classifier (benign vs. malignant), primary metric = recall at usable specificity. Trained on dermoscopy + TBP close-ups rather than clinical or smartphone photographs.

**Rationale.** (a) Binary first because the screening question is binary; subtype is downstream (Phase 3). (b) Recall-first because the clinical cost of a missed malignancy is much larger than the cost of an unnecessary referral, and accuracy at this prevalence is trivially maximized by predicting all-benign. (c) Dermoscopy/TBP because that's what ISIC provides at scale.

**Evidence.** Single-logit head (`efficientnet_b0.py:283`). Threshold sweep maximizes specificity *subject to* recall ≥ 0.80 (`trainer.py:469`). Training set assembled in `load_and_merge_metadata` (`trainer.py:129`).

**Honest limitation.** "Dermoscopy + TBP" is not what an end-user will photograph. The model was never asked to generalize to phone-camera close-ups — that's what Phase 2 is designed to test, not what it has tested.

### §2. Dataset assembly

**Choice.** Merge ISIC collections 390 (2024 TBP, ~401K), 70 (SIIM-ISIC 2020 dermoscopy), 212 (HAM10000 dermoscopy). Dedup by `isic_id`. Drop `Indeterminate`. Split at patient level.

**Rationale.** Multiple collections to get enough malignant examples. Dedup because the same `isic_id` can appear in two collections. Patient-level split because two photos of the same lesion in train and val is effectively a memorization test — ISIC literature notes 10–15pp AUC inflation from image-level random splits.

**Evidence.** Three collections merged at `trainer.py:150`. Dedup at `trainer.py:158`. Indeterminate dropped via `isin(["Benign", "Malignant"])` at `trainer.py:161`. Patient-level split at `efficientnet_b0.py:188`.

**Honest limitation.** Three problems live here:
- The three collections are not the same modality. TBP (390) has different image statistics and a much lower malignant prior than dermoscopy (70, 212). A small CNN can learn "looks like a TBP tile → predict benign" and score well on overall AUC without learning anything about lesions. Per-collection AUC is never computed (see W8).
- `patient_level_split` doesn't guard against NaN `patient_id` — all NaN rows become one synthetic patient (`efficientnet_b0.py:218`). Whether this fires in practice depends on per-CSV column presence; `scripts/audit_patient_id.py` answers it.
- First-write-wins dedup (`trainer.py:158`) silently picks a label if two CSVs disagree on `diagnosis_1`.

### §3. Architecture: EfficientNet-B0 + 2-layer MLP head

**Choice.** EfficientNet-B0 ImageNet-pretrained (5.3M params), GAP, Dropout(0.4), Linear(1280→256), ReLU, Dropout(0.2), Linear(256→1).

**Rationale.** B0 fits in 8GB VRAM at 300×300 batch 32 with room for stage-2 unfreezing; B3/B4 would not. ResNet-50 is comparable size but has no obvious dermoscopy advantage over EfficientNet. ViT-S would have needed substantially more data than 412K to outperform a pretrained CNN. Two-layer head with heavy dropout because it's trained from scratch on a heavily imbalanced binary task.

**Evidence.** Backbone at `efficientnet_b0.py:270`, head at `efficientnet_b0.py:278-284`. Head init Kaiming-normal (`efficientnet_b0.py:292`). Param counts verifiable via `efficientnet_b0.py:652` (`__main__`).

**Honest limitation.** Hardware-constrained choice. The 3060 Ti's 8GB VRAM is the binding constraint — say so plainly. There is no ablation against ResNet-50 or EfficientNet-B3; the choice is asserted from VRAM + reasonable defaults.

### §4. Loss function: focal with α=0.85, γ=2.5 — *post-correction*

**Choice.** Focal loss with α=0.85, γ=2.5. Weighted BCE is implemented as an alternative (`build_loss`, `efficientnet_b0.py:368`) but capped at pos_weight=80 to avoid gradient blowup at 161:1.

**Rationale.** In binary focal loss as implemented at `efficientnet_b0.py:343`, α directly weights the positive class: `alpha_t = alpha * targets + (1 - alpha) * (1 - targets)`. The default α=0.25 commonly seen in detection code means *positives get weight 0.25 and negatives get 0.75* — the wrong direction whenever positives are the rare class you actually care about catching. For skin screening, positives (malignant) are rare (0.6%) *and* are the class whose recall is the objective, so α has to be high. α=0.85 gives malignant ~5.7× the weight of benign, matching the intent.

The RetinaNet paper uses α=0.25 because γ=2 is already doing aggressive minority-focusing on its own; the paper notes explicitly that α and γ trade off (higher γ → lower α). The actual mistake is copying α=0.25 from detection code into a heavily-imbalanced binary classifier without re-tuning γ. The earlier "RetinaNet has the opposite imbalance" framing was wrong — the directions are the same; the calibration isn't.

**Evidence.** `FocalLoss.forward` (`efficientnet_b0.py:338-345`). CFG comment block at `efficientnet_b0.py:53-58` flags the inversion. Empirical evidence: stage-1 val AUC reaches 0.882 by epoch 5 (CLAUDE.md training table) — wouldn't happen if loss were upweighting benign.

**Honest limitation.** α=0.85 vs. weighted BCE was not directly ablated. γ=2.5 vs. γ=2.0 also unvalidated.

### §5. Two-stage transfer

**Choice.** Stage 1: backbone frozen, head only, 5 epochs at lr=3e-4 with `ReduceLROnPlateau`. Stage 2: full unfreeze, backbone lr=5e-6, head lr=1e-4, up to 30 epochs with `CosineAnnealingLR(T_max=30)`.

**Rationale.** Stage 1 prevents the randomly-initialized head from emitting wild gradients that destroy pretrained backbone features. Stage 2 uses differential LRs because the backbone has good features that need delicate nudges while the head still needs substantial updates. Cosine annealing in stage 2 instead of plateau-based scheduling because plateau-on-loss is unreliable when val loss is dominated by the benign majority (see §6); cosine is deterministic and avoids the LR-collapse failure mode.

**Evidence.** Stage 1 at `trainer.py:267-297`. Stage 2 at `trainer.py:299-364`. Cosine at `trainer.py:315-317`. Empirical: stage 1 ends at AUC 0.882, stage 2 best epoch (3) hits 0.928.

**Honest limitation.** Cosine annealing with `T_max=30` is mostly inert — ES typically fires around epoch 8–10, so the cosine completes about a third of one descent. Defensible but undertested (W5).

### §6. Early stopping on AUC, not val loss

**Choice.** In stage 2, ES monitors validation AUC and saves the checkpoint that maximizes it (`trainer.py:336-340`). Patience 7, delta 5e-4. The legacy `EarlyStopping` class in `efficientnet_b0.py:390` (loss-based) is *not* used in the UI path.

**Rationale.** Under 161:1 imbalance, val loss is dominated by ~50K benign predictions per epoch and ~70 malignant predictions. A model can lower its loss by becoming marginally more confident on the easy benigns while *worsening* on malignants. CLAUDE.md training table shows this: val loss falls from 0.0042 (epoch 3) to 0.0034 (epoch 10), but recall drops from 0.662 to 0.324 over the same span. Selecting on val loss would have given epoch 10; selecting on AUC gives epoch 3.

**Evidence.** ES logic at `trainer.py:336-364`. The comparison `if cur_auc > best_auc + cfg["es_delta"]` is the key line.

**Honest limitation.** None for the choice itself — this is the right call. Dead `EarlyStopping` in `efficientnet_b0.py` should be deleted or clearly marked legacy.

### §7. Threshold calibration — *post-correction*

**Choice.** Sweep the validation set ROC, find the threshold that maximizes specificity subject to recall ≥ 0.80, fall back to Youden's J. Settled on 0.347.

**Rationale.** Clinical-cost asymmetry. The val set has 53,675 images: 71 malignant, 53,604 benign. Comparing the two operating points on the actual val cohort (raw counts — avoid per-100K scaling, which mixes raw and ratio quantities and is what made the earlier draft inconsistent):

| Threshold | Caught | Missed | False alarms | Correctly cleared |
|---|---|---|---|---|
| 0.5 | 47 / 71 | 24 / 71 | 3,049 / 53,604 | 50,555 / 53,604 |
| 0.347 | 57 / 71 | 14 / 71 | ~6,647 / 53,604 | ~46,957 / 53,604 |

(Threshold-0.347 row computed from reported recall 0.803 and specificity 0.876: TP = ⌊71 × 0.803⌋ = 57, FN = 14; TN = ⌊53,604 × 0.876⌋ ≈ 46,957, FP ≈ 6,647.)

Net trade: **10 more cancers caught at the cost of ~3,600 additional unnecessary referrals.** Clearly worth it for screening (downstream action = dermatology visit). Not worth it for triage (downstream action = biopsy). **The threshold encodes which tool this is.**

**Evidence.** Sweep logic at `trainer.py:445-489`. UI exposes the calibrated threshold via the "Use optimal threshold from training" button (`app.py:354`).

**Honest limitation.** Threshold calibrated on the same val set that drove ES and best-epoch selection. On a properly held-out cohort, expect AUC to drop by roughly 0.005–0.02 from selection bias alone, and recall-at-the-fixed-threshold to drop by a few percentage points. Actual gap is unknown until W2 (held-out test eval) runs.

### §8. Evaluation metrics — *post-correction*

**Choice.** Headline = ROC-AUC + recall at the calibrated threshold. PR-AUC, F1, specificity, full confusion matrix reported but secondary.

**Rationale.** At 161:1 imbalance, accuracy is meaningless and F1 is dominated by precision, which is itself dominated by prevalence. AUC is prevalence-invariant. The operationally meaningful number for a screening tool is **recall at chosen specificity** — it directly answers "how many cancers caught, at what false-alarm cost." PR-AUC is reported but is a hard metric at this prevalence; the value of 0.03–0.05 is in the expected range for screening-style models on imbalanced ISIC subsets. Comparing PR-AUC directly to balanced-ISIC literature (0.15–0.25) without normalizing for prevalence is the standard mistake. Disclose this; do not lead with a "lift over random" framing — that framing invites the rebuttal *"5× a tiny number is still a tiny number"*, which is fair.

**Evidence.** AUC = ES metric (`trainer.py:336`). PR-AUC via `average_precision_score` (`trainer.py:428`).

**Honest limitation.** Per-epoch F1, recall, and specificity in the training plot are computed at threshold 0.5 (`trainer.py:424`), not the operating threshold. The recall curve in the plot shows the model becoming more confident over epochs, not getting worse at malignant detection. Anyone reading the plot without the calibration step gets a wrong impression.

---

## Phase 1 — Honest weaknesses (with minimum experiments)

| ID | What | Minimum experiment | Effort |
|---|---|---|---|
| W1 | Threshold sweep on the same val set used for ES | 4-way split (train/val/cal/test); cal owns the sweep | ~3h code + 1 training run |
| W2 | README claims a 54,729-image held-out test set; `trainer.py:235` discards it. Headline is val AUC of a val-selected checkpoint. | Return the test split, load `best_model.pth`, run `_val_ep`. If test AUC drops, that's the honest headline. | ~2h |
| W3 | Double balancing (`WeightedRandomSampler` + focal α=0.85) is asserted, not compared | Three 5-epoch stage-1 runs varying {sampler only, focal only, both}, report val AUC | ~4h |
| W4 | No global seeds; results not reproducible run-to-run | Fix seed in `Trainer._run`, train twice, report AUC delta | ~30m code + 2 runs |
| W5 | CosineAnnealingLR `T_max=30` while ES fires around ep 10 | One head-to-head: `T_max=10` vs current. If AUC within noise, collapse to "slowly-decaying LR" in the slide | ~3h |
| W6 | Augmentations not ablated (color jitter without hue/saturation defensibly conservative but undocumented) | Single ablation removing color jitter, or just document the reasoning ("hue/saturation deliberately excluded — color is diagnostic") | ~3h or 0h (just document) |
| W7 | Grad-CAM is a UI feature, not a systematic check | Run Grad-CAM on N=32 val images; compute fraction of CAM mass in outermost 15% of pixels; report distribution | ~4h — produces a slide-ready graphic |
| W8 | Per-collection performance breakdown not computed (modality-shortcut risk between TBP and dermoscopy) | Tag each image with source collection; compute per-collection AUC in `_val_ep` | ~2h |

**Prioritization for slide impact:** W2 → W7 → W8. Test eval changes the headline number; Grad-CAM check produces a slide-ready graphic and turns an unmeasured failure mode into a measured finding; per-collection AUC could *invalidate* the headline number. ~8 hours total. Everything else is nice-to-have.

---

## Voice and tone

Match `CLAUDE.md` / `README.md`:

- Direct, casual-technical, no hedging.
- Cite line numbers when referencing code.
- No bullet fountains. Prose where prose works, tables where they work, code blocks where they work.
- Honest about limitations is the rubric — "we didn't test this" is a fine sentence; hand-waving is not.
- Don't invent numbers. Estimates are fine if marked as estimates.
- Don't propose fixes that violate hard constraints in `CLAUDE.md` (e.g., α=0.25 for focal loss) without explicitly arguing why the constraint should be lifted.
