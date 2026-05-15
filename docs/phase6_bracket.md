# Phase 6 — Expected-result bracket (pre-run)

Written BEFORE Phase 6 launches, so the reaction to the actual number is
methodological rather than rationalization. These are priors against the
known structural changes (4.29:1 class balance vs old 161:1, dermoscopy-
only modality, genuine patient-level split via `effective_patient_id`,
4-way split with calibration on cal cohort, test set = 9,186 images /
1,767 malignant).

The retrained model's test AUC will land in one of these bands. The
slide-writing decision is locked to whichever band fires.

## Test AUC bands

| Band | Range | Interpretation | Action |
|---|---|---|---|
| **Alarm** | > 0.95 | The old 0.928 was likely inflated by easy benigns in the NaN-bucket-absorbed val (c=212 + c=390 dilution). Cleanly beating that on a more honest split with consistent dermoscopy modality would be surprising. Suspect residual leakage. | Investigate: re-run `audit_patient_id.py` against the new merged frame; check per-collection AUC in 7c; look for test_df rows whose `effective_patient_id` overlaps train. |
| **Expected (high)** | 0.88 – 0.95 | The W2/W8 fixes traded apparent AUC for honesty, but the data composition is friendlier (more malignant examples from BCN20000, consistent modality). Strong, presentable headline. | Headline number for slide 7. No caveats beyond the standard "single seed, single test set" disclaimers. |
| **Expected (low)** | 0.80 – 0.88 | The audit fixes cost some apparent AUC because the old number was partly artifactual. The "we measured this more cleanly and got a more conservative number" story is exactly the rubric the professor named — direction-of-result reasoning. | Headline number for slide 7 with the §§2/7/8 Phase 8 narrative-rewrite framing: "the new number is more honest than the old one." |
| **Investigate** | 0.65 – 0.80 | Smaller than expected but not impossible. BCN20000 contains heterogeneous melanocytic/non-melanocytic malignancies the old training pool didn't have — could be genuinely harder. Edges into investigate territory below ~0.75. | Run 7c (per-collection AUC) BEFORE writing slides; if c=249 AUC is meaningfully lower than c=70+c=212, the BCN20000 difficulty is the story. If all three are similarly lowered, something else broke. |
| **Broken** | < 0.65 | Below the Phase 3 ghost-eval floor of **0.6243** (old ckpt on c=70-only val). If the new training fails to beat what the old model achieves on a c=70 subset, something structural is wrong. | Stop before slide-writing. Re-run A1/A3 against the actual training run; check `experiments/run_*.log` for silent NaN losses or label flips; inspect a sample of training-set images for I/O corruption. |

## PR-AUC bands

Random baseline at 4.29:1 prevalence ≈ 0.189. Bands:

- **0.70 – 0.85** — expected band for a well-calibrated model on this prevalence.
- **0.55 – 0.70** — fine, presentable; reflects the difficulty of BCN20000.
- **0.45 – 0.55** — weak; check recall-precision tradeoff and threshold sweep convergence.
- **> 0.85** — unusual; cross-check against AUC band.

## Calibrated recall + specificity at the operating threshold

The sweep is "max specificity subject to recall ≥ 0.80." Bands:

- **Recall ~ 0.80, spec ~ 0.65 – 0.85** — sweep met the constraint cleanly. Expected.
- **Recall ~ 0.80, spec < 0.50** — sweep met recall but at a low threshold; the model isn't separating cleanly. Cross-check AUC.
- **Recall < 0.78** — sweep fell back to Youden's J because no threshold satisfied recall ≥ 0.80. This is itself a finding; the model can't hit the screening operating point on this data. Worth a slide bullet.

## Stage-1 milestone (epoch 5 check the user asked about)

Stage-1 head-only training at 4.29:1 imbalance with focal α=0.85 is a much
easier problem than at 161:1. Expected stage-1 final AUC: **0.85 – 0.92**.

- **Below 0.80 at epoch 5** — something broken in stage-1 setup. Likely
  candidates: focal loss misconfigured after my edits, transform pipeline
  applying wrong augmentations, train_loader corrupted by my seeding
  changes. Stop before stage 2 starts.
- **0.80 – 0.85** — low end of expected. Continue but watch stage 2.
- **0.85 – 0.92** — on track.
- **Above 0.92 at epoch 5** — converged fast. Stage 2 may have less to do
  than usual; ES could fire early. Not a problem.

## What I am NOT pre-bracketing

- **Run-to-run variance.** Phase 5 A3 showed bit-exact reproducibility, so
  a single-seed run is the same as the seed=42 result every time. The
  question of "would seed=43 give a meaningfully different number" is
  empirical, not a prior. Defer to a second-seed run if there's time.
- **PAD-UFES-20 generalization** (Phase 7b). That's a separate
  measurement against a different distribution; bracketing it here would
  conflate the in-distribution and OOD questions the audit explicitly
  separates.
- **Per-collection AUC breakdown** (Phase 7c). The bracket above assumes
  per-collection AUCs may diverge; that's the W8 measurement, not
  something to anchor on a prior.

## Reference: the user's priors offered in conversation

For triangulation against the priors above (NOT as anchors):

- "above 0.93 starts to feel 'is something still leaking' "
- "0.85 – 0.93 is the expected band"
- "0.78 – 0.85 is the 'fine but smaller, the W2/W8 fixes cost us some apparent AUC' range"
- "below 0.78 is the investigate-first zone"
- "PR-AUC at this prevalence should run somewhere in 0.50 – 0.75"

My bracket runs slightly higher on the alarm threshold (>0.95 vs >0.93)
because I weight the data-composition friendliness (more malignant
examples, consistent modality) more than the apparent-AUC deflation from
the W2/W8 fixes. My PR-AUC range is slightly higher (0.55 – 0.85) for
the same reason. Otherwise the brackets are broadly aligned.
