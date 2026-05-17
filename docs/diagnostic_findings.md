# Phase 3 diagnostic findings

Consolidates the four pre-Phase-4 diagnostic outputs (patient_id audit,
lesion_id grouping, merge consistency, pre-retrain bootstrap). This
document is the single source of truth that Phase 8 folds back into
`docs/midterm-prep-context.md` §§2/7/8. Until then, the audit doc
keeps its pre-rewrite anchors (161:1, 71/53,604 cohort, 0.347
threshold, etc.) -- they are still the canonical numbers for the
RETROSPECTIVE model that `best_model.pth` actually is.

Working dataset for these diagnostics: training-pool ISIC collections
`c=212` (HAM10000), `c=70` (SIIM-ISIC 2020), `c=249` (BCN20000). The
old `c=390` (Challenge 2024 TBP) was dropped per the Phase 2 plan.
PAD-UFES-20 (`c=406`) is held out for Phase 7b and is not part of any
of the per-collection numbers below.

---

## Patient-ID leak audit

Full output of `scripts/audit_patient_id.py`:

```
Scanning 3 CSV(s) under /home/elicer/AI_project/training_data

PER-CSV REPORT
------------------------------------------------------------------------------
  metadata_c212.csv
    rows                : 11,720
    has patient_id      : False
    patient_id NaN rate : 1.000   <- all-NaN; patient_level_split will collapse to one bucket
    has diagnosis_1     : True
    diagnosis_1 top-6   : [('Benign', 9415), ('Malignant', 2156), ('Indeterminate', 149)]
    isic_id duplicates within file: 0

  metadata_c70.csv
    rows                : 33,126
    has patient_id      : True
    patient_id NaN rate : 0.000
    has diagnosis_1     : True
    diagnosis_1 top-6   : [('Benign', 32539), ('Malignant', 584), ('Indeterminate', 3)]
    isic_id duplicates within file: 0

  metadata_c249.csv
    rows                : 18,946
    has patient_id      : False
    patient_id NaN rate : 1.000   <- all-NaN; patient_level_split will collapse to one bucket
    has diagnosis_1     : True
    diagnosis_1 top-6   : [('Malignant', 8871), ('Benign', 7831), (nan, 1156), ('Indeterminate', 1088)]
    isic_id duplicates within file: 0

CROSS-CSV REPORT
------------------------------------------------------------------------------
  isic_ids appearing in >1 CSV : 0
  ...of which have CONFLICTING diagnosis_1 between CSVs : 0

POST-MERGE REPORT (mimics trainer.load_and_merge_metadata)
------------------------------------------------------------------------------
  rows after concat+dedup+label filter : 61,396

  FAILURE MODE A: 28,273 rows (46.05% of merged set)
    have NaN patient_id and would all be bucketed into ONE synthetic
    NaN-patient by np.unique() at efficientnet_b0.py:218.
    Those rows land entirely in train OR val OR test depending on shuffle order.

  unique patient_ids (incl. NaN) : 2,057
  unique patient_ids (excl. NaN) : 2,056
  mean images per patient        : 29.86
```

**Conclusion.** Failure mode A fires on `c=212` and `c=249` (no
`patient_id` column), placing **28,273 / 61,396 rows (46.05%)** in the
synthetic-NaN bucket. Failure mode B does NOT fire because `c=70`
carries `patient_id`, so the merged frame has the column. The Phase 4a
fallback chain (`patient_id` -> `lesion_id` -> `isic_id`) is the
correct fix; the lesion_id grouping check below quantifies how honest
that fallback actually is per collection.

---

## Lesion-ID grouping

Full output of `scripts/audit_lesion_id.py`:

```
Scanning 3 per-collection CSV(s) under /home/elicer/AI_project/training_data

PER-COLLECTION REPORT
------------------------------------------------------------------------------
  metadata_c212.csv
    rows                  : 11,720
    has lesion_id         : True
    NaN lesion_id rows    : 0
    unique lesion_id      : 8,838
    mean rows / lesion    : 1.326
    median rows / lesion  : 1.0
    max rows / lesion     : 6
    unique/rows ratio     : 0.754  (1.0 = 1:1 image-level)
    -> Weak grouping (mean < 2). Honest patient-proxy but the
       leakage protection it provides is limited.

  metadata_c249.csv
    rows                  : 18,946
    has lesion_id         : True
    NaN lesion_id rows    : 0
    unique lesion_id      : 5,440
    mean rows / lesion    : 3.483
    median rows / lesion  : 3.0
    max rows / lesion     : 31
    unique/rows ratio     : 0.287  (1.0 = 1:1 image-level)
    -> Genuine multi-images-per-lesion grouping. Defensible as
       a patient-proxy for split purposes.

  metadata_c70.csv
    rows                  : 33,126
    has lesion_id         : True
    NaN lesion_id rows    : 0
    unique lesion_id      : 32,701
    mean rows / lesion    : 1.013
    median rows / lesion  : 1.0
    max rows / lesion     : 2
    unique/rows ratio     : 0.987  (1.0 = 1:1 image-level)
    -> NEAR 1:1 GROUPING. Using lesion_id as patient-proxy here
       is effectively image-level split. Disclose on slide as such.


POST-MERGE GROUPING
------------------------------------------------------------------------------
effective_patient_id = patient_id || lesion_id || isic_id

  rows after concat+dedup+label filter : 61,396
  effective_patient_id NaN rows        : 0
  unique effective_patient_ids         : 15,558
  mean rows per group                  : 3.946
  median rows per group                : 2.0
  max rows per group                   : 115

  effective_patient_id source breakdown:
    patient_id  : 33,123 rows  (53.95%)
    lesion_id   : 28,273 rows  (46.05%)
    isic_id     :      0 rows  ( 0.00%)
```

**Conclusions, one line per collection.**

- **c=212 HAM10000**: lesion_id is a **weak patient-proxy** (mean 1.33 rows/lesion, max 6). Honest fallback but leakage protection is limited; slide should disclose that ~75% of c=212 lesions appear in exactly one image.
- **c=249 BCN20000**: lesion_id is a **genuine patient-proxy** (mean 3.48 rows/lesion, max 31). Defensible as a split key. The user's "BCN20000 lesion_id is 1:1" trigger does NOT fire.
- **c=70 SIIM-ISIC 2020**: lesion_id is near-1:1 (mean 1.01, max 2), but irrelevant -- c=70 carries `patient_id` so the effective_patient_id for these rows comes from there, not lesion_id.

**Post-merge picture.** Phase 4a's fallback chain produces zero
isic_id-level rows. 53.95% of rows use `patient_id` (all from c=70);
46.05% use `lesion_id` (c=212 + c=249). Mean 3.95 images per
effective_patient_id group, max 115 -- defensible as a patient-level
split overall, with the per-collection caveat above for c=212.

---

## Merge consistency

Full output of `scripts/audit_merge_consistency.py`:

```
Concatenated 3 CSV(s) -> 63,792 rows total

CHECK (a): diagnosis_1 vocabulary
------------------------------------------------------------------------------
diagnosis_1
Benign           49785
Malignant        11611
Indeterminate     1240
NaN               1156

  trainer.py:161 will silently drop 2,396 rows in these classes:
    'Indeterminate'           : 1,240
    nan                       : 1,156

CHECK (b): image_type per source_collection
------------------------------------------------------------------------------
image_type         dermoscopic  NaN
source_collection                  
212                      11719    1
249                      18946    0
70                       33126    0

  All training-pool rows are dermoscopic (no 'clinical:*' types).

CHECK (c): isic_id cross-collection duplicates
------------------------------------------------------------------------------
  isic_id duplicate rows           : 0
  No cross-collection isic_id duplicates. trainer.py:158 dedup is a no-op.
```

**Three one-line conclusions.**

- **(a) diagnosis_1 vocabulary.** `trainer.py:161` will silently drop 2,396 rows -- 1,240 `Indeterminate` (149 from c=212, 3 from c=70, 1,088 from c=249) and 1,156 NaN (all from c=249). All non-`{Benign, Malignant}` values are reasonable drops; no remapping required. Filter is doing its job.
- **(b) image_type per collection.** All 63,792 training-pool rows are `dermoscopic` (one NaN in c=212). No `clinical:*` rows -- **the modality leak the c=390 drop was designed to retire is now absent from the training pool**. The W8 modality-shortcut concern at the data level is resolved by collection composition; what remains is per-collection AUC measurement (Phase 7c).
- **(c) cross-collection duplicates.** 0 duplicate `isic_id` rows across the three CSVs. `trainer.py:158` dedup is a no-op for this composition. The first-write-wins ambiguity is structurally impossible to trigger here.

**Incidental finding worth surfacing.** Post-filter class balance is
**49,785 Benign / 11,611 Malignant = 4.29:1**, NOT the audit doc's
161:1. 76.4% of all malignants (8,871/11,611) come from c=249
(BCN20000). This is a structural shift -- the dataset is no longer
catastrophically imbalanced. Threshold-calibration math, focal-α
sizing, and "F1 is meaningless at this prevalence" framing all need to
be revisited in Phase 8.

---

## Pre-retrain bootstrap (methodology, NOT a headline number)

`scripts/bootstrap_auc_ci.py`, old `best_model.pth` evaluated on the
NEW val split.

| Metric | Value |
|---|---|
| Val cohort | 5,168 images (79 pos / 5,089 neg, prevalence 1.53%) |
| Patient split | 1,441 train / 308 val / 308 test patients |
| ROC-AUC (point) | **0.6243** |
| ROC-AUC 95% CI (1000 bootstrap resamples, seed=42) | **[0.5561, 0.6866]** |
| CI half-width | ±0.0653 (wider than Hanley-McNeil's ±~0.04 expectation; smaller cohort) |
| Recall at threshold 0.347 | 0.899 (71/79) |
| Specificity at threshold 0.347 | 0.204 (1040/5089) |
| Confusion matrix at 0.347 | TN=1040, FP=4049, FN=8, TP=71 |

> This result is a floor estimate, not a headline. The old checkpoint
> was trained on {c=390, c=70, c=212} and is being evaluated on
> {c=70, c=212, c=249} -- a mismatched pair that does not appear anywhere
> else in the project narrative. The 0.803/0.876 reproduction check
> against CLAUDE.md is structurally impossible here because the val
> cohort composition has changed.
>
> Two legitimate uses for this number:
> 1. **Floor estimate** for the Phase 6 retrain. The new model trained
>    on the corrected pipeline should clear this AUC on the new val
>    split; if it doesn't, something in Phase 4 broke.
> 2. **Soft evidence on the c=390 drop.** If this AUC lands near the
>    original 0.928, that suggests c=390 was not the source of the
>    model's discriminative signal -- supporting the W8 modality-shortcut
>    concern at the dataset level. If it lands meaningfully lower, the
>    interpretation is murkier (could be missing c=390 features, could
>    be c=249 being out-of-distribution for an old checkpoint, can't
>    disambiguate from one number).
>
> The headline AUC for the midterm comes from the RETRAINED model on
> the new held-out test split (Phase 7a) plus PAD-UFES-20 (Phase 7b).
> This number does NOT go on slide 7.

---

## Ghost evaluation -- old ckpt on c=70 + c=212 only (filter was a no-op)

`scripts/eval_oldckpt_no_c249.py` was designed to filter the val to
c=70 + c=212 rows and re-run inference, so we could disambiguate
"c=390 wasn't carrying the discriminative signal" from "c=249 is OOD
for an old checkpoint." Result: **the filter dropped 0 rows.** The val
cohort is already c=70-only before any filtering.

```
device   : cuda
ckpt     : best_model.pth
threshold: 0.347

Split: train 50,723 imgs (1,441 patients) | val 5,168 (308) | test 5,505 (308)
Full val cohort (matches Phase 3)        : 5,168 (pos=79)
  source breakdown                       : {'70': 5168}
Ghost val (c=70 + c=212 only)            : 5,168 (pos=79, neg=5089)
  dropped (c=249 + any untagged)         : 0

ROC-AUC = 0.6243  [95% CI: 0.5561, 0.6866]  (1000 bootstraps, seed=42)
At threshold 0.347:
  recall      = 0.899  (71/79)
  specificity = 0.204  (1040/5089)
  CM: TN=1040  FP=4049  FN=8  TP=71
```

**Why it's a no-op (structural).** All c=212 and c=249 rows have NaN
`patient_id` and collapse into one synthetic NaN-patient bucket in
`patient_level_split` (`efficientnet_b0.py:218`). With seed=42 that
bucket lands in **train**, never in val. So val = 308 c=70 patients,
~5,168 c=70 images. The c=70+c=212 filter has nothing to filter out.

**What this actually says about the c=390 drop.** The naive ghost
comparison failed, but the structural picture is cleaner than I had it
in §3 above:

1. The new val is **c=70 only**. The Phase 3 bootstrap 0.6243 ± 0.065
   is a c=70-only-val result, NOT a "mixed cohort" result. My write-up
   in the section above this one framed it as "31% BCN20000
   composition" -- that framing is wrong and needs the Phase 8 rewrite
   to correct it.
2. CLAUDE.md's old val was 53,675 images. The new val is 5,168. The
   structural difference: the old val almost certainly absorbed the
   NaN bucket (c=212 + c=390 ≈ 413K rows, of which a portion landed in
   val per the same shuffle logic). Old val = c=70-val-cohort + NaN
   bucket (mostly c=390 TBP tiles + c=212 HAM10000 crops).
3. The 0.928 old headline was measured on that mixed dilution cohort.
   The 0.624 new bootstrap is c=70 dermoscopy only, with no easy TBP
   benigns to drag the AUC up.

**Implication for the W8 modality-shortcut concern.** The 0.928 -> 0.624
gap, viewed structurally, is indirect empirical support for the W8
concern: a substantial portion of the old AUC came from the easy-TBP
+ easy-HAM10000 dilution. When that dilution is removed (NaN bucket
shoved to train rather than val), the old checkpoint's c=70-only AUC
is 0.624. The c=390 drop story is therefore defensible empirically
through this lens -- just not in the way the original "ghost
evaluation" plan anticipated.

**Caveat.** This is one number on one shuffle. A cleaner version would
re-evaluate the old checkpoint on a TBP-rich slice (which we don't have
the c=390 data for anymore), or compare the new model after Phase 6
retrain across c=70-only vs c=70+c=212+c=249 val cohorts. The
structural reading above is consistent with W8 but doesn't rise to
"proven."

**Net effect on §§3/7/8 narrative.** The §3 "ghost evaluation can
disambiguate the c=390 question" framing in this document needs to
mark itself as superseded (the filter was a no-op, but the structural
reading provides the answer instead). Phase 8 should fold this in.
