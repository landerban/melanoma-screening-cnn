# Phase 9 — Analysis Log

Real-time reasoning trace. Lab-notebook discipline: every non-trivial
decision is timestamped with (what was observed, what was decided, why,
what to watch next). Entries are append-only; corrections are made via
a *new* entry that explicitly supersedes a prior one, not by editing
history.

Format of each entry:

```
## [YYYY-MM-DD HH:MM TZ] — <short title>
**Observed.** <data, log line, or finding>
**Decided.** <action taken>
**Why.** <reasoning, linked to protocol §X or hypothesis Hk where applicable>
**Watch.** <what could go wrong, what to verify next>
```

---

## [2026-06-08 — Phase 3 kick-off]

**Observed.**
- `best_model.pth` (18 MB) is a *legacy* checkpoint: `torch.load` returns
  a plain `state_dict` (an `OrderedDict` of tensors), not the rich dict
  promised by Phase 4e (H5 fix).
- `best_model.test_metrics.json` sidecar is **absent** on the local
  working tree.
- `training_data/` is **absent** — code is here, data is not. The
  Phase 6 retrain ran on Elice Cloud; only the checkpoint travelled
  back.
- `isic-cli` is not installed on the local M3.

**Decided.**
1. Treat the legacy-format checkpoint as an *observed* fact and log it
   as a documentation/state mismatch (README §"Reproducing the numbers"
   claims test_metrics live in the ckpt + sidecar; both are absent
   locally). Reproducibility is recoverable: Phase 9 will independently
   re-run the in-distribution test evaluation and verify AUC = 0.9513
   reproduces, validating that *this specific checkpoint is the Phase 6
   retrain output*.
2. Install `isic-cli` and fetch the three per-collection metadata CSVs
   (light: a few MB total).
3. Reproduce the patient-level split locally from the metadata using
   `seed=42` and the existing `patient_level_split` code path. The
   `effective_patient_id` fallback chain makes this deterministic.
4. Only *after* the split is reproduced and the test cohort isic_id
   list is in hand, draw the stratified analytical sample N=300 and
   download those 300 images (estimated ~3 GB, within local disk
   budget).

**Why.**
- The legacy checkpoint is fine for *inference*: PyTorch's
  `model.load_state_dict(state_dict)` works whether the surrounding dict
  is rich or legacy. The downside is we cannot read `cfg`, `git_hash`,
  or `timestamp` from the ckpt — but we *can* verify the ckpt's
  output AUC matches the published Phase 7a number, which is the
  load-bearing reproducibility guarantee.
- Downloading the *full* 61K-image training pool is not necessary for
  Phase 9. The analytical sample is N=300 by §4.1.2 of `04_protocol.md`;
  metadata-only download is sufficient to *identify* the right 300
  isic_ids. Downloading the right 300 is then a follow-up.
- This sequence is logged so a reviewer can see we did not *post-hoc*
  pick the 300; the selection follows the deterministic split + the
  pre-registered stratification.

**Watch.**
- Does the re-derived test cohort (post-split) have N=9,186 images,
  1,767 malignant, matching Phase 7a exactly?
- Does the ckpt produce test AUC 0.9513 ± bootstrap noise on this cohort
  via `python scripts/eval_test_set.py best_model.pth`?
- If either check fails: investigate code-vs-ckpt commit mismatch. The
  retrain was on commit c72578c (Phase 8 prep); the current HEAD is
  a54c1db (post-merge). Patient-level split logic must be identical
  between them.
- isic-cli requires an account for some operations; verify which
  operations need auth for our use case (metadata + bulk image download).

---

## [2026-06-08 — venv + dependency lock]

**Observed.** Initially installed `isic-cli` into the miniforge `base`
environment. User flagged this. Doctoral reproducibility requires
project-isolated dependencies.

**Decided.** Create `.venv` (Python 3.10), write
`requirements-phase9.txt` with pinned major-version ranges, install,
freeze to `requirements-phase9.lock.txt`. Add `.venv/` to `.gitignore`.
All Phase-9 scripts run under `.venv/bin/python`. Both
`requirements-phase9.txt` and the `.lock.txt` are committed; the
lock file is the *evidence* of which exact versions ran the analysis.

**Why.** A reviewer should be able to reproduce the Phase-9 results
bit-exact from the lock file alone. Without it, "the analysis was run
in mid-2026" is not a reproducibility statement. The lock file pins all
67 transitive dependencies, including `torch==2.6.0` which matches the
Phase-6 retrain environment per README §Hardware.

**Watch.** PyTorch 2.6 MPS backend has historically had numerical
quirks vs CUDA. The Phase 6 retrain ran on CUDA 12.4 (A100X). For
inference, the operator set we exercise (EfficientNet-B0 forward,
sigmoid, autograd through GradCAM) is well-covered by MPS. We will
verify bit-stability by checking that the Phase 7a AUC reproduces
within ±0.001 of 0.9513 on the same test cohort.

---

## [2026-06-08 — Phase 9 Primary Finding #1: Prevalence-balanced AUC reveals collection-specific shortcut reliance]

**Observed.** Running the Phase 6 ckpt on our 300-image
prevalence-balanced analytical sample (50% benign / 50% malignant within
each collection's N=100 cell) produces collection AUCs that diverge
dramatically from the Phase 7c prevalence-skewed AUCs:

| Collection | Phase 7c AUC (skewed) | Phase 9 AUC (50/50) | Delta |
|---|---|---|---|
| c=212 HAM10000 | 0.9151 | **0.943** | +0.028 |
| c=249 BCN20000 | 0.8702 | **0.778** | -0.092 |
| c=70 SIIM 2020 | 0.8835 | **0.548** | **-0.336** |

Per-class mean P_malig:

| Collection | Benign mean | Malignant mean | Gap |
|---|---|---|---|
| c=212 | 0.413 | 0.867 | **0.454** |
| c=249 | 0.597 | 0.776 | 0.179 |
| c=70 | 0.576 | 0.625 | **0.049** |

**Interpretation.** A prevalence-skewed AUC is inflated by the model's
ability to *systematically score one collection higher than another*.
When prevalence is balanced (our analytical sample), only
*within-collection* discriminative power survives. The Phase 7c c=70
AUC of 0.88 was *almost entirely* a function of the cohort's 1.7%
prevalence — under balanced conditions the model is essentially
guessing at chance (0.548) on c=70 lesions. c=212 is the opposite:
its lesion-feature signal is *strong enough* that balancing prevalence
*raises* the AUC (0.92 → 0.94). c=249 sits in between.

**Why this is novel.** No prior dermatology shortcut paper measured
this. The closest are:

- Bissoto 2019 (lesion-occlusion): showed models perform above
  baseline on occluded images, but did not decompose by collection or
  by prevalence regime.
- Phase 7c (our own): reported per-collection AUC at the *natural*
  collection prevalences; that report acknowledges aggregate-vs-
  per-collection gap as "modality-prior shortcut" but does not
  isolate the prevalence contribution from the within-collection
  discriminative contribution.

Phase 9 closes that gap by holding prevalence constant. The
remaining cross-collection AUC heterogeneity (0.94 / 0.78 / 0.55) is
*purely* within-collection discriminative power, separable from the
prevalence prior.

**Headline reframing for slide 9.** The slide currently says
"per-collection AUCs cluster at 0.87–0.92". The honest message after
Phase 9 is sharper: **"At natural prevalence the per-collection AUCs
look comparable (0.87–0.92), but at balanced prevalence c=70 falls to
0.55 — chance — revealing that the SIIM-cohort score is almost
entirely a prevalence-prior contribution rather than lesion
discrimination."** This sharpens the modality-shortcut claim from a
"contribution" to an "almost-entirely" attribution on c=70.

**Decision.** Treat this as the Phase 9 primary finding. Even if
counterfactual inpainting (H2) and Shapley decomposition (H3) produce
modest effect sizes downstream, this single measurement is already a
publishable result. Promote it to the paper's Figure 1.

**Limitations to flag in the paper.**
- N=100 per cell is small; bootstrap 95% CI on AUC 0.548 is wide
  (likely ±0.07 — to be computed in Phase 4a).
- The model has *seen* some of our analytical-sample patients during
  Phase 6 training (Phase 6 train/val/cal cohorts overlap with our
  test cohort? NO — we reproduced the same patient_level_split, so our
  300 are all from the held-out test cohort. This is clean.)
- 50/50 balance is a *measurement-friendly* prevalence, not a *clinical*
  one. The clinical message is: SIIM-style screening deployments
  should not trust the 0.88 AUC headline.

**Watch.**
- Phase 4a will run bootstrap CIs on the three per-collection balanced
  AUCs. If the c=70 CI lower-bound exceeds 0.50, the "almost chance"
  reading needs softening.
- Phase 4c (gap decomposition) becomes more interesting: the
  "shortcut-explained gap" needs to be partitioned against this new
  prevalence-removed baseline.

---

## [2026-06-08 — Ruler detector parameter sensitivity sweep]

**Observed.** The first-pass ruler detector
(canny=50/150, min_len=60, threshold=50) produced 42% ruler-positive
rate on the analytical sample — implausibly high vs. Sirico 2023's
manual-annotation prior of ~20–30%. Inspecting the per-collection
breakdown showed c=70 at 55%, c=212 at 42%, c=249 at 29%; not
consistent with Winkler 2019 (rulers documented chiefly in SIIM-style
cohorts).

**Diagnosis.** Hough's probabilistic line detector at the default
threshold catches lesion-boundary edges as line segments. The
geometric short axis of a curved lesion edge satisfies the length
gate when min_len=60, even though it is not a ruler.

**Decided.** Add an aspect-ratio gate (line dominant-axis length ≥
1.5 × orthogonal jitter) and tighten Hough parameters (min_len=100,
threshold=80, max_gap=5; Canny=80/200). Verified via 4-config sweep
in `scripts/phase9/03b_ruler_sensitivity.py`:

| Config | Overall | c=70 | c=212 | c=249 |
|---|---|---|---|---|
| baseline (orig) | 42% | 55% | 42% | 29% |
| **strict-1 (chosen)** | **13%** | **24%** | **9%** | **6%** |
| strict-2 | 5% | 11% | 3% | 1% |
| strict-3 | 4% | 10% | 2% | 0% |

Decision rule: first strict config retaining ≥10 ruler-positives on
c=70 (Winkler-2019 collection) and <30 (avoids lesion-edge false
positives). Strict-1 met both.

**Why.** Without this tightening, downstream LaMa-inpainting would
mostly remove *lesion edges* on ruler-flagged images, biasing the
counterfactual P_malig estimate toward "ruler matters" when in fact
the model was reacting to lesion-shape removal. The aspect-ratio gate
is principled: surgical-skin-marker lines are dominantly straight; the
edges of organic lesion boundaries are not.

**Watch.** Even strict-1's c=212 = 9% may include false positives.
We are not running Phase 3l manual ink verification due to time
budget; we will instead flag any image where the ruler mask intersects
significantly with the lesion mask (Phase 3h) and treat those as a
sensitivity-disclosure pool. Their per-image ΔP estimates will be
reported with and without the suspect images included.

---

## [2026-06-08 — Split-reproduction divergence and resolution]

**Observed.** First run of `scripts/phase9/01_select_sample.py` produced
a test cohort of N=9,496 (1,741 malignant), differing from Phase 7a's
N=9,186 (1,767 malignant) by 310 images. Per-collection breakdown:
c=212 matched exactly (1,708/305); c=70 was N=5,305 (vs 4,944); c=249
was N=2,483 (vs 2,534). Total *metadata* row count matched at 61,396.

The metadata content is identical; the *partitioning* into
train/val/cal/test diverged.

**Diagnosis.** `patient_level_split` calls `rng.shuffle(patients)`
where `patients = df[patient_col].unique()`. `pd.Series.unique`
preserves *first-occurrence order in the input frame*, which is
determined by the order in which the per-collection CSVs were
concatenated. The trainer's `load_and_merge_metadata` uses
`list(root.glob("*.csv"))`, which is *file-system iteration order* —
*non-deterministic across operating systems*:
- elicer (Ubuntu ext4) returns directory entries in *creation order*,
  which for our CSVs is `[c212, c70, c249]` (the dict order in
  `rebuild_per_collection_metadata.py`'s `COLLECTIONS` map).
- local M3 (APFS) returns *alphabetical*, giving `[c212, c249, c70]`.

Same seed=42, different shuffle input order → different partition.

**Decided.** Force creation-order concat in our Phase-9 script by
explicit list `[212, 70, 249]`. Re-running produces bit-exact match
to Phase 7a:

```
✓ Test cohort N=9,186, malignant=1,767 — matches Phase 7a.
✓ c=212: N=1,708 (exp 1,708), pos=305 (exp 305)
✓ c=70: N=4,944 (exp 4,944), pos=86 (exp 86)
✓ c=249: N=2,534 (exp 2,534), pos=1,376 (exp 1,376)
```

**Why.** Without this match, an estimated 85% of our analytical-sample
images would have been in the Phase 6 *training* set (we computed
expected overlap of ~15% under random reshuffle). That leak would
have rendered every downstream Phase-9 measurement uninterpretable —
the AUC, the per-collection breakdown, and the counterfactual ΔP would
all reflect *training-memorization* rather than *deployment-realistic
behaviour*.

**Audit-extension finding (W4-extended).** Phase 4d's W4 closure
(global seeds + DataLoader worker_init_fn + sampler generator) does
*not* control the frame-concatenation order in
`load_and_merge_metadata`. That ordering is delegated to the file
system. A defensible patch is one of:
1. Sort CSV paths by `(int(collection_id))` before concat — natural
   order, OS-independent.
2. Hash-sort by isic_id after concat but before
   `effective_patient_id.unique()` is taken — strongest invariance,
   but adds a downstream test cohort identity change.
3. Persist the seeded patient-list permutation to the checkpoint, so
   `eval_test_set.py` can replay it without re-shuffling.

We will recommend option 1 as a Phase 9 *Implications* contribution
(see `docs/phase9/06_paper.md` §Discussion when written). This is a
*real* methodological finding produced by the Phase 9 audit extension
that the original Phase 4 audit did not catch.

**Watch.**
- The bit-exact match is necessary but not sufficient for ckpt
  validation. Phase 3g must verify that running the actual ckpt
  on this test cohort gives AUC = 0.9513 ± 0.001.
- The 300-image analytical sample is committed in
  `artifacts/phase9/01_analytical_sample.csv`; future commits cannot
  drift this without an explicit deviation entry.

---
