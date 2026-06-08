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
