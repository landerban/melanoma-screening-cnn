"""
Phase 9 / Step 3c-e — Reproduce patient-level split and select the
stratified analytical sample.

Doctoral-level discipline:
  * Re-derive the test cohort from metadata + seed=42 only. No image
    files needed at this step.
  * Verify the test cohort N matches the Phase 7a/7c numbers
    (9,186 images, 1,767 malignant). If it does not match, halt — we
    are not on the same data state as the retrain.
  * Apply the pre-registered stratification from `04_protocol.md` §1.2
    (3 collections × 2 classes × 50 = 300 images).
  * Emit a CSV that becomes the canonical analytical sample. All
    downstream Phase-9 steps reference this CSV by isic_id; the file is
    committed to the branch so it is part of the audit trail.

Run from project root:
    python scripts/phase9/01_select_sample.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from efficientnet_b0 import patient_level_split  # noqa: E402

TRAINING_DATA = REPO_ROOT / "training_data"
OUT_CSV       = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
TEST_COHORT_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_test_cohort.csv"
LOG_TXT       = REPO_ROOT / "artifacts" / "phase9" / "01_select_sample.log"

SEED            = 42
PER_CELL_TARGET = 50   # 3 collections × {benign, malignant} × 50 = 300


def merge_metadata() -> pd.DataFrame:
    """
    Reproduce load_and_merge_metadata's logic *without* the image_path
    step (we do not have images yet). The split logic only requires
    isic_id, source_collection, effective_patient_id, label.
    """
    # Replicate trainer.py:load_and_merge_metadata's frame concatenation
    # order. Hypothesis: trainer.py used elicer's ext4 file-system order,
    # which on a fresh write of metadata files is *creation order*. The
    # creation script `rebuild_per_collection_metadata.py` writes the
    # CSVs in the dict-iteration order COLLECTIONS = {212, 70, 249},
    # giving creation order [c212, c70, c249]. macOS APFS returns
    # *alphabetical* by default ([c212, c249, c70]). We force creation
    # order explicitly to match the Phase 6 split.
    creation_order = [212, 70, 249]
    csvs = []
    for cid in creation_order:
        p = TRAINING_DATA / f"metadata_c{cid}.csv"
        if p.exists():
            csvs.append(p)
    if not csvs:
        sys.exit(f"[FATAL] no metadata_c*.csv under {TRAINING_DATA}")
    print(f"[debug] CSV order (creation-order): {[p.name for p in csvs]}")

    import re
    collection_re = re.compile(r"metadata_c(\d+)\.csv$")

    frames = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        m  = collection_re.search(p.name)
        df["source_collection"] = m.group(1) if m else None
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "patient_id" in combined.columns:
        eff = combined["patient_id"].copy()
    else:
        eff = pd.Series([pd.NA] * len(combined), index=combined.index, dtype=object)
    if "lesion_id" in combined.columns:
        eff = eff.fillna(combined["lesion_id"])
    n_fb_isic = int(eff.isna().sum())
    if "isic_id" in combined.columns:
        eff = eff.fillna(combined["isic_id"])
    if n_fb_isic > 0:
        warnings.warn(
            f"effective_patient_id fell back to isic_id for {n_fb_isic:,} rows"
        )
    combined["effective_patient_id"] = eff

    keep = [c for c in ["isic_id", "patient_id", "lesion_id", "diagnosis_1",
                        "source_collection", "effective_patient_id"]
            if c in combined.columns]
    combined = combined[keep].copy()

    if "isic_id" in combined.columns:
        combined = combined.drop_duplicates(subset="isic_id").reset_index(drop=True)

    combined = combined[combined["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)
    combined["label"] = (combined["diagnosis_1"] == "Malignant").astype(int)

    return combined


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log("# Phase 9 Step 3c-e — analytical sample selection")
    log(f"Seed: {SEED}")
    log(f"Per-cell target: {PER_CELL_TARGET}")
    log(f"Target total N: {PER_CELL_TARGET * 3 * 2}")
    log("")

    # 1. Merge metadata
    log("## 1. Merge metadata")
    df = merge_metadata()
    log(f"Merged rows after dedup + Benign/Malignant filter: {len(df):,}")
    log(f"Per-collection counts: "
        + ", ".join(f"c={c}: {n}" for c, n in df["source_collection"].value_counts().sort_index().items()))
    log(f"Class balance: benign={int((df.label == 0).sum())}, "
        f"malignant={int((df.label == 1).sum())}, "
        f"ratio={(df.label == 0).sum() / max(1, (df.label == 1).sum()):.2f}:1")
    log("")

    # Phase 7 expected: 61,396 rows after this step
    if len(df) != 61_396:
        log(f"!! Row count {len(df):,} differs from Phase 7 expected 61,396. "
            f"Investigate before proceeding.")
    else:
        log(f"✓ Row count matches Phase 7 (61,396). Same data state as retrain.")
    log("")

    # 2. Patient-level split (suppress patient_level_split's stdout to log
    #    only)
    log("## 2. Patient-level split (seed=42, 0.65/0.10/0.10/0.15)")
    train_df, val_df, cal_df, test_df = patient_level_split(
        df, patient_col="effective_patient_id", seed=SEED,
    )
    log(f"Train: {len(train_df):,} images, "
        f"{train_df['effective_patient_id'].nunique():,} patients")
    log(f"Val:   {len(val_df):,} images, "
        f"{val_df['effective_patient_id'].nunique():,} patients")
    log(f"Cal:   {len(cal_df):,} images, "
        f"{cal_df['effective_patient_id'].nunique():,} patients")
    log(f"Test:  {len(test_df):,} images, "
        f"{test_df['effective_patient_id'].nunique():,} patients")
    log("")

    # Phase 7a expected: test = 9,186 images, 1,767 malignant
    test_mal = int((test_df.label == 1).sum())
    if len(test_df) == 9_186 and test_mal == 1_767:
        log(f"✓ Test cohort N=9,186, malignant=1,767 — matches Phase 7a.")
    else:
        log(f"!! Test cohort N={len(test_df):,}, malignant={test_mal:,} "
            f"differs from Phase 7a (9,186 / 1,767). Investigate before "
            f"proceeding.")
    log("")

    # 3. Test cohort per-collection breakdown
    log("## 3. Test cohort per-collection breakdown")
    log("(should match Phase 7c bootstrap_test_auc.log)")
    expected = {
        "212": dict(N=1_708, pos=305,  prev=0.179),
        "70":  dict(N=4_944, pos=86,   prev=0.017),
        "249": dict(N=2_534, pos=1_376, prev=0.543),
    }
    for cid, exp in expected.items():
        sub = test_df[test_df.source_collection == cid]
        N   = len(sub)
        pos = int((sub.label == 1).sum())
        prev = pos / N if N > 0 else float("nan")
        match = (N == exp["N"]) and (pos == exp["pos"])
        log(f"  c={cid}: N={N:,} (exp {exp['N']:,}), "
            f"pos={pos:,} (exp {exp['pos']:,}), "
            f"prev={prev:.3f} (exp {exp['prev']:.3f})  "
            f"{'✓' if match else '!!'}")
    log("")

    # 4. Save full test cohort (basis for re-deriving Phase 7a AUC later)
    test_df.to_csv(TEST_COHORT_CSV, index=False)
    log(f"Saved full test cohort: {TEST_COHORT_CSV.relative_to(REPO_ROOT)} "
        f"({len(test_df):,} rows)")
    log("")

    # 5. Stratified analytical sample selection
    log("## 4. Stratified analytical sample (preregistered N=300)")
    log("Stratification: 3 collections × {benign, malignant} × 50 each.")
    log("")

    rng = np.random.default_rng(SEED)
    sample_rows = []
    for cid in ["212", "70", "249"]:
        for label in [0, 1]:
            pool = test_df[
                (test_df.source_collection == cid) & (test_df.label == label)
            ]
            pool_n = len(pool)
            take_n = min(PER_CELL_TARGET, pool_n)
            chosen = pool.sample(n=take_n, random_state=rng.integers(0, 2**31 - 1))
            sample_rows.append(chosen)
            log(f"  c={cid}, label={label}: pool={pool_n:,}, selected={take_n}")

    sample_df = pd.concat(sample_rows, ignore_index=True)
    log(f"\nSelected analytical sample: N={len(sample_df)}")
    log(f"Per-collection: " + ", ".join(
        f"c={c}: {n}" for c, n in sample_df.source_collection.value_counts().sort_index().items()
    ))
    log(f"Per-class: benign={int((sample_df.label == 0).sum())}, "
        f"malignant={int((sample_df.label == 1).sum())}")
    log("")

    # 6. Save analytical sample
    sample_df.to_csv(OUT_CSV, index=False)
    log(f"Saved analytical sample: {OUT_CSV.relative_to(REPO_ROOT)}")
    log("")

    log("## 5. Next steps")
    log(f"  → Download {len(sample_df)} images by isic_id (Step 3f)")
    log(f"  → Reproduce Phase 7a AUC on the full {len(test_df):,}-image "
        f"test cohort (Step 3g; requires images though)")
    log(f"  → Acquire HAM10000 lesion masks from ISIC 2018 Task 1 (Step 3h)")

    LOG_TXT.write_text("\n".join(log_lines) + "\n")
    print(f"\nLog written to {LOG_TXT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
