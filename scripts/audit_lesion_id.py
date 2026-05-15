"""
audit_lesion_id.py -- Phase 4a-prereq diagnostic.

Question: is `lesion_id` a genuine multi-images-per-lesion grouping, or
is it effectively 1:1 with `isic_id`? The Phase 4a fallback chain uses
`lesion_id` as a patient-proxy for collections lacking `patient_id`.

If a collection's lesion_id is roughly 1:1 with isic_id, falling back
to it is image-level split with extra steps -- still honest, but it
must be SAID that way on the slide (not called "patient-level").

Reports per-collection: rows total, unique lesion_id, mean rows-per-lesion,
max rows-per-lesion, NaN lesion_id count, unique/rows ratio.

Also reports the post-merge view using effective_patient_id =
patient_id if present-and-non-null else lesion_id else isic_id.

Run from project root:
    python scripts/audit_lesion_id.py [training_data_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def per_collection_report(name: str, df: pd.DataFrame) -> None:
    print(f"  {name}")
    print(f"    rows                  : {len(df):,}")

    if "lesion_id" not in df.columns:
        print(f"    has lesion_id         : False (cannot compute grouping)")
        print()
        return

    nan_lid = int(df["lesion_id"].isna().sum())
    non_nan = df.dropna(subset=["lesion_id"])
    n_unique = non_nan["lesion_id"].nunique()
    counts = non_nan.groupby("lesion_id").size()
    mean_per = float(counts.mean()) if len(counts) else 0.0
    max_per = int(counts.max()) if len(counts) else 0
    median_per = float(counts.median()) if len(counts) else 0.0
    ratio = n_unique / max(len(non_nan), 1)

    print(f"    has lesion_id         : True")
    print(f"    NaN lesion_id rows    : {nan_lid:,}")
    print(f"    unique lesion_id      : {n_unique:,}")
    print(f"    mean rows / lesion    : {mean_per:.3f}")
    print(f"    median rows / lesion  : {median_per:.1f}")
    print(f"    max rows / lesion     : {max_per}")
    print(f"    unique/rows ratio     : {ratio:.3f}  (1.0 = 1:1 image-level)")

    if mean_per < 1.2:
        print(f"    -> NEAR 1:1 GROUPING. Using lesion_id as patient-proxy here")
        print(f"       is effectively image-level split. Disclose on slide as such.")
    elif mean_per < 2.0:
        print(f"    -> Weak grouping (mean < 2). Honest patient-proxy but the")
        print(f"       leakage protection it provides is limited.")
    else:
        print(f"    -> Genuine multi-images-per-lesion grouping. Defensible as")
        print(f"       a patient-proxy for split purposes.")
    print()


def effective_patient_series(df: pd.DataFrame) -> pd.Series:
    """patient_id || lesion_id || isic_id, per row."""
    has_pid_col = "patient_id" in df.columns
    has_lid_col = "lesion_id" in df.columns
    has_iid_col = "isic_id" in df.columns

    pid_notna = df["patient_id"].notna() if has_pid_col else pd.Series([False] * len(df), index=df.index)
    lid_notna = df["lesion_id"].notna() if has_lid_col else pd.Series([False] * len(df), index=df.index)

    eff = pd.Series([pd.NA] * len(df), index=df.index, dtype=object)
    if has_pid_col:
        eff = eff.mask(pid_notna, df["patient_id"])
    if has_lid_col:
        eff = eff.mask(eff.isna() & lid_notna, df["lesion_id"])
    if has_iid_col:
        eff = eff.mask(eff.isna(), df["isic_id"])
    return eff, pid_notna, lid_notna


def main(training_data_dir: str = "training_data") -> None:
    root = Path(training_data_dir)
    if not root.exists():
        sys.exit(f"[ERROR] {root.resolve()} does not exist")

    csvs = sorted(root.glob("metadata_c*.csv"))
    if not csvs:
        sys.exit(f"[ERROR] no metadata_c*.csv found in {root.resolve()}")

    print(f"Scanning {len(csvs)} per-collection CSV(s) under {root.resolve()}\n")

    print("PER-COLLECTION REPORT")
    print("-" * 78)
    frames: dict[str, pd.DataFrame] = {}
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        frames[p.name] = df
        per_collection_report(p.name, df)

    print()
    print("POST-MERGE GROUPING")
    print("-" * 78)
    print("effective_patient_id = patient_id || lesion_id || isic_id")
    print()

    combined = pd.concat(list(frames.values()), ignore_index=True)
    keep = [c for c in ["isic_id", "patient_id", "lesion_id", "diagnosis_1"] if c in combined.columns]
    combined = combined[keep].copy()
    if "isic_id" in combined.columns:
        combined.drop_duplicates(subset="isic_id", inplace=True)
    combined = combined[combined["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)

    eff, pid_notna, lid_notna = effective_patient_series(combined)
    combined["effective_patient_id"] = eff

    print(f"  rows after concat+dedup+label filter : {len(combined):,}")
    n_nan_eff = int(combined["effective_patient_id"].isna().sum())
    print(f"  effective_patient_id NaN rows        : {n_nan_eff:,}")

    n_unique = combined["effective_patient_id"].nunique(dropna=True)
    print(f"  unique effective_patient_ids         : {n_unique:,}")

    counts = combined.dropna(subset=["effective_patient_id"]).groupby("effective_patient_id").size()
    if len(counts):
        print(f"  mean rows per group                  : {counts.mean():.3f}")
        print(f"  median rows per group                : {counts.median():.1f}")
        print(f"  max rows per group                   : {int(counts.max())}")

    # Source attribution: which column provided effective_patient_id for each row?
    src = pd.Series(["isic_id"] * len(combined), index=combined.index)
    if "lesion_id" in combined.columns:
        src = src.mask(lid_notna, "lesion_id")
    if "patient_id" in combined.columns:
        src = src.mask(pid_notna, "patient_id")

    print()
    print("  effective_patient_id source breakdown:")
    for name in ["patient_id", "lesion_id", "isic_id"]:
        n = int((src == name).sum())
        pct = 100.0 * n / len(combined)
        print(f"    {name:<12}: {n:>6,} rows  ({pct:5.2f}%)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_data")
