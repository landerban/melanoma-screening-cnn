"""
audit_patient_id.py — Phase 1 / W2-prereq diagnostic.

Answers two questions about whether the patient-level split in
`efficientnet_b0.patient_level_split` is actually patient-level:

  Failure mode A: `patient_id` column is PRESENT but some rows are NaN.
                  Those rows collapse into a single synthetic NaN-patient
                  via np.unique() and all land in one split.

  Failure mode B: `patient_id` column is ABSENT entirely (or 100% NaN).
                  `trainer.py:234` silently falls back to splitting on
                  `isic_id`, which is per-image-unique. That is an
                  image-level random split — CLAUDE.md hard constraint
                  #1 violation, dressed up as a patient-level call.

Run from the project root:
    python scripts/audit_patient_id.py [training_data_dir]

Default training_data_dir = ./training_data
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(training_data_dir: str = "training_data") -> None:
    root = Path(training_data_dir)
    if not root.exists():
        sys.exit(f"[ERROR] {root.resolve()} does not exist")

    csvs = list(root.glob("*.csv")) + list(root.glob("*/*.csv"))
    if not csvs:
        sys.exit(f"[ERROR] no CSVs found under {root.resolve()}")

    print(f"Scanning {len(csvs)} CSV(s) under {root.resolve()}\n")

    # --- Per-CSV ---------------------------------------------------------
    print("PER-CSV REPORT")
    print("-" * 78)
    frames: dict[str, pd.DataFrame] = {}
    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:
            print(f"  {p.name}: READ FAILED -- {exc}")
            continue
        frames[p.name] = df

        has_pid = "patient_id" in df.columns
        nan_pid = df["patient_id"].isna().mean() if has_pid else 1.0

        has_dx  = "diagnosis_1" in df.columns
        dx_counts = (
            df["diagnosis_1"].value_counts(dropna=False).to_dict()
            if has_dx else {}
        )

        has_isic = "isic_id" in df.columns
        n_dup    = (
            df["isic_id"].duplicated().sum() if has_isic else "no isic_id column"
        )

        print(f"  {p.name}")
        print(f"    rows                : {len(df):,}")
        print(f"    has patient_id      : {has_pid}")
        print(f"    patient_id NaN rate : {nan_pid:.3f}"
              f"{'   <- all-NaN; patient_level_split will collapse to one bucket' if nan_pid > 0.99 else ''}")
        print(f"    has diagnosis_1     : {has_dx}")
        if has_dx:
            top = list(dx_counts.items())[:6]
            print(f"    diagnosis_1 top-6   : {top}")
        print(f"    isic_id duplicates within file: {n_dup}")
        print()

    # --- Cross-CSV overlap and label conflicts ---------------------------
    print("CROSS-CSV REPORT")
    print("-" * 78)
    isic_to_csvs: dict[str, list[tuple[str, str]]] = {}
    for name, df in frames.items():
        if "isic_id" not in df.columns or "diagnosis_1" not in df.columns:
            continue
        for _, row in df[["isic_id", "diagnosis_1"]].iterrows():
            isic_to_csvs.setdefault(row["isic_id"], []).append((name, str(row["diagnosis_1"])))

    appearing_in_multiple = {k: v for k, v in isic_to_csvs.items() if len(v) > 1}
    conflicts = {
        k: v for k, v in appearing_in_multiple.items()
        if len({dx for _, dx in v}) > 1
    }
    print(f"  isic_ids appearing in >1 CSV : {len(appearing_in_multiple):,}")
    print(f"  ...of which have CONFLICTING diagnosis_1 between CSVs : {len(conflicts):,}")
    if conflicts:
        print("\n  First 5 conflicts:")
        for k, v in list(conflicts.items())[:5]:
            print(f"    {k}: {v}")
    print()

    # --- After load_and_merge_metadata semantics ------------------------
    print("POST-MERGE REPORT (mimics trainer.load_and_merge_metadata)")
    print("-" * 78)
    combined = pd.concat(list(frames.values()), ignore_index=True)
    keep = [c for c in ["isic_id", "patient_id", "lesion_id", "diagnosis_1"] if c in combined.columns]
    combined = combined[keep].copy()
    if "isic_id" in combined.columns:
        combined.drop_duplicates(subset="isic_id", inplace=True)
    combined = combined[combined["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)

    print(f"  rows after concat+dedup+label filter : {len(combined):,}")

    # Explicit failure-mode detection (A vs B) -- Phase 3 addition.
    print()
    if "patient_id" not in combined.columns:
        print("  FAILURE MODE B HIT: 'patient_id' column absent from merged metadata.")
        print("    trainer.py:234 will silently fall back to splitting on isic_id,")
        print("    which is per-image-unique. That is an IMAGE-LEVEL random split,")
        print("    masked as a patient-level call -- CLAUDE.md hard constraint #1 violation.")
    elif combined["patient_id"].isna().all():
        print("  FAILURE MODE B HIT: 'patient_id' column exists but is 100% NaN.")
        print("    Same effective outcome as column absent -- isic_id fallback applies.")
    else:
        nan_count = int(combined["patient_id"].isna().sum())
        nan_pct = 100.0 * nan_count / len(combined)
        if nan_count > 0:
            print(f"  FAILURE MODE A: {nan_count:,} rows ({nan_pct:.2f}% of merged set)")
            print(f"    have NaN patient_id and would all be bucketed into ONE synthetic")
            print(f"    NaN-patient by np.unique() at efficientnet_b0.py:218.")
            print(f"    Those rows land entirely in train OR val OR test depending on shuffle order.")
        else:
            print(f"  CLEAN: patient_id column present, no NaN rows.")
            print(f"    Both failure modes A and B do NOT fire on this merged set.")

    if "patient_id" in combined.columns:
        print()
        print(f"  unique patient_ids (incl. NaN) : "
              f"{combined['patient_id'].nunique(dropna=False):,}")
        print(f"  unique patient_ids (excl. NaN) : "
              f"{combined['patient_id'].nunique(dropna=True):,}")
        # Heuristic: if mean images-per-patient is very small (<= ~1.5), that means
        # patient_id was essentially functioning as lesion_id / per-image id anyway.
        n_pat = combined["patient_id"].nunique(dropna=True)
        if n_pat > 0:
            ipp = len(combined) / n_pat
            print(f"  mean images per patient        : {ipp:.2f}")
            if ipp < 1.5:
                print(f"    -> very low; patient_id may be a per-lesion or near-unique id,")
                print(f"       not a true patient grouping. Investigate the source CSV.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_data")
