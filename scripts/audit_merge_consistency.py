"""
audit_merge_consistency.py -- three cross-CSV checks on the training pool.

(a) diagnosis_1 vocabulary outside {Benign, Malignant} -- counted per
    non-standard value. trainer.py:161 silently drops these rows.

(b) image_type distribution per source_collection. All training-pool
    rows should be dermoscopic. Any 'clinical:*' is exactly the modality
    leak the c=390 drop was supposed to retire.

(c) isic_id cross-collection duplicates. For each duplicate isic_id,
    report source_collection values + whether diagnosis_1 agrees.
    trainer.py:158 dedup is first-write-wins and silently picks a label
    when CSVs disagree.

`source_collection` is synthesized inside this script from the CSV
filename (`metadata_c212.csv` -> `212`). It is NOT added to
load_and_merge_metadata in this commit -- that's Phase 4d's job.

Run from project root:
    python scripts/audit_merge_consistency.py [training_data_dir]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


COLLECTION_RE = re.compile(r"metadata_c(\d+)\.csv$")


def main(training_data_dir: str = "training_data") -> None:
    root = Path(training_data_dir)
    if not root.exists():
        sys.exit(f"[ERROR] {root.resolve()} does not exist")

    csvs = sorted(root.glob("metadata_c*.csv"))
    if not csvs:
        sys.exit(f"[ERROR] no metadata_c*.csv found in {root.resolve()}")

    frames = []
    for p in csvs:
        m = COLLECTION_RE.search(p.name)
        if not m:
            print(f"[WARN] {p.name} does not match metadata_c<digits>.csv -- skipping")
            continue
        cid = m.group(1)
        df = pd.read_csv(p, low_memory=False)
        df["source_collection"] = cid
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"Concatenated {len(frames)} CSV(s) -> {len(merged):,} rows total\n")

    # -------------------------------------------------------------------
    print("CHECK (a): diagnosis_1 vocabulary")
    print("-" * 78)
    vc = merged["diagnosis_1"].value_counts(dropna=False)
    print(vc.to_string())
    print()
    non_standard = {}
    for k, v in vc.items():
        if isinstance(k, str) and k in {"Benign", "Malignant"}:
            continue
        non_standard[k] = int(v)
    if non_standard:
        total_dropped = sum(non_standard.values())
        print(f"  trainer.py:161 will silently drop {total_dropped:,} rows in these classes:")
        for k, v in non_standard.items():
            print(f"    {repr(k):<25} : {v:,}")
    else:
        print("  All diagnosis_1 values are in {Benign, Malignant}. Nothing dropped.")
    print()

    # -------------------------------------------------------------------
    print("CHECK (b): image_type per source_collection")
    print("-" * 78)
    if "image_type" not in merged.columns:
        print("  image_type column absent across the training pool.")
    else:
        pivot = (
            merged.groupby("source_collection")["image_type"]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
        )
        print(pivot.to_string())
        print()
        clinical_mask = merged["image_type"].astype(str).str.startswith("clinical")
        n_clinical = int(clinical_mask.sum())
        if n_clinical > 0:
            print(f"  WARN: {n_clinical:,} rows have image_type starting with 'clinical' --")
            print(f"        modality leak; this is the very thing the c=390 drop targeted.")
            print()
            print("  First 10 offending rows:")
            cols = [c for c in ["isic_id", "source_collection", "image_type", "diagnosis_1"]
                    if c in merged.columns]
            print(merged.loc[clinical_mask, cols].head(10).to_string(index=False))
        else:
            print("  All training-pool rows are dermoscopic (no 'clinical:*' types).")
    print()

    # -------------------------------------------------------------------
    print("CHECK (c): isic_id cross-collection duplicates")
    print("-" * 78)
    dup_mask = merged["isic_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    print(f"  isic_id duplicate rows           : {n_dup_rows:,}")
    if n_dup_rows == 0:
        print("  No cross-collection isic_id duplicates. trainer.py:158 dedup is a no-op.")
    else:
        dups = (
            merged.loc[dup_mask, ["isic_id", "source_collection", "diagnosis_1"]]
            .sort_values(["isic_id", "source_collection"])
            .reset_index(drop=True)
        )
        print()
        print("  First 20 duplicate rows (sorted by isic_id, source_collection):")
        print(dups.head(20).to_string(index=False))

        agree, disagree = 0, 0
        disagreements: list[tuple] = []
        for iid, grp in dups.groupby("isic_id"):
            if grp["diagnosis_1"].nunique(dropna=False) == 1:
                agree += 1
            else:
                disagree += 1
                disagreements.append(
                    (iid,
                     list(zip(grp["source_collection"].tolist(),
                              grp["diagnosis_1"].tolist())))
                )

        n_unique_dups = agree + disagree
        print()
        print(f"  unique isic_ids appearing in >1 CSV : {n_unique_dups}")
        print(f"    -> diagnosis_1 AGREES across CSVs : {agree}")
        print(f"    -> diagnosis_1 DISAGREES          : {disagree}")
        if disagree > 0:
            print()
            print(f"  WARN: {disagree} isic_ids have CONFLICTING diagnosis_1 between collections.")
            print(f"        trainer.py:158 first-write-wins keeps whichever row came first in concat order.")
            print(f"        That label is non-deterministic across pipeline runs unless ordering is pinned.")
            print()
            print(f"  Sample conflicts (first 5):")
            for iid, pairs in disagreements[:5]:
                print(f"    {iid}: {pairs}")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_data")
