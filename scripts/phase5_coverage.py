"""
phase5_coverage.py -- Phase 5 / A1 dry-run.

Calls `load_and_merge_metadata` and verifies:
  * total rows returned
  * rows per source_collection
  * rows with non-null image_path (final return)
  * rows with null image_path (silent-drop count -- metadata rows whose image
    isn't on disk)

Cross-references against audit numbers (c=212: 11,720, c=70: 33,126,
c=249: 18,946; post-dedup pre-image-filter: 61,396).

Threshold: >50 missing image_path rows -> STOP, re-download required
before any retrain run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer import load_and_merge_metadata  # noqa: E402


COLLECTION_RE = re.compile(r"metadata_c(\d+)\.csv$")
EXPECTED_PRE_IMG = 61_396  # from artifacts/diagnostic_merge_consistency.log


def _replicate_until_image_filter(training_data_dir: str) -> pd.DataFrame:
    """
    Replicates load_and_merge_metadata up to but NOT including the image_path
    filter, so we can count rows that would be dropped because the image isn't
    on disk.
    """
    root = Path(training_data_dir)
    csvs = list(root.glob("*.csv")) + list(root.glob("*/*.csv"))
    frames = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        m = COLLECTION_RE.search(p.name)
        df["source_collection"] = m.group(1) if m else None
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["isic_id", "patient_id", "lesion_id", "diagnosis_1",
                        "source_collection"] if c in combined.columns]
    combined = combined[keep].copy()
    if "isic_id" in combined.columns:
        combined.drop_duplicates(subset="isic_id", inplace=True)
    combined = combined[combined["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)
    return combined


def main() -> int:
    print(f"Phase 5 / A1 -- dataset coverage verification\n")

    # Pre-image-filter cohort (what load_and_merge_metadata sees before
    # dropping rows whose images aren't on disk)
    pre = _replicate_until_image_filter("training_data")
    print(f"Pre-image-filter rows: {len(pre):,}")
    print(f"  by source: {pre['source_collection'].value_counts(dropna=False).to_dict()}")
    if len(pre) != EXPECTED_PRE_IMG:
        print(f"  [WARN] expected {EXPECTED_PRE_IMG:,} from audit, got {len(pre):,} "
              f"(delta {len(pre) - EXPECTED_PRE_IMG:+d})")
    print()

    # Post-image-filter cohort (what load_and_merge_metadata actually returns)
    df = load_and_merge_metadata("training_data", "training_data/images")
    print(f"load_and_merge_metadata returned: {len(df):,}")
    print(f"  by source: {df['source_collection'].value_counts(dropna=False).to_dict()}")
    print()

    # Image_path coverage
    n_with_path = int(df['image_path'].notna().sum())
    print(f"Rows with image_path (non-null): {n_with_path:,}")
    n_missing = len(pre) - len(df)
    print(f"Rows dropped (image not on disk): {n_missing:,}")
    if n_missing > 0:
        # Show which collections lost rows
        pre_counts = pre['source_collection'].value_counts().to_dict()
        post_counts = df['source_collection'].value_counts().to_dict()
        dropped_per_src = {k: pre_counts.get(k, 0) - post_counts.get(k, 0)
                           for k in pre_counts}
        dropped_per_src = {k: v for k, v in dropped_per_src.items() if v > 0}
        print(f"  by source: {dropped_per_src}")

    print()
    print(f"effective_patient_id NaN rows: {int(df['effective_patient_id'].isna().sum()):,}")
    print(f"label distribution: {df['label'].value_counts().to_dict()}")
    print()

    # Threshold check
    print("=" * 60)
    if n_missing > 50:
        print(f"  STOP: {n_missing:,} missing files > 50. Re-download required.")
        return 1
    elif n_missing > 0:
        print(f"  OK (tolerance): {n_missing:,} missing files (within <=50 threshold).")
    else:
        print(f"  CLEAN: all {len(pre):,} expected rows present on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
