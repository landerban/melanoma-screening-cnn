"""
rebuild_per_collection_metadata.py -- Phase 2b helper.

The three ISIC image downloads (c=212, c=70, c=249) all write into the same
training_data/images/ directory. Each one writes a metadata.csv at the end
of its run, so the three of them race and the last writer wins. That leaves
load_and_merge_metadata with metadata for only ONE collection.

Fix: after the image downloads finish, query isic-cli for each collection's
metadata separately and dump per-collection CSVs at the training_data/ root.
load_and_merge_metadata globs *.csv at one level so it picks them all up.

Also removes the race-loser training_data/images/metadata.csv since the
per-collection CSVs are authoritative.

Run from the project root:
    python scripts/rebuild_per_collection_metadata.py [training_data_dir]

Default training_data_dir = ./training_data
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Three training-pool collections per CLAUDE.md updated plan (TBP c=390 dropped).
COLLECTIONS = {
    212: "HAM10000",
    70:  "SIIM-ISIC-2020",
    249: "BCN20000",
}


def main(training_data_dir: str = "training_data") -> None:
    root = Path(training_data_dir).resolve()
    if not root.exists():
        sys.exit(f"[ERROR] {root} does not exist")

    if shutil.which("isic") is None:
        sys.exit("[ERROR] `isic` CLI not on PATH -- activate the venv first")

    failures: list[int] = []
    total_rows = 0
    for cid, name in COLLECTIONS.items():
        out_csv = root / f"metadata_c{cid}.csv"
        print(f"[{cid:>3}] {name:<16} -> {out_csv.name}")
        result = subprocess.run(
            ["isic", "metadata", "download", "-c", str(cid), "-o", str(out_csv)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not out_csv.exists() or out_csv.stat().st_size == 0:
            print(f"        [FAIL] rc={result.returncode}")
            if result.stderr:
                print(f"        stderr: {result.stderr.strip()[:200]}")
            failures.append(cid)
            continue
        with open(out_csv) as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
        total_rows += n_rows
        print(f"        wrote {n_rows:,} rows ({out_csv.stat().st_size / 1024:.1f} KB)")

    # Remove the race-loser metadata.csv. The per-collection CSVs are now
    # the authoritative source; load_and_merge_metadata will pick them up.
    race_loser = root / "images" / "metadata.csv"
    if race_loser.exists():
        sz = race_loser.stat().st_size
        race_loser.unlink()
        print(f"\nRemoved race-loser {race_loser.relative_to(root.parent)} "
              f"({sz/1024:.1f} KB) -- per-collection CSVs supersede it.")

    print(f"\nTotal metadata rows across collections: {total_rows:,}")
    if failures:
        sys.exit(f"\n[ERROR] {len(failures)} collection(s) failed: {failures}")
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_data")
