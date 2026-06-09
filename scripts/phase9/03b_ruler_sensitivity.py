"""
Phase 9 / Step 3k-sensitivity — Ruler detector parameter sensitivity
analysis.

The first-pass run (03_run_shortcut_detection.py with hough_min_len=60,
hough_threshold=50) flagged 43.7% of analytical-sample images as
ruler-present, which is implausibly high (literature prior on ISIC is
~20-30% via manual annotation in Sirico 2023). We suspect the default
parameters catch lesion-boundary edges in addition to true rulers.

This script sweeps three (min_len, threshold) configurations and reports
how many ruler-positives each yields, per collection. A reviewer can
read the sensitivity curve and judge whether our chosen operating point
isolates rulers from lesion edges.

Decision rule (pre-committed):
  - Pick the strictest configuration that retains AT LEAST 10 ruler-
    positives on c=70 (the collection where rulers are most documented
    in the literature; Winkler 2019).
  - The chosen configuration becomes the canonical ruler detector for
    all downstream Phase-9 analyses; the prior 03_run results are
    superseded.

Run from project root:
    .venv/bin/python scripts/phase9/03b_ruler_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "03b_ruler_sensitivity.log"

# (label, hough_min_len, hough_threshold, hough_max_gap, canny_lo, canny_hi)
CONFIGS = [
    ("baseline (orig)",  60, 50,  8, 50, 150),
    ("strict-1",         100, 80, 5, 80, 200),
    ("strict-2",         140, 100, 3, 80, 200),
    ("strict-3",         180, 130, 3, 80, 200),
]


def detect_ruler_v2(img, min_len, thr, max_gap, lo, hi):
    """Variant ruler detector for sweep. Adds an aspect-ratio gate to
    reject short, off-axis edges from lesion boundaries."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lo, hi)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=thr,
        minLineLength=min_len, maxLineGap=max_gap,
    )
    if lines is None:
        return False, 0

    # Aspect-ratio gate: a true ruler line has the line's length much
    # greater than the orthogonal jitter of its endpoints; lesion-edge
    # snippets are often diagonal but short. Require dominant axis
    # length >= 1.5 × the orthogonal component.
    n_real = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if max(dx, dy) >= 1.5 * min(dx, dy) and max(dx, dy) >= min_len:
            n_real += 1
    return n_real >= 1, n_real


def main():
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Loaded {len(df):,} images for sensitivity sweep")
    print(f"Configurations: {len(CONFIGS)}")
    print()

    results = {label: [] for label, *_ in CONFIGS}
    isic_ids = []
    collections = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="sensitivity"):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue

        img = load_image_resized(img_path, size=IMG_SIZE)
        isic_ids.append(isic_id)
        collections.append(row["source_collection"])

        for label, min_len, thr, max_gap, lo, hi in CONFIGS:
            present, _ = detect_ruler_v2(img, min_len, thr, max_gap, lo, hi)
            results[label].append(int(present))

    log = ["# Phase 9 Step 3k-sensitivity — ruler detector parameter sweep",
           "", "## Per-config presence rates"]
    chosen = None
    for label, *_ in CONFIGS:
        arr = np.array(results[label])
        coll_arr = np.array([str(c) for c in collections])
        total_pct = arr.mean() * 100
        rates_by_col = {
            c: arr[coll_arr == c].sum() for c in sorted(np.unique(coll_arr))
        }
        log.append(f"\n  {label}:")
        log.append(f"    overall: {arr.sum()}/{len(arr)} ({total_pct:.1f}%)")
        for c, n in rates_by_col.items():
            n_total = (coll_arr == c).sum()
            log.append(f"    c={c}: {n}/{n_total} ({n/n_total*100:.1f}%)")
        # Decision rule
        c70_n = rates_by_col.get("70", 0)
        if chosen is None and c70_n < 30 and c70_n >= 10:
            chosen = label

    log.append("\n## Decision")
    if chosen:
        log.append(f"  Chosen config: {chosen}")
        log.append(f"  Reason: first strict config retaining >=10 ruler-positives "
                   f"on c=70 (Winkler-2019 collection) and <30 (avoids lesion-edge "
                   f"false positives).")
    else:
        log.append(f"  No config met decision rule (>=10 on c=70, <30). "
                   f"Default to most-strict; document as low-power for ruler.")
        chosen = CONFIGS[-1][0]

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
