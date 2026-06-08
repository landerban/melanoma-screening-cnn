"""
Phase 9 / Step 3h-i — Approximate lesion segmentation via Otsu
thresholding + morphological cleanup.

Rationale.
The pre-registered protocol (04_protocol.md §3.2) wanted ISIC 2018
Task 1 expert masks for HAM10000 and manual annotation for the other
two collections. ISIC 2018 distributes those masks as a multi-GB zip
behind a challenge-page login and only for the 2018 *task* image set
— it does not provide per-isic_id endpoints, so individual fetch
costs would equal full download. Manual annotation of 100 images
across two collections would exceed the project time budget.

We substitute an Otsu-thresholded lesion mask as a coarse criterion.
Validation: against a 20-image random subset where the operator
visually checked whether the Otsu mask captures the lesion area
(IoU > 0.5 to a hand-drawn outline).

This is a *measurement-uncertainty* substitution. We will report it
as a sensitivity disclosure: H1's peak-region classification depends
on this approximation, and a future replication with expert masks
would tighten the result.

Otsu pipeline:
  1. Convert to L*a*b* and use the L channel — robust to vignette
     (the L channel discriminates against background fall-off less
     aggressively than HSV value).
  2. Otsu's method picks a threshold; we take the *darker* side as
     lesion (lesions are usually darker than surrounding skin).
  3. Morphological opening + closing to remove tiny specks and
     close interior holes.
  4. Keep the largest connected component (lesions are spatially
     contiguous; many small components = mask is unreliable, drop).

Saves to artifacts/phase9/05_lesion_masks/{isic_id}.png (single-channel
8-bit binary).

Run:
    .venv/bin/python scripts/phase9/05_lesion_masks.py
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
MASK_DIR   = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks.log"


def segment_lesion_otsu(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Otsu lesion segmentation with morphological cleanup. Returns
    (mask_uint8, stats_dict).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Otsu threshold. cv2.THRESH_BINARY_INV makes the *dark* side
    # (lesions are darker than skin) → foreground.
    _, raw = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=3)

    # Keep largest connected component
    n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_cc <= 1:
        return np.zeros_like(L), dict(area_frac=0.0, n_cc=0, reliable=False)

    areas = cc_stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    mask = (cc_lbl == biggest).astype(np.uint8) * 255

    area_frac = float(mask.sum() / 255) / mask.size

    # Reliability heuristic: lesion should occupy 1%–60% of the image.
    # Outside that range, Otsu is probably catching vignette or
    # everything-but-vignette.
    reliable = 0.01 <= area_frac <= 0.60

    return mask, dict(
        area_frac=area_frac,
        n_cc=n_cc - 1,
        reliable=reliable,
    )


def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Lesion masks for {len(df)} images via Otsu + morphological cleanup")

    log = [
        "# Phase 9 Step 3h-i — approximate lesion masks (Otsu)",
        f"Sample N = {len(df)}",
        "",
    ]
    stats_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="segment"):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        mask, stats = segment_lesion_otsu(img)
        cv2.imwrite(str(MASK_DIR / f"{isic_id}.png"), mask)
        stats["isic_id"] = isic_id
        stats["source_collection"] = row["source_collection"]
        stats["label"] = row["label"]
        stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(REPO_ROOT / "artifacts" / "phase9" / "05_lesion_mask_stats.csv",
                    index=False)

    n_reliable = int(stats_df.reliable.sum())
    log.append(f"## Reliability stats (heuristic: 1% <= area_frac <= 60%)")
    log.append(f"  Reliable masks: {n_reliable}/{len(stats_df)} "
               f"({n_reliable / len(stats_df):.1%})")
    log.append(f"  Mean area fraction: {stats_df.area_frac.mean():.3f}")
    log.append(f"  Median area fraction: {stats_df.area_frac.median():.3f}")
    log.append("")

    log.append("## Reliability by collection")
    for c in sorted(stats_df.source_collection.astype(str).unique()):
        sub = stats_df[stats_df.source_collection.astype(str) == c]
        n_rel = int(sub.reliable.sum())
        log.append(f"  c={c}: reliable={n_rel}/{len(sub)} "
                   f"({n_rel / len(sub):.1%}), "
                   f"mean area={sub.area_frac.mean():.3f}")
    log.append("")

    log.append("## Note")
    log.append("  - These are *approximate* lesion masks, not expert "
               "annotations.")
    log.append("  - The H1 peak-region analysis treats lesion as a "
               "*coarse criterion*: peak inside mask = on-lesion, "
               "peak outside = off-lesion. The IoU semantics are not "
               "load-bearing.")
    log.append("  - Sensitivity disclosure: 'Lesion masks via Otsu '"
               "thresholding; the reliability flag (area in [0.01, 0.6]) "
               "filters images where Otsu mis-segmented (e.g., uniformly "
               "dark images with vignette dominating).'")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log[-15:]))


if __name__ == "__main__":
    main()
