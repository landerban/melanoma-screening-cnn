"""
Phase 9 / Step 3o — Grad-CAM peak region classification.

For each image, compute the Grad-CAM peak (top-5% mass centroid) and
classify which region it falls into:

  - on_lesion        (peak inside lesion mask)
  - on_vignette      (peak inside vignette mask AND vignette present)
  - on_ruler         (peak inside ruler mask AND ruler present)
  - on_hair          (peak inside hair mask AND hair present)
  - on_skin          (none of the above)

Why top-5% mass centroid instead of argmax?
  CAMs are noisy at the pixel level. The argmax can sit on a single
  high-activation pixel inside a generally low-activation region. The
  top-5%-mass centroid is the median location of the model's strongest
  attention; it's robust to single-pixel noise and corresponds to a
  meaningful spatial region rather than a point.

Output:
  artifacts/phase9/06_peak_classification.csv
    Per-image peak classification + IoU with each region.

  artifacts/phase9/06_peak_classification.log
    Per-collection breakdown — direct H1 data.

Run:
    .venv/bin/python scripts/phase9/06_peak_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
CAM_DIR    = REPO_ROOT / "artifacts" / "phase9" / "04_gradcam"
SHORTCUT_MASK_DIR = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_masks"
LESION_MASK_DIR   = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
SHORTCUT_CSV = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.csv"
OUT_CSV    = REPO_ROOT / "artifacts" / "phase9" / "06_peak_classification.csv"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "06_peak_classification.log"


def top_mass_centroid(cam: np.ndarray, top_frac: float = 0.05) -> tuple[int, int]:
    """
    Centroid of the top-fraction-mass pixels in a normalized CAM.

    Why this metric: argmax is noisy at pixel scale; mean-of-CAM
    over-weights low-activation tails. Top-5% mass is a robust
    "where the model looks strongest" estimator.
    """
    threshold = np.quantile(cam, 1 - top_frac)
    mask = cam >= threshold
    if not mask.any():
        return cam.shape[0] // 2, cam.shape[1] // 2
    ys, xs = np.where(mask)
    return int(ys.mean()), int(xs.mean())


def peak_iou_with_mask(peak_y: int, peak_x: int, mask: np.ndarray,
                      neighborhood_px: int = 20) -> float:
    """
    Soft IoU: fraction of a (neighborhood × neighborhood) box around
    the peak that overlaps the mask. neighborhood=20 covers ~5% of a
    384×384 image radius, matching the CAM's intrinsic resolution
    (12×12 → 32px per cell).
    """
    h, w = mask.shape
    y0 = max(0, peak_y - neighborhood_px)
    y1 = min(h, peak_y + neighborhood_px + 1)
    x0 = max(0, peak_x - neighborhood_px)
    x1 = min(w, peak_x + neighborhood_px + 1)
    box = mask[y0:y1, x0:x1]
    if box.size == 0:
        return 0.0
    return float((box > 0).sum() / box.size)


def main():
    df = pd.read_csv(SAMPLE_CSV)
    shortcuts_df = pd.read_csv(SHORTCUT_CSV).set_index("isic_id")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="classify"):
        isic_id = row["isic_id"]

        cam_path = CAM_DIR / f"{isic_id}.npy"
        if not cam_path.exists():
            continue
        cam = np.load(cam_path)
        if cam.shape != (384, 384):
            # The CAM helper returns shape matching input; defensive
            cam = cv2.resize(cam, (384, 384), interpolation=cv2.INTER_LINEAR)

        py, px = top_mass_centroid(cam, top_frac=0.05)

        # Load region masks (each may not exist for a given image)
        def load_mask(name: str):
            p = name == "lesion" \
                and (LESION_MASK_DIR / f"{isic_id}.png") \
                or (SHORTCUT_MASK_DIR / f"{isic_id}__{name}.png")
            if not p.exists():
                return None
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if m.shape != (384, 384):
                m = cv2.resize(m, (384, 384), interpolation=cv2.INTER_NEAREST)
            return m

        lesion = load_mask("lesion")
        vignette = load_mask("vignette")
        ruler = load_mask("ruler")
        hair = load_mask("hair")

        # IoUs
        iou_lesion = peak_iou_with_mask(py, px, lesion) if lesion is not None else 0.0
        iou_vignette = peak_iou_with_mask(py, px, vignette) if vignette is not None else 0.0
        iou_ruler = peak_iou_with_mask(py, px, ruler) if ruler is not None else 0.0
        iou_hair = peak_iou_with_mask(py, px, hair) if hair is not None else 0.0

        # Classification priority order:
        #   1. on_lesion if iou_lesion >= 0.5 (peak well inside lesion)
        #   2. else, highest-IoU shortcut region if >= 0.3
        #   3. else on_skin
        # The 0.5 / 0.3 thresholds were calibrated by inspection: 0.5
        # means >half the peak neighborhood overlaps the mask
        # (high-confidence on-lesion); 0.3 for shortcuts is more
        # lenient since shortcut masks are smaller.
        category = "on_skin"
        if iou_lesion >= 0.5:
            category = "on_lesion"
        else:
            shortcut_ious = {
                "on_vignette": iou_vignette,
                "on_ruler":    iou_ruler,
                "on_hair":     iou_hair,
            }
            best_sc, best_iou = max(shortcut_ious.items(), key=lambda x: x[1])
            if best_iou >= 0.3:
                category = best_sc

        # Get shortcut presence from the detection step (filter the
        # IoU to only count if the shortcut was actually flagged)
        sc_flags = shortcuts_df.loc[isic_id]
        if category == "on_vignette" and not sc_flags["vignette_present"]:
            category = "on_skin"
        if category == "on_ruler" and not sc_flags["ruler_present"]:
            category = "on_skin"
        if category == "on_hair" and not sc_flags["hair_present"]:
            category = "on_skin"

        rows.append({
            "isic_id":           isic_id,
            "source_collection": row["source_collection"],
            "label":             row["label"],
            "peak_y":            py,
            "peak_x":            px,
            "iou_lesion":        round(iou_lesion, 4),
            "iou_vignette":      round(iou_vignette, 4),
            "iou_ruler":         round(iou_ruler, 4),
            "iou_hair":          round(iou_hair, 4),
            "category":          category,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    log = [
        "# Phase 9 Step 3o — Grad-CAM peak region classification",
        f"Sample N = {len(out_df)}",
        "",
        "## Overall peak distribution",
    ]
    cats = ["on_lesion", "on_vignette", "on_ruler", "on_hair", "on_skin"]
    for cat in cats:
        n = int((out_df.category == cat).sum())
        log.append(f"  {cat:<14}: {n:>3}/{len(out_df)} ({n/len(out_df):.1%})")
    log.append("")

    log.append("## Per-collection peak distribution (H1 direct data)")
    for c in sorted(out_df.source_collection.astype(str).unique()):
        sub = out_df[out_df.source_collection.astype(str) == c]
        log.append(f"\n  c={c}: (N={len(sub)})")
        for cat in cats:
            n = int((sub.category == cat).sum())
            log.append(f"    {cat:<14}: {n:>3} ({n/len(sub):.1%})")
    log.append("")

    log.append("## Per-class peak distribution")
    for lbl in [0, 1]:
        sub = out_df[out_df.label == lbl]
        lbl_str = "benign" if lbl == 0 else "malignant"
        log.append(f"\n  {lbl_str} (N={len(sub)}):")
        for cat in cats:
            n = int((sub.category == cat).sum())
            log.append(f"    {cat:<14}: {n:>3} ({n/len(sub):.1%})")
    log.append("")

    log.append("## H1 anchor: shortcut-attention rate per collection")
    log.append("  (any non-on_lesion non-on_skin peak counts as shortcut)")
    for c in sorted(out_df.source_collection.astype(str).unique()):
        sub = out_df[out_df.source_collection.astype(str) == c]
        n_short = int(sub.category.isin(["on_vignette", "on_ruler", "on_hair"]).sum())
        log.append(f"  c={c}: shortcut-attention = {n_short}/{len(sub)} "
                   f"({n_short/len(sub):.1%})")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
