"""
Phase 9 / Step 3j-n — Run all shortcut detectors on the analytical
sample and produce:

  artifacts/phase9/03_shortcut_detection.csv
    one row per image; columns include presence flags + scores for
    each shortcut.

  artifacts/phase9/03_shortcut_masks/<isic_id>__<shortcut>.png
    binary mask per (image, shortcut) for downstream inpainting.

  artifacts/phase9/03_color_signatures.npz
    per-collection mean histogram + per-image membership matrix.

Run from project root (in venv):
    .venv/bin/python scripts/phase9/03_run_shortcut_detection.py
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

from shortcut_detect import (  # noqa: E402
    detect_vignette, detect_ruler, detect_hair,
    compute_color_histogram, histogram_intersection,
    load_image_resized, IMG_SIZE,
)

SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
OUT_CSV    = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.csv"
MASK_DIR   = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_masks"
COLOR_NPZ  = REPO_ROOT / "artifacts" / "phase9" / "03_color_signatures.npz"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.log"


def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SAMPLE_CSV)
    print(f"Loaded {len(df):,} target images from {SAMPLE_CSV.name}")
    print(f"Detection at {IMG_SIZE}x{IMG_SIZE} (model input geometry)")
    print()

    rows = []
    histograms = []  # parallel to rows, for color-signature aggregation

    missing = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="detect"):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            missing.append(isic_id)
            continue

        img = load_image_resized(img_path, size=IMG_SIZE)

        # Detectors
        v_present, v_mask, v_score = detect_vignette(img)
        r_present, r_mask, r_score = detect_ruler(img)
        h_present, h_mask, h_score = detect_hair(img)
        hist = compute_color_histogram(img)
        histograms.append(hist)

        # Save masks (lossless PNG, single-channel)
        if v_present:
            cv2.imwrite(str(MASK_DIR / f"{isic_id}__vignette.png"), v_mask)
        if r_present:
            cv2.imwrite(str(MASK_DIR / f"{isic_id}__ruler.png"), r_mask)
        if h_present:
            cv2.imwrite(str(MASK_DIR / f"{isic_id}__hair.png"), h_mask)

        rows.append({
            "isic_id":              isic_id,
            "source_collection":    row["source_collection"],
            "label":                row["label"],
            "vignette_present":     int(v_present),
            "vignette_score":       round(v_score, 4),
            "ruler_present":        int(r_present),
            "ruler_score":          round(r_score, 4),
            "hair_present":         int(h_present),
            "hair_score":           round(h_score, 4),
        })

    out_df = pd.DataFrame(rows)

    # --- Color-cast signatures ---
    H = np.stack(histograms, axis=0)  # (N, 512)
    isic_ids = out_df["isic_id"].values
    collections = out_df["source_collection"].values.astype(str)

    sig_per_collection = {}
    for c in sorted(set(collections)):
        mask = collections == c
        sig_per_collection[c] = H[mask].mean(axis=0)

    # Per-image membership to each collection signature (3 columns)
    membership = np.zeros((len(H), len(sig_per_collection)), dtype=np.float32)
    coll_keys = sorted(sig_per_collection.keys())
    for j, c in enumerate(coll_keys):
        for i in range(len(H)):
            membership[i, j] = histogram_intersection(H[i], sig_per_collection[c])

    # Color-cast presence: the image's strongest membership is to its
    # *own* collection's signature, BUT we only flag it when the
    # within-collection membership exceeds the next-best by a margin.
    # Margin = 0.05 (calibrated by inspection of the membership
    # distribution; entries within 0.05 of one another are
    # collection-ambiguous).
    home_col_idx = np.array([coll_keys.index(str(c)) for c in collections])
    home_score = membership[np.arange(len(H)), home_col_idx]
    other_max = np.array([
        np.max(np.delete(membership[i], home_col_idx[i]))
        for i in range(len(H))
    ])
    color_cast_present = (home_score - other_max > 0.05).astype(int)

    out_df["color_cast_present"] = color_cast_present
    out_df["color_cast_home_score"] = home_score.round(4)
    out_df["color_cast_margin"] = (home_score - other_max).round(4)

    # Save outputs
    out_df.to_csv(OUT_CSV, index=False)
    np.savez(COLOR_NPZ,
             signatures={k: v for k, v in sig_per_collection.items()},
             per_image_membership=membership,
             coll_keys=np.array(coll_keys),
             isic_ids=isic_ids)

    # --- Summary ---
    log_lines = [
        f"# Phase 9 Step 3j-n — shortcut detection",
        f"Processed: {len(out_df):,} / {len(df):,} images",
        f"Missing on disk: {len(missing)}",
        "",
        f"## Per-shortcut prevalence in analytical sample",
    ]
    if missing:
        log_lines.append(f"  Missing isic_ids (first 10): {missing[:10]}")

    for sc in ["vignette", "ruler", "hair", "color_cast"]:
        col = f"{sc}_present"
        if col not in out_df.columns:
            continue
        n_pos = int(out_df[col].sum())
        log_lines.append(f"  {sc:<12} present: {n_pos:>3}/{len(out_df)} "
                         f"({n_pos / len(out_df):.1%})")

    log_lines.append("")
    log_lines.append("## Per-shortcut prevalence by collection")
    for sc in ["vignette", "ruler", "hair", "color_cast"]:
        col = f"{sc}_present"
        if col not in out_df.columns:
            continue
        log_lines.append(f"\n  {sc}:")
        for c in sorted(out_df.source_collection.unique()):
            sub = out_df[out_df.source_collection == c]
            n_pos = int(sub[col].sum())
            log_lines.append(f"    c={c}: {n_pos}/{len(sub)} "
                             f"({n_pos / len(sub):.1%})")

    log_lines.append("")
    log_lines.append("## By class (within analytical sample)")
    for sc in ["vignette", "ruler", "hair", "color_cast"]:
        col = f"{sc}_present"
        if col not in out_df.columns:
            continue
        log_lines.append(f"\n  {sc}:")
        for lbl in [0, 1]:
            sub = out_df[out_df.label == lbl]
            n_pos = int(sub[col].sum())
            lbl_str = "benign" if lbl == 0 else "malignant"
            log_lines.append(f"    {lbl_str}: {n_pos}/{len(sub)} "
                             f"({n_pos / len(sub):.1%})")

    log_lines.append("")
    log_lines.append("## Files emitted")
    log_lines.append(f"  CSV:           {OUT_CSV.relative_to(REPO_ROOT)}")
    log_lines.append(f"  Masks dir:     {MASK_DIR.relative_to(REPO_ROOT)} "
                     f"({len(list(MASK_DIR.glob('*.png')))} masks)")
    log_lines.append(f"  Color sigs:    {COLOR_NPZ.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log_lines) + "\n")
    print("\n" + "\n".join(log_lines[-30:]))


if __name__ == "__main__":
    main()
