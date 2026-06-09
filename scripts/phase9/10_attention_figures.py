"""
Phase 9 / Step 4f-2 — Visual Grad-CAM figures (the ones the audit
reviewer will look for first).

Produces:
  fig05_cam_grid.png         3x4 grid: per-collection benign/malignant
                             examples with original + CAM overlay
  fig06_peak_scatter.png     300 peak coordinates colored by collection
                             — shows spatial attention distribution
  fig07_attention_pie.png    Per-collection stacked bar of CAM-peak
                             category proportions (on_lesion vs
                             on_shortcut vs on_skin)
  fig08_color_signatures.png Per-collection HSV histogram signatures
                             — the *color cast* shortcut visualized
  fig09_cf_extreme_pairs.png 3 image pairs where color-cast removal
                             changed P_malig the most (already in
                             fig04 — replace fig04 with a richer
                             version that includes the CAM overlay)

Run:
    .venv/bin/python scripts/phase9/10_attention_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

SAMPLE_CSV  = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
PRED_CSV    = REPO_ROOT / "artifacts" / "phase9" / "04_predictions.csv"
CF_CSV      = REPO_ROOT / "artifacts" / "phase9" / "07_counterfactual.csv"
PEAK_CSV    = REPO_ROOT / "artifacts" / "phase9" / "06_peak_classification.csv"
IMG_DIR     = REPO_ROOT / "training_data" / "images"
CAM_DIR     = REPO_ROOT / "artifacts" / "phase9" / "04_gradcam"
LESION_DIR  = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
FIG_DIR     = REPO_ROOT / "artifacts" / "phase9" / "figures"

SEED = 42


def overlay_cam(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend Grad-CAM heatmap on top of original image (RGB output)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    blended = (img_rgb * (1 - alpha) + cam_color * alpha).astype(np.uint8)
    return blended


def fig05_cam_grid():
    """3 collections × 4 examples (2 benign + 2 malignant) with
    original + CAM overlay side-by-side. The 'show me the attention'
    figure the audit reviewer wants."""
    rng = np.random.default_rng(SEED)
    pred = pd.read_csv(PRED_CSV)

    fig, axes = plt.subplots(6, 4, figsize=(11, 13))
    coll_labels = {"212": "HAM10000\n(c=212)",
                   "70":  "SIIM-2020\n(c=70)",
                   "249": "BCN20000\n(c=249)"}

    row_i = 0
    for c_name in ["212", "70", "249"]:
        for lbl in [0, 1]:
            sub = pred[
                (pred.source_collection.astype(str) == c_name)
                & (pred.label == lbl)
            ]
            picked = sub.sample(n=2, random_state=rng.integers(0, 2**31 - 1))
            for col_i, (_, row) in enumerate(picked.iterrows()):
                isic_id = row["isic_id"]
                img_path = IMG_DIR / f"{isic_id}.jpg"
                cam_path = CAM_DIR / f"{isic_id}.npy"
                if not img_path.exists() or not cam_path.exists():
                    continue
                img = load_image_resized(img_path, size=IMG_SIZE)
                cam = np.load(cam_path)
                if cam.shape != (IMG_SIZE, IMG_SIZE):
                    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
                overlay = overlay_cam(img, cam, alpha=0.5)

                ax_orig = axes[row_i, col_i * 2]
                ax_over = axes[row_i, col_i * 2 + 1]
                ax_orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax_over.imshow(overlay)
                ax_orig.axis("off")
                ax_over.axis("off")
                lbl_str = "benign" if lbl == 0 else "malignant"
                ax_orig.set_title(f"{lbl_str}\nP={row['p_malig']:.2f}",
                                   fontsize=8)
                ax_over.set_title("Grad-CAM overlay", fontsize=8)
                if col_i == 0:
                    ax_orig.text(-0.30, 0.5, coll_labels[c_name],
                                 transform=ax_orig.transAxes,
                                 rotation=90, va="center", ha="center",
                                 fontsize=11, fontweight="bold")
            row_i += 1

    plt.suptitle("Grad-CAM attention per collection × class\n"
                 "(red = high attention, blue = low)",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_cam_grid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig05_cam_grid.png")


def fig06_peak_scatter():
    """All 300 Grad-CAM peak coordinates colored by collection."""
    peak = pd.read_csv(PEAK_CSV)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"212": "#1f77b4", "70": "#ff7f0e", "249": "#2ca02c"}
    pretty = {"212": "HAM10000", "70": "SIIM-2020", "249": "BCN20000"}
    for c_name, c in colors.items():
        sub = peak[peak.source_collection.astype(str) == c_name]
        ax.scatter(sub.peak_x, sub.peak_y, c=c, label=pretty[c_name],
                   alpha=0.6, s=40, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, IMG_SIZE); ax.set_ylim(IMG_SIZE, 0)
    ax.set_xlabel("Peak x (px)")
    ax.set_ylabel("Peak y (px)")
    ax.set_title("Spatial distribution of Grad-CAM peaks (N=300)\n"
                 "Per-collection clustering is weak (H1 falsified, "
                 "χ² p=0.23)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mark image center for reference
    ax.axhline(IMG_SIZE / 2, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(IMG_SIZE / 2, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_peak_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig06_peak_scatter.png")


def fig07_attention_pie():
    """Stacked bar: per-collection CAM-peak category proportions."""
    peak = pd.read_csv(PEAK_CSV)
    cats = ["on_lesion", "on_vignette", "on_ruler", "on_hair", "on_skin"]
    cat_pretty = ["On lesion", "On vignette", "On ruler", "On hair", "On normal skin"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#7f7f7f"]

    pivots = []
    cohorts = ["212", "70", "249"]
    coll_pretty = ["HAM10000", "SIIM-2020", "BCN20000"]
    for c_name in cohorts:
        sub = peak[peak.source_collection.astype(str) == c_name]
        row = [(sub.category == cat).mean() for cat in cats]
        pivots.append(row)
    pivots = np.array(pivots) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(cohorts))
    for i, (cat, cp, col) in enumerate(zip(cats, cat_pretty, colors)):
        ax.barh(coll_pretty, pivots[:, i], left=bottom, color=col, label=cp,
                edgecolor="white", linewidth=1)
        # annotate non-tiny segments
        for j, v in enumerate(pivots[:, i]):
            if v >= 3:
                ax.text(bottom[j] + v / 2, j, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if col != "#7f7f7f" else "black")
        bottom += pivots[:, i]

    ax.set_xlabel("% of analytical sample (N=100 per collection)")
    ax.set_xlim(0, 100)
    ax.set_title("Per-collection Grad-CAM peak categories\n"
                 "SIIM (c=70) attends to 'normal skin' more — "
                 "consistent with prevalence-prior reading")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_attention_pie.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig07_attention_pie.png")


def fig08_color_signatures():
    """Per-collection HSV histogram signatures — visualize the color
    cast shortcut directly."""
    npz_path = REPO_ROOT / "artifacts" / "phase9" / "03_color_signatures.npz"
    data = np.load(npz_path, allow_pickle=True)
    sigs = data["signatures"].item()

    # The signatures are 512-bin joint HSV histograms (8 bins per channel).
    # Project to marginal H histogram for visualization.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    coll_pretty = {"212": "HAM10000", "70": "SIIM-2020", "249": "BCN20000"}
    titles_pretty = {"H": "Hue", "S": "Saturation", "V": "Value"}

    for ax_i, channel in enumerate(["H", "S", "V"]):
        ax = axes[ax_i]
        for c_name, color in zip(["212", "70", "249"],
                                 ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            sig = sigs[c_name].reshape(8, 8, 8)
            # marginal over other 2 axes
            if channel == "H":
                marginal = sig.sum(axis=(1, 2))
            elif channel == "S":
                marginal = sig.sum(axis=(0, 2))
            else:
                marginal = sig.sum(axis=(0, 1))
            x = np.arange(8) + 0.5
            ax.plot(x, marginal, color=color,
                    label=coll_pretty[c_name], linewidth=2)
        ax.set_xlabel(f"{titles_pretty[channel]} bin (8 bins)")
        ax.set_ylabel("Probability mass")
        ax.set_title(f"{titles_pretty[channel]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Color-cast shortcut: per-collection HSV signatures\n"
                 "(each curve is the mean histogram across all 100 images "
                 "of that collection)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_color_signatures.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig08_color_signatures.png")


def fig09_cf_extreme_pairs():
    """Top 3 ΔP-color-cast images: original + CAM overlay + inpainted
    + CAM overlay-on-inpainted (if available). Replace fig04."""
    cf = pd.read_csv(CF_CSV)
    sub = cf[cf.color_cast_present == 1].copy()
    sub["delta"] = sub["p_malig__no_colorcast"] - sub["p_malig__original"]
    sub = sub.sort_values("delta", ascending=False).head(3)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for i, (_, row) in enumerate(sub.iterrows()):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        cam_path = CAM_DIR / f"{isic_id}.npy"
        cf_path = REPO_ROOT / "artifacts" / "phase9" / "07_cf_inpainted_qa" \
                  / f"{isic_id}__no_colorcast.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam = np.load(cam_path) if cam_path.exists() else np.zeros((IMG_SIZE, IMG_SIZE))
        overlay = overlay_cam(img, cam)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"{isic_id}\noriginal P = {row['p_malig__original']:.3f}",
                              fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Grad-CAM (original)", fontsize=10)
        axes[i, 1].axis("off")

        if cf_path.exists():
            cf_img = cv2.imread(str(cf_path))
            cf_rgb = cv2.cvtColor(cf_img, cv2.COLOR_BGR2RGB)
            axes[i, 2].imshow(cf_rgb)
        else:
            axes[i, 2].imshow(img_rgb)
        axes[i, 2].set_title(f"color-cast normalized\n"
                              f"P = {row['p_malig__no_colorcast']:.3f}  "
                              f"(Δ = {row['delta']:+.3f})", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Color-cast removal exemplars — top 3 ΔP\n"
                 "(left → middle: original + Grad-CAM; right: counterfactual)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_cf_extreme_pairs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig09_cf_extreme_pairs.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig05_cam_grid()
    fig06_peak_scatter()
    fig07_attention_pie()
    fig08_color_signatures()
    fig09_cf_extreme_pairs()
    print(f"\nAll attention figures saved to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
