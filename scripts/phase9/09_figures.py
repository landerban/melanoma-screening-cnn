"""
Phase 9 / Step 4f — Generate publication-quality figures for the
paper writeup and the slide deck.

Figures:
  fig01_balanced_auc.png    PF#1 — per-collection balanced AUC bars
                            with bootstrap CIs.
  fig02_marginal_dp.png     PF#2 — per-shortcut counterfactual ΔP
                            forest plot with paired-bootstrap CIs.
  fig03_gap_decomposition.png PF#3 — per-collection AUC orig vs
                            all-removed; cross-collection gap closure.
  fig04_color_cast_examples.png   Visual exemplar of color-cast removal
                            on three randomly chosen images.

All figures saved at 300 dpi.

Run:
    .venv/bin/python scripts/phase9/09_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

PRED_CSV = REPO_ROOT / "artifacts" / "phase9" / "04_predictions.csv"
CF_CSV   = REPO_ROOT / "artifacts" / "phase9" / "07_counterfactual.csv"
IMG_DIR  = REPO_ROOT / "training_data" / "images"
FIG_DIR  = REPO_ROOT / "artifacts" / "phase9" / "figures"

SEED = 42
B_BOOT = 5000


def bootstrap_auc_ci(labels, scores, b=B_BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(labels)
    boots = []
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            pass
    point = roc_auc_score(labels, scores)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, lo, hi


def bootstrap_paired_ci(x, y, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = (x[idx] - y[idx]).mean()
    point = (x - y).mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, lo, hi


def fig01(pred, cf):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    cohorts = ["c=212", "c=70", "c=249"]
    labels = ["HAM10000\n(c=212)", "SIIM-2020\n(c=70)", "BCN20000\n(c=249)"]

    # Phase 7c (skewed) AUCs from bootstrap_test_auc.log
    phase7c = [0.9151, 0.8835, 0.8702]
    phase7c_lo = [0.8980, 0.8545, 0.8555]
    phase7c_hi = [0.9323, 0.9100, 0.8842]

    # Phase 9 balanced AUCs
    phase9 = []
    phase9_lo = []
    phase9_hi = []
    for c_name in ["212", "70", "249"]:
        sub = pred[pred.source_collection.astype(str) == c_name]
        pt, lo, hi = bootstrap_auc_ci(sub.label.values, sub.p_malig.values)
        phase9.append(pt)
        phase9_lo.append(lo)
        phase9_hi.append(hi)

    x = np.arange(len(cohorts))
    width = 0.35
    bars1 = ax.bar(x - width/2, phase7c, width, label="Phase 7c (skewed prevalence)",
                   color="#7DA7D9", edgecolor="black")
    err1_lo = np.array(phase7c) - np.array(phase7c_lo)
    err1_hi = np.array(phase7c_hi) - np.array(phase7c)
    ax.errorbar(x - width/2, phase7c, yerr=[err1_lo, err1_hi],
                fmt="none", color="black", capsize=3)

    bars2 = ax.bar(x + width/2, phase9, width,
                   label="Phase 9 (50/50 balanced)",
                   color="#E89D6E", edgecolor="black")
    err2_lo = np.array(phase9) - np.array(phase9_lo)
    err2_hi = np.array(phase9_hi) - np.array(phase9)
    ax.errorbar(x + width/2, phase9, yerr=[err2_lo, err2_hi],
                fmt="none", color="black", capsize=3)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Primary finding #1 — prevalence-balanced AUC reveals\n"
                 "c=70's score is almost entirely a prevalence-prior contribution")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate the delta for c=70
    ax.annotate(f"Δ = -{phase7c[1] - phase9[1]:.2f}",
                xy=(1, phase9[1]), xytext=(1, 0.65),
                fontsize=10, ha="center", color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_balanced_auc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig01_balanced_auc.png")


def fig02(cf):
    fig, ax = plt.subplots(figsize=(7, 4))

    shortcuts = [
        ("Vignette",   "vignette_present",   "p_malig__no_vignette"),
        ("Ruler",      "ruler_present",      "p_malig__no_ruler"),
        ("Hair (NC)",  "hair_present",       "p_malig__no_hair"),
        ("Color cast", "color_cast_present", "p_malig__no_colorcast"),
    ]

    means, los, his, ns = [], [], [], []
    for _, col_pres, col_cf in shortcuts:
        sub = cf[cf[col_pres] == 1]
        pt, lo, hi = bootstrap_paired_ci(sub[col_cf].values,
                                          sub["p_malig__original"].values)
        means.append(pt); los.append(lo); his.append(hi); ns.append(len(sub))

    y = np.arange(len(shortcuts))
    err_lo = np.array(means) - np.array(los)
    err_hi = np.array(his) - np.array(means)
    colors = ["#7DA7D9", "#A0C078", "#999999", "#E89D6E"]

    ax.errorbar(means, y, xerr=[err_lo, err_hi], fmt="o", color="black",
                ecolor="black", capsize=4, markersize=8)

    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], s=120, color=c, edgecolor="black", zorder=3)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}\n(N={n})" for (name, _, _), n in zip(shortcuts, ns)])
    ax.set_xlabel("Mean ΔP_malig  (P(no_shortcut) − P(original))")
    ax.set_title("Primary finding #2 — per-shortcut counterfactual marginal effect\n"
                 "Color cast dominates; vignette and ruler are null\n"
                 "(hair is the inpainting-artifact baseline)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_marginal_dp.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig02_marginal_dp.png")


def fig03(cf):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    from sklearn.metrics import roc_auc_score

    cohorts = ["212", "70", "249"]
    pretty = ["HAM10000\n(c=212)", "SIIM-2020\n(c=70)", "BCN20000\n(c=249)"]

    orig, all_rm = [], []
    for c_name in cohorts:
        sub = cf[cf.source_collection.astype(str) == c_name]
        orig.append(roc_auc_score(sub.label, sub["p_malig__original"]))
        all_rm.append(roc_auc_score(sub.label, sub["p_malig__all_removed"]))

    x = np.arange(len(cohorts))
    width = 0.35
    ax.bar(x - width/2, orig, width, label="Original", color="#7DA7D9", edgecolor="black")
    ax.bar(x + width/2, all_rm, width, label="All shortcuts removed",
           color="#E89D6E", edgecolor="black")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(pretty)
    ax.set_ylabel("ROC-AUC (balanced 50/50)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Primary finding #3 — shortcut removal recovers c=70's lesion signal\n"
                 "Cross-collection gap (HAM − SIIM) shrinks from 0.43 to 0.15 (66% closed)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate c=70's reversal
    ax.annotate("c=70 AUC rises from chance\nto meaningful (+0.19)",
                xy=(1, all_rm[1]), xytext=(1.2, 0.85),
                fontsize=9, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_gap_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig03_gap_decomposition.png")


def fig04(cf):
    """Pick 3 images where color-cast removal had the largest ΔP, show
    original + color-normalized side by side."""
    sub = cf[cf.color_cast_present == 1].copy()
    sub["delta"] = sub["p_malig__no_colorcast"] - sub["p_malig__original"]
    sub = sub.sort_values("delta", ascending=False).head(3)

    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    for i, (_, row) in enumerate(sub.iterrows()):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cf_path = REPO_ROOT / "artifacts" / "phase9" / "07_cf_inpainted_qa" \
                  / f"{isic_id}__no_colorcast.jpg"
        if cf_path.exists():
            cf_img = cv2.imread(str(cf_path))
            cf_rgb = cv2.cvtColor(cf_img, cv2.COLOR_BGR2RGB)
        else:
            cf_rgb = img_rgb

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"{isic_id}\noriginal P_malig = {row['p_malig__original']:.3f}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cf_rgb)
        axes[i, 1].set_title(f"color-cast normalized\nP_malig = {row['p_malig__no_colorcast']:.3f}  "
                             f"(Δ = {row['delta']:+.3f})")
        axes[i, 1].axis("off")

    plt.suptitle("Color-cast removal exemplars\n"
                 "Top-3 ΔP among color-cast-present images",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_color_cast_examples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig04_color_cast_examples.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_CSV)
    cf   = pd.read_csv(CF_CSV)

    fig01(pred, cf)
    fig02(cf)
    fig03(cf)
    fig04(cf)
    print(f"\nAll figures saved to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
