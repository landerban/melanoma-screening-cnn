"""
Phase 10 — Comparison figures (fig14-17).

  fig14_pf1_before_after.png    Per-collection balanced AUC: Phase 9 vs Phase 10
  fig15_pf2_before_after.png    Per-shortcut ΔP: Phase 9 vs Phase 10
  fig16_pf3_before_after.png    Cross-collection gap: Phase 9 vs Phase 10
  fig17_fine_tune_trajectory    Fine-tune val_AUC over 5 epochs

fig13 (Grad-CAM Phase 6 vs Phase 10) — separate script (heavier).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR   = REPO_ROOT / "artifacts" / "phase10" / "figures"

# Phase 9 numbers (anchor)
P9 = {
    "auc": {"212": 0.9428, "70": 0.5480, "249": 0.7776},
    "dp":  {"vignette": +0.0070, "ruler": +0.0030, "hair": -0.0662, "colorcast": +0.1039},
    "gap_212_70": 0.434,
    "all_rm_auc": {"212": 0.871, "70": 0.723, "249": 0.706},
    "orig_auc":   {"212": 0.964, "70": 0.530, "249": 0.770},
}

P10 = {
    "auc": {"212": 0.9600, "70": 0.8668, "249": 0.8652},
    "dp":  {"vignette": -0.0081, "ruler": -0.0007, "hair": -0.0089, "colorcast": +0.0144},
    "gap_212_70": 0.0932,
    "all_rm_auc": {"212": 0.910, "70": 0.780, "249": 0.809},
    "orig_auc":   {"212": 0.962, "70": 0.841, "249": 0.850},
}

COLL_LABELS = {"212": "HAM10000\n(c=212)", "70": "SIIM-2020\n(c=70)", "249": "BCN20000\n(c=249)"}


def fig14_pf1():
    """Per-collection balanced AUC: Phase 9 vs Phase 10."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cohorts = ["212", "70", "249"]
    x = np.arange(len(cohorts))
    width = 0.35

    p9_vals = [P9["auc"][c] for c in cohorts]
    p10_vals = [P10["auc"][c] for c in cohorts]

    b1 = ax.bar(x - width/2, p9_vals, width, label="Phase 9 (before intervention)",
                color="#E89D6E", edgecolor="black")
    b2 = ax.bar(x + width/2, p10_vals, width, label="Phase 10 (color-invariant fine-tune)",
                color="#7DA7D9", edgecolor="black")

    for b, v in zip(b1, p9_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    for b, v in zip(b2, p10_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([COLL_LABELS[c] for c in cohorts])
    ax.set_ylabel("ROC-AUC (balanced 50/50)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("PF#1 — Per-collection balanced AUC: Phase 9 vs Phase 10\n"
                 "c=70 (SIIM) recovers from chance (0.548) to excellent (0.867)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate c=70 reversal
    ax.annotate(f"c=70: +0.32 (chance → excellent)",
                xy=(1, 0.87), xytext=(1.3, 0.95),
                fontsize=10, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_pf1_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig14_pf1_before_after.png")


def fig15_pf2():
    """Per-shortcut ΔP: Phase 9 vs Phase 10."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shortcuts = ["vignette", "ruler", "hair", "colorcast"]
    pretty = ["Vignette", "Ruler", "Hair (NC)", "Color cast"]
    x = np.arange(len(shortcuts))
    width = 0.35

    p9_vals = [P9["dp"][s] for s in shortcuts]
    p10_vals = [P10["dp"][s] for s in shortcuts]

    b1 = ax.bar(x - width/2, p9_vals, width, label="Phase 9 (before)",
                color="#E89D6E", edgecolor="black")
    b2 = ax.bar(x + width/2, p10_vals, width, label="Phase 10 (after)",
                color="#7DA7D9", edgecolor="black")

    for b, v in zip(b1, p9_vals):
        y = v + (0.005 if v > 0 else -0.012)
        ax.text(b.get_x() + b.get_width()/2, y, f"{v:+.3f}",
                ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    for b, v in zip(b2, p10_vals):
        y = v + (0.005 if v > 0 else -0.012)
        ax.text(b.get_x() + b.get_width()/2, y, f"{v:+.3f}",
                ha="center", va="bottom" if v > 0 else "top", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pretty)
    ax.set_ylabel("Counterfactual mean ΔP_malig")
    ax.set_title("PF#2 — Per-shortcut causal ΔP: Phase 9 vs Phase 10\n"
                 "Color cast effect drops 86% (+0.104 → +0.014)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_pf2_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig15_pf2_before_after.png")


def fig16_pf3():
    """Cross-collection gap closure: Phase 9 vs Phase 10."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: gap comparison
    ax = axes[0]
    gaps = [P9["gap_212_70"], P10["gap_212_70"]]
    bars = ax.bar(["Phase 9\n(before)", "Phase 10\n(after)"], gaps,
                  color=["#E89D6E", "#7DA7D9"], edgecolor="black")
    for b, v in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-collection gap (c=212 − c=70)")
    ax.set_ylim(0, 0.55)
    ax.set_title(f"Gap (c=212 − c=70): 0.434 → 0.093\n"
                 f"(79% reduction)")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-collection orig vs all-removed (Phase 10)
    ax = axes[1]
    cohorts = ["212", "70", "249"]
    x = np.arange(len(cohorts))
    width = 0.35
    orig = [P10["orig_auc"][c] for c in cohorts]
    rm = [P10["all_rm_auc"][c] for c in cohorts]
    ax.bar(x - width/2, orig, width, label="Original", color="#7DA7D9", edgecolor="black")
    ax.bar(x + width/2, rm, width, label="All shortcuts removed",
           color="#E89D6E", edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([COLL_LABELS[c] for c in cohorts])
    ax.set_ylabel("AUC (Phase 10 model)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Phase 10: smaller gap between original\nand all-removed (color cast 가 less load-bearing)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("PF#3 — Cross-collection gap closure", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_pf3_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig16_pf3_before_after.png")


def fig17_trajectory():
    """Fine-tune val_AUC over 5 epochs."""
    epochs = [0, 1, 2, 3, 4, 5]
    val_aucs = [0.7944, 0.9042, 0.9205, 0.9278, 0.9332, 0.9363]
    test_pre = 0.8017
    test_post = 0.9369

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_aucs, "o-", color="#1f77b4", linewidth=2, markersize=8,
            label="val AUC")
    ax.axhline(test_pre, color="#d62728", linestyle="--", alpha=0.7,
               label=f"Pre-FT test AUC = {test_pre:.4f}")
    ax.axhline(test_post, color="#2ca02c", linestyle="--", alpha=0.7,
               label=f"Post-FT test AUC = {test_post:.4f}")

    # Annotate each point
    for e, v in zip(epochs, val_aucs):
        ax.text(e, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    ax.annotate(f"ΔAUC = +{test_post - test_pre:+.4f}",
                xy=(5, test_post), xytext=(3.5, 0.85),
                fontsize=11, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Fine-tune trajectory — Color-invariant fine-tune (Phase 10)\n"
                 "5 epoch Stage-2 fine-tune on Colab A100, hue jitter only (sat=0, hue=0.10)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.78, 0.96)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_fine_tune_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig17_fine_tune_trajectory.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig14_pf1()
    fig15_pf2()
    fig16_pf3()
    fig17_trajectory()
    print(f"\n4 figures saved to {FIG_DIR.relative_to(REPO_ROOT)}")
    print("fig13 (Grad-CAM Phase 6 vs Phase 10) — run 04_fig13_cam_grid.py separately.")


if __name__ == "__main__":
    main()
