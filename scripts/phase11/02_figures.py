"""
Phase 11/12 — Publication figures.

  fig10_hsv_decomposition.png  Phase 11 per-channel ΔP forest plot
  fig11_lacn_comparison.png    Phase 12 LACN vs full vs lesion-only
  fig12_combined_color.png     Phase 11+12 combined: channel × spatial
                               grid of ΔP. THE summary figure.

Run:
    .venv/bin/python scripts/phase11/02_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
HSV_CSV  = REPO_ROOT / "artifacts" / "phase11" / "01_hsv_decomposition.csv"
LACN_CSV = REPO_ROOT / "artifacts" / "phase12" / "01_lacn.csv"
FIG_DIR_11 = REPO_ROOT / "artifacts" / "phase11" / "figures"
FIG_DIR_12 = REPO_ROOT / "artifacts" / "phase12" / "figures"

B = 5000
SEED = 42


def boot_paired_ci(deltas, b=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = deltas[idx].mean()
    return float(deltas.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def fig10_hsv():
    df = pd.read_csv(HSV_CSV)
    sub = df[df.color_cast_present == 1]

    spec = [
        ("H (hue)",        "dP_H",   "#d62728"),
        ("S (saturation)", "dP_S",   "#ff7f0e"),
        ("V (value)",      "dP_V",   "#1f77b4"),
        ("HSV (all)",      "dP_HSV", "#2ca02c"),
    ]
    means, los, his = [], [], []
    for _, col, _ in spec:
        m, l, h = boot_paired_ci(sub[col].values)
        means.append(m); los.append(l); his.append(h)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(spec))
    el = np.array(means) - np.array(los)
    eh = np.array(his) - np.array(means)
    for i, (name, _, col) in enumerate(spec):
        ax.errorbar([means[i]], [i], xerr=[[el[i]], [eh[i]]],
                    fmt="o", color="black", capsize=4, markersize=9,
                    markerfacecolor=col, markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in spec])
    ax.set_xlabel("Mean ΔP_malig (P(channel normalized) − P(original))")
    ax.set_title("Phase 11 — HSV channel decomposition of color-cast shortcut\n"
                 "Hue captures 93% of joint effect; saturation and value are null")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    FIG_DIR_11.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR_11 / "fig10_hsv_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig10_hsv_decomposition.png")


def fig11_lacn():
    df = pd.read_csv(LACN_CSV)

    spec = [
        ("full_norm\n(P9 reference)",  "dP_full",   "#1f77b4"),
        ("LACN\n(background only)",    "dP_LACN",   "#2ca02c"),
        ("lesion only",                "dP_lesion", "#d62728"),
    ]
    means, los, his = [], [], []
    for _, col, _ in spec:
        m, l, h = boot_paired_ci(df[col].values)
        means.append(m); los.append(l); his.append(h)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(spec))
    el = np.array(means) - np.array(los)
    eh = np.array(his) - np.array(means)
    for i, (name, _, col) in enumerate(spec):
        ax.errorbar([means[i]], [i], xerr=[[el[i]], [eh[i]]],
                    fmt="o", color="black", capsize=4, markersize=9,
                    markerfacecolor=col, markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in spec])
    ax.set_xlabel("Mean ΔP_malig (P(spatial variant) − P(original))")
    ax.set_title("Phase 12 — Lesion-Aware Color Normalization (LACN)\n"
                 "Background normalization captures 70% of full effect; lesion-only is weak")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    FIG_DIR_12.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR_12 / "fig11_lacn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig11_lacn_comparison.png")


def fig12_combined():
    """Combined channel × spatial decomposition — the summary figure."""
    hsv = pd.read_csv(HSV_CSV)
    sub_hsv = hsv[hsv.color_cast_present == 1]
    h_mean = sub_hsv.dP_H.mean()
    s_mean = sub_hsv.dP_S.mean()
    v_mean = sub_hsv.dP_V.mean()
    hsv_mean = sub_hsv.dP_HSV.mean()

    lacn = pd.read_csv(LACN_CSV)
    full_mean = lacn.dP_full.mean()
    bg_mean = lacn.dP_LACN.mean()
    les_mean = lacn.dP_lesion.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: channel decomposition
    ax = axes[0]
    bars = ax.bar(["H", "S", "V", "joint"], [h_mean, s_mean, v_mean, hsv_mean],
                   color=["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"],
                   edgecolor="black")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (0.005 if h > 0 else -0.012),
                f"{h:+.3f}", ha="center", va="bottom" if h > 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Phase 11 — channel decomposition")
    ax.set_ylabel("Mean ΔP_malig")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: spatial decomposition
    ax = axes[1]
    bars = ax.bar(["full", "LACN\n(bg)", "lesion\nonly"],
                   [full_mean, bg_mean, les_mean],
                   color=["#1f77b4", "#2ca02c", "#d62728"],
                   edgecolor="black")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                f"{h:+.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Phase 12 — spatial decomposition")
    ax.set_ylabel("Mean ΔP_malig")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Color-cast shortcut anatomy: Hue (93% of channel effect) ✕ Background (70% of spatial effect)\n"
        "Together: the shortcut is *background hue* — not lesion colour, not saturation, not value",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR_11 / "fig12_combined_color_anatomy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig12_combined_color_anatomy.png")


def main():
    fig10_hsv()
    fig11_lacn()
    fig12_combined()


if __name__ == "__main__":
    main()
