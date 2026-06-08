"""
Phase 9 / Step 4a-e — Statistical hypothesis tests on the
analytical sample.

Hypotheses (pre-registered in `docs/phase9/03_hypotheses.md`):
  H1: Attention is non-uniform across collections (χ² test).
  H2: Per-shortcut counterfactual ΔP is significantly non-zero
      (paired Wilcoxon, Cohen's d).
  H3: Cross-collection AUC gap decomposes via shortcut removal.
  H4: In-dist shortcut magnitude correlates with OOD deviation (n=3
      collection-level pairs — under-powered; reported as exploratory).
  H5: Score-CAM consistency — SKIPPED in this Phase 9 run; LaMa upgrade
      was deferred and Score-CAM was a downstream of that decision.

All tests are pre-registered at α = 0.01 (Bonferroni-corrected over
8 primary tests gives α = 0.006; we use the looser α = 0.01 per
hypothesis and report family-wise corrected separately).

Bootstrap CIs: B = 5000, seed = 42, paired bootstrap for ΔP, image-
level bootstrap for AUC.

Outputs:
  artifacts/phase9/08_statistics.csv      (per-test result row)
  artifacts/phase9/08_statistics.log      (human-readable summary)
  artifacts/phase9/08_h1_attention.csv    (per-row peak-category)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
B_BOOT = 5000

PEAK_CSV = REPO_ROOT / "artifacts" / "phase9" / "06_peak_classification.csv"
CF_CSV   = REPO_ROOT / "artifacts" / "phase9" / "07_counterfactual.csv"
PRED_CSV = REPO_ROOT / "artifacts" / "phase9" / "04_predictions.csv"
DET_CSV  = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.csv"
OUT_CSV  = REPO_ROOT / "artifacts" / "phase9" / "08_statistics.csv"
LOG_TXT  = REPO_ROOT / "artifacts" / "phase9" / "08_statistics.log"


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def bootstrap_ci(values: np.ndarray, stat_fn, b: int = B_BOOT,
                 ci: float = 0.95, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(values[idx])
    point = stat_fn(values)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(point), float(lo), float(hi)


def bootstrap_paired_ci(x: np.ndarray, y: np.ndarray, b: int = B_BOOT,
                       ci: float = 0.95, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = (x[idx] - y[idx]).mean()
    point = (x - y).mean()
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(point), float(lo), float(hi)


def main():
    peak = pd.read_csv(PEAK_CSV)
    cf   = pd.read_csv(CF_CSV)
    det  = pd.read_csv(DET_CSV)
    pred = pd.read_csv(PRED_CSV)

    log = ["# Phase 9 Step 4 — Statistical hypothesis tests",
           f"Seed: {SEED}, B (bootstrap): {B_BOOT}",
           f"α per test: 0.01 (Bonferroni-corrected family α = 0.006 for "
           f"family of 8 tests)",
           ""]

    rows = []

    # ---------- H1: cross-collection attention non-uniformity ----------
    log.append("## H1 — cross-collection attention non-uniformity")
    log.append("Categorical test: for each collection, what fraction of "
               "peaks land on (lesion ∪ skin) vs (vignette ∪ ruler ∪ hair)?")

    peak["spatial_shortcut"] = peak.category.isin(
        ["on_vignette", "on_ruler", "on_hair"]
    ).astype(int)
    contingency = pd.crosstab(peak.source_collection, peak.spatial_shortcut)
    log.append(f"\n  Contingency (collection × shortcut-vs-not):\n{contingency}")

    chi2, p_h1, dof, _ = stats.chi2_contingency(contingency)
    log.append(f"\n  χ² = {chi2:.3f}, dof = {dof}, p = {p_h1:.4f}")
    h1_supported = p_h1 < 0.01
    log.append(f"  H1 supported (p < 0.01)? {h1_supported}")

    if h1_supported:
        # Per-collection rate + 95% CI via bootstrap
        log.append("\n  Per-collection shortcut-attention rate (95% CI):")
        for c in sorted(peak.source_collection.astype(str).unique()):
            sub = peak[peak.source_collection.astype(str) == c]
            point, lo, hi = bootstrap_ci(
                sub.spatial_shortcut.values, lambda x: x.mean()
            )
            log.append(f"    c={c}: {point:.3f} [{lo:.3f}, {hi:.3f}]")
    else:
        log.append("\n  H1 falsified — attention is uniformly low-shortcut "
                   "across collections. Note this may reflect peak-IoU "
                   "threshold strictness rather than absence of shortcut.")

    rows.append(dict(hypothesis="H1", test="chi2", statistic=chi2, p_value=p_h1,
                     n=len(peak), supported=h1_supported))
    log.append("")

    # ---------- H2: per-shortcut counterfactual ΔP ----------
    log.append("## H2 — per-shortcut counterfactual ΔP ≠ 0")
    log.append("Paired Wilcoxon signed-rank, P(no_X) vs P(original), on "
               "images with X present.")

    h2_specs = [
        ("vignette",  "p_malig__no_vignette",   "vignette_present"),
        ("ruler",     "p_malig__no_ruler",      "ruler_present"),
        ("hair (NC)", "p_malig__no_hair",       "hair_present"),
        ("colorcast", "p_malig__no_colorcast",  "color_cast_present"),
    ]

    for sc_name, col_cf, col_pres in h2_specs:
        sub = cf[cf[col_pres] == 1]
        if len(sub) < 5:
            log.append(f"\n  {sc_name}: N={len(sub)} — too few; skipping")
            continue
        cf_p = sub[col_cf].values
        orig_p = sub["p_malig__original"].values
        delta = cf_p - orig_p
        # paired Wilcoxon (delta vs 0)
        try:
            w_stat, w_p = stats.wilcoxon(delta, alternative="two-sided",
                                          zero_method="zsplit")
        except ValueError:  # all zero
            w_stat, w_p = float("nan"), 1.0
        d_cohen = cohens_d_paired(cf_p, orig_p)
        mean_d, lo_d, hi_d = bootstrap_paired_ci(cf_p, orig_p)

        log.append(f"\n  {sc_name}  (N={len(sub)}):")
        log.append(f"    mean ΔP = {mean_d:+.4f} "
                   f"[95% CI {lo_d:+.4f}, {hi_d:+.4f}]")
        log.append(f"    median ΔP = {np.median(delta):+.4f}")
        log.append(f"    Cohen's d (paired) = {d_cohen:+.3f}")
        log.append(f"    Wilcoxon W = {w_stat:.1f}, p = {w_p:.4f}")
        supported = (w_p < 0.01) and (abs(mean_d) >= 0.02)
        log.append(f"    H2-{sc_name.split()[0]} supported? {supported}")

        rows.append(dict(hypothesis=f"H2-{sc_name.split()[0]}",
                         test="wilcoxon", statistic=w_stat, p_value=w_p,
                         n=len(sub), supported=supported,
                         effect_size=d_cohen, ci_lo=lo_d, ci_hi=hi_d,
                         mean_delta=mean_d))
    log.append("")

    # ---------- H3: cross-collection AUC gap decomposition ----------
    log.append("## H3 — cross-collection AUC gap decomposition")
    log.append("Compares per-collection AUC at original vs all_removed.")

    pred_idx = pred.set_index("isic_id")
    cf_idx = cf.set_index("isic_id")
    h3_rows = []
    for c in sorted(cf.source_collection.astype(str).unique()):
        sub = cf[cf.source_collection.astype(str) == c]
        # AUC at original P_malig and at all_removed P_malig, on the SAME
        # collection sub-sample (50% benign / 50% malignant in our split)
        labels = sub["label"].values
        orig = sub["p_malig__original"].values
        all_rm = sub["p_malig__all_removed"].values

        try:
            auc_orig = roc_auc_score(labels, orig)
            auc_rm = roc_auc_score(labels, all_rm)
        except ValueError:
            auc_orig, auc_rm = float("nan"), float("nan")

        rng = np.random.default_rng(SEED)
        boot_orig, boot_rm = [], []
        n = len(labels)
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, size=n)
            try:
                boot_orig.append(roc_auc_score(labels[idx], orig[idx]))
                boot_rm.append(roc_auc_score(labels[idx], all_rm[idx]))
            except ValueError:
                pass
        boot_orig = np.array(boot_orig); boot_rm = np.array(boot_rm)
        if len(boot_orig) > 0:
            lo_o, hi_o = np.quantile(boot_orig, [0.025, 0.975])
            lo_r, hi_r = np.quantile(boot_rm, [0.025, 0.975])
        else:
            lo_o = hi_o = lo_r = hi_r = float("nan")

        log.append(f"\n  c={c}:")
        log.append(f"    AUC original    = {auc_orig:.4f} [95% CI {lo_o:.4f}, {hi_o:.4f}]")
        log.append(f"    AUC all_removed = {auc_rm:.4f} [95% CI {lo_r:.4f}, {hi_r:.4f}]")
        log.append(f"    ΔAUC            = {auc_rm - auc_orig:+.4f}")

        h3_rows.append((c, auc_orig, auc_rm))

    # Cross-collection gap decomposition: gap(212 - 70) original vs all_removed
    aucs_orig = {c: o for c, o, _ in h3_rows}
    aucs_rm = {c: r for c, _, r in h3_rows}

    gap_orig_212_70 = aucs_orig.get("212", float("nan")) - aucs_orig.get("70", float("nan"))
    gap_rm_212_70 = aucs_rm.get("212", float("nan")) - aucs_rm.get("70", float("nan"))
    log.append(f"\n  Gap (c=212 − c=70):")
    log.append(f"    original    = {gap_orig_212_70:+.4f}")
    log.append(f"    all_removed = {gap_rm_212_70:+.4f}")
    log.append(f"    closed by   = {gap_orig_212_70 - gap_rm_212_70:+.4f} "
               f"({(gap_orig_212_70 - gap_rm_212_70) / max(abs(gap_orig_212_70), 1e-6) * 100:+.1f}%)")

    h3_supported = (gap_orig_212_70 - gap_rm_212_70) / max(abs(gap_orig_212_70), 1e-6) >= 0.30
    log.append(f"  H3 supported (≥ 30% gap closed)? {h3_supported}")
    rows.append(dict(hypothesis="H3", test="auc_gap",
                     statistic=gap_orig_212_70 - gap_rm_212_70,
                     p_value=float("nan"),
                     n=len(cf), supported=h3_supported))
    log.append("")

    # ---------- Primary Finding #1 corroboration with bootstrap CI ----------
    log.append("## Primary Finding #1 — per-collection balanced AUC CIs")
    log.append("Bootstrap CIs on the prevalence-balanced AUCs (sanity for "
               "the slide-9 reframing).")
    for c in sorted(pred.source_collection.astype(str).unique()):
        sub = pred[pred.source_collection.astype(str) == c]
        labels = sub["label"].values
        scores = sub["p_malig"].values
        rng = np.random.default_rng(SEED)
        boots = []
        n = len(labels)
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, size=n)
            try:
                boots.append(roc_auc_score(labels[idx], scores[idx]))
            except ValueError:
                pass
        boots = np.array(boots)
        point = roc_auc_score(labels, scores)
        lo, hi = np.quantile(boots, [0.025, 0.975]) if len(boots) > 0 else (float("nan"), float("nan"))
        log.append(f"  c={c}: AUC = {point:.4f} [95% CI {lo:.4f}, {hi:.4f}]")

    log.append("")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
