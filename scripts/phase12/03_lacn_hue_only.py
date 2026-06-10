"""
Phase 12 / Step 3 — Channel-selective LACN: Hue-only background normalization.

Motivation:
  Phase 12 / Step 2 (02_lacn_ood_recalibration.py) showed that bg-only
  Lab-mean shift recovers OOD recall but loses specificity (-0.080) and
  AUC (-0.052). Hypothesis: full Lab shift erodes legitimate Saturation
  and Value cues that PAD-UFES-20 carries.

  Phase 11 measured that 93% of the color-cast effect lives in the Hue
  channel. This script implements a channel-selective LACN that shifts
  *only* the Hue of the background, leaving Saturation and Value
  untouched. If the Phase 11 measurement is correct, Hue-only LACN
  should match Lab-LACN on recall while sparing specificity.

  Hue is circular (OpenCV: 0-180, semantically 0-360 deg). We use a
  circular mean to avoid wrap-around bias when computing the target.

Pipeline (reuses Step 2 infrastructure):
  1. Reuse PAD-UFES-20 Otsu masks from artifacts/phase12/02_padufes_lesion_masks/
  2. Compute target Hue (circular mean over Phase 9 analytical sample)
  3. For each OOD image with a reliable mask, shift background Hue only
  4. Forward Phase 6 ckpt -> P_malig
  5. Use the *same* calibration/test split (seed=42) as Step 2
  6. Threshold sweep on calibration -> tau_LACN_hue
  7. Compare three modes on test set:
       (A) No LACN          + tau=0.661   (Phase 9 baseline reference)
       (B) Hue-only LACN    + tau=0.661   (no recalibration)
       (C) Hue-only LACN    + tau_LACN_hue (recalibrated)

Outputs:
  artifacts/phase12/03_lacn_hue_ood.csv
  artifacts/phase12/03_lacn_hue_ood.log
  artifacts/phase12/figures/fig19_lacn_hue_vs_lab.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

# Paths
OOD_IMG_DIR    = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "images"
OOD_PRED_CSV   = REPO_ROOT / "artifacts" / "phase9" / "11_ood_predictions.csv"
LACN_LAB_CSV   = REPO_ROOT / "artifacts" / "phase12" / "02_lacn_ood.csv"
MASK_STATS_CSV = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_mask_stats.csv"
MASK_DIR       = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_lesion_masks"
P9_SAMPLE_CSV  = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
P9_IMG_DIR     = REPO_ROOT / "training_data" / "images"
CKPT_PATH      = REPO_ROOT / "best_model.pth"

OUT_CSV  = REPO_ROOT / "artifacts" / "phase12" / "03_lacn_hue_ood.csv"
LOG_TXT  = REPO_ROOT / "artifacts" / "phase12" / "03_lacn_hue_ood.log"
FIG_PATH = REPO_ROOT / "artifacts" / "phase12" / "figures" / "fig19_lacn_hue_vs_lab.png"

TAU_BASELINE  = 0.661
TARGET_RECALL = 0.79
SEED = 42


def pick_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(device):
    obj = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    m = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def forward_p(model, img_bgr, transform, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(x).squeeze()))


def circular_mean_hue_deg(hue_array_opencv):
    """Circular mean of an OpenCV Hue array (0-180). Returns degrees in 0-360."""
    h_360 = hue_array_opencv.astype(np.float32) * 2.0  # 0-180 -> 0-360
    h_rad = np.deg2rad(h_360.ravel())
    x = np.cos(h_rad).mean()
    y = np.sin(h_rad).mean()
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def shortest_signed_diff(target_deg, current_deg):
    """Smallest signed delta to rotate current -> target on a 360-deg circle."""
    diff = (target_deg - current_deg) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def normalize_hue_in_region(img_bgr, mask, target_hue_deg):
    """Shift the *circular Hue mean* of pixels inside `mask` (binary, 255=in)
    to `target_hue_deg` (degrees, 0-360). Saturation and Value are preserved.

    Returns BGR image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    inside = mask > 127
    if inside.sum() == 0:
        return img_bgr

    cur_hue_deg = circular_mean_hue_deg(hsv[inside, 0])
    shift_deg = shortest_signed_diff(target_hue_deg, cur_hue_deg)
    shift_opencv = shift_deg / 2.0  # OpenCV Hue scale is half-degree

    h_new = (hsv[inside, 0].astype(np.float32) + shift_opencv) % 180.0
    hsv_out = hsv.copy()
    hsv_out[inside, 0] = h_new.astype(np.uint8)
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def compute_target_hue_from_phase9():
    """Circular mean Hue over the Phase 9 analytical sample (in-distribution)."""
    df = pd.read_csv(P9_SAMPLE_CSV)
    xs, ys = [], []
    for isic_id in tqdm(df.isic_id, desc="Compute target Hue (Phase 9)"):
        p = P9_IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_360 = hsv[:, :, 0].astype(np.float32) * 2.0
        h_rad = np.deg2rad(h_360.ravel())
        xs.append(np.cos(h_rad).mean())
        ys.append(np.sin(h_rad).mean())
    mx, my = float(np.mean(xs)), float(np.mean(ys))
    return float(np.rad2deg(np.arctan2(my, mx)) % 360.0)


def metrics_at_tau(p_arr, y_arr, tau):
    pred = (p_arr >= tau).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-9)
    return dict(tau=tau, tp=tp, fn=fn, tn=tn, fp=fp,
                recall=recall, spec=spec, prec=prec, f1=f1)


def step2_apply_hue_lacn(ood_pred_df, mask_stats_df, target_hue_deg):
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"[device] {device}")
    print(f"Target Hue (Phase 9 circular mean): {target_hue_deg:.2f} deg")

    reliable_ids = set(mask_stats_df[mask_stats_df.reliable].isic_id)
    print(f"Reliable masks: {len(reliable_ids)}/{len(mask_stats_df)}")

    rows = []
    for _, row in tqdm(ood_pred_df.iterrows(), total=len(ood_pred_df),
                       desc="Hue-only LACN forward"):
        isic_id = row["isic_id"]
        img_path = OOD_IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        p_orig = float(row["p_malig"])

        if isic_id in reliable_ids:
            mask = cv2.imread(str(MASK_DIR / f"{isic_id}.png"), cv2.IMREAD_GRAYSCALE)
            if mask.shape != (IMG_SIZE, IMG_SIZE):
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
            bg_mask = 255 - mask
            img_hue_lacn = normalize_hue_in_region(img, bg_mask, target_hue_deg)
            p_hue = forward_p(model, img_hue_lacn, transform, device)
            mask_reliable = True
        else:
            p_hue = p_orig
            mask_reliable = False

        rows.append({
            "isic_id": isic_id,
            "label": int(row["label"]),
            "p_orig": p_orig,
            "p_hue_lacn": p_hue,
            "mask_reliable": mask_reliable,
            "dP_hue_lacn": p_hue - p_orig,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    return out_df


def stratified_split(out_df):
    """Same seed=42 split as Step 2 for direct comparability."""
    rng = np.random.default_rng(SEED)
    pos = out_df[out_df.label == 1].sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1)))
    neg = out_df[out_df.label == 0].sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1)))
    n_p = len(pos) // 2
    n_n = len(neg) // 2
    cal = pd.concat([pos.iloc[:n_p], neg.iloc[:n_n]], ignore_index=True)
    test = pd.concat([pos.iloc[n_p:], neg.iloc[n_n:]], ignore_index=True)
    return cal, test


def find_tau(cal_df, p_col, target_recall=TARGET_RECALL):
    p = cal_df[p_col].values
    y = cal_df.label.values
    taus = np.linspace(0.01, 0.99, 99)
    rows = []
    for tau in taus:
        pred = (p >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        recall = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        rows.append((tau, recall, spec))
    sweep = pd.DataFrame(rows, columns=["tau", "recall", "spec"])
    qualifying = sweep[sweep.recall >= target_recall]
    if len(qualifying) == 0:
        tau_star = float(sweep.tau.iloc[0])
        notes = (f"No tau achieves recall >= {target_recall}; max recall = "
                 f"{sweep.recall.max():.3f}. Falling back to tau=0.01.")
    else:
        best = qualifying.iloc[qualifying.tau.argmax()]
        tau_star = float(best.tau)
        notes = (f"tau* (largest tau with recall >= {target_recall}) = "
                 f"{tau_star:.3f}  (cal recall={best.recall:.3f}, "
                 f"cal spec={best.spec:.3f}).")
    return tau_star, sweep, notes


def make_figure(test_df, lab_df, eval_hue, eval_lab, tau_hue):
    """Compare LACN-Lab (Step 2) vs LACN-Hue (Step 3) on the held-out test."""
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: ROC -- no LACN, LACN-Lab, LACN-Hue
    ax = axes[0]
    y = test_df.label.values
    fpr0, tpr0, _ = roc_curve(y, test_df.p_orig.values)
    auc0 = roc_auc_score(y, test_df.p_orig.values)

    # Join LACN-Lab predictions on test isic_ids
    test_lab = test_df[["isic_id"]].merge(lab_df[["isic_id", "p_lacn"]],
                                          on="isic_id", how="left")
    fpr_lab, tpr_lab, _ = roc_curve(y, test_lab.p_lacn.values)
    auc_lab = roc_auc_score(y, test_lab.p_lacn.values)

    fpr_h, tpr_h, _ = roc_curve(y, test_df.p_hue_lacn.values)
    auc_h = roc_auc_score(y, test_df.p_hue_lacn.values)

    ax.plot(fpr0, tpr0, label=f"No LACN    (AUC={auc0:.3f})",
            color="steelblue", lw=2)
    ax.plot(fpr_lab, tpr_lab, label=f"LACN-Lab   (AUC={auc_lab:.3f})",
            color="darkorange", lw=2)
    ax.plot(fpr_h, tpr_h, label=f"LACN-Hue   (AUC={auc_h:.3f})",
            color="seagreen", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC: No LACN vs LACN-Lab vs LACN-Hue (test, N=100)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Metric comparison
    ax = axes[1]
    modes = [
        "A: No LACN\ntau=0.661",
        "B-Lab: LACN\ntau=0.661",
        "C-Lab: LACN\ntau_LACN",
        "B-Hue: LACN\ntau=0.661",
        "C-Hue: LACN\ntau_LACN",
    ]
    metric_names = ["recall", "spec", "f1"]
    colors = {"recall": "darkorange", "spec": "steelblue", "f1": "seagreen"}
    bar_data = {
        "recall": [eval_lab["A"]["recall"], eval_lab["B"]["recall"],
                   eval_lab["C"]["recall"], eval_hue["B"]["recall"],
                   eval_hue["C"]["recall"]],
        "spec":   [eval_lab["A"]["spec"], eval_lab["B"]["spec"],
                   eval_lab["C"]["spec"], eval_hue["B"]["spec"],
                   eval_hue["C"]["spec"]],
        "f1":     [eval_lab["A"]["f1"], eval_lab["B"]["f1"],
                   eval_lab["C"]["f1"], eval_hue["B"]["f1"],
                   eval_hue["C"]["f1"]],
    }
    x = np.arange(len(modes))
    w = 0.27
    for i, m in enumerate(metric_names):
        ax.bar(x + (i - 1) * w, bar_data[m], w, label=m,
               color=colors[m], alpha=0.85)
        for xi, v in zip(x + (i - 1) * w, bar_data[m]):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=7.5)
    ax.axhline(TARGET_RECALL, color="red", ls="--", alpha=0.3,
               label=f"target recall={TARGET_RECALL}")
    ax.set_ylim(0, 1.1)
    ax.set_title("Test-set metric comparison: Lab vs Hue LACN")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Delta vs baseline
    ax = axes[2]
    a_metrics = eval_lab["A"]
    deltas = {
        "B-Lab":   {m: eval_lab["B"][m] - a_metrics[m] for m in metric_names},
        "C-Lab":   {m: eval_lab["C"][m] - a_metrics[m] for m in metric_names},
        "B-Hue":   {m: eval_hue["B"][m] - a_metrics[m] for m in metric_names},
        "C-Hue":   {m: eval_hue["C"][m] - a_metrics[m] for m in metric_names},
    }
    dmodes = list(deltas.keys())
    x = np.arange(len(dmodes))
    for i, m in enumerate(metric_names):
        vals = [deltas[mode][m] for mode in dmodes]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], alpha=0.85)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + (0.005 if v >= 0 else -0.025), f"{v:+.02f}",
                    ha="center", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(dmodes, fontsize=9)
    ax.set_title("Delta vs (A) No-LACN baseline")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-0.15, 0.15)

    plt.suptitle("Phase 12 / fig19 -- LACN-Hue vs LACN-Lab: "
                 "channel-selective intervention recovers specificity?",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    log = ["# Phase 12 / Step 3 -- Channel-selective LACN (Hue-only)", ""]

    # Load Phase 9 OOD baseline
    ood_pred_df = pd.read_csv(OOD_PRED_CSV)
    log.append(f"OOD cohort: PAD-UFES-20, N = {len(ood_pred_df)}")
    log.append(f"Prevalence: {ood_pred_df.label.mean():.1%} malignant")
    log.append("")

    # Load Otsu masks (reuse Step 2)
    mask_stats_df = pd.read_csv(MASK_STATS_CSV)
    n_reliable = int(mask_stats_df.reliable.sum())
    log.append(f"## Reuse Otsu masks from Step 2: reliable={n_reliable}/{len(mask_stats_df)}")
    log.append("")

    # Target Hue (circular mean, Phase 9 sample)
    print("\n=== Compute target Hue (circular mean, Phase 9 sample) ===")
    target_hue_deg = compute_target_hue_from_phase9()
    log.append(f"Target Hue (circular mean over Phase 9 analytical sample): "
               f"{target_hue_deg:.2f} deg")
    log.append("")

    # Step: apply Hue-only LACN
    print("\n=== Apply Hue-only LACN + Phase 6 forward ===")
    if OUT_CSV.exists():
        out_df = pd.read_csv(OUT_CSV)
        print(f"[cached] {OUT_CSV.relative_to(REPO_ROOT)}")
    else:
        out_df = step2_apply_hue_lacn(ood_pred_df, mask_stats_df, target_hue_deg)
    log.append(f"Hue-LACN applied (reliable mask): {int(out_df.mask_reliable.sum())}/{len(out_df)}")
    log.append(f"Mean dP_hue_lacn (Hue-LACN - original P): {out_df.dP_hue_lacn.mean():+.4f}")
    log.append("")

    # Split (same seed)
    print("\n=== Stratified cal/test split (seed=42, same as Step 2) ===")
    cal_df, test_df = stratified_split(out_df)
    log.append(f"Calibration: N={len(cal_df)} (pos={int(cal_df.label.sum())}, "
               f"neg={int(len(cal_df) - cal_df.label.sum())})")
    log.append(f"Test:        N={len(test_df)} (pos={int(test_df.label.sum())}, "
               f"neg={int(len(test_df) - test_df.label.sum())})")
    log.append("")

    # Find tau_LACN_hue
    print("\n=== Threshold sweep on calibration (Hue-LACN P) ===")
    tau_hue, sweep, sweep_notes = find_tau(cal_df, "p_hue_lacn", TARGET_RECALL)
    log.append(f"## Threshold recalibration on calibration")
    log.append(f"  Target recall: {TARGET_RECALL}")
    log.append(f"  {sweep_notes}")
    log.append("")

    # Three-way Hue eval on test
    y = test_df.label.values
    p_orig = test_df.p_orig.values
    p_hue  = test_df.p_hue_lacn.values
    auc_orig = float(roc_auc_score(y, p_orig))
    auc_hue  = float(roc_auc_score(y, p_hue))

    eval_hue = dict(
        auc_no_lacn=auc_orig,
        auc_lacn=auc_hue,
        A=metrics_at_tau(p_orig, y, TAU_BASELINE),
        B=metrics_at_tau(p_hue,  y, TAU_BASELINE),
        C=metrics_at_tau(p_hue,  y, tau_hue),
    )

    log.append("## Hue-only LACN test-set evaluation (N={})".format(len(test_df)))
    log.append(f"  AUC (no LACN): {auc_orig:.4f}")
    log.append(f"  AUC (Hue-LACN):{auc_hue:.4f}")
    log.append("")
    for label, key in [("(A) No LACN  + tau=0.661  ", "A"),
                       ("(B) Hue LACN + tau=0.661  ", "B"),
                       ("(C) Hue LACN + tau_LACN   ", "C")]:
        m = eval_hue[key]
        log.append(f"  {label}: recall={m['recall']:.3f}  spec={m['spec']:.3f}  "
                   f"prec={m['prec']:.3f}  f1={m['f1']:.3f}  "
                   f"TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}")
    log.append("")

    # Cross-comparison with Step 2 LACN-Lab
    print("\n=== Cross-compare with Step 2 (LACN-Lab) ===")
    lab_df = pd.read_csv(LACN_LAB_CSV)
    test_lab = test_df[["isic_id", "label"]].merge(
        lab_df[["isic_id", "p_lacn"]], on="isic_id", how="left")
    y_lab = test_lab.label.values
    p_lab = test_lab.p_lacn.values
    cal_lab = cal_df[["isic_id"]].merge(lab_df[["isic_id", "p_lacn", "label"]],
                                        on="isic_id", how="left")
    tau_lab, _, _ = find_tau(cal_lab, "p_lacn", TARGET_RECALL)

    eval_lab = dict(
        auc_no_lacn=auc_orig,
        auc_lacn=float(roc_auc_score(y_lab, p_lab)),
        A=metrics_at_tau(p_orig, y, TAU_BASELINE),
        B=metrics_at_tau(p_lab,  y, TAU_BASELINE),
        C=metrics_at_tau(p_lab,  y, tau_lab),
    )

    log.append("## Lab-LACN (Step 2) test-set comparison (same test split)")
    log.append(f"  tau_LACN_Lab (re-derived from calibration): {tau_lab:.3f}")
    log.append(f"  AUC (Lab-LACN): {eval_lab['auc_lacn']:.4f}")
    log.append("")
    for label, key in [("(A) No LACN  + tau=0.661 ", "A"),
                       ("(B) Lab LACN + tau=0.661 ", "B"),
                       ("(C) Lab LACN + tau_LACN  ", "C")]:
        m = eval_lab[key]
        log.append(f"  {label}: recall={m['recall']:.3f}  spec={m['spec']:.3f}  "
                   f"f1={m['f1']:.3f}")
    log.append("")

    # Direct head-to-head (recalibrated mode C)
    log.append("## Head-to-head: recalibrated C-Hue vs C-Lab vs A-baseline")
    a, c_lab, c_hue = eval_lab["A"], eval_lab["C"], eval_hue["C"]
    log.append(f"  Recall    : A={a['recall']:.3f}  C-Lab={c_lab['recall']:.3f}  C-Hue={c_hue['recall']:.3f}")
    log.append(f"  Specificity: A={a['spec']:.3f}  C-Lab={c_lab['spec']:.3f}  C-Hue={c_hue['spec']:.3f}")
    log.append(f"  F1        : A={a['f1']:.3f}  C-Lab={c_lab['f1']:.3f}  C-Hue={c_hue['f1']:.3f}")
    log.append(f"  AUC       : {eval_lab['auc_no_lacn']:.3f} (none)  "
               f"{eval_lab['auc_lacn']:.3f} (Lab)  {eval_hue['auc_lacn']:.3f} (Hue)")
    log.append("")

    # Verdict
    spec_recovered = c_hue["spec"] >= c_lab["spec"] + 0.02
    auc_recovered  = eval_hue["auc_lacn"] >= eval_lab["auc_lacn"] + 0.02
    log.append("## Verdict")
    if spec_recovered and auc_recovered:
        log.append("  Hue-only LACN recovers BOTH specificity and AUC compared to Lab-LACN.")
        log.append("  Phase 11 channel decomposition (Hue carries 93% of color cast) is "
                   "validated for test-time intervention design: shifting only the "
                   "carrier channel preserves legitimate diagnostic cues.")
    elif spec_recovered:
        log.append("  Hue-only LACN partially improves on Lab-LACN (specificity recovered, "
                   "AUC similar). The channel-selective hypothesis is supported.")
    elif auc_recovered:
        log.append("  AUC improves but specificity remains below baseline. Hue-shift "
                   "still erodes some discrimination signal.")
    else:
        log.append("  Hue-only LACN does not outperform Lab-LACN on the held-out test. "
                   "The Lab-LACN trade-off appears to extend to Hue-only -- legitimate "
                   "OOD color cues may live across multiple channels.")
    log.append("")

    # Figure
    print("\n=== Generate fig19 ===")
    make_figure(test_df, lab_df, eval_hue, eval_lab, tau_hue)
    log.append(f"## Files emitted")
    log.append(f"  CSV:     {OUT_CSV.relative_to(REPO_ROOT)}")
    log.append(f"  Figure:  {FIG_PATH.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
