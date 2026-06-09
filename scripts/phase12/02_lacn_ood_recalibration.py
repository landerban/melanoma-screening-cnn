"""
Phase 12 / Step 2 — LACN OOD threshold recalibration.

Tests whether LACN (test-time intervention) is deployment-ready by
following an honest calibration/test split protocol that avoids
threshold-fitting leakage.

Pipeline:
  1. Extract Otsu lesion masks for the PAD-UFES-20 OOD cohort
     (200 images, the same cohort Phase 9 used for H10).
  2. Apply LACN (background-only Lab-mean shift) to each image whose
     mask is reliable; forward Phase 6 ckpt -> P_malig.
  3. Stratified split of the OOD cohort into calibration (100) and
     test (100) sets with seed=42.
  4. Threshold sweep on the calibration set: find tau_LACN that meets
     the Phase 9 baseline recall (~0.79).
  5. Evaluate on the test set three ways:
       (A) No LACN + tau=0.661   -- Phase 9 baseline reference
       (B) LACN + tau=0.661      -- LACN without recalibration
       (C) LACN + tau_LACN       -- recalibrated LACN

Outputs:
  artifacts/phase12/02_padufes_lesion_masks/{isic_id}.png
  artifacts/phase12/02_padufes_mask_stats.csv
  artifacts/phase12/02_lacn_ood.csv
  artifacts/phase12/02_lacn_ood.log
  artifacts/phase12/figures/fig18_lacn_ood_recalibration.png
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase12"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

# Reuse Phase 9 lesion mask extraction
import importlib.util
spec = importlib.util.spec_from_file_location(
    "p9_lesion", REPO_ROOT / "scripts" / "phase9" / "05_lesion_masks.py"
)
p9_lesion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p9_lesion)
segment_lesion_otsu = p9_lesion.segment_lesion_otsu

# Reuse Phase 12 LACN normalization
spec2 = importlib.util.spec_from_file_location(
    "p12_lacn", REPO_ROOT / "scripts" / "phase12" / "01_lacn.py"
)
p12_lacn = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(p12_lacn)
normalize_lab_in_region = p12_lacn.normalize_lab_in_region
compute_target_lab_mean = p12_lacn.compute_target_lab_mean

# Paths
OOD_IMG_DIR    = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "images"
OOD_PRED_CSV   = REPO_ROOT / "artifacts" / "phase9" / "11_ood_predictions.csv"
CKPT_PATH      = REPO_ROOT / "best_model.pth"

MASK_DIR       = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_lesion_masks"
MASK_STATS_CSV = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_mask_stats.csv"
OUT_CSV        = REPO_ROOT / "artifacts" / "phase12" / "02_lacn_ood.csv"
LOG_TXT        = REPO_ROOT / "artifacts" / "phase12" / "02_lacn_ood.log"
FIG_PATH       = REPO_ROOT / "artifacts" / "phase12" / "figures" / "fig18_lacn_ood_recalibration.png"

# Phase 9 reference threshold (from PF#3 / H10) and target recall
TAU_BASELINE   = 0.661
TARGET_RECALL  = 0.79  # match Phase 9 baseline recall

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


def step1_extract_masks(ood_pred_df):
    """Otsu lesion masks for every OOD image."""
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    for _, row in tqdm(ood_pred_df.iterrows(), total=len(ood_pred_df),
                       desc="Step 1: Otsu masks"):
        isic_id = row["isic_id"]
        img_path = OOD_IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        mask, stats = segment_lesion_otsu(img)
        cv2.imwrite(str(MASK_DIR / f"{isic_id}.png"), mask)
        stats["isic_id"] = isic_id
        stats["label"] = row["label"]
        stats_rows.append(stats)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(MASK_STATS_CSV, index=False)
    return stats_df


def step2_apply_lacn(ood_pred_df, mask_stats_df, target_lab):
    """LACN (bg-only) + Phase 6 ckpt forward."""
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"[device] {device}")

    reliable_ids = set(mask_stats_df[mask_stats_df.reliable].isic_id)
    print(f"Reliable masks: {len(reliable_ids)}/{len(mask_stats_df)}")

    rows = []
    for _, row in tqdm(ood_pred_df.iterrows(), total=len(ood_pred_df),
                       desc="Step 2: LACN forward"):
        isic_id = row["isic_id"]
        img_path = OOD_IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)

        # Original P (from Phase 9 baseline)
        p_orig = float(row["p_malig"])

        # LACN P (only if reliable mask)
        if isic_id in reliable_ids:
            mask = cv2.imread(str(MASK_DIR / f"{isic_id}.png"), cv2.IMREAD_GRAYSCALE)
            if mask.shape != (IMG_SIZE, IMG_SIZE):
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            bg_mask = 255 - mask
            img_lacn = normalize_lab_in_region(img, bg_mask, target_lab)
            p_lacn = forward_p(model, img_lacn, transform, device)
            mask_reliable = True
        else:
            # Fallback: LACN not applicable; reuse original P
            p_lacn = p_orig
            mask_reliable = False

        rows.append({
            "isic_id": isic_id,
            "label": int(row["label"]),
            "p_orig": p_orig,
            "p_lacn": p_lacn,
            "mask_reliable": mask_reliable,
            "dP_lacn": p_lacn - p_orig,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    return out_df


def step3_split(out_df):
    """Stratified 100/100 calibration/test split. Seed=42."""
    rng = np.random.default_rng(SEED)
    pos = out_df[out_df.label == 1].sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1)))
    neg = out_df[out_df.label == 0].sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1)))

    n_pos_half = len(pos) // 2
    n_neg_half = len(neg) // 2

    cal_df = pd.concat([pos.iloc[:n_pos_half], neg.iloc[:n_neg_half]], ignore_index=True)
    test_df = pd.concat([pos.iloc[n_pos_half:], neg.iloc[n_neg_half:]], ignore_index=True)
    return cal_df, test_df


def step4_find_tau_lacn(cal_df, target_recall=TARGET_RECALL):
    """Sweep tau on LACN-applied P; find smallest tau that meets target_recall."""
    p = cal_df.p_lacn.values
    y = cal_df.label.values

    # Use only positives where p > tau is correct
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
        rows.append((tau, recall, spec, tp, fn, tn, fp))

    sweep = pd.DataFrame(rows, columns=["tau", "recall", "spec", "tp", "fn", "tn", "fp"])

    # Find the *largest* tau that still meets the target recall (highest specificity).
    qualifying = sweep[sweep.recall >= target_recall]
    if len(qualifying) == 0:
        # No tau meets target recall -- pick lowest tau (max recall achievable)
        tau_lacn = float(sweep.tau.iloc[0])
        max_recall = float(sweep.recall.max())
        notes = (f"No tau achieves target recall >= {target_recall:.2f}; "
                 f"max achievable recall on calibration = {max_recall:.3f}. "
                 f"Falling back to tau=0.01.")
    else:
        # Largest tau (=> highest specificity) that still hits target recall
        best_row = qualifying.iloc[qualifying.tau.argmax()]
        tau_lacn = float(best_row.tau)
        notes = (f"Smallest tau achieving recall >= {target_recall:.2f}: "
                 f"tau_LACN={tau_lacn:.3f} (cal recall={best_row.recall:.3f}, "
                 f"cal spec={best_row.spec:.3f}).")
    return tau_lacn, sweep, notes


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


def step5_evaluate(test_df, tau_lacn):
    """Evaluate three modes on the test set."""
    p_no_lacn = test_df.p_orig.values
    p_lacn    = test_df.p_lacn.values
    y         = test_df.label.values

    auc_no_lacn = float(roc_auc_score(y, p_no_lacn))
    auc_lacn    = float(roc_auc_score(y, p_lacn))

    A = metrics_at_tau(p_no_lacn, y, TAU_BASELINE)
    B = metrics_at_tau(p_lacn,    y, TAU_BASELINE)
    C = metrics_at_tau(p_lacn,    y, tau_lacn)

    return dict(auc_no_lacn=auc_no_lacn, auc_lacn=auc_lacn,
                A=A, B=B, C=C)


def make_figure(test_df, eval_res, tau_lacn, sweep):
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: ROC curves
    ax = axes[0]
    fpr1, tpr1, _ = roc_curve(test_df.label.values, test_df.p_orig.values)
    fpr2, tpr2, _ = roc_curve(test_df.label.values, test_df.p_lacn.values)
    ax.plot(fpr1, tpr1, label=f"No LACN (AUC={eval_res['auc_no_lacn']:.3f})",
            color="steelblue", lw=2)
    ax.plot(fpr2, tpr2, label=f"LACN     (AUC={eval_res['auc_lacn']:.3f})",
            color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("OOD ROC: PAD-UFES-20 (test split, N=100)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Calibration sweep
    ax = axes[1]
    ax.plot(sweep.tau, sweep.recall, label="recall", color="darkorange", lw=2)
    ax.plot(sweep.tau, sweep.spec, label="specificity", color="steelblue", lw=2)
    ax.axhline(TARGET_RECALL, color="red", ls="--", alpha=0.5,
               label=f"target recall = {TARGET_RECALL}")
    ax.axvline(tau_lacn, color="green", ls="--", alpha=0.7,
               label=f"tau_LACN = {tau_lacn:.3f}")
    ax.axvline(TAU_BASELINE, color="gray", ls=":", alpha=0.7,
               label=f"tau_baseline = {TAU_BASELINE:.3f}")
    ax.set_xlabel("threshold tau")
    ax.set_ylabel("metric")
    ax.set_title("Threshold sweep on LACN-applied calibration set (N=100)")
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: Test-set 3-mode comparison
    ax = axes[2]
    modes = ["A: No LACN\ntau=0.661", "B: LACN\ntau=0.661", "C: LACN\ntau_LACN"]
    metrics = ["recall", "spec", "f1"]
    colors  = {"recall": "darkorange", "spec": "steelblue", "f1": "seagreen"}
    n_modes = len(modes)
    x = np.arange(n_modes)
    w = 0.27
    for i, m in enumerate(metrics):
        vals = [eval_res["A"][m], eval_res["B"][m], eval_res["C"][m]]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], alpha=0.85)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9)
    ax.axhline(TARGET_RECALL, color="red", ls="--", alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.set_title("OOD test-set metrics: 3-mode comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Phase 12 / fig18 — LACN OOD threshold recalibration",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    log = ["# Phase 12 / Step 2 — LACN OOD threshold recalibration", ""]

    # Load Phase 9 OOD baseline predictions (no LACN, tau=0.661 reference)
    ood_pred_df = pd.read_csv(OOD_PRED_CSV)
    log.append(f"OOD cohort: PAD-UFES-20, N = {len(ood_pred_df)}")
    log.append(f"Prevalence: {ood_pred_df.label.mean():.1%} malignant")
    log.append("")

    # Step 1: Otsu masks
    print("\n=== Step 1: Otsu lesion masks for PAD-UFES-20 ===")
    if MASK_STATS_CSV.exists():
        mask_stats_df = pd.read_csv(MASK_STATS_CSV)
        print(f"[cached] {MASK_STATS_CSV.relative_to(REPO_ROOT)}")
    else:
        mask_stats_df = step1_extract_masks(ood_pred_df)
    n_reliable = int(mask_stats_df.reliable.sum())
    log.append(f"## Step 1 — Otsu mask reliability")
    log.append(f"  Reliable masks: {n_reliable}/{len(mask_stats_df)} "
               f"({n_reliable / max(len(mask_stats_df), 1):.1%})")
    log.append(f"  Mean area fraction: {mask_stats_df.area_frac.mean():.3f}")
    log.append("")

    # Compute target Lab mean (same Phase 9 distribution reference as Phase 12)
    print("\n=== Computing target Lab mean (Phase 9 analytical-sample reference) ===")
    target_lab = compute_target_lab_mean()
    log.append(f"Target Lab mean (Phase 9 ref): {target_lab.round(2).tolist()}")
    log.append("")

    # Step 2: Apply LACN
    print("\n=== Step 2: Apply LACN + Phase 6 forward ===")
    if OUT_CSV.exists():
        out_df = pd.read_csv(OUT_CSV)
        print(f"[cached] {OUT_CSV.relative_to(REPO_ROOT)}")
    else:
        out_df = step2_apply_lacn(ood_pred_df, mask_stats_df, target_lab)
    n_lacn_applied = int(out_df.mask_reliable.sum())
    log.append(f"## Step 2 — LACN applied")
    log.append(f"  LACN applied (reliable mask): {n_lacn_applied}/{len(out_df)}")
    log.append(f"  Mean dP_LACN (LACN - original P): {out_df.dP_lacn.mean():+.4f}")
    log.append("")

    # Step 3: Split
    print("\n=== Step 3: Stratified calibration/test split ===")
    cal_df, test_df = step3_split(out_df)
    log.append(f"## Step 3 — Stratified split (seed={SEED})")
    log.append(f"  Calibration: N={len(cal_df)} (pos={int(cal_df.label.sum())}, neg={int(len(cal_df) - cal_df.label.sum())})")
    log.append(f"  Test:        N={len(test_df)} (pos={int(test_df.label.sum())}, neg={int(len(test_df) - test_df.label.sum())})")
    log.append("")

    # Step 4: Find tau_LACN on calibration
    print("\n=== Step 4: Threshold sweep on calibration ===")
    tau_lacn, sweep, sweep_notes = step4_find_tau_lacn(cal_df, TARGET_RECALL)
    log.append(f"## Step 4 — Threshold recalibration on calibration set")
    log.append(f"  Target recall: {TARGET_RECALL}")
    log.append(f"  {sweep_notes}")
    log.append("")

    # Step 5: Evaluate three modes on test
    print("\n=== Step 5: 3-mode evaluation on held-out test ===")
    eval_res = step5_evaluate(test_df, tau_lacn)
    log.append(f"## Step 5 — Test-set evaluation (N={len(test_df)})")
    log.append(f"  AUC (no LACN): {eval_res['auc_no_lacn']:.4f}")
    log.append(f"  AUC (LACN):    {eval_res['auc_lacn']:.4f}")
    log.append("")
    for label, key in [("(A) No LACN  + tau=0.661  ", "A"),
                       ("(B) LACN     + tau=0.661  ", "B"),
                       ("(C) LACN     + tau_LACN   ", "C")]:
        m = eval_res[key]
        log.append(f"  {label}: recall={m['recall']:.3f}  spec={m['spec']:.3f}  "
                   f"prec={m['prec']:.3f}  f1={m['f1']:.3f}  "
                   f"TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}")
    log.append("")

    # Verdict
    a_recall, c_recall = eval_res["A"]["recall"], eval_res["C"]["recall"]
    a_spec,   c_spec   = eval_res["A"]["spec"],   eval_res["C"]["spec"]
    a_f1,     c_f1     = eval_res["A"]["f1"],     eval_res["C"]["f1"]
    log.append("## Verdict")
    log.append(f"  Recall    : A={a_recall:.3f} -> C={c_recall:.3f}  (delta {c_recall - a_recall:+.3f})")
    log.append(f"  Specificity: A={a_spec:.3f} -> C={c_spec:.3f}    (delta {c_spec - a_spec:+.3f})")
    log.append(f"  F1        : A={a_f1:.3f} -> C={c_f1:.3f}       (delta {c_f1 - a_f1:+.3f})")
    if c_recall >= a_recall - 0.02 and c_spec >= a_spec - 0.02:
        log.append("  -> LACN recalibrated meets baseline. Deployment-ready (with calibration step).")
    elif c_recall >= a_recall - 0.02:
        log.append("  -> LACN recalibrated preserves recall but loses specificity. Partial win.")
    else:
        log.append("  -> LACN cannot recover Phase 9 baseline even after recalibration. "
                   "Fundamental trade-off: bg-only normalization erodes legitimate OOD color cues.")
    log.append("")

    # Figure
    print("\n=== Generating fig18 ===")
    make_figure(test_df, eval_res, tau_lacn, sweep)
    log.append(f"## Files emitted")
    log.append(f"  Masks dir: {MASK_DIR.relative_to(REPO_ROOT)}/")
    log.append(f"  Stats:    {MASK_STATS_CSV.relative_to(REPO_ROOT)}")
    log.append(f"  CSV:      {OUT_CSV.relative_to(REPO_ROOT)}")
    log.append(f"  Figure:   {FIG_PATH.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
