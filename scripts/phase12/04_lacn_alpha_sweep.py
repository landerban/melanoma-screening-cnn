"""
Phase 12 / Step 4 -- Partial-strength LACN: alpha-interpolation sweep.

Motivation:
  Phase 12 / Step 2 (Lab-LACN at full strength) costs 0.080 specificity
  and 0.052 AUC on PAD-UFES-20 OOD. Step 3 (Hue-only LACN) is worse.
  Hypothesis: the trade-off is monotonic in normalization *strength*,
  so partial-strength LACN at some intermediate alpha may sit at a
  better operating point.

  We define a continuous LACN intervention by alpha-interpolating
  between the original image and the full Lab-LACN image at the pixel
  level (BGR space):

      img_partial(alpha) = (1 - alpha) * img_orig + alpha * img_lacn

  alpha = 0 -> no intervention (Phase 9 baseline)
  alpha = 1 -> full Lab-LACN (Step 2)

  Sweep alpha in {0.0, 0.25, 0.5, 0.75, 1.0}.

Pipeline:
  1. Reuse PAD-UFES-20 Otsu masks (172/200 reliable, Step 2 cache)
  2. Reuse Phase 9 target Lab mean (Step 2 cache)
  3. For each alpha: build alpha-blended image -> Phase 6 forward
  4. Reuse same cal/test split (seed=42)
  5. For each alpha: threshold sweep on calibration -> tau_alpha
  6. Test-set evaluation per alpha:
       (B_alpha) tau=0.661  -- un-recalibrated
       (C_alpha) tau=tau_alpha -- recalibrated
  7. Plot operating curves vs alpha

Outputs:
  artifacts/phase12/04_lacn_alpha.csv
  artifacts/phase12/04_lacn_alpha.log
  artifacts/phase12/figures/fig20_lacn_alpha_sweep.png
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

# Reuse Phase 12 LACN normalization
import importlib.util
spec_lacn = importlib.util.spec_from_file_location(
    "p12_lacn", REPO_ROOT / "scripts" / "phase12" / "01_lacn.py"
)
p12_lacn = importlib.util.module_from_spec(spec_lacn)
spec_lacn.loader.exec_module(p12_lacn)
normalize_lab_in_region = p12_lacn.normalize_lab_in_region
compute_target_lab_mean = p12_lacn.compute_target_lab_mean

# Paths
OOD_IMG_DIR    = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "images"
OOD_PRED_CSV   = REPO_ROOT / "artifacts" / "phase9" / "11_ood_predictions.csv"
MASK_STATS_CSV = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_mask_stats.csv"
MASK_DIR       = REPO_ROOT / "artifacts" / "phase12" / "02_padufes_lesion_masks"
CKPT_PATH      = REPO_ROOT / "best_model.pth"

OUT_CSV  = REPO_ROOT / "artifacts" / "phase12" / "04_lacn_alpha.csv"
LOG_TXT  = REPO_ROOT / "artifacts" / "phase12" / "04_lacn_alpha.log"
FIG_PATH = REPO_ROOT / "artifacts" / "phase12" / "figures" / "fig20_lacn_alpha_sweep.png"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
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


def alpha_blend(img_orig_bgr, img_lacn_bgr, alpha):
    """Per-pixel alpha blend in BGR space."""
    a = float(alpha)
    return ((1.0 - a) * img_orig_bgr.astype(np.float32)
            + a * img_lacn_bgr.astype(np.float32)).clip(0, 255).astype(np.uint8)


def compute_per_image_p_all_alphas(ood_pred_df, mask_stats_df, target_lab):
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"[device] {device}")
    reliable_ids = set(mask_stats_df[mask_stats_df.reliable].isic_id)
    print(f"Reliable masks: {len(reliable_ids)}/{len(mask_stats_df)}")

    rows = []
    for _, row in tqdm(ood_pred_df.iterrows(), total=len(ood_pred_df),
                       desc="LACN alpha-sweep forward"):
        isic_id = row["isic_id"]
        img_path = OOD_IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)
        p_orig = float(row["p_malig"])

        row_out = {
            "isic_id": isic_id,
            "label": int(row["label"]),
            "p_alpha_0.00": p_orig,
            "mask_reliable": False,
        }

        if isic_id in reliable_ids:
            mask = cv2.imread(str(MASK_DIR / f"{isic_id}.png"), cv2.IMREAD_GRAYSCALE)
            if mask.shape != (IMG_SIZE, IMG_SIZE):
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
            bg_mask = 255 - mask
            img_lacn = normalize_lab_in_region(img, bg_mask, target_lab)
            row_out["mask_reliable"] = True
            for alpha in ALPHAS[1:]:
                blended = alpha_blend(img, img_lacn, alpha)
                row_out[f"p_alpha_{alpha:.2f}"] = forward_p(model, blended,
                                                            transform, device)
        else:
            # Unreliable mask -> fall back to p_orig at every alpha
            for alpha in ALPHAS[1:]:
                row_out[f"p_alpha_{alpha:.2f}"] = p_orig

        rows.append(row_out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    return out_df


def stratified_split(out_df):
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
        return float(sweep.tau.iloc[0]), float(sweep.recall.max())
    best = qualifying.iloc[qualifying.tau.argmax()]
    return float(best.tau), float(best.recall)


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


def make_figure(per_alpha_summary):
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_alpha_summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: AUC and recall/spec at tau_alpha
    ax = axes[0]
    ax.plot(df.alpha, df.auc, marker="o", color="black", lw=2,
            label="AUC (LACN-applied)")
    ax.plot(df.alpha, df.recall_tau_alpha, marker="s", color="darkorange",
            lw=2, label="Recall @ tau_alpha")
    ax.plot(df.alpha, df.spec_tau_alpha, marker="^", color="steelblue",
            lw=2, label="Spec @ tau_alpha")
    ax.axhline(df.iloc[0].auc, color="black", ls=":", alpha=0.4,
               label="alpha=0 ref")
    ax.set_xlabel("LACN strength alpha (0=no LACN, 1=full Lab-LACN)")
    ax.set_ylabel("metric")
    ax.set_title("Operating curve at recalibrated tau_alpha (test, N=100)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.4, 1.0)

    # Panel 2: B (un-recalib) vs C (recalib) deltas vs baseline alpha=0
    ax = axes[1]
    base_recall = df.iloc[0].recall_tau_alpha
    base_spec   = df.iloc[0].spec_tau_alpha
    base_auc    = df.iloc[0].auc
    ax.plot(df.alpha, df.recall_tau_alpha - base_recall, marker="s",
            color="darkorange", lw=2, label="Δrecall vs alpha=0")
    ax.plot(df.alpha, df.spec_tau_alpha - base_spec, marker="^",
            color="steelblue", lw=2, label="Δspecificity vs alpha=0")
    ax.plot(df.alpha, df.auc - base_auc, marker="o", color="black",
            lw=2, label="ΔAUC vs alpha=0")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("LACN strength alpha")
    ax.set_ylabel("delta vs alpha=0")
    ax.set_title("Trade-off curve: gains and losses as alpha increases")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-0.2, 0.1)

    plt.suptitle("Phase 12 / fig20 -- LACN partial-strength alpha sweep",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    log = ["# Phase 12 / Step 4 -- LACN partial-strength alpha sweep", ""]

    ood_pred_df = pd.read_csv(OOD_PRED_CSV)
    log.append(f"OOD cohort: PAD-UFES-20, N = {len(ood_pred_df)}")
    log.append(f"Prevalence: {ood_pred_df.label.mean():.1%} malignant")
    log.append("")

    mask_stats_df = pd.read_csv(MASK_STATS_CSV)
    log.append(f"Reuse Otsu masks: reliable={int(mask_stats_df.reliable.sum())}/{len(mask_stats_df)}")

    # Target Lab mean
    print("\n=== Compute target Lab mean (Phase 9 sample) ===")
    target_lab = compute_target_lab_mean()
    log.append(f"Target Lab mean (Phase 9 ref): {target_lab.round(2).tolist()}")
    log.append("")

    # Per-image, per-alpha forward
    print(f"\n=== Forward pass over alphas={ALPHAS} ===")
    if OUT_CSV.exists():
        out_df = pd.read_csv(OUT_CSV)
        print(f"[cached] {OUT_CSV.relative_to(REPO_ROOT)}")
    else:
        out_df = compute_per_image_p_all_alphas(ood_pred_df, mask_stats_df, target_lab)

    # Stratified split (same seed)
    cal_df, test_df = stratified_split(out_df)
    log.append(f"Stratified cal/test split (seed={SEED}): "
               f"cal N={len(cal_df)}, test N={len(test_df)}")
    log.append("")

    # Per-alpha analysis
    per_alpha = []
    log.append("## Per-alpha test-set metrics")
    log.append(f"{'alpha':>5} | {'AUC':>6} | {'tau_alpha':>9} | "
               f"{'B-recall':>8} {'B-spec':>7} {'B-f1':>6} | "
               f"{'C-recall':>8} {'C-spec':>7} {'C-f1':>6}")
    y_test = test_df.label.values
    for alpha in ALPHAS:
        col = f"p_alpha_{alpha:.2f}"
        p_test = test_df[col].values
        auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan")
        tau_alpha, _cal_recall = find_tau(cal_df, col, TARGET_RECALL)
        mB = metrics_at_tau(p_test, y_test, TAU_BASELINE)
        mC = metrics_at_tau(p_test, y_test, tau_alpha)
        log.append(f"{alpha:>5.2f} | {auc:>6.3f} | {tau_alpha:>9.3f} | "
                   f"{mB['recall']:>8.3f} {mB['spec']:>7.3f} {mB['f1']:>6.3f} | "
                   f"{mC['recall']:>8.3f} {mC['spec']:>7.3f} {mC['f1']:>6.3f}")
        per_alpha.append({
            "alpha": alpha,
            "auc": auc,
            "tau_alpha": tau_alpha,
            "recall_tau_baseline": mB["recall"],
            "spec_tau_baseline":   mB["spec"],
            "f1_tau_baseline":     mB["f1"],
            "recall_tau_alpha":    mC["recall"],
            "spec_tau_alpha":      mC["spec"],
            "f1_tau_alpha":        mC["f1"],
        })
    log.append("")

    # Identify best alpha
    df_sum = pd.DataFrame(per_alpha)
    best_f1_idx  = df_sum.f1_tau_alpha.idxmax()
    best_auc_idx = df_sum.auc.idxmax()
    log.append("## Sweet-spot identification")
    log.append(f"  Best F1 (recalibrated):   alpha={df_sum.alpha[best_f1_idx]:.2f}  "
               f"F1={df_sum.f1_tau_alpha[best_f1_idx]:.3f}")
    log.append(f"  Best AUC:                 alpha={df_sum.alpha[best_auc_idx]:.2f}  "
               f"AUC={df_sum.auc[best_auc_idx]:.3f}")
    log.append("")

    # Trade-off direction
    base = df_sum.iloc[0]
    log.append("## Verdict")
    auc_monotone_down = all(df_sum.auc[i] <= df_sum.auc[i-1] + 1e-6
                             for i in range(1, len(df_sum)))
    if best_f1_idx == 0 and best_auc_idx == 0:
        log.append("  Best operating point is alpha=0 (no LACN). On this OOD cohort, "
                   "LACN at any positive strength erodes more than it shifts. The "
                   "test-time intervention does not improve over Phase 9 baseline.")
    elif auc_monotone_down:
        log.append("  AUC is monotonically decreasing in alpha. No partial-strength "
                   "sweet spot exists; LACN strength controls only the recall/spec "
                   "balance, not discriminative quality.")
    else:
        log.append(f"  Non-monotone trade-off observed. Sweet spot at "
                   f"alpha={df_sum.alpha[best_f1_idx]:.2f} (F1) / "
                   f"alpha={df_sum.alpha[best_auc_idx]:.2f} (AUC).")
    log.append("")

    # Figure
    print("\n=== Generate fig20 ===")
    make_figure(per_alpha)
    log.append("## Files emitted")
    log.append(f"  CSV:     {OUT_CSV.relative_to(REPO_ROOT)}")
    log.append(f"  Figure:  {FIG_PATH.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
