"""
bootstrap_auc_ci.py -- bootstrap 95% CI on the headline AUC.

Loads best_model.pth, runs it over the validation set, and reports
AUC with a bootstrap 95% confidence interval, plus recall and specificity
at the calibrated threshold (passed as argv[2]; default 0.347).

Run:
    python scripts/bootstrap_auc_ci.py [ckpt_path] [threshold]

Defaults:
    ckpt_path = best_model.pth
    threshold = 0.347

Notes:
  * 1000 bootstrap resamples by default -- change `N_BOOT` for more/fewer.
  * Resamples per-image with replacement (the standard for AUC).
  * Expected CI width with N_pos ~ 71 is roughly +/- 0.04 in 95% terms
    (Hanley-McNeil SE ~ 0.02; 95% CI ~ +/- 1.96 * SE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

# Project imports -- adjust if running from a different cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from efficientnet_b0 import (  # noqa: E402
    EfficientNetB0Classifier,
    SkinLesionDataset,
    get_transforms,
    patient_level_split,
    CFG,
)
from trainer import load_and_merge_metadata  # noqa: E402


N_BOOT = 1000
SEED   = 42


def collect_val_probs(ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) over the validation split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_merge_metadata("training_data", "training_data/images")
    pcol = "patient_id" if "patient_id" in df.columns else "isic_id"
    _, val_df, _ = patient_level_split(df, patient_col=pcol)

    ds = SkinLesionDataset(
        val_df["image_path"].tolist(),
        val_df["label"].tolist(),
        get_transforms(CFG, "val"),
    )
    loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().numpy()
            all_probs.append(p)
            all_labels.append(lbls.numpy().astype(int))

    return np.concatenate(all_probs), np.concatenate(all_labels)


def bootstrap_auc(probs: np.ndarray, labels: np.ndarray,
                  n: int = N_BOOT, seed: int = SEED,
                  ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (point_auc, lo, hi)."""
    rng = np.random.default_rng(seed)
    aucs = []
    idx_all = np.arange(len(labels))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        # Skip resamples that have only one class -- would crash roc_auc_score
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return float(roc_auc_score(labels, probs)), \
           float(np.quantile(aucs, alpha)), \
           float(np.quantile(aucs, 1 - alpha))


def main(ckpt_path: str = "best_model.pth", threshold: float = 0.347) -> None:
    print(f"Loading {ckpt_path} and running over val split ...")
    probs, labels = collect_val_probs(ckpt_path)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"Val cohort: {len(labels):,} images  |  pos={n_pos}  neg={n_neg}\n")

    auc, lo, hi = bootstrap_auc(probs, labels)
    print(f"ROC-AUC = {auc:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]  "
          f"({N_BOOT} bootstrap resamples, seed={SEED})")

    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\nAt threshold {threshold}:")
    print(f"  recall      = {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  specificity = {spec:.3f}  ({tn}/{tn+fp})")
    print(f"  confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    thr  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.347
    main(ckpt, thr)
