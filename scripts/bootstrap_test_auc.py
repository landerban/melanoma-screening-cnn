"""
bootstrap_test_auc.py -- Phase 8 prep.

Bootstrap 95% CI on the held-out test AUC for the retrained model. Same
methodology as Phase 3's bootstrap_auc_ci.py (1000 resamples, seed=42),
but on the corrected 4-way split's test cohort using the new rich
checkpoint format.

Reports:
  * Aggregate test AUC + 95% CI
  * Per-collection AUC + 95% CI (c=70, c=212, c=249)
  * Recall + specificity at the calibrated threshold (point estimates;
    bootstrap CIs on operating-point metrics are less informative than
    on AUC since the threshold itself is fixed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from efficientnet_b0 import (  # noqa: E402
    CFG,
    EfficientNetB0Classifier,
    SkinLesionDataset,
    get_transforms,
    patient_level_split,
)
from trainer import load_and_merge_metadata  # noqa: E402


N_BOOT = 1000
SEED   = 42


def load_ckpt(ckpt_path: str, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("optimal_threshold")
    return obj, None


def bootstrap_auc(probs: np.ndarray, labels: np.ndarray,
                  n: int = N_BOOT, seed: int = SEED,
                  ci: float = 0.95) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    idx_all = np.arange(len(labels))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    point = float(roc_auc_score(labels, probs))
    return point, float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


def main(ckpt_path: str = "best_model.pth") -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, threshold = load_ckpt(ckpt_path, device)
    if threshold is None:
        threshold = 0.5
        print(f"[WARN] ckpt has no optimal_threshold; using 0.5")

    print(f"ckpt    : {ckpt_path}")
    print(f"device  : {device}")
    print(f"threshold (from ckpt): {threshold:.4f}")
    print(f"N_BOOT  : {N_BOOT}")
    print(f"seed    : {SEED}")
    print()

    df = load_and_merge_metadata("training_data", "training_data/images")
    _, _, _, test_df = patient_level_split(df, patient_col="effective_patient_id")
    print(f"Test cohort: {len(test_df):,} images  "
          f"(pos={int((test_df.label==1).sum())} / neg={int((test_df.label==0).sum())})")
    print(f"  by source: {test_df['source_collection'].value_counts().to_dict()}")
    print()

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = SkinLesionDataset(
        test_df["image_path"].tolist(),
        test_df["label"].tolist(),
        get_transforms(CFG, "val"),
    )
    loader = DataLoader(
        ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    probs_l, labels_l = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().numpy()
            probs_l.append(p)
            labels_l.append(lbls.numpy().astype(int))
    probs  = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    sources = test_df["source_collection"].to_numpy()
    assert len(probs) == len(test_df)

    # Aggregate AUC + CI
    auc, lo, hi = bootstrap_auc(probs, labels)
    pr_auc = average_precision_score(labels, probs)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("AGGREGATE TEST  (bootstrap N=1000, seed=42)")
    print("=" * 78)
    print(f"  N images        : {len(labels):,}")
    print(f"  prevalence      : {int(labels.sum()):,}/{len(labels):,} ({labels.mean():.1%})")
    print(f"  ROC-AUC         : {auc:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]  half-width {(hi-lo)/2:.4f}")
    print(f"  PR-AUC          : {pr_auc:.4f}")
    print(f"  recall (@{threshold:.3f}): {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  specificity     : {spec:.3f}  ({tn}/{tn+fp})")
    print(f"  CM              : TN={tn} FP={fp} FN={fn} TP={tp}")
    print()

    # Per-collection AUC + CI
    print("PER-COLLECTION  (bootstrap N=1000, seed=42)")
    print("=" * 78)
    for cid in sorted(np.unique(sources)):
        mask = sources == cid
        sub_labels = labels[mask]
        sub_probs  = probs[mask]
        n_sub = len(sub_labels)
        n_pos = int(sub_labels.sum())
        n_neg = n_sub - n_pos
        if n_pos < 2 or n_neg < 2:
            print(f"  c={cid:<5}: N={n_sub:,}  (insufficient class diversity)")
            continue
        a, l, h = bootstrap_auc(sub_probs, sub_labels)
        prev = sub_labels.mean()
        print(f"  c={cid:<5}: N={n_sub:>5,}  pos={n_pos:>4,} ({prev:>5.1%})  "
              f"AUC={a:.4f}  [95% CI: {l:.4f}, {h:.4f}]  half-width {(h-l)/2:.4f}")
    print()
    print("Use the aggregate CI for the slide-7 headline. The per-collection CIs")
    print("show whether the 0.87-0.92 within-collection range is statistically")
    print("stable -- non-overlapping CIs indicate genuinely different per-collection")
    print("quality; overlapping CIs say the apparent spread is within noise.")
    return 0


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    sys.exit(main(ckpt))
