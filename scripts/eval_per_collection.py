"""
eval_per_collection.py -- Phase 7c.

Per-source-collection AUC breakdown on the held-out test split. Answers
audit weakness W8: does the headline test AUC reflect uniform classifier
quality across the three training collections, or is one collection
(likely c=249 BCN20000 with its 47% malignant prevalence) dominating
the result?

If all three per-collection AUCs cluster around the headline number, the
0.9513 result is real cross-modality. If c=249 sits at 0.99 while
c=70/c=212 sit at ~0.85, the headline is dominated by one collection
and the slide needs to disclose this.

Reuses load_and_merge_metadata + patient_level_split with seed=42 so the
test split is identical to the one Phase 6 reported on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from efficientnet_b0 import (  # noqa: E402
    CFG,
    EfficientNetB0Classifier,
    SkinLesionDataset,
    get_transforms,
    patient_level_split,
)
from trainer import load_and_merge_metadata  # noqa: E402


def load_ckpt(ckpt_path: str, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("optimal_threshold")
    return obj, None


def report_row(label: str, labels: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    n = int(len(labels))
    n_pos = int((labels == 1).sum())
    n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        print(f"  {label:<22}  N={n:>6,}  pos={n_pos:>5}  (insufficient class diversity)")
        return
    auc    = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    preds  = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prev   = n_pos / n
    print(f"  {label:<22}  N={n:>6,}  pos={n_pos:>5,} ({prev:>5.1%})  "
          f"AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  recall={recall:.3f}  spec={spec:.3f}")


def main(ckpt_path: str = "best_model.pth") -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, threshold = load_ckpt(ckpt_path, device)
    if threshold is None:
        threshold = 0.5
        print(f"[WARN] ckpt has no optimal_threshold; using 0.5")

    print(f"ckpt    : {ckpt_path}")
    print(f"device  : {device}")
    print(f"threshold (from ckpt): {threshold:.4f}")
    print()

    # Same split as Phase 6 / 7a / Phase 5 / etc.
    df = load_and_merge_metadata("training_data", "training_data/images")
    _, _, _, test_df = patient_level_split(df, patient_col="effective_patient_id")
    print(f"Test cohort: {len(test_df):,} images  "
          f"({int((test_df.label==1).sum())} pos / {int((test_df.label==0).sum())} neg)")
    print(f"  by source: {test_df['source_collection'].value_counts().to_dict()}")
    print()

    # Build model + run inference once over the full test set
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
    assert len(probs) == len(test_df), \
        f"length mismatch probs={len(probs)} vs test_df={len(test_df)}"

    print(f"PER-COLLECTION BREAKDOWN  (threshold = {threshold:.3f})")
    print("=" * 100)
    report_row("Overall (test)", labels, probs, threshold)
    print()
    for cid in sorted(np.unique(sources)):
        mask = sources == cid
        report_row(f"c={cid}", labels[mask], probs[mask], threshold)

    # Interpretation hint
    print()
    print("Interpretation guide:")
    print("  - If per-collection AUCs cluster around the headline (~+/- 0.03),")
    print("    the test AUC reflects uniform quality across modalities (W8 clean).")
    print("  - If c=249 sits meaningfully higher than c=70/c=212 (gap > 0.05),")
    print("    BCN20000's higher malignant prevalence is doing more work than")
    print("    the model's discriminative quality -- slide must disclose.")
    return 0


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    sys.exit(main(ckpt))
