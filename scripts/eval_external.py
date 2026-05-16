"""
eval_external.py -- Phase 7b.

Out-of-distribution eval: best_model.pth (trained on dermoscopy c=70 + c=212
+ c=249) evaluated on PAD-UFES-20 (clinical smartphone close-ups, c=406).

The training pool is dermoscopy + dermoscopy + dermoscopy. PAD-UFES-20 is
NOT dermoscopy -- different sensor, no immersion fluid, variable lighting,
patient-held framing. The Phase 7c per-collection finding already showed
the model partly learned collection-specific priors; this script measures
whether any of the discriminative signal transfers to a held-out OOD
distribution.

The eval uses the ckpt's calibrated threshold (0.661), but that threshold
was sized for the in-distribution prevalence mix (19.2%). PAD-UFES-20 has
a different prevalence; recall/spec at the in-distribution threshold are
the most honest comparison, not a recalibrated PAD-UFES-20-specific
threshold.

Usage:
    python scripts/eval_external.py [ckpt_path] [dataset_dir]

Defaults: best_model.pth, training_data/eval/pad_ufes_20
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
)


def load_ckpt(ckpt_path: str, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("optimal_threshold")
    return obj, None


def load_padufes(dataset_dir: str) -> pd.DataFrame:
    """
    Load PAD-UFES-20 metadata + resolve image paths. The ISIC mirror
    standardizes diagnosis_1 to {Benign, Malignant, Indeterminate}; we
    filter to {Benign, Malignant} the same way load_and_merge_metadata
    does for the training pool.
    """
    root = Path(dataset_dir)
    meta_csv = root / "metadata.csv"
    if not meta_csv.exists():
        sys.exit(f"[ERROR] {meta_csv} not found")

    df = pd.read_csv(meta_csv, low_memory=False)
    print(f"PAD-UFES-20 raw metadata: {len(df):,} rows")
    print(f"  diagnosis_1 distribution: {df['diagnosis_1'].value_counts(dropna=False).to_dict()}")
    if "image_type" in df.columns:
        print(f"  image_type distribution: {df['image_type'].value_counts(dropna=False).to_dict()}")

    # Filter to binary classes
    df = df[df["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)
    df["label"] = (df["diagnosis_1"] == "Malignant").astype(int)

    # Resolve image paths (PAD-UFES-20 images are jpg per ISIC mirror)
    stem_to_path: dict[str, str] = {}
    for f in root.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            stem_to_path[f.stem] = str(f)
    df["image_path"] = df["isic_id"].map(stem_to_path)
    n_missing = int(df["image_path"].isna().sum())
    if n_missing > 0:
        print(f"  [WARN] {n_missing} rows missing image on disk -- dropping")
    df = df[df["image_path"].notna()].reset_index(drop=True)

    print(f"PAD-UFES-20 eval cohort  : {len(df):,} images  "
          f"(pos={int((df.label==1).sum())} / neg={int((df.label==0).sum())})")
    return df


def main(ckpt_path: str = "best_model.pth",
         dataset_dir: str = "training_data/eval/pad_ufes_20") -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, threshold = load_ckpt(ckpt_path, device)
    if threshold is None:
        threshold = 0.5
        print(f"[WARN] ckpt has no optimal_threshold; using 0.5")

    print(f"ckpt    : {ckpt_path}")
    print(f"device  : {device}")
    print(f"threshold (from ckpt, calibrated on dermoscopy cal cohort): {threshold:.4f}")
    print()

    df = load_padufes(dataset_dir)
    print()

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = SkinLesionDataset(
        df["image_path"].tolist(),
        df["label"].tolist(),
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

    auc    = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    preds  = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1     = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    prev   = float((labels == 1).mean())

    print("PAD-UFES-20 EVAL RESULT")
    print("=" * 78)
    print(f"  N images       : {len(labels):,}")
    print(f"  prevalence     : {prev:.1%}  (pos={int(labels.sum())} / neg={int(len(labels)-labels.sum())})")
    print()
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  PR-AUC         : {pr_auc:.4f}    (random baseline at this prevalence = {prev:.3f})")
    print(f"  PR-AUC lift    : {pr_auc/max(prev,1e-9):.2f}x random")
    print()
    print(f"  At in-distribution threshold {threshold:.3f}:")
    print(f"    recall       : {recall:.3f}  ({tp}/{tp+fn})")
    print(f"    specificity  : {spec:.3f}  ({tn}/{tn+fp})")
    print(f"    F1           : {f1:.3f}")
    print(f"    CM           : TN={tn} FP={fp} FN={fn} TP={tp}")
    print()
    print("Interpretation guide:")
    print("  - AUC >= 0.80: meaningful transfer; framing-aid Eigen-CAM hypothesis")
    print("                 (live heatmap localizes lesion on phone camera) is")
    print("                 supported by the discriminative-quality result.")
    print("  - 0.65 <= AUC < 0.80: partial transfer; model picks up SOME signal")
    print("                 on clinical close-ups but the in-distribution threshold")
    print("                 is miscalibrated for this distribution.")
    print("  - AUC < 0.65: collapse on OOD; modality shortcut dominates the")
    print("                 in-distribution result. Disclose on slide.")
    return 0


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    ddir = sys.argv[2] if len(sys.argv) > 2 else "training_data/eval/pad_ufes_20"
    sys.exit(main(ckpt, ddir))
