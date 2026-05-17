"""
eval_test_set.py -- Phase 7a.

Save/load sanity check: loads best_model.pth from disk and re-runs the
held-out test eval. The result should reproduce the test_metrics that
Trainer._run already saved at the end of training (AUC 0.9513, etc.).

If it doesn't, something is wrong with the rich-checkpoint round trip or
the build_ckpt_dict serialization.

Usage:
    python scripts/eval_test_set.py [ckpt_path]

Defaults: best_model.pth
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
        return (
            obj["state_dict"],
            obj.get("optimal_threshold"),
            obj.get("test_metrics"),
        )
    return obj, None, None


def main(ckpt_path: str = "best_model.pth") -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, threshold, frozen_test_metrics = load_ckpt(ckpt_path, device)
    if threshold is None:
        threshold = 0.5
        print(f"[WARN] ckpt has no optimal_threshold; using 0.5")

    print(f"ckpt    : {ckpt_path}")
    print(f"device  : {device}")
    print(f"threshold (from ckpt): {threshold:.4f}")
    print()

    df = load_and_merge_metadata("training_data", "training_data/images")
    _, _, _, test_df = patient_level_split(df, patient_col="effective_patient_id")
    print(f"Test cohort: {len(test_df):,} images  "
          f"(pos={int((test_df.label==1).sum())} / neg={int((test_df.label==0).sum())})")
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

    auc    = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    preds  = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1     = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    print("RE-RUN TEST EVAL (from saved checkpoint)")
    print("=" * 78)
    print(f"  AUC          : {auc:.4f}")
    print(f"  PR-AUC       : {pr_auc:.4f}")
    print(f"  recall       : {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  specificity  : {spec:.3f}  ({tn}/{tn+fp})")
    print(f"  F1           : {f1:.3f}")
    print(f"  CM           : TN={tn} FP={fp} FN={fn} TP={tp}")
    print()

    if frozen_test_metrics is not None:
        print("COMPARE vs frozen test_metrics stored in ckpt dict:")
        print("=" * 78)
        items = [
            ("auc",         auc),
            ("pr_auc",      pr_auc),
            ("recall",      recall),
            ("specificity", spec),
            ("tn",          int(tn)),
            ("fp",          int(fp)),
            ("fn",          int(fn)),
            ("tp",          int(tp)),
        ]
        ok = True
        for key, computed in items:
            frozen = frozen_test_metrics.get(key)
            if frozen is None:
                print(f"  {key:<14}: computed={computed}  frozen=(missing)")
                continue
            if isinstance(computed, float):
                diff = abs(computed - float(frozen))
                tag = "OK" if diff < 1e-3 else "MISMATCH"
                if tag == "MISMATCH":
                    ok = False
                print(f"  {key:<14}: computed={computed:.6f}  frozen={frozen:.6f}  diff={diff:.2e}  [{tag}]")
            else:
                tag = "OK" if computed == frozen else "MISMATCH"
                if tag == "MISMATCH":
                    ok = False
                print(f"  {key:<14}: computed={computed}  frozen={frozen}  [{tag}]")
        print()
        if ok:
            print("  All metrics match within tolerance. Checkpoint round-trip is honest.")
        else:
            print("  WARN: mismatches detected. Investigate save/load fidelity.")
            return 1
    return 0


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    sys.exit(main(ckpt))
