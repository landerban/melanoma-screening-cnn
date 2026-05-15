"""
eval_oldckpt_no_c249.py -- Ghost evaluation.

Runs the existing best_model.pth on the same Phase 3 val split, but
filtered to c=70 + c=212 rows only (i.e. excluding BCN20000 / c=249).

Rationale: bootstrap_auc_ci.py showed AUC 0.6243 on the full new val
(mixed collections). That number alone can't disambiguate two stories:

  (i) c=390 wasn't carrying the discriminative signal -- supporting
      the c=390 drop decision (audit weakness W8).
  (ii) c=249 is genuinely out-of-distribution for an old checkpoint
       trained on {c=70, c=212, c=390}, and that's why the AUC dropped.

This script restricts val to the two collections most likely overlapping
with the old checkpoint's training distribution. If the resulting AUC
lands near the original CLAUDE.md headline (0.928), story (i) is
supported. If it lands meaningfully lower, story (ii) cannot be ruled
out and the c=390 drop story has to stand on a-priori grounds alone.

Run from project root:
    python scripts/eval_oldckpt_no_c249.py [ckpt] [threshold]

Defaults: best_model.pth, threshold 0.347
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

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
SEED = 42
COLLECTION_RE = re.compile(r"metadata_c(\d+)\.csv$")


def tag_source_collection(df: pd.DataFrame, training_data_dir: str = "training_data") -> pd.DataFrame:
    """Map isic_id -> source_collection by reading each per-collection CSV."""
    root = Path(training_data_dir)
    isic_to_source: dict[str, str] = {}
    for p in sorted(root.glob("metadata_c*.csv")):
        m = COLLECTION_RE.search(p.name)
        if not m:
            continue
        cid = m.group(1)
        c_df = pd.read_csv(p, low_memory=False, usecols=["isic_id"])
        for iid in c_df["isic_id"]:
            isic_to_source[iid] = cid
    df["source_collection"] = df["isic_id"].map(isic_to_source)
    return df


def main(ckpt_path: str = "best_model.pth", threshold: float = 0.347) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device   : {device}")
    print(f"ckpt     : {ckpt_path}")
    print(f"threshold: {threshold}")
    print()

    df = load_and_merge_metadata("training_data", "training_data/images")
    df = tag_source_collection(df, "training_data")
    pcol = "patient_id" if "patient_id" in df.columns else "isic_id"

    # Same split logic as bootstrap_auc_ci.py -- same seed, same broken patient_id
    # column, so the val cohort matches Phase 3 before filtering.
    _, val_df, _ = patient_level_split(df, patient_col=pcol)

    full_n = len(val_df)
    full_pos = int((val_df["label"] == 1).sum())
    print(f"Full val cohort (matches Phase 3)        : {full_n:,} (pos={full_pos})")

    # Source-collection distribution in val
    src_counts = val_df["source_collection"].value_counts(dropna=False).to_dict()
    print(f"  source breakdown                       : {src_counts}")

    # Filter to c=70 + c=212 -- the collections the old checkpoint trained on
    # (alongside the dropped c=390).
    keep_mask = val_df["source_collection"].isin({"70", "212"})
    ghost_val = val_df.loc[keep_mask].reset_index(drop=True)
    n_dropped = int((~keep_mask).sum())
    n_pos = int((ghost_val["label"] == 1).sum())
    n_neg = int(len(ghost_val) - n_pos)
    print(f"Ghost val (c=70 + c=212 only)            : {len(ghost_val):,} (pos={n_pos}, neg={n_neg})")
    print(f"  dropped (c=249 + any untagged)         : {n_dropped:,}")
    print()

    if n_pos < 2 or n_neg < 2:
        sys.exit("[ERROR] ghost val has insufficient class diversity for AUC -- abort.")

    # ---- Inference ----------------------------------------------------
    ds = SkinLesionDataset(
        ghost_val["image_path"].tolist(),
        ghost_val["label"].tolist(),
        get_transforms(CFG, "val"),
    )
    loader = DataLoader(
        ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    probs_l, labels_l = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().numpy()
            probs_l.append(p)
            labels_l.append(lbls.numpy().astype(int))
    probs = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)

    # ---- Bootstrap CI -------------------------------------------------
    rng = np.random.default_rng(SEED)
    aucs = []
    idx_all = np.arange(len(labels))
    for _ in range(N_BOOT):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    aucs = np.array(aucs)
    auc_pt = float(roc_auc_score(labels, probs))
    lo, hi = float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))

    # ---- Threshold metrics --------------------------------------------
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"ROC-AUC = {auc_pt:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]  ({N_BOOT} bootstraps, seed={SEED})")
    print(f"At threshold {threshold}:")
    print(f"  recall      = {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  specificity = {spec:.3f}  ({tn}/{tn+fp})")
    print(f"  CM: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print()
    print("Reference points for interpretation:")
    print(f"  full new val (c=70+212+249)   AUC 0.6243 [0.5561, 0.6866]   (Phase 3 bootstrap)")
    print(f"  CLAUDE.md old-data headline   AUC 0.928                       (audit doc)")
    print()
    print("Interpretation rule:")
    print("  - if ghost AUC lands near 0.928 -> c=390 was not carrying the signal;")
    print("    the c=390 drop has empirical support (audit weakness W8).")
    print("  - if ghost AUC lands meaningfully lower -> can't disambiguate c=390-was-load-bearing")
    print("    from c=249-OOD; c=390 drop stands on a-priori grounds only.")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.347
    main(ckpt, thr)
