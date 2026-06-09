"""
Phase 9 / Step 3g + 3o — Model inference on the 300-image analytical
sample. Produces:

  artifacts/phase9/04_predictions.csv
    isic_id, source_collection, label, p_malig (original image)

  artifacts/phase9/04_gradcam/<isic_id>.npy
    (384, 384) float32 CAM heatmap normalized to [0, 1]

  artifacts/phase9/04_inference.log
    Per-collection AUC + per-class mean P_malig for sanity check.

The CAM is computed via the existing efficientnet_b0.GradCAM helper.
Target layer = last conv (efficientnet's backbone[-2], the 1280-channel
12x12 feature map before GAP).

Run from project root (in venv):
    .venv/bin/python scripts/phase9/04_model_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from efficientnet_b0 import (  # noqa: E402
    CFG, EfficientNetB0Classifier, get_transforms,
)
from trainer import GradCAM  # noqa: E402

SAMPLE_CSV  = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
IMG_DIR     = REPO_ROOT / "training_data" / "images"
CKPT_PATH   = REPO_ROOT / "best_model.pth"
OUT_CSV     = REPO_ROOT / "artifacts" / "phase9" / "04_predictions.csv"
CAM_DIR     = REPO_ROOT / "artifacts" / "phase9" / "04_gradcam"
LOG_TXT     = REPO_ROOT / "artifacts" / "phase9" / "04_inference.log"


def pick_device() -> torch.device:
    """Prefer MPS on Apple Silicon; fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ckpt(ckpt_path, device):
    """Load legacy state_dict or rich-dict checkpoint."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return (
            obj["state_dict"],
            obj.get("optimal_threshold"),
        )
    return obj, None


def main():
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    state_dict, threshold = load_ckpt(CKPT_PATH, device)
    if threshold is None:
        threshold = 0.661  # README/audit-known calibrated threshold
        print(f"[note] ckpt is legacy format; using README-known threshold {threshold}")

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    transform = get_transforms(CFG, "val")
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Sample: {len(df)} images")

    # GradCAM hooks backbone[-1] (last MBConv block) per trainer.py:97.
    # Forward stays in eval(); generate() does an extra backward pass
    # internally and zero_grads. Autograd must be enabled (default).
    gradcam = GradCAM(model)

    rows = []
    probs = np.zeros(len(df), dtype=np.float32)
    labels = np.zeros(len(df), dtype=np.int32)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="forward")):
        isic_id = row["isic_id"]
        label = int(row["label"])
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            print(f"[skip] missing {isic_id}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # Forward pass (no_grad) for P(malig)
        with torch.no_grad():
            logit = model(x).squeeze()
            p_malig = float(torch.sigmoid(logit))

        # Grad-CAM: generate() takes (C,H,W) no batch dim, returns
        # (H,W) numpy normalized [0,1]
        cam = gradcam.generate(x.squeeze(0))

        # Save artifacts
        np.save(CAM_DIR / f"{isic_id}.npy", cam.astype(np.float32))

        rows.append({
            "isic_id":           isic_id,
            "source_collection": row["source_collection"],
            "label":             label,
            "p_malig":           p_malig,
            "threshold":         threshold,
            "predicted_malig":   int(p_malig >= threshold),
        })
        probs[i] = p_malig
        labels[i] = label

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # Sanity check: AUC on the analytical sample (50/50 prevalence, so
    # AUC is well-defined per cell and overall)
    log = ["# Phase 9 Step 3g + 3o — model inference summary",
           f"Sample N = {len(out_df)}",
           f"Device   = {device}",
           f"Threshold (used for predicted_malig) = {threshold}",
           ""]

    overall_auc = roc_auc_score(out_df.label, out_df.p_malig)
    log.append(f"## Overall analytical-sample AUC: {overall_auc:.4f}")
    log.append("")
    log.append("(Note: this AUC is NOT comparable to Phase 7a's 0.9513 — "
               "the sample is 50/50 balanced rather than 19.2% prevalent. "
               "It is a model-functioning sanity check: AUC > 0.85 = ckpt valid.)")
    log.append("")

    log.append("## Per-collection AUC (on N=100 each)")
    for c in sorted(out_df.source_collection.unique()):
        sub = out_df[out_df.source_collection == c]
        auc = roc_auc_score(sub.label, sub.p_malig)
        log.append(f"  c={c}: N={len(sub)} AUC={auc:.4f}")
    log.append("")

    log.append("## Per-collection per-class mean P_malig")
    for c in sorted(out_df.source_collection.unique()):
        for lbl in [0, 1]:
            sub = out_df[(out_df.source_collection == c) & (out_df.label == lbl)]
            mean_p = sub.p_malig.mean()
            lbl_str = "benign" if lbl == 0 else "malignant"
            log.append(f"  c={c} {lbl_str:9s}: n={len(sub)} mean P_malig={mean_p:.4f}")
    log.append("")

    log.append("## Inputs to H3 (cross-collection separability — "
               "between-class mean gap)")
    for c in sorted(out_df.source_collection.unique()):
        sub = out_df[out_df.source_collection == c]
        mean_b = sub[sub.label == 0].p_malig.mean()
        mean_m = sub[sub.label == 1].p_malig.mean()
        log.append(f"  c={c}: mean(malig) - mean(benign) = "
                   f"{mean_m - mean_b:.4f}")
    log.append("")

    log.append("## Files emitted")
    log.append(f"  predictions: {OUT_CSV.relative_to(REPO_ROOT)}")
    log.append(f"  CAMs:        {CAM_DIR.relative_to(REPO_ROOT)} "
               f"({len(list(CAM_DIR.glob('*.npy')))} files)")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
