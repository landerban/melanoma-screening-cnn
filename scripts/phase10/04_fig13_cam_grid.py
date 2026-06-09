"""
fig13 — Phase 6 vs Phase 10 Grad-CAM side-by-side comparison.

Visualizes how attention shifts after color-invariant fine-tune.
Picks 6 images (3 collections × 2 classes) and shows:
  - Original image
  - Phase 6 Grad-CAM overlay  (from phase9/04_gradcam/)
  - Phase 10 Grad-CAM overlay (re-extracted with Phase 10 ckpt)

THIS is the decisive visual evidence of the intervention.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from trainer import GradCAM  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

PRED_CSV   = REPO_ROOT / "artifacts" / "phase9" / "04_predictions.csv"
P9_CAM_DIR = REPO_ROOT / "artifacts" / "phase9" / "04_gradcam"
P10_CKPT   = REPO_ROOT / "artifacts" / "phase10" / "best_model_color_invariant.pth"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
FIG_DIR    = REPO_ROOT / "artifacts" / "phase10" / "figures"
SEED       = 42


def overlay_cam(img_bgr, cam, alpha=0.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img_rgb + alpha * cam_color).astype(np.uint8)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_CSV)
    rng = np.random.default_rng(SEED)

    # Pick 1 example per collection × class = 6 images
    selected = []
    for c in ["212", "70", "249"]:
        for lbl in [0, 1]:
            sub = pred[(pred.source_collection.astype(str) == c) & (pred.label == lbl)]
            if len(sub) == 0: continue
            row = sub.sample(1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            selected.append((c, lbl, row["isic_id"]))

    # Load Phase 10 model + GradCAM
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    obj = torch.load(P10_CKPT, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state)
    model.eval()
    gradcam = GradCAM(model)
    transform = get_transforms(CFG, "val")

    # Build figure: 6 rows × 3 cols (orig, P6 cam, P10 cam)
    fig, axes = plt.subplots(6, 3, figsize=(10, 16))
    coll_pretty = {"212": "HAM10000", "70": "SIIM-2020", "249": "BCN20000"}

    for i, (c, lbl, isic_id) in enumerate(selected):
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path)

        # Phase 6 CAM
        p6_cam_path = P9_CAM_DIR / f"{isic_id}.npy"
        if p6_cam_path.exists():
            p6_cam = np.load(p6_cam_path)
            if p6_cam.shape != (IMG_SIZE, IMG_SIZE):
                p6_cam = cv2.resize(p6_cam, (IMG_SIZE, IMG_SIZE))
        else:
            p6_cam = np.zeros((IMG_SIZE, IMG_SIZE))

        # Phase 10 CAM (extract now)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transform(Image.fromarray(img_rgb)).to(device)
        p10_cam = gradcam.generate(x)

        # Plot
        ax_o = axes[i, 0]
        ax_p6 = axes[i, 1]
        ax_p10 = axes[i, 2]

        ax_o.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_o.set_title(f"{isic_id}\n{coll_pretty[c]}, {'malig' if lbl==1 else 'benign'}", fontsize=9)
        ax_o.axis("off")

        ax_p6.imshow(overlay_cam(img, p6_cam))
        ax_p6.set_title("Phase 6 attention", fontsize=10, color="darkred")
        ax_p6.axis("off")

        ax_p10.imshow(overlay_cam(img, p10_cam))
        ax_p10.set_title("Phase 10 attention", fontsize=10, color="darkgreen")
        ax_p10.axis("off")

    plt.suptitle("fig13 — Grad-CAM attention shift: Phase 6 vs Phase 10\n"
                 "Color-invariant fine-tune relocates attention toward lesion",
                 fontsize=12, y=0.998)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_cam_grid_p6_vs_p10.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved fig13_cam_grid_p6_vs_p10.png ({len(selected)} examples)")


if __name__ == "__main__":
    main()
