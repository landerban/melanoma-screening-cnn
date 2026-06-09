"""
Phase 10 — Quantitative attention analysis: peak-on-lesion rate
Phase 6 vs Phase 10.

For each of 300 analytical sample images:
  1. Phase 6 CAM (from artifacts/phase9/04_gradcam/)
  2. Phase 10 CAM (re-extract with Phase 10 ckpt)
  3. Lesion mask (from artifacts/phase9/05_lesion_masks/)
  4. Peak = top-5% mass centroid
  5. Peak inside lesion? (binary)

Stats: McNemar's test on paired binary (Phase 6 in-lesion vs Phase 10
in-lesion). Selects best 6 examples where Phase 6 peak is OUT of lesion
and Phase 10 peak moves INTO lesion (clearest visual evidence).

Outputs:
  artifacts/phase10/05_attention_quant.csv
  artifacts/phase10/05_attention_quant.log
  artifacts/phase10/figures/fig13_v2_attention_shift.png  (best 6 examples)
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
from scipy import stats
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from trainer import GradCAM  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

SAMPLE_CSV   = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
P9_CAM_DIR   = REPO_ROOT / "artifacts" / "phase9" / "04_gradcam"
LESION_DIR   = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
LESION_STATS = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_mask_stats.csv"
P10_CKPT     = REPO_ROOT / "artifacts" / "phase10" / "best_model_color_invariant.pth"
IMG_DIR      = REPO_ROOT / "training_data" / "images"

OUT_CSV      = REPO_ROOT / "artifacts" / "phase10" / "05_attention_quant.csv"
LOG_TXT      = REPO_ROOT / "artifacts" / "phase10" / "05_attention_quant.log"
FIG_OUT      = REPO_ROOT / "artifacts" / "phase10" / "figures" / "fig13_v2_attention_shift.png"


def top_mass_centroid(cam: np.ndarray, top_frac: float = 0.05) -> tuple[int, int]:
    """Centroid of top-fraction-mass pixels."""
    threshold = np.quantile(cam, 1 - top_frac)
    mask = cam >= threshold
    if not mask.any():
        return cam.shape[0] // 2, cam.shape[1] // 2
    ys, xs = np.where(mask)
    return int(ys.mean()), int(xs.mean())


def peak_in_lesion(py: int, px: int, lesion_mask: np.ndarray) -> bool:
    """Is the peak pixel inside the lesion mask?"""
    if lesion_mask is None or lesion_mask.sum() == 0:
        return False
    return bool(lesion_mask[py, px] > 127)


def overlay_cam(img_bgr, cam, alpha=0.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img_rgb + alpha * cam_color).astype(np.uint8)


def main():
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SAMPLE_CSV)
    rel = pd.read_csv(LESION_STATS).set_index("isic_id")

    # Phase 10 model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    obj = torch.load(P10_CKPT, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state)
    model.eval()
    gradcam = GradCAM(model)
    transform = get_transforms(CFG, "val")

    print(f"[device] {device}")
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="attention"):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue

        # Skip if lesion mask unreliable
        if isic_id not in rel.index or not rel.loc[isic_id, "reliable"]:
            continue

        # Phase 6 CAM (cached)
        p6_cam_path = P9_CAM_DIR / f"{isic_id}.npy"
        if not p6_cam_path.exists():
            continue
        p6_cam = np.load(p6_cam_path)
        if p6_cam.shape != (IMG_SIZE, IMG_SIZE):
            p6_cam = cv2.resize(p6_cam, (IMG_SIZE, IMG_SIZE))

        # Phase 10 CAM (extract now)
        img = load_image_resized(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transform(Image.fromarray(img_rgb)).to(device)
        p10_cam = gradcam.generate(x)

        # Lesion mask
        lm_path = LESION_DIR / f"{isic_id}.png"
        if not lm_path.exists():
            continue
        lesion_mask = cv2.imread(str(lm_path), cv2.IMREAD_GRAYSCALE)
        if lesion_mask.shape != (IMG_SIZE, IMG_SIZE):
            lesion_mask = cv2.resize(lesion_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Peaks
        p6_y, p6_x = top_mass_centroid(p6_cam)
        p10_y, p10_x = top_mass_centroid(p10_cam)
        p6_in = peak_in_lesion(p6_y, p6_x, lesion_mask)
        p10_in = peak_in_lesion(p10_y, p10_x, lesion_mask)

        rows.append({
            "isic_id": isic_id,
            "source_collection": row["source_collection"],
            "label": row["label"],
            "p6_peak_y": p6_y, "p6_peak_x": p6_x, "p6_in_lesion": int(p6_in),
            "p10_peak_y": p10_y, "p10_peak_x": p10_x, "p10_in_lesion": int(p10_in),
            "moved_into_lesion": int(p10_in and not p6_in),
            "moved_out_of_lesion": int(p6_in and not p10_in),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # --- Stats ---
    p6_rate = out_df.p6_in_lesion.mean()
    p10_rate = out_df.p10_in_lesion.mean()

    # McNemar's test on paired binary
    # b: P6 in, P10 out; c: P6 out, P10 in
    b = int(out_df.moved_out_of_lesion.sum())
    c = int(out_df.moved_into_lesion.sum())
    # McNemar: (|b-c|-1)^2 / (b+c)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) >= 5 else float("nan")
        # exact binomial
        from scipy.stats import binomtest
        p_value = binomtest(min(b, c), b + c, p=0.5).pvalue if (b + c) > 0 else 1.0
    else:
        chi2 = float("nan"); p_value = 1.0

    log = [
        "# Phase 10 attention analysis — peak-on-lesion rate",
        f"N = {len(out_df)} (reliable lesion masks only)",
        "",
        "## Per-model peak-on-lesion rate",
        f"  Phase 6  in-lesion rate: {p6_rate:.1%}",
        f"  Phase 10 in-lesion rate: {p10_rate:.1%}",
        f"  Δ: {p10_rate - p6_rate:+.1%}",
        "",
        "## McNemar's test (paired)",
        f"  b (P6 in → P10 out): {b}",
        f"  c (P6 out → P10 in): {c}",
        f"  χ² (continuity): {chi2:.3f}",
        f"  p (exact binomial): {p_value:.4g}",
        f"  Significant (p < 0.01)? {p_value < 0.01}",
        "",
        "## Per-collection breakdown",
    ]
    for c_name in sorted(out_df.source_collection.astype(str).unique()):
        sub = out_df[out_df.source_collection.astype(str) == c_name]
        r6 = sub.p6_in_lesion.mean()
        r10 = sub.p10_in_lesion.mean()
        log.append(f"  c={c_name}: P6 {r6:.1%} → P10 {r10:.1%} (Δ {r10-r6:+.1%}, N={len(sub)})")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))

    # --- Pick best 6 examples ---
    moved = out_df[out_df.moved_into_lesion == 1].copy()
    # Prefer examples spread across collections
    chosen = []
    for c_name in ["212", "70", "249"]:
        sub = moved[moved.source_collection.astype(str) == c_name]
        if len(sub) >= 2:
            picks = sub.sample(n=2, random_state=42)
        else:
            picks = sub
        chosen.extend(picks.to_dict("records"))
    chosen = chosen[:6]

    if len(chosen) < 6:
        # Fill in with any moved
        extras = moved[~moved.isic_id.isin([c["isic_id"] for c in chosen])]
        for _, row in extras.head(6 - len(chosen)).iterrows():
            chosen.append(row.to_dict())

    print(f"\nSelected {len(chosen)} best examples (peak moved INTO lesion)")

    # --- Build figure ---
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)

    coll_pretty = {"212": "HAM10000", "70": "SIIM-2020", "249": "BCN20000"}

    for i, rec in enumerate(chosen):
        isic_id = rec["isic_id"]
        img = load_image_resized(IMG_DIR / f"{isic_id}.jpg")
        p6_cam = np.load(P9_CAM_DIR / f"{isic_id}.npy")
        if p6_cam.shape != (IMG_SIZE, IMG_SIZE):
            p6_cam = cv2.resize(p6_cam, (IMG_SIZE, IMG_SIZE))
        # Recompute Phase 10 CAM
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transform(Image.fromarray(img_rgb)).to(device)
        p10_cam = gradcam.generate(x)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(
            f"{isic_id}\n{coll_pretty[str(rec['source_collection'])]}, "
            f"{'malig' if rec['label']==1 else 'benign'}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_cam(img, p6_cam))
        axes[i, 1].set_title("Phase 6 — peak outside lesion", fontsize=10, color="darkred")
        axes[i, 1].plot(rec["p6_peak_x"], rec["p6_peak_y"], "wx", markersize=14, markeredgewidth=2)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_cam(img, p10_cam))
        axes[i, 2].set_title("Phase 10 — peak inside lesion", fontsize=10, color="darkgreen")
        axes[i, 2].plot(rec["p10_peak_x"], rec["p10_peak_y"], "wx", markersize=14, markeredgewidth=2)
        axes[i, 2].axis("off")

    plt.suptitle(
        f"fig13 v2 — Best examples where Phase 10 attention moved INTO lesion\n"
        f"Quantitative: Phase 6 in-lesion rate {p6_rate:.0%} → Phase 10 {p10_rate:.0%} "
        f"(McNemar p={p_value:.3g})",
        fontsize=12, y=0.998,
    )
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {FIG_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
