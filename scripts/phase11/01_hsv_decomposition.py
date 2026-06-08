"""
Phase 11 — Decompose the Phase 9 color-cast shortcut by HSV channel.

For each image, generate 4 counterfactual variants:
  no_H   : hue replaced with sample-mean hue (S, V untouched)
  no_S   : saturation replaced with sample-mean S
  no_V   : value replaced with sample-mean V
  no_HSV : all three normalized (joint check vs Phase 9's Lab-shift)

Forward each through the Phase 6 ckpt and record P_malig. Compare to
the Phase 9-derived original P_malig.

Outputs:
  artifacts/phase11/01_hsv_decomposition.csv
  artifacts/phase11/01_hsv_decomposition.log

Run:
    .venv/bin/python scripts/phase11/01_hsv_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
DET_CSV    = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.csv"
CKPT_PATH  = REPO_ROOT / "best_model.pth"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
OUT_CSV    = REPO_ROOT / "artifacts" / "phase11" / "01_hsv_decomposition.csv"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase11" / "01_hsv_decomposition.log"

SEED = 42
B_BOOT = 5000


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


def compute_target_hsv_mean():
    """Mean HSV across the analytical sample (anchor for normalization)."""
    df = pd.read_csv(SAMPLE_CSV)
    means = []
    for isic_id in df.isic_id:
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        means.append(hsv.reshape(-1, 3).mean(axis=0))
    return np.stack(means, axis=0).mean(axis=0)


def normalize_channel(img_bgr, channel_idx, target_value):
    """Replace one HSV channel's mean with target while keeping per-pixel
    variation. variant = image's HSV - per-image-mean(channel) + target."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    cur = hsv[:, :, channel_idx].mean()
    hsv[:, :, channel_idx] = hsv[:, :, channel_idx] + (target_value - cur)
    if channel_idx == 0:  # hue wraps at 180
        hsv[:, :, 0] = hsv[:, :, 0] % 180
    else:
        hsv[:, :, channel_idx] = np.clip(hsv[:, :, channel_idx], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize_all(img_bgr, target_hsv):
    out = img_bgr.copy()
    for ch in range(3):
        out = normalize_channel(out, ch, target_hsv[ch])
    return out


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"[device] {device}")

    target_hsv = compute_target_hsv_mean()
    print(f"Target HSV mean: H={target_hsv[0]:.2f}, S={target_hsv[1]:.2f}, V={target_hsv[2]:.2f}")

    df = pd.read_csv(SAMPLE_CSV)
    det = pd.read_csv(DET_CSV).set_index("isic_id")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="HSV decomp"):
        isic_id = row["isic_id"]
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)

        p_orig = forward_p(model, img, transform, device)
        p_noH = forward_p(model, normalize_channel(img, 0, target_hsv[0]), transform, device)
        p_noS = forward_p(model, normalize_channel(img, 1, target_hsv[1]), transform, device)
        p_noV = forward_p(model, normalize_channel(img, 2, target_hsv[2]), transform, device)
        p_noAll = forward_p(model, normalize_all(img, target_hsv), transform, device)

        rows.append({
            "isic_id": isic_id,
            "source_collection": row["source_collection"],
            "label": row["label"],
            "color_cast_present": int(det.loc[isic_id]["color_cast_present"]),
            "p_orig": p_orig,
            "p_noH":  p_noH,
            "p_noS":  p_noS,
            "p_noV":  p_noV,
            "p_noHSV": p_noAll,
            "dP_H": p_noH - p_orig,
            "dP_S": p_noS - p_orig,
            "dP_V": p_noV - p_orig,
            "dP_HSV": p_noAll - p_orig,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # ---- Analysis ----
    sub = out_df[out_df.color_cast_present == 1]
    log = [
        "# Phase 11 — HSV channel decomposition of color-cast shortcut",
        f"Sample N = {len(out_df)} (color-cast present: N={len(sub)})",
        f"Target HSV mean: H={target_hsv[0]:.2f}, S={target_hsv[1]:.2f}, V={target_hsv[2]:.2f}",
        "",
        "## Marginal per-channel ΔP (color-cast-present subset)",
    ]

    def boot_paired_ci(deltas, b=B_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        n = len(deltas)
        boots = np.empty(b)
        for i in range(b):
            idx = rng.integers(0, n, size=n)
            boots[i] = deltas[idx].mean()
        return float(deltas.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    chan_results = {}
    for ch_name, col in [("H (hue)", "dP_H"),
                          ("S (saturation)", "dP_S"),
                          ("V (value)", "dP_V"),
                          ("HSV (all)", "dP_HSV")]:
        delta = sub[col].values
        mean, lo, hi = boot_paired_ci(delta)
        try:
            w, p = stats.wilcoxon(delta, alternative="two-sided", zero_method="zsplit")
        except ValueError:
            w, p = float("nan"), 1.0
        log.append(f"  {ch_name:<18}: mean Δ = {mean:+.4f} [{lo:+.4f}, {hi:+.4f}],  Wilcoxon p = {p:.4g}")
        chan_results[ch_name] = mean

    log.append("")
    log.append("## H11 — channel dominance")
    abs_means = {k: abs(v) for k, v in chan_results.items() if k != "HSV (all)"}
    joint = abs(chan_results["HSV (all)"])
    dominant_ch, dominant_val = max(abs_means.items(), key=lambda x: x[1])
    frac = dominant_val / joint if joint > 0 else float("nan")
    log.append(f"  Dominant channel: {dominant_ch} (|Δ|={dominant_val:.4f})")
    log.append(f"  Joint |Δ|={joint:.4f}; dominance fraction = {frac:.1%}")
    h11 = frac >= 0.60
    log.append(f"  H11 supported (one channel ≥ 60% of joint)? {h11}")
    log.append("")

    log.append("## H12 — additivity")
    sum_abs = sum(abs_means.values())
    discrepancy = abs(sum_abs - joint) / joint if joint > 0 else float("nan")
    log.append(f"  Σ|Δ_c| = {sum_abs:.4f}, |Δ_HSV| = {joint:.4f}")
    log.append(f"  Discrepancy = {discrepancy:.1%}")
    h12 = discrepancy <= 0.20
    log.append(f"  H12 supported (additivity within ±20%)? {h12}")
    log.append("")

    log.append("## H13 — per-collection channel profile")
    for c in sorted(sub.source_collection.astype(str).unique()):
        ssub = sub[sub.source_collection.astype(str) == c]
        if len(ssub) == 0:
            continue
        log.append(f"\n  c={c} (N={len(ssub)}):")
        for ch_name, col in [("H", "dP_H"), ("S", "dP_S"),
                              ("V", "dP_V"), ("HSV all", "dP_HSV")]:
            log.append(f"    {ch_name:<8}: mean Δ = {ssub[col].mean():+.4f}")

    log.append("")
    log.append("## Files emitted")
    log.append(f"  CSV: {OUT_CSV.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
