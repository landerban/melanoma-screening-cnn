"""
Phase 12 — Lesion-Aware Color Normalization (LACN).

For each image with a reliable lesion mask, generate 4 variants:
  original           untouched
  full_norm          full Lab-mean shift (same as Phase 9 no_colorcast)
  LACN_bg_only       shift only background (lesion preserved)
  lesion_only_norm   shift only lesion (background preserved)

Outputs:
  artifacts/phase12/01_lacn.csv
  artifacts/phase12/01_lacn.log
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
LESION_DIR = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
LESION_STATS = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_mask_stats.csv"
CKPT_PATH  = REPO_ROOT / "best_model.pth"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
OUT_CSV    = REPO_ROOT / "artifacts" / "phase12" / "01_lacn.csv"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase12" / "01_lacn.log"

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


def compute_target_lab_mean():
    df = pd.read_csv(SAMPLE_CSV)
    means = []
    for isic_id in df.isic_id:
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        means.append(lab.reshape(-1, 3).mean(axis=0))
    return np.stack(means, axis=0).mean(axis=0)


def normalize_lab_in_region(img_bgr, mask, target_lab):
    """Shift Lab-mean of pixels inside `mask` (binary, 255 = include)
    to `target_lab`. Pixels outside mask are untouched.

    Returns BGR image."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = mask > 127
    if inside.sum() == 0:
        return img_bgr  # nothing to do
    cur_mean = lab[inside].mean(axis=0)
    shift = target_lab - cur_mean
    lab[inside] += shift
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_lab_full(img_bgr, target_lab):
    full_mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    return normalize_lab_in_region(img_bgr, full_mask, target_lab)


def load_lesion_mask(isic_id):
    p = LESION_DIR / f"{isic_id}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is not None and m.shape != (IMG_SIZE, IMG_SIZE):
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return m


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"[device] {device}")

    target_lab = compute_target_lab_mean()
    print(f"Target Lab mean: {target_lab.round(2)}")

    df = pd.read_csv(SAMPLE_CSV)
    rel_df = pd.read_csv(LESION_STATS)
    reliable_ids = set(rel_df[rel_df.reliable].isic_id)
    print(f"Reliable lesion masks: {len(reliable_ids)}/{len(rel_df)}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LACN"):
        isic_id = row["isic_id"]
        if isic_id not in reliable_ids:
            continue
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        lesion_mask = load_lesion_mask(isic_id)
        if lesion_mask is None:
            continue
        bg_mask = 255 - lesion_mask

        # 4 variants
        p_orig = forward_p(model, img, transform, device)
        p_full = forward_p(model, normalize_lab_full(img, target_lab), transform, device)
        p_lacn = forward_p(model, normalize_lab_in_region(img, bg_mask, target_lab),
                           transform, device)
        p_lesion = forward_p(model, normalize_lab_in_region(img, lesion_mask, target_lab),
                             transform, device)

        rows.append({
            "isic_id": isic_id,
            "source_collection": row["source_collection"],
            "label": row["label"],
            "p_orig":      p_orig,
            "p_full_norm": p_full,
            "p_LACN":      p_lacn,
            "p_lesion_only": p_lesion,
            "dP_full":   p_full - p_orig,
            "dP_LACN":   p_lacn - p_orig,
            "dP_lesion": p_lesion - p_orig,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # --- analysis ---
    def boot_paired_ci(deltas, b=B_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        n = len(deltas)
        boots = np.empty(b)
        for i in range(b):
            idx = rng.integers(0, n, size=n)
            boots[i] = deltas[idx].mean()
        return float(deltas.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    log = [
        "# Phase 12 — Lesion-Aware Color Normalization (LACN)",
        f"Reliable-lesion sample N = {len(out_df)}",
        f"Target Lab mean: {target_lab.round(2)}",
        "",
        "## Counterfactual ΔP per variant",
    ]

    variants = {
        "full_norm (Phase 9 ref)": "dP_full",
        "LACN (bg-only)":          "dP_LACN",
        "lesion-only norm":        "dP_lesion",
    }

    for v_name, col in variants.items():
        delta = out_df[col].values
        mean, lo, hi = boot_paired_ci(delta)
        try:
            w, p = stats.wilcoxon(delta, alternative="two-sided", zero_method="zsplit")
        except ValueError:
            w, p = float("nan"), 1.0
        log.append(f"  {v_name:<28}: mean Δ = {mean:+.4f} [{lo:+.4f}, {hi:+.4f}],  Wilcoxon p = {p:.4g}")

    log.append("")
    log.append("## H14 — LACN (bg-only) reduces magnitude vs full_norm")
    mean_full, lo_full, hi_full = boot_paired_ci(out_df.dP_full.values)
    mean_lacn, lo_lacn, hi_lacn = boot_paired_ci(out_df.dP_LACN.values)
    log.append(f"  full_norm: mean = {mean_full:+.4f} [{lo_full:+.4f}, {hi_full:+.4f}]")
    log.append(f"  LACN     : mean = {mean_lacn:+.4f} [{lo_lacn:+.4f}, {hi_lacn:+.4f}]")
    log.append(f"  |LACN| < |full_norm|? {abs(mean_lacn) < abs(mean_full)}")
    log.append(f"  LACN sign retained (same as full_norm)? {np.sign(mean_lacn) == np.sign(mean_full)}")
    h14 = (abs(mean_lacn) < abs(mean_full)) and (np.sign(mean_lacn) == np.sign(mean_full))
    log.append(f"  H14 supported (smaller magnitude, same sign)? {h14}")

    log.append("")
    log.append("## H15 — lesion-only normalization affects malignants")
    mal_sub = out_df[out_df.label == 1]
    if len(mal_sub) > 0:
        mean_m, lo_m, hi_m = boot_paired_ci(mal_sub.dP_lesion.values)
        log.append(f"  Malignant subset N={len(mal_sub)}: mean Δ_lesion_only = {mean_m:+.4f} [{lo_m:+.4f}, {hi_m:+.4f}]")
        h15 = abs(mean_m) >= 0.02
        log.append(f"  H15 supported (|Δ| ≥ 0.02 on malignants)? {h15}")

    log.append("")
    log.append("## H16 — per-collection LACN ratio")
    per_coll_lacn = {}
    for c in sorted(out_df.source_collection.astype(str).unique()):
        sub = out_df[out_df.source_collection.astype(str) == c]
        m, l, h = boot_paired_ci(sub.dP_LACN.values)
        per_coll_lacn[c] = abs(m)
        log.append(f"  c={c} (N={len(sub)}): LACN mean Δ = {m:+.4f} [{l:+.4f}, {h:+.4f}]")
    ratio = per_coll_lacn.get("70", 0) / per_coll_lacn.get("212", 1e-6)
    log.append(f"  c=70 / c=212 ratio = {ratio:.2f}")
    log.append(f"  H16 supported (ratio > 1.5)? {ratio > 1.5}")

    log.append("")
    log.append("## Files emitted")
    log.append(f"  CSV: {OUT_CSV.relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
