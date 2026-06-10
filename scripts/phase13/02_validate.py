"""
Phase 13 / Step 2 -- Validate the LAJ + LACN-Consistency fine-tuned ckpt.

Compares Phase 13 against Phase 6 (base) and Phase 10 (hue jitter only)
on:
  - Per-collection balanced AUC (matches Phase 9 PF#1)
  - Color-cast counterfactual delta P (matches Phase 9 PF#2)
  - OOD AUC (PAD-UFES-20)
  - OOD AUC with LACN applied (the test Phase 12 ran)

Inputs:
  Phase 13 ckpt: artifacts/phase13/best_model_laj_consistency.pth
  Phase 9 analytical sample: 300 images already cached locally
  Phase 9 counterfactual masks: artifacts/phase9/03_shortcut_masks/
  Phase 12 OOD masks: artifacts/phase12/02_padufes_lesion_masks/

Outputs:
  artifacts/phase13/02_predictions.csv   -- per-image P on Phase 9 sample
  artifacts/phase13/02_counterfactual.csv-- per-image CF P (color cast)
  artifacts/phase13/02_ood_predictions.csv
  artifacts/phase13/02_ood_lacn.csv
  artifacts/phase13/02_validate.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

# Reuse Phase 12 LACN helper
import importlib.util
spec = importlib.util.spec_from_file_location(
    "p12_lacn", REPO_ROOT / "scripts" / "phase12" / "01_lacn.py"
)
p12_lacn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p12_lacn)
normalize_lab_in_region = p12_lacn.normalize_lab_in_region

PHASE9_DIR   = REPO_ROOT / "artifacts" / "phase9"
PHASE12_DIR  = REPO_ROOT / "artifacts" / "phase12"
PHASE13_DIR  = REPO_ROOT / "artifacts" / "phase13"

SAMPLE_CSV       = PHASE9_DIR / "01_analytical_sample.csv"
SHORTCUT_CSV     = PHASE9_DIR / "03_shortcut_detection.csv"
SHORTCUT_MASKDIR = PHASE9_DIR / "03_shortcut_masks"
P9_PREDS_CSV     = PHASE9_DIR / "04_predictions.csv"
P9_CF_CSV        = PHASE9_DIR / "07_counterfactual.csv"
P9_LESION_DIR    = PHASE9_DIR / "05_lesion_masks"

OOD_IMG_DIR      = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "images"
OOD_MASK_DIR     = PHASE12_DIR / "02_padufes_lesion_masks"
OOD_MASK_STATS   = PHASE12_DIR / "02_padufes_mask_stats.csv"
OOD_BASE_CSV     = PHASE9_DIR / "11_ood_predictions.csv"

IMG_DIR  = REPO_ROOT / "training_data" / "images"

# Phase 9 sample's mean Lab (Phase 12 reference)
TARGET_LAB = np.array([155.44, 142.81, 135.46], dtype=np.float32)

THRESHOLD = 0.661
SEED = 42
B_BOOT = 5000


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ckpt(path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if "best_state_dict" in (obj if isinstance(obj, dict) else {}):
        state = obj["best_state_dict"]
    m = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    m.load_state_dict(state)
    m.eval()
    return m, obj if isinstance(obj, dict) else {}


@torch.no_grad()
def forward_p(model, img_bgr, transform, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(x).squeeze()))


def boot_ci(arr, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(b)
    a = np.asarray(arr)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = a[idx].mean()
    return float(a.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def balanced_auc(probs, labels, seed=SEED, n_iter=200):
    """Mean AUC over n_iter balanced (50/50) subsamples."""
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_each = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_iter):
        ip = rng.choice(pos, size=n_each, replace=False)
        in_ = rng.choice(neg, size=n_each, replace=False)
        idx = np.concatenate([ip, in_])
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.mean(aucs))


# ============================================================================
# Step 1: Forward on Phase 9 analytical sample
# ============================================================================

def step1_phase9_predictions(model, transform, device, log):
    df = pd.read_csv(SAMPLE_CSV)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phase 9 forward"):
        isic_id = row["isic_id"]
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        p_malig = forward_p(model, img, transform, device)
        rows.append({
            "isic_id": isic_id,
            "label": int(row["label"]),
            "source_collection": str(row["source_collection"]),
            "p_malig": p_malig,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(PHASE13_DIR / "02_predictions.csv", index=False)

    log.append("## Per-collection balanced AUC (Phase 9 PF#1 analog)")
    log.append(f"{'coll':>5} | {'N':>4} | {'pos':>4} | {'P13 balanced AUC':>17}")
    overall = balanced_auc(out_df.p_malig, out_df.label)
    for c in sorted(out_df.source_collection.unique()):
        sub = out_df[out_df.source_collection == c]
        auc_b = balanced_auc(sub.p_malig.values, sub.label.values)
        log.append(f"{c:>5} | {len(sub):>4} | {int(sub.label.sum()):>4} | {auc_b:>17.4f}")
    log.append(f"  Overall balanced AUC = {overall:.4f}")
    log.append("")
    return out_df


# ============================================================================
# Step 2: Color-cast counterfactual delta P
# ============================================================================

def step2_color_cast_cf(model, transform, device, p13_preds, log):
    """Re-run the Phase 9 PF#2 color-cast counterfactual on Phase 13 ckpt."""
    df = pd.read_csv(SAMPLE_CSV)
    p13 = p13_preds.set_index("isic_id")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Color-cast CF"):
        isic_id = row["isic_id"]
        if isic_id not in p13.index:
            continue
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        p_orig = float(p13.loc[isic_id, "p_malig"])
        # Full Lab-mean shift (Phase 9 no_colorcast definition)
        full_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img_norm = normalize_lab_in_region(img, full_mask, TARGET_LAB)
        p_norm = forward_p(model, img_norm, transform, device)
        rows.append({
            "isic_id": isic_id,
            "label": int(row["label"]),
            "source_collection": str(row["source_collection"]),
            "p_orig": p_orig,
            "p_norm_colorcast": p_norm,
            "delta_p_norm_colorcast": p_norm - p_orig,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(PHASE13_DIR / "02_counterfactual.csv", index=False)

    mean, lo, hi = boot_ci(out_df.delta_p_norm_colorcast.values)
    try:
        w, p = stats.wilcoxon(out_df.delta_p_norm_colorcast.values,
                               alternative="two-sided", zero_method="zsplit")
    except ValueError:
        w, p = float("nan"), 1.0
    log.append("## Color-cast counterfactual delta P (Phase 9 PF#2 analog)")
    log.append(f"  Phase 13 mean delta P = {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    log.append(f"  Wilcoxon p = {p:.4g}")
    log.append("  Reference: Phase 9 (Phase 6 ckpt) = +0.1039")
    log.append("  Reference: Phase 10 ckpt          = +0.0144")
    log.append("")
    return out_df


# ============================================================================
# Step 3: OOD eval (PAD-UFES-20) -- baseline + LACN-applied
# ============================================================================

def step3_ood(model, transform, device, log):
    if not OOD_BASE_CSV.exists() or not OOD_IMG_DIR.exists():
        log.append("## OOD eval skipped (PAD-UFES-20 cohort or baseline preds not present).")
        log.append("")
        return None, None

    base = pd.read_csv(OOD_BASE_CSV)
    log.append(f"## OOD eval -- PAD-UFES-20 (N={len(base)})")

    rows = []
    for _, row in tqdm(base.iterrows(), total=len(base), desc="OOD forward"):
        isic_id = row["isic_id"]
        p = OOD_IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        p_p13 = forward_p(model, img, transform, device)
        rows.append({
            "isic_id": isic_id,
            "label": int(row["label"]),
            "p_p13": p_p13,
        })
    ood_df = pd.DataFrame(rows)
    ood_df.to_csv(PHASE13_DIR / "02_ood_predictions.csv", index=False)
    auc_p13 = roc_auc_score(ood_df.label, ood_df.p_p13)

    log.append(f"  Phase 13 OOD AUC (no LACN): {auc_p13:.4f}")
    log.append(f"  Reference: Phase 9 baseline (Phase 6 ckpt) ~= 0.85")
    log.append("")

    # OOD with LACN
    if OOD_MASK_DIR.exists() and OOD_MASK_STATS.exists():
        mask_stats = pd.read_csv(OOD_MASK_STATS)
        reliable_ids = set(mask_stats[mask_stats.reliable].isic_id)
        rows_l = []
        for _, row in tqdm(base.iterrows(), total=len(base), desc="OOD LACN-applied"):
            isic_id = row["isic_id"]
            p = OOD_IMG_DIR / f"{isic_id}.jpg"
            if not p.exists():
                continue
            img = load_image_resized(p)
            if isic_id in reliable_ids:
                mask = cv2.imread(str(OOD_MASK_DIR / f"{isic_id}.png"),
                                  cv2.IMREAD_GRAYSCALE)
                if mask.shape != (IMG_SIZE, IMG_SIZE):
                    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_NEAREST)
                bg_mask = 255 - mask
                img_lacn = normalize_lab_in_region(img, bg_mask, TARGET_LAB)
            else:
                img_lacn = img
            p_p13_lacn = forward_p(model, img_lacn, transform, device)
            rows_l.append({
                "isic_id": isic_id,
                "label": int(row["label"]),
                "p_p13_lacn": p_p13_lacn,
            })
        lacn_df = pd.DataFrame(rows_l)
        lacn_df.to_csv(PHASE13_DIR / "02_ood_lacn.csv", index=False)
        auc_p13_lacn = roc_auc_score(lacn_df.label, lacn_df.p_p13_lacn)
        log.append(f"  Phase 13 OOD AUC (LACN applied): {auc_p13_lacn:.4f}")
        log.append(f"  Reference: Phase 12 Step 2 (Phase 6 ckpt + LACN) = 0.798")
        log.append("  Phase 13 hypothesis: LACN-invariance training -> AUC restored to ~0.85+")
        log.append("")
        return ood_df, lacn_df
    return ood_df, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",
                        default=str(PHASE13_DIR / "best_model_laj_consistency.pth"))
    parser.add_argument("--out-log",
                        default=str(PHASE13_DIR / "02_validate.log"))
    args = parser.parse_args()

    PHASE13_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"[FATAL] ckpt not found: {ckpt_path}")

    device = pick_device()
    print(f"[device] {device}")
    model, obj = load_ckpt(ckpt_path, device)
    transform = get_transforms(CFG, "val")

    log = [
        f"# Phase 13 validation",
        f"ckpt: {ckpt_path}",
        f"device: {device}",
    ]
    if isinstance(obj, dict):
        log.append(f"best_val_auc (training-time): {obj.get('best_val_auc', 'n/a')}")
        log.append(f"best_epoch (training-time):   {obj.get('best_epoch', 'n/a')}")
    log.append("")

    # Step 1: Phase 9 sample
    p13_preds = step1_phase9_predictions(model, transform, device, log)
    # Step 2: CF
    step2_color_cast_cf(model, transform, device, p13_preds, log)
    # Step 3: OOD
    step3_ood(model, transform, device, log)

    Path(args.out_log).write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
