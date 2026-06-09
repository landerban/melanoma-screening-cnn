"""
Phase 9 / Step 3q-t — Counterfactual inpainting + forward-pass on
inpainted images + causal Shapley-approximation.

Method substitution from the pre-registered protocol.
The original Phase 2 protocol committed to LaMa for inpainting and a
full 2^n Causal Shapley enumeration (n=5 shortcuts → 32 configurations
per image). For the actual Phase 9 implementation we make two
methodological substitutions and disclose both:

1. *Inpainting*. We use OpenCV's Telea fast-marching inpainting
   (`cv2.inpaint(method=INPAINT_TELEA)`) instead of LaMa. Rationale:
   (a) the official big-lama checkpoint download endpoint is not
   stable; (b) the masked regions for vignette, ruler, and hair are
   *small* relative to LaMa's design target (large masks), so Telea's
   diffusion-based fill is adequate; (c) Telea is a built-in OpenCV
   primitive with zero additional setup cost. Disclosure: Telea
   introduces blurring near the mask boundary; we mitigate by using
   a 5-px inpaint radius and visually QA'ing a sub-sample.

2. *Causal Shapley*. We measure the *marginal contribution* of each
   shortcut separately AND the *joint effect* of removing all
   shortcuts together. With four shortcuts (vignette, ruler, hair,
   color-cast), the Shapley axiom (efficiency: marginals sum to
   joint) holds approximately if interactions are small; we report
   the joint vs. sum-of-marginals discrepancy as the interaction
   term. This is the "Shapley-lite" approximation common in feature
   attribution at scale, e.g., DeepLIFT's choice of single-feature
   ablations.

Color cast is operationalized as a CLAHE + channel-mean shift toward
a *uniform* (mean-of-all-collections) signature, applied as a global
transform rather than a localized inpaint.

Compute budget on M3 MPS: 5 configs × 300 images × ~150 ms/forward
= ~3 min for forward; inpainting ~20 ms/image × 4 = ~30s. Total < 5
minutes.

Outputs:
  artifacts/phase9/07_counterfactual.csv
    Per-image P_malig for each of: original, no_vignette, no_ruler,
    no_hair, no_colorcast, all_removed.

  artifacts/phase9/07_counterfactual.log
    Per-shortcut Δ(P_malig) summary stats + per-collection breakdown.

  artifacts/phase9/07_cf_inpainted/<isic_id>__<config>.jpg
    Inpainted images, saved for QA.

Run:
    .venv/bin/python scripts/phase9/07_counterfactual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
SHORTCUT_MASK_DIR = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_masks"
SHORTCUT_CSV = REPO_ROOT / "artifacts" / "phase9" / "03_shortcut_detection.csv"
CKPT_PATH  = REPO_ROOT / "best_model.pth"
OUT_CSV    = REPO_ROOT / "artifacts" / "phase9" / "07_counterfactual.csv"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "07_counterfactual.log"
QA_DIR     = REPO_ROOT / "artifacts" / "phase9" / "07_cf_inpainted_qa"

SHORTCUTS_SPATIAL = ["vignette", "ruler", "hair"]
INPAINT_RADIUS = 5


def load_mask(isic_id: str, name: str) -> np.ndarray | None:
    p = SHORTCUT_MASK_DIR / f"{isic_id}__{name}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != (IMG_SIZE, IMG_SIZE):
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return m


def inpaint_with_masks(img_bgr: np.ndarray, mask_names: list[str],
                        isic_id: str) -> np.ndarray:
    """Inpaint img by removing all listed shortcut regions."""
    combined = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for name in mask_names:
        m = load_mask(isic_id, name)
        if m is not None:
            combined = np.maximum(combined, m)
    if combined.sum() == 0:
        return img_bgr  # nothing to inpaint
    out = cv2.inpaint(img_bgr, combined, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out


def normalize_color(img_bgr: np.ndarray, target_lab_mean: np.ndarray) -> np.ndarray:
    """Color-cast removal: shift the image's Lab-space mean to the
    target mean. Used as the operationalization of "remove color
    cast" since color cast has no localized mask."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_mean = lab.reshape(-1, 3).mean(axis=0)
    lab += (target_lab_mean - cur_mean)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_target_lab_mean() -> np.ndarray:
    """Average Lab-mean across the analytical sample → the
    'uniform-signature' target for color-cast normalization."""
    df = pd.read_csv(SAMPLE_CSV)
    means = []
    for isic_id in df.isic_id:
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p, size=IMG_SIZE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        means.append(lab.reshape(-1, 3).mean(axis=0))
    return np.stack(means, axis=0).mean(axis=0)


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(device):
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def forward_p_malig(model, img_bgr: np.ndarray, transform, device) -> float:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = transform(pil).unsqueeze(0).to(device)
    logit = model(x).squeeze()
    return float(torch.sigmoid(logit))


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = load_model(device)
    transform = get_transforms(CFG, "val")
    print(f"Device: {device}")

    print("Computing target Lab-mean for color-cast normalization...")
    target_lab = compute_target_lab_mean()
    print(f"Target Lab-mean (L,a,b): {target_lab.round(2)}")

    df = pd.read_csv(SAMPLE_CSV)
    shortcuts_df = pd.read_csv(SHORTCUT_CSV).set_index("isic_id")

    configs = [
        ("original",      []),
        ("no_vignette",   ["vignette"]),
        ("no_ruler",      ["ruler"]),
        ("no_hair",       ["hair"]),
        ("no_colorcast",  ["__colorcast__"]),  # special handler
        ("all_removed",   SHORTCUTS_SPATIAL + ["__colorcast__"]),
    ]

    rows = []
    per_config_qa = {c[0]: 0 for c in configs}  # per-config counter
    QA_LIMIT = 15  # save up to 15 examples PER CONFIG for visual QA

    for _, row in tqdm(df.iterrows(), total=len(df), desc="cf"):
        isic_id = row["isic_id"]
        img_path = IMG_DIR / f"{isic_id}.jpg"
        if not img_path.exists():
            continue
        img = load_image_resized(img_path, size=IMG_SIZE)

        sc_flags = shortcuts_df.loc[isic_id]
        out = {"isic_id": isic_id,
               "source_collection": row["source_collection"],
               "label": row["label"],
               "vignette_present": int(sc_flags["vignette_present"]),
               "ruler_present":    int(sc_flags["ruler_present"]),
               "hair_present":     int(sc_flags["hair_present"]),
               "color_cast_present": int(sc_flags["color_cast_present"])}

        for config_name, removals in configs:
            current = img.copy()
            applied = False
            for r in removals:
                if r == "__colorcast__":
                    current = normalize_color(current, target_lab)
                    applied = True
                else:
                    # only inpaint if shortcut was actually present
                    if int(sc_flags[f"{r}_present"]):
                        current = inpaint_with_masks(current, [r], isic_id)
                        applied = True
            p_malig = forward_p_malig(model, current, transform, device)
            out[f"p_malig__{config_name}"] = round(p_malig, 6)

            # Save a few examples for visual QA — per-config limit
            if config_name != "original" and applied and per_config_qa[config_name] < QA_LIMIT:
                cv2.imwrite(str(QA_DIR / f"{isic_id}__{config_name}.jpg"), current)
                per_config_qa[config_name] += 1

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # --- Summary statistics ---
    log = [
        "# Phase 9 Step 3q-t — counterfactual P_malig",
        f"N = {len(out_df)}",
        f"Configs: {[n for n, _ in configs]}",
        "",
    ]

    # Per-shortcut marginal effect: delta = p_malig(no_X) - p_malig(original)
    # Computed only on images where X was actually present
    log.append("## Marginal effects per shortcut")
    log.append("(delta = mean[ P(no_X) - P(original) ] on images with X present)")
    for sc in SHORTCUTS_SPATIAL + ["colorcast"]:
        config_col = f"p_malig__no_{sc}"
        if config_col not in out_df.columns:
            continue
        if sc == "colorcast":
            sub = out_df[out_df.color_cast_present == 1]
        else:
            sub = out_df[out_df[f"{sc}_present"] == 1]
        if len(sub) == 0:
            log.append(f"  {sc:<12}: no images with shortcut present.")
            continue
        delta = sub[config_col] - sub["p_malig__original"]
        log.append(f"  {sc:<12} (N={len(sub):>3}): "
                   f"mean Δ = {delta.mean():+.4f}, "
                   f"median Δ = {delta.median():+.4f}, "
                   f"sd = {delta.std():.4f}")
    log.append("")

    # Joint vs sum-of-marginals (interaction proxy)
    log.append("## Joint vs sum-of-marginals (interaction term)")
    log.append("On images with ALL shortcuts present (sub-sample):")
    has_all = (
        (out_df.vignette_present == 1)
        & (out_df.ruler_present == 1)
        & (out_df.hair_present == 1)
        & (out_df.color_cast_present == 1)
    )
    sub = out_df[has_all]
    log.append(f"  N with all-4 present: {len(sub)}")
    if len(sub) > 0:
        d_joint = sub["p_malig__all_removed"] - sub["p_malig__original"]
        d_sum = sum(
            sub[f"p_malig__no_{sc}"] - sub["p_malig__original"]
            for sc in SHORTCUTS_SPATIAL + ["colorcast"]
        )
        log.append(f"  mean Δ_joint = {d_joint.mean():+.4f}")
        log.append(f"  mean Σ(Δ_marginal) = {d_sum.mean():+.4f}")
        log.append(f"  interaction (joint - sum) = "
                   f"{(d_joint - d_sum).mean():+.4f}")
    log.append("")

    # Per-collection breakdown
    log.append("## Per-collection marginal effects")
    for sc in SHORTCUTS_SPATIAL + ["colorcast"]:
        if sc == "colorcast":
            present_col = "color_cast_present"
        else:
            present_col = f"{sc}_present"
        log.append(f"\n  {sc}:")
        for c in sorted(out_df.source_collection.astype(str).unique()):
            sub = out_df[
                (out_df.source_collection.astype(str) == c)
                & (out_df[present_col] == 1)
            ]
            if len(sub) == 0:
                log.append(f"    c={c}: N=0 (no present)")
                continue
            delta = sub[f"p_malig__no_{sc}"] - sub["p_malig__original"]
            log.append(f"    c={c}: N={len(sub):>3}, "
                       f"mean Δ = {delta.mean():+.4f}")

    log.append("")
    log.append("## Notes")
    log.append(f"  - QA examples saved: {QA_DIR.relative_to(REPO_ROOT)} "
               f"({qa_save_count} images)")
    log.append("  - Inpainting: OpenCV Telea, radius=5px")
    log.append("  - Color-cast removal: Lab-space mean shift to "
               "analytical-sample mean")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
