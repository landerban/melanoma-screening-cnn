"""
Phase 10 / Step 2 — Validate the color-invariant fine-tuned ckpt by
re-running all Phase 9 measurements and testing H6-H10.

Prerequisites:
  - artifacts/phase10/best_model_color_invariant.pth   (from Colab)
  - artifacts/phase10/01_finetune.log                  (from Colab)
  - artifacts/phase9/01_analytical_sample.csv          (Phase 9 sample)
  - artifacts/phase9/01_test_cohort.csv                (Phase 9 test cohort)
  - training_data/images/ (the 300 analytical sample images already
    downloaded; OOD images cached if available)

Procedure (each step writes to artifacts/phase10/):
  1. Forward Phase 10 ckpt on 300 analytical-sample images →
     04_predictions.csv
  2. Counterfactual re-run on Phase 10 ckpt → 07_counterfactual.csv
  3. OOD re-eval → 11_ood_predictions.csv
  4. Compute H6-H10 statistics → 08_h6_h10_stats.log
  5. Generate Phase 9 vs Phase 10 comparison table →
     09_comparison.md
  6. Generate Phase 10 paper (auto-fill the numbers) →
     docs/phase10/02_results.md

Run:
    .venv/bin/python scripts/phase10/02_validate_with_phase9.py
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
from sklearn.metrics import roc_auc_score
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

PHASE9_DIR  = REPO_ROOT / "artifacts" / "phase9"
PHASE10_DIR = REPO_ROOT / "artifacts" / "phase10"

P10_CKPT      = PHASE10_DIR / "best_model_color_invariant.pth"
SAMPLE_CSV    = PHASE9_DIR / "01_analytical_sample.csv"
SHORTCUT_CSV  = PHASE9_DIR / "03_shortcut_detection.csv"
SHORTCUT_MASK_DIR = PHASE9_DIR / "03_shortcut_masks"
IMG_DIR       = REPO_ROOT / "training_data" / "images"
OOD_IMG_DIR   = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "images"
OOD_META_CSV  = REPO_ROOT / "training_data" / "eval" / "pad_ufes_20" / "metadata.csv"

OUT_PRED_CSV  = PHASE10_DIR / "04_predictions.csv"
OUT_CF_CSV    = PHASE10_DIR / "07_counterfactual.csv"
OUT_OOD_CSV   = PHASE10_DIR / "11_ood_predictions.csv"
OUT_STATS_LOG = PHASE10_DIR / "08_h6_h10_stats.log"
OUT_COMP_MD   = PHASE10_DIR / "09_comparison.md"
PAPER_OUT     = REPO_ROOT / "docs" / "phase10" / "02_results.md"

SEED = 42
B_BOOT = 5000
THRESHOLD = 0.661


# ---------- helpers ----------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_p10_ckpt(device):
    if not P10_CKPT.exists():
        sys.exit(f"[FATAL] Phase 10 ckpt missing: {P10_CKPT}\n"
                 f"  Did you download it from Colab and place it in "
                 f"{PHASE10_DIR}?")
    obj = torch.load(P10_CKPT, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    m = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    m.load_state_dict(state)
    m.eval()
    return m, obj if isinstance(obj, dict) else {}


@torch.no_grad()
def forward_pmalig(model, img_bgr_or_path, transform, device):
    if isinstance(img_bgr_or_path, np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr_or_path, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
    else:
        pil = Image.open(img_bgr_or_path).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(x).squeeze()))


def bootstrap_auc(labels, scores, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    boots = []
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            pass
    p = roc_auc_score(labels, scores)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return p, lo, hi


def bootstrap_paired(x, y, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = (x[idx] - y[idx]).mean()
    p = (x - y).mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return p, lo, hi


# ---------- inpaint helpers (mirror 07_counterfactual.py) ----------

INPAINT_RADIUS = 5
SHORTCUTS_SPATIAL = ["vignette", "ruler", "hair"]


def load_mask(isic_id: str, name: str):
    p = SHORTCUT_MASK_DIR / f"{isic_id}__{name}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is not None and m.shape != (IMG_SIZE, IMG_SIZE):
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return m


def inpaint_with_masks(img_bgr, mask_names, isic_id):
    combined = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for name in mask_names:
        m = load_mask(isic_id, name)
        if m is not None:
            combined = np.maximum(combined, m)
    if combined.sum() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, combined, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def normalize_color(img_bgr, target_lab_mean):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_mean = lab.reshape(-1, 3).mean(axis=0)
    lab += (target_lab_mean - cur_mean)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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


# ---------- pipeline ----------

def run_forward_300(model, transform, device):
    df = pd.read_csv(SAMPLE_CSV)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="P10 forward"):
        isic_id = row["isic_id"]
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        p_malig = forward_pmalig(model, p, transform, device)
        rows.append({
            "isic_id": isic_id,
            "source_collection": row["source_collection"],
            "label": row["label"],
            "p_malig": p_malig,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PRED_CSV, index=False)
    return out


def run_counterfactual_300(model, transform, device, target_lab):
    df = pd.read_csv(SAMPLE_CSV)
    sc = pd.read_csv(SHORTCUT_CSV).set_index("isic_id")

    configs = [
        ("original",      []),
        ("no_vignette",   ["vignette"]),
        ("no_ruler",      ["ruler"]),
        ("no_hair",       ["hair"]),
        ("no_colorcast",  ["__cc__"]),
        ("all_removed",   SHORTCUTS_SPATIAL + ["__cc__"]),
    ]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="P10 cf"):
        isic_id = row["isic_id"]
        p = IMG_DIR / f"{isic_id}.jpg"
        if not p.exists():
            continue
        img = load_image_resized(p)
        flags = sc.loc[isic_id]
        out = {
            "isic_id": isic_id,
            "source_collection": row["source_collection"],
            "label": row["label"],
            "vignette_present":    int(flags["vignette_present"]),
            "ruler_present":       int(flags["ruler_present"]),
            "hair_present":        int(flags["hair_present"]),
            "color_cast_present":  int(flags["color_cast_present"]),
        }
        for cfg_name, removals in configs:
            cur = img.copy()
            for r in removals:
                if r == "__cc__":
                    cur = normalize_color(cur, target_lab)
                else:
                    if int(flags[f"{r}_present"]):
                        cur = inpaint_with_masks(cur, [r], isic_id)
            p_m = forward_pmalig(model, cur, transform, device)
            out[f"p_malig__{cfg_name}"] = p_m
        rows.append(out)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CF_CSV, index=False)
    return out_df


def run_ood(model, transform, device):
    if not OOD_META_CSV.exists() or not OOD_IMG_DIR.exists():
        print("[skip] OOD assets missing; H10 will be marked N/A")
        return None
    # Use the SAME 200 isic_ids the Phase 9 OOD run selected
    p9_ood = PHASE9_DIR / "11_ood_predictions.csv"
    if p9_ood.exists():
        df = pd.read_csv(p9_ood)[["isic_id", "label"]]
    else:
        # fallback: sample again
        meta = pd.read_csv(OOD_META_CSV, low_memory=False)
        meta = meta[meta.diagnosis_1.isin(["Benign", "Malignant"])].copy()
        meta["label"] = (meta.diagnosis_1 == "Malignant").astype(int)
        rng = np.random.default_rng(SEED)
        b = meta[meta.label == 0].sample(100, random_state=rng.integers(0, 2**31 - 1))
        m = meta[meta.label == 1].sample(100, random_state=rng.integers(0, 2**31 - 1))
        df = pd.concat([b, m])[["isic_id", "label"]]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="P10 OOD"):
        p = OOD_IMG_DIR / f"{row.isic_id}.jpg"
        if not p.exists():
            continue
        p_m = forward_pmalig(model, p, transform, device)
        rows.append({"isic_id": row.isic_id, "label": int(row.label), "p_malig": p_m})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_OOD_CSV, index=False)
    return out


# ---------- H6-H10 evaluation ----------

def evaluate_hypotheses(pred, cf, ood, p10_meta):
    log = ["# Phase 10 — H6-H10 hypothesis tests",
           f"Seed: {SEED}, B (bootstrap): {B_BOOT}",
           ""]
    results = {}

    # ----- H6: in-dist test AUC -----
    log.append("## H6 — Test AUC preservation")
    log.append("Prediction: post-FT test AUC ∈ [0.93, 0.96]; lower CI ≥ 0.93")
    if p10_meta and "post_finetune_test_auc" in p10_meta:
        post_auc = p10_meta["post_finetune_test_auc"]
        pre_auc = p10_meta.get("pre_finetune_test_auc", 0.9513)
        log.append(f"  Pre-FT test AUC: {pre_auc:.4f}")
        log.append(f"  Post-FT test AUC: {post_auc:.4f}")
        log.append(f"  ΔAUC: {post_auc - pre_auc:+.4f}")
        h6 = (0.93 <= post_auc <= 0.96)
        log.append(f"  H6 supported (in band)? {h6}")
        results["H6"] = dict(supported=h6, post_auc=post_auc, delta=post_auc - pre_auc)
    else:
        log.append("  [N/A — ckpt metadata missing]")
        results["H6"] = dict(supported=None)
    log.append("")

    # ----- H7: color cast ΔP reduction -----
    log.append("## H7 — Color-cast ΔP reduction ≥ 50%")
    p9_cf = pd.read_csv(PHASE9_DIR / "07_counterfactual.csv")
    p9_sub = p9_cf[p9_cf.color_cast_present == 1]
    p9_delta = (p9_sub["p_malig__no_colorcast"] - p9_sub["p_malig__original"]).values
    p9_mean = float(p9_delta.mean())
    p9_lo, p9_hi = np.quantile(
        [p9_delta[np.random.default_rng(SEED).integers(0, len(p9_delta), len(p9_delta))].mean()
         for _ in range(B_BOOT)], [0.025, 0.975]
    )

    p10_sub = cf[cf.color_cast_present == 1]
    p10_delta = (p10_sub["p_malig__no_colorcast"] - p10_sub["p_malig__original"]).values
    p10_mean, p10_lo, p10_hi = bootstrap_paired(
        p10_sub["p_malig__no_colorcast"].values,
        p10_sub["p_malig__original"].values,
    )
    log.append(f"  Phase 9 mean ΔP: {p9_mean:+.4f} [{p9_lo:+.4f}, {p9_hi:+.4f}]")
    log.append(f"  Phase 10 mean ΔP: {p10_mean:+.4f} [{p10_lo:+.4f}, {p10_hi:+.4f}]")
    reduction = (p9_mean - p10_mean) / p9_mean if p9_mean != 0 else 0
    log.append(f"  Reduction: {reduction:.1%}")
    h7 = (abs(p10_mean) <= 0.05) and (p10_hi < p9_lo or p10_lo > p9_hi)
    log.append(f"  H7 supported (Phase 10 |ΔP| ≤ 0.05 AND CIs disjoint)? {h7}")
    results["H7"] = dict(supported=h7, p10_mean=p10_mean, p9_mean=p9_mean,
                          reduction=reduction)
    log.append("")

    # ----- H8: c=70 balanced AUC rise -----
    log.append("## H8 — c=70 balanced AUC ≥ 0.65")
    sub70 = pred[pred.source_collection.astype(str) == "70"]
    p10_auc70, lo70, hi70 = bootstrap_auc(sub70.label.values, sub70.p_malig.values)
    log.append(f"  Phase 9 c=70 balanced AUC: 0.548 [0.432, 0.663]")
    log.append(f"  Phase 10 c=70 balanced AUC: {p10_auc70:.4f} [{lo70:.4f}, {hi70:.4f}]")
    h8 = (p10_auc70 >= 0.65) and (lo70 > 0.50)
    log.append(f"  H8 supported (AUC ≥ 0.65 AND lo > 0.50)? {h8}")
    results["H8"] = dict(supported=h8, p10_auc=p10_auc70, lo=lo70, hi=hi70)
    log.append("")

    # ----- H9: cross-collection gap shrink -----
    log.append("## H9 — Cross-collection gap ≤ 0.30")
    sub212 = pred[pred.source_collection.astype(str) == "212"]
    auc212 = roc_auc_score(sub212.label, sub212.p_malig)
    gap = auc212 - p10_auc70
    log.append(f"  Phase 9 gap (c=212 − c=70): 0.434")
    log.append(f"  Phase 10 gap: {gap:.4f}")
    h9 = (gap <= 0.30)
    log.append(f"  H9 supported (gap ≤ 0.30)? {h9}")
    results["H9"] = dict(supported=h9, gap=gap)
    log.append("")

    # ----- H10: OOD recall safety check -----
    log.append("## H10 — OOD recall safety check ≥ 0.70")
    if ood is not None:
        preds = (ood.p_malig >= THRESHOLD).astype(int)
        tp = int(((preds == 1) & (ood.label == 1)).sum())
        fn = int(((preds == 0) & (ood.label == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        log.append(f"  Phase 9 OOD recall @ 0.661 (balanced 50/50): 0.790")
        log.append(f"  Phase 10 OOD recall: {recall:.3f}")
        h10 = (recall >= 0.70)
        log.append(f"  H10 supported (recall ≥ 0.70)? {h10}")
        results["H10"] = dict(supported=h10, recall=recall)
    else:
        log.append("  [N/A — OOD assets missing]")
        results["H10"] = dict(supported=None)
    log.append("")

    # ----- Summary -----
    log.append("## Summary")
    for h, v in results.items():
        log.append(f"  {h}: {v}")

    return log, results


def write_comparison(pred, cf, ood, results, p10_meta):
    p9_pred = pd.read_csv(PHASE9_DIR / "04_predictions.csv")
    p9_cf = pd.read_csv(PHASE9_DIR / "07_counterfactual.csv")

    md = ["# Phase 9 vs Phase 10 — Side-by-side comparison",
          ""]

    # ----- Primary finding #1 (balanced AUC per-collection) -----
    md.append("## PF#1 — Per-collection balanced AUC")
    md.append("")
    md.append("| Collection | Phase 9 AUC | Phase 10 AUC | Δ |")
    md.append("|---|---|---|---|")
    for c in ["212", "70", "249"]:
        s9 = p9_pred[p9_pred.source_collection.astype(str) == c]
        s10 = pred[pred.source_collection.astype(str) == c]
        a9 = roc_auc_score(s9.label, s9.p_malig)
        a10 = roc_auc_score(s10.label, s10.p_malig)
        md.append(f"| c={c} | {a9:.4f} | {a10:.4f} | {a10 - a9:+.4f} |")
    md.append("")

    # ----- Primary finding #2 (counterfactual ΔP) -----
    md.append("## PF#2 — Counterfactual ΔP per shortcut")
    md.append("")
    md.append("| Shortcut | Phase 9 mean ΔP | Phase 10 mean ΔP | Δ-of-Δ |")
    md.append("|---|---|---|---|")
    for sc, pres in [("vignette", "vignette_present"),
                      ("ruler", "ruler_present"),
                      ("hair", "hair_present"),
                      ("colorcast", "color_cast_present")]:
        s9 = p9_cf[p9_cf[pres] == 1]
        s10 = cf[cf[pres] == 1]
        d9 = (s9[f"p_malig__no_{sc}"] - s9["p_malig__original"]).mean()
        d10 = (s10[f"p_malig__no_{sc}"] - s10["p_malig__original"]).mean()
        md.append(f"| {sc} | {d9:+.4f} | {d10:+.4f} | {d10 - d9:+.4f} |")
    md.append("")

    # ----- Primary finding #3 (gap closure under all_removed) -----
    md.append("## PF#3 — Original vs all-removed AUC per collection")
    md.append("")
    md.append("| Collection | P9 orig | P9 all-rm | P10 orig | P10 all-rm |")
    md.append("|---|---|---|---|---|")
    for c in ["212", "70", "249"]:
        s9 = p9_cf[p9_cf.source_collection.astype(str) == c]
        s10 = cf[cf.source_collection.astype(str) == c]
        a9o = roc_auc_score(s9.label, s9["p_malig__original"])
        a9r = roc_auc_score(s9.label, s9["p_malig__all_removed"])
        a10o = roc_auc_score(s10.label, s10["p_malig__original"])
        a10r = roc_auc_score(s10.label, s10["p_malig__all_removed"])
        md.append(f"| c={c} | {a9o:.3f} | {a9r:.3f} | {a10o:.3f} | {a10r:.3f} |")
    md.append("")

    # ----- Hypothesis test summary -----
    md.append("## H6-H10 outcomes")
    md.append("")
    for h, v in results.items():
        if v.get("supported") is None:
            md.append(f"- **{h}**: N/A")
        else:
            md.append(f"- **{h}**: {'SUPPORTED' if v['supported'] else 'FALSIFIED'}  ({v})")

    OUT_COMP_MD.write_text("\n".join(md) + "\n")
    return md


def main():
    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[device] {device}")

    model, p10_meta = load_p10_ckpt(device)
    transform = get_transforms(CFG, "val")
    print(f"Phase 10 ckpt loaded.")

    # 1. Forward on 300 analytical sample
    pred = run_forward_300(model, transform, device)

    # 2. Counterfactual
    target_lab = compute_target_lab_mean()
    cf = run_counterfactual_300(model, transform, device, target_lab)

    # 3. OOD
    ood = run_ood(model, transform, device)

    # 4. H6-H10
    log_lines, results = evaluate_hypotheses(pred, cf, ood, p10_meta)
    OUT_STATS_LOG.write_text("\n".join(log_lines) + "\n")
    print("\n".join(log_lines[-20:]))

    # 5. Comparison
    write_comparison(pred, cf, ood, results, p10_meta)
    print(f"\nComparison: {OUT_COMP_MD}")
    print(f"Stats log:  {OUT_STATS_LOG}")


if __name__ == "__main__":
    main()
