"""
Phase 13 -- Lesion-Aware Jitter + LACN-Consistency Fine-Tune

Combines two design choices motivated by the post-defense analyses
(PRs #4 and #5):

  D1 (LAJ): Lesion-Aware Jitter
    Phase 10's ColorJitter(hue=0.10) shifts hue globally, including
    the lesion interior. PR #5 (IG attribution) shows the Phase 10
    intervention shifts attribution mostly off the background but is
    underpowered in the median -- consistent with global jitter also
    eroding some lesion color signal. LAJ restricts the hue shift to
    background pixels (outside the Otsu lesion mask), preserving the
    diagnostic color information that ABCDE's C-axis relies on.

  D2 (LACN-Consistency)
    PR #4 (LACN OOD) shows test-time LACN drops OOD AUC because the
    base model was never trained to be LACN-invariant. We add a
    consistency-regularization term that penalises differences in
    sigmoid(logit) between the LAJ-augmented sample and the LACN-
    applied version of the same image. The model is trained to be
    invariant to the test-time intervention it will see.

Loss:
    L_total =      L_focal(x_train)
            + 0.5 *L_focal(x_lacn)
            + lam *MSE(sigmoid(f(x_train)), sigmoid(f(x_lacn)))

Base ckpt: Phase 6 best_model.pth (same as Phase 10 for direct
           comparability).
Epochs: 5 (matches Phase 10).
Optimizer: AdamW lr=1e-5, weight_decay from CFG.
Sampler: WeightedRandomSampler (Phase 10 reuse).

Modes:
  --smoke      run a 1-epoch sanity-check on the Phase 9 analytical
               sample (N=300). Verifies all code paths. ~5 min on M3.
  --subset N   train on a stratified subset of N images per epoch
               (faster trend check, ~1-2h on M3 for N=5000, 2 epochs).
  default      full training data, all epochs.

Outputs:
  artifacts/phase13/best_model_laj_consistency.pth
  artifacts/phase13/01_finetune.log
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from efficientnet_b0 import (  # noqa: E402
    CFG, EfficientNetB0Classifier, FocalLoss,
    make_weighted_sampler, patient_level_split,
)
from trainer import load_and_merge_metadata  # noqa: E402


# ============================================================================
# Color-space utilities
# ============================================================================

def otsu_lesion_mask(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Otsu mask on L channel of Lab, largest connected component,
    morphologically cleaned. Returns binary uint8 (255 = lesion).
    """
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    _, raw = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=2)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n <= 1:
        return np.zeros_like(L)
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    mask = (lbl == biggest).astype(np.uint8) * 255
    area_frac = mask.sum() / 255 / mask.size
    if not (0.005 <= area_frac <= 0.65):
        return np.zeros_like(L)
    return mask


def bg_only_hue_jitter(img_rgb_uint8: np.ndarray, bg_mask: np.ndarray,
                       hue_strength: float, rng: np.random.Generator) -> np.ndarray:
    """Shift the OpenCV-scale Hue of background pixels by a random offset.
    Lesion pixels (inside bg_mask=False) are preserved.

    hue_strength = 0.10 means random shift in [-0.10*180, +0.10*180]
    on the OpenCV 0..180 scale = ~+-36 deg on the 0..360 hue circle.
    """
    if bg_mask.sum() == 0 or hue_strength <= 0:
        return img_rgb_uint8
    shift = int(round(rng.uniform(-hue_strength * 180.0, hue_strength * 180.0)))
    if shift == 0:
        return img_rgb_uint8
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.int16)
    bg = bg_mask > 127
    hsv[bg, 0] = (hsv[bg, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def lacn_apply_bg_only(img_rgb_uint8: np.ndarray, bg_mask: np.ndarray,
                       target_lab: np.ndarray) -> np.ndarray:
    """Apply Phase 12 LACN (background-only Lab-mean shift) to the image.
    Lesion pixels untouched. Used as the consistency-loss partner sample.
    """
    if bg_mask.sum() == 0:
        return img_rgb_uint8
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = bg_mask > 127
    cur = lab[inside].mean(axis=0)
    shift = target_lab - cur
    lab[inside] += shift
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# Phase 13 dataset: returns (x_train, x_lacn, label) per index
# ============================================================================

class Phase13LACNConsistencyDataset(Dataset):
    """
    Returns three tensors per item:
      x_train -- BackgroundOnlyHueJitter applied (train-time augmentation)
      x_lacn  -- LACN applied (Phase 12 background-only Lab shift)
      label

    Both x_train and x_lacn share the *same* geometric augmentation, so
    the consistency MSE between f(x_train) and f(x_lacn) measures only
    the color-space sensitivity difference -- the model behavior we
    want to make invariant.
    """

    def __init__(self, image_paths, labels, geom_tf, finalize_tf,
                 target_lab: np.ndarray, bg_hue_strength: float, seed: int):
        self.paths = list(image_paths)
        self.labels = list(labels)
        self.geom_tf = geom_tf            # PIL -> PIL (random)
        self.finalize_tf = finalize_tf    # PIL -> tensor (deterministic)
        self.target_lab = target_lab.astype(np.float32)
        self.bg_hue_strength = bg_hue_strength
        self.base_seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Per-item independent RNG, derived from worker_seed + idx
        worker_seed = torch.utils.data.get_worker_info()
        seed_offset = worker_seed.seed if worker_seed is not None else 0
        rng = np.random.default_rng((self.base_seed + seed_offset + idx) % (2**31 - 1))

        img = Image.open(self.paths[idx]).convert("RGB")

        # Apply random geom aug -- shared by both copies
        img_geom = self.geom_tf(img)
        img_geom_np = np.array(img_geom)

        # Otsu mask on the geom-augmented (RGB)
        lesion_mask = otsu_lesion_mask(img_geom_np)
        bg_mask = 255 - lesion_mask

        # x_train: lesion-aware jitter (background hue shift, random)
        x_train_np = bg_only_hue_jitter(img_geom_np, bg_mask,
                                        self.bg_hue_strength, rng)
        # x_lacn:  LACN applied (deterministic given target_lab)
        x_lacn_np = lacn_apply_bg_only(img_geom_np, bg_mask, self.target_lab)

        x_train = self.finalize_tf(Image.fromarray(x_train_np))
        x_lacn = self.finalize_tf(Image.fromarray(x_lacn_np))
        return x_train, x_lacn, int(self.labels[idx])


class PlainEvalDataset(Dataset):
    """Eval-time dataset: returns (x, label) only. No LACN, no jitter."""
    def __init__(self, image_paths, labels, val_tf):
        self.paths = list(image_paths)
        self.labels = list(labels)
        self.val_tf = val_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.val_tf(img), int(self.labels[idx])


# ============================================================================
# Target Lab mean (Phase 9 reference, cached)
# ============================================================================

def compute_target_lab(sample_csv: Path, img_dir: Path) -> np.ndarray:
    df = pd.read_csv(sample_csv)
    means = []
    for isic_id in df.isic_id:
        p = img_dir / f"{isic_id}.jpg"
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (CFG["input_size"], CFG["input_size"]))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        means.append(lab.reshape(-1, 3).mean(axis=0))
    return np.stack(means, axis=0).mean(axis=0)


# ============================================================================
# Transforms
# ============================================================================

def get_geom_tf(cfg):
    size = cfg["input_size"]
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
    ])


def get_finalize_tf(cfg):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


def get_val_tf(cfg):
    size = cfg["input_size"]
    return transforms.Compose([
        transforms.Resize(int(size * 1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


# ============================================================================
# Reproducibility
# ============================================================================

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# Eval helper
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs_l, labels_l = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        p = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
        probs_l.append(p)
        labels_l.append(lbls.numpy().astype(int))
    if not probs_l:
        return dict(auc=float("nan"), probs=np.array([]), labels=np.array([]))
    probs = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return dict(auc=float(auc), probs=probs, labels=labels)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt",
                        default="artifacts/phase10/best_model_color_invariant.pth",
                        help="Default = Phase 10 ckpt (cumulative: Phase 13 trains "
                             "on top of Phase 10's hue-jitter learning). For a clean "
                             "ablation vs Phase 10 use --base-ckpt best_model.pth.")
    parser.add_argument("--out-ckpt",
                        default="artifacts/phase13/best_model_laj_consistency.pth")
    parser.add_argument("--log-path",
                        default="artifacts/phase13/01_finetune.log")
    parser.add_argument("--training-data", default="training_data")
    parser.add_argument("--images-dir", default="training_data/images")
    parser.add_argument("--p9-sample",
                        default="artifacts/phase9/01_analytical_sample.csv")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bg-hue-strength", type=float, default=0.15,
                        help="Background-only hue jitter strength on OpenCV scale (0.15 ~ +-27 deg in OpenCV / +-54 deg in 360 deg scale)")
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--lacn-focal-weight", type=float, default=0.5,
                        help="Focal loss weight on the LACN-augmented sample")

    parser.add_argument("--smoke", action="store_true",
                        help="Sanity-check on Phase 9 analytical sample (N~300), 1 epoch.")
    parser.add_argument("--subset", type=int, default=0,
                        help="If >0, randomly subsample this many images per epoch (stratified by label).")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Persist a checkpoint after every epoch (Colab session safety).")
    parser.add_argument("--drive-backup-dir", type=str, default="",
                        help="If set, also copy the saved ckpt into this directory after every save (Colab Drive).")
    args = parser.parse_args()

    seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    out_ckpt = REPO_ROOT / args.out_ckpt
    log_path = REPO_ROOT / args.log_path
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "# Phase 13 LAJ+Consistency fine-tune log",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"mode: {'smoke' if args.smoke else (f'subset={args.subset}' if args.subset else 'full')}",
        f"base_ckpt: {args.base_ckpt}",
        f"device: {device}",
        f"epochs: {args.epochs}, lr: {args.lr}, batch: {args.batch_size}",
        f"bg_hue_strength: {args.bg_hue_strength}",
        f"lambda_consistency: {args.lambda_consistency}",
        f"lacn_focal_weight: {args.lacn_focal_weight}",
        "",
    ]

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg = dict(CFG)
    # Hard cap batch size for smoke / local
    cfg["batch_size"] = args.batch_size

    # Compute target Lab mean from the Phase 9 analytical sample
    log("## Computing target Lab mean (Phase 9 analytical sample)")
    target_lab = compute_target_lab(REPO_ROOT / args.p9_sample,
                                     REPO_ROOT / args.images_dir)
    log(f"target_lab = {target_lab.round(2).tolist()}")

    log("\n## Loading metadata")
    if args.smoke:
        # smoke: Phase 9 analytical sample, all labeled
        df9 = pd.read_csv(REPO_ROOT / args.p9_sample)
        df9 = df9.copy()
        df9["image_path"] = df9["isic_id"].apply(
            lambda x: str(REPO_ROOT / args.images_dir / f"{x}.jpg"))
        df9 = df9[df9["image_path"].apply(lambda p: Path(p).exists())]
        df9["effective_patient_id"] = df9.get("patient_id", pd.Series(df9.isic_id))
        df9["source_collection"] = df9.get("source_collection", "smoke").astype(str)
        df = df9.reset_index(drop=True)
        log(f"smoke: using Phase 9 sample, N = {len(df)}")
        # Use the smoke sample for train + val + test
        train_df = df.sample(frac=0.7, random_state=args.seed)
        rest = df.drop(train_df.index)
        val_df = rest.sample(frac=0.5, random_state=args.seed)
        test_df = rest.drop(val_df.index)
        log(f"smoke split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    else:
        df = load_and_merge_metadata(args.training_data, args.images_dir)
        log(f"Merged {len(df):,} rows")
        train_df, val_df, cal_df, test_df = patient_level_split(
            df, patient_col="effective_patient_id", seed=args.seed,
        )
        log(f"train={len(train_df):,} val={len(val_df):,} "
            f"cal={len(cal_df):,} test={len(test_df):,}")
        if args.subset > 0:
            n_subset = min(args.subset, len(train_df))
            pos = train_df[train_df.label == 1]
            neg = train_df[train_df.label == 0]
            n_pos = max(1, int(n_subset * len(pos) / len(train_df)))
            n_neg = n_subset - n_pos
            train_df = pd.concat([
                pos.sample(n=min(n_pos, len(pos)), random_state=args.seed),
                neg.sample(n=min(n_neg, len(neg)), random_state=args.seed),
            ], ignore_index=True)
            log(f"[subset] train downsampled to N = {len(train_df)} "
                f"(pos={int(train_df.label.sum())}, neg={int(len(train_df) - train_df.label.sum())})")

    geom_tf = get_geom_tf(cfg)
    finalize_tf = get_finalize_tf(cfg)
    val_tf = get_val_tf(cfg)

    train_ds = Phase13LACNConsistencyDataset(
        image_paths=train_df["image_path"].tolist(),
        labels=train_df["label"].tolist(),
        geom_tf=geom_tf,
        finalize_tf=finalize_tf,
        target_lab=target_lab,
        bg_hue_strength=args.bg_hue_strength,
        seed=args.seed,
    )
    val_ds = PlainEvalDataset(val_df["image_path"].tolist(),
                              val_df["label"].tolist(), val_tf)
    test_ds = PlainEvalDataset(test_df["image_path"].tolist(),
                               test_df["label"].tolist(), val_tf)

    gen = torch.Generator(); gen.manual_seed(args.seed)
    sampler = make_weighted_sampler(train_df["label"].tolist(), generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    log("\n## Loading base ckpt")
    obj = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state)
    log(f"Base ckpt loaded: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=cfg["weight_decay"])
    focal = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])

    # ------------------------------------------------------------------
    # Pre-fine-tune eval
    # ------------------------------------------------------------------
    log("\n## Pre-fine-tune eval")
    val_pre = evaluate(model, val_loader, device)
    test_pre = evaluate(model, test_loader, device)
    log(f"  pre-FT val  AUC = {val_pre['auc']:.4f}")
    log(f"  pre-FT test AUC = {test_pre['auc']:.4f}")

    # ------------------------------------------------------------------
    # Fine-tune loop
    # ------------------------------------------------------------------
    log("\n## Fine-tune trajectory")
    best_val_auc = val_pre["auc"]
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    lam = args.lambda_consistency
    w_lacn_focal = args.lacn_focal_weight

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss_focal_orig = 0.0
        ep_loss_focal_lacn = 0.0
        ep_loss_cons = 0.0
        n_batches = 0
        for x_train, x_lacn, lbls in tqdm(train_loader,
                                           desc=f"epoch {epoch}/{args.epochs}"):
            x_train = x_train.to(device, non_blocking=True)
            x_lacn = x_lacn.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            logits_orig = model(x_train).squeeze(1)
            logits_lacn = model(x_lacn).squeeze(1)

            L_focal_orig = focal(logits_orig, lbls)
            L_focal_lacn = focal(logits_lacn, lbls)
            L_cons = F.mse_loss(torch.sigmoid(logits_orig),
                                 torch.sigmoid(logits_lacn))

            L = L_focal_orig + w_lacn_focal * L_focal_lacn + lam * L_cons
            L.backward()
            optimizer.step()

            ep_loss_focal_orig += float(L_focal_orig.item())
            ep_loss_focal_lacn += float(L_focal_lacn.item())
            ep_loss_cons += float(L_cons.item())
            n_batches += 1

        train_focal_orig = ep_loss_focal_orig / max(1, n_batches)
        train_focal_lacn = ep_loss_focal_lacn / max(1, n_batches)
        train_cons = ep_loss_cons / max(1, n_batches)

        val_e = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log(f"  epoch {epoch}: focal_orig={train_focal_orig:.4f}  "
            f"focal_lacn={train_focal_lacn:.4f}  cons={train_cons:.4f}  "
            f"val_AUC={val_e['auc']:.4f}  time={dt:.0f}s")

        if val_e["auc"] > best_val_auc:
            best_val_auc = val_e["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            log("    new best val AUC; checkpointing")

        if args.save_every_epoch:
            epoch_payload = {
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "best_state_dict": best_state,
                "cfg": cfg,
                "epoch": epoch,
                "best_val_auc": float(best_val_auc),
                "best_epoch": int(best_epoch),
                "current_val_auc": float(val_e["auc"]),
                "phase13_modifications": {
                    "method": "LAJ + LACN-consistency (intermediate)",
                    "bg_hue_strength": args.bg_hue_strength,
                    "lambda_consistency": args.lambda_consistency,
                    "lacn_focal_weight": args.lacn_focal_weight,
                    "target_lab": target_lab.tolist(),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            ep_ckpt = out_ckpt.parent / f"epoch_{epoch}_{out_ckpt.name}"
            torch.save(epoch_payload, ep_ckpt)
            log(f"    [save-every-epoch] checkpoint: {ep_ckpt.relative_to(REPO_ROOT)}")
            if args.drive_backup_dir:
                try:
                    import shutil
                    drive_dir = Path(args.drive_backup_dir)
                    drive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ep_ckpt, drive_dir / ep_ckpt.name)
                    # also push the live log
                    log_path.write_text("\n".join(log_lines) + "\n")
                    shutil.copy(log_path, drive_dir / log_path.name)
                    log(f"    [drive-backup] -> {drive_dir / ep_ckpt.name}")
                except Exception as e:
                    log(f"    [drive-backup] FAILED: {e}")

    log(f"\nBest val AUC = {best_val_auc:.4f} at epoch {best_epoch}")

    # Load best for final eval
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_post = evaluate(model, test_loader, device)
    log(f"\n## Final test eval")
    log(f"  post-FT test AUC = {test_post['auc']:.4f}  (pre-FT was {test_pre['auc']:.4f})")
    log(f"  ΔAUC = {test_post['auc'] - test_pre['auc']:+.4f}")

    # Save ckpt
    save_payload = {
        "state_dict": best_state,
        "cfg": cfg,
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "pre_finetune_val_auc": float(val_pre["auc"]),
        "pre_finetune_test_auc": float(test_pre["auc"]),
        "post_finetune_test_auc": float(test_post["auc"]),
        "phase13_modifications": {
            "method": "LAJ (background-only hue jitter) + LACN-consistency",
            "bg_hue_strength": args.bg_hue_strength,
            "lambda_consistency": args.lambda_consistency,
            "lacn_focal_weight": args.lacn_focal_weight,
            "target_lab": target_lab.tolist(),
            "epochs_trained": args.epochs,
            "best_epoch": int(best_epoch),
            "smoke": args.smoke,
            "subset": args.subset,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
    }
    torch.save(save_payload, out_ckpt)
    log(f"\nCkpt saved: {out_ckpt}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
