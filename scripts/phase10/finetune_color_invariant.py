"""
Phase 10 — Color-invariant fine-tune intervention.

Brief Stage-2 fine-tune of the Phase 6 best_model.pth with HSV jitter
augmentation added to the training transform. Tests whether the
color-cast shortcut Phase 9 identified can be reduced via training
intervention without destroying in-distribution AUC.

Run on Colab A100 or M3 MPS:

    python scripts/phase10/finetune_color_invariant.py \
        --base-ckpt best_model.pth \
        --out-ckpt artifacts/phase10/best_model_color_invariant.pth \
        --epochs 5 --lr 1e-5 --batch-size 96

Pre-registered in docs/phase10/01_preregistration.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from efficientnet_b0 import (  # noqa: E402
    CFG, EfficientNetB0Classifier, FocalLoss,
    SkinLesionDataset, make_weighted_sampler, patient_level_split,
)
from trainer import load_and_merge_metadata  # noqa: E402


# ------------------------------------------------------------------
# Phase 10 transform — Phase 6's val/train + HSV jitter ADDED
# ------------------------------------------------------------------

def get_phase10_train_transform(cfg: dict) -> transforms.Compose:
    """
    Phase 10 training transform = Phase 6 transform + ColorJitter
    extended with HUE-only jitter (saturation jitter omitted).

    Rationale (post-Phase-11 update):
      Phase 11 (docs/phase11/02_results.md) decomposed the color-cast
      shortcut by HSV channel and found:
        hue:        ΔP = -0.243  (93% of joint effect, p < 1e-20)
        saturation: ΔP = +0.010  (null, p = 0.44)
        value:      ΔP = -0.005  (null, p = 0.89)
      Saturation jitter is therefore *mechanistically unnecessary* for
      breaking the shortcut. Hue jitter is the only intervention that
      maps to the identified causal cue.

      Initial pre-registration used (sat=0.15, hue=0.05); the amended
      protocol uses (sat=0.0, hue=0.10) — saturation off, hue jitter
      doubled to compensate for the removed channel.

      The decision and its empirical justification are logged in
      docs/phase10/01_preregistration.md §"Amendment 2026-06-09".
    """
    size = cfg["input_size"]
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        # KEY CHANGE: hue-only jitter (saturation off) vs Phase 6
        # Phase-11-informed: see Phase 11 channel decomposition above.
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.0, hue=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


def get_val_transform(cfg: dict) -> transforms.Compose:
    """Identical to Phase 6 val."""
    size = cfg["input_size"]
    return transforms.Compose([
        transforms.Resize(int(size * 1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ------------------------------------------------------------------
# Eval helper
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs_l, labels_l = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        p = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
        probs_l.append(p)
        labels_l.append(lbls.numpy().astype(int))
    probs = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return dict(auc=float(auc), probs=probs, labels=labels)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="best_model.pth")
    parser.add_argument("--out-ckpt",
                        default="artifacts/phase10/best_model_color_invariant.pth")
    parser.add_argument("--log-path",
                        default="artifacts/phase10/01_finetune.log")
    parser.add_argument("--training-data", default="training_data")
    parser.add_argument("--images-dir", default="training_data/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # Output paths
    out_ckpt = REPO_ROOT / args.out_ckpt
    log_path = REPO_ROOT / args.log_path
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"# Phase 10 fine-tune log",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"base_ckpt: {args.base_ckpt}",
        f"device: {device}",
        f"epochs: {args.epochs}, lr: {args.lr}, batch: {args.batch_size}",
        "",
    ]

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # Data
    log("## Loading metadata")
    df = load_and_merge_metadata(args.training_data, args.images_dir)
    log(f"Merged {len(df):,} rows")
    train_df, val_df, cal_df, test_df = patient_level_split(
        df, patient_col="effective_patient_id", seed=args.seed,
    )
    log(f"train={len(train_df):,} val={len(val_df):,} cal={len(cal_df):,} test={len(test_df):,}")

    cfg = dict(CFG)

    # Datasets
    train_tf = get_phase10_train_transform(cfg)
    val_tf = get_val_transform(cfg)
    train_ds = SkinLesionDataset(train_df["image_path"].tolist(),
                                  train_df["label"].tolist(), train_tf)
    val_ds = SkinLesionDataset(val_df["image_path"].tolist(),
                                val_df["label"].tolist(), val_tf)
    test_ds = SkinLesionDataset(test_df["image_path"].tolist(),
                                 test_df["label"].tolist(), val_tf)

    gen = torch.Generator(); gen.manual_seed(args.seed)
    sampler = make_weighted_sampler(train_df["label"].tolist(), generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model — start from Phase 6 ckpt
    log("\n## Loading base ckpt")
    obj = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(state_dict)
    log(f"Base ckpt loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=cfg["weight_decay"])
    criterion = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])

    # Pre-fine-tune eval
    log("\n## Pre-fine-tune eval (base ckpt on val + test)")
    val_pre = evaluate(model, val_loader, device)
    test_pre = evaluate(model, test_loader, device)
    log(f"  pre-FT val  AUC = {val_pre['auc']:.4f}")
    log(f"  pre-FT test AUC = {test_pre['auc']:.4f}")

    # Fine-tune loop
    log("\n## Fine-tune trajectory")
    best_val_auc = val_pre["auc"]
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for imgs, lbls in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            n_batches += 1
        train_loss = ep_loss / max(1, n_batches)

        val_e = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log(f"  epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_AUC={val_e['auc']:.4f}, time={dt:.0f}s")

        if val_e["auc"] > best_val_auc:
            best_val_auc = val_e["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            log(f"    new best val AUC; checkpointing")

    log(f"\nBest val AUC = {best_val_auc:.4f} at epoch {best_epoch}")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final test eval
    test_post = evaluate(model, test_loader, device)
    log(f"\n## Final test eval")
    log(f"  post-FT test AUC = {test_post['auc']:.4f} (pre-FT was {test_pre['auc']:.4f})")
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
        "phase10_modifications": {
            "augmentation_added": "saturation=0.0, hue=0.10 (Phase 11-informed)",
            "lr": args.lr,
            "epochs_trained": args.epochs,
            "best_epoch": int(best_epoch),
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
