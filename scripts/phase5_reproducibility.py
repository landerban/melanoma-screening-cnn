"""
phase5_reproducibility.py -- Phase 5 / A3 dry-run.

Verifies Phase 4d's seeding is end-to-end deterministic on a real batch:

  * Build train_loader the same way Trainer._run does.
  * Take ONE batch (image paths + tensors).
  * Forward through a freshly-built (seeded) model.
  * Repeat with the same seed in the same Python process.

Checks:
  * image-path lists must be identical (sampler reproducibility)
  * logits must match to ~1e-6 (model + augmentation + forward reproducibility)

If paths diverge, the sampler isn't getting a seeded generator. If paths
match but logits diverge by more than ~1e-5, there's a nondeterministic
op (cuDNN deterministic should have caught it; would be a real finding).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from efficientnet_b0 import (  # noqa: E402
    CFG,
    EfficientNetB0Classifier,
    SkinLesionDataset,
    get_transforms,
    make_weighted_sampler,
    patient_level_split,
)
from trainer import load_and_merge_metadata  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def one_batch_run(seed: int) -> tuple[list[str], torch.Tensor]:
    """
    Mirrors Trainer._run's data setup, then takes one batch and forwards.
    Returns the batch image-path list and the resulting logits (CPU tensor).
    """
    # 1. seed for model init
    seed_all(seed)
    device = torch.device("cuda")
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.eval()  # eval mode so dropout doesn't add nondeterminism

    # 2. re-seed for data ops
    seed_all(seed)
    df = load_and_merge_metadata("training_data", "training_data/images")
    train_df, val_df, cal_df, test_df = patient_level_split(
        df, patient_col="effective_patient_id"
    )

    paths  = train_df["image_path"].tolist()
    labels = train_df["label"].tolist()

    # Sampler with seeded generator (matches Trainer._run setup)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    sampler = make_weighted_sampler(labels, generator=sampler_gen)

    # Pull batch_size indices directly from the sampler (sidesteps DataLoader
    # worker nondeterminism; sampler IS the source of order)
    sampler_iter  = iter(sampler)
    batch_indices = [next(sampler_iter) for _ in range(CFG["batch_size"])]
    batch_paths   = [paths[i] for i in batch_indices]

    # Build dataset and read the batch. seed_all before reads so transform
    # augmentations are deterministic across runs.
    train_ds = SkinLesionDataset(paths, labels, get_transforms(CFG, "train"))
    seed_all(seed)
    imgs = torch.stack([train_ds[i][0] for i in batch_indices]).to(device)

    with torch.no_grad():
        logits = model(imgs).cpu()
    return batch_paths, logits


def main() -> int:
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return 1

    seed = int(CFG.get("seed", 42))
    print(f"Phase 5 / A3 -- reproducibility check (seed={seed})")
    print()

    print("Run 1 ...")
    paths1, logits1 = one_batch_run(seed)
    print(f"  batch size: {logits1.shape[0]}  logits shape: {tuple(logits1.shape)}")
    print(f"  logits[:3]: {logits1[:3].squeeze().tolist()}")

    print()
    print("Run 2 ...")
    paths2, logits2 = one_batch_run(seed)
    print(f"  batch size: {logits2.shape[0]}  logits shape: {tuple(logits2.shape)}")
    print(f"  logits[:3]: {logits2[:3].squeeze().tolist()}")

    print()
    paths_match = paths1 == paths2
    max_diff    = (logits1 - logits2).abs().max().item()
    mean_diff   = (logits1 - logits2).abs().mean().item()

    print("=" * 60)
    print(f"image paths identical : {paths_match}  (run1[0..3]: {paths1[:3]})")
    print(f"logits max |diff|     : {max_diff:.2e}")
    print(f"logits mean |diff|    : {mean_diff:.2e}")
    print()

    if not paths_match:
        print("  FAIL: image paths diverge -- sampler not deterministic across runs.")
        return 1
    if max_diff > 1e-5:
        print(f"  FAIL: logits diverge by {max_diff:.2e} > 1e-5 -- nondeterministic op somewhere.")
        return 1
    print(f"  OK: end-to-end deterministic. Same seed -> same paths, same logits to ~1e-6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
