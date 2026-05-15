"""
phase5_memory.py -- Phase 5 / A2 dry-run.

Forward + backward + optimizer.step() at the stage-2 config (backbone
unfrozen, full AdamW state) on a real-shape batch. Reports peak GPU
memory; threshold is 38 GB on the A100X MIG 3g.40gb (42.4 GB total,
~4 GB margin for fragmentation + DataLoader workers in real runs).

Uses random tensors, not real images -- the memory math is the same
and we save 5 min of disk IO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from efficientnet_b0 import CFG, EfficientNetB0Classifier  # noqa: E402


PEAK_LIMIT_GB = 38.0   # leave ~4 GB margin on the 42.4 GB MIG slice


def main() -> int:
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available -- run on GPU host")
        return 1

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    bs   = int(CFG["batch_size"])
    isz  = int(CFG["input_size"])
    print(f"Stage-2 config: batch={bs}  input={isz}x{isz}  backbone_lr={CFG['backbone_lr']}  head_lr_s2={CFG['head_lr_s2']}")

    # Mirror the Trainer._run stage-2 model setup
    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.unfreeze_backbone()
    model.train()

    # Mirror the Trainer._run stage-2 optimizer setup
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": CFG["backbone_lr"]},
            {"params": model.pool.parameters(),     "lr": CFG["backbone_lr"]},
            {"params": model.head.parameters(),     "lr": CFG["head_lr_s2"]},
        ],
        weight_decay=CFG["weight_decay"],
    )

    # Build a real-shape random batch
    imgs   = torch.randn(bs, 3, isz, isz, device=device)
    labels = torch.randint(0, 2, (bs, 1), device=device).float()

    # One full step: forward + loss + backward + optimizer
    opt.zero_grad()
    logits = model(imgs)
    loss   = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_resv  = torch.cuda.max_memory_reserved() / 1e9
    total      = torch.cuda.get_device_properties(0).total_memory / 1e9
    margin     = total - peak_resv

    print()
    print(f"peak_allocated : {peak_alloc:.2f} GB")
    print(f"peak_reserved  : {peak_resv:.2f} GB")
    print(f"total VRAM     : {total:.2f} GB")
    print(f"margin to OOM  : {margin:.2f} GB  (need >= 4 GB)")
    print(f"limit (38 GB)  : {'OK' if peak_alloc <= PEAK_LIMIT_GB else 'EXCEEDED'}")
    print()

    if peak_alloc > PEAK_LIMIT_GB:
        print(f"  STOP: peak {peak_alloc:.2f} GB exceeds {PEAK_LIMIT_GB:.0f} GB limit.")
        print(f"  Drop batch_size before Phase 6. Suggested: 64.")
        return 1
    print(f"  OK: peak {peak_alloc:.2f} GB <= {PEAK_LIMIT_GB:.0f} GB; safe for Phase 6 at batch={bs}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
