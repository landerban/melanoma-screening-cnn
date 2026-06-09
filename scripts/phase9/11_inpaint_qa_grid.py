"""
Inpainting quality visual QA — original vs inpainted side-by-side grid.

각 config 마다 대표 2 장 선정 → 원본 옆에 inpaint 결과 보여줌.
사용자가 *진짜로* 잘 지워졌는지 눈으로 검증 가능.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
QA_DIR    = REPO_ROOT / "artifacts" / "phase9" / "07_cf_inpainted_qa"
ORIG_DIR  = REPO_ROOT / "training_data" / "images"
FIG_DIR   = REPO_ROOT / "artifacts" / "phase9" / "figures"

CONFIGS = ["no_vignette", "no_ruler", "no_hair", "no_colorcast", "all_removed"]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 2 examples per config
    samples = {}
    for cfg in CONFIGS:
        files = sorted(QA_DIR.glob(f"*__{cfg}.jpg"))
        if len(files) >= 2:
            samples[cfg] = files[:2]
        else:
            samples[cfg] = files

    rows = len(CONFIGS)
    cols = 4  # orig1, inpaint1, orig2, inpaint2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))

    for r, cfg in enumerate(CONFIGS):
        for i, inpaint_path in enumerate(samples[cfg]):
            isic_id = inpaint_path.stem.split("__")[0]
            orig_path = ORIG_DIR / f"{isic_id}.jpg"
            if not orig_path.exists():
                continue

            orig = cv2.imread(str(orig_path))
            orig = cv2.resize(orig, (384, 384))
            inpaint = cv2.imread(str(inpaint_path))
            inpaint = cv2.resize(inpaint, (384, 384))

            ax_orig = axes[r, i * 2]
            ax_inp = axes[r, i * 2 + 1]
            ax_orig.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            ax_orig.set_title(f"{isic_id}\nORIGINAL", fontsize=9)
            ax_orig.axis("off")

            ax_inp.imshow(cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB))
            ax_inp.set_title(f"{cfg}", fontsize=9, color="red")
            ax_inp.axis("off")

        # Row label
        axes[r, 0].text(
            -0.2, 0.5, cfg.replace("no_", "").upper(),
            transform=axes[r, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=12, fontweight="bold",
        )

    plt.suptitle(
        "Inpainting Quality QA — Original vs Inpainted (red label)\n"
        "각 row 가 한 shortcut 의 제거 결과. 사용자가 *눈으로* 잘 지워졌는지 확인.",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out_path = FIG_DIR / "fig_inpaint_qa_grid.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.relative_to(REPO_ROOT)}")
    print(f"\n각 config 의 *모든* QA 이미지: {QA_DIR.relative_to(REPO_ROOT)}/")
    print("파일명 형식: {isic_id}__{config}.jpg")
    print("config 종류:", ", ".join(CONFIGS))


if __name__ == "__main__":
    main()
