"""
Phase 10 / Step 6 -- Integrated Gradients (IG) channel attribution.

Motivation:
  Grad-CAM (Phase 10 / fig13_v2) showed no significant peak-on-lesion
  shift between Phase 6 and Phase 10. That measurement is correct but
  the wrong instrument: Grad-CAM sums activations across channels and
  yields a spatial heatmap, so a *channel-level* sensitivity change
  cannot appear in it.

  IG (Integrated Gradients, Sundararajan et al. 2017) gives per-pixel
  per-input-channel attribution scores. We use IG to test two
  hypotheses that the audit -> intervention story implies:

  H_bg: Phase 10 reduces attribution magnitude over *background*
        pixels relative to *foreground* (lesion) pixels.
  H_rgb: Phase 10 reduces RGB-channel imbalance, i.e. R / G / B
        attribution shares are more uniform than Phase 6.

  Both are measurements Grad-CAM cannot make.

Method:
  - Sample 50 images from the Phase 9 analytical cohort, stratified by
    (collection, label). Reuse Otsu masks already in artifacts/phase9/
    05_lesion_masks/.
  - For each image, run IG with each ckpt (Phase 6, Phase 10) using a
    zero baseline and 50 integration steps.
  - Aggregate per-image: bg_attr_mean = mean |IG| over background;
    fg_attr_mean = mean |IG| over foreground; bg_fg_ratio = bg/fg.
  - Aggregate per-image RGB shares: r_share = sum |IG_R| / sum |IG_all|,
    similarly G, B.
  - Paired statistics (Wilcoxon) on per-image deltas Phase10 - Phase6.

Outputs:
  artifacts/phase10/06_ig_attribution.csv
  artifacts/phase10/06_ig_attribution.log
  artifacts/phase10/figures/fig21_ig_bg_fg_ratio.png
  artifacts/phase10/figures/fig22_ig_attribution_examples.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from tqdm import tqdm
from captum.attr import IntegratedGradients

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase9"))

from efficientnet_b0 import CFG, EfficientNetB0Classifier, get_transforms  # noqa: E402
from shortcut_detect import load_image_resized, IMG_SIZE  # noqa: E402

# Paths
P9_SAMPLE_CSV  = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
P9_LESION_DIR  = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_masks"
P9_LESION_STATS = REPO_ROOT / "artifacts" / "phase9" / "05_lesion_mask_stats.csv"
IMG_DIR        = REPO_ROOT / "training_data" / "images"
P6_CKPT        = REPO_ROOT / "best_model.pth"
P10_CKPT       = REPO_ROOT / "artifacts" / "phase10" / "best_model_color_invariant.pth"

OUT_CSV  = REPO_ROOT / "artifacts" / "phase10" / "06_ig_attribution.csv"
LOG_TXT  = REPO_ROOT / "artifacts" / "phase10" / "06_ig_attribution.log"
FIG_DIR  = REPO_ROOT / "artifacts" / "phase10" / "figures"

N_SAMPLE = 60
IG_STEPS = 32  # 32 keeps total runtime ~10min on M3 MPS for 60*2 images
SEED = 42


def pick_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    m = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


def stratified_sample(df, n=N_SAMPLE, seed=SEED):
    """Stratified sample by (collection, label). ~10 per stratum."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in sorted(df.source_collection.astype(str).unique()):
        for lbl in [0, 1]:
            sub = df[(df.source_collection.astype(str) == c) & (df.label == lbl)]
            take = min(n // 6, len(sub))  # 3 colls * 2 labels = 6 strata
            parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def load_image_tensor(isic_id, transform, device):
    p = IMG_DIR / f"{isic_id}.jpg"
    if not p.exists():
        return None
    img_bgr = load_image_resized(p, size=IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    return x, img_bgr


def load_mask_at_input_size(isic_id):
    """Load Otsu lesion mask and downsize to the input resolution."""
    p = P9_LESION_DIR / f"{isic_id}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # Resize to model input size
    in_size = CFG["input_size"]
    if m.shape != (in_size, in_size):
        m = cv2.resize(m, (in_size, in_size),
                       interpolation=cv2.INTER_NEAREST)
    return m


def run_ig(model, x, device, n_steps=IG_STEPS):
    """IG attribution for the malignant logit. Returns (3, H, W) numpy."""
    model.zero_grad()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x).to(device)
    # Captum target arg: index into output. EfficientNetB0Classifier outputs
    # a single logit per image (binary). target=None for scalar output.
    attr = ig.attribute(x, baselines=baseline, target=None, n_steps=n_steps,
                        internal_batch_size=1)
    return attr.squeeze(0).detach().cpu().numpy()  # (3, H, W)


def aggregate_attribution(attr, mask):
    """Aggregate IG attribution into bg/fg and RGB stats.

    attr: (3, H, W) numpy
    mask: (H, W) uint8 with 0/255 (foreground=255 = lesion)

    Returns dict with bg_attr_mean, fg_attr_mean, bg_fg_ratio,
    r_share, g_share, b_share, total_abs_attribution.
    """
    abs_attr = np.abs(attr)  # (3, H, W)
    abs_attr_2d = abs_attr.sum(axis=0)  # (H, W)

    inside = mask > 127
    outside = ~inside
    fg_mean = float(abs_attr_2d[inside].mean()) if inside.sum() > 0 else 0.0
    bg_mean = float(abs_attr_2d[outside].mean()) if outside.sum() > 0 else 0.0
    bg_fg = bg_mean / max(fg_mean, 1e-9)

    r_sum = float(abs_attr[0].sum())
    g_sum = float(abs_attr[1].sum())
    b_sum = float(abs_attr[2].sum())
    total = r_sum + g_sum + b_sum + 1e-12
    return dict(
        bg_attr_mean=bg_mean,
        fg_attr_mean=fg_mean,
        bg_fg_ratio=bg_fg,
        r_share=r_sum / total,
        g_share=g_sum / total,
        b_share=b_sum / total,
        total_abs=total,
    )


def make_paired_figure(df):
    """fig21 -- paired comparison of bg/fg ratio and RGB shares."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: paired scatter of bg_fg_ratio
    ax = axes[0]
    ax.scatter(df.bg_fg_ratio_p6, df.bg_fg_ratio_p10, alpha=0.6, color="steelblue")
    lim_lo = float(min(df.bg_fg_ratio_p6.min(), df.bg_fg_ratio_p10.min()) * 0.9)
    lim_hi = float(max(df.bg_fg_ratio_p6.max(), df.bg_fg_ratio_p10.max()) * 1.05)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("Phase 6 bg/fg ratio")
    ax.set_ylabel("Phase 10 bg/fg ratio")
    ax.set_title("bg/fg attribution ratio per image")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: distribution of delta (Phase10 - Phase6) per image
    ax = axes[1]
    delta = df.bg_fg_ratio_p10 - df.bg_fg_ratio_p6
    ax.hist(delta, bins=20, color="darkorange", alpha=0.75, edgecolor="black")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(delta.mean(), color="red", lw=2,
               label=f"mean = {delta.mean():+.3f}")
    ax.set_xlabel("Δ(bg/fg ratio) = Phase 10 - Phase 6")
    ax.set_ylabel("count")
    ax.set_title(f"Per-image delta (N={len(df)})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: RGB shares
    ax = axes[2]
    p6_shares = [df.r_share_p6.mean(), df.g_share_p6.mean(), df.b_share_p6.mean()]
    p10_shares = [df.r_share_p10.mean(), df.g_share_p10.mean(), df.b_share_p10.mean()]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, p6_shares, w, label="Phase 6", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, p10_shares, w, label="Phase 10", color="seagreen", alpha=0.85)
    for xi, v in zip(x - w/2, p6_shares):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, p10_shares):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["R", "G", "B"])
    ax.set_title("Mean RGB-channel attribution share")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Phase 10 / fig21 -- IG attribution: bg/fg shift and RGB-share rebalance",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig21_ig_bg_fg_ratio.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.relative_to(REPO_ROOT)}")


def make_examples_figure(examples, df_summary):
    """fig22 -- 4 example images: original, Phase 6 |IG|, Phase 10 |IG|."""
    n = min(len(examples), 5)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, ex in enumerate(examples[:n]):
        img_rgb = cv2.cvtColor(ex["img_bgr"], cv2.COLOR_BGR2RGB)
        attr_p6 = ex["attr_p6"]
        attr_p10 = ex["attr_p10"]
        mask = ex["mask"]

        attr_p6_2d = np.abs(attr_p6).sum(axis=0)
        attr_p10_2d = np.abs(attr_p10).sum(axis=0)
        vmax = max(attr_p6_2d.max(), attr_p10_2d.max())

        # Outline lesion on original
        ax = axes[i, 0]
        ax.imshow(img_rgb)
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color="yellow", lw=1.2)
        ax.set_title(f"{ex['isic_id']}\nc={ex['coll']}, label={ex['label']}",
                     fontsize=8)
        ax.axis("off")

        ax = axes[i, 1]
        ax.imshow(attr_p6_2d, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"Phase 6 |IG|\nbg/fg={df_summary.bg_fg_ratio_p6.iloc[ex['row_idx']]:.2f}",
                     fontsize=9, color="darkred")
        ax.axis("off")

        ax = axes[i, 2]
        ax.imshow(attr_p10_2d, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"Phase 10 |IG|\nbg/fg={df_summary.bg_fg_ratio_p10.iloc[ex['row_idx']]:.2f}",
                     fontsize=9, color="darkgreen")
        ax.axis("off")

    plt.suptitle("Phase 10 / fig22 -- example IG attribution maps",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    out = FIG_DIR / "fig22_ig_attribution_examples.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.relative_to(REPO_ROOT)}")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log = ["# Phase 10 / Step 6 -- Integrated Gradients channel attribution", ""]

    df = pd.read_csv(P9_SAMPLE_CSV)
    rel_df = pd.read_csv(P9_LESION_STATS)
    reliable_ids = set(rel_df[rel_df.reliable].isic_id)
    df = df[df.isic_id.isin(reliable_ids)]
    log.append(f"Phase 9 sample restricted to reliable masks: N = {len(df)}")

    sample = stratified_sample(df, N_SAMPLE, SEED)
    log.append(f"Stratified sample: N = {len(sample)} (per-stratum target = {N_SAMPLE // 6})")
    log.append(f"IG steps: {IG_STEPS}")
    log.append("")

    examples = []
    if OUT_CSV.exists():
        out_df = pd.read_csv(OUT_CSV)
        print(f"[cached] {OUT_CSV.relative_to(REPO_ROOT)} -- N={len(out_df)}")
        log.append(f"## Per-image attribution loaded from cache, N = {len(out_df)}")
    else:
        device = pick_device()
        print(f"[device] {device}")
        transform = get_transforms(CFG, "val")
        print(f"Loading Phase 6 ckpt: {P6_CKPT.relative_to(REPO_ROOT)}")
        m_p6 = load_model(P6_CKPT, device)
        print(f"Loading Phase 10 ckpt: {P10_CKPT.relative_to(REPO_ROOT)}")
        m_p10 = load_model(P10_CKPT, device)

        rows = []
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="IG attribution"):
            isic_id = row["isic_id"]
            loaded = load_image_tensor(isic_id, transform, device)
            if loaded is None:
                continue
            x, img_bgr = loaded
            mask = load_mask_at_input_size(isic_id)
            if mask is None:
                continue

            attr_p6 = run_ig(m_p6, x, device)
            attr_p10 = run_ig(m_p10, x, device)

            stats_p6 = aggregate_attribution(attr_p6, mask)
            stats_p10 = aggregate_attribution(attr_p10, mask)

            out_row = {
                "isic_id": isic_id,
                "coll": str(row["source_collection"]),
                "label": int(row["label"]),
                **{f"{k}_p6": v for k, v in stats_p6.items()},
                **{f"{k}_p10": v for k, v in stats_p10.items()},
            }
            rows.append(out_row)

            # Stash up to 5 examples for fig22
            if len(examples) < 5:
                examples.append(dict(
                    isic_id=isic_id, coll=str(row["source_collection"]),
                    label=int(row["label"]),
                    img_bgr=img_bgr, mask=mask,
                    attr_p6=attr_p6, attr_p10=attr_p10,
                    row_idx=len(rows) - 1,
                ))

        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUT_CSV, index=False)
        log.append(f"## Per-image attribution computed for N = {len(out_df)} images")

    # --- Statistics ---
    log.append("")
    log.append("## H_bg -- background/foreground attribution ratio")
    delta_bgfg = out_df.bg_fg_ratio_p10 - out_df.bg_fg_ratio_p6
    n = len(out_df)
    rng = np.random.default_rng(SEED)
    boots = np.empty(5000)
    arr = delta_bgfg.values
    for i in range(5000):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    try:
        w, p = stats.wilcoxon(arr, alternative="two-sided", zero_method="zsplit")
    except ValueError:
        w, p = float("nan"), 1.0
    log.append(f"  Mean bg/fg Phase 6  = {out_df.bg_fg_ratio_p6.mean():.4f}  "
               f"[median {out_df.bg_fg_ratio_p6.median():.4f}]")
    log.append(f"  Mean bg/fg Phase 10 = {out_df.bg_fg_ratio_p10.mean():.4f}  "
               f"[median {out_df.bg_fg_ratio_p10.median():.4f}]")
    log.append(f"  Paired delta (P10 - P6): mean = {arr.mean():+.4f}  "
               f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    log.append(f"  Wilcoxon p-value: {p:.4g}")
    h_bg = (arr.mean() < 0) and (ci_hi < 0)
    log.append(f"  H_bg supported (delta < 0, CI excludes 0)? {h_bg}")

    log.append("")
    log.append("## H_rgb -- RGB-channel share rebalance")
    for ch in ["r_share", "g_share", "b_share"]:
        d = out_df[f"{ch}_p10"] - out_df[f"{ch}_p6"]
        boots = np.empty(5000)
        arr = d.values
        for i in range(5000):
            idx = rng.integers(0, n, size=n)
            boots[i] = arr[idx].mean()
        lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        try:
            _, pp = stats.wilcoxon(arr, alternative="two-sided", zero_method="zsplit")
        except ValueError:
            pp = 1.0
        log.append(f"  Δ{ch}: mean = {arr.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                   f"Wilcoxon p = {pp:.4g}")

    # std of shares: high std = lopsided; reduction = more uniform
    p6_share_stds = np.std(out_df[["r_share_p6", "g_share_p6", "b_share_p6"]].values, axis=1)
    p10_share_stds = np.std(out_df[["r_share_p10", "g_share_p10", "b_share_p10"]].values, axis=1)
    log.append("")
    log.append(f"  Per-image RGB-share std (Phase 6  mean): {p6_share_stds.mean():.4f}")
    log.append(f"  Per-image RGB-share std (Phase 10 mean): {p10_share_stds.mean():.4f}")
    h_rgb = p10_share_stds.mean() < p6_share_stds.mean()
    log.append(f"  H_rgb supported (Phase 10 share std < Phase 6 share std)? {h_rgb}")

    log.append("")
    log.append("## Files emitted")
    log.append(f"  CSV: {OUT_CSV.relative_to(REPO_ROOT)}")

    # Figures
    print("\n=== Generate figures ===")
    make_paired_figure(out_df)
    if examples:
        make_examples_figure(examples, out_df)
        log.append(f"  Figure: {(FIG_DIR / 'fig22_ig_attribution_examples.png').relative_to(REPO_ROOT)}")
    else:
        log.append("  Figure: fig22 skipped (no preselected example images matched the sample)")
    log.append(f"  Figure: {(FIG_DIR / 'fig21_ig_bg_fg_ratio.png').relative_to(REPO_ROOT)}")

    LOG_TXT.write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
