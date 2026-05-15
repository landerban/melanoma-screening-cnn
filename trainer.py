"""
trainer.py — Background training thread + shared state.
Consumed by app.py (Gradio UI).
"""

import copy
import json
import os
import random
import re
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, roc_curve,
)

from efficientnet_b0 import (
    EfficientNetB0Classifier,
    SkinLesionDataset,
    build_loss,
    get_transforms,
    make_weighted_sampler,
    patient_level_split,
    CFG,
)


# ---------------------------------------------------------------------------
# Thread-safe training state
# ---------------------------------------------------------------------------

class TrainingState:
    """
    Shared object read by the UI thread and written by the trainer thread.
    All writes go through a lock; reads are best-effort (display only).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.running          = False
            self.stop_requested   = False
            self.stage            = 0        # 0=idle  1=stage1  2=stage2
            self.epoch            = 0
            self.total_epochs     = 0
            self.train_losses     : list[float] = []
            self.val_losses       : list[float] = []
            self.val_aucs         : list[float] = []
            self.val_pr_aucs      : list[float] = []
            self.val_recalls      : list[float] = []
            self.val_specs        : list[float] = []
            self.val_f1s          : list[float] = []
            self.es_counter       = 0
            self.best_auc         = 0.0
            self.best_val_loss    = float("inf")   # kept for display only
            self.optimal_threshold: float        = 0.5
            self.best_model_path  : str | None  = None
            self.test_metrics     : dict | None = None  # Phase 4b: held-out test eval
            self.log              : list[str]   = []
            self.error            : str | None  = None
            self.done             = False

    def log_line(self, msg: str):
        print(msg, flush=True)
        with self._lock:
            self.log.append(msg)

    def snapshot_log(self, tail: int = 40) -> str:
        with self._lock:
            return "\n".join(self.log[-tail:])


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.
    Hooks the last MBConv block (backbone[-1]).
    """

    def __init__(self, model: EfficientNetB0Classifier):
        self.model       = model
        self._acts       = None
        self._grads      = None
        target = model.backbone[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _i, out):
        self._acts = out.detach()

    def _bwd_hook(self, _m, _gi, gout):
        self._grads = gout[0].detach()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        img_tensor : (C, H, W) on the model's device — no batch dim.
        Returns    : (H, W) heatmap normalised to [0, 1].
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0)
        logit = self.model(x)
        self.model.zero_grad()
        logit.backward()

        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, img_tensor.shape[1:],
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_merge_metadata(training_data_dir: str, images_dir: str) -> "pd.DataFrame":
    """
    Scans `training_data_dir` for all CSVs, combines them, deduplicates on
    isic_id, drops Indeterminate, maps labels, and filters to images on disk.

    Returns a DataFrame with columns:
        isic_id, patient_id (optional), lesion_id (optional), diagnosis_1,
        source_collection, effective_patient_id, label, image_path.

    Phase 4a additions:
      * source_collection -- parsed from each CSV's filename
        (`metadata_c<id>.csv` -> `<id>`). Used downstream for the per-collection
        AUC breakdown (audit W8) and modality-shortcut analysis.
      * effective_patient_id -- patient_id || lesion_id || isic_id fallback
        chain. Built BEFORE drop_duplicates so each row's grouping is fixed at
        ingest. Guaranteed non-NaN; Trainer._run asserts. See
        docs/diagnostic_findings.md for the per-collection grouping breakdown
        that justifies this fallback chain.
    """

    # Scan training_data/ and any immediate subdirectories (e.g. images/)
    root = Path(training_data_dir)
    csvs = list(root.glob("*.csv")) + list(root.glob("*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {training_data_dir} or its subdirectories")

    collection_re = re.compile(r"metadata_c(\d+)\.csv$")
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
            # Phase 4a (W8): tag source_collection from filename. Per-collection
            # AUC breakdown in Phase 7c needs this; it also lets us track which
            # rows came from which collection through the merge.
            m = collection_re.search(p.name)
            df["source_collection"] = m.group(1) if m else None
            frames.append(df)
        except Exception:
            pass

    combined = pd.concat(frames, ignore_index=True)

    # Phase 4a: build effective_patient_id via patient_id || lesion_id || isic_id.
    # Done BEFORE drop_duplicates so each row's grouping is fixed at ingest.
    if "patient_id" in combined.columns:
        eff = combined["patient_id"].copy()
    else:
        eff = pd.Series([pd.NA] * len(combined), index=combined.index, dtype=object)
    if "lesion_id" in combined.columns:
        eff = eff.fillna(combined["lesion_id"])
    n_fb_isic_id = int(eff.isna().sum())
    if "isic_id" in combined.columns:
        eff = eff.fillna(combined["isic_id"])
    if n_fb_isic_id > 0:
        warnings.warn(
            f"effective_patient_id fell back to isic_id for {n_fb_isic_id:,} rows "
            f"(no patient_id and no lesion_id). Those rows are image-level-split "
            f"disguised as patient-level. Disclose on slide."
        )
    combined["effective_patient_id"] = eff

    # Keep only columns we need; tolerate missing ones
    keep = [c for c in ["isic_id", "patient_id", "lesion_id", "diagnosis_1",
                        "source_collection", "effective_patient_id"]
            if c in combined.columns]
    combined = combined[keep].copy()

    # Deduplicate on isic_id
    if "isic_id" in combined.columns:
        combined.drop_duplicates(subset="isic_id", inplace=True)

    # Binary label
    combined = combined[combined["diagnosis_1"].isin(["Benign", "Malignant"])].reset_index(drop=True)
    combined["label"] = (combined["diagnosis_1"] == "Malignant").astype(int)

    # Build a lookup of stem→full_path by scanning the directory ONCE (O(n) not O(n*k))
    imgs = Path(images_dir)
    stem_to_path: dict[str, str] = {}
    for f in imgs.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            stem_to_path[f.stem] = str(f)

    combined["image_path"] = combined["isic_id"].map(stem_to_path)
    combined = combined[combined["image_path"].notna()].reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Runs the full two-stage training loop in a background thread.
    Writes progress into `state` (TrainingState) after every epoch.
    """

    CKPT = "best_model.pth"

    def __init__(
        self,
        training_data_dir : str,
        images_dir        : str,
        cfg               : dict,
        state             : TrainingState,
    ):
        self.training_data_dir = training_data_dir
        self.images_dir        = images_dir
        self.cfg               = cfg
        self.state             = state

    # ------------------------------------------------------------------

    def launch(self):
        t = threading.Thread(target=self._safe_run, daemon=True)
        t.start()
        return t

    def _safe_run(self):
        try:
            self._run()
        except Exception as exc:
            self.state.error   = str(exc)
            self.state.running = False
            self.state.log_line(f"[ERROR] {exc}")
            raise

    # ------------------------------------------------------------------

    def _run(self):
        s   = self.state
        cfg = self.cfg

        # ---- Phase 4d: reproducibility ----
        # Seeds python.random, numpy, torch (CPU + CUDA), cuDNN. DataLoader
        # workers and the WeightedRandomSampler get separate generators seeded
        # below. Run-to-run AUC delta should be < ~0.001 with this in place;
        # if it's larger, something nondeterministic slipped in.
        seed = int(cfg.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        s.log_line(f"Seeded with seed={seed} (random, numpy, torch, cuDNN deterministic)")

        # ---- Data ----
        s.log_line("Loading and merging metadata CSVs …")
        df = load_and_merge_metadata(self.training_data_dir, self.images_dir)

        n_pos = int((df.label == 1).sum())
        n_neg = int((df.label == 0).sum())
        s.log_line(f"Dataset ready → {len(df)} images  |  Benign: {n_neg}  Malignant: {n_pos}")

        if n_pos == 0:
            raise RuntimeError("No malignant images found on disk — check images_dir.")

        # Phase 4a: explicit effective_patient_id key; no silent isic_id fallback.
        # load_and_merge_metadata guarantees effective_patient_id is non-NaN;
        # patient_level_split also re-validates and will hard-fail if violated.
        assert "effective_patient_id" in df.columns, \
            "load_and_merge_metadata must produce effective_patient_id (Phase 4a)"
        assert df["effective_patient_id"].notna().all(), \
            f"effective_patient_id has {int(df['effective_patient_id'].isna().sum())} NaN rows"
        # Phase 4c: four-way split. cal_df owns the threshold sweep so the
        # operating threshold isn't selected on the same cohort that drove ES
        # and best-epoch selection (audit weakness W1). test_df is the
        # held-out headline cohort (W2).
        train_df, val_df, cal_df, test_df = patient_level_split(
            df, patient_col="effective_patient_id"
        )
        s.log_line(
            f"Split → train: {len(train_df)}  val: {len(val_df)}  "
            f"cal: {len(cal_df)}  test: {len(test_df)}"
        )

        train_ds = SkinLesionDataset(
            train_df["image_path"].tolist(),
            train_df["label"].tolist(),
            get_transforms(cfg, "train"),
        )
        val_ds = SkinLesionDataset(
            val_df["image_path"].tolist(),
            val_df["label"].tolist(),
            get_transforms(cfg, "val"),
        )
        cal_ds = SkinLesionDataset(
            cal_df["image_path"].tolist(),
            cal_df["label"].tolist(),
            get_transforms(cfg, "val"),
        )
        test_ds = SkinLesionDataset(
            test_df["image_path"].tolist(),
            test_df["label"].tolist(),
            get_transforms(cfg, "val"),
        )

        nw       = min(cfg.get("num_workers", 4), os.cpu_count() or 1)
        use_cuda = torch.cuda.is_available()

        # Phase 4d: per-worker seeding so DataLoader workers are deterministic.
        # Each worker gets seed + worker_id; numpy and python.random inside
        # the worker use that.
        def _worker_init_fn(worker_id: int) -> None:
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Phase 4d: seeded generator for the WeightedRandomSampler so the
        # train batch composition is reproducible.
        sampler_gen = torch.Generator()
        sampler_gen.manual_seed(seed)

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            sampler=make_weighted_sampler(train_df["label"].tolist(), generator=sampler_gen),
            num_workers=nw, pin_memory=use_cuda,
            worker_init_fn=_worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=nw, pin_memory=use_cuda,
            worker_init_fn=_worker_init_fn,
        )
        cal_loader = DataLoader(
            cal_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=nw, pin_memory=use_cuda,
            worker_init_fn=_worker_init_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=nw, pin_memory=use_cuda,
            worker_init_fn=_worker_init_fn,
        )

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s.log_line(f"Device: {device}")

        model     = EfficientNetB0Classifier(freeze_backbone=True).to(device)
        criterion = build_loss(cfg, n_pos, n_neg).to(device)

        # ---- Stage 1 ----
        s.stage = 1
        s.log_line("═" * 50)
        s.log_line("Stage 1 — backbone frozen, training head only")
        s.log_line("═" * 50)

        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["head_lr"], weight_decay=cfg["weight_decay"],
        )
        sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt1, mode="min",
            factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"],
        )

        for ep in range(1, cfg["stage1_epochs"] + 1):
            if s.stop_requested:
                break
            s.epoch       = ep
            s.total_epochs = cfg["stage1_epochs"]

            tr  = self._train_ep(model, train_loader, criterion, opt1, device)
            vl, m = self._val_ep(model, val_loader, criterion, device)
            sch1.step(vl)
            self._record(s, tr, vl, m)
            s.log_line(
                f"  [S1 {ep:>2}/{cfg['stage1_epochs']}] "
                f"loss={tr:.4f}  val={vl:.4f}  "
                f"AUC={m['auc']:.3f}  recall={m['recall']:.3f}"
            )

        # ---- Stage 2 ----
        s.stage = 2
        s.log_line("═" * 50)
        s.log_line("Stage 2 — full model fine-tune  (ES monitors AUC, not loss)")
        s.log_line("═" * 50)

        model.unfreeze_backbone()
        opt2 = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg["backbone_lr"]},
                {"params": model.pool.parameters(),     "lr": cfg["backbone_lr"]},
                {"params": model.head.parameters(),     "lr": cfg["head_lr_s2"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
        # CosineAnnealingLR: smoothly decays LR without aggressive plateau collapse
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=cfg["stage2_epochs"], eta_min=1e-7,
        )

        es_counter   = 0
        best_auc     = 0.0
        best_weights = None

        for ep in range(1, cfg["stage2_epochs"] + 1):
            if s.stop_requested:
                break
            s.epoch        = ep
            s.total_epochs = cfg["stage2_epochs"]

            tr     = self._train_ep(model, train_loader, criterion, opt2, device)
            vl, m  = self._val_ep(model, val_loader, criterion, device)
            sch2.step()
            self._record(s, tr, vl, m)

            cur_auc = m["auc"]
            # ---- Early stopping on AUC (maximize), not val_loss ----
            if cur_auc > best_auc + cfg["es_delta"]:
                best_auc     = cur_auc
                es_counter   = 0
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, self.CKPT)
                s.best_model_path  = self.CKPT
                s.best_auc         = best_auc
                s.best_val_loss    = vl      # keep for display
                marker = " ✓ (best AUC)"
            else:
                es_counter += 1
                marker = f" ({es_counter}/{cfg['es_patience']})"

            cm_str = (f"  CM: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}"
                      if "tn" in m else "")
            s.es_counter = es_counter
            s.log_line(
                f"  [S2 {ep:>2}/{cfg['stage2_epochs']}] "
                f"loss={tr:.4f}  val={vl:.4f}  "
                f"AUC={cur_auc:.4f}  PR-AUC={m['pr_auc']:.4f}  "
                f"recall={m['recall']:.3f}  spec={m['specificity']:.3f}  "
                f"F1={m['f1']:.3f}{marker}"
            )
            if cm_str:
                s.log_line(cm_str)

            if es_counter >= cfg["es_patience"]:
                s.log_line(f"Early stopping at epoch {ep}. Best AUC: {best_auc:.4f}")
                break

        if best_weights:
            model.load_state_dict(best_weights)

        # ---- Phase 4c: threshold sweep on the held-out CAL set (not val) ----
        # The cal cohort was never touched during training or early stopping,
        # so the resulting threshold is honest -- this addresses audit W1.
        s.log_line("─" * 50)
        s.log_line("Threshold sweep on held-out cal set …")
        opt_thresh = self._sweep_threshold(model, cal_loader, device)
        s.optimal_threshold = opt_thresh
        s.log_line(f"Optimal threshold (recall≥0.80, max specificity): {opt_thresh:.3f}")

        # ---- Phase 4b: held-out test evaluation at the calibrated threshold ----
        # The headline AUC for the midterm comes from this number, not val.
        s.log_line("─" * 50)
        s.log_line(f"Held-out test evaluation at threshold {opt_thresh:.3f} …")
        test_loss, test_m = self._val_ep(model, test_loader, criterion, device, threshold=opt_thresh)
        test_m["loss"]      = float(test_loss)
        test_m["threshold"] = float(opt_thresh)
        test_m["n_pos"]     = int(test_m.get("tp", 0) + test_m.get("fn", 0))
        test_m["n_neg"]     = int(test_m.get("tn", 0) + test_m.get("fp", 0))
        s.test_metrics      = test_m
        s.log_line(
            f"  TEST  AUC={test_m['auc']:.4f}  PR-AUC={test_m['pr_auc']:.4f}  "
            f"recall={test_m['recall']:.3f}  spec={test_m['specificity']:.3f}  "
            f"F1={test_m['f1']:.3f}"
        )
        s.log_line(
            f"  TEST  CM: TN={test_m['tn']} FP={test_m['fp']} "
            f"FN={test_m['fn']} TP={test_m['tp']}  (n_pos={test_m['n_pos']}, n_neg={test_m['n_neg']})"
        )

        # Persist test metrics alongside the checkpoint. Phase 4e folds these
        # into the checkpoint dict; until then a sidecar JSON is the source of truth.
        test_metrics_path = Path(self.CKPT).with_suffix(".test_metrics.json")
        serializable = {k: (float(v) if hasattr(v, "item") else v) for k, v in test_m.items()}
        with open(test_metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)
        s.log_line(f"Saved test metrics to {test_metrics_path}")

        s.running = False
        s.done    = True
        s.log_line(f"Training complete!  Best val AUC: {best_auc:.4f}  Test AUC: {test_m['auc']:.4f}")

    # ------------------------------------------------------------------

    def _train_ep(self, model, loader, criterion, optimizer, device) -> float:
        model.train()
        total    = 0.0
        n_batch  = len(loader)
        for batch_idx, (imgs, lbls) in enumerate(loader, 1):
            if self.state.stop_requested:
                break
            imgs = imgs.to(device)
            lbls = lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)

            if batch_idx % 50 == 0 or batch_idx == n_batch:
                print(
                    f"  batch {batch_idx:>5}/{n_batch}  "
                    f"loss={loss.item():.4f}  "
                    f"gpu={torch.cuda.memory_reserved(device) // 1e6:.0f}MB"
                    if device.type == "cuda" else
                    f"  batch {batch_idx:>5}/{n_batch}  loss={loss.item():.4f}",
                    flush=True,
                )
        return total / len(loader.dataset)

    @torch.no_grad()
    def _val_ep(self, model, loader, criterion, device, threshold: float = 0.5):
        """
        Phase 4b: threshold parameterized so the held-out test eval can use
        the calibrated operating threshold from _sweep_threshold, not 0.5.
        Phase 4f changes the per-epoch threshold to a dynamic mini-sweep;
        the default 0.5 is kept for backward compatibility with any callers
        that haven't migrated.
        """
        model.eval()
        total      = 0.0
        all_probs  = []
        all_labels = []
        for imgs, lbls in loader:
            imgs  = imgs.to(device)
            lbls_ = lbls.to(device).unsqueeze(1)
            logits = model(imgs)
            total += criterion(logits, lbls_).item() * imgs.size(0)
            all_probs.append(torch.sigmoid(logits).squeeze(1).cpu())
            all_labels.append(lbls)

        probs  = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy().astype(int)
        preds  = (probs >= threshold).astype(int)

        try:
            auc    = roc_auc_score(labels, probs)
            pr_auc = average_precision_score(labels, probs)
        except ValueError:
            auc = pr_auc = float("nan")

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1     = f1_score(labels, preds, zero_division=0)

        return total / len(loader.dataset), {
            "auc": auc, "pr_auc": pr_auc,
            "recall": recall, "specificity": spec, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    @torch.no_grad()
    def _sweep_threshold(self, model, loader, device) -> float:
        """
        Sweep decision threshold over [0.05, 0.95].
        Strategy: maximize specificity subject to recall >= 0.80.
        Falls back to Youden's J (sensitivity + specificity - 1) if no
        threshold satisfies the recall constraint.
        """
        model.eval()
        all_probs, all_labels = [], []
        for imgs, lbls in loader:
            probs = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu()
            all_probs.append(probs)
            all_labels.append(lbls)

        probs  = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy().astype(int)

        fpr, tpr, thresholds = roc_curve(labels, probs)
        # specificity = 1 - fpr
        spec_arr    = 1.0 - fpr
        recall_arr  = tpr

        # Strategy 1: max specificity where recall >= 0.80
        mask = recall_arr >= 0.80
        if mask.any():
            best_idx   = int(np.argmax(spec_arr[mask]))
            opt_thresh = float(thresholds[mask][best_idx])
            self.state.log_line(
                f"  Threshold sweep: recall≥0.80 constraint met. "
                f"recall={recall_arr[mask][best_idx]:.3f}  "
                f"spec={spec_arr[mask][best_idx]:.3f}"
            )
        else:
            # Fallback: Youden's J
            j         = recall_arr + spec_arr - 1.0
            best_idx  = int(np.argmax(j))
            opt_thresh = float(thresholds[best_idx])
            self.state.log_line(
                f"  Threshold sweep: no threshold gave recall≥0.80. "
                f"Using Youden J={j[best_idx]:.3f}  "
                f"recall={recall_arr[best_idx]:.3f}  spec={spec_arr[best_idx]:.3f}"
            )

        return max(0.05, min(0.95, opt_thresh))

    @staticmethod
    def _record(s: TrainingState, tr, vl, m):
        with s._lock:
            s.train_losses.append(tr)
            s.val_losses.append(vl)
            s.val_aucs.append(m["auc"])
            s.val_pr_aucs.append(m.get("pr_auc", float("nan")))
            s.val_recalls.append(m["recall"])
            s.val_specs.append(m["specificity"])
            s.val_f1s.append(m["f1"])
