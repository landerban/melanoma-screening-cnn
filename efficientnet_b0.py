"""
Skin Lesion Binary Classifier — EfficientNet-B0
Task   : benign vs malignant (ISIC / HAM10000)
Input  : 300 × 300 × 3
Output : 1 logit  →  sigmoid  →  probability

Two-stage transfer learning
  Stage 1 : frozen backbone, head-only  (3–5 epochs, lr = 3e-4)
  Stage 2 : full fine-tune              (up to 30 epochs, backbone lr = 1e-5)

IMPORTANT — Data splitting:
  Always use PATIENT-LEVEL split.
  Image-level random split leaks lesion identity across sets → inflated metrics.
"""

import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from PIL import Image


# =============================================================================
# CONFIG
# =============================================================================

CFG = {
    # --- data ---
    "input_size"        : 300,
    "batch_size"        : 32,

    # --- model ---
    "backbone"          : "efficientnet_b0",
    "dropout_1"         : 0.4,    # after GAP
    "hidden_dim"        : 256,
    "dropout_2"         : 0.2,    # before final linear

    # --- loss ---
    # "focal"        : RECOMMENDED for severe imbalance (161:1).  alpha must be HIGH
    #                  to upweight the malignant minority.
    #                  alpha_t = alpha  for malignant (target=1)
    #                  alpha_t = 1-alpha for benign   (target=0)
    #                  → alpha=0.85 means malignant gets 5.7× more attention than benign
    # "weighted_bce" : alternative; pos_weight capped at 80 to avoid gradient explosion
    "loss"              : "focal",
    "focal_alpha"       : 0.85,   # HIGH — upweights malignant minority (was 0.25, which was WRONG)
    "focal_gamma"       : 2.5,    # slightly higher than default to focus harder on hard malignant cases

    # --- optimiser ---
    "weight_decay"      : 1e-4,
    "bce_pos_weight_cap": 80.0,   # caps pos_weight for weighted_bce; raw ratio ~161 causes instability

    # --- two-stage LRs ---
    "head_lr"           : 3e-4,   # Stage 1  (head only, backbone frozen)
    "backbone_lr"       : 5e-6,   # Stage 2  (backbone fine-tune) — halved; 1e-5 converges too fast
    "head_lr_s2"        : 1e-4,   # Stage 2  (head)

    # --- schedule ---
    # Stage 1 uses ReduceLROnPlateau; Stage 2 uses CosineAnnealingLR (avoids LR collapse)
    "scheduler_factor"  : 0.5,    # Stage 1 ReduceLROnPlateau factor
    "scheduler_patience": 3,      # Stage 1 ReduceLROnPlateau patience

    # --- training ---
    "stage1_epochs"     : 5,      # freeze backbone, train head only
    "stage2_epochs"     : 30,     # unfreeze and fine-tune
    "es_patience"       : 7,      # increased — AUC improves more slowly than loss
    "es_delta"          : 5e-4,   # min AUC improvement to reset patience

    # --- normalization (ImageNet) ---
    "mean"              : [0.485, 0.456, 0.406],
    "std"               : [0.229, 0.224, 0.225],
}


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(cfg: dict, mode: str) -> transforms.Compose:
    """
    mode = "train" : augmentation + normalization
    mode = "val"   : resize + center-crop + normalization only

    Augmentations are medically conservative:
      - no elastic deformation / extreme colour jitter that destroys lesion texture
    """
    size = cfg["input_size"]
    mean = cfg["mean"]
    std  = cfg["std"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(int(size * 1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


# =============================================================================
# DATASET
# =============================================================================

class SkinLesionDataset(Dataset):
    """
    Generic skin-lesion dataset for binary classification.

    Args:
        image_paths : list[str]   absolute paths to image files
        labels      : list[int]   0 = benign, 1 = malignant
        transform   : torchvision transform pipeline

    Expected usage with ISIC / HAM10000:
        1. Load metadata CSV (isic-cli or HAM10000 metadata).
        2. Map dx labels → binary:  benign=0 / malignant=1.
        3. Split by *patient_id* (never by image_id) to avoid leakage.
        4. Pass the resulting file lists here.

    WARNING:
        Do NOT perform random image-level splitting.
        Multiple images of the same lesion / patient MUST stay in the same split.
        Violation inflates AUC and recall by up to 10–15 pp.
    """

    def __init__(self, image_paths: list, labels: list, transform=None):
        assert len(image_paths) == len(labels), "Paths and labels must match."
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that up-samples the minority class
    so each batch sees a balanced view — without duplicating the dataset.
    """
    labels_t  = torch.tensor(labels)
    n_pos     = labels_t.sum().item()
    n_neg     = len(labels_t) - n_pos
    w_pos     = 1.0 / n_pos if n_pos > 0 else 0.0
    w_neg     = 1.0 / n_neg if n_neg > 0 else 0.0
    weights   = torch.where(labels_t == 1,
                            torch.tensor(w_pos),
                            torch.tensor(w_neg))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# =============================================================================
# PATIENT-LEVEL SPLIT HELPER
# =============================================================================

def patient_level_split(
    df,
    patient_col: str  = "patient_id",
    label_col: str    = "label",
    val_frac: float   = 0.15,
    test_frac: float  = 0.15,
    seed: int         = 42,
):
    """
    Split a DataFrame at the PATIENT level to prevent leakage.

    Args:
        df          : pandas DataFrame with at least [patient_col, label_col, "image_path"]
        patient_col : column holding the patient / lesion identifier
        label_col   : column holding the binary label (0/1)
        val_frac    : fraction of patients for validation
        test_frac   : fraction of patients for test
        seed        : random seed

    Returns:
        train_df, val_df, test_df

    Example:
        train_df, val_df, test_df = patient_level_split(metadata_df)

    WARNING: never pass image_id as the split key — use the patient or lesion id.
    """
    import pandas as pd

    rng      = np.random.default_rng(seed)
    patients = np.array(df[patient_col].unique())
    rng.shuffle(patients)

    n       = len(patients)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)

    test_patients  = patients[:n_test]
    val_patients   = patients[n_test: n_test + n_val]
    train_patients = patients[n_test + n_val:]

    train_df = df[df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df[patient_col].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df[patient_col].isin(test_patients)].reset_index(drop=True)

    print(f"Split  →  train: {len(train_df)} imgs ({len(train_patients)} patients) | "
          f"val: {len(val_df)} imgs ({len(val_patients)} patients) | "
          f"test: {len(test_df)} imgs ({len(test_patients)} patients)")

    # Leakage sanity check
    overlap = (set(train_patients) & set(val_patients)) | \
              (set(train_patients) & set(test_patients)) | \
              (set(val_patients)   & set(test_patients))
    if overlap:
        warnings.warn(f"Patient leakage detected in {len(overlap)} patient(s)!")

    return train_df, val_df, test_df


# =============================================================================
# MODEL
# =============================================================================

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with custom binary classification head.

    Head architecture:
        GlobalAveragePooling  (built into EfficientNet)
        Dropout(0.4)
        Linear(1280 → 256)
        ReLU
        Dropout(0.2)
        Linear(256 → 1)       ← raw logit

    Args:
        freeze_backbone : True during Stage 1, False during Stage 2
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features   # 1280

        # Keep feature extractor (includes GAP via AdaptiveAvgPool2d)
        self.backbone   = base.features
        self.pool       = base.avgpool                 # AdaptiveAvgPool2d(1)

        # Binary head
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )
        self._init_head()

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit, shape (B, 1)."""
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probability, shape (B, 1)."""
        return torch.sigmoid(self.forward(x))


# =============================================================================
# LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Binary focal loss.
      FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Use when the model over-predicts the benign majority class or when
    hard-example emphasis is needed.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1, 1)
        logits  = logits.view(-1, 1)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss    = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


def build_loss(cfg: dict, pos_count: int = None, neg_count: int = None) -> nn.Module:
    """
    Build the loss function.

    Focal loss:
      alpha must be HIGH (>=0.80) to upweight malignant.
      alpha_t = alpha  for malignant (target=1)
      alpha_t = 1-alpha for benign   (target=0)

    Weighted BCE:
      pos_weight = neg/pos, capped at bce_pos_weight_cap.
      Raw 161:1 ratio causes gradient instability; cap at 80.
    """
    if cfg["loss"] == "focal":
        alpha = cfg["focal_alpha"]
        gamma = cfg["focal_gamma"]
        print(f"FocalLoss  alpha={alpha}  gamma={gamma}  "
              f"(malignant weight={alpha:.2f}, benign weight={1-alpha:.2f})")
        return FocalLoss(alpha=alpha, gamma=gamma)

    if cfg["loss"] == "weighted_bce":
        cap = cfg.get("bce_pos_weight_cap", 80.0)
        if pos_count and neg_count:
            pw = min(neg_count / pos_count, cap)
        else:
            pw = 1.0
            warnings.warn(
                "pos_count / neg_count not supplied to build_loss(); "
                "using pos_weight=1.0 (no class weighting)."
            )
        pos_weight = torch.tensor([pw])
        print(f"WeightedBCE  pos_weight = {pw:.1f}  "
              f"(raw ratio={neg_count}/{pos_count}={neg_count/pos_count:.0f}, capped at {cap})")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss '{cfg['loss']}'. Choose 'weighted_bce' or 'focal'.")


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Halts training when val loss does not improve by `delta` for `patience` epochs.
    Best model weights are saved and can be restored via `.restore(model)`.
    """

    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience     = patience
        self.delta        = delta
        self.best_loss    = float("inf")
        self.counter      = 0
        self.stop         = False
        self._best_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss    = val_loss
            self.counter      = 0
            self._best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module):
        if self._best_weights:
            model.load_state_dict(self._best_weights)


# =============================================================================
# METRICS
# =============================================================================

@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device,
                    threshold: float = 0.5) -> dict:
    """
    Evaluate on a DataLoader.

    Returns dict with: loss, auc, recall (sensitivity), specificity, f1,
                       confusion_matrix
    """
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        probs  = model.predict_proba(images).squeeze(1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    preds  = (probs >= threshold).astype(int)

    cm            = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1           = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {
        "auc"             : auc,
        "recall"          : sensitivity,
        "specificity"     : specificity,
        "f1"              : f1,
        "confusion_matrix": cm,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_metrics(metrics: dict, prefix: str = ""):
    cm = metrics["confusion_matrix"]
    print(f"{prefix}AUC={metrics['auc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  "
          f"Specificity={metrics['specificity']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print(f"{prefix}Confusion matrix:\n{cm}")
    print(f"{prefix}  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")
    if metrics["fn"] > 0:
        print(f"{prefix}  ⚠  {metrics['fn']} false negatives "
              f"(missed malignant) — consider lowering threshold.")


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# =============================================================================
# TWO-STAGE TRAINING
# =============================================================================

def stage1_train(
    model      : EfficientNetB0Classifier,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    cfg         : dict,
    device      : torch.device,
    criterion   : nn.Module,
):
    """
    Stage 1 — backbone frozen, train head only.
    Uses head_lr = 3e-4.
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — Backbone frozen, training head only")
    print("=" * 60)

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["head_lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
    )

    _header()
    for epoch in range(1, cfg["stage1_epochs"] + 1):
        tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = _run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:>3}  {tr_loss:>10.4f}  {va_loss:>10.4f}  lr={lr_now:.2e}")


def stage2_train(
    model      : EfficientNetB0Classifier,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    cfg         : dict,
    device      : torch.device,
    criterion   : nn.Module,
) -> EfficientNetB0Classifier:
    """
    Stage 2 — full model fine-tune with differential learning rates.
    Backbone lr = 1e-5 (low to preserve pretrained features).
    Head lr    = 1e-4.
    Early stopping with patience = 5.
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — Full fine-tune (backbone lr=1e-5, head lr=1e-4)")
    print("=" * 60)

    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg["backbone_lr"]},
            {"params": model.pool.parameters(),     "lr": cfg["backbone_lr"]},
            {"params": model.head.parameters(),     "lr": cfg["head_lr_s2"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
    )
    es = EarlyStopping(patience=cfg["es_patience"], delta=cfg["es_delta"])

    _header()
    for epoch in range(1, cfg["stage2_epochs"] + 1):
        tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = _run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step(va_loss)

        es(va_loss, model)
        marker = " ✓" if es.counter == 0 else f" ({es.counter}/{es.patience})"
        lr_bb  = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:>3}  {tr_loss:>10.4f}  {va_loss:>10.4f}  bb_lr={lr_bb:.2e}{marker}")

        if es.stop:
            print(f"\n  Early stopping at epoch {epoch}. Best val loss: {es.best_loss:.4f}")
            break

    es.restore(model)
    return model


def _header():
    print(f"  {'Ep':>3}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("  " + "-" * 32)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def train_pipeline(
    train_loader : DataLoader,
    val_loader   : DataLoader,
    pos_count    : int = None,
    neg_count    : int = None,
    cfg          : dict = CFG,
    device       : torch.device = None,
) -> EfficientNetB0Classifier:
    """
    Full two-stage training pipeline.

    Args:
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        pos_count    : number of malignant samples in the training set
        neg_count    : number of benign samples in the training set
        cfg          : config dictionary
        device       : torch device (auto-detected if None)

    Returns:
        Trained model with best validation weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model     = EfficientNetB0Classifier(freeze_backbone=True).to(device)
    criterion = build_loss(cfg, pos_count, neg_count).to(device)

    # Stage 1
    stage1_train(model, train_loader, val_loader, cfg, device, criterion)

    # Stage 2
    model = stage2_train(model, train_loader, val_loader, cfg, device, criterion)

    return model


# =============================================================================
# SANITY CHECK  (no real data needed)
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.eval()

    dummy = torch.randn(4, 3, CFG["input_size"], CFG["input_size"]).to(device)
    with torch.no_grad():
        logit = model(dummy)
        prob  = model.predict_proba(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape  : {tuple(dummy.shape)}")
    print(f"Logit shape  : {tuple(logit.shape)}")
    print(f"Prob  shape  : {tuple(prob.shape)}")
    print(f"Total params : {total:,}")
    print(f"Trainable    : {trainable:,}")

    # Verify head-freeze works
    model.freeze_backbone()
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    head_only = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStage 1 mode:")
    print(f"  Frozen (backbone)  : {frozen:,}")
    print(f"  Trainable (head)   : {head_only:,}")
