# Skin Lesion Binary Classifier

Binary classification of skin lesions — **benign vs malignant** — using EfficientNet-B0 pretrained on ImageNet, trained on ISIC / HAM10000 dermoscopy data.

---

## Task

| Property | Value |
|---|---|
| Task | Binary classification |
| Classes | Benign (0) · Malignant (1) |
| Clinical goal | Screening support — maximize malignant recall while keeping specificity usable |
| Framework | PyTorch 2.6.0 + CUDA 12.4 |

---

## Dataset

| Source | Collection ID | Type |
|---|---|---|
| ISIC 2024 Challenge | 390 | TBP tile close-up |
| SIIM-ISIC Melanoma 2020 | 70 | Dermoscopy |
| HAM10000 | 212 | Dermoscopy |

Downloaded via `isic-cli`, deduplicated on `isic_id`, split at **patient level** to prevent lesion leakage.

| Split | Images | Patients |
|---|---|---|
| Train | 304,112 | 731 |
| Validation | 53,675 | 156 |
| Test (held out) | 54,729 | 156 |
| **Total** | **412,516** | **1,042** |

**Class distribution (full dataset)**

| Class | Count | % |
|---|---|---|
| Benign | 409,967 | 99.4% |
| Malignant | 2,549 | 0.6% |
| Imbalance ratio | ~161 : 1 | — |

> **Note:** `Indeterminate` labels (114 samples) are excluded. Patient-level splitting is mandatory — image-level random splits cause lesion memorization and can inflate AUC by 10–15 pp.

---

## Model Architecture

```
Input  (3 × 300 × 300)
       ↓
EfficientNet-B0 backbone  [ImageNet pretrained — IMAGENET1K_V1]
       ↓
GlobalAveragePooling  (AdaptiveAvgPool2d → 1×1)
       ↓
Dropout(0.4)
Linear(1280 → 256)
ReLU
Dropout(0.2)
Linear(256 → 1)
       ↓
Raw logit  →  sigmoid  →  P(malignant)
```

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B0 (~5.3M params) |
| Head | 2-layer MLP — 1280 → 256 → 1 |
| Input size | 300 × 300 × 3 |
| Output | 1 logit (binary) |

---

## Training Configuration

### Loss — Focal Loss

```
FL(p_t) = -alpha_t × (1 - p_t)^gamma × log(p_t)

alpha = 0.85   →  malignant weight = 0.85, benign weight = 0.15
gamma = 2.5    →  hard-example focusing
```

Focal loss chosen over weighted BCE because the 161:1 imbalance is too extreme for stable BCE weighting.
`alpha=0.85` is critical — the common default of `alpha=0.25` is inverted for this task, giving 3× more weight to benign than malignant.

### Optimizer — AdamW

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Head LR | 3e-4 | 1e-4 |
| Backbone LR | frozen | 5e-6 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 32 | 32 |

### Augmentations (train only)

- Random resized crop (scale 0.8–1.0)
- Horizontal flip
- Vertical flip
- Rotation ±15°
- Color jitter (brightness ±0.2, contrast ±0.2)
- ImageNet normalization — mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`

### Training Strategy — Two-Stage Transfer Learning

**Stage 1** — backbone frozen, head only (5 epochs, lr=3e-4)
Warms up the randomly-initialized MLP head without disturbing pretrained features.

**Stage 2** — full model fine-tune (up to 30 epochs)
- Backbone lr = 5e-6 (low to preserve pretrained features)
- Scheduler: `CosineAnnealingLR(T_max=30, eta_min=1e-7)`
- Early stopping: patience=7, monitors **AUC** — not val loss, which is dominated by the benign majority
- Balanced sampling: `WeightedRandomSampler` applied to training set

---

## Training Results

### Stage 1 — Head Only

| Epoch | Train Loss | Val Loss | AUC | Recall |
|---|---|---|---|---|
| 1/5 | 0.0132 | 0.0098 | 0.879 | 0.761 |
| 2/5 | 0.0113 | 0.0102 | 0.889 | 0.761 |
| 3/5 | 0.0106 | 0.0104 | 0.876 | 0.746 |
| 4/5 | 0.0101 | 0.0102 | 0.889 | 0.746 |
| 5/5 | 0.0099 | 0.0081 | 0.882 | 0.732 |

### Stage 2 — Full Fine-tune

| Epoch | Train Loss | Val Loss | AUC | PR-AUC | Recall | Spec | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0077 | 0.0071 | 0.9216 | 0.0296 | 0.775 | 0.886 | 0.018 | 47,495 | 6,109 | 16 | 55 | ✓ |
| 2 | 0.0058 | 0.0052 | 0.9203 | 0.0296 | 0.704 | 0.921 | 0.023 | 49,381 | 4,223 | 21 | 50 | |
| **3** | **0.0047** | **0.0042** | **0.9283** | **0.0265** | **0.662** | **0.943** | **0.030** | **50,555** | **3,049** | **24** | **47** | **✓ best** |
| 4 | 0.0039 | 0.0036 | 0.9241 | 0.0413 | 0.577 | 0.953 | 0.031 | 51,096 | 2,508 | 30 | 41 | |
| 5 | 0.0034 | 0.0032 | 0.9256 | 0.0376 | 0.507 | 0.967 | 0.038 | 51,815 | 1,789 | 35 | 36 | |
| 6 | 0.0030 | 0.0035 | 0.9262 | 0.0437 | 0.507 | 0.961 | 0.033 | 51,538 | 2,066 | 35 | 36 | |
| 7 | 0.0027 | 0.0030 | 0.9220 | 0.0298 | 0.451 | 0.973 | 0.041 | 52,159 | 1,445 | 39 | 32 | |
| 8 | 0.0024 | 0.0024 | 0.9218 | 0.0547 | 0.380 | 0.981 | 0.049 | 52,589 | 1,015 | 44 | 27 | |
| 9 | 0.0023 | 0.0029 | 0.9220 | 0.0413 | 0.394 | 0.978 | 0.044 | 52,440 | 1,164 | 43 | 28 | |
| 10 | 0.0021 | 0.0034 | 0.9158 | 0.0193 | 0.324 | 0.984 | 0.047 | 52,725 | 879 | 48 | 23 | |

Early stopping triggered at epoch 10 — AUC did not improve by ≥5e-4 for 7 consecutive epochs.

### Best Checkpoint (epoch 3)

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9283** |
| Val Loss | 0.0042 |
| Recall @ 0.5 threshold | 0.662 |
| Specificity @ 0.5 threshold | 0.943 |

### Threshold Calibration

After training, the validation set was swept to find the operating point satisfying **recall ≥ 0.80 at maximum specificity** (clinical screening standard):

| Threshold | Recall | Specificity | Strategy |
|---|---|---|---|
| 0.5 (default) | 0.662 | 0.943 | — |
| **0.347 (calibrated)** | **0.803** | **0.876** | recall ≥ 0.80, max specificity |

> **Recommended operating threshold: 0.347**
>
> At this threshold the model catches **80.3% of malignant lesions** while correctly dismissing **87.6% of benign lesions**.
> The 12.4% false positive rate is acceptable for a screening tool — a missed malignancy (false negative) carries far greater clinical cost than an unnecessary referral (false positive).

### Performance Notes

**Why F1 is low:** F1 at threshold=0.5 is not meaningful for 161:1 imbalance. At the calibrated threshold, recall is 80%. Use **AUC and recall@specificity** as primary metrics, not F1 or accuracy.

**Why PR-AUC is low:** A random classifier achieves PR-AUC ≈ prevalence ≈ 0.006 on this dataset. Our model achieves 0.03–0.05, a **5–8× lift** over random. State-of-the-art on balanced ISIC data reaches 0.15–0.25.

**Grad-CAM check:** If attention maps focus on rulers, image borders, or corner artifacts rather than the lesion itself, this indicates shortcut learning. Re-inspect augmentation and consider lesion-centered cropping.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| VRAM | 8.6 GB |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11.9 |
| OS | Windows 11 |

---

## Project Structure

```
AI_project/
├── efficientnet_b0.py        # Model, CFG, loss, transforms, dataset, patient split
├── trainer.py                # Background trainer thread, metrics, threshold sweep
├── app.py                    # Gradio UI — training monitor + inference + Grad-CAM
├── best_model.pth            # Saved checkpoint (best AUC — epoch 3)
├── README.md
└── training_data/
    ├── challenge-2024-training_metadata_2026-03-06.csv  (ISIC 2024, 401K rows)
    └── images/
        ├── metadata.csv      (HAM10000 metadata — downloaded by isic-cli)
        └── ISIC_xxxxxxx.jpg  (×445,906 images total)
```

---

## Quick Start

**1. Install dependencies**
```powershell
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
venv\Scripts\python.exe -m pip install gradio matplotlib pandas scikit-learn isic-cli
```

**2. Download datasets**
```powershell
isic image download -c 390 training_data\images   # ISIC 2024 Challenge
isic image download -c 70  training_data\images   # SIIM-ISIC Melanoma 2020
isic image download -c 212 training_data\images   # HAM10000
```
All three collections download into the same folder — duplicate `isic_id` filenames are naturally overwritten.

**3. Launch UI**
```powershell
venv\Scripts\python.exe app.py
# opens http://127.0.0.1:7860
```

**Training tab** — configure loss/epochs, Start Training, watch live loss + metric curves refresh every 3 s.

**Test Image tab** — upload a lesion image, click **"Use optimal threshold from training"** to load the calibrated threshold, then **Run Prediction** for probability + Grad-CAM heatmap.

---

## Clinical Disclaimer

This system is a **research and decision-support tool only**.

- It has **not** undergone regulatory review (FDA, CE, or equivalent).
- It must **not** replace dermatologist diagnosis.
- False negatives (missed malignancies) remain possible at any threshold.
- Performance may degrade on images from devices, skin tones, lighting conditions, or populations not represented in the ISIC training data.
- Always inspect Grad-CAM heatmaps to verify the model is attending to the lesion, not imaging artifacts.
