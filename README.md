# Skin Lesion Binary Classifier

Binary classification of skin lesions — **benign vs malignant** — using EfficientNet-B0 pretrained on ImageNet, trained on three ISIC dermoscopy collections (HAM10000, SIIM-ISIC 2020, BCN20000). The project includes a full audit of the original training pipeline, a corrected retrain on a cleaned data pool, and held-out evaluation against an out-of-distribution smartphone-clinical cohort (PAD-UFES-20).

---

## Task

| Property | Value |
|---|---|
| Task | Binary classification |
| Classes | Benign (0) · Malignant (1) |
| Clinical goal | Screening support — maximize malignant recall while keeping specificity usable |
| Primary metric | ROC-AUC; recall at calibrated specificity as the operating-point measure |
| Framework | PyTorch 2.6.0 + CUDA 12.4 |

---

## Headline numbers (held-out test cohort)

```
ROC-AUC = 0.9513   [95% CI 0.9465 – 0.9563]   (1000 bootstrap resamples)
PR-AUC  = 0.8353

At calibrated threshold 0.661 (selected on held-out cal cohort):
  recall      = 0.835  (1475/1767 cancers caught)
  specificity = 0.914
  F1          = 0.761
  CM          : TN=6784 FP=635 FN=292 TP=1475
```

Test cohort: 9,186 dermoscopy images / 1,767 malignant, patient-level disjoint from train / val / cal. The val/test AUC gap is 0.0001 — clean patient-level holdout.

**Honest disclosure.** The aggregate AUC is prevalence-weighted by the cohort composition. **Per-collection within-modality AUCs cluster at 0.87 – 0.92.** The aggregate exceeds per-collection numbers because the model has partly learned cross-collection prevalence priors. See `docs/midterm-prep-context.md` §9 and `docs/diagnostic_findings.md` for the full breakdown.

**Out-of-distribution result.** On PAD-UFES-20 (smartphone clinical close-ups, 1,568 images), AUC = **0.8055**. Partial transfer — meaningfully above chance, but the calibrated threshold is severely miscalibrated for this distribution (recall drops to 0.51).

---

## Dataset

Three ISIC dermoscopy collections, downloaded via `isic-cli`.

| Source | Collection ID | Type | Rows | Native `patient_id` |
|---|---|---|---|---|
| HAM10000 | 212 | Dermoscopy | 11,720 | absent (lesion_id only) |
| SIIM-ISIC Melanoma 2020 | 70 | Dermoscopy | 33,126 | present |
| BCN20000 | 249 | Dermoscopy | 18,946 | absent (lesion_id only) |

After deduplication on `isic_id` and dropping `Indeterminate` and NaN diagnoses (2,396 rows): **61,396 images, 49,785 benign / 11,611 malignant (4.29:1 imbalance)**.

PAD-UFES-20 (ISIC collection c=406, smartphone clinical) is held out for OOD evaluation only — not part of the training pool.

### Patient-level split via `effective_patient_id`

Because c=212 and c=249 ship without a native `patient_id` column, the codebase derives `effective_patient_id = patient_id || lesion_id || isic_id` per row, built in `load_and_merge_metadata` before deduplication. Without this fallback chain, 28,273 of 61,396 rows (46%) would collapse into one synthetic NaN-patient bucket and land entirely in one split.

Split fractions: train 0.65 / val 0.10 / cal 0.10 / test 0.15, all at the patient level.

| Split | Images | Patients |
|---|---|---|
| Train | 39,647 | 10,115 |
| Val | 6,634 | 1,555 |
| Cal | 5,929 | 1,555 |
| Test | 9,186 | 2,333 |

The **cal cohort** is the patient-disjoint sibling of val. It owns the threshold sweep — calibrating on val would have biased the operating point toward overfitting (val drove early stopping + best-epoch selection).

Patient-level splitting is mandatory — image-level random splits cause lesion memorization and can inflate AUC by 10–15 pp.

---

## Model architecture

```
Input  (3 × 384 × 384)
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
| Input size | 384 × 384 × 3 |
| Output | 1 logit (binary) |

EfficientNet-B0 was chosen under the original 8 GB VRAM constraint of an RTX 3060 Ti. The retrain runs on an A100X MIG 3g.40gb (40 GB) — the constraint is lifted, but B0 is retained to keep the methodology comparison clean ("is the AUC delta from the architecture or the data?"). B3 is a clean follow-up experiment with one variable changed.

---

## Training configuration

### Loss — Focal Loss

```
FL(p_t) = -alpha_t × (1 - p_t)^gamma × log(p_t)

alpha = 0.85   →  malignant weight = 0.85, benign weight = 0.15
gamma = 2.5    →  hard-example focusing
```

In this binary focal-loss implementation (`efficientnet_b0.py`), `alpha_t = alpha * targets + (1 - alpha) * (1 - targets)`. So **α directly weights the positive class.** The detection-code default `α=0.25` means *positives* get weight 0.25 — the wrong direction for a rare-positive problem. `α=0.85` gives malignant ~5.7× the weight of benign on the new 4.29:1 pool.

The methodology lesson is prevalence-invariant: copying defaults across imbalance regimes is the category error, regardless of the specific α value.

### Optimizer — AdamW

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Head LR | 3e-4 | 1e-4 |
| Backbone LR | frozen | 5e-6 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 96 | 96 |
| Input size | 384 × 384 | 384 × 384 |

### Augmentations (train only)

- Random resized crop (scale 0.8–1.0)
- Horizontal flip
- Vertical flip
- Rotation ±15°
- Color jitter on **brightness and contrast only** (±0.2 each)
- ImageNet normalization

Hue and saturation jitter are deliberately excluded — color is diagnostic for melanoma, and randomizing it would destroy signal the model needs.

### Two-stage transfer

**Stage 1** (5 epochs, head only, lr=3e-4, `ReduceLROnPlateau`). Warms up the randomly-initialized MLP head without disturbing pretrained features.

**Stage 2** (up to 30 epochs, full fine-tune).
- Backbone lr = 5e-6 (low to preserve pretrained features)
- Head lr = 1e-4
- Scheduler: `CosineAnnealingLR(T_max=30, eta_min=1e-7)`
- Early stopping: patience=7, **monitors AUC** (not val loss — which is dominated by the benign majority at this imbalance)
- Sampler: `WeightedRandomSampler` rebalances batches to roughly 50/50

### Reproducibility

`seed=42` is propagated through `random`, `numpy`, `torch`, `torch.cuda`, cuDNN deterministic mode, DataLoader workers (via `worker_init_fn`), and the `WeightedRandomSampler`'s generator. Two runs with the same seed produce bit-exact identical logits (verified by `scripts/phase5_reproducibility.py`).

---

## Training trajectory

Phase 6 retrain, A100X MIG 3g.40gb. Stage 1 / Stage 2 epoch-end summary from `experiments/run_20260515_172452.log`.

**Stage 1 (head only):**

| Epoch | Train loss | Val loss | Val AUC | Recall |
|---|---|---|---|---|
| 1/5 | 0.0256 | 0.0219 | 0.932 | 0.802 |
| 2/5 | 0.0221 | 0.0213 | 0.936 | 0.800 |
| 3/5 | 0.0214 | 0.0216 | 0.938 | 0.800 |
| 4/5 | 0.0211 | 0.0207 | 0.939 | 0.801 |
| 5/5 | 0.0207 | 0.0195 | 0.943 | 0.800 |

**Stage 2 (best AUC at epoch 13, ES at epoch 20):**

| Epoch | Train loss | Val loss | Val AUC | PR-AUC | Recall | Spec | F1 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0197 | 0.0203 | 0.9440 | 0.7861 | 0.802 | 0.913 | 0.729 |
| 6 | 0.0171 | 0.0189 | 0.9484 | 0.8057 | 0.800 | 0.919 | 0.736 |
| 9 | 0.0160 | 0.0184 | 0.9499 | 0.8119 | 0.802 | 0.925 | 0.747 |
| **13** | **0.0152** | **0.0170** | **0.9514** | **0.8212** | **0.801** | **0.926** | **0.749** |
| 17 | 0.0149 | 0.0169 | 0.9518 | 0.8218 | 0.800 | 0.928 | 0.751 |
| 20 | 0.0143 | 0.0167 | 0.9511 | 0.8222 | 0.800 | 0.926 | 0.747 |

Total wall time: 4 h 36 min.

---

## Threshold calibration

After training, the **held-out cal cohort** (1,555 patients, never touched during ES or best-epoch selection) is swept for the operating threshold satisfying **recall ≥ 0.80 at maximum specificity**.

| Threshold | Recall | Specificity | Source |
|---|---|---|---|
| 0.5 (default) | — | — | not the calibrated operating point |
| **0.661 (calibrated, this run)** | **0.835** on test | **0.914** on test | recall ≥ 0.80 / max spec, swept on cal |

The threshold encodes which tool this is. Same recall-at-max-specificity strategy as the original codebase; the value differs because the underlying score distribution differs (4.29:1 imbalance, stronger model → malignant probabilities cluster higher → operating point moves up).

**Per-collection miscalibration disclosure.** Threshold 0.661 is calibrated for the aggregate cohort prevalence (~20%). On individual collections at the same threshold: recall 0.23 on c=70 (1.7% prevalence), 0.89 on c=249 (54.3% prevalence). For deployment, the threshold must be recalibrated per institution's expected prevalence.

---

## Per-collection AUC breakdown

| Cohort | N (test) | Prevalence | AUC | 95% CI |
|---|---|---|---|---|
| **Aggregate test** | 9,186 | 19.2% | **0.9513** | [0.9465, 0.9563] |
| c=212 HAM10000 | 1,708 | 17.9% | 0.9151 | [0.8980, 0.9323] |
| c=70 SIIM-ISIC 2020 | 4,944 | 1.7% | 0.8835 | [0.8545, 0.9100] |
| c=249 BCN20000 | 2,534 | 54.3% | 0.8702 | [0.8555, 0.8842] |

**The aggregate exceeds every per-collection AUC** because predictions on c=249 patients cluster higher than predictions on c=70 patients (c=249 was 54% malignant in training; c=70 is 1.7%), and that between-collection score ordering is itself discriminative. The model has partly learned a modality-prior shortcut.

Statistical significance: c=212 CI [0.898, 0.932] does **not overlap** c=249 CI [0.856, 0.884] — c=212 is genuinely a better cohort for this model than c=249, with significance. c=70 and c=249 CIs overlap, so they are statistically indistinguishable.

---

## Audit summary

The project went through an 8-phase audit and rewrite. The audit's seven highest-priority findings were closed with provenance (commits + artifact logs); four open items are disclosure-grade and don't block the headline numbers.

**Closed:**

| Finding | Description | Fix |
|---|---|---|
| W1 | Threshold swept on the same val set used for ES | 4-way split; cal cohort owns the sweep |
| W2 | Test split discarded by the trainer | Test eval at calibrated threshold; metrics persisted |
| W4 | No global seeds; results not reproducible | Seed everywhere; bit-exact verified |
| W8 | Per-collection AUC not computed | Measured; see breakdown above |
| H5 | Checkpoint format opaque | Rich dict with cfg, val/test metrics, git_hash, timestamp |
| M1 | Per-epoch metrics at threshold 0.5 misleading | Dynamic mini-sweep per epoch |
| M4 | `app.py` reloads model on every inference | mtime-aware cache (~0 ms hit vs ~140 ms miss) |

**Open as slide-time disclosures:**

| Finding | Description |
|---|---|
| W3 | WeightedRandomSampler + focal α both active; not ablated which contributed |
| W5 | `CosineAnnealingLR T_max=30` while ES fired at epoch 20 (~67% of one descent) |
| W6 | Augmentation ablation not done; hue/saturation deliberately excluded |
| W7 | Systematic Grad-CAM artifact-attention check not done |

Full audit doc: `docs/midterm-prep-context.md`. Diagnostic outputs: `docs/diagnostic_findings.md`.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA A100X (MIG 3g.40gb slice) |
| VRAM | 40 GB (slice); ~25 GB peak used during stage 2 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10.13 |
| OS | Ubuntu (Elice Cloud container) |

The original codebase was developed against an RTX 3060 Ti (8 GB VRAM) on Windows 11. The retrain runs on A100X MIG for compute headroom; the model and methodology are unchanged.

---

## Project structure

```
AI_project/
├── efficientnet_b0.py           # Model, CFG, loss, transforms, dataset, patient_level_split
├── trainer.py                   # Background trainer thread, metrics, threshold sweep, ckpt format
├── app.py                       # Gradio UI — training monitor + inference + Grad-CAM
├── best_model.pth               # Trained checkpoint (rich dict: state_dict + cfg + metrics + git_hash)
├── best_model.test_metrics.json # Sidecar with the test eval metrics in human-readable form
├── README.md                    # This file
├── CLAUDE.md                    # Codebase conventions; hard constraints
├── docs/
│   ├── midterm-prep-context.md  # Canonical audit doc (8 sections + 9/10 added in Phase 8)
│   ├── diagnostic_findings.md   # Phase 3 diagnostic outputs (patient leak, lesion grouping, bootstrap)
│   ├── realtime-design.md       # Realtime framing-aid app design (Eigen-CAM live + Grad-CAM on capture)
│   ├── phase6_bracket.md        # Pre-run expected-result bracket (AUC bands, slide-7 framing)
│   └── midterm-slide-outline.md # 13-slide outline + 9 backup slides for the midterm talk
├── scripts/
│   ├── audit_patient_id.py            # Patient-leak audit (failure modes A and B)
│   ├── audit_lesion_id.py             # Per-collection lesion grouping + malignant-cluster ratio
│   ├── audit_merge_consistency.py     # diagnosis_1 vocabulary, image_type, cross-CSV dedup
│   ├── bootstrap_auc_ci.py            # Pre-retrain bootstrap on the old checkpoint (frozen artifact)
│   ├── bootstrap_test_auc.py          # Post-retrain bootstrap (aggregate + per-collection CIs)
│   ├── rebuild_per_collection_metadata.py  # Phase 2b: rebuilds per-collection CSVs via isic-cli
│   ├── eval_test_set.py               # Save/load sanity check on the rich ckpt
│   ├── eval_external.py               # PAD-UFES-20 OOD eval
│   ├── eval_per_collection.py         # Per-collection AUC breakdown
│   ├── eval_oldckpt_no_c249.py        # Ghost eval of old ckpt on c=70+c=212-only val
│   └── phase5_*.py                    # Pre-retrain validation scripts (coverage, memory, repro)
├── artifacts/                   # Diagnostic + eval output logs (one per script run)
├── experiments/                 # Per-training-run timestamped logs
└── training_data/
    ├── metadata_c212.csv        # HAM10000 metadata (rebuilt per-collection to avoid race-on-merge)
    ├── metadata_c70.csv
    ├── metadata_c249.csv
    ├── images/                  # ~63K dermoscopy JPGs across the three collections
    │   └── ISIC_xxxxxxx.jpg
    └── eval/
        └── pad_ufes_20/         # Held-out OOD eval cohort
            ├── metadata.csv
            └── ISIC_xxxxxxx.jpg
```

---

## Quick start

**1. Install dependencies**

```powershell
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
venv\Scripts\python.exe -m pip install gradio matplotlib pandas scikit-learn isic-cli pillow tqdm
```

**2. Download datasets**

```powershell
isic image download -c 212 training_data\images        # HAM10000
isic image download -c 70  training_data\images        # SIIM-ISIC 2020
isic image download -c 249 training_data\images        # BCN20000
isic image download -c 406 training_data\eval\pad_ufes_20   # PAD-UFES-20 (eval only)
```

All three training collections download into the same folder. Build per-collection metadata CSVs (avoids the race on `metadata.csv` when multiple `isic image download` runs write the same target):

```powershell
venv\Scripts\python.exe scripts\rebuild_per_collection_metadata.py
```

**3. Launch UI**

```powershell
venv\Scripts\python.exe app.py
# opens http://127.0.0.1:7860
```

The **Training tab** configures loss/epochs, starts training, and displays live loss + metric curves refreshing every 3 s.

The **Test Image tab** uploads a lesion image. Click *Use optimal threshold from training* to load the calibrated threshold from `state.optimal_threshold`, then *Run Prediction* for `P(malignant)` + a Grad-CAM heatmap.

---

## Reproducing the numbers in this README

Every number cited above traces to either an artifact log or the rich checkpoint:

| Number | Source |
|---|---|
| Test AUC 0.9513 [CI 0.9465, 0.9563] | `artifacts/bootstrap_test_auc.log` |
| Test recall/spec/CM at 0.661 | `artifacts/eval_test_indist.log` and the checkpoint's `test_metrics` field |
| Per-collection AUCs + CIs | `artifacts/bootstrap_test_auc.log` |
| PAD-UFES-20 OOD result | `artifacts/eval_padufes.log` |
| Training trajectory | `experiments/run_20260515_172452.log` |
| Audit findings | `docs/midterm-prep-context.md`, `docs/diagnostic_findings.md` |

To re-run any of the post-training evaluations against the saved checkpoint:

```bash
python scripts/eval_test_set.py best_model.pth          # in-distribution test, save/load sanity
python scripts/eval_per_collection.py best_model.pth    # per-collection AUC breakdown
python scripts/eval_external.py best_model.pth          # PAD-UFES-20 OOD
python scripts/bootstrap_test_auc.py best_model.pth     # 95% CIs (aggregate + per-collection)
```

---

## Follow-up experiments (post-midterm)

Two follow-ups motivated by the binary model's limitations, both designed in `docs/`:

- **Realtime framing-aid app** (`docs/realtime-design.md`): live Eigen-CAM attention overlay on phone-camera input as a framing aid (not live diagnosis). Capture-phase classification still goes through the standard pipeline. Grounded in the PAD-UFES-20 result — live attention has a weaker requirement (does the model look at the lesion?) than live classification.
- **Hierarchical multiclass** (slide 12 of `docs/midterm-slide-outline.md`): the frozen binary classifier gates a subtype head trained on the malignant pool only, with PAD-UFES-20 as the held-out subtype cohort. Hierarchical over end-to-end multiclass because the binary AUC 0.9513 is the project's strongest single piece of evidence; end-to-end risks degrading binary recall in exchange for marginal subtype accuracy.

---

## Clinical disclaimer

This system is a **research and decision-support tool only**.

- It has **not** undergone regulatory review (FDA, CE, or equivalent).
- It must **not** replace dermatologist diagnosis.
- False negatives (missed malignancies) remain possible at any threshold.
- Performance is bounded by the training distribution. The PAD-UFES-20 result above is the measured out-of-distribution gap on smartphone clinical images; performance may degrade further on devices, skin tones, lighting conditions, or populations not represented in the ISIC training data.
- The per-collection breakdown shows the model uses cross-collection prevalence priors in addition to lesion features. For deployment, the operating threshold must be recalibrated against the deployment population's expected prevalence.
- Always inspect Grad-CAM heatmaps to verify the model is attending to the lesion, not imaging artifacts (rulers, borders, ink markings).
