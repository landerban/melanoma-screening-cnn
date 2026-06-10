# Presentation Script — Speaker 1: Alexander Park

Skin Lesion Binary Classifier — Team 3 Final
Slides 1-5 (Setup + the crack)  ·  Target ~3.5 min  ·  ~445 words

---

## [Slide 1 — Title]

Good morning. We are Team 3. Our project is a skin lesion screening model — but more than that, it is an honest audit of what our model actually learned. We start from a strong result, a 0.95 AUC, and then we go looking for the shortcut hiding underneath it. I will set up the problem and the model, and then my teammates will walk you through the investigation.

## [Slide 2 — Problem & data]

First, the problem. Screening is a binary question: refer the patient, or don't. It is recall-first, because a missed melanoma costs far more than an unnecessary referral. At our four-to-one imbalance, a model can score high just by calling everything benign — so the cost asymmetry, not accuracy, drives our metric, our loss, and our threshold. The data is three dermoscopy collections from ISIC, about sixty-one thousand images. We hold out a smartphone cohort as an out-of-distribution test. And the split is patient-level, so no patient leaks between training and test.

## [Slide 3 — Model & training]

The model is EfficientNet-B0 with a small head, trained in two stages: first the head with a frozen backbone, then full fine-tuning. One detail matters here — we inverted the default focal-loss alpha. The standard value up-weights the majority class, which is the wrong direction for us. We set it so the rare malignant class carries the weight.

## [Slide 4 — Headline result]

And the headline is strong: a test AUC of 0.9513, on a patient-level split, with a validation-to-test gap of basically zero. Recall is 0.835, specificity 0.914. Out of 1,767 malignant cases, the model misses 292 — and that false-negative cell is exactly what we minimize against. So far, a clean, strong result.

## [Slide 5 — A crack in the result]

But here is where it gets interesting. When we break that 0.95 down per collection, every collection scores lower than the aggregate — the honest range is only 0.87 to 0.92. And on the out-of-distribution smartphone images, recall collapses to 0.51 — half the cancers missed. That raised a suspicion: maybe the model is keying on which collection an image comes from, not on the lesion itself. To find out, I'll hand over to Seongmin, who led the investigation.

---

Delivery notes
- Pace is relaxed; this is the setup. Land slide 4 ("strong result") with confidence, then shift tone on slide 5 ("but here is where it gets interesting") to create the turn.
- End cleanly on the handoff line to Seongmin.
