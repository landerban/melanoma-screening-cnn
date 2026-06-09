# Phase 9 — Slide Integration

How Phase 9 modifies the main midterm slide deck
(`docs/midterm-slide-outline.md`). Each section below is keyed to a
specific main-deck slide.

The deliberate position is *augmentation, not replacement*. The Phase
7c slide-9 message remains valid; Phase 9 sharpens it and adds a
follow-up slide that quantifies the prevalence-prior contribution.

---

## Slide 9 — REWRITTEN

**Current title.** "Per-collection AUC: the aggregate hides a modality-
prior contribution."

**New title.** "The 0.95 aggregate hides a 70%-prevalence-prior contribution on SIIM-2020."

**New table (replaces current per-collection table).**

| Cohort | N | Prev | Phase 7c AUC | **Phase 9 balanced AUC** | Within-collection signal |
|---|---|---|---|---|---|
| Aggregate test | 9,186 | 19.2% | **0.9513** | — | (prevalence-weighted mix) |
| c=212 HAM10000 | 1,708 | 17.9% | 0.9151 | **0.943** [0.895, 0.979] | strong (gains under balance) |
| c=70 SIIM 2020 | 4,944 | 1.7% | 0.8835 | **0.548** [0.432, 0.663] | **~chance** |
| c=249 BCN20000 | 2,534 | 54.3% | 0.8702 | 0.778 [0.680, 0.862] | partial |

**New narrative bullets.**

- The Phase 7c per-collection AUCs (0.87 – 0.92) clustered close
  together because *all three* received a prevalence-prior boost from
  the heterogeneous cohort prevalences.
- Phase 9 holds prevalence constant (50/50 per collection) and
  re-evaluates. c=212's AUC *rises* (0.94) — its lesion-feature signal
  survives the loss of cross-collection cues. **c=70's AUC collapses
  to 0.548 — chance.** The Phase 7c 0.88 was almost entirely a
  prevalence-prior contribution.
- This is the strongest "the headline number understates its own
  caveat" finding in the audit pipeline. Disclose it on the slide
  alongside the 0.9513 number.

**Forward reference (kept).** The Phase 9 work decomposes which
*specific* shortcuts drive this prevalence-prior contribution.

**One-line Q&A defense.** "How do we know c=70's 0.88 isn't real?"
→ "When we equalize prevalence the AUC collapses to 0.55. The 0.88
required the 1.7% prevalence specifically — it doesn't transfer to a
50/50 cohort, which is what a balanced screening population would
look like."

---

## NEW Slide 9.5 — "What's driving the prevalence-prior contribution?"

**Slot.** Between slides 9 and 10 (or replaces slide 10's first row).

**Title.** "Color cast — not rulers, not vignettes — is the dominant causal shortcut."

**Bullets.**

- We removed each candidate shortcut via inpainting + measured ΔP on
  the analytical sample (N=300).
- **Vignette** (literature: Bevan 2022): ΔP = +0.007, ns. *Null.*
- **Ruler** (literature: Winkler 2019): ΔP = +0.003, ns. *Null.*
- **Hair** (negative control): ΔP = −0.066, p < 0.0001 — inpainting
  artifact baseline.
- **Color cast (non-spatial, exploratory)**: ΔP = **+0.104**,
  p < 0.0001 — **the dominant causal shortcut**.

**Punchline.** "The literature's named spatial shortcuts (rulers,
vignettes) are *null* on this checkpoint. The dominant shortcut is a
non-spatial colour drift across source collections — a finding the
prior dermatology AI shortcut literature does not name."

**One-line Q&A defense.** "Why color cast?" → "Each collection has a
distinct colour distribution from its acquisition device. The model
learned the distribution as a class prior. Removing the colour drift
(by normalizing to a mean signature) changes the prediction by +0.10
on affected images — a 1.6× larger effect than the inpainting-artifact
baseline."

**Figure.** `figures/fig02_marginal_dp.png` — forest plot of the four
ΔP estimates with 95% CIs.

---

## NEW Slide 9.6 — "Causal proof — shortcut removal closes 66% of the cross-collection gap"

**Slot.** After slide 9.5.

**Title.** "Shortcut removal closes 66% of the HAM-vs-SIIM gap; SIIM rises from chance to 0.72."

**Bullets.**

- Cross-collection AUC gap (HAM − SIIM) is **+0.43 at original**,
  **+0.15 after all-shortcut removal** — 66% closed.
- c=70 AUC **rises** from 0.53 to 0.72 under shortcut removal. The
  shortcuts were *masking* its lesion signal.
- c=212 AUC *falls* (0.96 → 0.87) because removing the colour-
  signature transform also strips legitimate lesion-discriminative
  pigment cues.

**Punchline.** "This is direct causal evidence — not Simpson-paradox
inference. Removing the shortcuts in a fixed model produces a measurable
AUC shift in the predicted direction. The Phase 7c modality-prior
attribution was correct; this slide quantifies it."

**Figure.** `figures/fig03_gap_decomposition.png` — per-collection
bar plot, original vs all-removed, with the gap-closure annotation.

---

## Slide 10 — Audit closures, with Phase 9 additions

**Current closures.** W1 / W2 / W4 / W8 / H5 / M1 / M4 (7 items).

**Phase 9 adds.**

- **W4-extended (NEW)**: `trainer.py:load_and_merge_metadata` uses
  file-system iteration order for CSV concat. ext4 and APFS return
  different orders, so the seed-42 patient split differs across
  operating systems. Closed in Phase 9 by forcing explicit
  creation-order in the new pipeline. Will fold back into trainer.py
  with a one-line `sorted(... key=lambda p: int(re.search(r"c(\d+)",
  p.name).group(1)))`.
- **W7-partially-closed**: Phase 9 ran a Grad-CAM peak-region audit;
  the H1 χ² test was falsified (attention is uniform across
  collections) — so we cannot say "the model attends differently."
  But the H2 counterfactual shows it *reacts differently* — attention
  and counterfactual response are orthogonal axes. Note this on the
  slide as "W7 measurement complete; the finding is that attention
  patterns are not the right diagnostic — counterfactual ΔP is."

**Phase 9 adds to "open" list.**

- *Inpainting-artifact baseline disclosure*: Hair NC produced a
  −0.066 mean ΔP under Telea inpainting. All effect sizes in §3.2
  should be read as "raw" or "hair-adjusted."
- *LaMa upgrade as future work*: would tighten the noise floor.

---

## Slide 11 — Realtime framing-aid

**Current rationale.** "Live attention overlay is a weaker requirement
than diagnostic classification; PAD-UFES-20's 0.81 AUC supports the
weaker requirement holding."

**Phase 9 nuance to add (one bullet).**

- **The dominant shortcut on this model is non-spatial (color cast).
  A live attention overlay cannot show color cast.** The framing aid
  remains useful (the user still needs to frame the lesion); but its
  failure mode is "the user produces a well-framed shot whose colour
  signature still matches the model's collection-prior." Disclose
  this on the slide.

---

## Slide-by-slide deltas summary

| Slide | Phase 9 effect |
|---|---|
| 1 (problem framing) | unchanged |
| 2 (dataset) | unchanged |
| 3, 4, 5, 6 (methods) | unchanged |
| 7 (headline) | "footnote" disclosure: balanced AUCs in 9 |
| 8 (PR-AUC × prevalence) | unchanged |
| **9 (per-collection breakdown)** | **rewritten** — balanced AUC column + c=70 chance disclosure |
| **9.5 (NEW)** | **per-shortcut ΔP forest plot** |
| **9.6 (NEW)** | **gap-closure bar chart + c=70 reversal** |
| 10 (audit closures) | adds W4-extended, W7-partial, inpainting disclosure |
| **11 (realtime framing-aid)** | **non-spatial-shortcut nuance bullet** |
| 12 (multiclass) | unchanged |
| 13 (timeline) | adds Phase 9 line; "post-midterm: LaMa upgrade, OOD chain" |
| B1–B9 (backup) | B7 (Grad-CAM artifact) becomes a real B7 — see fig01 forest |

---

## What the presenter says when transitioning

> "We had a clean Phase 7c finding — the aggregate 0.95 includes a
> modality-prior contribution. Phase 9 takes that one step deeper:
> *which* modality cues, and *how much* of the headline?
>
> Holding prevalence constant collapses c=70's AUC from 0.88 to 0.55.
> The aggregate's 0.95 was even more prevalence-driven than Phase 7c
> disclosed.
>
> Counterfactually removing the candidate shortcuts shows the
> well-documented spatial cues (rulers, vignettes) are null on this
> checkpoint. The real cue is non-spatial — a colour drift across
> the source collections. Inpainting away the colour drift produces
> a +0.10 prediction shift.
>
> Removing all candidates together closes 66% of the HAM-vs-SIIM gap
> and lifts SIIM's AUC from chance to 0.72. The shortcuts were
> *masking* SIIM's lesion signal. This is direct causal evidence
> rather than the associational reasoning of slide 9.
>
> Three new slides, ten minutes total in the second half of the deck.
> The original Phase 7c narrative stays — Phase 9 sharpens its
> implications and corrects which shortcut is the load-bearing one."

That is the 90-second elevator pitch the presenter delivers on the
new slides 9.5 and 9.6.
