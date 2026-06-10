# Presentation Script — Speaker 3: Chen Qixuan

Skin Lesion Binary Classifier — Team 3 Final
Slides 13-15 (Positioning + conclusion + close)  ·  Target ~2 min  ·  ~250 words

---

## [Slide 13 — Positioning: contribution]

Thanks. I want to be clear about what is new here. Prior work has audited single shortcuts — Winkler on the ruler, Nauta on a color patch — one cue at a time. Others remove bias holistically, without isolating the cue, and FastDiME does counterfactuals for a single shortcut. What this study adds is the combination: multi-shortcut, per-cue attribution; a cross-collection gap decomposition; a shortcut-to-OOD link; and two fixes — a validated training fine-tune, and a promising test-time candidate. The point is this — we did not just re-train a published model and report accuracy. We measured what our model actually learned, found the mechanism, and fixed it.

## [Slide 14 — Conclusion]

To conclude. We reported a 0.95 AUC — and instead of stopping there, we found the shortcut beneath it, attributed its mechanism, and built two fixes. We also report the limits honestly: the evidence is attribution-level, on a single checkpoint and one out-of-distribution cohort, so multi-seed and multi-OOD validation are future work. And along the way, our audit even caught a reproducibility bug in our own pipeline. Every number you saw has a script, a log, and a commit behind it.

## [Slide 15 — Thank you]

That is our work. Thank you — and we are happy to take your questions.

---

Delivery notes
- This part is short, so you can speak slowly and clearly; no need to rush.
- The strongest line is "we did not just re-train a published model and report accuracy" — say it deliberately; it is the answer to the implementation question.
- After the last line, the team can step forward together for Q&A.
