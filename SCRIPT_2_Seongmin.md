# Presentation Script — Speaker 2: Lee Seongmin

Skin Lesion Binary Classifier — Team 3 Final
Slides 6-12 (the investigation: audit, mechanism, fixes)  ·  Target ~4.5 min

---

## [Slide 6 — The pivot]

Okay, so this is where the real work starts. We had a suspicion that the model was using the collection a photo comes from, not the lesion itself. Our goal was to actually prove that, and then to fix it. I also want to point out that we preregistered everything. Before we looked at any results, we wrote down sixteen hypotheses, each with a clear condition for when it would count as wrong, and we committed them to git. So we couldn't just fit a story to the results afterwards. The investigation has four parts: the audit, the mechanism, a test-time fix, and a training fix.

## [Slide 7 — Finding 1: prevalence-balanced AUC]

First, we re-tested each collection at a balanced, fifty-fifty split. For the SIIM collection, the AUC dropped from 0.88 to 0.55. That is basically chance. So most of that 0.88, around seventy percent, was not the model reading the lesion. It was just using how common cancer is in that collection.

## [Slide 8 — Finding 2: counterfactual delta-P]

Next, we wanted to know what the actual cue is. So we used counterfactual inpainting: we remove one suspected cue at a time, and we measure how much the prediction changes. The shortcuts that are well known in the literature, the ruler and the vignette, turned out to be null here. We had preregistered them, and they were falsified. Hair is our negative control; it tells us the baseline noise that the inpainting itself adds. The real cue was the color cast. Its effect was more than one and a half times that baseline, and it is not something reported in the literature. You can see it on real images on the right: when we normalize the color cast, the prediction moves by more than 0.6.

## [Slide 9 — Finding 3: gap decomposition]

Third, if color is really a shortcut, then removing it should help. And it does. When we remove the detected cues, the gap between collections drops by sixty-six percent. And the SIIM AUC actually goes up, from 0.53 to 0.72. So the shortcut was not just noise. It was hiding the real lesion signal.

## [Slide 10 — Mechanism: background hue cast]

So then, what exactly is this cue? We broke it down two ways. By color channel, hue is ninety-three percent of the effect; saturation and value are basically zero. By location, about seventy percent of it is in the background, not on the lesion. So we call it a background hue cast. Why is it there? Each collection was shot on different equipment, so each one has its own systematic hue. You can see those signatures here. One more thing we checked: after the fix, the model's attention does not actually move. And that makes sense, because this is a color cue, not a location cue.

## [Slide 11 — Solution: two interventions]

So we have two ways to fix it. The first one is LACN, which works at test time. At inference, we normalize the background color, with no retraining. It removes about seventy percent of the cue. But one of our hypotheses failed here: it shifts the scores, so the threshold has to be recalibrated before you could deploy it. So it is a promising option, but it is not finished yet. The second one works during training. We add jitter only on the hue and fine-tune for a few epochs, so the model learns not to rely on color.

## [Slide 12 — Result: training fine-tune]

And the training fix works. All five of our preregistered hypotheses held up. The main result is that the SIIM balanced AUC went from 0.55, which is chance, up to 0.87. The color-cast effect dropped by eighty-six percent, down to statistically null. And the gap closed by seventy-nine percent. Now, I should be upfront about one thing: the overall AUC does drop a little, by about 0.014. That is a real trade-off. We give up a small amount of accuracy on familiar data, and in return we get a lot more robustness across collections. For a screening tool, we think that is the right trade. I will pass it to Chen to compare this with previous work and wrap up.

---

Delivery notes
- This is the core of the talk, so speak a little slower than feels natural.
- Pause briefly after the key numbers so they land: "0.88 to 0.55", "0.55 ... up to 0.87", "eighty-six percent", "seventy-nine percent".
- On slide 12, say the trade-off part plainly and calmly. Owning the small drop is a strength, not an apology.
- End on the line that hands over to Chen.
