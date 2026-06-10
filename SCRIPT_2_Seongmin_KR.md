# 발표 대본 (한국어 이해용 + 영어 발표용) — 발표자 2: Lee Seongmin (나)

슬라이드 6-12 (핵심 조사: audit, mechanism, fixes)  ·  목표 ~4.5분
각 슬라이드: [한국어 - 무슨 말인지] → [English - 실제로 말할 것]
* 한국어는 영어와 같은 내용/같은 숫자입니다. 둘 다 실제 사람이 발표하듯 자연스럽게 썼습니다.

---

## [슬라이드 6 — 전환점(The pivot)]

[한국어]
자, 여기서부터가 진짜 본론이에요. 저희한테는 이 모델이 병변이 아니라 어느 데이터셋에서 온 사진인지를 보고 찍는 것 같다는 의심이 있었습니다. 목표는 그걸 실제로 증명하고, 또 고치는 거였어요. 그리고 하나 짚고 싶은 게, 저희는 전부 사전등록을 했습니다. 결과를 보기 전에, 가설 16개랑 각각 "이러면 틀린 거다"라는 기준을 미리 적어서 git에 올려놨어요. 그래서 결과 나온 다음에 말을 맞출 수가 없습니다. 조사는 네 부분으로 진행돼요. audit, mechanism, 그리고 테스트 단계 fix랑 학습 단계 fix입니다.

[English]
Okay, so this is where the real work starts. We had a suspicion that the model was using the collection a photo comes from, not the lesion itself. Our goal was to actually prove that, and then to fix it. I also want to point out that we preregistered everything. Before we looked at any results, we wrote down sixteen hypotheses, each with a clear condition for when it would count as wrong, and we committed them to git. So we couldn't just fit a story to the results afterwards. The investigation has four parts: the audit, the mechanism, a test-time fix, and a training fix.

## [슬라이드 7 — Finding 1: 유병률 균형 AUC]

[한국어]
먼저, 각 데이터셋을 50 대 50으로 균형 맞춰서 다시 테스트했어요. SIIM 데이터셋은 AUC가 0.88에서 0.55로 떨어졌습니다. 거의 찍기 수준이에요. 그러니까 그 0.88의 대부분, 한 70% 정도는 모델이 병변을 본 게 아니라, 이 데이터셋에 암이 얼마나 흔한지를 이용한 것뿐이었던 거죠.

[English]
First, we re-tested each collection at a balanced, fifty-fifty split. For the SIIM collection, the AUC dropped from 0.88 to 0.55. That is basically chance. So most of that 0.88, around seventy percent, was not the model reading the lesion. It was just using how common cancer is in that collection.

## [슬라이드 8 — Finding 2: 반사실 ΔP]

[한국어]
다음으로, 그럼 진짜 단서가 뭔지 알고 싶었어요. 그래서 반사실적 inpainting을 썼습니다. 의심되는 요소를 하나씩 지우고, 예측이 얼마나 바뀌는지 재는 거예요. 학계에서 잘 알려진 단서인 자(ruler)랑 가장자리 어둠(vignette)은 우리 모델에선 효과가 없었어요. 미리 가설로 등록해뒀는데 틀린 걸로 나온 거죠. 털(hair)은 음성 대조군인데, inpainting 자체가 만드는 기본 잡음을 알려주는 역할입니다. 진짜 단서는 색감(color cast)이었어요. 그 효과가 기준선의 1.5배가 넘었고, 기존 논문엔 보고된 적 없는 거예요. 오른쪽 실제 사진을 보시면, 색감을 보정했을 때 예측이 0.6 넘게 움직입니다.

[English]
Next, we wanted to know what the actual cue is. So we used counterfactual inpainting: we remove one suspected cue at a time, and we measure how much the prediction changes. The shortcuts that are well known in the literature, the ruler and the vignette, turned out to be null here. We had preregistered them, and they were falsified. Hair is our negative control; it tells us the baseline noise that the inpainting itself adds. The real cue was the color cast. Its effect was more than one and a half times that baseline, and it is not something reported in the literature. You can see it on real images on the right: when we normalize the color cast, the prediction moves by more than 0.6.

## [슬라이드 9 — Finding 3: 격차 분해]

[한국어]
세 번째로, 색감이 정말 단서라면 그걸 없앴을 때 오히려 좋아져야겠죠. 실제로 그렇습니다. 감지된 단서를 다 지웠더니 데이터셋 사이 격차가 66% 줄었어요. 그리고 SIIM 점수는 오히려 0.53에서 0.72로 올라갔습니다. 그러니까 그 단서는 단순 잡음이 아니라, 진짜 병변 신호를 가리고 있었던 거예요.

[English]
Third, if color is really a shortcut, then removing it should help. And it does. When we remove the detected cues, the gap between collections drops by sixty-six percent. And the SIIM AUC actually goes up, from 0.53 to 0.72. So the shortcut was not just noise. It was hiding the real lesion signal.

## [슬라이드 10 — 메커니즘: 배경 색조 캐스트]

[한국어]
그럼 이 단서가 정확히 뭘까요. 두 방향으로 쪼개봤어요. 색 성분으로 보면 hue가 효과의 93%고, 채도랑 밝기는 거의 0입니다. 위치로 보면 70% 정도가 병변이 아니라 배경에 있어요. 그래서 저희는 이걸 배경 색조(background hue cast)라고 부릅니다. 왜 생기냐면, 데이터셋마다 촬영 장비가 달라서 각자 고유한 색조를 갖거든요. 여기 그 색조 차이가 보입니다. 하나 더 확인한 게 있는데, 고치고 나서도 모델이 보는 위치는 안 바뀝니다. 당연한 게, 이건 색 단서지 위치 단서가 아니니까요.

[English]
So then, what exactly is this cue? We broke it down two ways. By color channel, hue is ninety-three percent of the effect; saturation and value are basically zero. By location, about seventy percent of it is in the background, not on the lesion. So we call it a background hue cast. Why is it there? Each collection was shot on different equipment, so each one has its own systematic hue. You can see those signatures here. One more thing we checked: after the fix, the model's attention does not actually move. And that makes sense, because this is a color cue, not a location cue.

## [슬라이드 11 — 해결: 두 가지 개입]

[한국어]
그래서 고치는 방법이 두 가지예요. 첫 번째는 LACN인데, 테스트 단계에서 동작합니다. 추론할 때 배경 색을 표준으로 맞춰주는 거고, 재학습은 없어요. 단서를 한 70% 없애줍니다. 근데 가설 하나가 여기서 실패했어요. 점수가 전체적으로 밀려서, 실제로 쓰려면 기준값을 다시 잡아야 합니다. 그래서 유망하긴 한데 아직 완성은 아니에요. 두 번째는 학습 단계에서 동작합니다. hue에만 jitter를 줘서 몇 epoch 동안 미세조정하는 건데, 그러면 모델이 색에 의존하지 않게 배웁니다.

[English]
So we have two ways to fix it. The first one is LACN, which works at test time. At inference, we normalize the background color, with no retraining. It removes about seventy percent of the cue. But one of our hypotheses failed here: it shifts the scores, so the threshold has to be recalibrated before you could deploy it. So it is a promising option, but it is not finished yet. The second one works during training. We add jitter only on the hue and fine-tune for a few epochs, so the model learns not to rely on color.

## [슬라이드 12 — 결과: 학습 단계 fine-tune]

[한국어]
그리고 이 학습 쪽 방법은 제대로 됐습니다. 미리 등록한 가설 5개가 전부 통과했어요. 제일 중요한 결과는, 찍기 수준이던 SIIM이 0.55에서 0.87로 올라간 겁니다. 색감 효과는 86% 줄어서 통계적으로 0이 됐고, 격차는 79% 닫혔어요. 그리고 한 가지는 솔직하게 말씀드릴게요. 전체 AUC는 0.014 정도 살짝 떨어집니다. 이건 진짜 trade-off예요. 익숙한 데이터에서의 정확도를 조금 내주는 대신, 여러 데이터셋에 걸친 견고함을 훨씬 많이 얻는 겁니다. 스크리닝 도구 입장에선 이게 맞는 교환이라고 봅니다. 이제 첸이 이걸 기존 연구랑 비교하고 마무리하겠습니다.

[English]
And the training fix works. All five of our preregistered hypotheses held up. The main result is that the SIIM balanced AUC went from 0.55, which is chance, up to 0.87. The color-cast effect dropped by eighty-six percent, down to statistically null. And the gap closed by seventy-nine percent. Now, I should be upfront about one thing: the overall AUC does drop a little, by about 0.014. That is a real trade-off. We give up a small amount of accuracy on familiar data, and in return we get a lot more robustness across collections. For a screening tool, we think that is the right trade. I will pass it to Chen to compare this with previous work and wrap up.

---

전달 팁 (내 파트가 핵심이라 중요)
- 평소보다 약간 천천히.
- 핵심 숫자 직후엔 잠깐 멈추기: "0.88 to 0.55" / "0.55 ... up to 0.87" / "eighty-six percent" / "seventy-nine percent".
- 슬라이드 12 trade-off 부분은 차분하고 담담하게. 0.014 떨어진 걸 인정하는 게 약점이 아니라 강점.
- 마지막은 첸에게 넘기는 문장으로 끝.
