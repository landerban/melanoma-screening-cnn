# 발표 대본 (한국어 이해용 + 영어 발표용) — 발표자 1: Alexander Park

슬라이드 1-5 (도입 + 모델 셋업)  ·  목표 ~3.5분
각 슬라이드: [한국어 - 무슨 말인지] → [English - 실제로 말할 것]

---

## [슬라이드 1 — 타이틀]

[한국어]
안녕하세요, 저희는 3조입니다. 저희가 만든 건 피부 병변을 양성/악성으로 가려내는 스크리닝 모델인데요, 사실 이 발표의 핵심은 모델 자체보다 "이 모델이 진짜로 뭘 보고 판단하는지"를 정직하게 파헤친 부분입니다. 일단 AUC 0.95라는 좋은 점수에서 출발하지만, 저희는 그 점수 밑에 숨어 있는 함정(shortcut)을 찾으러 갑니다. 제가 먼저 문제 상황이랑 모델을 소개하고, 그다음 팀원들이 본격적인 조사 내용을 설명할게요.

[English]
Good morning. We are Team 3. Our project is a skin lesion screening model — but more than that, it is an honest audit of what our model actually learned. We start from a strong result, a 0.95 AUC, and then we go looking for the shortcut hiding underneath it. I will set up the problem and the model, and then my teammates will walk you through the investigation.

## [슬라이드 2 — 문제 & 데이터]

[한국어]
먼저 문제부터요. 이건 결국 "환자를 병원에 보낼까 말까" 둘 중 하나를 고르는 문제예요. 그리고 정확도보다 recall, 즉 놓치지 않는 게 훨씬 중요합니다. 암을 한 번 놓치는 대가가 괜히 한 번 더 검사받게 하는 것보다 비교도 안 되게 크니까요. 게다가 데이터가 양성 대 악성이 4.3 대 1로 치우쳐 있어서, 그냥 전부 "양성"이라고 찍어도 정확도는 높게 나와요. 그래서 정확도가 아니라 이 "놓치면 큰일 난다"는 비대칭이 우리 평가지표와 손실함수, 기준값을 정합니다. 데이터는 ISIC 피부경 사진 세 묶음, 약 6만 1천 장이고요. 스마트폰으로 찍은 사진 묶음은 일부러 빼서 "처음 보는 환경" 테스트용으로 남겨뒀습니다. 그리고 같은 환자 사진이 학습이랑 테스트에 섞이지 않게 환자 단위로 나눴습니다.

[English]
First, the problem. Screening is a binary question: refer the patient, or don't. It is recall-first, because a missed melanoma costs far more than an unnecessary referral. At our four-to-one imbalance, a model can score high just by calling everything benign — so the cost asymmetry, not accuracy, drives our metric, our loss, and our threshold. The data is three dermoscopy collections from ISIC, about sixty-one thousand images. We hold out a smartphone cohort as an out-of-distribution test. And the split is patient-level, so no patient leaks between training and test.

## [슬라이드 3 — 모델 & 학습]

[한국어]
모델은 EfficientNet-B0에 작은 분류기를 얹은 구조고, 두 단계로 학습했어요. 처음엔 본체는 고정하고 분류기만, 그다음에 전체를 미세조정했습니다. 한 가지 짚고 갈 게, focal loss라는 손실함수의 alpha 값을 기본값과 반대로 뒤집었어요. 기본값은 많은 쪽(양성)에 무게를 더 주는데, 우리한텐 그게 거꾸로거든요. 그래서 드문 쪽인 악성에 무게가 실리도록 바꿨습니다.

[English]
The model is EfficientNet-B0 with a small head, trained in two stages: first the head with a frozen backbone, then full fine-tuning. One detail matters here — we inverted the default focal-loss alpha. The standard value up-weights the majority class, which is the wrong direction for us. We set it so the rare malignant class carries the weight.

## [슬라이드 4 — 헤드라인 결과]

[한국어]
결과는 좋습니다. 환자 단위로 나눈 테스트에서 AUC 0.9513, 검증 점수랑 테스트 점수 차이가 거의 0이에요. recall은 0.835, specificity는 0.914고요. 악성 1,767건 중에 292건을 놓치는데, 바로 이 "놓친 칸"이 저희가 줄이려고 애쓰는 숫자입니다. 여기까지만 보면 깔끔하고 강한 결과죠.

[English]
And the headline is strong: a test AUC of 0.9513, on a patient-level split, with a validation-to-test gap of basically zero. Recall is 0.835, specificity 0.914. Out of 1,767 malignant cases, the model misses 292 — and that false-negative cell is exactly what we minimize against. So far, a clean, strong result.

## [슬라이드 5 — 결과의 균열]

[한국어]
그런데 여기서부터가 흥미롭습니다. 이 0.95를 데이터셋별로 쪼개보면, 어느 하나도 0.95에 못 미쳐요 — 정직하게 보면 0.87에서 0.92 사이입니다. 게다가 스마트폰 사진, 즉 처음 보는 환경에서는 recall이 0.51로 뚝 떨어집니다. 암의 절반을 놓치는 거예요. 여기서 의심이 들었습니다. "혹시 이 모델이 병변 자체가 아니라, 어느 데이터셋에서 온 사진인지를 보고 찍는 거 아닐까?" 이걸 확인하기 위해, 조사를 맡은 성민이에게 넘기겠습니다.

[English]
But here is where it gets interesting. When we break that 0.95 down per collection, every collection scores lower than the aggregate — the honest range is only 0.87 to 0.92. And on the out-of-distribution smartphone images, recall collapses to 0.51 — half the cancers missed. That raised a suspicion: maybe the model is keying on which collection an image comes from, not on the lesion itself. To find out, I'll hand over to Seongmin, who led the investigation.

---

전달 팁
- 도입부라 편하게. 슬라이드 4("강한 결과")는 자신감 있게 마무리하고, 슬라이드 5에서 톤을 바꿔 "근데 이상하다"로 분위기 전환.
- 마지막은 성민에게 넘기는 문장으로 깔끔하게 끝.
