# 추가 컨텍스트 - HANDOFF_FOR_PPT.md 후속

> PPT 제작용 보조 문서. HANDOFF_FOR_PPT.md 와 짝으로 사용.
> 메인 storyline / 팀 framing / 발표 flow / Q&A 보완 / 할루시네이션 검증.

---

## A. 5단계 실험 narrative 흐름 - PPT 의 메인 storyline (가장 중요)

학술 발표 표준 흐름. 상황 > 문제 파악 > 문제 내용 > 원인 분석 > 해결 > 기대 결과.

### 단계 1. 상황 (Setup)

- ISIC 3 개 dermoscopy 데이터셋 (HAM10000 + SIIM-2020 + BCN20000, 총 61,396 image) 으로 EfficientNet-B0 학습
- Test AUC 0.9513 달성
- 동시에 11 개 audit weakness 식별 + 7 개 fix (patient-level split, threshold calibration, seed reproducibility 등)
- 다만 per-collection 으로 보면 AUC 0.87-0.92 - aggregate (0.95) 보다 낮음
- "aggregate 가 부분의 평균보다 높다" 이상 패턴. Modality-prior shortcut 의 간접 의심 - 직접 증거 없는 상태.

### 단계 2. 문제 파악 (Problem Identification)

만약 진짜 shortcut 이면:
- 모델이 환자 의학적 상태가 아니라 데이터셋 출신을 단서로 씀
- 임상 배포 시 다른 병원, 다른 카메라에서 recall 폭락 - 환자 놓침
- PAD-UFES-20 (스마트폰) 에서 recall 0.51 로 추락 - 절반 놓침. 간접 증거 강화

이 의심을 quantitative 한 직접 증거로 격상해야 함.

### 단계 3. 문제 내용 (Problem Description)

(a) Prevalence-balanced 환경에서:
- c=70 SIIM-2020 AUC: 0.88 > 0.55 (chance 수준)
- 0.88 의 70% 가 prevalence prior contribution

(b) Counterfactual inpainting 으로 4 shortcut 측정:

| Shortcut | dP | 결과 |
|---|---|---|
| Vignette (Bevan 2022) | +0.007 | null |
| Ruler (Winkler 2019) | +0.003 | null |
| Hair (negative control) | -0.066 | inpainting baseline |
| Color cast | +0.104 | dominant cue |

학계 유명 shortcut 가설 (룰러, 비네팅) 이 이 모델엔 null. 진짜 cue 는 color cast (literature 에 없음).

(c) Cross-collection gap 분석:
- 인공물 모두 제거 시 gap 의 66% 닫힘
- c=70 AUC: 0.53 > 0.72 상승 - shortcut 이 lesion signal 가렸음

### 단계 4. 원인 분석 (Cause Analysis)

"Color cast" 가 정확히 무엇인가? 추가 분해:

(a) HSV channel decomposition:
- Hue dP = -0.243 (93% of joint)
- Saturation dP = +0.010 (null)
- Value dP = -0.005 (null)
- Hue 가 진짜 cue.

(b) Spatial decomposition:
- 전체 normalize: dP = +0.135
- Background 만 normalize: dP = +0.094 (70%)
- Lesion 만 normalize: dP = +0.053 (weak)
- Color shortcut 이 lesion 주변 (background) 에 있음.

(c) Mechanism 통합:
- "Color cast" = "Background Hue Cast"
- 각 ISIC 컬렉션이 다른 dermoscope + institution + processing 에서 acquisition. 그 hue 차이가 systematic. BCN20000 학습 셋 54% malignant > 모델이 "BCN 풍 hue > 악성" 외움.

### 단계 5. 해결 (Solution)

(a) Test-time intervention - LACN (Lesion-Aware Color Normalization):
- 재학습 X. 즉시 deployment 가능
- 추론 시 Otsu segment > background 만 Lab-mean shift > classify
- 5ms overhead per image

(b) Training-time intervention - Color-Invariant Fine-Tune:
- Saturation jitter 빼고 (null), Hue jitter 만 (0.10, 두 배)
- 기존 ckpt + 5 epoch 짧은 fine-tune
- 학습 중 매 batch 마다 hue 흔들기 > 모델이 hue 단서 못 믿음 > 진짜 병변 봐야 함

### 단계 6. 기대 결과 (Expected Outcome)

| 가설 | 기대 변화 | 현재 상태 |
|---|---|---|
| H6 | Test AUC 0.93-0.96 유지 | Supported (0.9369 검증됨) |
| H7 | Color cast dP 50% 이상 감소 | 측정 대기 |
| H8 | c=70 balanced AUC 0.65+ 상승 | 측정 대기 |
| H9 | Cross-collection gap 30% 이상 감소 | 측정 대기 |
| H10 | OOD recall 유지 (safety) | 측정 대기 |

H7-H10 도 supported 되면: "4-axis 완성형 연구" - Audit > Mechanism > Test-time fix > Training-time fix

### 5 단계 > 슬라이드 매핑

| 단계 | 슬라이드 번호 | 시간 |
|---|---|---|
| 1. 상황 | 1-7 (모델 + 0.9513) | 7m |
| 2. 문제 파악 | 9-11 (per-collection 의심 + OOD) | 2m |
| 3. 문제 내용 | 12-15 (PF#1, PF#2, PF#3) | 4m |
| 4. 원인 분석 | 16-17 (Phase 11, 12 mechanism) | 2m |
| 5. 해결 | 18-19 (Phase 10 + LACN) | 2m |
| 6. 기대 결과 + 종합 | 20 (H6-H10 + future work) | 1m |

---

## B. 팀 프로젝트 framing 가이드

- 모든 작업을 "우리 팀 (we)" 으로 통합
- 누가 무엇 했다 명시 X
- 1-2 단계 (중간 발표 분량) 와 3-6 단계 (기말 추가) 가 자연스럽게 흐름으로 이어지게
- "한계 발견 + 해결 시도" 가 팀의 honest 자기 비판
- 청중이 "정직하고 깊이 있는 팀이네" 인상.

---

## C. 발표 dramatic flow 팁

- 슬라이드 7 (헤드라인 0.9513) > 청중 "오 좋은 결과"
- 슬라이드 9 (per-collection 의심) > 청중 "음 뭔가 이상한데"
- 슬라이드 13 (PF#1: 0.55) > 청중 "0.88 이 사실 chance 였어?" <- 발표의 결정적 순간
- 슬라이드 15 (PF#3: 0.53 > 0.72 상승) > 청중 "오 진짜 shortcut 이었네"
- 슬라이드 18 (Phase 10: 0.94 유지) > 청중 "해결도 가능하구나"

이 5 모먼트가 발표의 결정적 강조점.

---

## D. Self-critique narrative 강력 추천

발표에서 "우리 모델의 한계를 우리가 발견하고 해결 시도" 형식이 최강. 슬라이드 톤 예시:

"Test AUC 0.9513 까지 만들었지만, 정직한 self-evaluation 으로 prevalence prior shortcut 을 발견하고 mechanism 까지 식별했습니다. 그리고 그 발견을 fix 까지 가져갔습니다 - 방치하지 않고."

self-critique + intervention 의 완성형 narrative.

---

## E. 추가 Q&A

Q: "Hair (negative control) 이 왜 유의했나요?"
A: Inpainting 자체가 blurring artifact 도입. Hair 제거 시 P_malig 가 -0.07 떨어짐 = inpainting baseline noise. Color cast 의 +0.10 은 그 baseline 의 1.6 배라 진짜 effect.

Q: "Vignette/ruler null 인데 그럼 literature 가 틀린 건가요?"
A: 아닙니다. literature 의 finding 들은 그 논문의 특정 ckpt + 특정 데이터셋 위에서 valid. 우리 ckpt 는 4-way patient-level split + focal alpha=0.85 + 의도적 색 augmentation 제외 등 다른 학습 셋팅이라 vignette/ruler 의존도가 다르게 됨. Color cast 가 dominant 인 게 이 ckpt 의 특이성. 학계 generalization 이 안 된 게 아니라 ckpt 마다 shortcut profile 이 다름.

Q: "발표 후 어떻게 발전?"
A: Multi-OOD cohort 검증 (FitzPatrick17), expert lesion mask 으로 LACN refinement, Phase 11-informed hue-only ablation (saturation 빼고만으로도 충분한지). Phase 10 의 H7-H10 검증 (현재 진행 중) 도 곧 완료.

---

## F. 할루시네이션 검증 - 솔직 평가

수치는 다 진짜, 다만 주의할 부분 있음.

### 검증된 진짜 수치 (log 파일에 있음)

| 수치 | 어디서 검증 가능 |
|---|---|
| Test AUC 0.9513 | artifacts/bootstrap_test_auc.log (팀원 작업) |
| c=70 balanced AUC = 0.548 | artifacts/phase9/04_inference.log |
| Color cast dP = +0.104 | artifacts/phase9/07_counterfactual.log |
| Vignette/Ruler/Hair dP | 같은 log |
| 66% gap closure | artifacts/phase9/08_statistics.log |
| Hue dP = -0.243 (93%) | artifacts/phase11/01_hsv_decomposition.log |
| Background = 70% | artifacts/phase12/01_lacn.log |
| Phase 10 Pre 0.80 > Post 0.94 | 본인이 Colab 스크린샷으로 직접 본 결과 |

발표에서 인용할 모든 핵심 수치가 실제 측정 결과. 할루시네이션 X.

### 주의해야 할 부분

1. 주관적 평가 표현 - 그대로 발표 X
   - "박사급 standard", "학부 term project 상위", "논문 publishable" 같은 표현은 평가지 객관적 사실 X
   - 발표에선 제거하거나 완화
   - 대신: "4-axis 완성형 분석" 같은 fact 기반 표현

2. Q&A 답안 - 외워서 답 X, 본인이 이해해서 답
   - Q&A 답은 우리 작업 기반으로 생성한 것
   - 본인이 각 답의 reasoning 진짜로 이해해야 자연스럽게 답 가능
   - 특히 "color cast 가 왜 shortcut?", "왜 hue jitter 0.10?" 같은 깊은 질문은 본인 이해 필수

3. Prior work 인용 - 정확한 paper, 다만 세부 인용 확인 권장
   - Geirhos 2020, Bissoto 2019/2022, Nauta 2022, Winkler 2019, FastDiME 2023, MaskMedPaint 2024 - 모두 실제 paper
   - 다만 paper 제목/저자/year 의 정확성 100% 보장은 안 됨
   - 발표 슬라이드에 인용 박을 때 Google Scholar 에서 한 번 확인 권장 (1분)

4. 팀원 작업 (Phase 1-8) 의 미세 detail
   - README 와 docs/midterm-prep-context.md 에서 직접 확인 가능
   - Hyperparameter, audit weakness 번호 (W1, W2, ...) 등은 정확
   - 다만 팀원의 의도 (왜 그렇게 했나) 는 코드/문서 기반 추론 - 100% 정확 X

5. Phase 10 의 2nd run 결과 (H7-H10)
   - 아직 측정 안 됨 (Colab 2nd run 진행 중)
   - 기대값으로 표시 (예: "c=70 0.55 > 0.65 예상") - 예측이지 결과 X
   - 발표에서 "측정 진행 중" 으로 표현, 결과로 표현 X

### 가장 안전한 발표 자료 준비 방법

1. 수치는 log/csv 에서 직접 확인 - artifacts/phase9/, phase11/, phase12/ 폴더
2. figure 는 직접 봐서 확인 - artifacts/phase*/figures/*.png
3. paper 문서는 검토 - docs/phase9/06_paper.md 가 가장 정리됨
4. 주관적 표현 (박사급 등) 삭제 - fact 만
5. Phase 10 의 H7-H10 결과 - 측정 진행 중으로 표시 (아직 실제로 없음)

### 결론

| 카테고리 | 신뢰도 |
|---|---|
| 측정 수치 (AUC, dP, gap closure) | 100% - log 확인 가능 |
| Phase 10 의 Test AUC 0.94 | 100% - 본인이 스크린샷 직접 봄 |
| Phase 10 의 H7-H10 결과 | 0% - 아직 측정 안 됨, 예측만 |
| Prior work 인용 (paper 명) | 95% - 실제 paper, 세부 확인 권장 |
| 5 단계 narrative 흐름 | 100% - 학술 발표 표준 구조 |
| 주관적 평가 ("박사급" 등) | 추측 - 발표에서 제거 권장 |
| Q&A 답안 | 우리 작업 기반 생성 - 본인 이해 필수 |

발표에서 인용할 핵심 수치 다 진짜. 주관적 표현만 빼면 안전.

---

문서 끝. HANDOFF_FOR_PPT.md 와 이 파일을 같이 컨텍스트로 사용.
