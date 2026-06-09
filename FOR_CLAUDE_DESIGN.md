# FOR CLAUDE DESIGN - 기말 발표 PPT 제작 지시서 (Phase 10 반영 최종판)

> 같이 넘기는 파일: 이 문서 + HANDOFF_FOR_PPT.md + HANDOFF_PHASE10_RESULTS.md + HANDOFF_FOR_PPT_2_narrative.md + 피겨 14장.
> 슬라이드 텍스트는 전부 영어 (기존 팀원 덱이 영어라 통일). 지시는 한국어.
> 상태: Phase 1-12 전부 완료. Phase 10 fine-tune 5/5 hypothesis SUPPORTED. placeholder 없음.

---

## 0. 한 줄 요청

기존 팀원 최종 덱(SkinLesionClassifier.pptx)의 비주얼 스타일을 유지하면서, 기말 추가 작업(Phase 9-12)을 반영한
완성형 최종 발표 덱을 만든다. 발표 시간 10분 + Q&A 5분에 맞춰 약 14장으로 압축. 출력은 .pptx.

---

## 1. 발표 조건 (중요)

- 발표 10분 + 질의응답 5분.
- 분량 감각: 중간 발표 PDF(15장) 정도. 즉 14-15장이 적정. 절대 20장 넘기지 말 것.
- 페이스: 슬라이드당 약 40-45초. 텍스트 과밀 금지, 피겨와 핵심 숫자 위주.

---

## 2. 현재 상태 + 발표 반전 포인트

- 팀원 pptx(14장)는 중간 발표 내용까지만 있음. Phase 9-12(색 shortcut 발견 -> mechanism -> 해결)가 전부 빠짐.
- 이번 덱은 그 빠진 부분을 핵심으로 넣되, 기초(모델/결과)는 압축해서 빠르게 통과.

### 발표 반전(plot twist) - 반드시 활용
팀원 덱 슬라이드 7에 "Color-sensitive diagnostic cues intentionally preserved (색은 진단 신호라 일부러 보존)"
라고 적혀 있다. 그런데 Phase 9가 밝혀낸 것은 그 색이 진단 신호가 아니라 shortcut(컨닝 단서)이었다는 것.
이 모순을 발표의 반전 포인트로 연결. "우리는 색이 진단적이라 믿고 보존했으나, 정직한 검증으로 그것이
오히려 collection-specific shortcut임을 발견했고, 그것을 고치는 데까지 갔다."

---

## 3. 전체 storyline (탐정 6단계)

| 단계 | 내용 |
|---|---|
| 1. 상황 (Setup) | EfficientNet-B0로 ISIC 3개 학습, Test AUC 0.9513, audit로 7/11 weakness fix |
| 2. 문제 파악 | per-collection AUC가 aggregate보다 낮음 + OOD recall 0.51 추락 = shortcut 의심 |
| 3. 문제 내용 | PF#1(balanced AUC 0.55) / PF#2(color cast가 진짜 범인) / PF#3(66% gap 닫힘) |
| 4. 원인 분석 | Phase 11(Hue 93%) + Phase 12(background 70%) = "Background Hue Cast" |
| 5. 해결 | LACN(test-time) + color-invariant fine-tune(training-time) |
| 6. 결과 | Phase 10: 5/5 hypothesis SUPPORTED. c=70 0.55->0.87. 4-axis 완성형 검증 |

---

## 4. 최종 슬라이드 구성안 (목표 14장, 10분)

### Part A - 기초 (압축, 약 4분, 5장)

1. **Title** - Skin Lesion Binary Classifier (+ 고려대 로고 코너에, 로고 제공 시). Team 3 / June 2026.
2. **Problem & Dataset** - 흑색종 screening = recall-first binary. 3 ISIC dermoscopy(HAM/SIIM/BCN) 61,396장, 4.29:1 imbalance + OOD PAD-UFES-20. patient-level split(effective_patient_id).
3. **Method** - EfficientNet-B0 + MLP head, 2-stage transfer, Focal loss alpha=0.85(양성 가중), threshold 0.661(cal cohort). (핵심만, 한 장에 압축)
4. **Main Result** - Test AUC 0.9513 [0.9465-0.9563], recall 0.835, spec 0.914. val/test gap 0.0001(honest split). (간단히 confusion matrix)
5. **The headline doesn't tell the whole story** - per-collection AUC 0.87-0.92 (aggregate 0.95보다 낮음) + OOD recall 0.51. -> "aggregate가 부분 평균보다 높다" 이상 패턴. shortcut 의심 제기. (전환점)

### Part B - 기말 핵심 (약 6분, 8-9장)

6. **From suspicion to causal evidence** - research question. 슬라이드 5의 의심을 직접 증거로 격상. Audit -> Mechanism -> Fix 4-axis 소개.
7. **[PF#1] Prevalence-balanced AUC** - fig01_balanced_auc.png. c=70 SIIM 0.88 -> 0.55(chance). "0.88의 70%가 prevalence prior."
8. **[PF#2] Color cast is the dominant shortcut** - fig02_marginal_dp.png. counterfactual inpainting으로 4 shortcut 측정. vignette/ruler(학계 가설) null, color cast ΔP +0.104 (p<1e-5).
9. **[PF#3] Shortcut removal closes 66% of gap** - fig03_gap_decomposition.png. c=70 AUC 0.53 -> 0.72 상승(shortcut이 lesion signal 가렸음).
10. **Mechanism: Background Hue Cast** - fig12_combined_color_anatomy.png(통합) 또는 fig10+fig11 나란히. Hue가 93%(채널) + background가 70%(공간). 한 장에 통합 권장.
11. **Two interventions** - LACN(test-time, 재학습 X, 5ms) + Color-Invariant Fine-Tune(hue-only jitter 0.10, 5 epoch). 같은 mechanism 기반 두 경로.
12. **[Phase 10] Intervention works - 5/5 hypotheses supported** - fig14_pf1_before_after.png + fig17_fine_tune_trajectory.png. c=70 0.55->0.87, color cast ΔP 86%↓, gap 79%↓.
13. **Conclusion** - Audit -> Mechanism -> Test-time fix -> Training-time fix의 4-axis 완성형, empirically validated. Future work 한 줄.
14. **Q&A / Thank you**

(선택 여유 슬라이드: fig13_cam_grid_p6_vs_p10.png를 슬라이드 12에 보조로, fig05_cam_grid.png를 슬라이드 8/10에 직관 evidence로. 시간 빡빡하면 생략.)

---

## 5. 기본 정보 (중간 발표에서 검증된 fact - 일관 유지)

- 제목: Skin Lesion Binary Classifier - EfficientNet-B0 Based Benign-Malignant Screening
- 팀/날짜: Team 3 / June 2026. 소속: 고려대학교 정보대학 컴퓨터학과
- Framework: PyTorch 2.6.0 + CUDA 12.4
- Model: EfficientNet-B0 (ImageNet pretrained, ~5.3M params), input 384x384x3
- Head: GAP -> Dropout(0.4) -> Linear(1280->256) -> ReLU -> Dropout(0.2) -> Linear(256->1) -> Sigmoid
- Data: HAM10000(c=212, 11,720, ~18%) / SIIM-ISIC 2020(c=70, 33,126, ~1.7%) / BCN20000(c=249, 18,946, ~54%). Merged 61,396 (49,785 benign / 11,611 malignant = 4.29:1). OOD: PAD-UFES-20 (1,568)
- Split (patient-level via effective_patient_id): Train 65% / Val 10% / Cal 10% / Test 15% (test 9,186)
- Loss: Focal (alpha=0.85, gamma=2.5). Optimizer AdamW, seed=42, ES patience=7 on val AUC
- Headline: ROC-AUC 0.9513 [0.9465-0.9563], Recall 0.835, Specificity 0.914, F1 0.761, threshold 0.661
- Confusion matrix: TN 6,784 / FP 635 / FN 292 / TP 1,475
- Per-collection AUC (skewed): Aggregate 0.9513 / HAM 0.9151 / SIIM 0.8835 / BCN 0.8702

### Phase 9 핵심 수치 (실측)
- PF#1 balanced(50/50): HAM 0.9151->0.943, BCN 0.8702->0.778, SIIM 0.8835->0.548(chance, 강조)
- PF#2 ΔP: vignette +0.007(null), ruler +0.003(null), hair -0.066(NC baseline), color cast +0.104(dominant, p<1e-5)
- PF#3 gap(c=212 - c=70): 0.434 -> 0.148 = 66% closed. c=70 original 0.530 -> all_removed 0.723

### Phase 11/12 핵심 수치 (실측)
- HSV: Hue ΔP -0.243 (92.8%), Saturation +0.010(null), Value -0.005(null)
- Spatial: full +0.135, background-only(LACN) +0.094(70%), lesion-only +0.053

### Phase 10 결과 (실측, 5/5 SUPPORTED - placeholder 아님)
| 가설 | 예측 | 실제 | 결과 |
|---|---|---|---|
| H6 Test AUC 유지 | [0.93, 0.96] | 0.9369 | SUPPORTED |
| H7 Color cast ΔP 감소 | 50% 이상 | 86% 감소 (+0.104 -> +0.014, 통계적 null) | SUPPORTED |
| H8 c=70 balanced AUC | >= 0.65 | 0.867 (chance -> excellent) | SUPPORTED |
| H9 Cross-collection gap | <= 0.30 | 0.093 (79% 감소) | SUPPORTED |
| H10 OOD recall safety | >= 0.70 | 0.710 | SUPPORTED |

- PF#1 Phase 9 -> Phase 10: HAM 0.943->0.960, BCN 0.778->0.865, SIIM 0.548->0.867(+0.319 핵심)
- PF#2 color cast ΔP: +0.104 -> +0.014 (86% 감소)
- PF#3 gap: 0.434 -> 0.093 (79% 감소). 균일화: HAM 0.96 / SIIM 0.87 / BCN 0.87
- Fine-tune trajectory: Pre-FT 0.80 -> ep1 0.904 -> ep5 0.936 -> Post-FT test 0.9369 (ΔAUC +0.135). 2개 독립 Colab session에서 동일 trajectory(reproducible)
- Method: hue-only ColorJitter(hue=0.10, saturation=0.0), 5 epoch, lr=1e-5

---

## 5.5 FIGURE MANIFEST - 파일명 -> 슬라이드 -> 캡션 (업로드 시 원본 파일명 유지)

| 파일명 | 슬라이드 | 보여주는 내용 | 영어 캡션 |
|---|---|---|---|
| fig01_balanced_auc.png | 7 (PF#1) | Phase 7c(skewed) vs Phase 9(balanced) per-collection AUC. c=70 0.88->0.55 폭락 | "Prevalence-balanced AUC: c=70 SIIM collapses 0.88 -> 0.55 (chance)" |
| fig02_marginal_dp.png | 8 (PF#2) | 4 shortcut ΔP forest plot + 95% CI. color cast만 유의 | "Counterfactual delta-P: color cast is the only dominant shortcut (+0.104, p<1e-5)" |
| fig03_gap_decomposition.png | 9 (PF#3) | collection별 original vs all_removed AUC. c=70 상승 | "Removing shortcuts closes 66% of the gap; c=70 rises 0.53 -> 0.72" |
| fig12_combined_color_anatomy.png | 10 (Mechanism) | channel x spatial 통합 요약 | "Mechanism: Background Hue Cast (Hue 93% x Background 70%)" |
| fig10_hsv_decomposition.png | 10 보조 | H/S/V 채널별 ΔP, Hue만 큼 | "HSV decomposition: Hue accounts for 93%" |
| fig11_lacn_comparison.png | 10 또는 11 | full vs background(LACN) vs lesion ΔP | "Spatial: 70% of the cue is in the background" |
| fig14_pf1_before_after.png | 12 (Phase 10) | c=70 0.55 -> 0.87 막대 | "After fine-tune: c=70 balanced AUC 0.55 -> 0.87" |
| fig17_fine_tune_trajectory.png | 12 (Phase 10) | Pre 0.80 -> ep1-5 -> Post 0.94 곡선 | "Fine-tune trajectory: +0.135 AUC over 5 epochs" |
| fig13_cam_grid_p6_vs_p10.png | 12 보조(선택) | Phase 6 vs 10 같은 사진 Grad-CAM, attention이 lesion으로 이동 | "Attention shifts onto the lesion after intervention" |
| fig15_pf2_before_after.png | 12 보조(선택) | color cast ΔP +0.10 -> +0.01 | "Color-cast effect removed (ΔP +0.10 -> +0.01)" |
| fig16_pf3_before_after.png | 12 보조(선택) | gap closure | "Cross-collection gap: 0.43 -> 0.09 (79% down)" |
| fig05_cam_grid.png | 8/10 보조(선택) | collection별 Grad-CAM 그리드 | "Grad-CAM: where the model looks" |
| fig09_cf_extreme_pairs.png | 8 보조(선택) | color cast top-3 전후 + Grad-CAM | "Most color-sensitive cases: before/after inpainting" |
| fig_inpaint_qa_grid.png | (선택/백업) | inpainting QA 검증 그리드 | "Inpainting QA: artifact-free removal" |

필수: fig01, fig02, fig03, fig12(or fig10+fig11), fig14, fig17. 나머지는 시간/여백 여유 시 보조.

---

## 6. 주의사항 (반드시 지킬 것)

1. 이모지 절대 사용 금지 (슬라이드, 노트, 어디에도).
2. 주관적/과장 표현 제거: "PhD-level", "publishable", "top-tier", "excellent" 같은 자평 금지. fact 기반 표현만. (수치 자체로 말하게)
3. 팀 framing: 모든 작업을 "we / our team"으로. 누가 무엇 했는지 명시 금지.
4. 슬라이드 텍스트는 영어로 통일. 16:9 (960x540 pt).
5. 비주얼 스타일: 기존 팀원 덱의 색감/폰트/레이아웃 톤 유지. 새 슬라이드도 같은 디자인 언어로.
6. 10분 분량 엄수: 14장 내외. 텍스트 과밀 금지, 슬라이드당 핵심 숫자/피겨 위주.
7. 발표 강조 5개 모먼트가 살게: slide4(0.9513 좋은 결과) -> slide5(per-collection 이상) -> slide7(0.55 반전) -> slide9(0.53->0.72 상승) -> slide12(0.55->0.87 해결). 이 흐름이 dramatic arc.
8. 슬라이드별 발표자 노트(speaker notes) 영어로 첨부.

---

## 7. 제공 자산 체크리스트

문서 4개 (전부 repo 루트):
- HANDOFF_FOR_PPT.md
- HANDOFF_PHASE10_RESULTS.md
- HANDOFF_FOR_PPT_2_narrative.md
- FOR_CLAUDE_DESIGN.md (이 파일)

피겨 14장 (artifacts/phase{9,10,11,12}/figures/) - 전부 존재 확인됨:
- Phase 9: fig01_balanced_auc, fig02_marginal_dp, fig03_gap_decomposition, fig05_cam_grid, fig09_cf_extreme_pairs, fig_inpaint_qa_grid
- Phase 11: fig10_hsv_decomposition, fig12_combined_color_anatomy
- Phase 12: fig11_lacn_comparison
- Phase 10: fig13_cam_grid_p6_vs_p10, fig14_pf1_before_after, fig15_pf2_before_after, fig16_pf3_before_after, fig17_fine_tune_trajectory

참고/선택:
- 팀원 SkinLesionClassifier.pptx (디자인 톤 참고)
- 중간 midterm-deck.pdf (참고)
- 고려대학교 / 정보대학 공식 로고 PNG (타이틀용, 선택)

---

## 8. 그대로 복붙할 프롬프트 (Claude design용)

```
첨부한 네 문서(HANDOFF_FOR_PPT.md, HANDOFF_PHASE10_RESULTS.md, HANDOFF_FOR_PPT_2_narrative.md,
FOR_CLAUDE_DESIGN.md)와 피겨 14장을 바탕으로 딥러닝 학부 term project 기말 최종 발표 PPT(.pptx)를 만들어줘.

핵심:
- 기존 팀원 덱(SkinLesionClassifier.pptx)의 비주얼 스타일을 유지하되, Phase 9-12(색 shortcut 발견 ->
  mechanism -> 해결 -> 검증)를 핵심으로 넣는다.
- 발표 10분 + Q&A 5분 분량. 약 14장으로 압축(20장 넘기지 말 것).
- FOR_CLAUDE_DESIGN.md 섹션 4 슬라이드 구성안(14장)과 섹션 3 storyline(탐정 6단계)을 그대로 따른다.
- 섹션 5의 기본 정보 + Phase 9/10/11/12 실측 수치를 정확히 사용한다(Phase 10은 5/5 supported, placeholder 아님).
- 섹션 5.5 FIGURE MANIFEST대로 피겨를 배치하고 캡션을 단다.
- 섹션 6의 주의사항 전부 준수(이모지 금지, 과장표현 금지, 팀 framing, 영어, 16:9, 10분 분량).
- 발표 반전 포인트(색을 진단신호로 믿고 보존했으나 사실 shortcut이었다는 모순)를 살린다.
- 슬라이드별 영어 speaker notes 첨부.
```

---

문서 끝.
