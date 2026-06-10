# 발표 대본 (한국어 이해용 + 영어 발표용) — 발표자 3: Chen Qixuan

슬라이드 13-15 (위치잡기 + 결론 + 마무리)  ·  목표 ~2분
각 슬라이드: [한국어 - 무슨 말인지] → [English - 실제로 말할 것]
* 한국어는 영어 문장과 같은 내용을 자연스럽게 풀어쓴 것입니다.

---

## [슬라이드 13 — 위치잡기(Positioning): 기여]

[한국어]
감사합니다. 여기서 "우리가 새로 한 게 뭔지"를 분명히 하고 싶어요. 기존 연구들은 함정을 하나씩만 다뤘습니다 — Winkler는 자(ruler), Nauta는 색 패치, 이런 식으로 한 번에 하나씩요. 또 어떤 연구들은 함정을 따로 분리하지 않고 편향을 통째로 없애려 했고, FastDiME은 함정 하나에 대해서만 반사실 분석을 했어요. 저희가 더한 건 이걸 다 묶은 겁니다 — 여러 함정을 하나하나 정량적으로 따지고, 데이터셋 간 격차를 분해하고, 함정이 OOD 성능으로 이어지는 연결을 보고, 그리고 두 가지 고치는 법까지. 핵심은 이거예요 — 저희는 남이 발표한 모델을 그냥 다시 돌려서 정확도만 보고한 게 아닙니다. 우리 모델이 실제로 뭘 학습했는지 측정하고, 원인을 찾아내고, 직접 고쳤습니다.

[English]
Thanks. I want to be clear about what is new here. Prior work has audited single shortcuts — Winkler on the ruler, Nauta on a color patch — one cue at a time. Others remove bias holistically, without isolating the cue, and FastDiME does counterfactuals for a single shortcut. What this study adds is the combination: multi-shortcut, per-cue attribution; a cross-collection gap decomposition; a shortcut-to-OOD link; and two fixes — a validated training fine-tune, and a promising test-time candidate. The point is this — we did not just re-train a published model and report accuracy. We measured what our model actually learned, found the mechanism, and fixed it.

## [슬라이드 14 — 결론]

[한국어]
마무리하겠습니다. 저희는 AUC 0.95를 보고했지만, 거기서 멈추지 않고 그 밑에 숨은 함정을 찾아 원인을 규명하고 두 가지 해결책을 만들었습니다. 한계도 솔직히 말씀드려요 — 증거는 원인 귀속(attribution) 수준이고, 모델 하나와 OOD 데이터 하나에 기반합니다. 그래서 여러 seed, 여러 OOD로 더 검증하는 게 앞으로 할 일입니다. 그리고 그 과정에서 저희 점검이 우리 코드 자체의 재현성 버그까지 잡아냈어요. 오늘 보신 모든 숫자 뒤에는 스크립트, 로그, 커밋이 다 있습니다.

[English]
To conclude. We reported a 0.95 AUC — and instead of stopping there, we found the shortcut beneath it, attributed its mechanism, and built two fixes. We also report the limits honestly: the evidence is attribution-level, on a single checkpoint and one out-of-distribution cohort, so multi-seed and multi-OOD validation are future work. And along the way, our audit even caught a reproducibility bug in our own pipeline. Every number you saw has a script, a log, and a commit behind it.

## [슬라이드 15 — 감사합니다]

[한국어]
이상입니다. 감사합니다 — 질문 받겠습니다.

[English]
That is our work. Thank you — and we are happy to take your questions.

---

전달 팁
- 파트가 짧으니 천천히 또박또박 말해도 됨. 서두를 필요 없음.
- 가장 강한 문장은 "we did not just re-train a published model and report accuracy" — 천천히, 또렷하게. 이게 "그냥 구현한 거 아니냐"는 질문에 대한 답이다.
- 마지막 문장 후, 팀이 함께 앞으로 나가 Q&A.
