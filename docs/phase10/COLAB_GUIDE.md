# Phase 10 Colab 실행 가이드 (한국어)

## 무엇을 하는 거냐

Phase 9 가 *color cast 가 shortcut 이다* 라고 발견했음. Phase 10 은 *그걸 고치는 새 학습 방법* 을 제안 + 검증. 이게 *진짜 새 연구* (audit 가 아닌).

**구체적**: Phase 6 의 학습 transform 에 *HSV jitter* (색 흔들기) 를 추가해서 *fine-tune* (짧은 추가 학습). 새 모델이:

- 기존 AUC 0.95 *유지* 하나? (H6)
- Color cast ΔP +0.10 → +0.05 이하로 *줄어드나*? (H7)
- c=70 의 balanced AUC 0.55 → 0.65 이상으로 *오르나*? (H8)
- Cross-collection gap 0.43 → 0.30 이하로 *줄어드나*? (H9)
- OOD recall 0.79 *유지* 하나? (H10, safety check)

## Colab 에서 할 일 — 총 5단계

### 1단계 — Colab 열기 + A100 활성화

1. https://colab.research.google.com 접속
2. 좌상 *파일* → *노트북 업로드*
3. 로컬 레포의 [scripts/phase10/colab_phase10.ipynb](scripts/phase10/colab_phase10.ipynb) 업로드
4. 메뉴 *런타임* → *런타임 유형 변경* → 하드웨어 가속기 **A100 GPU** 선택 (Colab Pro 필요)
5. *연결*

### 2단계 — best_model.pth 업로드 준비

로컬 laptop 의 `best_model.pth` (18 MB) 가 필요. 위치:

```
/Users/zs_olef.x/Work/Major/DeepLearning/term-proj/melanoma-screening-cnn/best_model.pth
```

Colab 노트북의 **셀 4 (파일 업로드)** 실행 시 *파일 선택* 창 나옴 → 위 파일 선택.

### 3단계 — 모든 셀 순서대로 실행

`런타임` → `모두 실행` 또는 위에서부터 `Shift+Enter`.

**예상 시간** (A100 기준):

| 셀 | 시간 |
|---|---|
| 1. GPU 확인 | 5초 |
| 2. Repo clone | 10초 |
| 3. 의존성 설치 | 1분 |
| 4. best_model.pth 업로드 (수동) | 30초 |
| 5. ISIC 데이터 다운로드 | **30-40분** (가장 오래) |
| 6. Fine-tune 5 epoch | **60-75분** |
| 7. 결과 확인 | 5초 |
| 8. ckpt 다운로드 | 1분 |

**총 ~ 90-120분**

### 4단계 — fine-tune 결과 확인

셀 7 출력 봐:

```
Pre-FT test AUC: 0.9513      ← Phase 6
Post-FT test AUC: 0.940-0.955 ← Phase 10 (예상)
ΔAUC: -0.01 ~ +0.01 사이면 OK (H6 supported)
```

만약 ΔAUC < -0.02 (예: 0.92) → H6 falsified, 우리 가설 *부분 실패*. 그래도 *publishable negative result*.

### 5단계 — ckpt 다운로드

셀 8 자동 다운로드:
- `best_model_color_invariant.pth` (~18 MB)
- `01_finetune.log`

내려받은 파일을 **로컬** 의 `artifacts/phase10/` 에 넣음.

## 그 다음 (로컬에서)

내가 (Claude) 가 *Phase 9 분석을 새 ckpt 로 재실행* 하는 스크립트 만들어둠. 본인이 ckpt 만 받아오면:

```bash
# 로컬에서
mkdir -p artifacts/phase10
mv ~/Downloads/best_model_color_invariant.pth artifacts/phase10/
mv ~/Downloads/01_finetune.log artifacts/phase10/

# Phase 9 재실행 (~30분)
.venv/bin/python scripts/phase10/02_validate_with_phase9.py
```

→ H6-H10 검정 결과 + Phase 10 paper 자동 생성.

## 문제 시

### Colab 끊김
A100 사용 시 12시간 무료. 90분 작업이라 *충분*. 다만 *백그라운드 탭으로 가만 두면 끊김* — 가끔 탭 활성화.

### 디스크 부족
Colab 100 GB 줌. 50 GB 데이터 + 18 MB ckpt = 충분.

### 데이터 다운로드 실패
isic-cli 가 가끔 실패. 셀 5 의 명령어 재실행하면 *이어받기* 됨.

### Fine-tune AUC 가 0.85 이하로 떨어짐
*심각한 catastrophic forgetting*. lr 너무 크거나 augmentation 너무 강함.
- 다시 시도: 셀 6 의 `--lr 1e-5` 를 `--lr 5e-6` 으로 수정
- 그래도 안 되면 augmentation 약화: `finetune_color_invariant.py` 의 `hue=0.05` 를 `hue=0.02` 로

### Out of memory
batch_size 96 → 48 또는 32 로 줄임. 셀 6 의 `--batch-size 48`.

## 보안

- ISIC 다운로드: *공개 데이터, 인증 불필요*
- 모델 weight: 환자 정보 없음 (training 자체에 PHI 없음)
- Colab 작업 완료 후: *런타임 → 런타임 해제* 로 깨끗이 종료

## 비용

- Colab Pro: $9.99/월. A100 priority access
- 이 작업 1회: A100 90분 → 컴퓨팅 unit ~20개 (월 100개 무료 제공 내)

## 다음 작업

본인이 *지금* 해야 할 것:

1. Colab Pro 구독 (없으면)
2. notebook 업로드 + best_model.pth 업로드
3. *모두 실행* → 90분 기다림
4. ckpt 다운 → 나한테 알려줘

내가 Phase 9 재실행 + Phase 10 paper 작성 (사용자는 결과 보고 검토만).

질문 있으면 물어봐.
