# CLI 실행 지시문 — 과거 기반 표현·지연 그래프의 가능성과 실패 위치 진단

## 0. 승인된 작업과 최상위 목표

이 문서는 사용자가 합의한 6개 분석 실험을 실행에 넘기는 지시문이다. 문서 작성만 하고 멈추지 말고, 아래 사전 점검을 통과하면 승인된 계산 예산 안에서 실제 구현·학습·추론·검산·결과 보고까지 수행하라. 전 단계마다 승인을 다시 묻지 않는다. 다만 목표·데이터·비용의 실질적 변경은 자동 진행하지 않는다.

**최상위 목표:** 각 시점까지 관측된 정보로 표현과 그래프를 만드는 방식이 (1) 실제 사용 시점에 계산 가능하고, (2) 지연된 채널의 예측 정보를 남기며, (3) Chronos-Bolt 또는 단순한 예측기와 함께 유용하게 작동하는 조건을 구분한다. 사용자는 결과를 보고 표현 추출만 수정할지, 표현을 재학습할지, 그래프 또는 예측기를 바꿀지 판단한다.

기존 LEVEL을 옹호하거나 개선하는 작업이 아니다. PCA, D8, 잔차 G, 마지막 잔차 유지 경로를 필수 구성으로 되살리지 않는다. 이미 있는 자료·모델·검증 코드는 읽고 재사용하되, 과거 성능을 이번 비교의 결과로 대신하지 않는다.

### 판단할 여섯 질문

1. **E1:** 실제 예측 경계 이후의 관측이 현재 예측에 들어가는가? 과거 표현이 후속 관측에 따라 바뀌는가?
2. **E2:** 자기 채널 과거 외에 다른 채널의 과거가 추가 예측 가치를 주는가?
3. **E3:** 같은 인코더의 추출을 시점별 과거 전용으로 바꾸는 것만으로 예측이 달라지는가?
4. **E4:** 표현·잠재 미래 예측·디코더 중 어디에 문제가 있고, Bolt를 유지할 근거가 있는가?
5. **E5:** 과거 전용 인코더를 학습할 때 재구성·대조학습·미래 예측 중 어떤 목적이 필요한가?
6. **E6:** 고정 지연 그래프로 충분한가? 과거 기반 갱신이 필요한 조건이 있는가?

각 실험을 시작하기 전, 목적 / 단순 대안 / 바꾸는 요인 / 고정하는 조건 / 개선·무차이·악화별 판단 / 중단 조건을 실행 manifest에 기록한다. 어떤 결과도 다음 판단을 바꾸지 않는 작업은 실행하지 말고 이유를 남긴다.

**증거 표기:** 실제 코드·원문·실행 기록에 근거한 내용은 [확인], 원인 해석은 [추정], 아직 실행하지 않은 설계는 [추정·미검증]으로 구분한다. 아래 새로운 수치·모델 크기·분할·계산 상한은 빠른 진단을 위한 합의안과 그 실행상 구체화이지, 최적값이나 검증된 효과 기준이 아니다.

---

## 1. 시작 위치·기존 자산·금지 사항

### 1.1 저장소 확인

- 대상 저장소: `CanelE452/hierarchical-tsfm-peft`.
- 지시문 작성 시 확인한 공개 기준 커밋: `60488e6cb398df1462bcd7ff44221b2c35881061`.
- 실제 실행 PC의 저장소 위치·HEAD·branch·미커밋 변경·실행 중 작업부터 확인한다. Windows 또는 Ubuntu라고 추정해서 강제하지 않는다.
- 원격 또는 로컬에 이후 작업이 있으면 읽고 중복·충돌을 확인한다. 강제 checkout/reset/clean/stash, 자동 pull, 기존 프로세스 강제 종료는 금지한다.
- 현재 장비의 이미 설치된 환경을 사용한다. 공유 환경 업그레이드, 대규모 의존성 변경, 클라우드 결제, 새 대형 모델 다운로드는 승인 범위 밖이다.

### 1.2 실제로 확인된 읽기 출발점

저장소 root 기준으로 다음은 기존 파일이다. 먼저 읽고 실제 로컬 버전과 대조하라.

```text
research/professor_compressor_direct_20261011/README.md
research/professor_compressor_direct_20261011/protocol.json
research/professor_compressor_direct_20261011/runs.json
research/professor_compressor_direct_20261011/public/SOURCE_README.md
research/professor_compressor_direct_20261011/public/DATA_NOTE_KO.md
research/professor_compressor_direct_20261011/public/GT_METHOD_DELTA.md
research/professor_compressor_direct_20261011/public/TS2_METHOD_DELTA.md
research/professor_compressor_direct_20261011/public/METHOD_FAILURE_DIAGNOSIS_KO.md
research/professor_compressor_direct_20261011/data_io.py
research/professor_compressor_direct_20261011/models_gt.py
research/professor_compressor_direct_20261011/ts2_adapter.py
research/professor_compressor_direct_20261011/foundation.py
```

로컬 모델·배열·로그는 Git에 없을 수 있다. 기존 manifest에서 실제 자산 경로를 확인하고 존재·형상·해시를 검증한다. 없으면 `BLOCKED_MISSING_ASSET`으로 표시하고 가능한 독립 실험만 진행한다. 없는 자산을 재현했다고 하거나 이름으로 경로를 만들어내지 않는다. 누락된 정상 크기 TS2Vec를 무단 재학습해 예산을 소모하지 않는다.

[확인·공개 구현] 기존 `data_io.py`의 예측 시작점은 **첫 미래 인덱스**이고, 과거는 `x[o-L:o]`, 미래는 `x[o:o+H]`이다. 기존 loader는 TRAIN 전체 통계와 TRAIN 전체 그래프를 만든다. 따라서 새 FIT/TUNE 분할에서 그대로 호출한 출력을 ‘FIT-only 적합’이라고 부르면 안 된다.

### 1.3 새 산출물 위치 — 아래 이름은 새 제안

```text
research/past_only_repr_diagnosis_20261012/
.cache/past_only_repr_diagnosis_20261012/
```

이미 존재하면 완료 여부를 읽고 검증된 결과를 재사용하거나 새 attempt 하위 경로를 쓴다. 기존 결과를 덮어쓰지 않는다. 새 runner의 모든 쓰기는 새 작업 경로 안에 한정한다. 공유 source를 고칠 필요가 있으면 먼저 새 wrapper/복사본으로 분리한다.

### 1.4 엄격한 금지

- 보호된 TEST 파일을 열기·복사·해시·배열 로드·평가하지 않는다. 지시문 검색/manifest 작성 때문에 TEST 내용을 읽는 것도 금지한다.
- 기존 중단된 LEVEL/A–H 작업을 재개하지 않는다.
- 기존 결과, 기존 실패 로그, 원자료, 모델 가중치를 삭제·정리하지 않는다.
- `git add .`, commit, push, 교수님께 메시지 전송은 이번 승인 범위에 없다. 로컬 산출물을 만들고 변경 목록과 게시 후보만 보고한다.
- 논문·발표 완성본, 다수의 장식용 그림을 만드는 데 시간을 쓰지 않는다.
- 성능이 나쁘다고 데이터·seed·horizon·지표를 바꾸거나 실패 조건을 표에서 빼지 않는다.

---

## 2. 선행 원리와 이번 실험의 위치

구현 전 아래 원문·공식 문서의 필요한 부분만 읽고 사용한 원리와 변경점을 기록한다. 이미 로컬에 출처가 검증된 사본이 있으면 재사용하라. 긴 문헌 수집이나 논문 숫자 채우기는 하지 않는다.

1. **TS2Vec: Towards Universal Representation of Time Series (2022/AAAI)**
   - https://ojs.aaai.org/index.php/AAAI/article/view/20881
   - 문맥 대조학습의 원리. 공식 forecasting의 과거 종료 표현 추출과 이번 E3 전체 문맥/공통 시작점 비교를 구분한다.
2. **Graph WaveNet for Deep Spatial-Temporal Graph Modeling (2019/IJCAI)**
   - https://www.ijcai.org/proceedings/2019/264
   - 그래프 관계와 과거 방향 시간 연산을 결합한 선행. 이번 간단한 고정/갱신 지연 그래프를 원논문 재현이나 물리 인과 발견으로 부르지 않는다.
3. **Time-Series Representation Learning via Temporal and Contextual Contrasting (2021/IJCAI)**
   - https://www.ijcai.org/proceedings/2021/324
   - 미래와 연결된 표현학습의 선행. E5의 작은 공통 인코더 목적 비교는 이 논문의 전체 방법 재현이 아니다.
4. **Graph-Time Convolutional Autoencoders (2022/LoG)**
   - https://proceedings.mlr.press/v198/sabbaqi22a.html
   - 기존 재구현의 시간 압축·대표 시점·디코더 위상 의미를 확인하는 근거.
5. **statsmodels: Vector Autoregressions**
   - https://www.statsmodels.org/stable/vector_ar.html
   - E2·합성 자료·지연 선형 회귀의 표준 출발점. multi-horizon direct ridge를 그대로 VAR 원형이라고 부르지 않는다.
6. **scikit-learn: TimeSeriesSplit / Common pitfalls / Ridge**
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
   - https://scikit-learn.org/stable/common_pitfalls.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
   - 시간순 평가·학습되는 전처리의 분리·회귀 규제 근거.

과거 전용 연산 자체를 신규성으로 주장하지 않는다. 확인하려는 추가 가치는 ‘표현의 시점별 정의를 일관되게 했을 때 예측이 달라지는가’, ‘학습 목표/그래프/예측기 중 무엇이 필요한가’이다.

---

## 3. 공통 시간·데이터 계약

### 3.1 시간 좌표를 먼저 고정

새 지시문에서 `o`는 **첫 미관측 미래 인덱스**로 통일한다.

```text
history = X[o-512:o]        # o-1까지 관측, 정확히 512시점
future  = X[o:o+48]         # 맞힐 대상 48시점
L = 512, H = 48
```

가장 최근 관측 시점을 t라고 쓰면 t=o-1이다. 코드와 설명의 off-by-one을 검사한다.

- 예측 함수는 원칙적으로 과거 배열과 이미 고정된 학습 산출물만 받는다.
- 평가 정답은 별도 scorer 또는 진단 전용 함수로 분리한다.
- 학습용 미래 정답을 손실로 사용하는 것과 미래 정답을 예측 입력에 넣는 것을 구분한다.
- 합성 미래 잠재값/실제 미래 잠재값 진단은 별도 네임스페이스·파일로 격리한다.

### 3.2 실자료는 Jena·Robin만

현재 데이터 계약에서 원자료·채널 순서·단위·간격·TRAIN/VAL 경계·관측 마스크를 읽고 기록한다. 별도의 Weather 자료를 추가하지 않는다. Jena는 기상 자료다.

두 자료를 각각 보고하고 평균 하나로 서로의 실패를 가리지 않는다. 비교 단위마다 동일 시작점, 미래 정답, 관측 마스크, 원공간 평가 스케일을 보장한다.

### 3.3 새 학습과 저장 인코더 진단의 독립성을 구분

**NEW_FIT 프로토콜(E2·E5·E6의 새 학습):**

- 기존 TRAIN 구간을 시간순 앞 80% FIT / 뒤 20% TUNE으로 나눈다. 실제 경계는 사전에 저장한다.
- 표준화·초기 결측 대체 통계·고정 그래프·학습 가중치는 FIT에서만 적합한다.
- 모든 FIT supervised target은 FIT 안에서 끝나야 한다. TUNE target은 TUNE 안에 있어야 한다.
- TUNE의 이미 알려진 과거 문맥에 FIT가 포함되는 것은 허용한다. 정답 구간이 경계를 넘는 창은 제외한다.
- 기존 VAL은 반복 노출된 최종 개발 진단 자료다. 모델 선택은 TUNE으로 하고 VAL에서 재선택하지 않는다.
- 원자료가 이전 전처리를 거쳤다면 실제 관측 mask와 원래 mean/std로 물리값을 복원 가능한지 확인한다. 불가능하면 전처리 독립성을 주장하지 말고 해당 비교를 보류한다.
- 이번에는 FIT+TUNE으로 사후 refit하지 않는다. 선택된 FIT 학습 모델을 그대로 VAL에 적용한다.

**FROZEN_ENCODER_DIAGNOSTIC(E1·E3·E4):**

- 기존 인코더는 이미 TRAIN 전체와 과거 VAL 선택에 노출됐을 수 있다. 이를 숨기지 않는다.
- 같은 인코더 비교에는 원래 checkpoint에 맞는 입력 전처리를 유지한다. FIT-only 통계로 입력을 무단 변경해 인코더 분포를 깨뜨리지 않는다.
- 새 decoder/head의 계수와 새 feature scaler는 FIT에서 적합하고 TUNE에서 선택한다.
- TUNE을 인코더까지 독립적인 backtest라고 부르지 않는다. frozen representation에 조건부인 개발 진단이다.
- 과거 학습 시점의 표현은 그때 아직 학습되지 않았던 모델로 계산할 수 없었다. 이번 frozen 진단을 역사적 온라인 배포 재현으로 해석하지 않는다.

### 3.4 시작점·계산량 선택

- 빠른 supervised 비교는 같은 규칙으로 최대 FIT 256 / TUNE 64개의 legal origins를 사용한다.
- 시간순 legal origins에서 균등한 인덱스 선택으로 뽑고, 성능/target 값으로 고르지 않는다. 부족하면 legal 전체를 사용한다.
- 기존의 ‘직접 릿지 stride1 전체’ 대 ‘F-decoder stride128 소수’ 비교를 반복하지 않는다.
- 기본 VAL은 기존 legal 전체다. 다만 E3의 시점별 prefix export가 과도하게 비싸면 §4의 사전 비용 규칙에 따라 작은 공통 진단 묶음을 고른다. 범위를 바꾸면 `FULL_VAL`과 `DIAG_SUBSET`을 명확히 나눈다.
- 독립 표본 수는 origin 수·horizon 행 수와 같지 않다. unique timestamp 수, target 중복 횟수와 구간별 표본 수를 보고한다.

### 3.5 회귀·지표 공통 규칙

[추정·운영값] 새 ridge 계열은 feature를 FIT에서 규격화하고, 출력별 **평균 제곱오차 + λ×계수 제곱합**, 벌점 없는 intercept를 공통 계약으로 한다. λ 후보는 `[1e-4, 1e-2, 1, 1e2, 1e4]` 한 번만 사용한다. 기존 API의 sum SSE 목적을 쓰면 유효 관측 수에 맞춰 변환하고 실제 정의를 기록한다. TUNE macro MSE 최소, 동률은 선언된 grid 순서로 선택한다.

- 관측 원공간 MSE·MAE, 채널 macro 평균, seed별 값, origin별 대응 차이를 저장한다.
- horizon은 전체 1~48과 1~8 / 9~24 / 25~48을 함께 보고한다. 주 지표는 전체 MSE, MAE는 필수 동반 지표다.
- 기상·전력의 물리 단위를 섞어 평균하지 않는다.
- 잠재 오차는 같은 E·같은 export 안에서만 직접 비교한다. 서로 다른 E의 latent MSE로 표현 순위를 매기지 않는다.
- 신뢰구간이나 p값을 억지로 만들지 않는다. 두 seed·겹치는 창을 독립 표본으로 t-검정하지 않는다.

---

## 4. 계산 예산과 실행 순서

### 4.1 승인된 상한

- 1차 E1~E4: **GPU 사용 작업의 경과시간 합계 최대 90분**.
- 2차 E5~E6 및 조건부 작은 LSTM: **추가 최대 90분**.
- 합계 최대 180분. 실패·재시도·GPU smoke·표현 추출·예측·검산·benchmark도 해당 작업 시간에 포함한다.
- GPU 독립 작업은 동시 1개. [추정·보수적 실행 상한] GPU 작업 시간에 포함되지 않는 CPU-only 수치 계산·graph 추정·합성 진단은 누적 wall 60분에서 중단한다. 긴 graph 반복 적합으로 GPU 예산을 우회하지 않는다. 구현·읽기 시간은 이 계산 상한과 구분한다. CPU 작업/전처리 시간은 별도로 보고한다. wall·GPU 사용 작업 wall·optimizer steps를 서로 혼동하지 않는다.
- 이는 완료 보장이 아니라 중단 상한이다. 새 장비 대여·대형 파운데이션 모델 학습은 하지 않는다.

RAM·VRAM·디스크는 현재 가용량을 관찰하고 기존의 더 엄격한 보호 한도를 유지한다. 출력 배열 크기를 미리 계산하고 큰 all-window/full-prefix tensor를 통째로 메모리에 만들지 않는다. 학습 배치 감소는 비교 묶음 전체에 사전에 적용한다. 실제 batch/누적 batch/업데이트 횟수를 모두 기록한다.

### 4.2 성능을 보기 전 비용으로만 범위 선택

작은 고정 입력의 순전파로 per-origin·per-step 시간을 측정한다. 실제 신경망 업데이트를 했다면 smoke도 계산 원장에 넣고 science 모델은 원래 seed에서 새로 시작한다.

E3/E4의 기본은 FIT256/TUNE64/전체VAL, d16·d320·두 seed·두 자료다. 예산을 초과할 전망이면 성능을 보지 않은 상태에서 먼저 다음 축소 묶음을 선택할 수 있다.

```text
FIT64 / TUNE32 / VAL의 시간순 균등 최대32개
```

이 경우 해당 실험은 `DIAG_SUBSET`으로만 보고하고 기존 full VAL 성능과 직접 비교하지 않는다. 작은 묶음도 예산에 안 맞으면 우선 Jena의 d16 두 seed와 모든 필요한 대조를 완성하고 나머지를 `BUDGET_HELD`로 남긴다. d320·Robin을 하지 않았으면 성능을 추정하지 않는다.

- 각 묶음 시작 전에 예상 비용과 실제 잔여 예산을 확인한다.
- 하나의 seed/한쪽 대조만 완료한 채 균형 비교로 선언하지 않는다.
- 벽시계 상한에 닿은 모델을 수렴한 최종 결과로 쓰지 않는다.
- 다음 과학적 선택지나 지표를 성능을 본 뒤 추가하지 않는다.

### 4.3 진행 경로

1. 경로·계약·자산 점검 → synthetic unit test → E1.
2. E2로 지연 정보 가치를 보고 E3·E4의 동일 조건 진단을 수행한다.
3. E3에서 학습-추출 불일치가 남거나 E4에서 표현/목적 문제를 배제하지 못하면 E5.
4. E2에서 지연 신호를 포착한 자료 또는 관계 변화 합성 조건에 한해 E6. 실자료에서 신호가 없으면 합성 가능성 확인과 실자료 후순위를 구분한다.
5. 각 gate 판정은 결과를 보고 임의로 새로 정하지 말고 아래 결정 규칙을 따른다. 사용자의 아이디어가 실패했다고 성급하게 종료하지 않는다.

---

## 5. E1 — 전체 정보 경로·표현의 시간 일관성 검사

### 목적과 고정 조건

학습 없이, 가중치·전처리 계약·정적 그래프를 고정하고 예측 경계 누출과 과거 표현의 사후 변경을 분리한다. 검사할 시점은 각 자료의 legal VAL에서 미리 균등 선정한 최대16개다.

### E1-A: 예측 경계 뒤 실제 관측 교란

`X[:o]`를 동일하게 두고, 메모리의 허용 TRAIN/VAL 기록에서 `X[o:]`와 해당 mask를 고정·잡음·순열 등으로 바꾼다. TEST는 쓰지 않는다. 현재 예측은 실제 배포 pipeline을 입력 준비부터 다시 호출한다.

검사할 것은 raw preprocessing → input normalization → missing fill → graph → representation → forecast 전체다. 모델에 이미 잘라진 history만 주는 단일 검사로 전처리 안전성을 대신하지 않는다.

- fit되는 preprocessing/정적 graph는 FIT 또는 기존 TRAIN cutoff 밖을 바꿔도 동일해야 한다.
- 무작위 입력은 현 자료 척도로 정하고 반복 seed를 고정한다.
- 예측과 현재 representation·사용 graph의 절대/상대 차이, hash, 호출한 파일 경로를 기록한다.
- dropout/eval 모드를 통일한다. 비교 시 seed도 고정한다.
- 검사 중 가중치 업데이트는 하지 않는다. 기본 검사는 inference 결과 캐시를 우회한 cold 경로로 수행하고, 별도 warm 경로와 비교한다. 원래 결과를 캐시에서 그대로 돌려받아 미래 교란이 무효화된 검사를 통과시키지 않는다.

### E1-B: 과거 내부의 뒤쪽 교란

고정 history 시작점 a와 내부 시점 s<o-1을 정한다. `X[a:s+1]`은 유지하고 `X[s+1:o]`만 바꾼다. `z_s`의 변화를 검사한다.

변화가 있어도 그 자체가 o 이후 정답 누출은 아니다. `WITHIN_HISTORY_LOOKAHEAD` 또는 `REPRESENTATION_REVISION`으로 구분한다.

### E1-C: append/prefix 불변성과 창 이동 분리

- 왼쪽 시작점 a를 고정한 상태에서 관측 suffix를 추가한다. 이미 계산한 z_s가 변하는지 검사한다.
- 그다음 별도 검사로 왼쪽 시작점을 옮겨 오래된 문맥이 사라지는 효과를 측정한다.
- 서로 다른 왼쪽 문맥이나 time pooling phase를 가진 값을 같아야 한다고 검사하지 않는다.
- GTConvAE의 잠재 tick은 4개 원시점 묶음의 오른쪽 끝이 대표 시점이다. 묶음 중간을 대표 시점으로 취급하거나 pooling phase를 이동시켜 허위 누출을 만들지 않는다.
- 동적 graph는 G_s도 당시까지의 관측으로 계산해야 한다. G_o를 과거 모든 z_s에 소급 적용해 과거 전용이라고 부르지 않는다.

### 검사기 자체 검증·판정

합성 자료에서 (a) 미래를 일부러 읽는 잘못된 함수/미래 전체로 fit한 scaler·graph는 검출하고, (b) 단순 과거 전용 함수는 통과해야 한다.

수치 허용치: 같은 입력 반복 오차·dtype·CPU reference 대조로 먼저 정하고 protocol에 봉인한다. 편의상 큰 atol을 올려 통과시키지 않는다. 실패는 최대 오차·영향받은 값 수·원시 예시를 남긴다.

- A 실패: 해당 경로의 성능 결론 중단. 새 wrapper에서 국소 수정 → 검사 재통과 → 새 attempt. 원본은 보존.
- A 통과, B/C 변화: 누출이 아니라 과거 표현 정의 문제. E3로 진행.
- 모두 통과: 검사한 범위에서 시간 정보 계약 만족. 효과/정확도 판정은 미완료.

---

## 6. E2 — 다른 채널의 과거가 필요한가?

### 비교 모델

같은 origin·target·규제 선택으로 다음 direct multi-horizon ridge를 비교한다. lag는 관측 마지막 시점 기준 0~32다.

1. `SELF_LAG`: 출력 채널 자신의 현재와 과거.
2. `SELF_LAG_PLUS_CROSS_NOW`: 자기 과거 + 나머지 채널의 현재.
3. `ALL_CHANNEL_LAG`: 모든 채널의 현재와 과거.
4. 학습 없는 persistence(채널 마지막 관측 유지)를 공통 참고 기준선으로 둔다.

출력은 미래 48×C다. 다른 모델에 미래 실제 채널값을 설명변수로 제공하지 않는다. 차원 증가·계수 수·학습 표본·λ 효과를 함께 보고한다. 한 λ의 우연한 실패로 관계가 없다고 단정하지 않는다.

### 합성 검사 — 새 설계, 검증된 실자료가 아님

표준 지연 선형 생성 모델에 기반한 구현 시험대다. 채널6, 유효 길이4096, burn-in512, 조건별 고정된10개 seed의 독립 잡음 기록을 만든다. 생성은 CPU만 사용한다.

기본 조건의 설명용 생성식:

```text
u_j[t] = 0.3*u_j[t-1] + epsilon_j[t],                   j=1,2,3
v_j[t] = 0.3*v_j[t-1] + 0.6*u_j[t-lag_j] + eta_j[t]
lag = (4, 8, 16); 독립 표준정규 잡음; source→receiver 단방향
```

- 무관계 조건: cross coefficient를 0으로 둔다.
- 고정 관계 조건: 위 구조 유지.
- 변화 조건: FIT/TUNE에서는 위 구조, 사전 고정된 평가 구간 중간에서 source 대응과 lag를 바꾼다. 결과를 보고 변화 위치를 고르지 않는다.
- 생성 계수·seed·분할·정확한 변화 timestamp를 저장한다. 안정성/유한값과 알려진 source→receiver 방향을 unit test한다.
- 합성 분할은 전체 시간의 앞60% FIT / 다음20% TUNE / 마지막20% 진단으로 하고, 입력·target 경계와 burn-in 제외를 명시한다. 실제 change는 마지막 평가 구간 중간 한 지점이다.
- 두 graph regime 모두 자기회귀 계수는 그대로 두고 feedback cycle을 추가하지 않는다. 무관계 조건과 marginal 분산이 완전히 같다고 주장하지 않는다.
- 생성식상 실제 과거에서 알 수 있는 신호는 지연별 가까운 horizon에 집중된다. 48 전체 평균만 보지 않고 receiver별·horizon별로 읽는다.
- 잡음 없는 정렬 unit test와 잡음 있는 예측 비교를 구분한다. 신호를 심었으므로 성공을 신규성/실자료 효과로 주장하지 않는다.

### 결과와 gate

원공간 MSE/MAE 및 고정4개 연속 평가 구간의 대응 차이를 기록한다. 단일 record/unit test를 통계적 확증으로 부르지 않는다.

E6 실자료 진행용 `LAG_SIGNAL`은 사전 정한 주 전체 MSE에서 ALL_CHANNEL_LAG가 SELF_LAG보다 낮고, 네 구간 중 적어도 세 구간에서 같은 방향일 때로 둔다. 이것은 **계산 분기용 임시 규칙**이지 유의성/효과의 합격선이 아니다. MAE가 나빠지면 `LAG_SIGNAL_MSE_ONLY`로 표시한다. 전체에서 신호가 없더라도 사전 horizon 구간의 차이를 함께 기록하며 ‘선형·32lag 범위 미확보’로 한정한다.

- 알려진 noiseless 지연도 계산하지 못하면 구현 검증 실패로 관련 후속을 차단한다.
- 합성만 성공하면 ‘가능성 확인, 실자료 추가 가치 미확보’이다.
- 실자료에서 lag signal이 있으면 E6 대상에 포함한다. 무관계 조건에서도 복잡한 모델의 자동 우위가 나오도록 자료를 고치지 않는다.

---

## 7. E3 — 동일 인코더의 전체 문맥/과거 전용 추출 비교

### 비교의 고정 조건

현재 저장된 정상 크기 TS2Vec d16·d320 두 seed를 우선 사용한다. 같은 E 가중치·같은 입력 전처리·같은 history `[a:o]`, a=o-512를 유지한다. d320을 압축 성공이라고 부르지 않는다.

```text
FULL:   z_s = E(X[a:o])[s-a]
PREFIX: z_s = E(X[a:s+1])[-1],    a <= s < o
```

PREFIX가 예전 공식 last201 export보다 더 많은/다른 문맥을 쓰는 점을 기록한다. FULL512와 PREFIX201을 이번 동일 추출 대조로 잘못 비교하지 않는다.

### 구현·성능 최적화 규칙

- 모든 prefix의 거대한 padded tensor를 한 번에 만들지 않는다. chunk·정확한 길이별 묶음·disk cache를 사용한다.
- 가변 길이 padding이 양방향 합성곱의 hidden activation을 바꿀 수 있다. 단순 오른쪽 zero/NaN padding이 실제 짧은 prefix 실행과 같은지 반드시 대조한다.
- 최적화 구현을 실제 단일 prefix reference와 경계·중간·마지막 위치에서 검증한다. 다르면 느리더라도 정답 경로를 쓰거나 예산 미완료로 남긴다.
- FULL의 마지막 표현과 PREFIX의 마지막 표현은 같은 입력일 때 일치해야 한다. 이는 중요한 negative control이다. 마지막 벡터만 쓰는 head에서는 차이가 원래 없을 수 있다.
- 비교할 새 D는 같은 구조 `d→128→ReLU→C`, 같은 초기화·FIT origins·업데이트 수로 각 export에 별도 학습한다. E는 고정한다.
- 기존 FULL 전용 D를 PREFIX에 붙인 결과는 입력 분포 불일치 진단으로만 남기고 주 비교로 쓰지 않는다.
- [추정·운영값] D는 Adam 1e-3, 최대600 update, 공통 batch(예산/메모리 preflight로 결과 전 고정). epoch0 및 100 update 간격 TUNE reconstruction MSE로 선택한다. 양쪽 모두 같은 상한을 사용하고 최선이 상한이면 명시한다.
- F는 고정된 Chronos-Bolt-small, 입력은 `[B*d,512]`, 미래48, 공식 median 출력 계약을 유지한다. 가중치·양자화·미래 길이를 바꾸지 않는다.

### 무엇을 측정할지

1. 같은 원시점에서의 FULL/PREFIX 표현 차이 및 append revision.
2. 두 export 각각의 입력 재구성 MSE/MAE.
3. 각 export→같은 F→각 export에 맞춰 학습한 D의 미래 MSE/MAE.
4. 초반 문맥이 부족한 위치와 나머지 위치를 사전 구간별로 보고하되 전체를 주 결과로 유지.
5. 표현 추출 및 전체 실제 경로 비용.

`z_{s|a}`는 왼쪽 시작점 a에 조건부인 표현이다. 창 시작점이 달라진 두 표현을 자동으로 같은 시계열 값으로 취급하지 않는다. 이번 E3는 같은 a의 정보 제한 비교이지 영구적인 streaming state의 효율을 입증하는 실험은 아니다.

### 해석

- PREFIX가 좋아지면 같은 E에서 추출 규칙을 바꿀 가치가 있다.
- PREFIX가 나빠져도 causal 학습 전체를 기각하지 않는다. E의 학습 문맥과 export 문맥 차이가 남는다.
- FULL/PREFIX 차이가 작으면 이 E에서 추출 규칙이 주원인이라는 주장은 약해진다.
- E3가 budget-held이면 없던 결과를 추정하지 않는다. E5의 작은 공통 인코더는 별도 질문으로 진행 가능하다.

---

## 8. E4 — 같은 표현에서 잠재 예측기·복원 문제 분리

### 주 대상

E3의 PREFIX 인코더와 새 reconstruction D를 고정한다. 비용상 Jena d16 두 seed를 먼저 완성하고, d320·Robin은 사전 범위/예산에 따라 동일 대조를 완료한다. 기존 GT checkpoint는 E1과 참조값 치환 진단에서 재사용 가능하나, 모든 predictor×family×dimension의 전조합을 의무적으로 늘리지 않는다.

### 고정 D에 넣을 잠재 미래

1. `LATENT_PERSISTENCE`: 마지막 잠재값 유지.
2. `LATENT_SELF_RIDGE`: 잠재 채널 자신의 lag0..32에서 미래48을 직접 예측.
3. `LATENT_JOINT_RIDGE`: 모든 잠재 채널의 lag0..32에서 미래48을 직접 예측.
4. `BOLT_LATENT`: 고정 Bolt가 과거512 잠재 시점에서 미래48 예측.
5. `REFERENCE_LATENT_DIAGNOSTIC`: 실제 미래 관측에서 계산한 잠재값. 배포 경로 아님.
6. `DIRECT_RAW_RIDGE`: 같은 과거 잠재 lag0..32에서 미래 원채널을 직접 예측. D를 거치지 않는 전체 대안.

Direct raw와 latent ridge는 같은 lag 입력·FIT/TUNE origins·정답 horizon·선택 규칙을 공유한다. Bolt는 더 긴 잠재512 문맥과 사전학습을 사용한다는 정보/학습 차이를 명시한다. 모든 경로를 ‘TSFM 유무만 다른 순수 인과 비교’라고 부르지 않는다.

d320 joint ridge는 33×320 특징과 48×320 출력을 가질 수 있으므로 parameter matrix를 무작정 만들지 않는다. dual ridge 등 동치인 계산으로 메모리를 제한하고 작은 배열에서 primal/dual 일치를 검증한다. 계산 방식 변경을 추가 모델 효과로 부르지 않는다.

### 잠재 정답의 정의와 참조 진단 격리

- a=o-512를 고정하고, 실제 미래 잠재 z_s는 `E(X[a:s+1])[-1]`로 계산한다. 과거와 미래를 서로 다른 왼쪽 문맥으로 만들지 않는다.
- reference latent는 **실제 미래를 사용하는 진단 전용 값**이다. 예측 입력·graph·P/D calibration에 평가 정답이 들어가지 않게 별도 함수/파일에 격리한다.
- TRAIN 잠재 정답을 P의 감독으로 만드는 것은 허용하되 모든 그 원시점은 FIT 경계 안에 있어야 한다. TUNE 잠재 정답은 선택/채점에만 쓴다.
- 실제 미래를 인코딩할 때 기존 z_history를 새로 FULL-context 재인코딩해 바꾸지 않는다.
- GT 진단이면 4점 묶음의 대표 시점·global phase·동일 native decoder history를 지킨다.
- reference 결과는 수학적으로 보장된 최저 오차/상한이 아니다. 실제 잠재값 치환 시 현재 D의 동작을 보는 진단이다.

### 측정과 판단

- 같은 E 안에서 잠재 예측 오차 및 원공간 미래 MSE/MAE.
- 동일 D 고정 비교를 주 결과로 쓴다. predictor별 D 추가 학습을 하려면 모든 비교 경로에 동일 감독·예산을 주는 별도 블록으로 분리하고, 기본 블록 완료 뒤 남은 예산에서만 시행한다.
- 참조 잠재값에서도 복원이 나쁘면 E/D 조합의 표현·복원 일반화 문제를 남긴다. E와 D 중 하나로 단정하지 않는다.
- 참조 복원은 양호하지만 predicted latent는 나쁘면 잠재 예측 문제가 후보가 된다.
- joint ridge가 self ridge/Bolt보다 유리하면 잠재 채널 공동 예측의 가치를 의심할 근거가 생긴다. 모델 크기·감독 유무 차이도 보고한다.
- direct raw가 좋은데 잠재 예측→D가 모두 나쁘면 ‘표현 자체가 쓸모없다’보다 연결 목표의 불일치 가능성을 남긴다.
- reconstruction과 future의 다른 표본 MSE를 빼서 Bolt 책임이라고 계산하지 않는다.

### 조건부 작은 LSTM과 TiRex-2 범위

참조 복원이 가능하고 공동 예측이 자기 채널보다 유리한 근거가 있으며 잔여 2차 예산이 있을 때만 작은 단방향 LSTM 한 설정×두 seed를 허용한다. latent 입력·FIT/TUNE·horizon·고정 D를 맞춘다. 모델 크기/최적화 설정은 결과 전 기록하고 추가 sweep하지 않는다. 미실행이면 필요성은 보류로 남긴다.

TiRex-2 새 다운로드·대형 비교는 이번 기본 실행에 없다. 필요 조건과 예상 비교의 차이만 보고한다. causal 표현이 된다는 이유만으로 예측기를 반드시 교체하지 않는다.

---

## 9. E5 — 과거 전용으로 처음부터 학습할 때 목표가 무엇이어야 하나?

### 실행 gate

E1 안전성 이후, E3에서 학습/추출 불일치가 남거나 E4에서 표현 학습 목적의 영향을 배제하지 못할 때 실행한다. 기존 인코더 자산이 없어 E3가 막혔어도 이 작은 공통 모델 실험은 가능하다. 새로운 대규모 TS2Vec/GTConvAE full 재학습으로 확장하지 않는다.

### 공통 인코더와 네 조건 — 신규 교육용 진단 구현

[추정·운영값] 작은 길이 보존 시간 합성곱 인코더, hidden32, 출력d16, kernel3·dilation1/2/4/8/16의 다섯 층을 출발 설정으로 한다. 실제 receptive field·parameter 수를 코드에서 계산하라. output은 항상 `[B,L,16]`이다. 새로운 모델 이름을 TS2Vec/GTConvAE라고 붙이지 않는다.

| 조건 | 시간 연산 | 학습 목표 |
|---|---|---|
| FULL_REC | 입력 창 안의 양방향 | 재구성 |
| CAUSAL_REC | 각 시점까지 | 재구성 |
| CAUSAL_CONTRAST | 각 시점까지 | 대응 표현 대조 |
| CAUSAL_FORECAST | 각 시점까지 | 미래 원변수 예측 |

- 첫 두 행은 같은 kernel·층수·parameter 수, padding/정보 방향만 다르게 한다. 총 receptive-field 크기는 맞아도 과거/미래 배분은 달라진다는 점을 기록한다.
- causal 경로에서 시간 전체 평균을 쓰는 normalization, 양방향 보간, 대칭 padding이 숨어 있지 않아야 한다. normalization은 timestamp별 channel 방향 또는 FIT 고정 통계를 쓴다.
- 같은 seed마다 공통 E 파라미터의 초기 tensor를 동일하게 복사하고 배치 순서도 최대한 맞춘다. objective별 추가 head와 FLOPs 차이는 기록한다.
- Adam1e-3, 최종600 update 고정. 같은 묶음에서 상한 도달과 실제 sample/timestep/미래 정답 노출량을 각각 기록한다.
- GPU 메모리에 맞는 실제 batch를 성능 전에 묶음 공통으로 정한다. 업데이트 수가 같다고 연산·정답 노출까지 같다고 주장하지 않는다.
- 학습 초깃값·100 update 간격·최종 곡선을 남긴다. 서로 다른 학습 손실의 절댓값으로 목적 우열을 비교하지 않는다.

### 대조학습의 구체적인 계약

표준 NT-Xent/InfoNCE 계열의 대칭 대조 손실을 사용한다. 이것은 신규 손실 제안이 아니다. positive는 같은 원시점의 서로 다른 마스킹 관측이다. 양쪽 representation은 해당 시점 이후 입력을 보지 않는다.

[추정·운영값] 관측 입력의 10% 마스킹, 작은 projection head, L2-normalized projection, temperature0.2를 출발값으로 봉인한다. 학습 효과가 확인된 최적값은 아니다. 진폭 재조정·시점 순열·time warping은 이번에 넣지 않는다.

- positive 쌍의 원시 timestamp와 정보 끝시점을 실제로 기록한다.
- 같은 원시점·겹치는 중복 관측을 negative로 취급하지 않는다. 가까운 temporal negative의 제외 간격은 encoder의 실제 receptive field를 기준으로 정한다.
- instance contrast를 포함할 때는 실제 batch의 독립 사례 수≥2를 검사한다. 구현이 B1에서 해당 항을0으로 만드는 경우 그대로 통과시키지 않는다.
- Robin처럼 사례 수가 적으면 학습 구간 안에서 겹치지 않는 사례를 구성하거나, 사전에 temporal 대조만의 별도 조건으로 명시한다. 후자는 두 항을 썼다고 보고하지 않는다. 대조 항을 바꾸는 경우 결과를 보기 전 objective manifest에 확정한다.
- source에서 충분한 쌍을 못 만들면 이 조건은 `BLOCKED_PAIR_CONSTRUCTION`이다. 무조건 실행 개수를 채우지 않는다.

### 재구성/예측 감독과 공통 평가

- REC는 입력 원본을 observed MSE로 재구성한다. causal D를 사용해 causal E를 우회하는 미래 정보 경로를 만들지 않는다.
- FORECAST의 target은 FIT의 실제 미래48 원채널이다. 원래 입력에는 미래를 넣지 않는다.
- 학습 후 모든 E를 고정하고 학습 당시 붙인 head를 제외한 **같은 새 direct ridge readout**을 각각 학습한다. 같은 FIT/TUNE·input lag·target·선택 규칙을 사용한다.
- 같은 reconstruction probe D도 붙여 예측과 복원을 구분한다. probe의 실제 학습 횟수·비용을 포함한다.
- 파운데이션 모델 연결은 최소한 CAUSAL_REC와 사전 gate를 통과한 목적의 비교에만 추가한다. 하나만 Bolt에 붙여 모든 목적의 Bolt 호환성을 결론내리지 않는다.
- 표현 분산, 유효 rank, 상수 붕괴 여부는 기술 통계로 보고하며 임의 rank 점수를 합격선으로 쓰지 않는다.

### 범위와 판정

우선 Jena 네 조건×두 seed의 균형 묶음을 완성한다. Robin 확장은 남은 예산에서 네 조건 묶음 단위로 수행한다. 한 조건만 좋은 데이터에서 추가학습하지 않는다.

- FORECAST 우세: 예측 목적 학습의 후보 가치. 비지도만의 이득으로 표시하지 않는다.
- CONTRAST 우세: 현재 통제 범위에서 대조 목적의 가치.
- REC와 차이 없음: 더 단순한 목적이 충분할 수 있음.
- 모두 불리/불안정: 이 예산·구조에서 가치 미확보. causal 표현의 일반 불가능으로 확대하지 않음.

---

## 10. E6 — 고정/과거 갱신 지연 그래프 비교

### gate와 독립 목표

E2의 알려진 지연 합성 검사부터 통과해야 한다. 합성의 고정·변화 조건에서 가능성을 보고, `LAG_SIGNAL`이 있는 실자료로 확장한다. 신호가 없는 자료는 보류하되 이유를 적는다. 그래프 그림만 만들어 성공이라고 하지 않는다.

### 그래프의 의미

G는 물리적 인과 그래프가 아니라 **선형 지연 예측 관계를 요약한 계산 구조**이다. 동시점 절대 상관만으로 directed lag를 발견했다고 주장하지 않는다.

[추정·운영 정의] 각 도착 채널 x_j[v]를 모든 채널의 v-1..v-32로 예측하는 ridge 계수에서 source·destination·lag 관계를 얻는다. 전처리·규제 hyperparameter는 FIT/TUNE에서 고정하고 평가 중 재선택하지 않는다. signed 계수/정규화/edge 선택 규칙을 protocol에 명시한다. 이것은 지연 회귀 기반 prototype이며 Graph WaveNet/GTConvAE 원형 재현이 아니다.

기본은 도착 채널당 다른 채널의 최대2개 source-lag 항을 |계수| 순서로 남기고, 이웃 계수 절댓값 합이 1 이하가 되도록 정규화한다. 자기 과거 경로는 별도 공통 분기다. sparsity2는 빠른 진단용 고정값이지 확인된 topology가 아니다.

### 네 조건

1. `SELF_ONLY`: 다른 채널에서의 메시지 없음.
2. `FIXED_LAG_GRAPH`: FIT에서 만든 그래프 고정.
3. `PAST_UPDATED_LAG_GRAPH`: 과거 관측만으로 갱신.
4. `WRONG_EDGE_GRAPH`: 이웃 수·lag별 가중치 규모를 맞추고 source 연결을 미리 고정한 순열로 바꾼 대조.

WRONG_EDGE의 순열 뒤에도 self-edge 제외, 도착 노드별 edge 수, lag와 가중치 규모가 계약대로 유지되는지 검사한다. 학습부터 각 조건의 graph를 사용해 별도로 모델을 적합한다. 정상 graph로 학습하고 추론 때만 파괴한 결과를 주 비교로 삼지 않는다. 음성 대조는 graph가 있다는 사실보다 올바른 연결이 필요한지 묻는 것이다.

### 시간·정보 예산 계약

- [추정·운영값] graph 추정 최대 과거512시점, 32시점마다 갱신.
- E1처럼 각 과거 표현 z_s에 들어가는 graph는 G_s이고, G_o로 과거를 소급 재작성하지 않는다.
- 평가 미래48 동안 실제 미래 graph를 사용하지 않는다. 기본은 예측 origin에서 확정한 graph를 유지한다. 예측값으로 graph를 다시 갱신하는 방식은 이번에 추가하지 않는다.
- graph 추정 때문에 signal history보다 더 이전의 관측을 쓰는 경우, ‘모든 방법에 똑같이512시점만 제공했다’고 하면 안 된다. 기본은 해당 origin의 허용 history 안에서만 G_s를 추정한다. prefix가 짧으면 자기 연결로 시작하고 유효 회귀 표본 수를 기록한다.
- 엄격한 global streaming graph가 필요하면 같은 과거 정보 예산을 모든 대조에 제공하는 별도 프로토콜 변경으로 보고한다. 이번 실험에서 조용히 입력 범위를 늘리지 않는다.
- FIXED와 UPDATED의 데이터량 효과를 줄이기 위해 FIXED도 FIT 마지막 최대512시점으로 추정한 것을 주 대조로 쓴다. 전체 FIT graph와 동일 정보량이라고 주장하지 않는다.
- 동적 graph에 검증 구간에서 이미 관측한 과거를 사용하는 것은 온라인 적응 조건이다. 이를 인코더 가중치를 validation 정답으로 튜닝하는 것과 구분해서 기록한다. 평가 오차를 보고 graph hyperparameter를 선택하지 않는다.

### 공통 모델과 평가

같은 작은 그래프–시간 인코더, d16, causal 시간 처리, 동일 D/readout, 목적과 update 예산을 사용한다. 파라미터 수·edge mask·규제만의 차이를 명시한다. 동적 graph의 계산 비용을 전체 latency에 포함한다.

재구성 또는 미래 목적 중 무엇을 쓸지는 E5의 사전 결정 규칙으로 정하고, graph 네 조건에 동일 적용한다. E5가 미완료이면 사전 기본값인 미래 예측 목적의 작은 모델로 하되 조건부 실험으로 명시한다. 각 조건 최대600 update, 두 seed.

합성은 E2의 생성 기록을 재사용한다. 새 neural graph 모델은 시나리오별 FIT 구간을 합쳐 학습할 수 있으나 train/진단 시간 경계와 record ID를 보존한다. 여러 기록을 합쳐 학습했다면 10개 기록의 점수를 10개 독립 모델 반복처럼 부르지 않는다. 알려진 edge·lag 회복은 합성에서만 평가한다.

실자료는 오차·horizon·연속 시간 구간·graph 변경량·latency를 함께 본다. graph 변화량 자체는 성공 지표가 아니다.

- FIXED가 SELF/WRONG보다 유리: 관계 활용의 후보 근거.
- UPDATED가 변화 조건에서만 유리: 갱신의 성립 조건 발견.
- FIXED와 UPDATED 비슷함: 고정 graph로 충분할 가능성.
- E2 단순 다변량 회귀보다 나쁨: 관계가 있더라도 신경망 graph의 추가 가치 미확보.

---

## 11. 검산·성능·비용의 필수 보고

### 11.1 모든 결과 행에 남길 최소 계약

아래 필드 이름은 이번 산출물의 새 제안이다. 기존 JSON에 이 키가 있다고 가정하지 않는다.

```text
experiment_id, attempt_id, dataset, seed, status
source_revision, source_hash, checkpoint_hash
fit_bounds, tune_bounds, eval_bounds, forecast_origins_hash
input_support_start, input_support_end, latent_timestamp_rule
effective_graph_fit_cutoff, graph_update_rule, export_mode
normalization_fit_scope, target_mask_hash
encoder_family, latent_shape, predictor, decoder, trainable_parts
actual_batch, optimizer_steps, checkpoint_selection
supervised_origins, unique_target_timestamps, duplicated_target_rows
mse, mae, horizon_metrics, per_channel_metrics
wall_seconds, gpu_job_seconds, peak_allocated_vram, peak_reserved_vram, sampled_peak_rss
failure_reason, remaining_uncertainties
```

### 11.2 독립 재계산과 부정 검사

- 저장된 예측·target·mask에서 별도 NumPy 계산으로 MSE/MAE를 재계산한다.
- source/checkpoint/graph가 고정되어야 할 비교에서 실제 hash가 바뀌지 않았는지 확인한다.
- causal forward의 future perturbation, split 경계, 원공간 역변환, latent timestamp와 phase를 검사한다.
- teacher/reference latent가 일반 예측 파일에 섞이지 않았는지 검사한다.
- 데이터 누출 guard는 파일명에 TEST 문자열이 있다는 이유만으로 정상 안전 코드를 차단하는 식으로 만들지 않는다. 실제 file access와 허용 경로·배열 경계를 검사한다.
- 오차 계산 일치 PASS를 수렴·표현 유용성·논문 성공으로 바꾸지 않는다.

### 11.3 비용 측정

공통 입력 묶음 B1/B32에서 warmup5/timed20을 기본으로 하고 GPU 동기화 벽시계 시간을 잰다. 짧은 데이터에서 batch32를 만들려고 일부 입력을 반복하면 반복 수를 명시한다. 모델 로딩·디스크 읽기는 제외하고 실제 전처리·표현 추출·graph 갱신·예측·디코딩·CPU 출력은 포함한다.

- 캐시된 표현에서의 head-only 시간과 실제 입력부터의 end-to-end 시간을 구분한다.
- prefix export가 느리면 그것도 방법의 현재 비용이다. 숨기거나 캐시 시간으로 대체하지 않는다.
- graph가 매32시점 갱신되면 갱신 시점/비갱신 시점의 비용을 따로 보고한다.
- GPU allocated·reserved·CPU RSS는 합산하지 않는다. sampling peak와 진짜 전체 peak도 구분한다.
- seed별 timing 중앙값의 평균/범위를 신뢰구간이라고 부르지 않는다.

---

## 12. 상태·중단·조건부 진행 규칙

상태는 최소 다음을 구분한다.

```text
NOT_STARTED
READY
RUNNING_CONFIRMED
COMPLETE_VERIFIED
FAILED_PRESERVED
BLOCKED_MISSING_ASSET
BLOCKED_DATA_CONTRACT
BLOCKED_PAIR_CONSTRUCTION
BUDGET_HELD
INCONCLUSIVE_OPTIMIZATION
NOT_APPLICABLE_WITH_REASON
```

실행 시작/완료 수, source versions, 실패 후 실제 소비한 update를 별도 원장에 쓴다. 끝난 결과의 재사용은 완료 fit 수에 새로 더하지 않는다.

### 즉시 또는 관련 묶음 중단

- 보호된 TEST 접근 시도, 미래 target의 입력 유입, split 경계 오류.
- 잘못된 채널/시간 대응, mask·정규화 불일치, 사라진/다른 checkpoint.
- NaN/Inf, 메모리 위험, 예산 초과, 동시 GPU 작업 충돌.
- positive/negative 구조가 선언한 목적과 다른 상태.

안전성 오류가 국소 wrapper 버그로 확인되면 새 attempt에서 수정·재검증할 수 있다. 데이터 범위·손실·모델 목표를 바꿔야 하는 경우는 별도 변경안으로 남기고 임의 진행하지 않는다. 오래 걸린다는 이유로 문제를 다른 연구로 전환하지 않는다.

### 결론 표기

- **유지:** 현재 같은 조건에서 일관된 이득을 보여 후속 후보로 남김. 독립 확증과 다름.
- **조건부 유지:** 특정 자료/horizon/지표/변화 구간에서만 도움.
- **보류:** 시간·자산·학습 안정성·비교 조건으로 판단 불가.
- **현재 구성 기각:** 완료된 공정 비교에서 단순 대안보다 오차/비용이 함께 불리. 전체 방법 계열의 불가능을 뜻하지 않음.

‘개선5%’ 같은 과학적 합격선을 새로 만들지 않는다. 선택 기준과 다음 실험 gate, 최종 성능 주장 수준을 구분한다. 성공하지 않아도 후보 배제나 조건 발견은 목표에 기여한다.

---

## 13. 최소 산출물과 사용자에게 할 최종 보고

파일을 많이 만드는 것보다 판단이 가능한 보고를 우선한다. 다음은 새 제안 파일명이며 실제 생성 여부를 확인해 보고하라. 하나의 JSONL에 여러 receipt를 모아도 된다.

```text
PLAN.md                       목적·왜·비교·예산·결정표
PROTOCOL.json                 결과 전 고정된 계약
ASSETS_AND_PREFLIGHT.json      로컬 환경/자산/보호 범위/예상 비용
STATUS.json                   실험별 실제 상태와 다음 판단
runs.jsonl                    완료/실패/재사용/비용 원장
metrics.csv                   새 실험의 비교 수치
EVIDENCE.json                 검산·hash·안전성 결과
REPORT_KO.md                  결과와 해석, 기존 결과와 분리
PROFESSOR_UPDATE_KO.md         교수님께 설명할 짧은 보고 초안
CHANGED_FILES.txt              새로 만들거나 수정한 파일; 자동 게시하지 않음
figures/                      판단에 필요한 그림만
```

그림은 최대8개: 정보 사용 검사, 과거 표현 revision, horizon별 지연 이득, 동일 export 비교, 참조 잠재값/예측기 비교, 학습 목적 비교, graph 변화 조건 비교, 전체 비용 중 실제 완료한 항목만 만든다. 축·범례·캡션은 영어, 보고 설명은 한국어로 한다. 결과가 없는 패널을 예상 곡선으로 채우지 않는다. t-SNE/UMAP 장식용 군집 그림은 이번 목적상 제외한다.

`REPORT_KO.md`의 앞부분은 여섯 질문에 대한 답과 근거 위치부터 제시한다. 방법 나열이나 실행 횟수로 시작하지 않는다. 각 질문에 아래를 채워라.

```text
무슨 질문인가?
무엇을 실제로 실행했나? 무엇은 미실행인가?
같은 조건/다른 조건은 무엇인가?
관측한 원수치는 무엇인가?
무엇을 지지하고 무엇은 지지하지 못하는가?
다음 판단은 유지/조건부 유지/보류/현재 구성 기각 중 무엇인가?
판단을 바꿀 최소 추가 근거는 무엇인가?
```

최종 사용자 보고에는 다음을 간단히 포함한다.

1. E1 실제 미래 누출과 단순 과거 표현 변경을 구분한 결과.
2. E2 지연 정보가 도움이 된 자료와 예측 거리.
3. E3 과거 전용 추출의 효과 또는 미완료 이유.
4. E4 표현/잠재 예측/복원 중 남은 병목, Bolt 유지 여부와 근거.
5. E5·E6 실제 실행한 조건과 실행하지 않은 이유.
6. MSE/MAE 불일치·seed·시간 구간 편차·반복 VAL 한계.
7. 소비한 계산량, 실패와 보존 위치, 생성된 파일 경로.

교수님 보고 초안에는 좋은 결과만 쓰지 않는다. ‘과거 전용이면 누출이 없어 좋다’와 ‘예측 정확도가 좋아졌다’를 구분하고, 원인이 확인되지 않았으면 미확정이라고 쓴다. 메시지는 자동 전송하지 않는다.

---

## 14. 지금 시작할 순서

**읽기·자산 확인 → 목적/계약 확정 → 안전 smoke와 비용 예측 → E1 → E2 → E3·E4 → gate에 따른 E5·E6 → 원수치 검산 → 보고**로 진행하라.

이미 승인된 범위에서 실제로 실행하되, 자료·모델이 없거나 안전성/예산 gate가 실패하면 그 지점과 가능한 독립 작업을 정확히 구분해서 보고한다. 계획만 다시 제시하고 멈추거나, 전혀 다른 대규모 실험으로 대체하지 않는다.
