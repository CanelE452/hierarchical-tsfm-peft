# 공통 정보 gate 재검토와 다음 최소 비교안

상태: **기존 결과·소스·원수치 재검토 완료 / gate 수정 및 신규 표현 실험 승인 대기**. 이번 메모는 main `da1114945f90b32017b3caf39b81538557af6fd1` 게시 뒤 작성한 후속 검토다. 새 학습·실제 모델 추론을 수행한 보고서가 아니며 자동 게시하지 않는다. [게시된 완료 결과](publication_main_20261014_v1/README_KO.md)와 [기존 승인 계획](PHASE1_PLAN_KO.md)은 보존한다.

## 목적과 현재 판단

최상위 목적은 작은 K의 과거 시계열을 고정 Bolt에 넣었을 때 원미래 정확도와 전체 비용에서 쓸 이유가 있는지 판단하는 것이다. 다음 실험의 소비처는 교수님과 연구 방향을 결정하는 미팅이다. SELF/JOINT ridge 승패를 해결 그 자체로 바꾸지 않는다.

[확인·논리] 모든 채널이 같은 s(t)를 관측한다면 자기 채널만으로 s를 알고 다른 채널을 추가해도 정보 이득은 없다. 동시에 한 채널을 선택해 모든 채널로 복사하면 K=1로 정확히 표현할 수 있다. 따라서 **JOINT의 추가 이득은 공통 패턴의 압축 가능성에 필요한 조건이 아니다**. 이 반례가 실제 Robin·Beijing에 적용된다는 증거는 없다.

[확인] 이번 negative ridge 결과는 같은 시간 요약·선택 규칙에서 얻은 유효한 실패 경계다. 그러나 SELF33/JOINT33은 Robin에서 33/561, Beijing에서 33/363개의 특징을 쓰며 같은 함수 추정 난도를 갖지 않는다. 유한 표본·규제·상관된 특징 때문에 저차원 제약이 더 잘 일반화할 가능성은 남아 있다. 그 가능성이 실제 개선을 증명하지 않는다.

## 첨부 비판에서 다시 검산한 것

[메타데이터 검산](GATE_REVIEW_METADATA_AUDIT_v1.json)은 기존 CSV와 terminal JSON만 사용한다. 실제 과거/미래 NPZ·원자료·checkpoint 내용은 읽지 않는다.

```text
Robin             PERSIST                 RAW_BOLT
정확히 MSE=0      TUNE 10/17 · VAL 1/17    TUNE 0/17 · VAL 0/17
MSE ≤ 1e−8        TUNE 10/17 · VAL 1/17    TUNE 3/17 · VAL 1/17
TUNE macro MSE    .541433527292            .056951090934
VAL macro MSE     1.308464073360           .340401887028
```

모델과 zero 정의가 다른 두 관찰을 혼동하지 않는다. 이것은 선택된 창에서 target이 해당 예측과 일치한 특성이며 원자료 전체의 상수성·미래 누출을 입증하지 않는다. TUNE와 개발 기간의 차이는 남지만, 개발 점수를 보고 새 방법·K·평가 기준을 유리하게 바꾸지 않는다.

## 이미 수행한 예측 목적 표현과 중복 제거

[확인·소스] [ANCHOR_POINTWISE](../smallk_predictive_alignment_20261013/engine.py#L92)는 affine C→K E와 affine K→C D를 함께 실제-Y 예측 손실로 학습한다. 경로는 `X_last + D(F(E(X))) - D(E(X_last))`이며 F 가중치는 동결된다. Jena K5/Robin K4 × 두 seed, FIT256/TUNE64·600 update·정규화 macro MSE 선택의 4조건이 있다. checkpoint 파일 4개 존재까지 확인했고 내용을 다시 읽거나 실행하지 않았다.

```text
자료     pointwise TUNE MSE          pointwise 개발 VAL MSE
Jena     1.105706 / 1.092841          .882636 / .863282
Robin     .076319 / .078170          .566753 / .576821
```

중요한 차이는 초기화다. [basis 선택 소스](../smallk_predictive_alignment_20261013/engine.py#L456)는 ANCHOR_POINTWISE에 TEACHER basis를 사용한다. full-C 예측 모델을 TUNE로 선택한 뒤 FIT 예측 innovation 공분산으로 얻은 초기화다. 이를 PCA 초기화와 같은 실험으로 부르지 않는다. 기존 v11 learned-U는 원 C NLinear 우회와 tied orthogonal U/Gamma를 갖고 이전 통계·선택 계약을 쓰므로 현재 결과와 직접 순위를 합치지 않는다.

[판단] '재구성 대신 예측 E/D를 하자'는 제안만으로는 새 실험 공백을 메우지 못한다. 같은 구조의 반복 학습보다 **분산 기반 투영과 시간 예측성 기반 투영을 같은 복원 조건에 놓고, 같은 표현에서 F와 단순 예측기를 바꾸는 비교**를 우선한다. Beijing pointwise의 미실행·PCA 초기화의 미실행도 별도 공백으로 보존하되 한 번에 모두 학습하지 않는다.

## 선택한 최소 후속: PCA와 ForeCA 원리 투영의 같은 E/D 비교

경쟁 설명은 ①K가 원미래 변화에 필요한 정보를 보존하지 못함 ②정보는 남지만 고정 F가 혼합된 좌표의 미래를 예측하지 못함 ③복원 D/현재값·일반화가 제한함이다. 높은 latent forecastability만으로 성공을 선언하지 않는다.

선택 후보는 다음과 같다.

- **고정 PCA와 ForeCA 원리의 고정 선형 투영:** variance와 spectral entropy의 기준을 바꾸되, K와 D 클래스·FIT 입력·정규화·현재값 조건을 맞춘다. ForeCA는 whitening과 비상수/유효 rank 제약을 사용해 정적 표현 붕괴를 막는다. 원리 기반 Bolt 적용이며 공식 R 구현의 정확한 benchmark 재현이 아니다. [ForeCA 공식 논문](https://proceedings.mlr.press/v28/goerg13.pdf).
- **예측 목적 pointwise E/D의 PCA 초기화:** 기존 teacher 초기화와 비용 차이를 제거하는 보충 대조지만 구조 자체는 이미 수행했다. 이번 첫 묶음에서는 재학습하지 않고 기존 결과로 분리한다. [AdaPTS ICML 서지](https://proceedings.mlr.press/v267/benechehab25a.html)는 표준 adapter의 중복을 확인하는 출처이며 기존 v1 열람을 최종판 본문 열람으로 확대하지 않는다.
- **실제 채널 대표 K개를 선택:** 새로운 좌표 혼합을 피할 가능성이 있지만 채널/cluster 선택과 복원 규칙이 별도 자유도를 만든다. 첫 묶음에서는 실행하지 않는다. 이 단순 대안도 실패한 ridge에 의해 논리적으로 배제된 것은 아니다.

**첫 묶음의 계약:** Robin C17/K4와 Beijing C11/K2, L512/H48, 원 FIT256/TUNE64·Robin VAL30·이미 노출된 Beijing EVAL146, FIT-only scaler·창별 causal FF·원 관측 mask를 유지한다. 보호 TEST·새 자료·긴 L은 없다. E는 시점별 C→K이므로 과거 내부 prefix/future 누출이 없는 선형 경로다. 동일 FIT 입력으로 얻은 선형 D를 고정하고 현재 원 C의 마지막 값 anchor를 두 경로에 동일하게 넣는다. 따라서 K stream + current C이며 strict K-only로 부르지 않는다. 다른 E의 D 가중치가 byte 동일하다는 뜻은 아니고, D 클래스·적합 입력·학습 규칙을 같게 한다.

투영 적합은 동일한 FIT 과거 창256개를 사용하고, 시간/창 중복의 가중치를 두 경로에 동일하게 둔다. 이를 독립 표본으로 세지 않는다. PCA는 FP64 공분산의 상위 K성분이다. ForeCA는 같은 FIT 공분산으로 whitening한 뒤, 전체 FIT 평균을 뺀 과거에512점 Hann을 적용한 단측257빈 스펙트럼을 창 평균한다. DC/Nyquist의 power는1배·내부 빈은2배이며, 반사 padding의7빈 이동평균·확률 정규화 뒤 entropy를 줄이는 원리 적용이다. 성분별 최대200반복·결정적 초기 방향2개만 FIT entropy로 선택하고, 단위 분산/직교 제약·최대 고유값 대비 rank cutoff1e−8을 고정한다. 수식과 작은 fixture로 power/제약 검사를 먼저 검증하고 TUNE/DEV로 spectral 설정을 탐색하지 않는다. 유효 rank<K 또는 유한값/제약 검사 실패는 해당 경로를 INFEASIBLE/FAILED로 남기며 임의 규제로 성공으로 바꾸지 않는다. D는 두 경로 모두 같은 FIT 과거의 선형 최소제곱 복원으로 적합하고 미래 label로 재적합하지 않는다.

배포식은 `X_last + D(F(E(X_past))) - D(E(X_last))`다. 원 C 전체 과거를 읽는 추가 residual bypass는 넣지 않는다. PCA/ForeCA 각각에서 E와 D를 고정한 뒤 **채널별 native 선형 time head 하나**를 actual-Y 관측 macro 정규화 MSE로 학습해 F만 교체한다. 두 neural seed 92601/92602, 각600 update·batch4·checkpoint0/100/200/400/600과 같은 TUNE 선택을 적용한다. 표현/decoder를 native 경로에서 다시 학습하지 않는다. 결정적 PCA·ForeCA를 seed로 중복하지 않는다. native 학습은 2자료×2표현×2seed, 최대8조건이다.

원 E(Y_future)를 넣은 복원은 **진단 전용 oracle**로만 별도 표기한다. 실제 future는 배포 입력·TUNE 선택·graph·correction에 들어가지 않는다. decoder의 정보 손실이 남는지 검사할 보조이며 모든 비선형 decoder의 최적 하한이라는 주장은 하지 않는다. E/D는 FIT 과거 복원으로 적합한 조건임을 명시하고, 미래 복원 적합성을 가정하지 않는다.

총량은 결정적 표현4개·고정 D4개·native 학습 최대8개이며 신경망 encoder/대형 backbone 학습은0이다. 실제 Raw parity와 F/설정/normalizer/source SHA 전후, FIT 통계·origin·입력 고정, 미래 교란, whitening/rank·저장 예측/지표·TUNE 선택을 검증한다. 비용은 Raw/PCA-F/ForeCA-F/PCA-native/ForeCA-native ×2자료×B1/B8, 최대20조건·120회, native seed92601의 배포 경로로 측정한다. 전처리부터 원단위 CPU 반환까지 같은 trial 안에 넣고 타이머 밖 작업을 표시한다. 실패 경로는 보존하고 독립 가능한 경로만 계속한다.

## 결과별 판단과 자원

첫 묶음은 JOINT ridge gate와 관계없이 끝까지 실행·검산·보고한다. ForeCA가 PCA보다 latent entropy만 낮고 원미래 오차가 줄지 않으면 압축 성공으로 세지 않는다. oracle가 낮은데 F만 나쁘면 F 호환성의 단서이고, native도 나쁘면 특정 native의 학습·예측 범위와 representation을 함께 검토한다. oracle가 높으면 이 E/D의 복원/보존 한계가 남는다. 어떤 결과도 단독 원인으로 확정하지 않는다.

채택은 TUNE의 원미래 MSE·MAE와 Raw·기존 단순 대안 대비 전체 비용의 원수치로 판단한다. 상대 변화1%는 이전과 같은 탐색 해상도 표기로 유지하고 이를 유의성으로 부르지 않는다. Raw 대비 TUNE MSE 증가1% 이내·MAE 증가1% 이내이며 같은 B8 whole 중앙 지연이10% 이상 줄 때만 '개발 조건의 비용 절충 후보'로 표시한다. 이는 첫 연구 성공의 필요조건이나 독립 일반화 기준이 아니다. B1·개발 평가의 악화도 모두 보존한다. 결과를 보고 새 margin/K/자료를 바꾸지 않는다. 개발 평가를 보고 추가 모델을 선택하지 않는다.

새 hyperparameter 계열을 더하지 않고 위 고정 묶음1회 후 보고한다. 이를 전체 연구 목표 완료나 불가능의 증명으로 축소하지 않는다. 다음 supervised E/D나 graph·contrastive 분기는 이번 묶음에 자동 포함하지 않는다.

**자원은 기존 승인 상한을 늘리지 않는다:** CPU process120분·GPU process90분·저장6GiB·RSS8GiB·GPU reserved9GiB의 실제 누적 사용량에 이어 기록하고, 기존 고정 wall 종료 **2026-10-15 00:14:37 KST**를 유지한다. 마지막 보고 snapshot은 CPU 약97초·GPU 약40초였지만 새 실행 직전에 receipts를 다시 합산한다. 2026-10-14 22:31 조회 당시 시계상 약103분이 남았으며 approval 대기·구현 시간도 지나간 wall이다. [추정] 구현/학습/실제 추론/검산/보고에45–75분, 추가 GPU process20분·CPU process30분·저장1GiB 이내를 예상하며 완료·개선은 보장하지 않는다. 종료 시각을 넘기면 해당 새 실행을 시작하지 않고 미완료/사용량/재개 요구를 기록한다.

승인이 필요한 변경은 **'JOINT gate 미통과 후 새 표현 금지'를 이 고정 묶음1회에 한해 해제하는 것**이다. [기존 계획](PHASE1_PLAN_KO.md)은 “이때도 이득이 없으면 이 계열 추가 학습을 멈추고”라고 명시한다. 잔여 자원만으로 그 조건을 자의적으로 없애지 않는다. 원 계획·승인·시계·source·결과·게시 manifest를 수정하지 않고 별도 gate 변경 기록을 추가한다. 같은 범위의 하위 실험마다 승인을 다시 요구하지 않는다.
