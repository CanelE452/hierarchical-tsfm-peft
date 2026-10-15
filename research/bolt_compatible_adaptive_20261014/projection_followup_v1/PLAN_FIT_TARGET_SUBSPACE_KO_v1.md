# FIT 실제 미래 변화량을 사용한 고정 출력 공간 비교

상태: **SOURCE·계획·작은 CPU 합성 검산 단계**. 이 문서는 실제 신규 FIT, native 학습, Bolt 추론, 비용 측정 또는 검산 완료를 뜻하지 않는다. 실제 실행은 별도 ROOT controller 준비 후 수행한다. 기존 producer·검산·승인 기록·선택 잠금·결과는 수정하지 않는다.

## 목적과 기존 비교의 빈틈

과거를 잘 복원하거나 과거 스펙트럼의 예측 가능성을 높이는 방향이 실제 미래 변화량을 보존하는 방향이라는 보장은 없다. 현재 고정 PCA/ForeCA와 동일 cached future의 디코더 개입은 이 문제를 해결했다고 보여주지 않았다. 다음 최소 대조는 **FIT의 실제 미래 변화량을 직접 보존하는 선형 공간을 학습하면 같은 K·같은 과거 입력·같은 예측기·같은 디코더 규칙에서 달라지는가**다. 결과가 나빠도 FIT 공간의 일반화, 잠재 미래 예측, 복원 중 남는 문제를 분리하는 데 사용한다.

이전 D1은 FIT의 RAW 예측 변화량으로 공간을 만들었다. 이전 POINTWISE에는 실제-Y 예측 목적 E/D 공동 학습이 있었다. 따라서 이번을 첫 예측 목적 표현학습이라고 주장하지 않는다. 이번의 차이는 공동 최적화 없이 **FIT 실제 Y로 만든 결정적 출력 공간을 고정하고**, 그 안의 과거 표현을 whitening한 뒤 frozen Bolt와 같은 native head를 비교하는 통제다. 그래프나 대조학습의 필요성·새 방법의 독창성을 입증하는 설계가 아니다.

## 데이터·학습·배포 계약

Robin C17/K4와 Beijing Aotizhongxin C11/K2, L512/H48, 기존 FIT256/TUNE64와 기노출 DEV를 유지한다. 신규 dataset·K 탐색·TEST·대규모 backbone 학습은 없다. 정규화는 기존 FIT 통계, 평가·native 학습은 관측 실제-Y 채널 macro 정규화 MSE다. 현재 원 C값을 anchor로 사용한다. complete-C 미래 위치만 공간 학습에 사용하고, 전체 관측 평가와 complete-C 보조 평가를 구분한다.

`fit_target_subspace`에는 FIT history·과거 관측 mask·FIT 실제 Y·미래 관측 mask·현재 C·origins만 들어간다. 모든 원 C가 관측된 FIT 미래 위치에서 `V = Y64 − currentC64`를 만들고 **평균을 빼지 않는 SVD**의 top K 오른쪽 방향 W를 취한다. 미래 평균이나 intercept를 추가하지 않는다. 겹친 window의 반복 위치를 그대로 포함하고 해당 counts·배열 SHA를 남긴다. 미래 관측이 불완전한 위치 전체는 V에서 제외한다.

FIT 과거의 causal forward-fill 값을 모두 포함해 원 global 과거 mean을 구하고, `(X−mean)W`의 K차원 공분산 안에서만 whitening한다. 상대 고유값 hardcut `1e−8`로 K개의 과거 분산 방향이 유지되어야 한다. regularization이나 K 변경으로 실패를 숨기지 않는다. 목표 변화량 자체의 수치 rank도 `1e−15` 상대 cutoff에서 K 이상인지 명시적으로 확인한다. SVD·encoder의 부호는 최대 절댓값 좌표가 양수가 되도록 정하지만 동률 특이공간이 다른 수치 라이브러리에서 유일하다고 주장하지 않는다.

배포 E는 고정 pointwise C→K 선형 연산이다. 미래 정답이나 현재 예측 horizon은 입력으로 들어가지 않는다. FIT의 과거→미래 pair로 공간을 학습하는 **지도학습**이며, 이를 unsupervised/past-only training이라고 쓰지 않는다. **배포가 과거만 사용하는 것**과 **학습에 FIT 미래 정답을 쓰는 것**을 구분한다. 기존 train-FIT/selection-TUNE 계약을 전제로 하며 array별 실제 split 결속은 controller와 독립 검산이 맡는다.

## 같은 디코더·예측기 조건

두 조건은 `PCA_ENCODER_PINV`와 `FIT_TARGET_SUBSPACE`다. PCA의 기존 audited E와 center를 그대로 사용한다. 둘 다 `D32 = pinv(E64→FP32→FP64,rcond=1e−15)→FP32`로 구성하고 decoder 재구성 bias는 원 FIT 과거 center다. anchor 예측은 `currentC + (futureK−lastK)D32`로 동일하며 bias가 상쇄된다. 단위 FIT 잠재분산, pointwise 구조, K, 예측기, 실제-Y 목적과 decoder 규칙을 맞춘다. 과거 최소제곱 decoder와 이번 inverse decoder를 동일한 통제로 섞지 않는다.

F는 frozen Chronos-Bolt를 유지한다. 새 비교에서 native는 두 자료×두 공간×두 seed의 **8개 새 학습**을 같은 600 updates·batch4·동일 seed·TUNE macro MSE checkpoint 선택으로 수행할 예정이다. 기존 native는 원 past-LS D를 위해 학습했으므로 새 inverse D의 최적화된 PCA/native 대조로 재사용하지 않는다. 기존 native/새 D 개입 결과는 역사적 원인 진단으로 보존한다.

모든 E/D는 FIT 이후 고정한다. Bolt는 같은 frozen 모델·정규화·중앙 예측·batch 계약을 사용하고 실제 추론한다. native head만 FIT 실제-Y 손실로 학습한다. TUNE은 head 선택에만 쓰고 공간·K·whitening cutoff·branch를 선택하지 않는다. DEV는 기노출 재검사이며 독립 일반화 검증으로 바꾸어 말하지 않는다.

## 이전 oracle와 해석 범위

이미 알려진 미래 행렬에서 split별로 방향까지 최적화한 rank-K oracle는 배포할 수 없는 진단이다. 그 TUNE/DEV 방향, singular vectors를 이번 parameter로 재사용하지 않는다. 이번 W는 controller가 SHA로 확정한 **FIT 실제 Y만으로 새로 계산**한다. K·cutoff를 oracle 수치로 최적화하거나 oracle 지표로 현재 run의 head checkpoint를 선택하지 않는다. TUNE checkpoint 선택은 기존 실제-Y macro MSE 규칙을 사용한다.

이 후속은 이미 DEV·고정-D span·rank-K oracle 결과를 본 뒤 정한 **탐색적·적응적 연구**다. 전체 후속 설계 선택의 동기에 DEV/oracle가 미친 영향이 0이었다고 입증하지 않는다. 확인 가능한 제한은 현재 run의 FIT-only 공간 학습, 미리 고정한 K/cutoff, TUNE에 한정한 head 선택, oracle 방향의 parameter 재사용 금지다. 이전 진단이 연구 판단에 주는 맥락과 실행 내부에서 parameter/checkpoint를 고르는 것은 구분한다. 따라서 이번 기노출 DEV 결과를 독립 확증·독립 일반화 검증으로 제시하지 않는다.

FIT 목표 공간이 미래 오차를 줄여도 공간 보존과 latent forecastability가 모두 검증됐다고 말하지 않는다. FIT에서는 목표를 잘 보존해도 TUNE/DEV에서 출력 공간이 달라질 수 있다. 출력 공간이 적절해도 frozen Bolt가 mixed latent의 변화량을 잘 예측하지 못할 수 있다. 개선이 없으면 단순 FIT 목표 공간 대안의 실패 범위를 수치로 보고하며 전체 표현학습·비선형 방법·다른 K의 불가능성으로 확대하지 않는다.

## 실행·검산 단계

이번 ownership은 [순수 수학 모듈](future_target_projection_v1.py), [합성 검산](test_future_target_projection_v1.py), 이 계획이다. 실제 data/NPZ/checkpoint/model은 이 모듈의 import에서 읽지 않는다. 회전된 목표와 input-PCA 방향 차이, 단위 과거 분산, 미래 평균을 빼지 않는 목표, 불완전 미래 위치 제외, rank 실패, inverse D, CP0 anchor, 배포 prefix 불변, TUNE가 없는 fit API, serialization과 C/Fortran layout을 작은 합성 데이터로 확인한다.

ROOT controller가 기존 COMPLETE producer·PASS audit·FIT array/origin/mask/stat SHA를 묶은 뒤 신규 공간을 fit하고, 같은 조건의8 native 학습·실제 Bolt/native 추론·두 mask의 지표·전체 배포 비용·독립 검산·한국어 보고를 이어갈 예정이다. 새 파일과 결과는 fresh 경로로 남긴다. 독립 검산은 producer의 SVD helper를 가져다 쓰지 않고 FIT target Gram 및 restricted past covariance로 재계산해야 한다. 원 예측·선택·캐시·소스 보존도 검산한다.

`[판단]` 이 설계는 사용자가 계속 수행하도록 한 예측 목적 표현·원인 진단 목표의 최소 통제 후속으로 해석한다. 새 exact-human 개별 승인이 있다고 주장하지 않고 기존 승인 문서를 수정하지 않는다. 최신 지속 수행 지시와 자원 상한 해제는 ROOT가 관리한다. TEST 접근·기존 작업 종료·설치·큰 모델 학습·자동 commit/push는 포함하지 않는다.
