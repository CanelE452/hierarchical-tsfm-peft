# 선택된 과거 전용 shrinkage 디코더와 새 native head의 조건 일치 비교

## 왜 이 비교가 필요한가

[확인] 기존 shrinkage 진단의 native 예측은 정적 디코더를 사용해 학습했던 head를 그대로 두고, 출력에 사용하는 디코더만 시점별 과거 전용 `D[n]`으로 바꾼 개입이다. 그 결과가 나빠졌다는 사실만으로 새 디코더에 맞춰 학습하는 작은 예측기까지 실패한다거나, Chronos-Bolt가 필요하다고 결론낼 수 없다. 이번 비교는 이미 선택된 같은 표현·축소 규칙·현재값 anchor·디코더 조건에서 작은 head를 새로 학습해 이 제한된 학습 불일치를 검사한다.

[판단] 원래 목적은 작은 `K`와 고정 Bolt를 사용하면서 원 `C` 예측의 정확도·비용을 유지할 가능성을 확인하는 것이다. 이번 단계는 그 목적을 `native가 이기는가`로 바꾸지 않는다. 고정 Bolt와 맞춘 경량 head의 오차를 비교해, 표현 공간·새 디코더·미래 잠재 예측 중 어디에 문제가 남는지 좁힌다. 새 head가 이겨도 모든 native 예측기의 우월성이나 Bolt의 불필요성을 증명하지 않으며, 져도 Bolt의 필요성을 증명하지 않는다.

## 이미 확정된 조건과 근거

[확인] 선행 CPU shrinkage 결과와 독립 감사는 실제 완료됐다. 이 문서를 작성하는 시점에 새 matched head의 실제 학습·추론·비용 측정은 수행하지 않았다. source와 합성 fixture만 준비한다.

- [shrinkage 실행 결과](runs/local_decoder_shrinkage_v1/RESULT.json), SHA256 `aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532`.
- [독립 shrinkage 감사](audits/LOCAL_DECODER_SHRINKAGE_AUDIT_v1.json), `PASS`, SHA256 `145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9`.
- 이전 [FIT-target 결과](runs/actual_fit_target_v1/RESULT.json)와 [독립 감사 v2](audits/FIT_TARGET_COMPARISON_AUDIT_v2.json)가 표현·정규화·원 native 학습 계약의 기준이다. 기존 소스·결과·잠금 기록을 보존한다.

선행 감사의 88개 양의 rho backend state, 13,504개 window matrix fit, 187,664개 양의 lambda solve, 4개 선택 잠금은 이전 단계의 실제 횟수다. 이번 새 head 학습 횟수와 합산해 같은 실험처럼 말하지 않는다.

```text
데이터    표현                   K   선택 rho   새 head seed          새 fit 수
Robin     PCA_PINV               4   0.1        92601, 92602          2
Robin     FIT_TARGET_PINV        4   0.1        92601, 92602          2
Beijing   PCA_PINV               2   0.1        92601, 92602          2
Beijing   FIT_TARGET_PINV        2   1.0        92601, 92602          2
```

[확인] rho는 선행 단계의 Bolt 전체 TUNE 정규화 MSE로 선택됐으며 이번 단계에서 다시 선택하지 않는다. 두 예측기는 같은 선택 규칙과 rho를 공유한다. 다만 `BOLT_NUMPY`와 `NATIVE_TORCH32`의 실제 FP32 과거 잠재값은 backend 연산 때문에 다를 수 있다. 따라서 native에는 선택된 `NATIVE_TORCH32`의 `D32[n]`과 잠재 과거를 사용한다. Bolt의 디코더·lambda를 복사하거나 두 backend의 행렬·lambda가 byte-identical이라고 주장하지 않는다. 같은 native 조건의 두 seed는 과거 잠재값·origin·디코더가 같아야 한다.

## 입력과 예측식

동일한 FIT 256개, TUNE 64개, 과거 길이 512, 미래 길이 48, 원 데이터의 observed mask·원점·FIT 정규화 통계를 유지한다. Robin은 원 채널 17개, Beijing은 11개다. 배포 시 입력은 관측 가능한 과거와 현재값이다. FIT-target 표현은 이전 단계에서 FIT의 미래 정답을 감독 신호로 학습했으므로 이를 비지도 또는 미래 정답을 전혀 쓰지 않은 표현으로 설명하지 않는다. 새로운 head forward에 미래 정답·미래 encoding은 들어가지 않는다.

`Z[n]`은 고정 E로 만든 `512 × K` 과거 잠재값이다. `D32[n]`은 해당 origin의 과거 `C` 값과 과거 observed mask만으로 선행 단계에서 계산한 `K × C` 행렬이다. 새 head는 각 잠재 차원마다 독립적인 affine `512 → 48` 맵을 둔다.

```text
latent_future[n,h,k] = Z[n,-1,k]
                       + sum_t W[k,h,t] * (Z[n,t,k] - Z[n,-1,k])
                       + b[k,h]

delta[n,h,k] = latent_future[n,h,k] - Z[n,-1,k]
prediction[n,h,c] = currentC[n,c]
                    + sum_k delta[n,h,k] * D32[n,k,c]
```

실제 native FP32 계약은 `future=head(Z)`, `delta=future-lastZ`, `currentC + torch.einsum('nhk,nkc->nhc', delta, D32[n])`이다. 축소 디코더의 절편(bias)은 현재값 anchor를 사용하는 innovation 식에서 상쇄되므로 forward 입력·학습 변수에 포함하지 않는다. 절편과 E/D는 고정이고, head의 W/b만 학습한다. head 입력은 잠재 과거이며 디코더 계수 자체를 head에 추가 입력하지 않는다. 다만 전체 출력 시스템은 `K` 흐름 외 현재 `C`와 과거에서 만든 `K × C` 디코더 계수를 사용하므로 strict K-only 정보 인터페이스가 아니다.

CP0은 W/b가 0이므로 현재값 예측과 FP32 수치상 같다. 음수 0의 부호 비트까지 항상 같은 것으로 주장하지 않는다. 같은 origin의 `Z`, `currentC`, `D[n]`, target, mask를 함께 batch 인덱싱해야 한다. 행렬만 다른 origin 순서로 섞으면 다른 예측이 된다.

## 학습·선택 계약

- 8개 새 fit, 조건당 2개 seed. 모든 head는 0 초기화이며 seed는 minibatch 순서를 바꾼다. 두 seed를 서로 다른 무작위 초기화라고 설명하지 않는다.
- 기존과 같은 AdamW, learning rate 0.001, weight decay 0, gradient norm clip 1, batch 4, 600 updates. 전체 최대 4,800 updates와 40개 CP이며 기존 head를 재사용하거나 warm-start하지 않는다.
- 손실은 해당 minibatch에서 관측된 실제 미래 `C`의 정규화 제곱오차다. 각 채널의 관측 개수로 나눈 뒤 관측이 있는 채널을 균등 평균한다. 전체 FIT 관측 개수를 minibatch 분모로 사용하지 않는다. 관측되지 않은 target의 NaN은 손실·gradient에서 제외한다.
- CP는 0/100/200/400/600만 평가한다. 전체 TUNE의 관측 개수로 계산한 채널 macro 정규화 MSE가 가장 작은 CP를 선택하며 exact tie는 이른 update를 선택한다. 현재 CP 점수에 쓰인 canonical batch32 예측 배열을 그대로 저장한다. full-N 재계산 예측으로 저장 배열을 바꾸지 않는다.
- rho나 K, decoder, 학습량, objective, CP 규칙을 새 결과에 맞춰 바꾸지 않는다. 기존 Bolt는 고정된 예측 결과를 비교에 재사용하며 새 backbone 학습은 없다.

8개 head의 선택 CP·소스·입력 lineage를 잠근 후 새 DEV 예측을 수행한다. 일부 조건이 실패하면 원인을 해당 조건에 기록하고 가능한 다른 조건을 계속한다. 부분 실패가 생길 때는 각 조건을 `COMPLETE` 또는 명시적 실패로 해결한 registry를 잠근 뒤 DEV를 열며, 실패 조건을 완료하거나 통과한 것으로 계산하지 않는다. 이미 노출된 DEV와 oracle가 후속 설계 동기에 영향을 주었으므로 전체 설계가 DEV-independent라고 주장하지 않는다.

## 최소 검증과 후속 실행 순서

1. 새 소스·기존 source의 존재와 SHA, 실행 중 작업을 점검한다. 기존 작업을 종료하거나 결과를 덮지 않는다.
2. 독립 shrinkage 감사 `PASS`와 producer SHA, 원 FIT-target `PASS v2`를 exact bind한다. 선택 잠금·같은 native 두 seed의 잠재 과거·origin·D[n]가 일치하는지 실제 controller가 검산한다. 여기 준비한 head 모듈은 과학 파일을 직접 읽지 않는다.
3. 합성 fixture에서 head별 독립성, per-window pairing, 동결 gradient, 관측 개수 macro 손실, 미관측 NaN, CP exact tie, batch32 저장 점수 일치, 직렬화 parity, 초기값/seed 의미를 확인한다.
4. FIT/TUNE만으로 8개 head를 학습하고 실제 loss·gradient·CP·선택 상태·자원 로그를 저장한다. head 입력·디코더·현재값의 전후 SHA를 확인한다. 관측 정답이 아닌 입력에서 nonfinite가 생기면 해당 경로를 중단한다.
5. 모든 선택을 잠근 뒤 과거 전용 forward로 실제 FIT/TUNE/노출 DEV 예측을 저장한다. 원 Raw, 선택된 Bolt, 이전 frozen-head 개입, 새 matched head를 같은 origin·mask·정규화 조건으로 비교한다. 전체 관측 지표와 완전 `C` mask 진단을 섞지 않는다.
6. 독립 감사 후 한국어 보고서에 원수치·조건·실패·불확실성을 기록한다. 비용은 아직 수행하지 않은 것으로 표시하고, 새 native 전체 causal 전처리→E→head→D→physical 반환을 실제 측정할 경우 별도 producer·감사·타이머 근거를 둔다. 기존 frozen native 비용이나 Bolt 비용을 새 matched native 비용으로 대체하지 않는다.

모듈 API는 `train_native_shrunk(fit_payload, tune_payload, seed, progress_hook, checkpoint_callback)`다. payload는 FP32 `latent_history`, `decoder`, `raw_last`, `target`, Bool `observed`, Int64 `origins`를 받는다. CP callback은 `(update, state, score, prediction)` 네 인자를 받으며 `prediction`은 그 CP 선택에 실제 사용한 batch32 배열이다. 반환은 선택 model/state, metadata, learning curve와 모든 CP다. 실제 512/48·8조건·600update 계약과 감사 lineage를 적용하는 controller는 별도로 준비·검토한다.

## 해석 한계와 다음 판단

[확인] 새 비교는 정적 D로 학습한 head를 조건부 D에 붙인 학습 불일치의 영향을 검사한다. 새로운 목적 함수는 이미 원 native/POINTWISE/MAIN에서 사용한 실제 미래 `C` MSE이며, 이전 전체 실패를 재구성 학습만의 문제로 설명하지 않는다. 새 D 조건에서 같은 목적을 다시 학습하는 것이 변화다.

[판단] matched head가 좋아지면 기존 frozen-head 개입의 일부 실패는 학습·디코더 불일치와 양립한다. 여전히 Raw보다 나쁘면 표현 손실, 잠재 미래 예측의 어려움, 제한된 affine head, 디코더 공간·gain·학습량 등의 후보가 남는다. 반대로 matched head가 나빠져도 모든 학습 문제나 모든 native 구조의 한계를 확정할 수 없다. 시점별 D는 공간 방향과 gain을 함께 바꾸므로 증폭 하나를 유일 원인으로 지목하지 않는다.

새 head는 해당 D에 대해 실제 미래 C 손실을 최적화하지만 D는 과거 적합의 결과로 고정돼 있다. 기존 정적 D floor나 split별 임의 rank-K oracle를 새 예측의 하한으로 그대로 쓰지 않는다. 필요 시 독립 감사가 각 origin의 실제 D32[n]를 대상으로 같은 완전 C mask에서 재계산한 normalized-MSE-only 진단을 사용한다. 알려진 미래로 산출한 해당 진단은 실제 예측이나 과거 정보의 충분성을 증명하지 않으며 MAE·원단위 MSE의 최소값도 아니다.

이번 비교는 노출 DEV에 대한 탐색적 비교다. FIT/TUNE/DEV window·horizon이 독립 표본이라는 가정을 하지 않고, 두 seed의 수치 차이를 일반화 유의성으로 바꾸지 않는다. 보호된 TEST를 사용하지 않으며, 설치·큰 백본 재학습·기존 중단 작업 재개·기존 작업 종료·commit/push를 수행하지 않는다. 새 head 학습과 비용이 실제로 완료되고 독립 검산될 때까지 이를 해결·완료 결과로 보고하지 않는다.
