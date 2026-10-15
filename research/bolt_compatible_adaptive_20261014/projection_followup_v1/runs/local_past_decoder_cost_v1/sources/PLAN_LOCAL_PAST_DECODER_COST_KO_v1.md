# LocalD의 전체 Bolt 배포 비용: 실행 전 계약

[판단] 이 비교는 진행 중인 과거 정보 기반 디코더 원인 진단을 위한 비용 검증이다. 별도의 정확한 사용자 승인 발언이 있었다고 주장하지 않는다. 기존 CPU LocalD 진단의 완료·독립 PASS와 원 FIT-target 비교의 완료·독립 PASS를 SHA로 연결한 후에만 실제 자산과 GPU를 사용한다. 사용자가 해제한 누적 자원 hard cap을 다시 적용하지 않는다. 보호된 TEST, 새 데이터, 새 K 선택, 설치, backbone·encoder·head 재학습, 과거 작업 종료·재개, commit/push는 포함하지 않는다.

## 질문과 비교 조건

같은 원 데이터·FIT scaler·C/K·512 과거·48 미래·frozen Bolt·current C anchor에서, 실제 배포 때 origin별 masked 과거 LS를 수행하면 얼마나 시간이 드는가? CPU의 cached future를 재복원한 시간과 실제 E→F→D 전체 배포 시간을 구별한다.

경로는 RAW_BOLT, PCA_PINV_BOLT, FIT_TARGET_PINV_BOLT, PCA_LOCAL_BOLT, FIT_TARGET_LOCAL_BOLT의 다섯 개다. Robin C17/K4와 Beijing C11/K2, 각 기존 development split, B1/B8이므로 계획은 20셀이다. 셀별 warmup 2회·측정 6회로 120개 측정 샘플을 기대하되, 실제 완료 수를 기록한다. 처음 다섯 circular rotation만 완전히 균형이고 여섯 번째 seeded permutation은 추가 반복이다. 여섯 번 전체가 완전 Latin 균형이라고 하지 않는다. 같은 첫 B개 origin을 모든 경로·반복에 사용한다.

Native head를 로드하거나 측정하지 않는다. 기존 native 결과는 고정된 head에 대한 디코더 개입이며 새 D로 재최적화한 모델 결과가 아니다. 이번 비용 결과가 그 native 비용을 대신하지 않는다.

## 실제 배포와 타이머

타이머는 원 물리 단위 과거의 causal fill 이전에 시작한다. CPU FIT normalization → 고정 NumPy E → local 경로의 observed-centered slope LS(rcond 1e-4)와 함수에 구현된 진단·hash 계산 → H2D/F/D2H → local 또는 static D → current C anchor → CPU 물리 단위 반환과 최종 동기화까지 포함한다. local D는 매 callback에서 새로 계산한다. 원 origin별 D를 cache하거나 타이머 밖에서 재사용하지 않는다. 기존 `fit_local_past_decoder`의 진단 계산을 비용 절감 목적으로 제거하지 않는다.

데이터·model·state 로드, source 검증, preflight, warmup, 배열 저장·저장물 SHA, memory snapshot은 측정 밖이다. 타이머의 source와 계산 경계는 기존 `whole_fit_target_cost_v1.py`에서 복사했으며 경로·mask callback·설명만 바꾼다. 같은 resident frozen F의 process RSS·peak VRAM을 기록한다. 각 방법의 독점적 RAM/VRAM이라고 주장하지 않고 peak counter도 초기화하지 않는다. stage별 비용을 별도로 분해했다고 하지 않는다.

## 수치 재현 gate

먼저 실제 물리 과거 first8을 fill/normalize하고, cached normalized history·current C·NumPy E 표현에 비교한다. 원 observation mask와 origin은 byte-exact이어야 한다. 같은 함수로 local D를 재적합하여 deployed FP32 계수·intercept와 singular values를 검증하고, 관측 수·slope rank·affine rank·rank deficient·low observed count·no-observation fallback 결정은 exact로 비교한다. rcond 1e-4도 exact 계약이다. 전처리 rounding이 작은 잠재 방향이나 rank 결정에 영향을 주는 경우 output parity만으로 통과시키지 않는다.

각 B1/B8 실제 F 출력과 physical return은 원 cached static/local forecast와 비교한다. normalized atol 1e-5/rtol 2e-6, physical atol 2e-3/rtol 2e-5를 기존대로 유지한다. 계수 검증에도 normalized 수치 허용오차를 그대로 적용하며 실패를 지우려고 완화하지 않는다. normalized 공차와 physical 공차는 둘 다 통과해야 한다. 계수 gain이 작은 F·preprocess rounding을 증폭해 기존 결과와 달라질 수 있다. 그런 경우 해당 경로의 원인과 fresh 실패 artifact를 보존하고 측정을 완료로 세지 않는다.

## 해석·정보조건·실패 정책

Local D는 forecast origin의 전체 C 과거와 mask를 사용한 N-specific K×C 계수다. 입력 시점과 encoder는 같지만 decoder side information은 static D와 다르다. strict K-only 압축이나 표현학습 자체의 개선을 증명한 실험이 아니다. TUNE/DEV known past를 예측 시점에서 재적합하는 것은 명시하며, 해당 미래 label은 fit·threshold·checkpoint 선택에 쓰지 않는다. 기존 static D floor 및 split 전체의 공통 rank-K oracle floor는 local D의 하한으로 재사용하지 않는다.

한 데이터의 preflight·측정 실패는 그 fresh 결과와 reason을 보존하고 다른 데이터의 가능한 측정을 계속한다. 공유 F 자산·보호된 TEST·source SHA 같은 공통 gate 실패는 전체 FAILED_PRESERVED 사유다. 셀·샘플이 20/120에 못 미치면 PARTIAL_COMPLETED 또는 FAILED_PRESERVED로 표시하며, 실패한 데이터의 비용은 미측정이라고 남긴다. 기존 과학 결과·선택 lock·원 예측 배열은 before/after SHA로 검증한다. new whole 비용도 독립 감사 PASS 뒤에만 보고서에 확정 수치로 연결한다. 비용이 작아도 성능 개선·안정성·일반화·채택 근거로 바꾸지 않는다.

## 소스와 실행 방법

새 소스는 `run_local_past_decoder_cost_v1.py`와 `whole_local_past_decoder_cost_v1.py`이다. 합성 fixtures만으로 pending audit가 실제 import/실행 이전에 차단되는지, SHA·count·TEST 계약, exact rank 비교, 관측 mask, 실제 local LS가 매 callback 안에 있는지, 20/120 실제 저장·timer 산술·출력 parity를 검증한다. 기존 실행 소스는 수정하지 않는다.

루트의 source review와 테스트 이후 새로운 tag로 실행한다:

```text
python run_local_past_decoder_cost_v1.py --tag local_past_decoder_cost_v1 --local-diagnostic runs/local_past_decoder_v1/RESULT.json --local-audit audits/LOCAL_PAST_DECODER_PROBE_AUDIT_v1.json
```

실제 완료 이후 gate·source snapshot·physical precheck·20셀·120개 timer와 output 배열을 독립 검산한다. 이 계획 작성과 합성 통과는 실제 GPU 비용 측정이 아니다.
