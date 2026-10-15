# Follow-up / deployment 소스 리뷰

작성: 2026-10-15. 소유 변경은 이 문서뿐이다. 실행 중인 R2 소스, `followup.py`, `deployment.py`, native/cost 모듈을 수정하지 않았다. 이 리뷰에서 실제 자료·F·GPU를 로드하거나 source를 import/실행하지 않았다. `[확인]`은 소스 호출과 차원·시간 좌표를 끝까지 추적했다는 뜻이며 실제 학습/비용/성능 검증을 뜻하지 않는다.

최종 재확인한 소스 SHA256:

```text
followup.py     b04422c80514df82b29948c29db8ebb44746301d47d5dbb6664615c23d73b089
deployment.py   3e8fcc9458ade07fe89a68fba9dc498350d43335ce485be647243c20bbe621d2
native_ridge.py be7d914f4131921e76675f0aad654c0f290d07dac31b3e3db6c6c39a1d89e0ea
cost_measure.py 73515b5ef176a8a09ddbef992d9d25b23b87e97cd45d6b854fddd12bbf899116
```

아래 수정은 이 리뷰 담당이 코드를 바꾼 것이 아니라 루트/비용 담당에게 전달한 뒤 소스에 반영된 상태를 다시 읽어 확인한 것이다. 이후 실행은 별도 동결 source snapshot과 대조해야 한다.

## 실행 전 전달했고 소스 반영을 확인한 문제

1. **[P1, 확인·반영됨] 비용 경로의 LoRA hook 잔류.** 최초 `cost_measure.execute`는 `del route`만 호출했다. hook은 공유 Bolt layer가 LoRA pair를 계속 참조하므로 Python route 삭제로 제거되지 않으며, 다음 Raw/Adapter와 반복 측정까지 delta가 누적될 수 있었다. 현재 `finally`는 `route.close()`를 호출하고 cleanup 예외도 ERROR trial로 보존한다. `Route.close`는 hook 제거·row_microbatch 복원·model CPU 이동을 수행한다. F의 parameter state hash만으로 hook 잔류를 검출할 수 없다는 점은 계속 유효하다. 이 수정의 실제 hook 제거 동작은 해당 담당의 별도 테스트/receipt 범위다.

2. **[P1, 확인·반영됨] 둘째 fold의 seed 선택에 C가 포함되는 문제.** 최초 secondary refit은 primary B[60%,80%) 점수로 고른 seed를 사용했다. secondary C가 [70%,100%)이므로 [70%,80%) 정답이 seed 선택에 들어갔다. 현재 secondary B0/B2/B4는 사전 seed `SEEDS[0]`로 고정한다. 가족 목록과 K는 이미 고정되어 있으며 secondary scaler/PCA/E/D/G의 적합 및 checkpoint 선택은 새 secondary A/B에서 이루어진다. 현재 흐름에서 이 primary-B→secondary-C seed 선택 경로는 제거됐다.

3. **[P1, 확인·반영됨] LoRA 생성/checkpoint 예외 cleanup.** 생성 중 OOM/assert 또는 checkpoint load 실패가 나면 이미 부착한 일부 hook이 남을 수 있었다. 현재 LoRA constructor는 예외 시 `close()`하고, `build`의 checkpoint 예외도 내부 LoRA를 찾아 닫는다. 정상 단일 LoRA fit이 끝날 때에도 `group_train`이 닫는다. 후속 driver가 group_train의 연장 예외를 프로세스 안에서 잡고 계속하는 새 흐름을 추가한다면 그 경계의 `finally` cleanup도 유지해야 한다.

4. **[P2, 확인·반영됨] G/직접 모델의 seed가 명세와 달랐던 문제.** 최초 G는 base adapter의 seed와 구조별 RNG 소비량을 물려받았고 DIRECT는 이전 route RNG 상태를 사용했다. 현재 DIRECT는 생성 전에 spec seed를 고정한다. G는 base를 생성·로드한 후 G 생성 직전에 G seed를 고정하므로 base 가족과 무관한 동일 G 초기화/동일 schedule 비교가 가능하다. LoRA도 build 시 명시 seed를 고정한다.

5. **[P2, 확인·반영됨] 비용의 active weights buffer 누락.** 최초 `active_parameter_bytes`는 parameters만 세어 Ridge의 buffer weights를 0 bytes로 기록했다. 현재 parameters와 buffers의 storage를 중복 제거해 합산한다. center와 native weights도 포함된다. 실제 peak allocated/reserved 측정과 이 회계값은 별개다.

6. **[P2, 확인·반영됨] Robin 가족 선정의 교차 자료 요약 기준.** 최초 주석은 relative-to-Raw였지만 실제로는 두 자료 B MSE의 절대 평균을 썼다. 현재 각 자료의 R1 A0 Raw B MSE로 나눈 비율을 동일 가중 평균한다. Raw 오차가 0에 가까운 특수 경우의 비율 해석은 계획의 절대차 기준을 따라야 하며 실제 값은 이 리뷰에서 확인하지 않았다.

7. **[P2, 확인·반영됨] Ridge 수치 미해결 상태 전달.** 현재 선택 solver의 `numeric_status` 또는 `numeric_unresolved_equations`가 미해결이면 결과 `training_status`에 `UNDERTRAINED_OR_UNRESOLVED`를 전달한다. 운영상 결과 저장/합법적 score의 `status=COMPLETE_VALID`와 수치적으로 충분히 해결된 최적화 여부를 분리한다. 보고서는 두 필드를 함께 읽고 미해결 LS를 최적 선형 대조가 충분히 풀렸다는 근거로 사용하면 안 된다.

## 요청한 주요 연결 검토

**group_train dict 중복키:** [확인] 현재 R3, R4 DIRECT/LoRA, R5 ablation/intermediate/secondary/stress의 호출자는 모두 새 spec을 만든다. 이 flat spec에는 `checkpoint/updates/selected_update/selected_B/training_status`가 없으므로 `dict(**spec, ...)`의 중복 인자 TypeError 경로는 현재 호출 흐름에 없다. base의 선택 metadata는 `base`라는 nested dict 안에 있으므로 flat key와 충돌하지 않는다. 다만 선택 결과인 `core_specs()`나 group_train 반환값을 그대로 재학습 spec으로 넘기는 API 사용은 지원되지 않는다. 그 경우 `cp.train` 자체의 `dict(**spec, updates=...)`에서 중복이 생길 수 있다. 후속 재시도 driver는 새 ID/새 훈련 spec을 만들어야 하며 기존 core 소스를 바꾸지 않는다.

**LoRA와 frozen F:** [확인] removable hook pair만 `LoRA.parts`에 등록되고 Bolt 원 parameter는 외부 frozen 모델에 남는다. `cp.train` optimizer가 보는 것은 pair의 down/up뿐이며 원 F weight가 섞이지 않는다. q/v 전체 linear를 대상으로 r8, scaling=16/8=2, dropout0, up=0으로 시작한다. LoRA가 붙은 F는 원 weight가 같아도 출력이 달라지는 별도 비교군이다. 정상 group_train은 단일 LoRA spec을 사용하므로 여러 후보의 hook을 동시에 붙인 채 비교하지 않는다. 294,912 parameter assertion은 고정 small 모델의 layer 구성에 의존하므로 다른 모델로 이식하면 그대로 사용할 수 없다.

**R4 native 차원과 원 C 목적:** [확인] F-swap은 잠재마다 공유하는 mean8/64/512−last 세 특징과 H48 계수, 총 3×48=144개의 실 계수를 사용한다. `swap_design`의 `[B,48,C,144]`, Gram `[144,144]`, rhs `[144]`, fitted weights `[3,48]`과 `swap_predict`의 feature-major flatten 순서가 맞는다. D는 `[7*r,C]`이고 같은 origin-reference lag 식을 사용한다. 적합 target은 `Y-current_C-offset`이므로 잠재 teacher가 아니라 실제 원 C 미래 목적이다. stream chunk 뒤 충분통계만 유지하여 전체 훈련 design을 모으지 않는다.

[확인] Raw Ridge는 feature `[B,C,33]`, Gram `[C,48,33,33]`, rhs `[C,48,33]`, weight `[C,33,48]`, `einsum('bcp,cph->bhc')`의 연결이 맞다. 33번째 current-centered feature는 항등 0이므로 nominal parameter 수와 유효 rank가 다르며 native module에 그 사실과 rank 진단이 명시되어 있다. 이 항목을 유효 33차원 정보라고 쓰면 안 된다.

**ZERO 후보의 의미:** [확인] Raw Ridge weights=0은 current persistence다. Swap weights=0은 latent 미래 head의 변화량이 0이라는 뜻이다. 학습한 D의 lag prefix/reference offset은 여전히 남아 있으므로 Swap ZERO가 항상 원 C persistence라는 뜻은 아니다. 보고서의 “no correction”은 어느 head/공간의 보정이 0인지 구분해야 한다.

**R3 G 정보/예산:** [확인] Raw, Current, B0-K, B2-K, B4-K의 G는 모두 동일 원 C history에서 현재값을 뺀 정보를 입력받고 같은 Residual 클래스/seed/schedule을 사용한다. base는 동결하며 base forward만 no_grad이고 G 학습 경로는 남는다. 두 자료 primary 5개씩 10 fit, secondary 5개씩 10 fit으로 G는 총 최대20 fit이다. extension은 같은 fit의 추가 update이며 별도 새로운 seed fit으로 세지 않는다. 실패로 생략한 경로는 완료 fit에 포함하면 안 된다.

**R4 구성 수:** [확인] native는 두 alpha×두 가족×두 자료=8회 계수 적합, DIRECT는 두 가족×두 자료=4 fit, LoRA 2 fit, Raw Ridge 2 fit로 최대16 fitting configuration이다. ZERO는 새로운 계수 적합이 아니다. source의 고정 `fitting_configurations=16`은 계획 수이며 실제 성공 수는 개별 result/failure에서 다시 집계해야 한다.

**C 점수/비용 순서:** [확인] followup의 alpha/checkpoint/가족/중간-r 선택은 A/B만 사용한다. 이 파일은 모든 route를 결정한 뒤 `SELECTION_LOCK.json`을 쓰며 C를 점수화하지 않는다. 비용보다 먼저 쓰는 이 파일은 **선택 잠금**이고 비용 완료 증명이 아니다. 루트가 별도로 작성 중인 C driver는 이 잠금과 비용 완료/오류 상태를 함께 확인한 뒤 C를 점수화해야 한다. 아직 작성 중인 그 driver의 enforcer가 이 두 파일에 없다는 사실은 알려진 범위이며, 이 리뷰를 C 실행 승인/완료 증거로 해석하지 않는다.

**물리 deployment 입력:** [확인] Route의 window-local 누적 관측 인덱스는 해당 past row 이후의 값을 채움에 사용하지 않는다. 초기 미관측은 A mean이고 표준화·GPU 전송·실제 경로 예측·CPU 반환·원 단위 복원을 수행한다. Adapter+G 비용 경로도 선택된 같은 base에 같은 G를 한 번 더한다. R1의 F 호출은 동기화된 nested timer로 누적한다. native/direct의 전체 wall은 포함되지만 현재 fallback branch는 native head/D 계산도 PE timer에 함께 넣는다. 따라서 native의 PE/DG stage 분해를 Adapter와 동일한 의미로 해석하지 말고, 전체 시간 또는 묶음 단계라는 점을 표시해야 한다.

최종 소스 판정: 두 P1 문제를 포함한 전달 항목들의 수정 반영을 확인했다. 현재 명시된 호출 흐름에서 즉시 실행을 막는 dict 중복키·R4 weight 차원 불일치 또는 신규 C 선택 누출 경로는 발견하지 못했다. 실제 hook lifecycle, 실제 Raw/선정 예측 parity, fitting 수, 수치 해결 상태, 전용 비용 측정과 최종 C 잠금은 별도 실행 receipt로 검증해야 한다.
