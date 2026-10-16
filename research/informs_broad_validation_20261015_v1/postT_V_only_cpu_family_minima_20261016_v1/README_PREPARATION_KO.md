# 원 필수 family 비교 4조건의 새 CPU 실행 준비

현재는 새 실행기와 합성 검사 소스를 준비하는 단계다. 이 문서·검사 작성 담당자는 실제 테스트, metadata CLI, compile, Task·모델·배열 로딩, probe, 학습, 추론, 자식 프로세스 실행을 수행하지 않았다. **테스트 상태는 NOT_RUN이며, 준비 코드의 존재를 probe 성공·학습 완료·성능 개선으로 해석하지 않는다.** ROOT가 정밀 CPU 비용 측정의 종료와 기존 작업의 정확한 PID/create_time closure를 확인한 뒤 소스 검토·검사·실제 입장을 수행한다.

소유 소스는 [실행기](family_minima_runner.py), [stdlib 합성 검사](test_family_minima_runner.py)다. 합성 검사9개를 정의했으며 이 담당자는 실행하지 않았다. 별도 PLAN/REGISTRY의 실제 봉인과 과학 실행은 ROOT가 담당한다. 원 campaign의 source39, registry, 선택 잠금, 원 점수·비용·T 결과와 기존 partial run을 수정하거나 재개하지 않는다. T를 관찰한 이후 원 미완료 비교를 보강하는 V-only 개발이므로 독립적인 새 일반화 검증으로 부르지 않는다.

## 정확한 네 조건과 원 recipe

원 `runs/<원ID>/SOURCE_CONTRACT.json`에 있는 spec을 복제하고, 새 run ID, CPU device, 원 등록의 전체 search 일정 `0.25 → 1 → 3`, post-T stage/provenance만 명시적으로 바꾼다. 새 손실·encoder·데이터·분할·seed를 만들지 않는다.

- Jena B4 K6: 원 `primary_jena_E1_B4_k6_s92601` → 새 `postT_cpu_jena_E1_B4_k6_s92601_family_v1`. 실제 spec family는 `B4`, lr0.001이다.
- Jena AdaPTS port K6: 원 `primary_jena_E1_ADAPTS_BOLT_PORT_k6_s92601` → 새 `postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1`. 표시명과 달리 원 실행 spec family는 `ADAPTS`, lr0.001이다.
- Jena LoRA rank8: 원 `primary_jena_E1_LORA_R8_k0_s92601` → 새 `postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1`. ID의 k0는 등록 라벨이며 **원 spec family는 `LORA`, 실제 `latent_k=21`, F 입력 폭은 원 C21, lr0.0001**이다. 채널 압축 K0 또는 F 호출0으로 계산하면 안 된다.
- Beijing B4 K11: 원 `primary_beijing_E1_B4_k11_s92601` → 새 `postT_cpu_beijing_E1_B4_k11_s92601_family_v1`. 실제 spec family는 `B4`, lr0.001이며 K=C11이다.

공통 recipe는 seed92601, primary fold, L512/H48, microbatch4/effective16, F row microbatch128, `checkpoint_inputs=False`, AdamW/weight_decay0이다. LoRA rank8/alpha16도 원 spec 그대로다. AdaPTS는 저장소의 `AdaPTSLinearPort`이며, 원 논문의 모든 설정을 재현한 것처럼 표현하지 않는다.

## 데이터와 학습 경계

원 PCA PLAN `b4207ede…`의 원 데이터 계약 전체와 search/refit scaler SHA를 재사용한다. Jena는 shape20965×21, A50 끝10482와 legal origins9923, A60 끝12579와 legal origins12020이다. Beijing은 shape21038×11, A50 끝10519와 legal origins9960, A60 끝12622와 legal origins12063이다. 두 자료 모두 원 V_select의 horizon-stride origin43개를 유지한다. columns·배열 fingerprint·data SHA·scaler SHA·foundation revision·실제 normalization source SHA도 고정한다.

원 `fit_base.fit_base`는 A50/A_stop에서 0/0.25/1/3 search를 마친 뒤 A_stop으로 epoch를 고르고, 새 A60 model을 만들어 선택된 epoch만 refit한다. **검색3epoch 완료와 선택된 refit epoch는 별개**다. selected epoch0이면 실제 refit update0이 유효하며, 선택3epoch라는 이유만으로 수렴을 주장하지 않는다. 원 `training_status`도 그대로 보존한다.

기존 run/checkpoint를 이어 학습하지 않는다. 새 namespace의 partial 결과·claim·prediction도 덮거나 재개하지 않는다. 다른 조건의 실패·미완료는 그 조건 상태로 남기며 독립적으로 가능한 조건을 막는 근거로 삼지 않는다.

## 실제 probe의 두 gradient 계약

각 조건의 fit 전에 같은 조건의 실제3 effective-update probe가 필요하다. 전체 A50 legal population을 손실 정규화에 그대로 쓰고, probe 후 model·optimizer·hook을 폐기한다. probe 상태를 fresh fit에 재사용하지 않는다. probe 성공은 장시간 CPU 완주·RSS·수렴을 보장하지 않는다.

기대 F stream은 `16 × updates × effective_F_width`, physical calls는 `4 × updates × ceil(4 × effective_F_width / 128)`이다. 여기서 폭은 Jena B4/AdaPTS6, Jena LoRA21, Beijing B4 11이다. 따라서 3update/48sample probe의 기대 streams는 각각288/288/1008/528, 기대 calls는 각12다. 이 수치는 설계 기대값이며 실제 측정값이 아니다.

B4와 AdaPTS는 frozen foundation 모듈의 `register_full_backward_hook`에서 `grad_input[0]`만 기록한다. 원 module 경계를 통과한 `[4,512,K]` 이벤트 `4 × updates`개, 모두 finite, 적어도 하나 nonzero가 필요하며 scope는 `F_module_grad_input_only`다. decoder도 잠재 표현을 직접 쓸 수 있으므로 입력 tensor의 gradient나 `requires_grad=True`만으로 F 경유 backward를 증명하지 않는다. F parameter gradient 없음과 F state SHA 불변은 별도다. hook은 gradient를 변경하지 않고 제거해야 하며 실제 fit에는 probe hook을 붙이지 않는다.

LoRA는 frozen 원 입력이므로 F 입력 gradient를 같은 기준으로 요구하지 않는다. 대신 원 q/v rank8 adapter의 실제 gradient가 존재하고 모두 finite이며 적어도 하나 nonzero인지 확인한다. 원 backbone parameter gradient 없음, adapter가 아닌 base weight SHA 불변, `close()` 이후 원 weight/grad flags 복원도 확인해야 한다. 원 `LoRAQvAdapter.close()`의 hook 제거·원 weight 복원·base hash 검사 계약을 유지한다. 두 gradient 경로를 하나의 통과 플래그로 대체하지 않는다.

## 봉인·자원·실행 순서

기본 명령은 stdlib metadata만 출력하며 과학 모듈·Task·모델·배열을 읽거나 파일을 생성하지 않아야 한다.

```powershell
python -B family_minima_runner.py
```

실행에는 ROOT가 봉인한 별도 PLAN/REGISTRY, 해당 PLAN SHA, 원 public/cache/asset roots, 정확한 단일 새 ID, live deadline, fresh output을 명시한다. 봉인 전의 예시 SHA나 가상 성공 영수증으로 실제 fit을 허용하지 않는다. 원 `SOURCE_CONTRACT` 네 개와 source39·immutable9·원 PCA PLAN·required runner `e11c3491…`·진단 utility `36c75b38…`의 정확한 bytes를 전후 검증한다. source39에는 JSON/MD도 있으므로 39개 Python 파일로 바꾸지 않는다.

자원 계약은 CPU2, 각 RSS3GiB, 외부 최대2worker·총6GiB, 원 public+cache30GiB와256MiB 입장 여유 및 물리 디스크 여유, 원 deadline1792222550이다. 실제 CPU2/offline/CUDA 비활성 환경을 사용한다. 자체 RSS/deadline watchdog은 외부 process-tree supervisor의 PID/create_time closure 증거를 대신하지 않는다. watchdog의 `os._exit(98)`에서는 최종 `RESULT.json` 없이 `STARTED.json`·`RESOURCE_STOP.json`만 남을 수 있으므로 ROOT가 실제 종료와 보존된 증거를 별도로 보고해야 한다. 기존 CPU 학습·추론과 정밀 비용 측정이 모두 끝난 quiet window를 ROOT가 확인해야 한다. 이 문서와 테스트 작성 담당자는 현재 외부2slot의 실제 상태, prior queue closure, 비용 종료를 확인하지 않았다.

실행 순서는 ROOT 소스 검토와 합성 검사 → metadata 후보 봉인 → 실제 per-condition probe와 guard closure → 같은 조건의 fresh search/refit → full V43 생성이다. 실행기 API에서 원 BASE receipt·stage receipts·곡선·선택 artifact·V manifest/statistics를 새 namespace에 보존한다. full V의 기대 F streams는 Jena B4/AdaPTS258, Jena LoRA903, Beijing B4 473이다. cached V를 실제 새 forward로 세지 않는다.

## 완료 후에도 남는 검증

학습과 full V 생성만으로 독립 검산 완료를 주장하지 않는다. ROOT는 같은 parent/scaler/source/model/array binding으로 원 FP64 full-V auditor와 고정6-origin CPU checkpoint replay를 별도로 실행하고 실제 실행 여부·공차·F counters를 확인해야 한다. 이 준비 문서와 합성 검사는 실제 점수나 비용을 생산하지 않는다.

CPU 결과와 기존 CUDA 비용을 섞어 속도 우위를 주장하지 않는다. 원 registry 상태, 원 T selection, 기존 GPU-label coverage를 자동으로 바꾸지 않는다. 원 safeFIT loader가 파일 내부 T bytes를 물리적으로 역직렬화할 수 있다는 사실과, 새 T target/scoring을 요청하지 않는 계산 범위를 구분한다. 보호된 T 평가, 새 선택, 원 파일 덮기, 자동 commit/push는 이 실행 준비의 범위가 아니다.
