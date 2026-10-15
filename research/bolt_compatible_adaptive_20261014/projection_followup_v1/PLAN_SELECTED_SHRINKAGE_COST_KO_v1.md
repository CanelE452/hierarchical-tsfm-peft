# 선택된 과거 shrinkage 디코더의 실제 전체 Bolt 비용

[판단] 진행 중인 decoder 안정성 진단의 비용 검증이다. 새로운 개별 사용자 승인 발언이 있었다고 주장하지 않는다. 사용자의 지속 수행과 누적 자원 상한 해제를 따르며 기존 승인·gate·실패 기록은 보존한다. 새 데이터/K, TEST, 설치, E/F/head 학습, 과거 job 종료·재개, Git 게시는 포함하지 않는다. 현재 소스와 합성 fixtures 준비 단계이며 실제 비용 실행을 마쳤다는 문서가 아니다.

## 고정된 조건과 lock

Robin C17/K4, Beijing C11/K2, 과거512/미래48, 기존 FIT scaler·origin·frozen E/F·currentC anchor를 그대로 사용한다. 별도 shrinkage CPU producer COMPLETE와 독립 PASS를 정확한 SHA로 확인한다. 모든 선행 source JSON·source code SHA와 네 selection lock의 경로/SHA를 scientific import·GPU 로드 전에 검증한다.

각 데이터×E 공간의 기존 Bolt TUNE 정규화 macro MSE가 선택한 양수ρ만 사용한다. ρ는 lock JSON에서 읽고, 해당 lock의 기존 7후보·tie 순서·선정 수치가 독립 감사와 일치하는지 검증한다. 비용·DEV·native·oracle 수치로 ρ를 다시 고르지 않는다. 원 TUNE 정답은 이미 정상 hyperparameter 선택에 사용됐다. 미래 정답을 전혀 안 썼다는 주장을 하지 않는다. D fitting은 각 origin의 관측 과거만 사용한다.

다섯 경로는 RAW_BOLT, PCA_PINV_BOLT, FIT_TARGET_PINV_BOLT, PCA_SHRUNK_BOLT, FIT_TARGET_SHRUNK_BOLT다. 두 데이터×B1/B8로 계획20셀, 셀당 warmup2/측정6으로120시간 표본이다. 동일한 첫B개 origin을 모든 경로·반복에 사용한다. 다섯 circular rotation만 위치 균형이고 여섯 번째 seeded permutation은 추가 반복이다. 6회 전체가 완전 Latin 균형이라고 하지 않는다.

## 실제 계산과 타이머

기존 `whole_local_past_decoder_cost_v1.py`를 새 파일로 복사한 타이머 경계를 유지한다. causal physical fill 전 타이머 시작 → FIT normalize → 원 NumPy E → shrinkage D와 구현된 진단/hash 계산 → H2D/Bolt/D2H → D와 currentC anchor → 물리 단위 CPU 반환과 최종 synchronization까지 포함한다. 매 callback에서 `fit_shrunk_local_decoder(history32, true_observed_mask, NumPyZ32, stored_static_D_cast32, locked_rho)`를 실제 호출한다. 원 static prior의 FP32 cast를 고정하고 매 origin/channel마다 `lambda=rho*trace(observed_centered_Z.T@Z/n)/K`로 구한다. 기존 unpenalized analytic intercept·trace0 prior 유지·관측0 prior/current-intercept 규칙과 함수 내 condition·stationarity·norm·reconstruction·input hash 진단을 모두 보존한다.

D를 타이머 밖에서 cache하지 않고 진단 코드를 비용 절감 목적으로 제거하지 않는다. production 최소 비용으로 해석하지 않는다. 이 단계는 prior로 향하는 ridge이며 구현된 fit input binding hashes는 타이머 안에 포함한다. 모델/데이터/state 로드, preflight, warmup, 출력 저장/SHA, source/file/state metadata hashes generated outside timed callback, guard·memory snapshot은 타이머 밖이다. 따라서 모든 hash를 제외한 비용이라는 주장이 아니다. native head를 로드하거나 측정하지 않는다. 메모리는 shared resident F가 있는 process RSS/peak VRAM이고 방법의 독점 비용이나 VRAM 절감이라는 주장이 아니다. 이전 절대 latency와 비교해 가속을 주장하지 않는다. 다른 CPU 활동이 전혀 없었다고 보장하지 않으며 짧은 반복 기술통계의 불확실성을 유지한다.

## 수치 gate와 실패 보존

physical first8은 두 batch를 모두 포함한다. 원 origin·mask를 exact로 확인하고 history/currentC/E·계수·bias·actualλ·covariance trace·ridge condition·stationarity를 같은 normalized atol1e-5/rtol2e-6로 비교한다. static prior 배포 cast SHA는 exact다. 관측 수, zero-trace prior 유지, 관측0 prior 유지, past valid, condition valid와 λ>0 solver 결정은 exact여야 한다. 최종 출력 허용오차만으로 내부 결정을 통과시키지 않는다.

각 B1/B8의 실제 F 연결 예측은 원 선택된 `BOLT_NUMPY` candidate artifact의 BOLT_prediction과 비교한다. static 대조는 같은 CPU 조건의 보존된 원 BOLT_static_prediction과 비교한다. normalized atol1e-5/rtol2e-6와 물리 단위 atol2e-3/rtol2e-5를 모두 요구한다. 공차·ρ·λ·prior를 결과에 맞춰 바꾸지 않는다. 물리 전처리 rounding이나 F 수치 차이가 solver/gain에 증폭되면 해당 실패와 fresh artifact를 남긴다.

한 데이터의 preflight/측정 실패는 그 partial receipt·saved sample·reason을 보존하고 다른 데이터의 가능한 측정을 계속한다. 공통 F/source/보호 gate 오류는 전체 FAILED_PRESERVED다. saved samples와 completed-cell samples를 분리하며 20/120 미완료를 COMPLETE로 쓰지 않는다. 원 source·input·cache·checkpoint·selection lock·새 lock의 before/after SHA와 model/config/F 서명을 확인한다. root가 actual CLI를 한 번 실행하고 새 independent cost PASS 이후에만 보고서에 비용을 확정 연결한다.

이 decoder는 원 C의 관측 과거를 K×C 계수로 추가 전달한다. static D와 decoder side information이 같거나 strict K-only 압축인 비교가 아니다. 원 static/global-rankK oracle 하한을 새로운 per-origin D에 적용하지 않는다. shrinkage의 상대λ는 절대 gain이나 미래 성능을 보장하지 않는다. 이전 native는 static D에서 학습한 고정 head 개입이며 새 matched-head benchmark는 별도 단계다. 비용 개선·실행 성공이 표현학습·일반화·전체 목표 해결을 뜻하지 않는다.

## 실행

root의 source review와 합성 테스트 이후, 기존 결과를 덮지 않는 fresh tag로 실행한다.

```text
python run_selected_shrinkage_cost_v1.py --tag selected_shrinkage_cost_v1 --shrinkage-diagnostic runs/local_decoder_shrinkage_v1/RESULT.json --shrinkage-audit audits/LOCAL_DECODER_SHRINKAGE_AUDIT_v1.json
```

새 timer/controller/plan은 실행 전에 SHA로 동결한다. 합성 검증은 actual 자료·모델·NPZ/F/GPU를 실행했다는 근거가 아니다.
