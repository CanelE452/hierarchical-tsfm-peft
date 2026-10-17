# Adapter 진단 결과

[확인] SCIENTIFIC_PARTIAL. ENGINEERING_COMPLETE는 실제 전체 검증·게시·원격 SHA 확인 이후에만 판정한다.
모든 정확도 결과는 반복 사용한 개발 V다. 기존 결과 재사용과 이번 신규 실행/재채점은 RUNS_FULL_PRECISION의 origin 열로 구분한다.
상태 수: COMPLETE=106, FAILED=1, SKIPPED_BUDGET=2.

현재 수집된 Jena 최소 MSE는 Ridge `0.5566463357048567`, MAE `0.40552180175406594`이다. 서로 다른 기능·용량·학습량의 결과를 포함하므로 순수 구조 효과로 읽지 않는다.
정확도 허용폭·실용 속도 목표는 UNREGISTERED다. 2seed는 각각 보이며 범위 이상의 일반화·수렴·계층적 모든 K-prefix를 입증하지 않는다.

[전체 job 상태](JOB_STATUS.csv) · [정확도](RUNS_FULL_PRECISION.csv) · [공통 update](SAME_UPDATE_RESULTS.csv) · [같은 endpoint 대조](SAME_UPDATE_COMPARISONS.csv) · [계약 확인 비교](MATCHED_COMPARISONS.csv) · [제한 있는 맥락 비교](CONTEXTUAL_COMPARISONS.csv) · [비용](COST_FULL_PRECISION.csv) · [질문별 판단](QUESTIONS_AND_DECISIONS_KO.md) · [감사](VALIDATION.json)

M5: 로컬 파일의 헤더 schema·SHA를 확인했다. [M5_LOCAL_ASSET_DISCOVERY.json](M5_LOCAL_ASSET_DISCOVERY.json)과 [M5_LOCAL_ASSETS_KO.md](M5_LOCAL_ASSETS_KO.md)에 근거와 범위를 보존한다. 숫자 행 파싱=0, fit/evaluation=NOT_RUN, C/L/H·split=NOT_REGISTERED다. 성능·자료 품질·단위·첫 사용·사전학습 미노출을 확인한 결과가 아니다. 원 감사의 2026-10-16 HTTP 401·등록 inventory 부재 기록은 과거 접근 기록으로 보존하며, 현재 전체 PC의 파일 부재나 라이선스 거부를 뜻하지 않는다.

[확인] Jena cpu B1 실측 warm ms/origin: Raw=139.44280007854104 (FP32 core; FP64 inverse; CPU return); B2=48.65384986624122 (FP32 core; FP64 inverse; CPU return); Ridge=0.2991999499499798 (FP64 Ridge core; FP32 normalization; FP64 inverse; CPU return). 장치·dtype·scope가 같은 행만 공정 비교하며, B1을 다른 batch로 일반화하지 않는다.

[확인] Jena cuda B1 실측 warm ms/origin: Raw=12.843200005590916 (FP32 core; FP64 inverse; CPU return); B2=22.455150028690696 (FP32 core; FP64 inverse; CPU return). 장치·dtype·scope가 같은 행만 공정 비교하며, B1을 다른 batch로 일반화하지 않는다.

미완료: C01_pems32_B0=SKIPPED_BUDGET (Actual disposable profiles exceed complete pair budget/reserve); C01_pems32_B2=SKIPPED_BUDGET (Actual disposable profiles exceed complete pair budget/reserve); E04_jena_lora=FAILED (ValueError)

실패 시도는 그대로 보존하며 교정 실행으로 해결했다: E04_jena_lora → E04_jena_lora_parent_v2 COMPLETE. T3의 superseded_attempts와 원 receipt를 함께 확인한다.

[원격 main 보고서](https://github.com/CanelE452/hierarchical-tsfm-peft/tree/main/reports/adapter_diagnostics_20261017_v2). 이번 추가 산출물의 main 포함 여부는 정상 push 후 별도 원격 SHA 조회로 확인한다.

[T1 정확도·비용](PPT_T1_ACCURACY_COST.csv) · [T2 메커니즘](PPT_T2_MECHANISM_ABLATION.csv) · [T3 근거·미완료](PPT_T3_COVERAGE_CLAIMS.csv) · [T4 자료·프로토콜](PPT_T4_DATASET_PROTOCOL.csv)

[자료 감사 보완](DATA_AUDIT_SUPPLEMENT_KO.md) · [자료 출처·단위](DATA_SOURCE_DETAILS.json) · [horizon 구간 요약](HORIZON_BAND_SUMMARY.csv) · [paired-origin 결속](PAIRED_ORIGIN_BINDINGS.json)

![accuracy_cost](figures/accuracy_cost.png)

![learning_curves](figures/learning_curves.png)

![horizon_delta](figures/horizon_delta.png)

![channel_delta](figures/channel_delta.png)

![backbone_comparison](figures/backbone_comparison.png)

![qualitative_forecasts](figures/qualitative_forecasts.png)
