# 최종 저장 결과 감사 준비 v2

[확인] 이 버전은 준비만 마쳤다. AST 2개와 합성 metadata 분기 테스트 15개가 통과했으며, 기본 준비 CLI도 exit0이었다. 기존 453개 manifest 재독, 배열·체크포인트·gzip 해제, 실제 예측·재채점, GPU·T 실행은 모두 0이다. 합성 fixture는 실제 결과의 성능이나 checkpoint replay를 검증하지 않는다.

기존 `diagnostics/whole_available_audit_1792092300000/discovery_wrapper.py`와 453개 감사 결과는 수정하지 않았다. v2는 해당 wrapper의 고정 SHA를 확인한 뒤 정확한 사후 M4 발견 분리 context를 그대로 사용한다. `audit_campaign.audit_all`의 채점·registry·legacy 코드는 변경하지 않는다.

실제 실행 시 원본 453개(V405 + 원래 dense M48), 정확히 SHA가 고정된 제외 M4, source39 및 registry 과학 조건을 유지해야 한다. 추가 manifest는 다음 두 종류만 허용한다.

- 실제 `SELECTION_LOCK.json`의 선택 ID에 속하는 공식 `route_predictions/<ID>/T_final/PREDICTION_MANIFEST.json`. lock fingerprint·원 소스·registry·route·data·scaler가 같아야 한다.
- `legacy_e0/evaluation/<legacy ID>/cp<update>/PREDICTION_MANIFEST.json`의 historical B. 봉인 `e0_b_execution_ready_20261016T012806`의 정확한 9ID/55CP 중 하나여야 하며 공식 legacy schema와 모델 파일/state hash·경로·old source·통계·foundation·기록된 B curve가 같아야 한다. 새 V/T 증거로 세지 않는다.

legacy는 campaign의 `cache_relative_path` 필드를 전제하지 않는다. 실제 writer와 원 감사 흐름에 따라 해당 폴더의 `REFERENCE_CONTRACT.json`, `PUBLIC_ERROR_STATS.csv.gz`, `INDEPENDENT_RESCORE.json`, 비공개 같은 상대 폴더의 `predictions.npz`를 결박한다. 봉인된 asset/cache checkpoint와 원 metadata도 byte hash로 보존한다. reference 내용·배열 seal·FP64 수치·공개 통계·독립 영수증/curve 검증은 원 감사기가 그대로 수행한다. checkpoint tensor를 로드하거나 모델을 재실행하지 않는다.

실제 추가 T 수를 t, 실제 historical B 수를 b라고 할 때 원 report의 관계를 다음처럼 확인한다.

```text
manifests_found                    453 + t + b
prediction_rescoring_passed         453 + t
public_stats_reaggregation_passed   405 + t
dense_private_rescoring_passed       48
legacy_B_rescoring_passed            b
global_scores_checked               true
```

historical B의 자체 공개 통계 검사는 legacy 감사 안에 있으며 위 405+t에 합치지 않는다. 선택된 T나 봉인 B가 일부 없으면 누락 ID/CP와 별도 coverage=false를 기록한다. 저장되어 있는 결과의 감사 PASS를 T/E0 전체 완료·수렴·캠페인 완료·독립 일반화로 바꾸어 말하지 않는다.

기본 명령은 작은 봉인/소스 준비 검사만 수행한다. `$R`은 공개 연구 루트, `$C`는 비공개 캐시, `$A`는 기존 asset 프로젝트 루트, `$PY`는 해당 Python이다. 같은 의미의 환경 변수 `INFORMS_PUBLIC_ROOT`, `INFORMS_CACHE_ROOT`, `INFORMS_ASSET_ROOT`도 지원한다. 출력은 항상 새 폴더로 지정한다.

```powershell
& $PY -X utf8 -u -B "$R/diagnostics/post_final_audit_v2_1792093500000/discovery_wrapper_v2.py" --public-root $R --output "$R/diagnostics/새_준비_폴더"
```

ROOT가 최종 T와 E0의 실제 실행 종료 및 CPU 감사 시점을 확인한 다음에만 아래를 명시 실행한다. 이 준비 작업은 실제 실행을 예약하거나 자동 시작하지 않는다. 현재 정밀 GPU 비용 측정 중에는 실행하면 안 된다.

```powershell
& $PY -X utf8 -u -B "$R/diagnostics/post_final_audit_v2_1792093500000/discovery_wrapper_v2.py" --public-root $R --private-root $C --asset-root $A --output "$R/diagnostics/새_최종감사_폴더" --execute
```

실제 감사는 CUDA=-1/계산 스레드2이며 새 GPU lease·추론·학습을 하지 않는다. 실제 전역 SCORES/selection lock가 없으면 실패하고, 허용되지 않은 추가 manifest나 봉인 변경은 무시하지 않는다. 원 감사기 출력과 wrapper 입력/출력 영수증을 함께 보존해야 하며, 최종 공개 전 경로 검사도 별도로 필요하다.
