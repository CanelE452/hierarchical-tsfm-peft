# 원 감사기 + 정확한 사후 M4 발견 분리

이 wrapper는 `audit_campaign.audit_all`의 실제 채점·registry·legacy 검사를 변경하지 않는다. 공개 루트 자체의 `rglob("PREDICTION_MANIFEST.json")` 호출에서 SHA가 고정된 사후 감사용 M4만 분리한다. 다른 패턴·하위 루트·경로는 그대로 탐색한다. 원 소스 39개와 wrapper SHA는 별도 기록한다.

[확인] 준비 검사는 전체 457개 중 원본 453개(V405, 원래 dense M48)와 사후 계약 M4를 구분했다. M4의 독립 8개 감사 근거는 `diagnostics/raw_deep_g_saved_finalize_1792091230000/RESULT.json`이다. 최초 8개 수치 검산 PASS 이후 후처리 exit1과 별도 readback exit0은 그대로 보존되어 있다. 이번 발견 범위 분리는 그 원본이나 registry를 수정하지 않는다.

환경 루트 `$R`은 공개 연구 디렉터리, `$C`는 기존 비공개 캐시, `$PY`는 프로젝트 Python이다. 새 출력 폴더를 지정한다.

```powershell
& $PY -X utf8 -u -B "$R/diagnostics/whole_available_audit_1792092300000/discovery_wrapper.py" --public-root $R --private-root $C --output "$R/diagnostics/새_감사_출력" --execute
```

`--execute`가 없으면 metadata만 읽으며 채점하지 않는다. 실제 실행은 CUDA=-1, 계산 스레드 2로 원 감사기를 1회 호출하고 원본/입력/registry 과학 필드 보존을 확인한다. 원래 auditor의 coverage incomplete는 캠페인 완료로 바꾸어 말하지 않는다. checkpoint 재실행·재학습·모델 추론·TEST 접근은 0이다.

최종 T 저장 출력과 실제 전역 SCORES/SELECTION_LOCK가 생긴 뒤에는 새 출력 폴더와 `--expect-global-scores`를 함께 지정한다. 기존 원본 453개와 정확한 제외 4개의 SHA는 유지하고, 추가 manifest는 locked ID의 실제 T_final만 허용한다. 이는 미래 T를 생성하거나 전체 T 완료를 보장하는 기능이 아니다.

준비 명령: metadata exit0, scope checker 최종 exit0. scope checker 첫 호출은 필수 `--public-root`를 누락해 argparse exit1로 종료했으며 채점·배열 로드는 없었다. 수정 후 실행은 다른 패턴/하위 루트 비간섭, 원 탐색 복원, 잘못된 제외 SHA 거부를 확인했다. 실제 FP64 실행은 이 문서 작성 시 아직 하지 않았다.

현재 wrapper의 후속 사용 한계: postmode는 기존 453개와 실제 추가 T_final만 지원하며, 모든 대상의 표준 REFERENCE_CONTRACT를 전제로 한다. 이후 E0 legacy-B manifest가 추가되면 이 wrapper를 그대로 재사용할 수 없다. 해당 경우 원 감사기의 legacy 예외 처리를 보존하는 별도 후속 wrapper 또는 ROOT의 직접 호출 범위를 다시 정해야 한다. 이번 pre-final 453개 감사에는 legacy manifest가 없으므로 이 한계가 적용되지 않는다.
