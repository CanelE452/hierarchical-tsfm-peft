# 코드·원수치·재현 조건

이번 게시에서 새 학습이나 추론을 실행하지 않았다. 실험 당시 실행 기록과 코드, 최신 보고서의 근거를 원문 그대로 게시하고, 별도의 파일·CSV·링크 검증과 작은 fixture 회귀 검사를 수행했다. `[확인]`은 저장된 실제 결과/검증 기록에 한정한다.

## 실행과 수학

- [자기 채널 masked ridge·pool·gate 수학](../readout_math_v1.py), [첫 SELF33/SELF512 비교 및 기존 JOINT33 검증 후 재사용](../run_readout_v1.py)
- [첫 FIT512 시도](../run_expanded_readout_v1.py), [수치 재현 실패 진단](../diagnose_dense_layout_v1.py), [C-contiguous canonical payload로 수선한 FIT512 실행](../run_expanded_readout_v2.py)
- [첫 whole 배포 비용](../cost_readout_v1.py), [수선된 FIT512 CPU 비용](../cost_expanded_readout_v1.py)
- [첫 원배열 독립 감사 v2](../audit_readout_v2.py), [FIT512 감사](../audit_expanded_readout_v1.py), [비용 감사 v2](../audit_cost_v2.py), [추가 비용 감사](../audit_expanded_cost_v1.py)
- [최신 보고서 생성기](../make_readout_report_v3.py), [수학 회귀 검사](../test_readout_math_v1.py), [수치 수선 회귀 검사](../test_expanded_readout_v2.py), [실제 JSON cohort 보고서 회귀 검사](../test_make_readout_report_v2.py)
- [실험 당시 승인·고정 시계·자원 제한](../adaptive_runtime_v1.py), [guarded runner](../run_guarded_v1.py)

현재 SELF/JOINT 비교는 현재값 anchor와 실제 원미래 innovation, FIT-only feature center/population scale, observed channel-macro MSE 손실을 쓴다. λ=αNc, 비규제 절편, PERSIST→공유 FIT 평균→7개 ridge pool의 첫 exact tie 규칙을 보존했다. SELF512와 SELF33의 시간 해상도 차이, JOINT의 차원·계수 용량 차이는 별도 진단이다.

첫 FIT512는 입력 값·저장 byte 해시가 같아도 array stride가 FP32 block mean의 합산 순서를 바꾸어 α0 예측이 기존 허용 오차를 벗어났다. 수정 실행은 원값·origin·mask·target·scaler·α·gate를 유지하고 payload를 fit 전부터 C 순서로 통일했다. 오차 한계를 완화하지 않았으며 실패 기록을 성공 결과에 합치지 않았다.

## 결과·증거

- [Robin 첫 producer](../runs/first_robin_v1/RESULT.json), [Beijing 첫 producer](../runs/first_beijing_v1/RESULT.json), [수선 전 FIT512 producer](../runs/expanded_robin_v1/RESULT.json), [유효 FIT512 producer](../runs/expanded_robin_v2/RESULT.json)
- [첫 배포 비용 producer](../cost_runs/first_readout24_v2/RESULT.json), [추가 CPU 비용 producer](../cost_runs/expanded_robin4_v1/RESULT.json)
- [최신 원 보고서 manifest](../reports/first_and_dense_v3/MANIFEST.json), [실제 보고서 파일 검산](../audits/FINAL_REPORT_ARTIFACT_AUDIT_v2.json), [종료 시 자원 관찰](../RESOURCE_SNAPSHOT_AFTER_REPORT_v3.json)

`runs/`, `cost_runs/`의 source snapshots와 `resources_v1/`의 종료 receipts도 파일 목록에 포함한다. NPZ·원자료·checkpoint는 포함하지 않아 Git clone만으로 새 실험 재현이 가능하다고 주장하지 않는다. 데이터 계약 및 로컬 자산 SHA는 JSON에 남겼다. 원 지시문·승인 시계가 고정되어 있으므로 그대로 실행하면 종료된 승인 범위를 재활성화할 수 없다. 새 실험을 원하는 경우 기존 파일을 수정하지 않고 새 승인/시계/출력 경로를 마련해야 한다.

## 게시 파일 자체 확인

다음은 실험을 재시작하지 않는 파일 검증이다. NPZ/모델 가중치/보호 TEST를 읽지 않는다. 작은 수학 fixture의 합성 array를 사용한 테스트는 게시 전 기록에서 연구 성능 실험과 구분한다.

```powershell
python research/bolt_compatible_adaptive_20261014/publication_main_20261014_v1/verify_snapshot.py
```

게시 후 실제 commit의 raw 파일 SHA와 GitHub Markdown 이미지 경로는 다음 도구로 확인한다. 외부 네트워크 읽기가 필요하다.

```powershell
python research/bolt_compatible_adaptive_20261014/publication_main_20261014_v1/verify_remote.py --commit <published-commit-sha>
```

[게시 전 검사 기록](PUBLICATION_CHECKS.json), [게시 파일 목록](FILES.json), [링크 검사](LINK_AUDIT.json)를 함께 읽는다. 게시 후 검증 receipt는 commit SHA를 대상으로 검사하므로 로컬 `.cache/bolt_compatible_adaptive_20261014/publication_main_20261014_v1/REMOTE_VERIFICATION.json`에 기록한다.
