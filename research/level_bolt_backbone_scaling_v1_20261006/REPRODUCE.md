# 실행·재사용 안내

이 결과는 기존 Robin v6, Jena v7, Peacock Education v12의 선택된 LEVEL 두 seed와 Small F0 예측을 재사용합니다. Mini/Tiny만 고정 원점에서 새로 추론했고 새 적합은 0회입니다. 원자료·부모 checkpoint·예측 배열·공식 모델 가중치는 Git에서 제외하며 각 manifest의 경로와 SHA256으로 연결합니다. 공개 저장소만으로 모든 비공개 캐시가 제공되는 것은 아닙니다.

기존 프로젝트 환경과 원본 캐시를 사용하는 실행 순서는 아래와 같습니다. 완료된 TEST를 다시 평가하는 명령으로 사용하지 않습니다. `predict`는 완료 receipt를 확인해 재사용하고, 완료된 `score`와 `cost`는 덮어쓰기를 거부합니다. 저장 점수의 재검산은 `verify`로 수행합니다.

```text
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/prepare.py
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/preflight.py
# evaluation_seal.json: 실제 구현·모델·자료·재사용·preflight를 TEST 전에 봉인
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/evaluate_controls.py predict
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/evaluate_controls.py score
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/evaluate_controls.py verify
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/cost_controls.py --device cuda
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/report_controls.py
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/report_layout_fix.py
.venv/Scripts/python.exe research/level_bolt_backbone_scaling_v1_20261006/verify_controls.py
```

실제 실행 상태·자원은 `ledger.json`, 단계 종료는 `evaluation_attempts/`, 원점별 예측 바인딩은 `prediction_receipts/`, 비용 원행은 `cost_rows/cuda/`에 있습니다. 준비 path 오류와 수정 전 manifest도 보존합니다. 봉인 후 TEST 설정을 바꾸지 않았는지와 원본 보호는 `final_checks.json`의 실제 체크 결과를 따릅니다.

이전 Chronos-2 비용과 이번 값을 나누지 않습니다. 네 arm의 동일 CPU 입력→전체 온라인 추론→H48C CPU 출력 비용을 이번 회차에서 측정했습니다. 통계 구간은 고정 checkpoint에 조건부이며, 자체 검산은 독립 재현이나 논문 신규성의 인증이 아닙니다.
