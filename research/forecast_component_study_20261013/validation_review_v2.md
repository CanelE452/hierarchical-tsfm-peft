# Forecast Component Study 검산 메모

[확인] Stage A 저장 prediction 38개를 별도 코드로 다시 채점했다.
Stage A 판정: PASS. 실패 체크 수: 0.
Stage A2 판정: PASS. 실패 체크 수: 0.

## 핵심 확인

- 저장 NPZ의 prediction/target/observed/origins 형상과 VAL origin 연결을 확인했고, 물리 단위 MSE/MAE와 정규화 MSE/MAE를 재계산했다.
- RESULT, metrics.csv, prediction SHA, checkpoint SHA, source SHA, training record SHA 연결을 다시 확인했다.
- checkpoint 선택은 원래 head의 TUNE physical channel-macro MSE 최솟값과 일치한다. 전체 checkpoint VAL metric 표면은 저장되어 있지 않아 VAL로 전 checkpoint를 골랐다고 볼 근거는 없고, 선택된 checkpoint와 final600만 VAL 비교 대상으로 저장되어 있다.
- Robin의 원래 FORECAST 학습 origin은 seed별 2개이고, 공통 FIT readout은 256개다. 두 조건을 같은 표본수 실험으로 해석하면 안 된다.
- split origin은 각 split target bounds 안에 있고, contract에는 FIT normalization이 선언되어 있다. 보호 TEST 경로는 audit hook으로 차단했다.
- Stage A2는 sparse replay와 dense256의 batch 파일, checkpoint SHA, TUNE 최저 update 선택, selected/final VAL prediction 재채점을 별도로 확인했다.

## 주의

- 이 검산은 저장 산출물의 일관성 검산이다. Stage A가 독립 TEST 일반화나 Stage B 결합 가치를 입증한다는 뜻은 아니다.
- Stage B는 실제 파일이 아직 발견되지 않은 경우 스키마를 추정하지 않고 발견 목록만 기록한다.

## Stage A2/Stage B 발견 상태

```json
{
  "stage_a2*": [
    {
      "path": "research\\forecast_component_study_20261013\\stage_a2_dense_pool_v1",
      "is_dir": true,
      "children": [
        "RESULT.json",
        "jena_s92601_dense256.json",
        "jena_s92601_sparse_replay.json",
        "jena_s92602_dense256.json",
        "jena_s92602_sparse_replay.json",
        "robin_s92601_dense256.json",
        "robin_s92601_sparse_replay.json",
        "robin_s92602_dense256.json",
        "robin_s92602_sparse_replay.json"
      ],
      "sha256": null
    },
    {
      "path": "research\\forecast_component_study_20261013\\STAGE_A2_PROTOCOL.md",
      "is_dir": false,
      "children": null,
      "sha256": "8cdc43b21770393d50bc2f2b733257ef721369fdbf9c9faa728b04244b128381"
    }
  ],
  "stage_b*": [
    {
      "path": "research\\forecast_component_study_20261013\\stage_b_preflight_v1",
      "is_dir": true,
      "children": [
        "RESULT.json",
        "jena_contract.json",
        "robin_contract.json"
      ],
      "sha256": null
    },
    {
      "path": "research\\forecast_component_study_20261013\\stage_b_resources",
      "is_dir": true,
      "children": [
        "preflight_v1_73416_1791876141267478900.json"
      ],
      "sha256": null
    }
  ],
  "component*": [
    {
      "path": "research\\forecast_component_study_20261013\\component_experiment.py",
      "is_dir": false,
      "children": null,
      "sha256": "ff4cf8c9ceeaec280a6af69e8c4e586685457030feece790faba464afd282d03"
    }
  ]
}
```
