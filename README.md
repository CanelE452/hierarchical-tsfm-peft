# Hierarchical TSFM PEFT screen

동일 Chronos-Bolt-small + LoRA에서 실제 hierarchy 관계의 추가 예측 가치를 평가합니다.
유일한 실행 계약은 [MASTER](docs/00_MASTER_CLI_hierarchical_tsfm_peft_new_pc_20260921.txt)입니다.
실행 기록은 `results/hier_peft_screen_v1/`에 저장합니다. 논문 PASS를 주장하지 않습니다.

## Data

```python
from datasetsforecast.hierarchical import HierarchicalData
Y, S, tags = HierarchicalData.load("./data", "Labour")
Y2, S2, tags2 = HierarchicalData.load("./data", "TourismLarge")
```

공식 source: https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip
Raw data, model weights, checkpoints는 Git에 포함하지 않습니다.

## Model and reconciliation

- Model: `amazon/chronos-bolt-small`
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Nixtla: https://github.com/Nixtla/hierarchicalforecast
- Context 48개월, horizon 12개월, CALIBRATION으로만 MinT-shrink 추정.
- VALIDATION으로 LR/checkpoint 선택, TEST로 선택 금지.

## Setup and execution

Python 3.11로 새 `.venv`를 만들고 GPU에 맞는 공식 PyTorch CUDA wheel을 설치합니다.
검증된 `requirements-lock.txt`로 정확한 의존성을 재현합니다.
현재 환경은 Python 3.11.16, PyTorch 2.10.0+cu128 / CUDA 12.8, RTX 4070입니다.
환경의 출처·패키지 wheel 해시는 `ENVIRONMENT.json`에 있습니다.

```text
python -c "from pathlib import Path; Path('.cache').mkdir(exist_ok=True)"
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --report .cache/torch-install.json
python -m pip install -r requirements-lock.txt --report .cache/dependencies-install.json
python -m pip install --no-deps -e .
python -m hier_peft.env --verify
python -m hier_peft.data
python -c "from hier_peft.chronos_wrapper import download_model; download_model()"
python experiments/hier_peft_screen_v1/run_preflight.py
python experiments/hier_peft_screen_v1/run_train.py --stage A
python experiments/hier_peft_screen_v1/finalize.py
```

Preflight 미통과 시 학습을 실행하지 않습니다. Stage B는 Labour continuation rule 통과 시에만 허용합니다.
Labour gate가 통과한 경우 finalize 전에 동일 CLI의 `--stage B`를 실행합니다.
이미 main update가 존재하면 자동 재실행을 거부하여 fit/update 예산을 보호합니다.
새 머신에서는 기존 실행 결과를 덮어쓰지 않는 별도 workspace/results가 필요합니다.

## 완료된 Labour screen

`STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN`: 16 fits / 8,192 main updates를 완료했습니다.
HIER는 SELF·POOL 대비 두 repeat seed에서 일관된 추가 가치를 보이지 않아 TourismLarge 학습은 실행하지 않았습니다.

- [한국어 보고서](results/hier_peft_screen_v1/REPORT_KO.md)
- [최종 결정](results/hier_peft_screen_v1/FINAL_DECISION.md)
- [Seed effects](results/hier_peft_screen_v1/SEED_EFFECTS.csv)
- [Verification](results/hier_peft_screen_v1/VERIFICATION.json)

실행 후 JSON boolean 직렬화 문제만 수정했습니다. 실행 당시 source seal과 후처리 수정 내역을 모두 보존했습니다.

## 승인된 채널 압축 잔차 PEFT 개발 (2026-09-26)

동결 TSFM의 입력 채널 압축에서 남은 재구성 잔차를 작은 시간 예측기로 보완하는 방법을 개발 중입니다. 위 hierarchy screen의 단발 계약과 별도로 승인된 연구이며, 아직 주제를 확보한 상태는 아닙니다.

- [승인된 계획 원문](research/tsfm_peft_development_20260926/PLAN.md)
- [현재 진행 상태와 다음 판단](research/tsfm_peft_development_20260926/STATUS.md)
- [재현 명령과 검산 범위](research/tsfm_peft_development_20260926/commands.json)
- [최초 16fit 개발 비교](research/tsfm_peft_development_20260926/initial_development.json) 및 [교정된 평가 비용](research/tsfm_peft_development_20260926/initial_cost_recheck_development.json)
- [잔차 예측 rank의 제한적 CPU 진단](research/tsfm_peft_development_20260926/residual_rank_diagnostics.json)
- [학습량 보완 후 개발 비교](research/tsfm_peft_development_20260926/extended_development.json)

학습량을 보완한 뒤에도 잔차 경로는 압축만 한 대조와 원입력 보정보다 나았지만, 강화한 선형 대조 대비 이득은 1.30%로 작고 기간에 따라 달랐습니다. CPU 용량 진단을 근거로 잔차 rank만 8에서 32로 늘린 동일 용량 대조를 진행 중이며, 주제 확보를 선언하지 않았습니다.

Electricity의 기존 노출 구간은 개발에만 사용합니다. BDG2 Bull Office의 보호 구간은 최종 설정을 확정하기 전까지 평가하지 않습니다. 원자료·가중치·예측 배열은 Git에 포함하지 않습니다.
