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
python -m pip install -r requirements-lock.txt
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
