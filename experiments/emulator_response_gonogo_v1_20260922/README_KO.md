# 예측 변화 보존 대리모델 PEFT — CLI 번들

`MASTER_CLI.txt`가 실행 계약이다. 이미 있는 Jena gap-screen 코드를 재사용해 실행한다.

## 제공 파일

- `MASTER_CLI.txt`: 자료·비교군·수식·예산·선택·판정·보고서/push 계약.
- `config_proposal.json`: 새 실험의 명시적인 고정값. 기존 프로젝트 config를 덮어쓰지 않는다.
- `reference_core.py`: activation-aware factorization, 실 factorized linear, additive LoRA,
  raw 변화분 학습, 예측 반응 loss, scaling을 포함한 EMLoC-style correction, 비용/판정 부품.
- `test_reference_core.py`: 25개 CPU unit tests.
- `CPU_TEST_LOG.txt`, `LOCAL_VALIDATION.json`: 제작 환경에서 실제로 실행한 검사 범위.
- `BUNDLE_MANIFEST.json`: 파일 SHA-256 목록.

## CLI가 실행할 검사

```bash
python -m unittest discover -s <번들 경로> -p 'test_reference_core.py' -v
```

그 뒤 기존 `gap_screen`의 native 모델 로딩·data panel·loss/score·journal·보고서 코드를
새 experiment 경로에서 재사용해 `MASTER_CLI.txt`의 6군을 구현한다.
본 번들은 완성된 `run_all.py`가 아니다. 원래 모델과 대리모델의 실제 연결은 CLI가 검증해야 한다.

## 여기에서 검증한 것

CPU 작은 텐서에서 압축 산술, 초기 예측 동일성, gradient, raw/native 변환,
LoRA correction 선형 항등식, 준비비 합산, 판정 논리를 검사했다.
실제 Jena 다운로드·Chronos/PEFT·CUDA·EMLoC BF16 코드와의 parity·학습시간·성능은 검사하지 않았다.
가상의 GPU 실행시간이나 성능을 파일에 넣지 않았다.

## 주제 경계

EMLoC 자체의 최초 제안이나 재현 논문을 목표로 하지 않는다.
새로 시험하는 것은 출발 예측 보존 + 최종 예측의 작은 적응 반응 보존의 추가 가치다.
기존 방법만으로 충분하면 `GO_STANDARD_ONLY`다.
현재 결과가 나쁘다고 즉석에서 메모리 연구로 목표를 바꾸거나 Jena 외 자료를 추가하지 않는다.
