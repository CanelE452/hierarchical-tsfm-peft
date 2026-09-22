# Internal-adaptation gap screen

같은 긴 과거를 사용하는 **출력 보정 / LoRA / LoRA+**를 ETTm2와 Jena에서 독립 비교하는 실행 코드다. 새로운 PEFT 방법을 제안하거나 예전 STOP 후보를 재개하지 않는다. 자세한 목적·범위·선택·비용 정의는 [PROTOCOL_KO.md](PROTOCOL_KO.md)와 [config.json](config.json)에 있다.

## 빠른 실행

번들의 `execute.py`가 설치 → CPU 테스트 → 자료 → GPU 검사 → 본학습 → 예측 저장 → MD·PNG·CSV → 검산 → 선택적 push를 순서대로 실행한다.

```bash
python /path/to/unpacked_bundle/execute.py --repo /path/to/tsfm-peft-method-screen --publish
```

따옴표가 필요한 Windows 경로에서도 Python 스크립트를 그대로 사용할 수 있다. 새 private repo를 만드는 작업이 아니다. 기존 repo에 **새 experiment/results/cache 경로만** 추가한다.

설치한 코드에서 직접 실행할 수도 있다.

```bash
python experiments/internal_adaptation_gap_v1_20260922/run.py all --repo .
python experiments/internal_adaptation_gap_v1_20260922/run.py verify --repo . --device cpu
python experiments/internal_adaptation_gap_v1_20260922/publish.py --repo .
```

하나씩 실행하려면 `--panel ettm2` 또는 `--panel jena`. 두 자료가 모두 끝나야 전체검산 PASS와 publish가 허용된다. 부분 fit은 자동 재실행하지 않으며, 종료된 fit만 재사용한다.

## 환경

이미 작동하는 Python 3.11 / CUDA 환경을 우선 사용한다. `requirements.txt`는 호환 범위이며 **이 번들 제작 환경에서 실제 설치 검증한 lock이 아니다.** 실제 사용 버전과 `pip freeze`는 실행 때 저장한다. 기존 CUDA torch를 무조건 업그레이드하지 않는다. 다른 프로젝트 패키지·드라이버·시스템 전역 설정을 바꾸지 않는다.

필요한 패키지가 없으면 같은 Python 3.11의 새 venv를 만들고, 장치에 맞는 CUDA PyTorch를 먼저 준비한 뒤 requirements를 설치한다. 실험기에서 실제 `torch.cuda.is_available()`와 BF16 지원을 검사한다. CUDA가 없으면 본학습을 CPU로 조용히 돌리지 않는다.

Chronos-2 모델은 pinned HF revision으로 다운로드한다. 이미 있는 실제 snapshot을 사용하려면 `--model-dir /real/snapshot`을 명시한다. 이때 실제 가중치 fingerprint를 저장하지만 local override가 원격 pinned 파일과 완전히 같다고 자동 보증하지는 않는다.

자료를 이미 받았다면 실제 CSV 경로만 전달한다.

```bash
python execute.py --repo /repo --ettm2-file /data/ETTm2.csv --jena-file /data/mpi_roof_2024.csv --publish
```

CSV가 아닌 Jena ZIP은 자동 다운로드 경로에서 풀어 읽는다. 컬럼·시간간격이 예상과 다르면 설정을 조용히 바꾸지 않고 BLOCK한다.

## 자동 실모델 사전검사

공식 forward/손실과 local encode wrapper, 초기 HEAD=F0, 미래 target 교체 불변성, origin별 group isolation, trainable 실제개수, LoRA+ 공식 optimizer 파라미터별 LR mapping, 실제 gradient·파라미터 변화, frozen hash, 저장·복원을 확인한다. 제작 환경의 CPU tests는 이를 대신하지 않는다.

## 무엇이 생성되나

`results/internal_adaptation_gap_v1_20260922/SUMMARY_KO.md`

`ettm2/REPORT_KO.md`, `jena/REPORT_KO.md` 안에는 각5개 figure와 데이터·선택·cost·seed·계열별 결과가 들어간다. `VERIFICATION.json`은 실제 로컬 가중치·원시 예측까지 재검산하지만, GitHub에는 그 대용량 원시물 대신 hash와 작은 점수표를 올린다. GitHub 보고서만으로 모든 local NPZ를 다시 채점할 수 있다고 주장하지 않는다.

`publish.py`는 먼저 CPU artifact 검산을 다시 수행하고, 올바른 remote/branch와 다른 staged 또는 unpushed 변경이 없는지 확인한다. 강제 push·자동 merge/rebase는 하지 않는다. 성공 후 실제 artifact commit과 보고서 URL을 출력한다. 자체 commit의 해시를 자기 안에 넣는 순환 receipt는 만들지 않는다.

## BLOCKED·음성 결과

과학적 음성 결과도 정상 완료 보고서로 push한다. 자료/실행 오류는 `ERROR.json`과 통합 INCOMPLETE에 남고, 완료로 위장한 publish는 차단된다. 일부 fit이 애매하게 중단됐다면 같은 key로 다시 학습해 덮어쓰지 않는다. CLI는 error 위치를 사용자에게 보고하고 임의 rescue하지 않는다.

## 제작 환경의 테스트 범위

`tests/`는 CPU 작은 tensor와 모의 backbone으로 계약·metric·scheduler·최적화 그룹·전체 저장/보고서 경로를 검증한다. 모의 예측은 **실제 TSFM 결과가 아니며**, 실제 결과로 게시하는 것은 차단된다. 실제 Chronos 모델/데이터/CUDA 검증은 사용자의 실행 환경에서 수행한다.
