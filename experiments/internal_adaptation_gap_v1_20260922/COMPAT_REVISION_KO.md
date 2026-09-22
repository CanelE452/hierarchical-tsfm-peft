# 실행 환경 호환 revision r1 (TRAIN 전, 2026-09-22)

기준 번들: `internal_adaptation_gap_cli_bundle_20260922.zip` (sha256 `770c4828c6af69d5214439a5ef00b89432e9c6ea5f088eca2f6267dd3a47f6e9`). `install_into_repo.py`의 manifest 해시 검증을 통과한 뒤 설치했다. 이 문서는 설치 뒤, 환경 봉인(SOURCE_SEAL)과 모든 학습보다 먼저 작성했다.

실험 계약(`config.json`, `PROTOCOL_KO.md`)의 설정·검사·assert·허용오차·예산·seed·학습률·rank는 바꾸지 않았다. `config.json`은 번들과 바이트 단위로 같다.

## 1. 실행 저장소 변경 (사용자 지시)

- 번들의 원래 대상은 `CanelE452/tsfm-peft-method-screen`(commit `9c692e7`)이다. 사용자가 2026-09-22에 `CanelE452/hierarchical-tsfm-peft`에서 실행하라고 지시했다. 설치 시점의 HEAD는 `ab5ef0d5392c62d38beb6c2839192b0227c83e66`이다.
- `config.json`의 `reference_repo`와 `reference_commit`은 원래 대상을 가리키는 기록값이다. 코드는 이 값을 읽지 않는다. 계약 파일을 보존하려고 고치지 않았다. 실제 실행 저장소와 HEAD는 `ENVIRONMENT.json`의 `existing_head`에 기록된다.
- `publish.py` 두 줄을 바꿨다. remote 허용 패턴을 `CanelE452/tsfm-peft-method-screen`에서 `CanelE452/hierarchical-tsfm-peft`로 바꿨고, `PUSH_RECEIPT.json`의 보고서 URL도 같은 저장소로 바꿨다. 다른 게시 안전장치(검산 재실행, staged 파일, 선행 unpushed commit, 원격 선행, 확장자·크기·secret 검사)는 그대로다.

## 2. GPU 점유 검사의 Windows WDDM 거짓 양성

- 증상: Windows WDDM에서는 `nvidia-smi --query-compute-apps`가 `dwm.exe`, `explorer.exe` 같은 데스크톱 그래픽 클라이언트(프로세스 표의 Type `C+G`)까지 모두 반환한다. 원래 검사는 자기 PID가 아닌 행이 하나라도 있으면 `GPU_BUSY`로 차단한다. 따라서 화면이 이 GPU에 연결된 이 PC에서는 다른 GPU 작업이 없어도 항상 차단된다.
- 수정 위치: `gap_screen/runner.py`의 `environment()`. `os.name=='nt'`일 때만, `nvidia-smi` 프로세스 표에서 Type이 `C`(순수 compute)인 PID만 다른 GPU 작업으로 센다. Linux의 동작은 원래와 같다. 차단할 때의 메시지와 "종료하지 않고 무기한 기다리지 않는다"는 동작도 원래와 같다.
- 실측(2026-09-22, RTX 4070, driver 595.79): 유휴 상태에서 원래 목록은 38행이었고 모두 `C+G`라서 필터 후 0행이었다. 실제 torch CUDA 자식 프로세스를 띄우면 그 프로세스가 Type `C`로 잡혔다. 즉 필터를 거쳐도 실제 다른 학습 작업은 여전히 차단된다.
- 한계: 데스크톱 앱(브라우저·메신저 등)은 GPU를 간헐적으로 쓸 수 있다. 이것은 차단 대상이 아니다. 그 영향은 계약의 타이밍 CV 공개와 `QUALITY_GAP_TIMING_UNCERTAIN` 규칙으로 드러난다.

## 3. 코드 수정 없이 둔 실행 환경 설정

- 모든 단계(pytest·run.py·publish.py)를 같은 Python으로 실행한다: `E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe` (3.11.16, torch 2.10.0+cu128). 이 저장소의 직전 실험이 쓴 환경이다.
- 환경변수 `PYTHONUTF8=1`과 `PYTHONIOENCODING=utf-8`을 설정한다. 이유: 제공된 `tests/test_workflow.py`가 `REPORT_KO.md`(UTF-8)를 인코딩 지정 없이 `read_text()`로 읽는다. 한국어 Windows의 기본 인코딩은 cp949라서 `UnicodeDecodeError`가 난다(36개 중 1개 실패, 실측). 테스트와 assert는 고치지 않았다.
- 패키지 범위 이탈: 설치된 `peft 0.21.0`은 `requirements.txt` 범위 `peft>=0.18,<0.21` 밖이다. 동작이 확인된 기존 환경을 우선하라는 계약에 따라 업그레이드나 다운그레이드를 하지 않았다. LoRA+ 공식 그룹 검사는 설치된 이 peft의 `create_loraplus_optimizer`와 비교한다. `transformers 5.17.0`, `huggingface_hub 1.32.0`은 범위 안이다.

## 4. 자료와 모델 출처

- ETTm2: `--ettm2-file E:/CODING/proj/mltimeseries/data/ETT-small/ETTm2.csv` (sha256 `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd`). 2026-09-22에 공식 URL(`config.json`의 `source_urls.ettm2`)에서 새로 받은 파일과 sha256이 같다.
- Jena 2024: 2026-09-22에 `config.json`의 공식 URL(`https://weather.bgc-jena.mpg.de/mpi_roof_2024.zip`)과 이전 경로(`https://www.bgc-jena.mpg.de/wetter/mpi_roof_2024.zip`)가 모두 HTTP 404였다. 그래서 `--jena-file E:/CODING/proj/mltimeseries/data/jena_mpi_roof/mpi_roof_2024.csv` (sha256 `65a47db98ad85b0600f2deab1e11055af8fef9c5aedceadbdcd0011748120558`)를 명시했다. 이 CSV는 같은 폴더의 공식 archive `_zip/mpi_roof_2024.zip` (sha256 `3ce424b9d1671ebc26c09921c7d4f9444282acb1e8a5f020d90a156287ba83aa`, 2,583,427 bytes)에 들어 있는 유일한 파일과 sha256이 같다. 이 archive의 해시는 `mltimeseries/data/oa_resolution_pilot_v1_source_hashes.json`에 공식 URL 다운로드로 기록돼 있다. 오늘 원격에서 다시 받아 비교하지는 못했다.
- 모델: `--model-dir`를 쓰지 않았다. `amazon/chronos-2` revision `29ec3766d36d6f73f0696f85560a422f50e8498c`를 Hugging Face hub 경로로 로드한다. 이 revision의 snapshot은 로컬 HF 캐시에 이미 있다. 실제 가중치 fingerprint는 `MODEL_RECEIPT.json`에 저장된다.

## 5. 실행 경로

수정한 설치 코드는 번들 payload와 다르다. 그래서 번들의 `execute.py`를 다시 실행하면 installer가 `existing different file`로 멈춘다. 계약 §4의 설치 코드 직접 실행 경로를 같은 순서로 따른다. 순서는 pytest → `run.py all` → `VERIFICATION.json` PASS 확인 → `publish.py`다.
