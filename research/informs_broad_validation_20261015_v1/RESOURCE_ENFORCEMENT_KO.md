# 단일 GPU 소유권과 누적 자원 집행

`resources.py`는 GPU·모델·데이터를 import하지 않는다. 새 캠페인의 private cache에만 lock, 활성 lease, 사용량과 receipt를 기록한다. 기존 외부 프로세스를 종료하지 않는다.

## 실행 API

```python
from resources import GPUReservation, check_storage_budget

with GPUReservation(cache, total_gpu_seconds=43200, reserve_seconds=3600,
                    wall_deadline=1792050350 + 172800,
                    purpose="training", kind="base_fit_queue") as lease:
    lease.check()
    # 반복 update/추론/하위 작업을 시작하기 전에 check한다.
    run_queue(deadline=lease.deadline)
```

`lease.deadline`은 campaign wall 종료와 GPU 잔여 process wall 중 더 이른 시각이다. `remaining`은 지금부터 그 시각까지의 초다. `purpose="training"`은 3,600초를 검산·비용 측정에 남긴다. 최종 검산은 `purpose="audit"`로 이 보류분을 사용할 수 있다. 두 목적 모두 기존 `GPU_USAGE.jsonl`의 probe·batching probe·base queue 사용량을 포함한 전체 43,200초 한도를 공유한다. 이 보류분은 새 예산이 아니다.

GPU 커널 시간 대신 **lease가 유지된 전체 process wall**을 보수적으로 계상한다. 모델 로딩, CPU 준비, child 대기와 마감 시 OS/server 정리도 포함한다. 동일 lease의 자식은 부모 시간 안에 포함되므로 따로 더하지 않는다. 외부 캠페인 GPU 작업을 이 private lock으로 차단할 수는 없다. 정밀 비용 실행 전의 다른 GPU 작업 확인은 실행기 책임이다.

`check()`와 deadline은 협력적 경계다. 이미 제출한 긴 CUDA 연산을 OS 수준에서 중단하지 않는다. 실행기는 update/배치/측정 전후마다 검사하고, 제한 직전에는 남은 시간보다 오래 걸릴 일을 시작하지 않아야 한다. 정상 종료 receipt의 `deadline_overrun_seconds`는 정리나 단일 연산이 마감을 넘겼는지도 숨기지 않는다. 자동 예산 증액이나 프로세스 강제 종료는 없다.

## 실제 OS lock

Windows에서는 `msvcrt.LK_NBLCK`로 파일 offset 0의 **정확히 1 byte**를 잠근다. POSIX에서는 `fcntl.flock(LOCK_EX|LOCK_NB)`를 사용한다. 경쟁자는 즉시 `ResourceBusy`를 받는다. PID 파일의 존재만으로 소유권을 인정하지 않는다. 활성 private `ACTIVE_GPU_LEASE.json`에는 PID, process create_time, lease ID, start/deadline, 비밀 token과 로컬 인증 endpoint가 있다.

OS lock 획득 후 이전 활성 기록을 발견하면 실제 PID와 create_time을 비교한다. 같은 실제 owner가 살아 있으면 재사용하지 않는다. owner가 죽었거나 PID가 재사용되었어도 기록된 인증 child가 살아 있으면 회수하지 않는다. 모두 종료된 이전 lease만 회수한다. crash 사용량은 `max(0, min(now, prior.deadline) - prior.started_unix)`를 `CRASH_RECOVERED_UNKNOWN_CONSERVATIVE`로 기록한다. GPU 활동의 실제 끝을 알 수 없다는 의미이며, 누락분을 0으로 처리하지 않는다.

정상/실패 종료는 `gpu_receipts/<lease_id>.json`을 exclusive create로 먼저 보존하고 journal을 append한다. journal append 전에 crash가 나더라도 같은 receipt를 재사용해 복구한다. 동일 lease ID는 한 번만 추가하며, 이미 journal에 중복 ID가 있으면 오류로 중단한다. 기존 journal 행을 수정하지 않는다. 공개용 receipt에는 token, private 경로, 명령행, 오류 메시지를 넣지 않는다. private 활성 기록과 child 환경변수는 공개하지 않는다.

## fresh child 비용·legacy 실행

```python
import os
import subprocess

with GPUReservation(cache, purpose="audit") as lease:
    environment = dict(os.environ, **lease.child_environment())
    subprocess.run(command, env=environment, check=True)
```

자식은 GPU import·모델 로드 전에 다음을 수행하며, 새 `GPUReservation`을 만들지 않는다.

```python
from resources import validate_gpu_child

with validate_gpu_child(cache) as child:
    child.check()
    run_inference(deadline=child.deadline)
```

검사는 (1) 실제 parent PID/create_time, (2) private 활성 lease의 일치, (3) OS lock의 실제 경합, (4) nonce를 포함한 token HMAC 인증 응답을 함께 요구한다. 응답 서버는 lock을 실제 보유한 부모 프로세스의 thread이며, child PID/create_time와 실제 조상 관계를 확인한다. 단순 환경변수나 잠긴 파일만으로 허용하지 않는다. 자식 인증마다 private 기록에 자식의 process identity를 남긴다. 부모는 인증된 모든 자식을 `wait`/`join`한 다음 lease를 닫아야 한다. 살아 있는 자식이 있으면 닫기를 오류로 처리하고 기록을 보존하여 다음 작업의 무단 점유를 막는다.

## 저장 공간과 다운로드

`check_storage_budget(cache, max_bytes=30*1024**3, pending_bytes=...)`는 새 private cache의 실제 파일 byte와 예상 다음 쓰기를 합산한다. symlink/junction을 따라 이전 자산을 훑지 않는다. 공개 캠페인 파일처럼 cache 밖에서 이번에 생성한 파일은 실행기가 그 byte를 `pending_bytes`에 더하거나 별도 합계 검사를 해야 한다. 저장·체크포인트 전에 검사한다. 파일을 생성하는 도중 OS가 디스크를 제한하는 quota는 아니다.

`record_download(cache, source_id, byte_count)`는 **다운로드 전** 전송량을 예약한다. 첫 호출에서 기존 `sources/*_download_receipt.json` 및 `source_evidence_receipt.json`의 `download_bytes_this_invocation`을 메타데이터만 읽어 `DOWNLOAD_BASELINE.json`에 동결한다. 이미 받은 데이터와 원문 확인용 metadata 다운로드도 5GiB에 포함된다. 이후 새 전송은 `DOWNLOAD_USAGE.jsonl` 예약을 합산하며 실패 예약도 환급하지 않는다. 같은 파일이 후속 receipt에도 등장해 기존 baseline을 다시 더하지 않는다. 초기 header/API 조사 보수분은 별도 source ID로 한 번 예약할 수 있다. 이후 전송을 예약 없이 수행하는 코드는 이 API의 보장을 받지 못한다.

## 실제 검증 범위

CPU toy의 최초 5개 시험이 구현 부재로 실패한 후 통과했다. 이어 고아 child 보호 및 과거 다운로드 합산 시험도 각각 실패를 확인하고 수정 후 통과했다. 최종 7개 시험은 별도 Python 프로세스를 실제 생성하여 다음을 확인한다.

- 부모가 보유한 OS lock과 자식의 실제 충돌 및 token 없는 종료 receipt.
- 올바른 살아 있는 조상의 child 인증, 잘못된 token과 종료된 owner의 거부.
- `os._exit(7)` crash 후 보수 계상, 같은 lease의 중복 복구 시 한 번만 charge.
- PID 재사용 오판 방지, GPU reserve 및 campaign wall 한도.
- 기존 usage 행 보존, 저장 공간과 다운로드 누적 상한.
- owner가 죽어도 실제 살아 있는 인증 child가 남아 있으면 reclaim 금지.
- 이미 받은 데이터·원문 metadata byte와 신규 전송 예약의 합산.

```text
python -X utf8 research/informs_broad_validation_20261015_v1/test_resources.py
Ran 7 tests in 0.985s
OK
```

이 CPU 검증은 GPU 모델·원데이터를 열거나 F를 실행하지 않았다. 단일 GPU 실험 queue의 실제 사용은 호출부 통합 후 별도 실행 receipt로 확인해야 한다.
