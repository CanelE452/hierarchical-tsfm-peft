# 우선순위 변경에 따른 기존 큐 중단

2026-10-16 사용자 지시로 기존 큰 큐를 먼저 중단했다. 이 문서는 새 핵심 비교의 완료 보고가 아니다. 방법론 판단을 보고하기 전에는 기존 전체 큐를 다시 실행하지 않는다.

## 실제 중단과 보존

[확인] 16:15 KST에 실제 scheduler PID54524의 생성 시각·명령·작업 폴더와 현재 자손을 대사했다. 학습은 Beijing shared-K와 Max-K 두 조건이었고, 나머지 13개 fresh fit은 시작되지 않았다. 고정 nested PCA 3건은 이미 V·FP64 재채점·CPU6 재추론을 마친 상태로 보존했다.

새 작업 배정을 먼저 막기 위해 해당 scheduler만 일시 정지했다. 이후 각 worker의 콘솔에 연결된 프로세스가 확인된 해당 worker·launcher와 중단 도구뿐임을 확인하고 인터럽트를 보냈다. 두 학습 worker와 소유 launcher/console은 종료됐다. 정지된 scheduler는 다시 실행시키지 않고 종료했다. 16:17 KST에 확인한 7개 소유 PID/생성 시각은 모두 종료 상태였고, 기존 실행 세션도 exit1로 끝났다. 다른 프로젝트·현재 CLI·프로세스 이름을 기준으로 종료하지 않았다.

원 checkpoint 10개는 ZIP CRC와 SHA를 확인한 뒤 별도 private 디렉터리에 바이트 그대로 복사했다. 중단 후 원본과 사본 모두 SHA가 같았다. 이후 실제 저장 구조도 확인했다. 모델·AdamW·sampler 순서/위치/노출 및 전용 NumPy RNG·Torch RNG·contract·best state·완료 epoch 정보가 있다. 해당 학습에는 LR scheduler와 AMP가 사용되지 않았다. Python/global NumPy RNG는 별도 저장되지 않는다.

```text
run_id                                        최신 search   search updates/exposures   최신 refit
postT_cpu_beijing_E4_ORDERED_MULTI_K_k11_s92601_v1  epoch3       1868 / 29880              epoch0
postT_cpu_beijing_E4_MAXK_PCA_TRAINED_k11_s92601_v1 epoch3       1868 / 29880              epoch0
```

각 search의 선택 best epoch도 3이었다. 이는 A_stop 선택이며 V 성능이나 수렴의 증거가 아니다. 각 run의 `search/checkpoint_000003.pt`와 `refit/checkpoint_000000.pt`가 마지막 정상 저장점이다. 원 cache 안의 `postT_V_only_cpu_multik_20261016_v1/runs/<run_id>/`에 보존했다. 별도 사본과 중단 증거는 private `priority_pause_validate_20261016_v1/`에 있다. 원자료·체크포인트·명령행 전체와 개인 절대경로는 공개하지 않는다.

## 복구 가능한 범위와 손실

[확인] 현재 실행되던 코드는 다음 optimizer update에서 저장하고 종료하는 협조적 중단을 지원하지 않았다. 인터럽트가 새 checkpoint를 만들지 않았다. 마지막 refit 저장점은 0 update다. 저장 뒤 경과는 shared 약51분, Max-K 약8분이지만, 이 시간을 정확한 학습량으로 환산하지 않는다. 저장 뒤 update/exposure 기록이 없어 미저장 진행량은 **미확정**이다.

중단 순간 그대로의 재개는 불가능하다. 마지막 정상 저장점의 상태 형식은 남아 있으나 실제 재개 실행은 아직 검증하지 않았다. 기존 `multik_runner.py --mode fit`은 fresh-only 경로라 기존 run/claim을 거부하며, 같은 명령을 다시 실행하는 것은 재개가 아니다. 후순위에서 필요한 조건만 선정한 뒤 source/config/data/scaler/모델/장치·sampler를 대사하는 명시 재개 진입점을 사용해야 한다. 그 진입점이 준비되기 전에는 자동 재개하지 않는다.

기존 54조건 비용 연결 도구는 private 미검증 초안 한 파일만 남아 있고, CONFIG·CLAIM·실행 worker·자동 재시작은 생성되지 않았다. 전체56·모든 K·shared/max-K·새 그래프·대규모 재포장은 핵심 판단 보고 이후의 후순위다. 기존 TEST 잠금과 원 예산은 그대로다.

## 핵심 검증으로 전환

완료된 Jena B0/B2/lag, 정확한 Raw/Ridge/CPU fresh G를 우선 재사용한다. PCA는 K16 대신 원 K6 자산을 사용한다. 기존 primary LoRA는 .25epoch 제한 학습임을 표시한다. 첫 새 비교는 R4의 동일 P/E/D를 고정한 전체512×6 no-Bolt 시간 선형 head이며, 이후 같은 구조를 별도 학습한 R5b를 구분해 확인한다. 실제 실행·검산·비용·판정은 별도 결과 보고서에 반영한다.

사용한 Windows 인터럽트의 대상 범위는 [Microsoft의 콘솔 신호 설명](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent)과 [콘솔 연결 프로세스 목록](https://learn.microsoft.com/en-us/windows/console/getconsoleprocesslist)을 따랐다. 콘솔 신호 자체를 자동 checkpoint 보장으로 해석하지 않았다.
