# 실제 실행 계약과 명령

이번 범위는 B2 outer RevIN의 Jena·PeMS32 두 seed 비교, 조건부 동일 그룹의 6 epoch 확인, Jena 구성 대조 두 seed, 최종 모델의 비용 측정이다. 첨부된 combined 지시문은 기술 참고자료이며, 사용자가 승인한 최소 실행안이 실행 범위를 결정한다. `CAMPAIGN_CONTRACT.json`과 변경하지 않은 `IMPLEMENTATION_PLAN_KO.md`가 등록 계약이다.

## 실행 순서

실제 실행에는 기존 원자료, frozen F 가중치, 정확히 대응하는 OFF checkpoint·예측, private campaign 장부가 필요하다. 이 자산들은 공개 payload에 포함하지 않는다. `ASSET_ROOT`는 이 자산들이 있는 기존 프로젝트 폴더이며, 작업은 별도 관리 worktree에서 수행했다. Python3.11 및 라이브러리 버전은 `ENVIRONMENT.json`과 `AUXILIARY_SOURCE_PINS.json`을 따른다.

```powershell
python -m research.b2_revin_20261018_v1.run audit --asset-root <ASSET_ROOT>
python -m research.b2_revin_20261018_v1.run plan --asset-root <ASSET_ROOT>
python -u -m research.b2_revin_20261018_v1.run execute --asset-root <ASSET_ROOT>
python -m research.b2_revin_20261018_v1.run verify --asset-root <ASSET_ROOT>
python -u -m research.b2_revin_20261018_v1.run costs --asset-root <ASSET_ROOT>
python -m research.b2_revin_20261018_v1.run report --asset-root <ASSET_ROOT>
```

기본 인자 없는 실행은 도움말만 출력한다. 이 명령들은 private 자산과 기존 장부를 가진 승인 실행 환경의 순서를 기록한다. 새 환경에서 공개 파일만으로 학습이나 기존 optimizer continuation을 재현했다는 뜻은 아니다. 공개 집계 재생성은 `REPRODUCE_KO.md`의 별도 명령을 따른다.

## 선택과 중단 범위

- 최초 checkpoint는 기존 OFF와 같은 0/.25/1/3 epoch다. epoch2를 삽입해 update 경계를 바꾸지 않는다.
- A_stop 마지막 checkpoint가 엄격한 최저값인 B2 arm이 있으면 그 데이터의 OFF/ON×두 seed를 모두 4/5/6 epoch까지 이어 간다. terminal model·optimizer·sampler·CPU/CUDA RNG를 유지한다.
- 모든 신규 search와 데이터별 공통 정규화를 먼저 봉인한다. `SELECTIONS_SEALED_BEFORE_NEW_V.json` 이후 선택 epoch에서 fresh A60 refit하고 V를 평가한다.
- Jena B0는 최종 B2의 공통 정규화와 상한에 맞춘 두 seed 대조다. B0가 상한에서 개선 중이면 구성 결론의 학습량 한계로 남긴다.
- 6 epoch에서도 개선 중이면 수렴 미확정으로 표시한다. 추가 학습을 자동으로 등록하지 않는다.
- 반복 사용한 V는 개발 평가다. T 접근과 독립 테스트 주장은 포함하지 않는다.

## 측정과 검증

검산은 저장 예측을 원채널 macro MSE/MAE와 FP64 충분통계로 다시 계산하고, target·mask·origin·채널 순서 및 선택/refit 출처를 연결한다. 비용 측정은 실제 저장 모델의 재로딩 예측을 전체 V와 먼저 대사한 뒤, raw+mask부터 CPU 출력까지 GPU batch1/8 경로를 측정한다. Ridge의 기존 CPU 비용은 별도 조건으로 표시한다. 실패나 미상 값을 정상 결과로 채우지 않는다.

기존 기준선의 `cap=0`은 이번 신규 학습 상한에 속하지 않는 역사적 reference 표식이다. 실제 기존 LoRA와 DirectG의 선택 epoch는 각각 1과 3이며, `selected_epoch`와 원 출처를 읽어야 한다. `REUSED_VERIFIED` OFF 행에 남아 있는 학습 시간은 과거 시간이다. 신규 search 시간 합산은 `source_kind=NEW_TRAINED` 행의 `new_stage_search_seconds`만 사용하고, cap6의 상속된 cap3 시간도 신규 사용량으로 중복 합산하지 않는다.

모델 실행·비용 측정·전체 프로젝트 테스트는 한 번에 하나의 GPU lease를 사용한다. 실제 사용량은 `RESOURCE_LEDGER.json`, 실제 종료·검산·테스트·재생성 상태는 각 검증 JSON을 따른다. 기존 봉인 자산의 byte SHA와 원래 dirty checkout의 Git 상태는 별도로 대사한다. 후자의 검사는 사용자 dirty 파일 전체 내용의 SHA 검증을 의미하지 않는다.

새 source와 집계·문서·그림만 명시 allowlist로 정상 main에 반영한다. 실제 원격 확인은 `REMOTE_PROOF.json` 및 최종 전체 allowlist Git blob 대사로 수행하며, 코드 검토만으로 게시 완료를 선언하지 않는다.
