# 실제 실행·검산·재생성 명령

Python은 기존 FP32/CUDA 환경에서 실행한다. ASSET_ROOT는 데이터·F snapshot·기존 부모를 보관한 로컬 checkout으로 설정한다. 공개 결과에 private 절대경로를 넣지 않는다.

```powershell
$env:ASSET_ROOT = "<local-asset-root>"
python -m research.raw_context_multiscale_20261018_v1.run audit --asset-root "$env:ASSET_ROOT"
python -m research.raw_context_multiscale_20261018_v1.run smoke --asset-root "$env:ASSET_ROOT" --device cpu
python -m research.raw_context_multiscale_20261018_v1.run execute --asset-root "$env:ASSET_ROOT" --plan reports/raw_context_multiscale_20261018_v1/EXPERIMENT_PLAN_REPAIR_002.json
python -m research.raw_context_multiscale_20261018_v1.run verify --asset-root "$env:ASSET_ROOT"
python -m research.raw_context_multiscale_20261018_v1.run report --asset-root "$env:ASSET_ROOT"
python -m research.raw_context_multiscale_supplement_20261019_v1.cp0 --help
python -m research.raw_context_multiscale_supplement_20261019_v1.cp0 --help
python -u -m research.raw_context_multiscale_supplement_20261019_v1.cp0 --asset-root <local-asset-root> --plan reports/raw_context_multiscale_20261018_v1/SUPPLEMENT_PLAN_001.json --device cuda
python -m research.raw_context_multiscale_supplement_20261019_v1.report generate --asset-root "$env:ASSET_ROOT" --plan reports/raw_context_multiscale_20261018_v1/SUPPLEMENT_PLAN_001.json --terminal .cache/raw_context_multiscale_20261018_v1/TERMINAL_COMPLETION_RECOVERY_002.json
python -m research.raw_context_multiscale_supplement_20261019_v1.report replay --source-dir reports/raw_context_multiscale_20261018_v1/final --output-dir .cache/raw_context_multiscale_20261018_v1/final_replay_001
```

audit는 기존 봉인과 부모를 재검증한다. 최초 smoke 명령은 실제 검증된 인터페이스이나 기존 immutable CPU 파일이 있으면 같은 이름을 덮어쓰지 않고 실패할 수 있다. 반복 TRAIN smoke가 필요하면 gates CLI의 새 --output-id로 별도 attempt를 등록한다.

execute 재개는 기존 optimizer·sampler·RNG·checkpoint를 보존하며 같은 immutable CORE_EXECUTION_START와 deadline을 사용한다. 새 실행으로 deadline을 연장하지 않는다. 기본 상한은 core wall 6시간·신규 GPU process wall 상한 4시간이며 마지막 20%를 예측·비용·검산·보고에 남긴다. 미입장 연장을 과거 decision을 덮어써서 수행하지 않는다.

generate는 main terminal COMPLETE/exit0, primary 독립 검산 PASS, 전체 SCORES/COST_INDEX, CP0 독립 검산 PASS와 등록 source SHA를 먼저 요구한다. 실제 생성 전에 primary 결과를 수정하지 않는다. 기본 final 폴더는 새 출력이어야 하며 replay는 별도 새 폴더에만 쓴다.

PARTIAL branch는 이 wrapper에서 독립 검산이 구현돼 있지 않아 fail closed한다. 그 경우 완료 subset의 prediction·cost 독립 검산과 누락 사유·재개 명령을 갖춘 별도 PARTIAL 보고 경로를 먼저 등록해야 한다. 메타데이터만 보고 검산 완료로 표시하지 않는다.
