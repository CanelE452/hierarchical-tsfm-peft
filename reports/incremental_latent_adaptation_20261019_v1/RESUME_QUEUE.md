# 실행·재개 상태

새 신경망 구성 22개, search/refit 내부 단계 48개 중 48개 완료. epoch 0 refit 2개는 최적화 없는 별도 단계다.

실험·보고·검산 범위의 미완료 없음.


명령은 저장된 봉인과 완료 파일을 확인하며 재개한다. 예산 종료로 남은 단계는 새 결과로 간주하지 않는다.

```powershell
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run register --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run audit --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run probes --asset-root "E:\CODING\proj\hierarchical-tsfm-peft" --dataset jena
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run probes --asset-root "E:\CODING\proj\hierarchical-tsfm-peft" --dataset pems32
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run train-B --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run finish-training --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run final-accuracy --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.run measure --asset-root "E:\CODING\proj\hierarchical-tsfm-peft"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.verify --manifest "C:\Users\User\.codex\worktrees\adapter-diagnostics-v2\hierarchical-tsfm-peft\.cache\incremental_latent_adaptation_20261019_v1\FINAL_MANIFEST.json" --outdir "C:\Users\User\.codex\worktrees\adapter-diagnostics-v2\hierarchical-tsfm-peft\reports\incremental_latent_adaptation_20261019_v1"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir "C:\Users\User\.codex\worktrees\adapter-diagnostics-v2\hierarchical-tsfm-peft\reports\incremental_latent_adaptation_20261019_v1"
& "E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe" -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir "C:\Users\User\.codex\worktrees\adapter-diagnostics-v2\hierarchical-tsfm-peft\reports\incremental_latent_adaptation_20261019_v1" --verify-replay
```

정리 순서: 비용까지 완료 → build_status()로 원장·공개 카운터 해석 고정 → report 및 report --verify-replay → verify → build_status(refresh_ledger=False)로 상태만 갱신. 최종 검산 PASS 후 다시 verify하여 포장 상태를 확인한다.

훈련 단계 시간과 프로세스 전체 시간은 범위가 겹쳐 합산하지 않는다. 과거 A60 부모 비용은 재사용 비용이며 신규 자원 한도와 구분한다. F가 없는 방법도 학습 그룹 프로세스에는 공통 F가 상주했으므로 standalone 훈련 메모리는 N/A다.

Includes conservative upper-bound component; total is not entirely exact
