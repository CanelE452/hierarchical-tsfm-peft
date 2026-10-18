# 메모리 예산 실험 완료 감사

[확인] 과학 실행과 독립 검증은 완료됐고 게시 단계는 `PENDING_PUSH`다. 이 문서는 마지막 push 이후 전체 allowlist 원격 검증을 미리 완료했다고 주장하지 않는다.

```text
phase  screens  selected_confirms
A      46       45
B      20       20
```

Phase B `PROCEED` / `REGISTERED_GATE_PASSED`. 동일 checkpoint/B 반복 피크 분리 조건의 고유 그룹은 4건이다. 중복 증거 행을 독립 실험으로 세지 않았다. 신규 모델은 4개다.

정확한 MSE/MAE·원경로/분할 비용·B1 지연·B8 처리량·TSFM/전체 GPU/CPU Ridge 범위는 [RESULTS_KO](RESULTS_KO.md), [T1](T1_ORIGINAL_PATH.csv), [T2](T2_CHUNK_COMPARISONS.csv), [Pareto](PARETO_TABLE.csv)를 따른다. 여러 chunk는 새 정확도 seed가 아니다.

전체 pytest: `604 passed in 93.01s (0:01:33)`. 실험 검증과 실제 공개 replay는 PASS이며 현재 proof/receipt/product/source SHA를 다시 확인했다. 필요한 과학 작업은 0개다. 비용 과다로 등록 규칙상 screen만 남은 후보는 RESUME_QUEUE의 optional 항목이고 강제 confirm을 하지 않는다.

Jena16772/PeMS14284 pre-T 숫자 prefix, A_fit/A_stop/fresh A60/repeated V 역할, 이전 T 노출·legacy 정규화 ancestry·pretraining overlap UNKNOWN을 유지했다. V는 독립 테스트가 아니다. 기존 발표 MD/PPT/PDF와 원 checkout HEAD/status를 다시 확인했으며 원 dirty 파일 전체 content hash를 수행했다는 주장은 하지 않는다.

GPU 값은 allocated/reserved 텐서·allocator 계층이다. PID total은 이번 hook/nvidia-smi에서 미확보했고 device total로 대체하지 않았다. CPU RSS와 GPU memory/speed를 같은 축으로 비교하지 않았다. 고정 첫 B개 창의 20회 p95는 서비스 tail 보장이 아니며 RSS 표본은 모델 적재·full V를 포함한 전체 생애 피크가 아니다.

메모리 절감만으로 신규성·입력별 적응성을 주장하지 않는다. K별 신규 모델을 수행했어도 독립 학습 모델 비교이며 동일 모수 고정 대조나 단일 checkpoint 즉시 K 전환이 아니다.

시작부터 이 마감 snapshot까지 실제 경과 120.86분이다. RUNTIME의 report snapshot과 이 마감 시각은 구분한다. reservation/launcher/worker/loop/search/refit/test 시간은 겹치는 범위이며 GPU kernel busy time은 측정하지 않았다.

게시 proof의 범위는 VALIDATION/STATUS에 있다. payload 단계에서는 정상 main push와 실제 원격 검증이 남아 있고, closing 단계에서는 stable payload 원격 일치만 확인됐다. 마지막 정상 push 뒤의 final allowlist 검증 결과와 실제 closing SHA는 root의 마지막 외부 proof 및 최종 사용자 응답에 남긴다. [요구항목별 근거](REQUIREMENT_COVERAGE.csv), [재개 범위](RESUME_QUEUE.json), [재현](REPRODUCE.md).
