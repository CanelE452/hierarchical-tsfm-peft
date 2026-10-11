# 직접 실험의 GitHub 게시 범위와 검증

사용자가 지금까지의 결과를 main에 push하도록 승인했습니다. 이번 게시 범위는 [328개 explicit allowlist](publication_files.json)의 direct 실험 source·수치·81개 PNG·receipt·원형 대비 변경·실패 기록, 저장소 README 색인과 해당 날짜 history입니다. 새 동일 조건 Jena 비교와 관련 없는 untracked 작업은 이 게시에 포함하지 않습니다.

- [결과·비용·개별 그림](public/REPORT_KO.md), [결과 색인](README.md), [실행 source·원문·재현 범위](public/SOURCE_README.md)
- [MSE/MAE](metrics.csv), [비용88개 조건](COSTS.csv), [학습·추론 stage202개](TRAINING_STAGES.csv)
- [완료 당시 과학 감사](FINAL_EXECUTION_AUDIT.json), [게시 전 원본 SHA와 문서 변환](publication_before.json)
- [표준 라이브러리 게시 검사](check_publication.py), [게시 검사 receipt](publication_checks.json)

실험 source/result/protocol/그림/최종 감사는 변경하지 않습니다. 모든 공개 문서의 링크·이미지는 GitHub 상대경로를 사용합니다. 자료·weight·checkpoint·prediction/diagnostic NPZ·cache·로그·이미지 재생성 archive·private attachment는 게시하지 않습니다. 이미 tracked인 최소 runtime/vendor 의존성은 manifest에 따로 나열합니다.

## 과학 snapshot과 게시 사본

`FINAL_EXECUTION_AUDIT.json`은 HEAD `591ce5ce388f6b70edd79a2897cff2b986ad0fc3`에서 완료된 과학 snapshot이며 byte 그대로 유지합니다. 그 안의 `commit_push:false`는 완료 당시 상태입니다. 이후 승인된 Git 게시를 실패 또는 누락으로 해석하지 않습니다. 문서/history SHA도 그 시점의 값이며 게시 이후 HEAD나 전체 작업공간이 같은 상태라는 주장이 아닙니다.

실험 완료 당시 한국어 문서는 수정하지 않고 로컬에 보존했습니다. 공개 한국어 문서10개는 `public/` 사본으로 원본 SHA를 명시하고 수치·판단을 유지한 채 링크를 이동했습니다. 게시 시점 안내만 본문 앞에 추가했습니다. 공개 감사의 원본 문서 SHA는 로컬 원본과 비교하거나 `publication_before.json`의 변환 provenance로 확인합니다. 공개 clone에 그 원본 Markdown 파일이 있다고 가정하지 않습니다.

과거 Electricity의 protected target materialization 사고 원본은 로컬 byte 그대로 남깁니다. 공개 [범위 기록](public/PROTECTED_ACCESS_CONTEXT.json)에는 원본 SHA, old campaign guard 실패, 모델 학습·추론·scoring·selection에 사용하지 않았다는 범위만 제공합니다. 해당 사고의 protected target 진단 통계는 재게시하지 않습니다. **전체 campaign의 TEST 접근0은 주장하지 않습니다.** 신규 direct Robin/Jena scope 접근0과 구 사건을 구분합니다.

Old LEVEL A–H와 clean-cost revision2는 중지된 미완료 범위입니다. [중지 receipt](OLD_FIT_CANCELLED.json)를 게시하지만 구 checkpoint와 자동 실행 queue는 실행하거나 복원하지 않습니다. GT guard 실패1개, TS2Vec 완료 pilot1개, Jena 부분 실패8개도 성공 canonical16개와 구분해 공개합니다. `FIGURE_MANIFEST.json`에는 과거 pilot 이미지의 SHA 기록도 남아 있지만 공개 PNG81개는 canonical80개와 전체 비교1개입니다. Pilot 이미지와 summary archives는 allowlist에서 제외합니다.

## 공개 clone에서 가능한 확인

Python 표준 라이브러리와 Git만으로 상대 링크·PNG signature·Python syntax·JSON/CSV·공개 sealed artifact SHA·canonical81개 이미지·완료 receipt 수를 검사할 수 있습니다. 모델·원자료·weights·GPU·TEST를 열지 않습니다.

```text
python research/professor_compressor_direct_20261011/check_publication.py
```

`--local`은 이 머신의 미게시 원본 문서·게시 전 backup·구 incident 원본 SHA까지 비교합니다. `--staged`는 변경 파일이 allowlist와 정확히 일치하고 index bytes가 작업공간과 같은지 검산합니다. Receipt는 commit 전 게시 준비 시점입니다. 실제 push 이후 원격 main SHA와 파일·문서·PNG 가용성은 별도로 확인합니다.

전체 실험·NPZ 기반 독립 metric 재검산·원본 checkpoint 전체 재구성에는 로컬 TRAIN/VAL archive, 해당 revision의 Chronos weights와 각 A/B/encoder checkpoint·예측 배열이 필요합니다. 경로와 SHA는 공개 receipt 및 [소스 안내](public/SOURCE_README.md)에 있지만 binary는 공개 Git에 없습니다. 단순 clone으로 전체 실험이 즉시 재현된다고 보장하지 않습니다. 완료된 ID를 재실행하면 기존 cache를 덮을 수 있으므로 별도 run ID/출력 계약이 필요합니다. 일부 epoch 상한 도달과 반복 VAL, Jena batch4/Robin effective batch1의 차이도 유지합니다.
