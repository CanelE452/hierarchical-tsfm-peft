# Robin 실험의 GitHub 게시 범위와 검증

사용자 명시 승인에 따라 이번 Robin 실험 코드·보고서·그림을 main에 게시한다. 대상은 [파일 allowlist](publication_files.json)의 Robin 폴더와 저장소 README 색인, 해당 날짜 history다. 부모 공통 소스·결과는 이미 tracked이며 게시 준비에서 실험 fitting, F/TEST 추론, benchmark를 추가하지 않는다.

- 결과와 해석: [REPORT_KO](REPORT_KO.md), [result](result.json), [summary](summary.json)
- 실행 계약: [PLAN](PLAN.md), [protocol](protocol.json), [실행 기록](EXECUTION_NOTE_KO.md)
- 실험·검증 소스: [run](run.py), [tests](test_run.py), [verify](verify.py), [analysis](analyze.py), [report generator](write_report.py)
- PNG/PDF: [비교 PNG](figures/01_comparison.png) / [PDF](figures/01_comparison.pdf), [규제 경로 PNG](figures/02_ridge_paths.png) / [PDF](figures/02_ridge_paths.pdf), [원인 PNG](figures/03_mechanisms.png) / [PDF](figures/03_mechanisms.pdf), [채널·시점 PNG](figures/04_channel_origin_sensitivity.png) / [PDF](figures/04_channel_origin_sensitivity.pdf)
- 과학 검증: [21,415 checks](verification.json), [4,996 checks의 완료 당시 감사](final_audit.json), [source seal](source_seal.json)
- 게시 검증: [검사 코드](check_publication.py), [검사 결과](publication_checks.json), [게시 전 원본 SHA](publication_before.json)

모든 Markdown 대상은 해당 문서 위치를 기준으로 한 상대경로다. 원자료·모델 weight·checkpoint·예측 NPZ·cache·로그·관련 없는 작업 파일은 게시하지 않는다.

## 원래 과학 기록과 게시 검사의 차이

`final_audit.json`은 HEAD26a5593에서 실험을 완료한 로컬 snapshot이다. 게시 전 원래29개 artifact와 문서·색인·history 원본6개를 보존했다. 게시 단계에서는 문서의 상태 설명·색인만 수정하며 실험 수치, protocol/run/test의 source seal, 결과·검증 receipt, 그림의 bytes를 유지한다.

`report_review.json`의 문서 SHA는 과학 완료 당시 원본에 해당한다. 그 원본은 로컬 backup SHA로 대조하며 게시 후 문서의 SHA는 `publication_checks.json`에 기록한다. 과학 감사가 게시 이후 Git HEAD를 검증한 것처럼 읽으면 안 된다. `audit.py check`는 원래 snapshot 전제를 사용하고 기존 감사 파일을 덮어쓰므로, 게시 경로 검증에는 아래 공개 검사 코드를 쓴다.

## 공개 clone에서 가능한 확인

게시 검사에는 Python 표준 라이브러리와 Git만 필요하고 데이터·모델을 로드하지 않는다.

```text
python research/robin_readout_diagnosis_20261011/check_publication.py
```

합성 invariant10개는 NumPy와 threadpoolctl을 설치한 환경에서 원자료 없이 확인할 수 있다.

```text
python research/robin_readout_diagnosis_20261011/test_run.py
```

한국어 문서는 공개 result/summary/verification을 읽어 생성한다. NumPy가 필요하며 새 fitting이나 그림 재생성은 하지 않는다.

```text
python research/robin_readout_diagnosis_20261011/write_report.py
```

전체 실험·배열 기반 독립 검증·그림 재생성에는 부모의 Raw F/TRAIN forecast NPZ, 정규화된 TRAIN/VAL asset, 이번 실험의 inputs 및 arm/fold grid NPZ가 필요하다. 이 cache들은 공개 Git에 없다. 전체 실행은 덮어쓰기 거부와 과학 완료 당시의 보존 manifest·HEAD 전제도 있으므로 cache 복원만으로 새 clone에서 바로 반복할 수 있다고 보장하지 않는다. 새 실행은 원래 증거를 보존하는 별도 실행 계약이 필요하다.

게시 준비의 `--local --staged` 검사는 로컬 원본·cache 보존과 개별 stage된 파일의 정확한 bytes도 확인한다. 이 결과는 commit 전 snapshot이다. 실제 원격 main SHA, 파일 목록, report 렌더링과 이미지 다운로드는 push 이후 별도로 확인한다.
