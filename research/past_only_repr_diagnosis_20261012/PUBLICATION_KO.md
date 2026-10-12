# 이번 실험의 GitHub 게시와 확인 범위

[확인] 이 게시물은 과거 전용 표현·지연 그래프 실험의 코드, 결과 메타데이터, 상세 보고서, 원수치·비용 CSV, 그림 8개를 함께 제공한다. 최신 판단은 [결과 보고서](README.md)와 [최종 상세 보고서 v5](final_report_v5_20261012/REPORT_KO_v5.md)를 따른다. 이전 LEVEL 전체 캠페인의 완료 보고가 아니다.

## 실험 종료 기록과 게시 승인을 구별하기

실험 종료 당시에는 사용자가 commit/push를 금지했다. 따라서 봉인된 `STATUS_FINAL_v5.json`, `EVIDENCE_FINAL_v5.json`, INDEX 및 v5 보고서의 `commit_push: 0` 또는 “commit/push는 하지 않았다”는 그 회차의 실제 상태다. 이후 사용자가 main 게시를 명시적으로 승인했다. 이 게시 작업은 새 학습·추론·TEST 사용 없이 기존 산출물을 점검하고 안내 문서와 게시 검증을 추가한다. 당시 기록을 사후 수정하지 않는다.

실험 계획·결과·실패 attempt와 변경 전 source snapshot은 당시 경로와 해시를 보존한다. `FINAL_VERIFY_v3.json`이라는 파일명은 출력 형식의 역사적 이름이고, 최종 채택 검산은 [final_verify_v5.py](final_verify_v5.py)의 산출물이다. 초반 부분 완료 보고서·옛 최종 보고서 v4와 두 실패 검산의 큰 JSON은 로컬 이력으로 보존하며 이번 최신 보고서 진입점에 사용하지 않는다. 실패 이유와 후속 조치는 최신 보고서와 실행 원장에 남아 있다.

## 무엇을 점검했는가?

실험 완료 시의 [독립 검산 원문](receipts/final_proof_v5_20261012_1948/FINAL_VERIFY_v3.json)은 저장된 예측 배열로 지표·선택 조건을 재계산했다. E3/E4, E5, E5-Bolt, E6 및 비용의 primary section은 PASS이며, 보고서 채택 연결은 3,429/3,429 PASS, 제외한 완료 과학 조건은 0개다. 의도적으로 보존한 TF32 불일치 대조까지 모두 precision PASS라는 뜻은 아니다. 검사 수는 독립 표본 수가 아니다.

이번 게시 검증은 그 실험을 다시 수행하지 않는다. [verify_publication.py](verify_publication.py)는 표준 라이브러리만 사용하여 게시 파일 목록, 봉인 근거·보고서·CSV·그림의 SHA256, PNG 청크 CRC, 코드 문법과 문서의 상대 링크를 확인한다. 저장 예측 배열·모델·보호 TEST를 열지 않으며, 기존 과학 검산의 파일과 상태가 유지되는지를 검사한다. [게시 전 점검 결과](publication_receipts/PRE_PUSH_AUDIT_v2.json)와 [게시 파일의 명시적 목록](PUBLICATION_FILES_v1.txt)을 함께 제공한다. 초판 게시 점검 이후 평균 그림의 해석 문구를 명확히 했고, 그 문서를 포함한 최종 파일은 v2에서 다시 점검했다.

```powershell
# 저장소 루트에서 실행. 실험·모델 실행 없이 게시 파일만 점검한다.
python research/past_only_repr_diagnosis_20261012/verify_publication.py
```

위 명령은 새 clone에서도 게시된 파일과 기존 tracked 근거만 확인한다. 약 17MB인 독립 검산 원문은 GitHub 웹 미리보기 제한이 있을 수 있어 Raw 보기를 이용한다. 빠른 확인에는 위의 작은 게시 점검 결과와 상세 보고서가 적합하다.

## 코드와 로컬 자산의 재현 범위

핵심 실행 소스와 필요한 외부 import 1개인 [ts2_export_followup.py](../professor_compressor_followup_20261012_run01/ts2_export_followup.py)를 함께 게시한다. 하위 의존인 기존 `common_direct.py`, `ts2_adapter.py`, `models_gt.py`, pinned TS2Vec vendor와 데이터 계약은 이미 저장소에서 추적 중이다. 외부 source 1개의 게시를 이전 LEVEL 작업의 재개로 해석하지 않는다.

원자료·모델 가중치·체크포인트·NPZ/NPY 예측 배열은 Git에 포함하지 않는다. 실제 재학습이나 전체 배열 재검산에는 SHA가 일치하는 다음 로컬 자산이 필요하다.

- TRAIN/VAL: `.cache/tsfm_peft_practical_controls_v7_20260928/data`의 원 자료와 tracked v18 데이터 계약.
- E1/E3/E4: 기존 TS2Vec·GTConvAE 체크포인트와 기록된 SWA 상태. 옛 인코더의 TRAIN_ALL/VAL 노출 한계는 유지한다.
- Bolt: `.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8`. 소스는 `local_files_only=True`로 읽으며 자동 다운로드하지 않는다.
- 이번 저장 예측·학습 상태: `.cache/past_only_repr_diagnosis_20261012`의 결과 배열·체크포인트.

봉인된 preflight·plan·result에는 실행 당시 Windows 절대경로가 남아 있다. 이것을 상대경로로 바꾸면 원래 해시 연결이 깨지므로 JSON을 수정하지 않는다. GitHub 문서·그림·원수치의 탐색 링크만 저장소 상대경로로 제공한다. 새 clone에서 즉시 전체 학습이나 배열 검산이 가능하다고 주장하지 않는다. 실제 재실행은 원 자산 복원, SHA·경로 조건 확인, 새 출력 경로 및 예산 검토가 선행되어야 한다. 기존 종료 결과를 덮는 실행 명령을 안내하지 않는다.

## 게시 범위와 원본 보존

Git의 자동 개행 변환으로 봉인 source의 SHA가 달라지지 않도록 이 실험 폴더에만 `.gitattributes`의 `-text`를 적용했다. 그림 01~07은 원 보고서 manifest에, 가독성 개선판 08은 [별도 receipt](final_report_v5_20261012/COVERAGE_READABILITY_RECEIPT_v1.json)에 연결되어 있다. 원본 그림 07과 manifest를 바꾸지 않고 보고서 진입점에는 개선판 08을 사용한다.

이번 게시 파일은 명시적 목록으로만 stage한다. 관련 없는 미추적 실험, 기존 history의 다른 작업 변경, 원자료와 대형 모델은 포함하지 않는다. 원래 지시의 보호 TEST 금지와 기존 중단 실험 재개 금지는 유지한다. 공개 보고서와 증거 확인을 위한 게시이며 독립 일반화 검증이나 전체 모델 재현 패키지가 아니다.
