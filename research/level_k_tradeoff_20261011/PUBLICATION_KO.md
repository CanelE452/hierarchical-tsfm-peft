# GitHub 게시 범위와 검증 기록

사용자의 명시 요청에 따라 이번 실험 코드·보고서·시각자료를 `main`에 게시한다. 게시 작업에서는 새 학습·F 추론·TEST 평가·benchmark를 실행하지 않는다.

## 경로와 포함 파일

보고서 이미지 4종과 근거 링크를 문서 위치 기준 상대경로로 바꿨다. [보고서 생성 코드](write_report.py)도 같은 상대경로를 출력한다. 저장소 root README와 [이번 결과 색인](README.md)에서 보고서·실행 기록·코드·PNG/PDF를 찾을 수 있다.

이번 폴더의 코드·문서·JSON 결과/검증/실패 원장, TS2Vec pinned source·MIT LICENSE·provenance, 그림 PNG/PDF를 포함한다. 코드가 직접 참조하는 이전 budget run/benchmark/result 및 encoder runner/encoders·lag pilot도 원래 경로와 byte 그대로 포함한다. 이전 문헌 검토·pilot 보고서의 Markdown 링크 대상 12개는 `context/`에 복사했으며, 링크만 상대경로로 변환했다. 원본 경로·원본/게시본 SHA는 [context provenance](context/provenance.json)에 있다. 이전 연구의 원본 파일은 수정하지 않았다.

원자료·모델 가중치·학습 checkpoint·예측 배열·cache·Python cache·임시 파일·로그·CPU fixture model pickle은 게시 대상에서 제외했다. 원래 JSON manifest가 이 파일의 경로와 SHA를 기록한 것은 로컬 실행 근거이며 공개 파일이 모두 포함됐다는 뜻이 아니다. parent preservation에는 개인 첨부의 파일명·SHA·크기 정보가 있으나 첨부 본문은 게시하지 않는다.

## 원래 과학 감사와 게시 감사

[독립 과학 검증](verification.json)의 2,518항목 PASS와 [실험 완료 당시 최종 감사](final_verification.json)의 115항목 PASS는 게시 전 로컬 상태에서 수행됐다. 그 결과·warning·SHA를 사후에 변경하지 않았다. 보고서 링크 수정·README 색인 추가·게시 커밋으로 문서 SHA와 Git HEAD가 바뀌므로, 당시의 ‘HEAD 불변’과 문서 SHA는 당시 상태를 나타낸다. [실행 기록](EXECUTION_NOTE_KO.md)의 ‘커밋·푸시 없음’도 실험 완료 시점의 기록이다.

[게시 전 snapshot](publication_before.json)은 당시 artifact hash 및 수정할 문서의 보존 위치를 기록한다. 문서 원본은 로컬 `.cache/level_k_tradeoff_20261011/publication_before/`에 보존했다. 공개 보고서와 원본 보고서를 비교할 때 Markdown 대상 경로를 제외한 문장·수치·표는 같다. 새 [게시 검사](publication_checks.json)는 현재 링크 대상·이미지·source seal·science JSON·context copy·게시 파일의 SHA를 확인한다. 과학 검증 2,518개를 새로 실행한 것으로 표현하지 않는다.

공개 bundle의 Markdown 링크·이미지·source seal은 원자료 없이 검사할 수 있다.

```powershell
python -B -X utf8 research/level_k_tradeoff_20261011/check_publication.py
```

전체 `verify.py`와 `final_audit.py`에는 미게시 로컬 asset과 게시 전 보존 상태가 필요하다. `final_audit.py`는 기존 receipt를 덮어쓰지 않으며, 이 게시 커밋 이후의 HEAD를 게시 전 HEAD와 같다고 판정하도록 수정하지 않았다. 외부 재현은 데이터·checkpoint를 복원한 별도 workspace에서 실행 계약과 경로를 확인한 뒤 진행해야 한다.
