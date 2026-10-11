# 실행 기록

이전 연구·cache 4,945개 파일과 HEAD26a5593을 먼저 hash snapshot했다. 사전 protocol에7 arms, 공통10λ,3fold,230/3910 budget을 고정했다. Fixture10개 PASS 후8개 science 소스를 봉인하고 CPU closed-form 비교를 실행했다. 완료 결과의 덮어쓰기를 거부한다.

실제 wall 19.6336초 / process CPU 19.1250초, 신규230ridge후보/3910채널solve, 재사용10후보. 새 F/TEST/GPU/encoder fit0. 실패·retry·후보 추가·λ grid 확장 없음. [과학 결과](result.json), [source 봉인](source_seal.json), [검증](verification.json)은 독립 evidence다. process CPU 및 wall에는 입력검증·hash·NPZ 저장도 포함되어 순수 fit benchmark가 아니다.

독립 NumPy verifier 21,415 PASS, 실패0. 상한λ1e6 경고4건은 실험을 실패로 처리하지 않되 finite-grid 선택 한계로 유지한다. 부모 normalization 때문에 TRAIN rolling CV는 conditional decoder CV다. 전체 forecast 재생성은 없었다.

그림은 [figures](figures/01_comparison.png) 아래4 PNG/4 PDF, 오차 분해는 [summary](summary.json), 상세 해석은 [보고서](REPORT_KO.md). 실험 완료 단계에는 Git/cache cleanup·원본 삭제·기존 소스 수정·commit/push가 없었고 history는 보존 prefix 뒤에 기록했다. 이후 사용자가 승인한 게시 단계는 [별도 안내](PUBLICATION_KO.md)에 기록하며 원래 과학 결과·검증 receipt는 보존한다.

캐시가 이미 준비된 Windows workspace에서 사용한 명령:

```powershell
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/test_run.py
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/audit.py capture
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/audit.py seal
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/run.py --execute
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/verify.py
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/analyze.py
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/write_report.py
& .venv/Scripts/python.exe -B -X utf8 research/robin_readout_diagnosis_20261011/audit.py check
```

위 블록은 원래 실험 시점의 단계별 명령 순서다. 이미 완료한 폴더에서 capture/seal/run을 재실행하면 기존 evidence 덮어쓰기를 막기 위해 중단한다. 원래 보존 manifest와 HEAD 전제도 있어 게시 이후 clone에서 그대로 전체 실험을 재실행하는 안내가 아니다. 원본 과학 결과 없이 analyze/report/verify가 새 실험을 만들어 주지는 않는다. 부모 trace와 TRAIN/VAL 및 새 grid cache도 필요하다. 공개 파일만으로 가능한 fixture 테스트·문서 생성·경로 검사는 [게시 안내](PUBLICATION_KO.md)에 구분했다.
