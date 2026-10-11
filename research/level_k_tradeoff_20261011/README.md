# 동결 TSFM: K·표현·미래 디코더의 비용–오차 비교

**2026-10-11 완료된 두 자료·두 seed의 반복 VAL 개발 실험.** 같은 FORECAST_RIDGE_D8 계약에서 PCA·point MLP의 K sweep과 공식 TS2Vec 경량 설정을 비교했다. Raw F와 Raw + D8을 함께 유지했다. GitHub 게시에서 실험 수치·선택 규칙·봉인된 science 소스는 변경하지 않았다.

## 결과와 연구 판단

- [전체 결과 보고서](REPORT_KO.md): 설정, MSE/MAE·K·seed·허용 오차, B1/B32 전체 비용, 디코더 진단, 문제 정의·기여 후보·한계.
- [교수님 제안 방법과 TS2Vec 설정](METHOD_CHOICE_KO.md): 공식 코드 revision, 작은 architecture, causal export, GTConvAE 원형과의 차이.
- [실행·실패 복구 기록](EXECUTION_NOTE_KO.md): 실제 session/PID/종료코드, 실패 원장, 재사용과 학습 예산, 검증.
- [여러 압축기 문헌 검토](context/level_encoder_contract_20261010/METHODS_KO.md): 2026-10-10 당시 검토의 링크 정리 복사본. TS2Vec의 최신 실행 상태는 위 결과 보고서를 따른다.
- [게시 범위와 기존 검증 기록의 해석](PUBLICATION_KO.md), [게시 검사](publication_checks.json).

Jena MLP K6는 Raw F 대비 MSE 평균 +3.89%, MAE +19.38%이며, B32 추론은 3.18배 빠르고 CUDA peak allocated는 56.43% 낮다. 한 seed는 MSE 증가 5%를 초과한다. Robin은 MSE 증가를 20%까지 허용해도 압축 후보가 없었다. TS2Vec의 이번 작은 설정은 같은 K의 단순 표현보다 오차와 전체 시간이 나빴다. 시간 표현 자체·TS2Vec 전체의 예측 능력·새 알고리즘 신규성을 판정한 결과는 아니다.

## 시각자료

각 그림은 보고서에도 삽입돼 있다. 아래 PNG는 GitHub에서 바로 볼 수 있고 PDF는 별도로 열거나 내려받을 수 있다.

![MSE와 전체 B32 추론 시간](figures/accuracy_cost.png)

[벡터 PDF](figures/accuracy_cost.pdf)

![B1/B32 추론 시간과 메모리](figures/batch_cost.png)

[벡터 PDF](figures/batch_cost.pdf)

![원채널 디코더의 TRAIN/VAL MSE](figures/readout_generalization.png)

[벡터 PDF](figures/readout_generalization.pdf)

![공통 ridge grid와 선택](figures/ridge_sensitivity.png)

[벡터 PDF](figures/ridge_sensitivity.pdf)

## 코드와 원시 수치

- [실험 계획](PLAN.md), [PCA/MLP 계약](protocol.json), [TS2Vec 계약](method_protocol.json).
- [K sweep](sweep.py), [보존된 실패 이후 continuation](continuation.py), [TS2Vec 실행](method_run.py), [공식 구현 연결](ts2vec_adapter.py).
- [전체 추론 비용 측정](benchmark.py), [소스 봉인·보존](seal.py), [원시 배열·checkpoint 검증](verify.py), [실험 완료 당시 최종 감사](final_audit.py).
- [PCA/MLP·원채널 결과](result.json), [TS2Vec 결과](method_result.json), [52개 비용 측정](benchmark.json), [집계·적용 조건](summary.json).
- [독립 검증 2,518개 항목](verification.json), [실험 완료 당시 115개 최종 감사](final_verification.json), [실패 복구 AST 대조](recovery_source_diff.json).
- [수치 집계와 그림 생성](analyze.py), [공개 JSON에서 보고서 생성](write_report.py), [게시용 경로·파일 검사](check_publication.py).
- [TS2Vec MIT LICENSE](vendor/ts2vec/LICENSE), [공식 revision과 원본 파일 SHA](vendor/ts2vec/provenance.json).

## 실행과 재현 범위

실제 검증 환경은 Windows, Python 3.11 계열, torch 2.10.0+cu128, RTX4070다. 원래 명령과 정확한 receipt는 [실행 기록](EXECUTION_NOTE_KO.md)에 있다. 저장소의 [의존성 잠금](../../requirements-lock.txt)을 함께 참고한다.

공개된 JSON만으로 보고서의 상대경로와 수치를 다시 생성할 수 있다. 저장소 root에서 실행한다.

```powershell
python -B -X utf8 research/level_k_tradeoff_20261011/write_report.py
python -B -X utf8 research/level_k_tradeoff_20261011/check_publication.py
```

전체 학습·benchmark·원시 검증과 `analyze.py`의 디코더 재계산에는 Git에 포함하지 않은 TRAIN/VAL NPZ, 고정 모델 snapshot, 이전 F0/initial checkpoint, 예측 trace·학습 checkpoint가 필요하다. 공개 bundle만으로 전체 과학 실행을 재현했다고 주장하지 않는다. 기존 완료 결과가 있는 이 폴더에서 과학 실행을 다시 시작하면 덮어쓰기 거부가 적용된다. `continuation.py`는 이 회차의 실패/재사용 상태에 맞춘 복구 코드다.

미추적이었던 공통 소스인 budget run/benchmark, encoder runner/encoders, lag pilot을 원래 경로와 byte 그대로 함께 게시했다. 데이터·모델·checkpoint·cache·개인 첨부 내용은 포함하지 않았다. JSON의 로컬 경로는 실행 provenance이며 파일 다운로드 링크가 아니다. 기존 과학 검증은 로컬 asset을 대상으로 완료된 당시의 기록이다.
