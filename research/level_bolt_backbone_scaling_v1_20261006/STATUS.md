# 진행 상태

기존 LEVEL/F0 prediction·data·checkpoint/source 연결과 공식 Mini/Tiny 모델 준비를 마쳤습니다. 첫4 VAL 원점의 실제 사전검증이 통과했으며 신규 fit0입니다. 기존 LEVEL은 재학습하지 않습니다.

실제 실행·실패·사용량은 `ledger.json`, 모델 바인딩은 `model_manifest.json`, 자료와 부모 재사용은 `input_contract.json`·`reuse_manifest.json`을 기준으로 읽습니다. 준비 과정에서 외부 원자료의 절대경로를 repository-relative artifact로 잘못 변환한 path 오류가 발생했습니다. 원본은 보존했고 공개 참조 정리와 실제 비교 입력의 내부 바인딩을 분리해 수리했습니다. 준비 실패 job도 원장에 남깁니다. TEST/모델 forward 실행 전 오류입니다.

사전검증은 `preflight_checks.json`의 실행 receipt로 확인합니다. 모델 manifest에 실제 로드 수치를 추가한 작업은 `preflight_model_binding_update.json`에 기록했고, 변경 전 manifest를 보존했습니다. 점수 핵심 함수와 paired resampling 계산의 원본 동등성은 `scoring_math_reuse.json`에 기록했습니다.

## 실행·보고

TEST 전 `evaluation_seal.json`으로 20개 구현·자료·모델·재사용·사전검증 파일을 봉인했습니다. Mini/Tiny의 6개 canonical TEST 추론 단위가 완료됐고, 기존 LEVEL/F0의 9개 저장 예측을 재사용했습니다. 새 fit·optimizer update는 0회입니다. `evaluation_attempts/`의 predict/score/verify 종료 기록과 `accuracy_verification.json`의 486개 점수·CSV 일치 체크를 확인합니다.

같은 회차 CUDA 비용은 108행(주 측정90, sentinel18), 각 조건의 3블록을 완료했고 OOM은 0행입니다. 첫24 VAL 원점, B1/B4, warmup10와 10회 전체 pass를 사용했습니다. 비용 job wall time은 398.427초, 사전검증·새 추론을 포함한 GPU job 합계는 416.868초입니다. 새 모델 materialization은 119,580,688 bytes입니다. 정확한 최종 CPU 사용량과 자원 상한 검산은 원장과 `final_checks.json`을 기준으로 읽습니다.

공정성 표·기간별 MSE/MAE·대응 구간·배포 파라미터·원블록·반론·주장 범위를 `FINAL_REPORT_KO.md`에 작성했고, 원본 PPT/PDF는 편집하지 않았습니다. 자료별 PNG/SVG 6개를 만들었습니다. 첫 렌더의 마커 잘림은 봉인된 평가·비용 코드와 수치를 그대로 두고 별도 표시 보정으로 해결했으며, 전후 manifest와 동일 plotted values를 `figures/layout_correction.json`에 연결했습니다. 수정된 세 PNG도 실제 시각 확인했습니다.

## 최종 검산과 기술 수리

첫 종합 검산의 실패는 `final_checks_attempt01.json`과 원장에 보존합니다. Jena의 combined SSE 합산 순서 차이(max absolute 8.73115e-11, relative 3.92853e-15)를 metric 오차와 구분하고, 같은 모델 경로의 CSV 절대경로/JSON 상대경로 표기를 비교하는 검산기를 수리합니다. 원본 평가·비용·CSV와 TEST 봉인은 유지하며, 모델 추론·성능 재평가는 반복하지 않습니다. 수리 기록은 `verification_repair.json`, 최종 완료 여부는 `final_checks.json`의 실제 pass·coverage와 종료 원장을 따릅니다.

자체 검산 PASS는 독립 재현이나 논문 신규성을 인증하지 않습니다. 사전학습 중복·보호된 외부 확인·순수 방법 인과효과는 해결하지 않았습니다.

## 게시 범위

현재 사용자가 승인한 ordinary main commit/push의 대상은 이 비교의 코드·작은 receipt·CSV·보고·그림과 root README입니다. 캐시의 가중치·checkpoint·큰 예측과 무관한 dirty 파일은 제외합니다. 무관 파일 보호 전체 경로 목록은 local-only guard이며, 공개 검산에는 범위·실제 체크 수·SHA만 남깁니다. 원격 게시 성공은 live origin/main SHA와 게시 파일의 GitHub blob 확인으로 별도 검증하며, 문서의 완료 문장만으로 판정하지 않습니다.
