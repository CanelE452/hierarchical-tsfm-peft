# V6 status

[확인] 승인된 새 집단 비교·비용·영문 논문 초안·최종 검산을 완료했다. 결과 commit `9bcb58d67aeafd3e2ed654ca96d50739e37aee35`를 origin/main에 push했고 git ls-remote의 실시간 SHA 일치를 확인했다. [게시 확인](publication_receipt.json)을 남겼다. 새 적합/예측/벤치마크 작업은 없고 원장상 활성 job도 없다.

Robin Office17/K5 고정 PCA에서 신경망10 + CPU ridge2 = 12/16 attempt를 완료했다. 모두 조기 종료했고 예비 연장 대상0, 모델 실패·재시도0이다. 합성 optimizer1/6세션은 총10update였다. 14개 선택/무학습 모델을 함께 TEST 평가했으며 이후 모델·자료·선택을 바꾸지 않았다.

LEVEL 전체 MSE0.275196: OLD 대비48.92%, matched RAW 대비20.61%, LEVEL_ONLY 대비21.16% 낮았다. F0/LoRA 대비20.12%/23.07% 손해는 남았다. 같은 회차 batch4 peak allocated는214.49 vs265.49MiB로19.21% 작았지만 안정적인 처리량 우위는 확보되지 않았다. 제한된 방법 기여는 유지하며 실제 배포 주장은 메모리–정확도 절충으로 한정한다.

- [승인 계약](PLAN.md), [승인 원문](APPROVAL.txt), [원래 채팅 계획과 보존 시점](approved_plan_receipt.json)
- [최종 판단](TOPIC_DECISION.md), [영문 초안](PAPER_DRAFT.md), [선행·기여 경계](CLAIMS_PRIOR_ART.md)
- [고정 평가](evaluation01.json), [비교 JSON](final_comparison.json), [전체 채널 CSV](final_channel_metrics.csv)
- [GPU 비용](cost_gpu01.json), [CPU 비용](cost_cpu01.json), [한 차례 상주 진단](cost_diagnostic01.json), [최종 그림](figures_layout02/manifest.json)
- [최종 검산 PASS](final_checks.json), [게시 내용 점검 PASS](publication_content_check01.json), [실행 명령](commands.json)

검산은 원본644개 추적 파일 보존, 예산/노출 순서, 실제 갱신/동결/복원 및 VAL 선택, 저장된 예측14종의 MSE/MAE, 기간 합산과 seed 평균, 대응 블록구간8종, 비용78GPU+18CPU행, JSON/CSV와 원고 숫자·그림 원본 연결을 확인했다. 이 실행팀의 검산은 독립 재현이 아니다. 논문 초안 완료와 논문 제출 준비 완료도 구분한다.

최종 게시 확인 뒤 자원 snapshot(아래 확인 시점의 원장 누적값):
```json
{
  "real_fit_attempts": 12,
  "synthetic_sessions": 1,
  "gpu_seconds": 1536.831430200371,
  "cpu_check_seconds": 23.726517399638983,
  "data_prepare_seconds": 0.8988480000989512,
  "cpu_analysis_seconds": 52.88517740019597,
  "active_jobs": [],
  "storage_bytes": 27884701,
  "limits": {
    "real_fit": 16,
    "synthetic_sessions": 6,
    "synthetic_steps": 10,
    "gpu_seconds": 14400,
    "cpu_check_seconds": 900,
    "storage_bytes": 5368709120,
    "download_bytes": 0,
    "conflict_wait_per_incident_seconds": 1800,
    "conflict_wait_total_seconds": 3600
  }
}
```

GPU 작업점유1536.831초=25.61분/240분. 신규 저장은 약27.9MB/5GiB, 다운로드0이다. 원본/백본/예측/가중치는 cache에 두고 게시하지 않는다. 기준 main/origin/main은9ddc0368716d32007b9af89859c7e91aa66be6fa이며, 결과 게시 완료 SHA는 publication_receipt에 기록했다. 이 STATUS/receipt를 반영하는 후속 일반 commit은 게시 메타데이터만 변경한다. 기존 v1–v5와 무관한6개 미추적 항목을 유지한다.

계상 한계: 최초 AST3건(0.3345136/0.3937/0.3522초), 문서 diff검사0.2246446초와 cp949 실패·UTF8 재시도0.2688814/0.2691553초는 사전예약을 놓쳐 ledger에 사후 계상했다. 모델/실자료 fit/optimizer 실행이 아닌 정적·문서 검사이며 숨기지 않는다. v2의22/20회 위반을 소급 승인하지 않는다.

보고 코드의 표시 숫자 연결·GPU/CPU 역할 guard를 수리했고, 첫 그림의 축 잘림을 원본을 보존하며 수정했다. 원수치·모델·자료·선택은 바뀌지 않았다. 최종 검산 실패/재시도는 없었다. 수리 diff와 첫 산출물은 repair_reporting01.md 및 figure_review.json에 있다.

남은 과학적 한계: 한 site·두 학습 seed, 백본 사전학습 비중복 미확정, F0/LoRA보다 큰 오차, 안정적 처리량 우위 미확보, 독립 외부 재현 없음. 다음 결정 하나는 이 고정 구성을 다른 한 도메인으로 확인할지이며 별도 자료 계약이 필요하다. 현재 승인으로 추가 실험을 시작하지 않는다.

게시 점검에서 생성 SVG의 경로 구문과 원문 repair diff의 행 끝 공백을 확인했다. 봉인 hash/원본 보존을 위해 그9개 파일의 공백만 허용했고 다른 코드·본문은 git diff --check를 통과했다. 설정·hook은 우회하지 않았다. 자세한 기록은 [publication_whitespace_review.json](publication_whitespace_review.json)에 있다.
