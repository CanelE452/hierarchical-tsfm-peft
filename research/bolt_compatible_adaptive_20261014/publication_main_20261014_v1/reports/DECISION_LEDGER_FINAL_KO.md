> 사용자의 명시 요청에 따른 GitHub 게시용 링크 사본입니다. 수치와 본문은 보존했습니다. 원문 SHA256 `2c890b8c53b172c3ec2f5540f8885843a84cae9d2b19b8b759989386f6af1007`. 원문에 있는 게시 금지/미게시 표현은 실험 당시 기록입니다.

# 실제 결과를 반영한 6항목 결정 기록

이 문서는 종료 시점의 대조 기록이다. 사전 선택은 [승인 계획](../../PHASE1_PLAN_KO.md), [최초 결정 기록](../../DECISION_LEDGER_PHASE1_v1.md), 각 실행의 FIT/TUNE `BRANCH_LOCK.json`으로 확인한다. 아래 문서를 모든 결과보다 먼저 작성한 기록이라고 주장하지 않는다.

## D1 — 같은 시간 특징의 self/joint

1. **질문:** 다른 채널의 같은 시간 요약을 추가하면 자기 채널 기준보다 원미래 예측의 실용 이득이 생기는가?
2. **현 증거:** [확인] 기존 비교는 특징/α 선택 계약이 달랐다. 이번에는 SELF33와 JOINT33의 시간 요약·label·FIT/TUNE·통제·α 선택을 맞추고 SELF512/Raw를 별도 강한 기준으로 둔다.
3. **경쟁 설명:** 유용한 공통 신호의 이득 부족 / 작은 FIT와 큰 joint 용량·규제 / 시간 요약 또는 모델 가정의 부족. 이 모델로 비선형/저차원 정보 부재를 증명할 수 없다.
4. **다음 검사:** 최초2자료·신규SELF28묶음·기존JOINT14묶음 재사용,9pool54비교. 공통 gate는 TUNE MSE1% 이상 개선·MAE 악화1% 이내·시간4블록 중3개 이상 개선이며 Raw 채택 gate와 별개다. fit만 유리하면 같은 FIT의 최대512origin1회 진단, 긍정 gate면 작은K후속, 아니면 분기 종료를 사전 고정했다.
5. **결과:** [확인] Robin TUNE SELF/JOINT .139777591/.262253305(0/4), Beijing .843454174/.953965639(1/4). 신규28·재사용14·54pool 실제 실행, 독립 검산 PASS. Robin 선택 모델의 FIT만 joint가 조금 좋았고 Beijing 선택 모델은 FIT에서도 불리했다. [고정 분기](../../runs/first_robin_v1/BRANCH_LOCK.json) / [Beijing 분기](../../runs/first_beijing_v1/BRANCH_LOCK.json). 개발 점수로 분기를 바꾸지 않았다.
6. **다음 결정:** Robin1회 표본·규제 진단, Beijing 신규 표현 경로 종료. Beijing 개발 EVAL의 joint MSE 이득·MAE 악화 반례는 불확실성으로 보존한다.

## D2 — 동일 FIT 구간의512origin

1. **질문:** Robin joint의 FIT/TUNE 차이가 기존256origin의 학습 밀도·위치 조건 때문인가?
2. **현 증거:** [확인] D1에서 선택 joint의 FIT MSE .278969183은 self .285825349보다 낮으나 TUNE는 나쁘다.
3. **경쟁 설명:** 학습 origin 지원 부족 / 고차원 모델의 과적합·규제 / 시점에 따라 변하는 관계 / 부적합한 선형 요약. 추가 origins도 중첩이라 독립 표본 수 증가는 아니다.
4. **다음 검사:** 같은 FIT 시간 구간·원 FIT 통계·TUNE/DEV·손실·global9pool로 self/joint를512origins에서 함께 학습한다. 교집합128이며 위치/밀도도 달라진다. gate가 계속 부정이면 이 경로의 후속 새 학습을 종료한다.
5. **결과:** [확인] 최초14fit에는 저장 전후 배열 순서에 따른 FP32 요약 재현 오류가 발견돼 본 비교에서 제외했다. 같은 조건의 수정14fit를 독립 검산했다. 유효 TUNE SELF/JOINT .140339388/.260887098, MSE85.90% 악화·MAE33.59% 악화·개선0/4다. [수정된 분기](../../runs/expanded_robin_v2/BRANCH_LOCK.json), [18pool 검산](../../audits/EXPANDED_READOUT_AUDIT_v2.json).
6. **다음 결정:** 계획의 동일 계열 무이득 조건에 따라 종료. ForeCA/linearE-D 실패 또는 공통 신호 부재로 일반화하지 않는다. 예산 잔여 때문에 새 모델을 의무적으로 학습하지 않는다.

## D3 — 저장 입력으로 학습 산술을 재현하지 못한 원인

1. **질문:** dense 검산 실패가 실제 데이터/모델 불일치인가, 배열 순서에 따른 계산 차이인가?
2. **현 증거:** [확인] 원256/새512 raw값·관측 마스크 재구축은 통과했지만 saved FIT feature center의 exact 검사가 실패했다.
3. **경쟁 설명:** 원자료·통계 변경 / 배열 strides에 따른 FP32 평균 / 잘못된 checkpoint/계수 선택 / NumPy 실행 환경. 오차 허용 범위를 넓혀 덮지 않는다.
4. **다음 검사:** 같은 값의 생성 layout/C/F/NPZ layout, 특징 SHA·stat·전α 저장 예측을 비교한다. 원 layout만 재현되면 C 배열을 전체 payload에 명시하고 동일 조건의 수선1회만 허용한다. 별도 실험·새 방법으로 세지 않는다.
5. **결과:** [확인] history byte SHA는 같지만 원 strides `[34816,4,2048]`만 feature/stat/저장 예측을 exact 재현했다. C/NPZ center 차이 최대5.25469659e−8, jointα0예측 차이4.37498093e−5로 기존 parity도 실패했다. [진단](../../audits/DENSE_LAYOUT_DIAGNOSTIC_v1.json), [원 검산 실패](../../audits/EXPANDED_READOUT_AUDIT_v1.json)를 보존했다. C 수선 후5fixture와 실제 재학습·독립 검산 PASS.
6. **다음 결정:** 수정 producer만 유효 숫자로 사용. 물리56신규fit와 유효42를 구분하고, 이 구현 오류를 과거 전체 연구 실패 원인으로 확대하지 않는다.

## D4 — 정확도와 실제 전체 추론비용

1. **질문:** 단순 기준과 Raw의 정확도 차이가 실제 전처리·반환을 포함한 비용 차이와 어떤 관계에 있는가?
2. **현 증거:** [확인] 첫54 및 수정18pool의 수치 검산 PASS. Raw 개발 MSE .340401887/.589575414는 새 선택 ridge보다 낮지만 단독 Raw의 일반화나 압축-F 효과를 입증한 것은 아니다.
3. **경쟁 설명:** 모델 연산 / 채널 수·B1/B8·전송 / 처리 순서·하드웨어 상태 / 동일 출력 반환 계약 차이.
4. **다음 검사:** 첫6경로×2자료×B1/B8의24조건, warmup2·순서 균형6반복·실제 원단위 과거부터 원단위 CPU 반환까지. 수정된 dense2경로×B1/B8 CPU4조건은 별도 trial로, 새 Raw 측정 없이 교차 trial 우위를 주장하지 않는다.
5. **결과:** [확인] 28조건·168샘플 실제 추론, 독립 저장 배열·타이밍 산술 검산 PASS. Raw B1 약12.80–13.35ms/창, B8 약2.51–3.22ms/창; 첫 선택 SELF33 약.09–.24ms/창. 실제 weights/config SHA 및 정규화 forward parity도 확인했다. [24비용](../../cost_runs/first_readout24_v2/RESULT.json), [4비용](../../cost_runs/expanded_robin4_v1/RESULT.json).
6. **다음 결정:** 빠른 ridge가 더 정확한 방법이라는 결론은 아니다. 비용 검산을 독립 타이머·메모리 재측정으로 바꾸지 않는다. 새 압축/디코더/잔차/native 학습 분기는 공통 gate가 확보되지 않아 열지 않는다.

## D5 — 보고서를 실제 결과로 검증할 수 있는가?

1. **질문:** 유효/무효 시도·비용·미실행·그림·선택의 수치를 독자가 파일로 추적할 수 있는가?
2. **현 증거:** [확인] 네 독립 수치/비용 검산 PASS. metadata·환경·report schema 실패5process와 행정 bootstrap 실패1건은 보존됐다.
3. **경쟁 설명:** 실제 원수치 불일치 / stale 상태 링크 / producer-COMPLETE와 scientific-PASS 혼동 / 그림 범위·개발 노출의 과장.
4. **다음 검사:** 한국어 보고·8PNG/8SVG·CSV·6장 스토리 MD를 정확한 producer/audit SHA에 묶고, 별도 도구로 수치·상대 링크·이미지 파일을 검사한다. 과학 수치·scope가 달라지면 완료로 쓰지 않는다.
5. **결과:** [확인] 최종v3 파일 감사는4,288수치항목·44파일SHA·27상대링크·8PNG/8SVG PASS. PNG8개를 직접 봤고 최종v3와 확인한v2의 byteSHA가 동일하다. [최종 파일 감사](../../audits/FINAL_REPORT_ARTIFACT_AUDIT_v2.json). 변경 내용은 현재 상태 링크와 합성 미래 교란 검사 범위 명확화다.
6. **다음 결정:** [이번 캠페인 종료](CAMPAIGN_CLOSURE_KO.md). 새목표·자료·비용 계약이 바뀌는 추가 연구와 보호 TEST는 별도 범위다. 이번 단계의 성공적인 수행과 전체 연구 해결을 구분한다.
