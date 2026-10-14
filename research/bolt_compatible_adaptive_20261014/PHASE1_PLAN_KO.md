# Phase 1 승인 계획: 다른 채널 정보와 Bolt의 추가 가치부터 판별

상태: **승인 대기 / 실제 실행0**. 목적은 작은 K가 원미래 정보를 보존하고, 고정 Bolt가 단순 대안보다 정확도·전체 비용에 기여하는 조건을 판단하는 것이다. 소비처는 교수님과 연구 방향의 유지/조건부 유지/보류/기각을 결정하는 미팅이다. [Phase0 근거](PHASE0_AUDIT_KO.md)와 [신규성 반론](LITERATURE_REVIEW_KO.md)을 재사용한다. 성공점을 얻기 위한 무한 탐색은 하지 않는다.

## 첫 질문과 최소 비교

**질문:** 같은 시간 특징·표본·선택 조건에서 원채널 joint 모델이 자기 채널 모델보다 미래 C를 더 잘 예측하는가? 경쟁 설명은①공통 예측 정보의 실용 이득이 작음②정보는 있지만 작은 FIT/큰 차원·규제가 joint에 불리함③기존 시간 요약/선택 조건이 비교를 흐림이다.

첫 대상은 Robin C17·Beijing C11, 기존 FIT256/TUNE64/VAL30·노출 EVAL146, L512/H48이다. 원 FIT 통계·창별 causal forward-fill·origins/target mask를 유지한다. 새로운 자료/TEST/다른 H는 없다. Jena C21·Appliances C26는 계약이 확인된 조건부 후속 자료이며 TUNE 진단과 근거의 모순을 해소할 때만 사용한다. 최종 평가에서 잘 나온 자료를 골라 확대하지 않는다. 과거 노출 구간은 개발 진단이고 독립 확증이 아니다.

비교3경로는 **SELF33**(자기 채널32×16 평균+현재값), **JOINT33**(동일 요약의 모든 C), **SELF512**(자기 채널 과거 전체512)다. 모든 경로는 현재값 anchor와 actual-Y innovation, FP32 차감→FP64 centered masked ridge→FP32 innovation+현재값을 쓴다. 손실은 관측 channel-macro MSE, λ=αNc, 비규제 절편이다. α=0/1e−4/.001/.01/.1/1/10을 모두 동일하게 **전체 macro TUNE**로 선택한다. 채널별 α를 self에만 허용하지 않는다. p/rank/계수 차이는 보고하고 정보만의 인과 효과로 과장하지 않는다.

각 경로 pool은 PERSIST(innovation0)→동일 FIT 평균 innovation→7ridge다. 통제는 공유하고 exact tie는 앞 후보다. Raw Bolt의 동일 origin 예측도 강한 self 실용 기준으로 둔다. SELF512는 JOINT33보다 입력 해상도가 다르므로 채널 정보 대조는 SELF33 대 JOINT33이고, SELF512는 강한 단순 대안 기준이다.

공유 MEAN은 같은 FP32 actual-Y minus current-C-last와 FIT 관측 mask에서 계산한 H×C 평균이다. 원자료의 FIT 정규화와 별개로, ridge 특징의 center/population scale도 FIT에서만 계산해 고정한다. JOINT 재사용은 이 두 통계 상태를 확인한다. 재사용 producer는 결과를 보기 전에 고정한 `matched_robin_s92601_v1/RESULT.json`과 `matched_beijing_s92601_v1/RESULT.json`이며 실제 source SHA와 후보 artifact는 새 실행 manifest에서 검증한다.

JOINT33은 기존 matched14개 α 후보를 계보/FP32/통계/마스크/점수 gate 후 재사용한다. 기존7ridge 선택을 덮지 않고 동일 PERSIST/MEAN을 추가한9pool로 새 공정 선택을 기록한다. 새 SELF2종×2자료×7α=**28개 모델 묶음**, 재사용14ridge, 6경로×9=**54pool 비교**다. 묶음 안의 채널별 하위 계수와 mask-group factorization 수는 별도로 기록한다. parity 실패·캐시 부재 시 해당 JOINT33만 새7fit로 대체(최대14추가), 실패 원본을 보존한다. 결정적 ridge에2신경망 seed를 붙여 동일 fit를 중복하지 않는다.

## 결과별 후속 분기: 범위 안에서 자동 선택

첫 결과의 FIT/TUNE/개발 원수치·α/선택 여유·MSE/MAE·channel/block/동일 origin 차이를 기록한다. 중첩 창은 독립 표본으로 세지 않으며 최종 평가로 후속 대표 자료/방법을 고르지 않는다.

매 반복은 FIT/TUNE로 다음 분기를 먼저 기록·고정한 뒤 해당 개발 구간을 점수화한다. 이미 본 과거 개발 숫자는 조사 근거로만 남기고 신규 분기 선택에는 사용하지 않는다. **공통 정보 gate**는 JOINT33 대 SELF33의 TUNE MSE 상대 감소1% 이상, MAE 상대 악화1% 이내, 시간순4블록 중3개 이상에서 MSE 감소를 탐색상 자원 배분 기준으로 삼는다. 이는 유의성 검정이나 독립 일반화 기준이 아니다. 기준 MSE/MAE가1e−8 이하이거나 관측이 없으면 비율을 계산하지 않고 불충분으로 표시한다. gate를 넘지 못하는 작은 변화는 약한 근거로 보존한다. **방법 채택 gate**는 Raw·SELF512·Raw+동등 보정 대비 원미래 정확도와 whole 비용이다. 두 gate를 혼동하지 않는다.

- **joint 이득 없음/악화:** 공통 정보 없음을 일반화하지 않는다. FIT만 좋고 TUNE가 나쁘면 각 자료의 동일 FIT 시간 구간에서 결정적 legal origins 최대512로 self/joint를 동시에 확대하는 **1회 표본/규제 진단**만 허용한다. 모든 대조가 같은 추가 FIT origins·기존 TUNE를 쓰며 중첩 증가와 기존256 결과를 별도 보고한다. 이때도 이득이 없으면 이 계열 추가 학습을 멈추고 현재 자료/계약에서의 공통 표현 근거 부족을 보고한다.
- **공통 정보 gate 통과:** 아직 Raw나 SELF512보다 나빠도, 작은 K가 그 이득을 보존하며 비용 절충이 가능한지1차 비교를 허용한다. PCA 재사용/동일 조건 기준, ForeCA 원리 기반 projection, 표준 linear 예측 E/D의 **새 표현 계열 최대2개**만 선택한다. K는 Robin4/Beijing2(조건부 Jena5/Appliances6)부터 시작하며 학습 신경망2seed92601/92602를 보고한다. ForeCA는 FIT spectral 목적이며 공식 재현이라고 쓰지 않고, linear E/D는 actual-Y 예측 목적을 사용하는 AdaPTS와 겹치는 표준 대안이다. 평가·선택은 같은 actual-Y/TUNE 계약이고 원채널 복원 성능·latent 분산/rank·비용을 함께 검사한다. E/D 구조 차이를 단독 E 효과로 확대하지 않는다.
- **같은 E/D의 Bolt 가치가 불명확:** 완료된 shared native F-swap을 재사용한다. 필요한 경우에만 같은 E/D를 동결하고 보다 유연한 채널별 native 선형 head를 actual-Y loss로600updates 학습·같은 TUNE checkpoint(0/100/200/400/600)로 비교한다. future E teacher/oracle로 목표를 바꾸지 않는다.
- **공통 경로가 유용하나 개별 정보가 남음:** 기존 LEVEL/current-C 구조를 재사용한다. 선택된 공통 경로·PCA·Raw Bolt에 **같은 RAW33C 잔차 feature·ZERO/MEAN/7α**를 붙이고 전체 전송·메모리·비용을 비교한다. 이미 끝난 MAIN past-residual84fit를 이름만 바꿔 반복하지 않는다. 준비된 future-conditioned p49K SOURCE는 참고만 하며 이번 실행 범위에서는 제외한다. actual-future E/oracle를 배포 입력이나 보정으로 사용하지 않는다.
- **입력 범위가 경쟁 설명으로 남음:** Raw L512/1024/2048을 지원 자료/공통 origin으로1회만 검사한다. Robin 현 FIT 내부 L2048은 INFEASIBLE이다. L별 origin 감소/target 일치·창 시작 forward-fill을 확인하며 E의 RF가 늘었다고 주장하지 않는다. Raw도 같이 좋아지면 압축 고유 효과가 아니다.

각 분기는 DECISION_LEDGER의 질문/확인 대 추정/경쟁 설명/최저 비용 다음 검사/결과/다음 결정6항목으로 기록한다. 같은 계열2회 무이득이면 중단한다. 독립 평가가 필요할 정도의 TUNE 근거가 생기면 알고리즘 탐색을 멈추고 별도 평가 설계를 제안한다. **보호 TEST는 이 승인에 포함하지 않는다.** 기존 강한 대안보다 추가 기여가 없어도 실패 경계와 불충분을 결과로 허용한다.

계열 상한은 새 **표현 학습 계열2개**다. predictor 대조는 채널별 native 선형 head1계열, 보정은 공통 RAW33C ridge1계열로 한정하고 별도 fit 수·자원을 기록한다. deterministic projection/PCA의 동일 fit를 seed 수로 부풀리지 않는다. representation/native 신경망은 조건마다600updates·batch4·checkpoint0/100/200/400/600을 기본 상한으로 삼는다. 더 많은 loss·head·graph·contrastive 모델을 이 명목으로 추가하지 않는다.

## 지표·검산·비용·자원

Primary는 FIT 통계 기반 normalized 관측 channel-macro MSE, secondary는 MAE/채널별 원단위 MSE·MAE/시간 block/계수·학습 비용·whole 추론·VRAM이다. 값 유지/mean/Raw/Raw+동등 보정/압축만·결합·native를 조건이 맞는 것만 COMPARISON에 넣는다. cost는 selection 고정 후 B1/B8·순서 균형·동일 warmup/repeat·causal forward-fill→정규화→전송→모델/보정→inverse/반환 전체를 측정한다. baseline parity·미래 교란·FIT 통계/asset SHA·정규방정식/저장 예측·선택/CSV/보고 링크·그림을 검증하고 유효 원수치 없는 그림은 만들지 않는다.

새 캠페인 상한안: **GPU process 누적90분, CPU process 누적120분, 전체 wall4시간, 새 저장6GiB, RSS8GiB, GPU reserved9GiB**. CPU와 메모리 상한은 원문에 빠진 부분을 보완한 승인안이다. 사전 baseline에서 더 작은 GPU 상한이 필요하면 낮추고 기존 job을 중지하지 않는다. clock은 승인 후 활성화하며 이전 clock/receipt를 리셋하지 않는다. 첫 readout/검산은 CPU 약10–30분, 비용/보고까지 wall 약30–60분의 **미측정 추정**이며 후속 training은 상한이 남고 gate가 지지할 때만 한다.

소스·결과는 새 시도에 저장하고 원본/TEST/큰 backbone 학습/광범위 새 자료/설치 다운로드/commit/push/무제한 탐색은 하지 않는다. 기존10개 source/manifest SHA를 Phase0 전후 대조한다. 승인 후 RUN_STATUS, DECISION_LEDGER, 동일 계약 COMPARISON.csv, REPORT_KO, 유효 실험 그림, 교수님용5–7장 스토리의 PRESENTATION_BRIEF_KO.md를 쓴다. 실제 PPT/논문은 만들지 않는다.

**이 계획 승인 후에는 위 범위 안에서 중간 실험별 승인 없이 자동 진행하고, 상한 초과·목표/자료 계약 변경 시 해당 부분을 멈추고 보고하겠습니다.** 현재는 첨부 문서 A 승인 gate에 따라 사용자 긍정 응답을 기다리며 구현·모델/NPZ/TEST 열람·fixture·학습·추론·commit/push는0이다.
