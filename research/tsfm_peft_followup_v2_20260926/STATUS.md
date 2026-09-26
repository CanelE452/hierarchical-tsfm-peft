# 채널 압축 잔차 PEFT 후속 개발 v2

- 상태: COMPLETE_PUBLISHED — 개발·비용·최종 검산 및 결과 게시 완료. 과학적 판정은 부분 개선/주력 확대 보류이며 승인 attempt 초과를 함께 보고한다.
- 승인: 현재 스레드 목표에 기록된 전체 v2 승인. PLAN.md는 직전 계획 응답 원문이다. v1 종료 계약은 원본에 보존하고 v2에 자동 적용하지 않는다.
- 기준: main ef59f1e921bc70c93b919e095c4d3d6f048ca319, origin=https://github.com/CanelE452/hierarchical-tsfm-peft.git. 로컬/원격 동일 확인. 무관한 미추적 4항목 보존.
- 목적: CURRENT의 큰 정확도 손해를 줄이거나 더 유효한 정확도–비용 구성을 확인. 개발 완료와 방법 개선을 분리한다.
- 현재: 최종 VAL 선택은 Electricity CURRENT K8, Bull FIXED K4; RAW는 두 자료 CURRENT. K8의 개발 점수 향상은 선택되지 않은 결과로 보존.
- 예산: 신규 실자료20fit 완료, fit실패0. GPU 원장 최종3489.4492/14400초(58.157분; 실패·진단·재측정 포함). CPU 합성 optimizer 검사도 포함해야 하는 PLAN 규칙을 누락했으므로 22/20attempt로 상한 초과. 원본 기록 보존 후 정정하며 추가 실험 없음.
- 실행 상태: GPU 원장10job 모두 종료(9complete/1failed), 활성 연구 프로세스0. 캐시30,423,573bytes, 공개 후보 약8MiB로 저장상한10GiB 미만.
- 데이터: 기존 TRAIN 업데이트/VAL 선택. Electricity DEV, Bull E1/E2는 노출 개발 평가. v1 코드·봉인·기록·캐시를 덮어쓰지 않는다. 신규 확인 자료 미확보이며 새 자료 탐색 없음.
- 게시: 결과 commit `e025d70ab53a55e0e2a052809fb450206eaab1e8`를 `CanelE452/hierarchical-tsfm-peft`의 origin/main에 push했고 원격 SHA 일치를 확인했다. 이 상태 갱신은 별도 일반 commit으로 남긴다. 승인 회차의 추가 실험은 종료했다.
- 판정: Bull E1의 부분 정확도 개선은 확인. 논문 주력 확대와 배포 추가 가치 미확보/보류. 동일 FIXED RAW 대비 E1 이득은 남지만 E2 악화·새확증 미확보·시간 블록 변동이 있다. Electricity의 처리량 이점은 기존 압축 경로의 공통 이점이며 v2 새 개선이 아니다.
- 남은 결정 한 가지: 같은 E/D·G에서 F0만 학습형 시간 선형 경로로 대체하는 대조를 별도 승인 회차에 수행할지. 이번 미실행 결과를 실패로 세지 않는다.

## 변경 이력

1. 승인 후 실제 상태 재검사: v2 파일/실행 없음. 계획 원문을 스레드 메시지에서 추출해 PLAN.md로 보존. 기존 결과 재실행 없이 신규 폴더에 착수.
2. 구현 소유권 분리: 학습기·모델, 기존 예측 CPU 진단, 평가기. root가 GPU 작업을 단독 순차 운영한다. 연구 계획·변경 전 점검·실행 모니터링 원칙 적용. 승인 범위의 수정을 매번 재승인받는 추가 조건은 두지 않는다.
3. 실제 CURRENT 초기/선택 상태 진단 완료: GPU 원장 26.895648초, optimizer update0. 네 선택 모델의 저장 VAL replay 최대오차0. 고정 TRAIN96 초기→선택 MSE: Electricity0.251987→0.173418/0.180712, Bull0.604197→0.563624/0.562117. 선택값에서 TRAIN·VAL 모두 초기보다 좋아지므로 epoch1 선택만으로 학습 실패를 확정하지 않는다. 모든 epoch 가중치는 존재하지 않으며 재학습으로 채우지 않았다.
4. 저장 예측 CPU 최소 진단 완료(existing_diagnostics.json): Bull E1 CURRENT−LoRA 추가 MSE의 공간 안/밖은 seed1 +0.003393/+0.121174, seed2 +0.003439/+0.083593. Q=DD+에서 동일 비교군을 투영했고 rank4 유지. 완전 관측 벡터 진단이며 인과 분해로 부르지 않는다. E/D 비교 후 손해가 남으면 K4→8 대응 비교의 근거로 검토한다. 채널 제외 없음.
5. 첫8fit 실행 전 결정: 공동학습 불안정과 표현공간 제약을 구별하기 위해 계획대로 FIXED_ED/SLOW_ED 직접 비교. 같은 seed 초기값·표본·G rank32/LR .001/최대120epoch, VAL-only선택. CPU 합성10검사 PASS, 실제 각 fit에 gradient/freeze/갱신/복원검사 통합. 예상 배분70분은 미검증 상한 배분이며 완료시간 약속이 아니다.

6. 첫 Electricity seed92601 완료: FIXED_ED VAL0.178669(best28/stop34,90.44초), SLOW_ED0.173783(best35/stop41,240.02초). CURRENT0.167624보다 둘 다 불리하다. 두 결과 모두 정체 종료/복원오차0이며 한 seed 부분 결과로 다른 자료·seed를 중단하지 않는다.

7. 첫8fit 전체 검산 PASS: full cohort 완료, VAL최저/최초선택, 원점 schedule, 모드간 초기값, SLOW LR비율, 고정E/D불변, 저장checkpoint와예측정합. all8 정체 종료, 학습상한 미해결 없음.
8. 첫 노출개발 평가(first_modes_development.json): Electricity CURRENT MSE0.165850 vs FIXED0.172642/SLOW0.170440로 공동학습 유지. Bull FIXED는 VAL0.374377로 CURRENT0.405068보다 낮아 선택됐고 E1 MSE0.210365(기존0.267849 대비21.46%감소), E2는1.246580(기존1.156667 대비7.77%증가). E1개선은 두seed/두기간 모두이나 E2절충을 숨기지 않는다. 신규확증 아님.
9. RAW 실행 전 근거: 제안만 새로운 모드 선택을 받으면 잔차 경로의 추가 가치를 과대평가할 수 있다. RAW에도 동일2모드/2seed/2자료의8fit를 제공하고 VAL 선택 후 same-mode와 selected-vs-selected를 모두 비교한다. Bull FIXED 대응쌍부터 시작하며 원래16fit총범위를 유지한다.

10. 기본16fit 검산 PASS, 실패0, 모두120epoch 이전 정체 종료(최대70). 동일 선택 기회에서 RAW는 두 자료 모두 CURRENT 유지. Bull FIXED RES는 같은 FIXED RAW 대비 E1 MSE60.60% 감소, E2는6.48% 감소이나 E2 조건부 블록구간은0을 포함. 잔차 가치가 남으면서 E1 미압축 LoRA 손해도 남으므로 branch_decision.json에 관찰·반증·예산을 먼저 기록하고 Bull FIXED_ED K4→8 RES/RAW×2seed의4fit를 선택했다. 이 분기와 비용 측정 후 신규fit 상한20으로 추가 탐색하지 않는다. 새 확인은 미확보다.

11. K8 진행 중 초기 checkpoint를 읽어 cross-K 통제의 한계를 확인: seed92601에서 K4 PCA basis는 K8 첫4열과 비트 단위 동일, G 출력층은 모두0이나 G 앞층 초기값은 다르다. E/D 층 크기에 따른 생성 난수 소비 차이 때문이다. 같은 K의 RES/RAW·모드 비교는 대응 초기값을 유지한다. K8 결과는 같은 seed/레시피의 구성 비교이며 K만의 순수 인과 절제로 주장하지 않는다. 결과를 숨기거나 승인 상한을 늘려 재실행하지 않는다.

12. 전체20fit 검산 PASS/실패0. final_selection은 Electricity CURRENT K8, Bull FIXED K4; RAW는 두 자료 CURRENT. Bull K8 FIXED의 E1/E2 개발 MSE0.202656/1.239798은 K4보다 조금 낮지만 VAL0.413678이 K4 0.374377보다 불리해 선택을 바꾸지 않았다. final_development.json에 모든 구성·대응비교·기간·seed를 보존했다.
13. 비용 첫 neural_cost는 첫 F0 이후 allocator 잔여9,568,256bytes 때문에 중단했다. model weakref 해제 후에도 잔여가 있고 cuBLAS workspace 해제 후0이 되는 최소 재현을 cost_cleanup_check.json에 기록했다. 모델 교체/예열 전에 workspace를 비우는 수정만 적용하고 neural_cost_corrected로 전체 측정을 재시작했다. 원본 실패 attempt/partial과 cost_cleanup_fix.patch 보존. 초기 실패 두 측정행은 최종 비용 근거에서 제외하며 진단·실패·재실행 GPU 시간 모두 원장에 합산한다. 설치 PyTorch2.10 자체 메모리 검사와 공식 CUDA workspace 설명을 근거로 삼았으며 전역 환경은 바꾸지 않았다.

14. 비용156행 완료: 병합 LoRA4checkpoint 정합 PASS(max1.431e-6),24checkpoint 저장 VAL replay PASS,모든 교체 allocator기준0. 선택 RES batch1/batch4는 Electricity26.623ms/140.334origins/s, Bull26.638ms/140.529origins/s. 병합 LoRA는 각각26.979/119.910,26.066/148.828. Bull 추론 속도 이점 미확보. Electricity 처리량 이점은 RAW도 비슷해 압축 경로 공통 이점이다. Bull F0 block별11.483/25.787/27.160ms 등 큰 drift가 있어 작은 속도 차이의 안정적 우열은 보류한다.
15. 최종 CPU 검산의 최초 seed grouping 오류를 실패 JSON에 보존하고 checker만 수정해 retry1 PASS.68예측행·40seed평균그룹의 최대 산술오차5.385e-14,비용요약오차0. 신규20fit 초기→선택 state에서 Bull FIXED K4 RES는17920scalar 변경/E/D0, K8RAW는epoch0선택으로변경0. 기존v1초기checkpoint 부재는 미확인으로 명시. 검산 wall4.150초/CPU4.000초, 새로운 inference/fit0.
16. 최종 공개 사전점검: 원자료·예측배열·weight·secret 없음, PLAN hash 유지/v1 191파일hash일치. 기존 CURRENT 진단3그림과 최종 방법·학습·오차·비용5그림을 원자료·선택·source hash와 연결했다. 검토는 실행팀 자체 검산이며 독립 재현이 아니다.
17. 게시 전 승인 범위 재확인에서 attempt 계상 오류 발견: CPU 합성 optimizer 단위검사2회(각2step)를 실자료 fit와 구분해 상한에서 제외했으나 PLAN은 별도 optimizer 검사도 포함하도록 명시했다. 22/20attempt로 초과한 사실을 TOPIC과 여기에 정정하고 사용자에게 알렸다. 원본 PLAN/테스트receipt를 바꾸거나 예외를 소급 추가하지 않는다. 추가 계산은 하지 않는다.
18. 최초 확인의 최소21회 추산을 실제 세션 출력으로 정정: 합성검사는 성공2회/실패0회(각10tests,optimizer2step)였다. 첫 출력chunk861d1d는unittest1.148초/tool3.142948초,둘째chunk066dd5는0.849초/2.8370554초다. 기존 training_tests.json은 둘째만 기록했으므로 원본을 보존하고 attempt_accounting_correction.json에 누락·계상 오류를 기록했다. 총22/20회이며 추가 GPU시간은0이다.
19. 관련138파일을 개별 staging해 일반 commit/push 완료. v1·무관한 미추적4항목은 포함하지 않았다. diff 공백검사에서 보존용 unified patch의 빈 context행은 patch 문법이므로 해당 파일에만 blank-at-eol/EOF 검사를 제외했고 다른 파일은 기본 검사에 통과했다. hooks 우회·force push·새 branch/PR·이력 재작성 없음.
