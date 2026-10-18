# B2 + RevIN 최소 실행 계획

승인 근거는 현재 사용자 목표의 8개 항목이다. 첨부 통합 문서는 기술 명세를 확인하는 자료이며, 그 안의 더 넓은 실험 목록과 12/24 epoch 규칙은 이번 범위로 승격하지 않는다.

목표: 같은 B2 부모에서 원채널 창별 RevIN 하나의 효과와 실제 GPU 추론 비용을 확인하고, 검산 가능한 세 묶음을 main에 게시한다.

구조: 기존 데이터·FrozenBolt·B2·AdamW·선택/refit 코드를 읽기 전용으로 재사용한다. 신규 코드와 산출물은 `research/b2_revin_20261018_v1`, `reports/b2_revin_20261018_v1`, 전용 private cache에 둔다. 원 E: 사용자 체크아웃과 이전 봉인 source/receipt는 보존한다.

## 고정 계약

- Jena C21/K6/L512/H48, PeMS32 C32/K8/L512/H12, seed92601/92602.
- 기존 causal fill와 A_fit global z-score 뒤 원 C채널 RevIN; biased variance, eps1e-5, detached mean/std, gamma1/beta0. 공식 inverse 분모 gamma+eps². 창별 통계는 per-call context.
- OFF는 원 모델 자체. ON/OFF P/E/D·PCA center·global scaler 초기값을 같게 유지한다. anchor를 포함한 forecast 전체에 inverse 한 번, loss는 기존 global-normalized 원채널 macro MSE.
- FP32, AdamW lr.001/weight_decay0, clip1, effective batch16/micro4, F row128, checkpoint_inputsFalse. F weights/eval 고정, 입력 gradient 유지.
- 초기 선택 checkpoint 0/.25/1/3: 기존 OFF와 같은 update 경계를 보존한다. A_fit train stride1, A_stop/V strideH. strict A_stop MSE minimum, 정확한 동률이면 이른 epoch.
- 해당 데이터의 네 대응 arm 중 하나라도 epoch3이 strict best이고 직전 epoch1보다 개선되면, 그 데이터의 OFF/ON·두 seed 전부 epoch4/5/6까지 확인한다. epoch6이 같은 조건이면 TRAINING_CAP_STILL_IMPROVING으로 명시한다.
- 확장에는 search의 마지막 model/optimizer/sampler/CPU+CUDA RNG 상태를 사용한다. old receipt/config를 수정하지 않고 새 branch의 원 checkpoint SHA와 import proof를 남긴다.
- cap3 대응 비교와, 조건부 cap6 대응 비교를 각각 보존한다. 각 cap의 선택 뒤 fresh A60 scaler/PCA/모델을 해당 epoch까지만 refit한다. 선택/확장/N 결정은 V를 읽기 전에 봉인한다.
- 데이터별 두 seed의 최소 A_stop MSE 평균으로 공통 N(OFF/ON)을 정하고 동률 OFF. Jena 구성 기여는 같은 N·일정·장치의 B0 두 seed만 추가한다. B0 학습의 cap은 Jena 최종 B2 cap과 같다.
- Raw·Ridge·DirectG·LoRA는 기존 검증된 prediction/source를 연결한다. 최종 accuracy–cost에 필요한 Raw/DirectG/LoRA/B2 OFF/ON과 Jena B0만 GPU B1/B8 보완한다. Ridge 원 FP64 NumPy 비용은 기존 CPU 범위로 표시한다.
- D 결과는 이번 최종 주장의 근거로 쓰지 않는다. 따라서 PeMS D seed2는 조건부 미적용이다. SOFTS/FreTS/LIFT/N-HiTS/CycleNet/TimesNet/ForeCA/fullAdaPTS/K·init sweep은 제외한다.
- V는 반복 개발 평가이며 독립 테스트가 아니다. T·M_cal로 선택하지 않는다. 기존 T 노출과 pretraining overlap UNKNOWN을 공개한다.
- 유료 자원·새 dataset/model 다운로드 없음. GPU worker1, 비용과 학습 격리. 이전 원장을 리셋하지 않고 새 실제 작업 시간을 별도 기록한다. 별도 사용자 hard budget은 없으며 3/6 epoch 범위만 유한하게 실행한다.

## 구현과 검증

- [ ] `revin.py`/`deployment.py`: OFF parity, 공식 수식, constant/causally filled 창, interleaved context, B1/B8·순서 독립, anchor inverse, F 입력 gradient를 CPU fixture로 검증.
- [ ] `training_jobs.py`: stage3/6 immutable continuation, sampler/RNG·tail updates 동일, fresh refit, V 선택 금지, source/data/F/scaler/cache binding 검증.
- [ ] `run.py`/`contracts.py`: 4 ON과 검증된 4 OFF, 데이터 단위 확장 decision 및 conditional B0, safe resume·등록 ID·source seal·resource ledger.
- [ ] 기존 OFF의 CP0/terminal/refit/prediction/source/recipe를 실제 대사한다. exact reuse 불가이면 원인과 신규 matched OFF 실행을 기록한다.
- [ ] 실제 순차 GPU 실행 후 cap3 및 final cap의 saved predictions를 독립 FP64 충분통계로 재집계하고, Raw/강한 기준선의 cohort/mask/origins/scaler를 연결한다.
- [ ] scored bundle을 두 번 reload한 물리 입력 전체 경로의 full-V parity와 B1/B8 비용을 측정한다. RevIN norm/inverse·2C params 포함. cache forecast로 latency를 대신하지 않는다.
- [ ] 정확도–비용, RevIN 대응 비교, 구성 기여·한계의 CSV/그림/한국어 분석과 학습 곡선을 생성하고 public source-only replay를 실행한다.
- [ ] fresh 읽기 전용 reviewer의 material finding을 고친다. 필요한 focused tests·private verification·public replay·보존/공개 검사 후 explicit staging, 정상 main push, 실제 원격 blob 검증.

## 검토 초점

같은 B2라는 이름으로 CPU B03와 CUDA C02를 섞지 않기; epoch endpoint tail 업데이트 차이; deployment가 forward를 우회하는 RevIN 누락; selected refit을 search continuation으로 착각하기; V를 본 뒤 정규화·확장 결정 바꾸기; 전체 anchor inverse 누락; 통계 context 교차 오염; GPU 비용과 기존 CPU 비용을 섞어 speedup 주장하기.

파일 소유: RevIN worker는 revin/deployment와 수학 테스트, training worker는 staged training와 상태 테스트, root는 등록/실행/검산/게시, 별도 reporting worker는 표·그림·한국어 해석을 맡는다. 다른 작업자의 변경을 되돌리지 않는다.
