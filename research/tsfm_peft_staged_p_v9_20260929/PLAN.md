# V9 실행계획: 학습된 LEVEL을 보존하는 단계적 P 보정

사용자는 원문 PLAN/GOAL 파일을 제공한 뒤 **“이거를 /plan /goal이라고 생각하고 시행해”**라고 지시했다. 이에 계획 구체화와 후속 실행을 함께 승인한 것으로 적용한다. 설계 요청 원문은 REQUEST_PLAN.txt, GOAL 원문은 APPROVAL.txt, 현재 채팅 승인문은 APPROVAL_CHAT.txt로 byte 내용 또는 원문 그대로 구분해 보존한다. 이 문서는 실제 저장소 확인으로 구체화한 실행계획이며 원래 요청문과 같다고 표시하지 않는다. 동일 범위 구현·수리·실행·게시마다 재승인을 요구하지 않는다.

## 목적과 확인된 출발점

목적은 학습된 LEVEL을 고정한 뒤 P 보정만 학습하면 기존 LEVEL의 장점을 보존하며 v8 공동 재학습보다 유효한 후속 후보가 되는지 판단하는 것이다. 양성 결과·모든 자료 승리·범용성·최초성·논문 채택은 완료 조건이 아니다. 새로운 PEFT 주제나 새 자료·Hog 튜닝·K/rank/loss/비선형성 탐색은 하지 않는다.

[확인] 작업 위치 E:/CODING/proj/hierarchical-tsfm-peft, origin https://github.com/CanelE452/hierarchical-tsfm-peft.git, main 및 live origin/main 61fbbff0959294eb996bf6ae37a39dab7df2bfb3. 전역 C:/Users/User/.codex/AGENTS.md가 적용되며 조회한 상위/프로젝트 경로에 추가 AGENTS.md는 없었다. 기존 무관 untracked6개와 v1–v8는 보존한다. 다른 Python 연구 프로세스는 조회에서 발견되지 않았고 RTX4070 12GB/GPU 사용률0%, 드라이브 가용약480GB를 확인했다. 이 값은 시작 시점 조회이며 실행 전에 다시 충돌을 검사한다.

[확인] v9 폴더/동명 cache는 생성 전에 없었다. 공개 산출물은 research/tsfm_peft_staged_p_v9_20260929/, 배열·새 adapter checkpoint는 .cache/tsfm_peft_staged_p_v9_20260929/에 둔다. raw/백본은 복제·다운로드하지 않는다.

## 부모와 실제 재사용

아래 경로는 저장소 루트 상대경로이며 네 checkpoint의 존재/manifest SHA256 일치를 읽기 전용으로 확인했다. 부모 재학습은0이다.

```text
Robin v6 LEVEL, C17/K5
.cache/tsfm_peft_level_confirmation_v6_20260928/runs/v6_robin_level_res_92601/best.pt
fffabef217686fdd7193206ee6011f833b696b6bbcaa3655d6c4e8d4f88ca647
.cache/tsfm_peft_level_confirmation_v6_20260928/runs/v6_robin_level_res_92602/best.pt
b65baa1d1dc13d0146e238345b8aba4af38718cc62fbdcb72d78a8f0c568d02c
Jena v7 LEVEL, C21/K6
.cache/tsfm_peft_practical_controls_v7_20260928/runs/v7_jena_level_res_92601/best.pt
2dd129ab4483e1cc359647f9522c35afff74d01f8d45578d5c92943134ec0459
.cache/tsfm_peft_practical_controls_v7_20260928/runs/v7_jena_level_res_92602/best.pt
4057b762a43c83d6d762cbd15fe12857cc481226e2ddcd08da905643c359e5e8
```

각 부모의 실제 model_v6.py/model_v7.py restore_model을 고유 모듈 이름으로 불러 학습된 state까지 복원한다. 초기 함수 LEVEL_ONLY나 v8 Q head를 부모로 대신하지 않는다. 부모 초기 파일·코드·선택 epoch·과거 적합 시간·기존 VAL/TEST 예측도 reuse_manifest에 hash로 연결한다. pin된 Chronos-Bolt-small revision은772f3d25d38aec6d914c8949dab4462e2d46f5d8, FP32다.

Robin 배열은 .cache/tsfm_peft_level_confirmation_v6_20260928/data/{trainval,test}.npz를 재사용한다. 각각 hash04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14 / ecb32c192c56140a2b896b16d7936f8e9a983d78bef0bdec1ef6d35e2ddb6aa1이다. Jena는 .cache/tsfm_peft_practical_controls_v7_20260928/data/{jena_trainval,jena_test}.npz, hash29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933 / f188ee288c1a3c4722d673a4882a0c271f79ea3f161a725f976869da780161e5다. TRAIN 통계/PCA/입력대체/mask를 다시 적합하지 않는다.

```text
자료    TRAIN/VAL/TEST-A/TEST-B 원점   간격       L512/H48
Robin   1945 / 30 / 40 / 40           1시간      512시간/48시간
Jena    25648 / 371 / 365 / 365       10분       85시간20분/8시간
```

분할·컬럼·원점·mask의 전체 계약은 byte 동일한 data_contract_robin/jena.json과 protocol.datasets에 연결한다. Robin TRAIN [2016-01-23,04-15), VAL [04-17,05-18), A [05-20,06-30), B [06-30,08-10). Jena TRAIN [2024-01-01 00:10,07-01), VAL [07-01,09-01), A [09-01,11-01), B [11-01,2025-01-01)이며 원래 시간 인덱스를 바꾸지 않는다. 두 자료 모두 노출 개발자료다.

## 방법과 가까운 선행

P=UUᵀ, Q=I−P는 각 자료의 고정 TRAIN PCA이고 deltaX=X−stopgrad(last(X))다. 부모 f*는 같은 자료·seed의 학습된 LEVEL이다.

```text
BASE_LEVEL    = f*(X)
STAGED_P      = f*(X) + H_P(deltaX P)
STAGED_RAW    = f*(X) + H_RAW(deltaX)
보조 정책      = f*(X) + alpha H(input), alpha∈{0,.25,.5,.75,1}
```

새 H는 공유·무편향·무활성화512→32→48, 학습 변수17,920개다. 부모 F0/E/D/G·buffer·통계·eval 상태를 모두 고정하고 optimizer에서 제외한다. 부모 G까지 누적 적합 변수는35,840개지만 이번 갱신은17,920개뿐이다. 배포에는 동결 백본도 필요하다. P 수준을 다시 가산하지 않는다.

새 H의 down과 zero-up은 실제 v8 initial/{dataset}/edg_seed_{seed}.pt의 retained tensor를 복사한다. 학습된 v8 H는 쓰지 않으며 두 새 head 객체는 독립이다. 동일 seed 두 방법의 초기값·부모·TRAIN 순서가 같고 초기 전체 함수는 학습된 부모와 같다. RAW는 같은 학습된 LEVEL 위의 추가 원입력 보정으로, 과거 RAW32/RAW64와 다른 대조다.

Side-Tuning(ECCV2020)의 고정 base+side 결합, Friedman(Annals of Statistics2001)의 단계적 가산 및 축소는 선행이다. Side-Tuning 최종 방법절과 Friedman1999 preprint 방법절을 읽은 범위를 PRIOR_ART.md에 구분한다. AdaPTS/NLinear 및 VAL 선택 편향은 v7의 확인 범위를 재사용한다. v9은 부모 예측계수1, P에 한정한 단일 선형 보정, zero-up 초기 함수 보존, 학습 후 유한 VAL alpha 선택의 직접 비교다. 조합·alpha fallback만으로 신규성을 선언하지 않는다.

완전관측·동일가중 제곱오차에서 직교 P/Q 손실이 분리되므로 필연적 gradient 충돌을 원인으로 단정하지 않는다. v8와의 차이는 부모 학습·Q 고정·head 최적화·clip/scheduler/checkpoint 선택까지 포함한다. STAGED_RAW가 같은 추가 단계와 선택 기회를 받는 핵심 대조다.

## 실행과 선택

STAGED_P/RAW × Robin/Jena × seed92601/92602 = 기본8fit을 고정한다. 한 자료 결과로 다른 자료를 취소하지 않는다. 부모 재학습0, 기존 BASE/JOINT_PQ_V8/RAW64/F0/LoRA/DIRECT 예측은 재사용한다. 새 부모 복구가 불가피하면 기술 예비4 내에서 정확한 기존 레시피로만 복구하며 그 새 부모를 대응 기준으로 삼는다.

새 head만 AdamW LR1e-3, wd0, clip1, batch4, 512 TRAIN origins/epoch, 최대120epoch, strict VAL nonimprovement6, ReduceLROnPlateau factor.5/patience2/relative threshold1e-4. Robin24/Jena144 위상 sampling, 동일 seed/epoch 원점 순서. 전체 원채널 observed-mask macro MSE로 학습한다. 새 projected loss/LR/seed 탐색 없음. 이번 고정 계획에는 자동240epoch 연장을 넣지 않는다. 상한에서 개선 중이면 수렴 제한을 보고하며 정상 음성을 버그로 재분류하지 않는다.

기본 best는 alpha1 전체 VAL MSE 최소(step0 포함), 정확 동률은 먼저 나온 상태다. best-trained는 epoch>=1 중 alpha1 전체 VAL 최소 한 개이며 별도 저장한다. 그 checkpoint 한 개의 5개 alpha를 같은 전체 VAL로 선택하며 동률은 작은 alpha다. 각 run이 동일 기회를 갖고 총40개 보조 VAL 후보를 계상하되 새 fit0이다. alpha1이어도 basic-best와 best-trained checkpoint가 다르면 동일 모델이 아니다. alpha0은 부모 복귀이지 추가 예측 효능이 아니다.

두 자료의 모든 학습·기본 선택·alpha 선택을 마친 뒤 두 TEST 비교를 수행한다. 이후 신규 fit/alpha/채널/기간/주지표 변경 금지. 반복 VAL 사용의 선택 편향을 보고하며 VAL fallback은 TEST 비열화 보장이 아니다.

## 검증과 평가

부모 및 P 입력 계산은 no_grad, 새 H는 autograd 안에 둔다. wrapper.train 호출에도 부모.eval을 강제한다. 전체 부모 parameter/buffer hash·optimizer 제외·새 head의 실제 업데이트·첫 두 step gradient·유한 loss·선택 및 optimizer/scheduler/RNG 복원을 확인한다. 초기 zero-up에 따른 첫 down gradient0과 영구 단절을 구분한다.

초기 full output=부모는 최대절대차0, 부모 기록/출력 parity는 atol1e-5/rtol1e-4, 선택 checkpoint 재생은 atol1e-6/rtol0. STAGED_P correction Q는 atol1e-5/rtol0, 최종 prediction Q와 부모 Q는 atol1e-5/rtol1e-4로 고정하고 max차·위반원소수를 남긴다. 큰 차이를 통과시키기 위한 tolerance 확대는 없다. 이 성질은 미래 전체 MSE 보장이 아니다. 합성 optimizer 검사는 한 세션 전체 최대10update로 묶고 실제 자료 smoke는 별도 실행하지 않는다.

필수 비교는 기본 P 대 부모/JOINT_PQ/대응 RAW와 실용 F0/LoRA/DIRECT/RAW64, 보조 alpha 정책 대 기본 및 부모다. MSE 주 지표, MAE·seed·기간·채널·signed bias를 보조로 남긴다. 전체는 채널별 기간 SSE/count 합산 후 채널평균, seed 손실 평균이며 ensemble이 아니다. 자료 간 MSE를 합치지 않는다. 공간 분해는 동일 완전관측 target subset만 사용하고 coverage를 보고한다. 결측 target을0으로 투영하지 않는다. 블록 bootstrap은 Robin7/Jena42원점,기간 내부2,000회,고정 seed이며 학습/방법선택 불확실성 전체를 포괄하지 않는다.

## 조건부 비용과 예산

실용적으로 유지 검토할 nonzero 후보가 남은 자료에서만 같은 회차 비용을 측정한다. Robin 이득에 Jena timing을 쓰거나 반대로 이식하지 않는다. 자료마다 BASE2, P기본2/P-alpha2, RAW기본2/RAW-alpha2, JOINT2,F01,병합LoRA2,DIRECT2의 최대17instance에 B1/B4를 적용한다. alpha0은 같은 부모로 묶고, 그 외는 부모·head tensor·입력 정책·alpha가 실제 같은 경우만 중복을 줄인다.

FP32/TF32off/CPU4threads, 공통 VAL24, 예열10,전체24원점20passes,3순서블록. CPU표준화입력→전체CPU출력이며 온라인 부모+head/H2D/D2H·reshape를 포함한다. load·disk·채점·원장은 밖이다. 학습 cache를 배포 timing으로 쓰지 않는다. F0 B1/B4 전후sentinel을 각 블록에 둔다. 미압축 F0/LoRA B4의 M=B*K(Robin20/Jena24)도 같은 세션에서 포함해 소배치 대안을 유지한다. 최대37configs×3+sentinel12=123행/자료, 양쪽 최대246행이며 동일 함수 중복은 줄인다. batch1 latency/batch4 throughput/allocated/reserved/배포bytes를 분리한다. CPU배포/profile/compile/양자화 추가 없음. 불안정하면 비용 미확정으로 제한한다.

부모 과거 학습 이력+새 head 적합은 총 적응 비용의 역사적 합산으로만 보고한다. 이번 재사용 증분 GPU시간과 구분하고 same-session 학습 가속비로 주장하지 않는다.

```text
신규 fit        기본8 + 기술복구/부모복구 예비4 = 최대12
GPU             1장 / 동시학습1 / 작업점유7200초
계획 배분       부모·검사600 / 학습·VAL3000 / 평가900 / 비용1800 / 복구900초
CPU 합성        최대6세션 × 세션 전체10update 이하
CPU 검사        누적900초, 실패도 포함
저장·다운로드   신규5GiB / 다운로드0
충돌 대기       회당30분·누적60분; 다른 작업 종료 금지
```

배분은 [추정·미검증] 운영 계획이며 시간 예측이나 학술 합격선이 아니다. 총상한 내에서 배분을 조정할 수 있지만 자동 증액/예산 채우기는 없다. 모든 fit·실패·재시도·합성 optimizer·GPU/CPU검사는 공통 원장에 먼저 예약한다. 부모GPU job과 내부fit 시간을 이중 가산하지 않는다. root만 실행을 시작하며 worker는 코드 작성/읽기만 한다. 전역환경·driver/power·새 서버·유료서비스·권한 우회 없음.

## 결과별 결정과 산출물

두 자료에 P 이득이 남으면 조건부 대체 후보로 검토하고 비용·단순대안·반례를 함께 보고한다. alpha0 복귀가 대부분이면 함수 보존과 추가 효능 미확보를 구분한다. RAW도 충분하면 P 제한의 고유 기여를 축소한다. Q 보존에도 악화되면 Q 재학습만이 원인이 아니며 새 모듈을 무단 추가하지 않는다. 한 자료만 개선이면 그 조건부 발전을 인정하면서 TEST 기반 자료별 routing을 만들지 않는다. 기술무효/접근제한/상한은 방법 음성과 구분한다.

PLAN/APPROVAL/STATUS, 부모·자료·초기값 manifest, 실행별 result/curve/receipt, 기본/alpha 선택, 비교JSON/CSV·Q보존·P/Q진단·그림, 조건부비용, TOPIC_DECISION, 짧은 방법 결과 업데이트, 최종검산/게시receipt를 연결한다. 부모 및 예측 배열은 local cache에만 둔다. 같은 범위 오류는 원본/영향/diff를 보존하고 수리한다. 기존 원고 전체를 덮어쓰거나 논문을 제출하지 않는다.

예정 비교 완료 또는 실제차단/상한이면 수행/미수행과 다음 결정 하나를 보고한다. 관련 신규파일·색인·기존 history체계만 개별stage·일반commit·origin/main push하고 live SHA를 확인한다. force/reset/이력삭제/hook우회/새branch 없음. 재개 시 실제PID·원장·STATUS부터 확인하며 중복 적합하지 않는다.
