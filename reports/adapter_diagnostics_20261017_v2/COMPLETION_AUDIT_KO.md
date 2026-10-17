# 원 지시문 완료 감사

[확인] 원 지시문 0~12장을 [요구사항별 대조표](REQUIREMENT_COVERAGE.csv)의 81개 항목으로 대조했다. 판정 기준은 2026-10-17 19:19 UTC에 동결된 실제 산출물과 검증 영수증이다. 기록된 실험·보고 범위는 검증되었으나 과학적 완료는 `SCIENTIFIC_PARTIAL`이며, 최종 게시 상태는 `PENDING_REMOTE_VERIFICATION`이다. 전체 `ENGINEERING_COMPLETE`는 이 문서에서 선언하지 않는다. M5는 후속 로컬 자산 발견으로 접근 상태가 교정되었으며, metadata-only 범위로 검증되었다.

감사 원문은 `CLI_TSFM_Diagnostics_20261017_KO.txt`이며 SHA256은 `6fc62193d0a48c319d339819e409ef25dad06ce4f4a3acdffb0908f4292e2955`다. 시작 main/HEAD는 `2bbdd09dc5763ed381bb86e385de58b9a6351057`이고, 현재 계획 SHA256은 `fac668f230fe99ab914bcd8192c8f6a559d5ecc794f990737d7e33997372a081`이다. 과거 시도·추가 job·교정 경로는 [START_STATE](START_STATE.json), [계획](EXPERIMENT_PLAN.json), [시도 이력](ATTEMPT_LINEAGE_KO.md)에 보존되어 있다.

**검증한 범위**

[확인] [VALIDATION](VALIDATION.json)은 19:19:02.396192 UTC 기준 exit0/PASS이며, [최종 전체 테스트 영수증](TEST_VALIDATION.json)은 19:16:28.864640 UTC 기준 `182 passed in 20.59s`, exit0다. 감사자는 테스트를 재실행하지 않고 이 동결 영수증과 공개 CSV의 실제 상태를 대조했다. 과거 테스트 통과 기록을 최종 소스 검증으로 대신하지 않았다.

G1은 실제 pre-T 경계·mask·관측 0·scaler 및 합성 fill 범위와 별도 `G1_M5_LOCAL_ASSETS=LOCAL_ASSETS_SCHEMA_VERIFIED` 범위, G2는 D01의 적용 가능한 공식 경로·frozen F·입력 gradient 범위, G3는 계약 동등성이 확인된 비교 부분집합, G4는 등록 합성 solver 범위에서 통과했다. G5의 실제 상태는 `CURRENT_RUNTIME_AND_HISTORICAL_EVIDENCE`다. 이를 원 CPU 학습의 완료나 CPU/CUDA bitwise 동일성으로 확대하지 않는다. Moirai의 adapter 검사와 Chronos2 native Raw의 gradient/optimizer/channel-isolation NA도 구분했다.

[확인] G6는 총 147개 score 행의 공개 채널별 충분통계에서 FP64 macro MSE/MAE를 독립 재집계했다. 구성은 RUNS 59행과 SAME_UPDATE 88행이다. 다만 **147개 전부의 horizon×channel 통계가 완비된 것은 아니다.** C02의 보조 A_fit train endpoint 16개는 채널 marginal만 저장되어 `G6_horizon_coverage`가 PENDING으로 남는다. 주 V와 등록 비교의 전체 horizon 및 세 band는 완비되어 있다. 가용 horizon×channel 자료의 4,903개 채널을 교차 검산했으며, 채널별 평균 최대 차이 `1.4163785943159984e-13`은 허용치 `1e-10` 안에 있고 정수 count는 일치한다. 큰 비정규화 합의 최대 차이 `6.746267899870872e-08`을 평균 오차와 혼동하지 않는다.

[확인] G7은 현재 source/private artifact binding을 확인했다. G8은 109개 job을 누락 없이 집계했고, 별도 독립 검증은 [paired-origin coverage](PAIRED_ORIGIN_COVERAGE.csv)의 등록 고유 pair 118/118, [horizon band](HORIZON_BAND_SUMMARY.csv)의 303/303을 COMPLETE로 확인했다. Jena band는 H1..16, H17..32, H33..48이며 각 band의 채널별 SSE/SAE/count를 먼저 합친 뒤 masked channel macro를 계산했다. 같은 origin을 독립 실험 반복으로 간주한 CI·p-value는 없다. 실제 sampler snapshot은 A03 24/24, B03 job은 2/2, fit-run join은 6개 확인되었다.

G9의 동결된 `TYPES_AND_SIZE` 검사는 PASS(당시 공개 파일 1,208개, 165,579,208바이트)다. 이 수치는 아래 감사 파일을 포함해 최종 게시 패키지를 재생성하기 전의 범위다. 최종 staged allowlist/checksum 검사와 원격 main 포함 확인은 게시자가 뒤이어 수행할 단계로 남는다.

**실험·비교의 실제 상태**

[확인] [JOB_STATUS](JOB_STATUS.csv)는 109개 중 COMPLETE 106, SKIPPED_BUDGET 2, FAILED 1이다. 원 CPU `C01_pems32_B0/B2`는 실제 profile에 따른 pair 예산/예비분 부족으로 생략되었다. 별도 CUDA B0/B2 pair는 완료했으나 이를 원 CPU exact continuation 완료로 바꾸지 않았다. 기존 `E04_jena_lora` 실패는 원시 상태에 남아 있고, calibration parent의 기존 manifest와 deployment identity 범위를 구분한 `E04_jena_lora_parent_v2`가 성공하여 해당 비용 공백을 해소했다. 실패 행을 삭제하거나 COMPLETE로 덮어쓰지 않았다.

A02는 두 seed의 저장 배열을 FP64로 재채점했다. A03은 B0/B2/CAP×2 seed×4 endpoint×3 split의 72행을 갖춘다. B01은 고정 PE의 full-history probe 6개, B02는 Jena21/PeMS32의 own/cross Ridge, B03은 두 seed의 original/shared/Dh와 fit/V를 분리했다. B01은 lambda1 고정이며, B03은 A_fit PE로 A_stop lambda를 고른 뒤 대응 A60 PE를 fresh refit하는 계약이다. C02는 PCA/QR×두 seed의 실제 초기화 진단 4행이다. D02는 Moirai Raw/B0/B2의 두 seed trio, D03는 Jena21과 PeMS170의 native 실용 대안을 포함한다.

[확인] [비용 CSV](COST_FULL_PRECISION.csv)는 42행(COMPLETE 38, COMPLETE_PROTOCOL 4)이다. B1/B8의 warm10/timed20과 전체 physical 경로→FP64 inverse→CPU return을 구분하며 warm/cold/fit, parameter/resident/logical F/RSS/GPU peak 범위를 보존했다. Jena LoRA와 PeMS170 Raw/B2/Ridge/LoRA/native의 E04 비용은 새 학습이 없는 `EXISTING_SELECTED_ROUTE_DEVICE_PARITY_REPLAY`다. 전체 V Jena43/PeMS148 origins의 device replica 6개가 별도 RUNS 행으로 남고, 원 CPU/selected reference도 보존된다. 비용 정확도는 해당 replica 자신의 score·prediction SHA·run ID에 연결된다. B1/B8의 timed 출력 20개씩은 측정시간 밖에서 해당 origin의 저장 replica와 atol1e-5/rtol1e-4로 비교되어 20/20 PASS다. 공정 speedup은 동일 physical cohort·device/backend·dtype·thread facet에만 적용되며 Ridge CPU64는 별도 facet이다.

Chronos2 native의 seed는 None이다. native 실용 대안 비교에만 명시한 primary supervised seed92601 fallback 규칙을 적용하며 reference_seed를 보존한다. native를 supervised seed 반복으로 취급하지 않는다. 백본과 grouping이 함께 바뀌는 비교는 동일 adapter 효과의 증거가 아니다. 두 seed의 개별값과 평균/범위, MATCHED 24개·CONTEXTUAL 22개·SAME_UPDATE 72개의 차이를 유지한다.

**필수 산출물과 재현성**

[확인] 원문 필수 보고서·CSV·plan/job/ledger/validation/source binding/reproduce 문서와 F1~F6 PNG/SVG 12개, T1~T4가 존재한다. T1은 85행, T2는 134행, T3는 질문별 6행, T4는 7행이다. T2의 facet별 행 수를 고유 pair 수로 해석하지 않는다. T3에서 Q2는 PARTIAL, Q1 및 Q3~Q6은 `SATISFIED_REGISTERED_SCOPE`이며 이는 가설 입증·수렴·독립 일반화를 뜻하지 않는다. 성능 합격선은 UNREGISTERED다. B2 full-history 선형 probe의 반례와 MSE/MAE 방향 차이, Jena/PeMS Ridge 방향 차이도 지우지 않았다.

[확인] [공개 재현 영수증](PUBLIC_REPRODUCTION_VALIDATION.json)의 `--reaggregate`와 `--figures` 명령은 exit0이며 새 model/F 호출 0, private 수치 배열 접근 false다. 147행 점수를 재집계했고, 최종 F1~F5는 공개 CSV로 생성한 뒤 같은 renderer를 다시 실행하여 PNG/SVG 10/10 SHA가 정확히 같았다. 첫 in-memory writer와 CSV renderer 비교는 F3/F4의 seed 정수/실수 label 표기 때문에 4개 파일이 달랐다(6/10 동일). 이를 최초 writer까지 byte-identical이었다고 주장하지 않는다. F5의 native 실용 대안 패널은 adapter 패널과 분리되어 있다.

F6는 고정 first/middle/last origin index [0,21,42], channel index [0,10,20], seed92601의 정성 그림이다. T 접근은 false이고 원시 curve 값은 공개하지 않았다. [그림 메타데이터](FIGURE_METADATA.json)의 SHA로 묶인 승인 private curve/prediction 자산이 있어야 정확히 재생성할 수 있다. 공개 재현 명령의 10/10 판정에는 F6가 포함되지 않는다.

**한계·자원·게시 대기**

[확인] M5는 사용자 요청 뒤 넓힌 로컬 검색에서 sales validation/evaluation·calendar·sell_prices 4개 CSV와 원 ZIP을 찾았다. [로컬 발견 검증](M5_LOCAL_ASSET_DISCOVERY.json)은 SHA·크기·CSV 첫 헤더 4줄만 확인했고 `LOCAL_ASSETS_SCHEMA_VERIFIED`다. 수치 값 파싱·fit·evaluation·model/F 호출·다운로드는 모두 0이다. M5의 C/L/H·입력·split·origin·평가 계약은 `NOT_REGISTERED`, fit/evaluation은 `NOT_RUN`이다. 따라서 현재 접근 block은 Q1에서 제거되었지만 M5 품질·성능 검증을 뜻하지 않는다. [별도 설명](M5_LOCAL_ASSETS_KO.md)과 T4가 이 범위를 보존한다. 원 A01의 block은 제한된 등록 inventory와 과거 외부 HTTP401 기록이며 삭제하지 않았다. 현재 모든 로컬 폴더에 파일이 없다는 뜻이나 라이선스 거부의 증거로 해석하지 않는다. 라이선스와 독립 출처 확인은 UNKNOWN으로 남는다.

원 CPU pair의 예산 생략, C02 보조 train endpoint 16개의 h×c 미보존은 그대로 남는다. 상류 PeMS 보간 이력, pretraining overlap과 일부 legacy 원 URL/units는 UNKNOWN이다. 과거 T46 노출·반복 개발 V·fixed K의 범위도 독립 일반화로 확대하지 않는다. 이 감사는 새 학습·F 추론·private archive 재검사·추가 실험을 수행하지 않았다.

[확인] [동결 자원 장부](RESOURCE_LEDGER.json)는 wall 14,388.088579초, GPU 6,255.156718초, CPU-worker 1,157.715853초, process CPU 7,860.421875초, download reserved 166,798,057바이트를 기록한다. 한도는 wall21,600/GPU14,400초, 예비분4,320/2,880초, download5GiB/cache30GiB이며 실제 CPU worker 설정은 1/BLAS2다. actual cache byte 총량은 snapshot에 없어 UNKNOWN으로 남긴다. 이전 사용량과 실패 비용을 reset하지 않았다. 이후 게시 처리까지의 최종 wall 사용량을 이 snapshot으로 미리 확정하지 않는다.

원격 게시 판정은 `PENDING_REMOTE_VERIFICATION`이다. 최종 publisher가 감사 파일을 포함해 manifest/checksums와 staged allowlist를 검증한 뒤 정상 push하고, 최종 응답에 **실제 commit SHA와 원격 main SHA/포함 여부 조회**를 남겨야 한다. 이 문서가 들어갈 미래 commit SHA를 자체 본문에 미리 확정할 수 없다. 최종 응답의 실제 SHA와 원격 조회가 이 요구의 마지막 증거다.

**동결 근거의 SHA256**

아래 값은 실제 작은 공개 파일을 읽어 확인했다. 항목별 경로·SHA는 대조표에 있으며, 큰 공개 파일은 PUBLIC_REPRODUCTION_VALIDATION 또는 publication manifest의 파일별 SHA를 참조했다. 이 감사와 manifest/checksum 자체 SHA는 순환을 피하려 본문에 넣지 않는다. 최종 publication manifest/checksum이 감사 두 파일을 묶는다.

```text
VALIDATION.json
  06ebe19e9c5cbe94683d8df1983c5864d9c25be1450fbd0ce18d22383784c131
SOURCE_BINDINGS.json
  078131a8f3033fc2e3bc3576b04d309515ec7be65b115d4981a233e28ff39214
TEST_VALIDATION.json
  393a5b51b2a162a64cda461eb9f2ae1f7025c703d2f7e7b44801e55be44768d4
RESOURCE_LEDGER.json
  314aff8fe97285f61ba546babcea6f30d94a08ad52c589b229176864839ab353
JOB_STATUS.csv
  f69420a560a170ab6ad383b76092a8c50b92fb56f29183249c75623ccf6454fd
PAIRED_ORIGIN_COVERAGE.csv
  84de2890be5d9626152d0c3086dfe80b4a71e65d04a695afd26b1f076a4ae7ae
HORIZON_BAND_SUMMARY.csv
  15ddf7f849f46de6d4bc5da663945bd14765fade9bbb8c637fb8935317974609
FIGURE_METADATA.json
  93fbb9141125f3d24115e2f4283519fafafc4132209cad1951186e7602c41f44
PUBLIC_REPRODUCTION_VALIDATION.json
  72ba6bf1405b966c8291531c46ddf7ce8a1152a3c46cbd4463f384f3bf7afd91
M5_LOCAL_ASSET_DISCOVERY.json
  28b065be07780377a9e2a6e5188bbc9ce086e5682e0c6aa6fa489ad759c62705
```
