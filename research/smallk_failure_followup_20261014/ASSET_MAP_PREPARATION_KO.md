# Robin·Beijing 동일 조건 자산 지도

상태: **PREPARATION_ONLY**. 현재 Bi 연구 종료 뒤 root가 실행 범위를 전환할 때 사용한다. 이 준비에서는 문서·Python 소스·RESULT/contract JSON과 파일 존재만 확인했다. 실제 scientific NPZ/CP 내용, 모델, GPU, 보호 TEST는 열거나 실행하지 않았다. 아래 SHA는 **생산 JSON의 선언값**이며 이번 준비의 독립 파일 재해시 결과가 아니다. 현재 비용 계측을 방해할 대형 해시는 미뤘다.

모든 경로의 기준은 repository root `E:/CODING/proj/hierarchical-tsfm-peft`다. 원 실행 폴더와 bytes를 보존하고 새 후속 폴더만 사용한다. 경로 별칭은 아래와 같다.

```text
R = .cache/smallk_predictive_alignment_20261013/run_dev_v1/robin
RN = .cache/smallk_predictive_alignment_20261013/run_new_v1/robin
B = .cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin
A = .cache/smallk_predictive_alignment_20261013/run_checkpoint_audit_v1
P = .cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac
S = research/smallk_predictive_alignment_20261013
```

## 고정 비교 조건

[확인: 소스·JSON] 두 자료 모두 L512/H48, seed92601·92602, 동결 Chronos-Bolt-small FP32, TF32 off, 현재 상태 C개 anchor 우회와 nonlinear D를 사용했다. Robin은 C17/K4, Beijing은 C11/K2다. 엄격한 K-only 정보 압축이 아니다.

```text
자료        split     창 수   origin 첫/끝    target bounds       역할
Robin       fit       256     1040/2073       [528,2121)          training
            tune       64     2121/2472       [2121,2520)         model_selection
            val        30     2568/3264       [2568,3312)         reused_development
Beijing     fit       256      512/20990      [0,21038)           training
            tune       64    21038/28003      [21038,28051)       model_selection
            eval      146    28051/35011      [28051,35064)       exposed_evaluation
```

Robin VAL 및 이미 열린 Appliances/Beijing EVAL은 후속 작업의 개발 증거다. `run_new_resolved_v1/RESULT.json.protocol_deviation`의 원 순서 위반은 보존한다: Appliances EVAL이 먼저 열렸고, Beijing parser repair와 그 FIT/TUNE 적합·EVAL은 뒤에 진행됐다. 공동 새자료 모델 봉인 계약은 통과하지 않았다. 보호 TEST는 계속 닫는다.

## D0 정본 저장 예측

[확인: 코드 흐름] `engine.add_prediction`은 **정규화 값**을 저장한다. 물리 지표를 JSON에 함께 기록한다고 NPZ가 물리 단위인 것은 아니다. 공통 `mean/std [C]`, 각 split의 `prediction/target/observed [N,48,C]`, `origins [N]` schema다. observed는 bool, 원 source origin은 int64다. 실제 NPZ header·dtype·값은 후속 실행 때 검증해야 한다.

```text
Robin RAW
R/RAW_BOLT.npz
f936ae6aa84d6b3024cfbe0de24e79c86487c2a7ec217a8e55e52b4935eb4851

Robin MAIN92601
R/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/BOLT.npz
a40312e145e17352653a5bda229bb75f3aaee027e1b4cbe8dae10297dd502778

Robin MAIN92602
R/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/BOLT.npz
bfd8ae6ad23647048f0eb6fc0b0bb74268ce9c2c418821258db250177aa25377

Beijing RAW
B/EVAL/RAW_CONTROLS/RAW_BOLT.npz
2a557027349fc8763995ade7b8150e14e776f6bd67beb73d53cd0e84d4d7bcd8

Beijing MAIN92601
B/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/BOLT.npz
5c9985ade70b792e117b4d2273f18b82e937e2f0fcb393e3cdc174fc0a9eb2b9

Beijing MAIN92602
B/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/BOLT.npz
8fd04590facced935425be634ee2f6db4284a6696bf46cf00e9a816eb797ed75
```

Robin는 fit/tune/val, Beijing의 EVAL 폴더 안 정본 NPZ는 fit/tune/eval을 함께 포함하는 저장 흐름이다. 같은 이름의 원 FIT/TUNE-only NPZ와 혼동하지 않는다. Robin `RN/RAW_CONTROLS/RAW_BOLT.npz`는 R RAW와 선언 파일 SHA가 같다.

선언값의 출처:

```text
S/run_dev_v1/RESULT.json             7cbb19cf333211a77869726a73fbd4ee3f1a32d407688f4ca454122a6afb92c9
S/run_new_v1/RESULT.json             115dc6e7f5508e0b729acaf6118c906a83de311720a16fdc1d27321a44dba5e3
S/run_beijing_repair_v1/RESULT.json  692c2bee884fefd55f2dbbdc00daaf34805b856f115517a47d415b430899df77
S/run_new_resolved_v1/RESULT.json    45e14c4852174267babe76116f3e380e6450b981f6c09f7e84b928a6d271d858
S/run_checkpoint_audit_v1/RESULT.json SHA unknown in this preparation
```

## 선택 MAIN CP와 복원 API

```text
Robin92601 R/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/u100.pt
file SHA  9fee31847c8546a9ffcd0237271d3e8b65b45f646004e75a3be7f07167630946
codec SHA e453cda947c8812721ba2c19594132ea1c93764865528871584d94b88ffc85c3

Robin92602 R/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/u100.pt
file SHA  f32d2fbd3fbaad83c455a5fe21cd5201682c066a1dacfea870c2a02380f2280c
codec SHA cc936accf917d95e959bd7bb54f2de872b1c7d6e82a4c9b3c7cf08fe6b8df50a

Beijing92601 B/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/u400.pt
file SHA  c585c780cd49224676546d58ef87fc79ae518acaa1f658c5b33bb9c416c9000d
codec SHA 5fbe188ed8b1a9b33be3c12b1ace1ce2c7f51ed7b769edc906417ba959c483cf

Beijing92602 B/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/u600.pt
file SHA  398a0d4a2985f98feaadd0716615a1a39a47e53fee95022a70bba7209bf6d74d
codec SHA 758cb683a2e54ab31b438d51be7b81add038998c8d5dcf7005cc1ced69ac51e5
```

[확인: 코드] `controls.load_codec(case)`가 CP file SHA를 검증하고 `torch.load(..., map_location='cpu', weights_only=True)['codec']`에서 `fixed_basis [C,K]`를 읽는다. `en.Codec(np.zeros(C,np.float32), basis, seed, 'ANCHOR_NONLINEAR_D')`를 구성한 뒤 **전체 state**를 복원한다. 초기 zeros mean은 constructor placeholder이며 CP의 project/decode bias가 복원된다. 마지막으로 selected_codec_hash와 비교한다. 이 API는 `.to('cuda')`를 하므로 현재 준비 단계에서 호출하지 않는다.

`engine.Codec`의 E는 linear project + causal TCN(RF127); D는 linear decoder + `Linear(K,32)→GELU→Linear(32,C)` residual이다. 실제 경로는 다음과 같다.

```text
z_past [N,512,K] = E(x)
z_last [N,1,K]   = z_past[:,-1:]
z_F    [N,48,K]  = F(z_past)
prediction       = x_last + D(z_F) - D(z_last)
latent delta     = z_F - z_last
```

F constructor는 `en.ex.previous.FrozenBolt()`, 이어 `en.replace_normalizer(bolt)`다. 로컬 model snapshot은 `.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8`; 생산 모델 state SHA는 `c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874`다. snapshot의 개별 weight 파일 SHA 및 현재 설치 Chronos source SHA는 이번 준비에서 unknown이다. frozen median quantile(.5)을 사용한다. `ZeroSafeInstanceNorm`은 같은 forward와 상수 분산의 유한 backward convention을 적용한 기존 조건이다.

## 원 source·정규화·과거 창

Robin의 기준은 `S/run_dev_v1/robin_contract.json`이다. 원 TRAIN/VAL source는 `.cache/tsfm_peft_level_confirmation_v6_20260928/data/trainval.npz`, SHA `04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14`다. `common.load_data('robin', sizes=(256,64,None))`는 저장 x/mean/std에서 FP32 물리값을 역변환하고 원 finite mask를 보존하며 FIT만으로 통계를 계산했다. **정확 재현에는 이번 contract 또는 원 prediction NPZ의 저장 mean/std를 사용하고 재계산값으로 바꾸지 않는다.** 전체 과거 입력은 `ex.arrays(data,512)`/`ex.preprocess`: 각 창 시작을 FIT mean으로 reset한 causal observed-only FF다. origin o에서 과거 `[o-512,o)`, 미래 `[o,o+48)`이고 last는 전처리된 과거 마지막 값이다.

Robin origin array SHA는 fit `db58ec09a892cd5bd9a932d268351c646aa10542f9dfb2d89fa3bdf2694d2cc1`, tune `e131da45c7670183aafb89adadae732ee54fe7d456464f7358c277b569b74e4a`, val `1364df723f7c92c3f2bb41a52c573eed3fa8572b8f8753cc4971e5312678feea`다. 채널명 17개는 원 contract의 `columns`를 그대로 가져온다; 끝 채널이 Robin_office_Zelma다.

Beijing 기준 contract는 `P/contract.json`, SHA `9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f`다. 원 zip은 `.cache/smallk_predictive_alignment_20261013/raw_dataset/beijing_aotizhongxin/download/uci_501.zip`, SHA `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`. parser repair가 `PRSA2017_Data_20130301-20170228.zip!PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv` 하나를 선택했다.

```text
P/fit_tune_payload.npz
78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277
P/eval_past_input_payload.npz
18710be9776de39cd09a4e1b3844c4108b92edb778253b0447ccdb13e6e67ca0
P/repair_eval_past_physical_payload.npz
d90da33192522c2d39684d5880efd449ad83bb9bc98ef3b16c0317286dca7e31
```

Beijing channels는 `PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM`, interval3600초다. FIT mean/std array SHA는 각각 `b4d71a45e629117b3e1c3d4d99b0815703c293579698b0f7d544350fe27e962e`/`c420046547719b94ff8acd4d0435965ddad22ffb31671864153efe9d628e0ff1`. 원 parser의 FP32 F-contiguous reduction과 저장 NPZ의 C-layout은 같지 않을 수 있으므로 **저장 통계를 재사용**한다. `newdata_repair.load_training(dataset, S/METHOD_LOCK.json)`은 등록된 FIT/TUNE payload를 읽고 `newdata_repair.load_evaluation_inputs(...)`는 저장 normalized EVAL history만 읽을 수 있다. 후속 EVAL target은 이미 노출된 정본 prediction NPZ의 target/mask를 재사용하면 된다. 원 `open_evaluation`의 첫 개봉 흐름을 다시 실행할 필요는 없다.

## FIT teacher·last·basis

```text
Robin 각 seed: R/TEACHER_FIT_s92601.npz, R/TEACHER_FIT_s92602.npz
두 파일의 선언 SHA 동일:
698cb8fcb6decf5a7ca666d73e1985a7309012bb078866b059a2bf1ad310d9b3
keys forecast [256,48,17], last [256,17], basis [17,4]
basis array SHA 7a7069bff99b28386f3a631c2bb151acfc36cfee16d454a61d3a485cb7c5fb42
출처 S/run_dev_v1/robin/TEACHER_s92601.json, TEACHER_s92602.json

Beijing92601 B/TEACHER_FULL_IDENTITY_s92601/FIT_TEACHER_BASIS.npz
d4ff7281f8a966f26c422ddf3dbabb71c6ba3fb44edf925afedad513762c602c
basis array SHA e1a464f6677dafb4fffbf7985c55240319a53ccfe33759c643ca1f1f53686bcf
teacher selected u200 CP SHA d879d7c3c3ebcc385ede48d8dacba1a069d98e03f6b8b25ac1116ba4a080fbd2

Beijing92602 B/TEACHER_FULL_IDENTITY_s92602/FIT_TEACHER_BASIS.npz
b2158eca431f01aebc40cffca17d3c74deee7ad1693d775f6c90e4feaf60d90d
basis array SHA 0771fd900c1b22a9a0fabf6192924ea34eea73c377f9420395058d2edfb9e730
teacher selected u600 CP SHA 5ce5dc1fb7ae1264fbc39f27a903efb702eb3b37965aca45b80aed76a67862fb
Beijing keys fit_forecast [256,48,11], fit_last [256,11], basis [11,2]
```

Robin teacher는 이전 full-C identity PRED의 선택 u0이다. scope 문서의 RAW equality 증거와 일치하는 조건이지만 이번 준비에서 배열 maxdiff를 재계산하지 않았다. Beijing teacher는 실제 학습된 full-C E/D이므로 RAW와 다르다. **D1의 RAW rank-K basis는 RAW FIT forecast−last에서 별도로 고정**해야 하며 Beijing의 학습 teacher basis를 RAW basis로 바꿔 부르면 안 된다. 기존 teacher last는 FIT만 있고 공통 mean/std·split origins가 없는 schema라 D1용 last NPZ에 그대로 넘길 수 없다.

## 저장 FIT latent와 같은 E native

[확인: JSON 비교] BASE_Q의 selected_codec_hash는 위 MAIN hash와 두 자료·두 seed 모두 정확히 같다. whole CP file SHA는 q와 method metadata 때문에 다르다. state equality의 선언 증거와 실제 CP state 비교는 구분한다. 아래 FIT delta NPZ를 MAIN에 귀속하려면 후속 gate에서 양 CP를 실제 복원·비교한다.

```text
A/robin/S2_ANCHOR_NONLINEAR_D_BASE_Q_w0_s92601/SELECTED_FIT_ALIGNMENT.npz
a14af75528646dc3d245d0a78d83ee08b5b57bc56f79db904c082be54bbf596a
A/robin/S2_ANCHOR_NONLINEAR_D_BASE_Q_w0_s92602/SELECTED_FIT_ALIGNMENT.npz
1b188df002c6aa0f335dd2c6fea09e7c89f1323ccab9d2991c12a29fb29f10f9
A/beijing_aotizhongxin/S3_ANCHOR_NONLINEAR_D_BASE_Q_w0_s92601/SELECTED_FIT_ALIGNMENT.npz
fd05ca806b05a04a28ad6dd722ed535ceb87b82a6c8d740c22c942ad4a1f7d3c
A/beijing_aotizhongxin/S3_ANCHOR_NONLINEAR_D_BASE_Q_w0_s92602/SELECTED_FIT_ALIGNMENT.npz
3e27f55a0ce62105e59a41e5073392616e46139dbe5c3e53615c3c793a7e68cf
```

`checkpoint_audit.py` schema는 `latent_fit_pre_q/latent_fit_post_q/target_latent [256,48,K]`, `valid [256,48]`, summary scale·target, `origins [256]`, q weight/bias다. **pre_q는 z_F−z_last이며 절대 z_F가 아니다.** target_latent는 실제 미래의 좌표, post_q는 학습 보조 readout 좌표다. 둘을 F 출력이나 normalized original-C 오차로 혼합하지 않는다. MAIN의 TUNE/VAL/EVAL latent 저장 자산은 이 소스 흐름에서 확인되지 않았다(unknown). nonlinear D 재적합은 delta만으로 불가능하므로 z_F와 z_last 둘 다 새로 저장해야 한다.

같은 MAIN E/D를 동결한 native 대조는 BASE_Q latent audit와 별도로 실제 존재한다. `controls.fit_native`는 `NativeShared(512)`의 공유 `Linear(512,48)`만 600 updates 학습하고 TUNE normalized MSE로 선택한다. source_checkpoint path·SHA는 위 MAIN CP와 동일하다. 새 native 적합을 하기 전에 이 기존 대조부터 비교할 수 있다.

```text
RN/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/NATIVE.npz
9c4a831ddf1b1caafd0dfacb388845e7724dcbc8b3f6bda33326f3e0cdeafa5e
RN/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/models.pt
4eccd228c8f5f4bb831d72c47da2525a6a415da8668c1838e5e83e5c3b1079e9

RN/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/NATIVE.npz
e1aeded615e35fec1f380e0af934575d28a3e763892b6b090ed2ae5060b6c7bf
RN/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/models.pt
ab3080dd34e40f789a40a52e7bb95f714cf6ff2e0c72118d8f687beca6575543

B/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/NATIVE.npz
0101e8e31d20e3885c91007d06296be0032a45ef492fd401a862e9b5756e8465
B/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/models.pt
5a7b2e986468b26ef193525e12a3a953ba6a2257e79a9ce5eb5f69d2ccfb8dbf

B/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/NATIVE.npz
06e8c17bb32205b75c3db1dac3d6b94f63ea42c2fa6b5eb0de4fe4643d8f237a
B/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/models.pt
5029d7e3070bbe5da1d0937f1c3552eb48653213ee785d9947222decdf4bf872
```

native selected update는 각 자료 seed1=400/seed2=600이다. `controls.models_for_case(control, source_cases)`는 원 MAIN을 복원하고 `models.pt['codec']`의 실제 torch.equal, native/ridge state를 확인한다. 이 native는 Bolt용으로 학습된 E/D의 readout 대조이며 native에 최적화한 encoder의 최선 성능이라고 하지 않는다.

## 실행 전환 뒤 순서

1. Bi의 필요한 학습·검산·비용·보고를 완료하고 root가 실제 후속 실행으로 전환한다. 그 전에는 아래 작업을 실행하지 않는다. 신규 후속 resource ledger와 fresh output을 쓰고 원 `engine.run`, `new_run.run`, `checkpoint_audit.run`, 기존 중단 job을 재실행하지 않는다.
2. 선언 SHA와 실제 파일 SHA·source snapshot을 대조하고 source 조건을 봉인한다. SOURCE JSON hash, 각 NPZ/CP hash, 현재 설치 Chronos source·model snapshot 개별 hash는 이때 확인한다. 원 source/CP 불일치가 있으면 선택 의미를 바꿔 자동 진행하지 않는다.
3. D0: 자료별 manifest `storage_scale=normalized`, `normalization_origin_split=fit`, 위 RAW와 두 MAIN candidates, 정확 channels·split roles를 지정한다. k/last를 생략하여 먼저 [diagnose_saved_outputs.py](diagnose_saved_outputs.py)를 실행한다. 실제 배열 열기 전에 SHA와 역할 gate를 검사하고 `--execute-saved-diagnostics --source-manifest <fresh_manifest> --output <nonexistent_directory>`를 사용한다. `.venv/Scripts/python.exe`, OMP/MKL/OPENBLAS threads2를 유지한다.
4. D1 저장 RAW projection: source past에서 split별 last를 추출하여 새 NPZ에 `mean/std`, `fit_last/tune_last/val_last` 또는 `eval_last [N,C]`, 각각 `_origins [N]`를 저장한다. 원 RAW와 저장 scale·stats·origin이 정확히 같아야 한다. FIT teacher last의 공통 FIT equality를 검산하고 다른 split에 복사하지 않는다. 그 새 SHA와 k4/2를 manifest에 추가한다. 기존 helper는 FIT-only through-origin second moment와 mean+centered covariance를 별도로 적합한다.
5. D1 변환 순서: `project(RAW forecast)` 대 `F(projected past)` 경로는 현재 미구현이다. 동일 basis와 normalized past·같은 Bolt를 사용하되 input projection의 mean/last centering 및 output inverse/anchor를 먼저 명시한다. 현재 mean-centered RAW delta projection을 기존 anchor decoder로 조용히 대체하면 mean 항의 의미가 달라질 수 있다. root의 새 조건 봉인 뒤 필요한 실제 추론만 보충한다.
6. D2: 선택 MAIN/BASE_Q state equality와 output parity를 먼저 확인한다. FIT 기존 pre_q delta는 유지하고, 같은 MAIN에서 z_past/z_last/z_F를 FIT/TUNE/개발 평가에 저장하여 nonlinear D의 입력을 완성한다. 기존 MAIN native NPZ를 같은 split에서 비교한다. D 재적합은 E/F latent를 고정하고 FIT만 학습·TUNE만 선택하며 `x_last + D_new(z_F) − D_new(z_last)`라는 원 anchor 정보/비용을 유지한다. 새 decoder 후보·loss·grid·update 예산은 아직 봉인되지 않았으므로 unknown이다.

## RAW forecast distillation은 기존 alignment와 다르다

[확인: 해당 캠페인 소스] `teacher_basis`/`fit_teacher`의 과거 기반 forecast는 FIT covariance basis를 정하는 데 사용했다. `Alignment`는 실제 FIT 미래 `(Y−x_last)`를 observed-weighted LS 좌표로 바꾸고, 실제 미래 summary/prototype을 q 보조 loss target으로 사용한다. `engine.train`의 main loss도 실제 target이다. **RAW_BOLT의 과거 기반 forecast 자체를 작은 경로의 학습 target으로 맞추는 distillation은 이 캠페인의 실행 source에 없다.** 전체 저장소의 모든 과거 실험에서 never tested라고 확장하지 않는다. 신규 distillation을 진행한다면 실제 미래 정렬 결과를 그 방법의 기존 검증처럼 인용하지 않는다.

원 source 선언 SHA는 engine `05f2cf63b3f7f6a01bf722c51771455be9c73c21ab301d7eb950b3dd58bab817`, controls `324c75a65a001ac0cae30610abb148499718cbad838a69e0012df4a85bd83393`, alignment `84d5be3e310251987fb4f7403399b9a7b9d37f3a54410e85c68613bfeeac4537`, newdata `ccef24b64c9beef0e6e31f9bc49dbad4e8e6ad57a6ce7c99973ccbce691de3bd`, newdata_repair `8cc75ffc4c86329939ae5155192afa03494f7bfff70fcf1ab2946c8c13f83900`이다. 직접 소스를 현재 실행하지 않은 준비 지도이며 원인 확정·진단 완료 보고서가 아니다.
