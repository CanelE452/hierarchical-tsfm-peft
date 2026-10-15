# 이번 캠페인의 안전 자산과 새 시간 경계

작성: 2026-10-15. 이 문서는 데이터 담당의 SOURCE·JSON·파일 메타데이터 조사와 로더 구현 결과다. **이 담당자는 실제 NPZ 배열·원 CSV·모델을 열거나 학습하지 않았다.** 배열 로드·원수치·실제 학습 완료는 루트 실행자의 별도 receipt로 확인해야 한다. `describe_assets()`는 존재·크기만 확인하며 데이터 SHA를 새로 계산하지 않는다. `load_dataset()` 호출 시에만 허용 파일을 해시한 뒤 필요한 NPZ member를 읽는다.

## 자산과 재사용 범위

**[확인·계약/파일 메타데이터] Beijing:** [기존 contract](references/beijing_contract.json)에 연결된 `fit_tune_payload.npz`가 존재하며 1,524,625 bytes다. 허용 FIT는 원 절대행 `[0,21038)`, 채널 11개, 간격 1시간이다. 새 로더는 `physical/mask/times/columns`만 읽고 FIT로 바로 자른다. 기존 `x/filled/mean/std/PCA/E/D`를 재사용하지 않는다. 원래 TUNE `[21038,28051)`와 EVAL `[28051,35064)`는 새 학습 구간에 포함하지 않는다.

**[확인·계약/파일 메타데이터] Jena:** [기존 contract](../level_matched_learned_ed_v18_20261006/data_contract_jena.json)의 safe `jena_trainval.npz`가 존재하며 1,561,064 bytes다. 채널 21개, 간격 10분, 기존 TRAIN은 `[0,26207)`다. [기존 로더](../past_only_repr_diagnosis_20261012/common.py)는 TRAIN의 앞 80%인 FIT `[0,20965)`와 TUNE `[20965,26207)`를 사용했다. 이번 허용 범위는 FIT만이다. Jena가 존재하므로 점수에 따른 Appliances 대체를 하지 않는다.

Jena [생성 SOURCE](../tsfm_peft_practical_controls_v7_20260928/data_prepare_v7.py)의 `standardize_train_only`와 `prepare_jena`는 전체 TRAIN 통계로 FP32 `x`를 만들고 원 관측 `finite` mask를 보관한다. 현재 safe archive에 별도 원 단위 배열은 없다. 이번에는 FIT `x`만 잘라 `x*legacy_std+legacy_mean`으로 FP32 역변환한다. **전체 TRAIN 통계와 반올림을 이미 거친 계보가 남으므로 원시 A-only 값의 무손실 회수라고 부르지 않는다.** 새 A 통계를 다시 적합하는 것과 이 계보 문제는 별개다.

**[확인·계약/파일 메타데이터] Robin:** [기존 contract](../level_matched_learned_ed_v18_20261006/data_contract_robin.json)의 safe `trainval.npz`가 존재하며 111,474 bytes다. 채널 17개, 간격 1시간, TRAIN은 `[528,2520)`이며 이번 허용 FIT는 `[528,2121)`다. 기존 TUNE `[2121,2520)`는 전환하지 않는다. [생성 SOURCE](../tsfm_peft_level_confirmation_v6_20260928/prepare_v6.py)의 `make_arrays`는 전체 TRAIN 통계를 거친 FP32 `x`를 저장했다. Jena와 같이 조건부 FP32 물리값 복구이며 무손실 raw A-only가 아니다. 짧은 기간의 스트레스 진단용이다.

```text
dataset  archive metadata SHA256 (실배열 로드 때 다시 검증)
Beijing  78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277
Jena     29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933
Robin    04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14

contract SHA256 (이번 담당자가 JSON bytes로 직접 확인)
Beijing  9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f
Jena     678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820
Robin    b7c02be4f69617e1c60dd874ef6337facde1a4165f8cbd30233aefb0601a85b2
```

## 데이터 접근의 실제 의미

[data.py](data.py)는 데이터 파일의 정확한 경로 3개와 SHA·크기·계약 SHA를 고정한다. 기존 repair/full-loader를 호출하지 않아 EVAL-past 생성·원 ZIP 재파싱·자동 다운로드로 진입하지 않는다. Python audit hook은 허용 archive와 이번 새 campaign 경로 이외의 `.npz/.npy/.csv/.zip/.parquet/.arrow/.feather` 접근을 거부한다. 보호 TEST와 Beijing EVAL payload는 허용 목록에 없다. 이 가드는 Python `open` 관측 범위이며 OS sandbox나 모든 native-library 접근의 부재 증명은 아니다.

NPZ의 압축된 member는 전체 압축 해제 뒤 행을 자른다. 따라서 FIT/TUNE 또는 TRAIN/VAL을 함께 담은 안전 파일의 필요한 member를 읽을 때 기존 TUNE/VAL 행도 물리적으로 역직렬화된다. 반환·통계·학습은 FIT만으로 제한하지만 **원래 TUNE/VAL의 값을 물리적으로 전혀 읽지 않았다고 주장하지 않는다.** 기존 보호 TEST 파일·별도 EVAL 정답 파일은 열지 않는다. 전체 원 CSV/ZIP도 열지 않는다.

새 loader의 반환 `physical/observed`는 허용 FIT만이며, `indices`는 원 archive 절대행을 보존한다. Robin 로컬행 0은 원 절대행 528이다. `times`는 원 naive 시간축을 UTC/DST 변환 없이 int64 nanosecond로 통일하고 간격을 검증한다. 원 mask가 참인 값의 유한성·채널 순서도 로드 시 검사한다. 복원하지 않은 target은 NaN으로 유지하고, 학습용 수치 placeholder는 원 mask에서 반드시 제외한다.

## 결과 확인 전 고정한 두 fold

루트와 합의한 주 fold는 전체 허용 FIT의 `A/B/C=60/20/20`, 둘째 fold는 `50/20/30`이다. 둘째 fold는 첫 fold의 C에서 새 모델을 선택하는 절차가 아니다. 전체 scaler/PCA/초기 E/D/학습/보정을 각 fold A에서 다시 적합하고 선택을 B까지만 수행한다.

```text
dataset  fold       absolute A       absolute B       absolute C
Beijing  primary    [0,12622)        [12622,16830)    [16830,21038)
Beijing  secondary  [0,10519)        [10519,14726)    [14726,21038)
Jena     primary    [0,12579)        [12579,16772)    [16772,20965)
Jena     secondary  [0,10482)        [10482,14675)    [14675,20965)
Robin    primary    [528,1483)       [1483,1802)      [1802,2121)
Robin    secondary  [528,1324)       [1324,1643)      [1643,2121)
```

`make_fold`는 로컬행 bounds와 원 절대행 bounds를 모두 반환한다. L512/H48에서 A는 모든 합법 origin을 제공한다. B/C는 각 split의 첫 합법 origin부터 stride48에 마지막 합법 origin을 더한 고정 cohort다. 마지막 두 창은 간격이 48 미만이면 일부 겹칠 수 있으며 독립 표본이라고 세지 않는다. A target의 마지막 행은 B 첫 origin보다 작고, B target의 마지막 행은 C 첫 origin보다 작다. 과거 history가 앞 split의 이미 관측된 값을 포함하는 것은 허용한다. `common_origins(fold,lengths)`는 요청한 최장 lookback이 가능한 같은 target origin의 교집합을 반환한다. Robin에서 L2048은 전체 FIT보다 길어 불가능하므로 인위적으로 늘리지 않는다.

C는 이번 알고리즘에서 잠가 두는 후속 점수 구간이며 과거 캠페인에서 원래 FIT로 사용됐던 시간이다. `historically_unseen=False`를 고정하고, 이번 루트 selection lock 준수 여부와 별도로 기록한다. 두 fold나 C를 새로운 독립 자료/완전 미노출 일반화 검증으로 부르지 않는다.

## 작은 합성 검산

실제 자료를 열지 않은 작은 배열에서 관측 mask 전용 평균/표준편차, window별 A 평균 초기화와 과거 방향 채움, 정답 mask 유지, 두 시간 경계, target purge, 입력/정답 합집합과 중복 횟수를 검사했다. 2026-10-15 `.venv/Scripts/python.exe -X utf8` 실행은 종료코드 0과 `SYNTHETIC_DATA_HELPERS_PASS`를 반환했다. 이는 실제 데이터 품질이나 모델 성능 검증이 아니다.
