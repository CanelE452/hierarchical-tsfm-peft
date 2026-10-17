# 데이터·Ridge 감사

[확인] 상태: `PASS_WITH_M5_BLOCK`. 실제 실행 시각: 2026-10-17T15:41:05.982834+00:00.
기존 승인 자산에서 V 끝까지만 수치 변환했다. 보호 T·Electricity 원자료 tail 수치 접근, 새 다운로드·모델 로드·F 호출은 0회다.
전체 blob SHA는 opaque bytes 검증이며 수치 변환 경계와 구분한다. 전체 자료 hash는 기존 계약에서 상속했으며 T를 다시 읽어 재계산하지 않았다.

모든 성능 해석은 반복 사용한 개발 V에 한정된다. 기존 T의 46조건 평가 이력을 지우지 않으며 새 독립 test라고 부르지 않는다.

## beijing

[확인] `COMPLETE`, 계약 shape [21038, 11], L/H=512/48.
실제 materialized pre-T 행 16830, 관측 0 17580, 결측 4990.
A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.
A60 scaler SHA256: `3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb`.
Ridge: `REUSED_VERIFIED`.

## jena

[확인] `COMPLETE`, 계약 shape [20965, 21], L/H=512/48.
실제 materialized pre-T 행 16772, 관측 0 59088, 결측 0.
A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.
A60 scaler SHA256: `5f0c7d6aeddcfe9577e381375d36cadc47d2572469118da73e6ad837031fa2b9`.
Ridge: `REUSED_VERIFIED`.

## robin

[확인] `COMPLETE`, 계약 shape [1593, 17], L/H=512/48.
실제 materialized pre-T 행 1274, 관측 0 0, 결측 1.
A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.
A60 scaler SHA256: `9ee88501873c78f2240a92fdfe14748034e936a750b6d84e7a6b397db3823888`.
Ridge: `BLOCKED_MISSING_ASSET`.

## pems08

[확인] `COMPLETE`, 계약 shape [17856, 170], L/H=512/12.
실제 materialized pre-T 행 14284, 관측 0 8507, 결측 0.
A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.
A60 scaler SHA256: `94e8c71c42f8d58a1d258fb200d86a24595fe1793203da1543b3e47a60569c60`.
Ridge: `REUSED_VERIFIED`.

## electricity

[확인] `COMPLETE`, 계약 shape [24240, 321], L/H=512/48.
실제 materialized pre-T 행 19392, 관측 0 66243, 결측 0.
A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.
A60 scaler SHA256: `ee837cab095dacd518cb44f57372fd70ad2bc21ff1093e5b8d6c03d2571c83ad`.
Ridge: `REUSED_VERIFIED`.

Beijing은 기존 Aotizhongxin safe FIT [0,21038)의 11채널 hourly 물리 FP32와 원 mask다. Robin은 원 source [528,2121)의 17채널 hourly legacy normalized FP32 계보다. Robin의 짧은 stress 범위를 주 도메인 평균으로 확대하지 않는다. 이번 감사는 두 자료도 각각 V까지만 수치 접근했다.
Jena는 legacy normalized FP32를 기존 mean/std로 역변환한 계보다. bit-exact 원자료도, 역변환 잔차가 성능에 영향을 준다는 결론도 아니다. 컬럼별 단위는 이름을 유지하고 10분 실제 timestamp를 사용한다.
PeMS08은 저자 배포 [17856,170,3]의 feature0 flow·원 센서 축 순서다. 5분 경과 인덱스이며 1970년 실제 관측이 아니다. 외부 센서 ID·정확한 물리 counting 단위·상류 보간의 미래 접근은 UNKNOWN이다.
Electricity는 기존 승인된 321 hourly Lai panel이다. 2012-01-01 00:00:01을 임의 보정하지 않았고 보호 cutoff 2014-10-07 이전 prefix 계약을 유지했다. 이번 검사는 그 안에서도 V까지만 읽었다. 집계·단위변환·상류 보간은 재구축하지 않았다.

## Ridge와 solver

[확인] own-channel 512과거 → 32개 비중첩 block 평균−마지막 값 + intercept, 33 features, horizon별 계수다. alpha=(.01,.1,1,10,100), A_stop stride H, A60 scaler fresh refit 계약이다. 저장 alpha .01은 grid 하한이며 전역 최적값이 아니다.
목적식은 채널별 전체 관측 target 수 N_c로 SSE를 나눈 뒤 alpha L2를 더한다. intercept는 비규제다. sklearn의 비정규화 SSE 관례에 대응하는 alpha는 채널별 alpha×N_c다.
[확인] synthetic masked regression에서 독립 augmented least-squares와 기존 solver, 정상방정식 잔차, 저장·로드 예측을 확인했다. coefficient 최대 차이 `1.7763568394002505e-15`, KKT 최대 잔차 `1.1102230246251565e-16`.
공유 D를 D_h에 복제하면 같은 예측·목적값임을 확인했다. penalty는 shared alpha||D||²와 D_h alpha mean_h||D_h||²로 맞췄다. 이 합성 검사는 아직 실제 B03 decoder 구현의 검증을 대신하지 않는다.
기존 Ridge artifact/source/data/scaler/model hash와 PASS replay 영수증이 일치하는 경우 REUSED_VERIFIED다. 실제 데이터의 KKT는 기존 receipt 수치를 재사용했고 새 대규모 fit/KKT를 주장하지 않는다.

## M5·상류 한계

[확인] M5는 `BLOCKED_ACCESS_OR_LICENSE`다. 기존 공식 재감사는 인증401과 로컬 원자산0을 기록했다. 이번에는 인증 재시도·mirror·새 다운로드·범위 확대 검색을 하지 않았다. 승인된 sales/calendar/prices가 실제 제공되어야 AVAILABLE로 바꿀 수 있다.
[미검증] 정확한 시계열·시간구간의 pretraining overlap은 UNKNOWN이다. [공식 model card 대조](BACKBONE_SOURCES.json)에서 Moirai2는 GiftEvalPretrain·GiftEval train context·Chronos corpus 일부·합성/내부 자료를, Chronos2는 chronos_datasets·GiftEvalPretrain을 기술하지만 현재 Jena/PeMS/Electricity 구간의 명시적 포함·제외까지 입증하지 못했다. NO_OVERLAP_EVIDENCE도 무누출 보증이 아니다.

공식 근거: [Ridge 정의](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Electricity 배포판](https://zenodo.org/records/3889830), [저자 PeMS feature0 전처리](https://github.com/guoshnBJTU/ASTGCN-r-pytorch/blob/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296/prepareData.py).

재현: `python -m research.adapter_diagnostics_20261017_v2.data_audit --execute --asset-root ASSET_ROOT --report-root REPORT_ROOT`. 인수 없는 기본 실행은 source-only no-op이다. 전체 hashes·접근 영수증·검증 수치는 [DATA_AUDIT.json](DATA_AUDIT.json)에 있다.
