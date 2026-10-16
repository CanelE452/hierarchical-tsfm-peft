# 데이터 자산·시간·사용권 검토

이 문서는 원자료의 공개 허가서가 아니다. 실제 취득·변환 fingerprint는 [ASSET_SOURCES.json](ASSET_SOURCES.json)에 기록한다. 원본, 원정답 배열, 큰 예측 배열은 비공개 캐시에 두며 main에 게시하지 않는다. 로컬 원자산 위치는 환경변수 `INFORMS_ASSET_ROOT`, 새 캐시는 `INFORMS_CACHE_ROOT`로 전달한다.

## 전체 계약

사용 가능한 시간 prefix 안에서 A_fit/A_stop/M_cal/V_select/T_final을 50/10/10/10/20으로 고정한다. A_stop으로 epoch를 정한 뒤 A_fit+A_stop에 통계·PCA·모델을 다시 적합한다. M_cal 이전까지 학습한 부모만 결합기에 사용한다. 두 번째 cutoff는 첫 분할의 앞 80% 안에서 같은 비율로 만들어 첫 T_final을 다른 fold의 학습에 흡수하지 않는다. 이전 연구에서 본 Beijing/Jena/Robin 기간의 새 T는 역사적으로 새로운 holdout이 아니다.

origin o의 입력은 [o-L,o), 정답은 [o,o+H)이며 정답 전체가 해당 split 안에 있어야 한다. 학습 stride1, 평가 strideH, 마지막 끝점을 덧붙이지 않는다. L512/H48, PeMS08만 H12다. 상수 채널은 표준편차 1로 남기고 양의 저분산 채널을 제거하지 않는다. 원관측 mask만 통계·목표에 사용한다. 입력 결측은 각 창에서 A평균으로 상태를 초기화한 뒤 과거 방향으로 채운다. PCA는 해당 A의 과거 방향 채움·정규화 뒤 실제 feature 평균을 빼며 `pca_center`를 별도 반환한다.

## 기존 개발 자료

- Beijing: pin된 안전 `fit_tune_payload.npz`의 FIT [0,21038), 11채널, hourly. 실제 물리 FP32와 원 mask를 사용한다.
- Jena: pin된 `jena_trainval.npz`의 FIT [0,20965), 21채널, 10분. legacy TRAIN 정규화 FP32를 원 mean/std로 역변환한다. 전체 TRAIN 통계의 계보와 반올림이 남으므로 무손실 raw A-only 자료라고 부르지 않는다.
- Robin: pin된 `trainval.npz`의 원 인덱스 [528,2121), 1593행·17채널·hourly. Jena와 같은 legacy 정규화 한계를 갖는다. 짧은 stress 진단이며 주 도메인 평균과 구분한다.

세 safe NPZ는 필요한 member를 decompress한 뒤 즉시 FIT를 자른다. 따라서 기존 TUNE/VAL 일부가 물리적으로 역직렬화될 수 있지만 반환 배열·역변환 계산·새 통계·학습·평가는 FIT에 한정된다. 기존 보호 TEST 파일은 열거나 hash하지 않는다. 원 source 및 계약의 정확한 size/SHA를 확인하고 컬럼 순서·시간 간격·FIT 경계를 대조한다.

## Electricity

[Zenodo 3889830](https://zenodo.org/records/3889830)의 `electricity_hourly_dataset.ts`를 사용한다. 공식 metadata는 35,932,338 bytes, MD5 `142c94e21026db3c15c2d5c5ca7e2af9`, CC-BY-4.0이다. 원 370고객 UCI 15분 자료에서 임의 321개를 골라 만든 파일이 아니다. 배포된 Lai et al. 계열 321 hourly panel의 순서를 그대로 유지한다.

파일 내부는 TSF형식 `series_name:start_timestamp:comma_values`다. 실제 시작 표기는 `2012-01-01 00-00-01`이므로 1초를 보정해 지우지 않는다. source의 시간대/DST 의미는 명시되지 않았다. 기존 프로젝트 보호 경계인 `2014-10-07T00:00:00`를 같은 clock의 엄격한 exclusive cutoff로 적용한다. 원본 배포 blob 취득·checksum 계산과 텍스트 길이 확인은 전체 파일에 적용하지만 이 경계 이후 값은 float로 변환하지 않는다. parser는 보호 tail을 opaque text로 남긴다. 원 321개 모두 시간범위와 길이가 같아야 한다.

source는 hourly aggregation, `@missing false`, `@equallength true`를 명시한다. 유한 0은 관측이다. 단위는 원 UCI kW 계보와 hourly aggregate라는 설명까지만 확인했으며 원 전처리의 정확한 집계/단위변환·보간을 재구축하지 않았다. 원 배포 preprocessing을 미래 누출이 전혀 없는 것으로 보증하지 않는다.

## PeMS08

[저자 ASTGNN commit 9c2e19b98c4cedf1f35214d8789685b6381b3aad](https://github.com/guoshnBJTU/ASTGNN/tree/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08)의 `PEMS08.npz`를 pin한다. 실제 18,525,419 bytes, Git blob SHA1 `f5f3e84a00417573aa8487fe1e9e8cd13c6026c8`, SHA256 `e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075`가 일치한다.

실제 key는 `data`, shape는 [17856,170,3], dtype float64다. [ASTGCN-r-pytorch 2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296](https://github.com/guoshnBJTU/ASTGCN-r-pytorch/tree/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296)의 README는 이 ASTGNN 자료 다운로드와 연구 재현을 안내하고, `prepareData.py`는 feature0을 미래 flow target으로 사용한다. config는 num_of_vertices170, points_per_hour12를 명시한다. 따라서 전체 170센서의 feature0을 사용하며 순서를 바꾸거나 0을 결측으로 대체하지 않는다.

NPZ 안에 실제 calendar timestamp나 외부 sensor ID가 없으므로 시간은 1970-01-01부터 300초 간격의 synthetic elapsed index다. 1970년 관측이라는 의미가 아니다. 컬럼 ID도 보존된 sensor axis index다. 원 upstream 결측처리·미래값 보간 여부는 확인 불가다. 명시적인 정식 데이터 라이선스는 발견하지 못했으며 `AUTHOR_RESEARCH_DISTRIBUTION_FORMAL_LICENSE_UNSPECIFIED`로 표시한다. 이 근거를 raw 재배포 허가로 확대하지 않으며 source와 raw 배열을 공개 main에 넣지 않는다. L512/H12는 TimeMixer L96/H12와 달라 원논문 점수 재현이 아니다.

## M5 세 패널

로컬 승인 자산에서 M5 판매/달력 원파일을 찾지 못했다. 이번 실행의 공식 Kaggle competition data-list API 무인증 요청은 실제 HTTP401이었다. 사용자 계정·약관 동의를 대신 사용하거나 출처 불명 mirror로 대체하지 않았다. `m5_panel01`, `m5_panel02`, `m5_panel03`은 모두 `BLOCKED_ACCESS_OR_LICENSE`이며 상품/부서 schema나 채널 수를 추정해 만들지 않는다. 적법한 공식 자산이 제공되면 schema를 확인하고 metadata ID hash 순서로 3개 점포×부서, 각 최대128상품을 결과 전에 고정해야 한다.

## 검사 범위

합성 6개 테스트에서 split/target 경계·평가 끝점 미추가·secondary T 비침범·A 밖 값 변경에 대한 통계/PCA 불변·창별 fill reset·관측0과 결측 구분·상수/저분산·금지 경계 밖 history/target 거부·Electricity 보호 tail poison 미변환을 실제 확인했다. 이 검사는 모델 성능 개선이나 독립적인 일반화의 증거가 아니다. 별도의 checkpoint replay·저장 예측 재채점은 평가 담당 산출물의 범위다.
