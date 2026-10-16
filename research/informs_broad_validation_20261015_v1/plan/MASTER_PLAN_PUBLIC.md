> 공개용 사본. 원문 SHA256: `34d242626f163a8c510d4f6750170bdf9a34af87a610b6aee9df7781fb708569`. 개인 상담 요약 한 문단만 제외했으며 과학·실행 계약은 바꾸지 않았다. 원문은 로컬에 보존한다.

# TSFM INFORMS: 데이터·K·학습·결합 확대 실험과 검증·main 게시 지시문

- 작성일: 2026-10-15
- 대상 저장소: `CanelE452/hierarchical-tsfm-peft`
- 설계 작성 시 확인한 원격 기준: `c7a06e522e1cd8e100a3e2291a7f693dd3f7125f`
- 상태: **실행 설계. 이 문서 작성 자체가 새 실험 실행 또는 push 완료를 뜻하지 않는다.**
- 요청 범위: 앞서 논의한 데이터 확대, K 민감도, 학습 미해결, 예측 결합, 지연 압축, 가변 K 공동학습, 결과 분석, 재현 가능한 공개 자료, `main` 게시.
- 아래 새 방법·수치·중단 규칙은 모두 **[추정·미검증 설계]**다. 실제 로그·원문 확인으로 뒷받침되는 기존 사항만 [확인]으로 표시한다.
- 과학 결과가 음성이어도 게시한다. 실행·산술의 유효성과 성능·신규성·일반화는 각각 판정한다.

## 0. 실행 범위와 완료의 의미

사용자의 요청은 설계 내용을 CLI가 실제 수행하고 결과를 main에서 확인할 수 있게 만드는 것이다. 이 파일을 전달받은 CLI는 계획만 재작성하고 끝내지 않는다. 이미 유효하게 실행된 동일 계약의 결과는 재사용하고, 다른 계약의 결과는 새 비교로 재적합한다.

### 허용하는 작업

1. 현재 저장소와 그 작업에 필요한 로컬 데이터·모델 자산 읽기. 실제 저장소의 지침 파일과 활성 작업부터 확인한다.
2. **새 캠페인 경로** 안의 구현·테스트·학습·평가·수치 분석·문서 생성.
3. 공개 데이터·공개 모델·공식 연구 코드의 적법한 무료 다운로드. 출처·버전·라이선스·해시를 확인하고 제한을 기록한다. 인증·약관 동의가 필요한 것은 기존 적법한 접근만 사용한다.
4. 의존성 추가는 필요할 때 격리된 환경에서 버전을 고정한다. 현재 환경을 업그레이드하거나 다른 작업을 깨뜨리지 않는다.
5. 이번 결과의 코드·집계 원수치·재현 자료·그림·요약을 안전 검사 후 **일반 fast-forward 방식으로 `origin/main`에 push**. 이번 사용자의 명시 요청이 이 범위의 게시 권한이다.

### 허용하지 않는 작업

- 새 결과를 좋게 만들기 위한 보호 TEST 사용, 데이터/채널/시점 사후 제외, 사후 지표 변경.
- 유료 GPU/API 구입, 약관 대리 동의, 비밀키 공개, 허가 없는 사내 데이터 다운로드/전송.
- force push, main 보호 규칙 우회, 원격 이력 재작성, 다른 작업 종료/삭제, 기존 결과 덮어쓰기.
- 원자료·개인 연락처·상담 녹취·비공개 센서 시계열·백본 체크포인트를 공개 main에 올리기.
- 모든 미실행 항목을 “진단상 불필요”로 바꾼 뒤 전체 완료로 선언하기.

### 자원 상한과 현실적인 범위

[추정·운영 상한] 신규 GPU 활성 작업 합계 12시간, 경과시간 48시간, 신규 로컬 저장 30GiB, 자동 무료 다운로드 합계 5GiB를 기본으로 한다. 기존에 승인된 별도 자원 한도가 명확히 있으면 더 제한적인 쪽부터 적용한다. 12시간은 **초기 자원 구간이며 모든 데이터·모든 K·충분 수렴의 완료 예상이 아니다**.

시작 시 30분 이내의 자산 조사·실제 짧은 probe로 전체 비용을 추정한다. 전체 범위가 상한을 넘으면 임의로 “다 했다”고 축소하지 말고, 핵심 가족별 최소 실험을 breadth-first로 우선 수행하고 `RESUME_QUEUE`를 만든다. 한도를 다 쓰기 전에 공개 자료 생성·검산 시간을 남긴다. 한도 초과 실행은 추가 사용자 승인 없이 하지 않는다. 부분 결과도 검산 후 main에 게시하고 미완료 범위를 뚜렷하게 표시한다.

**전체 실행 완료**는 필수 블록의 실제 최소 커버리지를 충족하고, 실행 불가 항목이 있으면 그 차단을 인정한 `PARTIAL` 상태를 쓰는 것이다. **연구 성공**은 별도로 Raw와 강한 비교군 대비 정확도–비용의 추가 가치가 검증된 경우다.

## 1. Phase 0: 목적, 반론, 무엇을 알아내야 하는가

### 최상위 목표

고정된 단변량 시계열 파운데이션 모델(TSFM)에 제공할 시간·채널 정보를 구성하여, 원채널 미래 예측의 정확도와 전체 추론비용 사이에서 언제 실질적인 이점이 생기는지 밝힌다.

교수님과 INFORMS 청중이 판단할 것은 “어떤 데이터·채널 수·표현 예산에서 Raw, 내부 적응, 외부 어댑터, 작은 직접 예측기 중 무엇을 사용할 이유가 있는가”다. 특정 방법을 이기게 만드는 것이 목적이 아니다.

### 연구의 출발점과 아직 가설인 것

[공개 범위] 원문에 포함된 개인 상담의 시간대 및 내용 요약 한 문단은 공개에서 제외했다. 아래 연구 질문·실험 조건·데이터 경계·권한·예산·판정 기준은 원문과 같다.

[확인·기존 캠페인] 다음 실제 파일을 시작점으로 확인한다:

- `research/temporal_frontend_campaign_20261015_v1/README_KO.md`
- `research/temporal_frontend_campaign_20261015_v1/CAMPAIGN_PLAN.md`
- `research/temporal_frontend_campaign_20261015_v1/runtime.py`
- `research/temporal_frontend_campaign_20261015_v1/campaign.py`
- `research/temporal_frontend_campaign_20261015_v1/models.py`
- `research/temporal_frontend_campaign_20261015_v1/data.py`
- `research/temporal_frontend_campaign_20261015_v1/REPORT_KO_v3.md`
- `research/temporal_frontend_campaign_20261015_v1/SCORES.csv`

기존 결과는 신경망 80조건·121,200 update이며, 주 자료 Beijing/Jena와 짧은 Robin 진단이다. 12조건은 학습 미해결이다. Jena의 작은 K MSE 개선을 전부 무시하지 말고, G-only·B0+G·LoRA의 강한 반례도 남긴다. README의 summary만으로 내부 계약을 추정하지 말고 사용 함수와 실제 기록을 읽는다.

### 목적별 질문과 최소 대조

| 질문 | 필수 대조 | 결과가 바꾸는 결정 |
|---|---|---|
| 자료가 너무 좁았는가? | 작은 이질적 센서 vs 전력·교통·소매 패널 | 적용 범위 유지/제한 |
| K가 너무 작았는가? | 같은 자료·같은 모델에서 여러 K와 K=C | 유효 압축 범위 및 실용 예산 |
| 앞단 자체가 필요한가? | 시간 앞단 vs 같은 K의 B0 vs 고정 PCA | 단순 표준 해법으로 충분한가 |
| G와 Bolt가 보완적인가? | G-only, 단순 예측 결합, 예측값 조건부 보완 | 결합 정보/추가 용량 중 무엇인가 |
| 시간 의미의 일관성이 중요한가? | 기존 창 전체 혼합 vs 고정 규칙의 지연 압축 | 제약 있는 단순 표현의 필요성 |
| 하나의 모델로 K를 조절할 가치가 있는가? | 순서화된 다중 K 학습 vs 각 K 별도 모델·사후 절단 | 재학습·저장·예산 대응의 이점 |
| 학습이 부족했는가? | 실제 epoch/노출/학습률/초기점·중단곡선 | 방법 실패 vs 자원 내 미해결 |

### 가장 강한 반론

- 알려진 AdaPTS·예측 결합·공통/개별 결합을 이름만 바꾼 것 아닌가? → 실제 표준 포트·단순 결합·B0와 비교하며 신규성은 보류한다.
- 더 작은 Raw 백본·병합 LoRA·G-only가 더 정확하고 저렴하지 않은가? → 같은 정보·전체 비용으로 남긴다. 불리하다는 이유로 제외하지 않는다.
- 개선은 K·자료·seed·시점·지표를 사후 선택한 결과 아닌가? → 등록된 전체 시도·V 선택·봉인된 T 평가를 분리한다.
- 표현이 이미 예측을 직접 수행하고 Bolt가 장식이 된 것 아닌가? → 같은 표현의 단순 예측기, forecast ablation, 비용 포함 비교를 한다.
- 특정 모델에 유리한 데이터만 고른 것 아닌가? → 새 도메인은 성능과 무관한 메타데이터 기준으로 등록하고 모든 결과를 게시한다.

“상류 먼저”: 데이터·시간 경계·정보 계약이 잘못되었으면 해당 경로를 고친다. 단순 SELF/JOINT 선형 비교의 음성 결과를 모든 표현 연구의 중단 조건으로 쓰지 않는다. **정보 중복을 압축할 수 있다는 것과 다른 채널이 개별 예측 정확도를 추가 개선한다는 것은 다른 질문**이다.

## 2. 선행연구를 실행 설계에 연결하는 규칙

아래는 검색·원문 확인 범위를 구분한 근거다. CLI는 실제 구현을 이식하기 전에 해당 원문/공식 코드의 사용 부분을 읽고 commit 또는 문서 버전을 기록한다.

| 원리/문헌 | 원래 지지하는 내용 | 이번 차이·위험 | 실행상 역할 |
|---|---|---|---|
| TSMixer: An All-MLP Architecture for Time Series Forecasting (2023/TMLR) | 시간·피처 혼합, 데이터에 따른 피처 정보의 상이한 이득 | Bolt 앞단으로 바꾸는 것은 원모델 재현이 아님 | B2/직접 모델, 자료 확대 |
| TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (2024/ICLR) | 성분별 다중 해상도 과거 혼합과 미래 예측 결합 | 시간 축소≠채널 축소, 앞단만 이식함 | B4 비교, P-only 효과 분리 |
| AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting (2025/ICML) | 고정 단변량 FM의 잠재공간 어댑터 | Bolt·결정론·7시차 D는 별도 변형 | 표준 어댑터 포트 필수 검토 |
| The Combination of Forecasts (1969/Operational Research Quarterly; 현행 출판사 페이지 JORS) | 과거 오차로 정한 예측 결합의 가능성 | 단순 결합이 새 기여는 아님 | 가장 저렴한 조합 기준 |
| Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting (2019/NeurIPS), §5.2 | 공통 예측을 개별 예측기의 추가 입력으로 사용 | 원모델은 예측기도 함께 학습, 우리는 고정 Bolt | 예측값 조건부 보완의 근거 |
| Learning Ordered Representations with Nested Dropout (2014/ICML) | 표현 좌표의 순서화·유연한 차원 | 시계열 미래에 대한 보장은 없음 | 다중 K 원리 |
| Matryoshka Representation Learning (2022/NeurIPS), §3 | 여러 prefix 차원의 과제 손실을 함께 학습 | 현재는 잠재 시계열·고정 FM·D까지 포함 | 가변 K 파일럿 |
| Lightweight Online Adaption for Time Series Foundation Model Forecasts (2025/ICML), ELF | 예측기와 결합기를 이용한 FM의 온라인 적응 | 현재는 정적 학습 계약; 온라인 정답 피드백이 다름 | 가까운 선행연구로 반드시 명시, 조용히 온라인 학습을 추가하지 않음 |

[확인 범위] TSMixer/TimeMixer는 첨부 원문을 사용한다. DeepGLO와 MRL은 원문을 확인했고, 나머지는 공식 출판 페이지·초록/문서를 확인한 범위를 넘겨 주장하지 않는다. 성능 수치·논문의 전체 재현 여부를 만들지 않는다. SPRINT 기존 포트가 발견되면 수정된 horizon/grid에 맞춰 검토하되, 이 캠페인의 새로운 기여라고 이름 붙이지 않는다.

새 원리 3개는 **예측 결합(통계적 보완성)**, **지연 압축(시간 의미의 일관성)**, **다중 K(용량·자원 조절)**다. 원래 분야의 성공과 이번 성능은 별개다.

## 3. 자산 조사와 새 캠페인 분리

### 3.1 먼저 읽고 확인할 것

- 실제 `origin` URL, branch, `HEAD`, `origin/main`, working tree 변경, 진행 중 PID/명령/시작 시각.
- 해당 저장소에서 발견한 실제 지침 파일을 읽는다. 이름만 보고 파일 존재를 가정하지 않는다.
- 기존 데이터·모델 파일은 메타데이터를 먼저 검사한다. 파일명·경로·컬럼·배열 key는 직접 열어 확인한 것만 코드에 쓴다.
- public benchmark 재다운로드도 과거 노출 이력을 지우지 못한다.
- 실행 초기에 `REUSE_MATRIX`를 만들어 source/data/scaler/split/origin/mask/objective/weight/checkpoint가 모두 맞는 결과만 재사용한다.

### 3.2 새로 제안하는 경로

`research/informs_broad_validation_20261015_v1/`는 **이번 문서가 제안하는 새 경로**다. 이미 존재하면 내용과 캠페인 ID를 확인하여 동일 작업을 재개하거나 새 run 하위폴더를 만든다. 기존 다른 결과를 지우지 않는다.

데이터·큰 체크포인트·예측 배열은 별도 비공개 캐시에 둔다. 구체적 캐시 위치는 저장소 상태·용량을 확인하고 정한다. 공개 경로와 private path는 manifest에서 분리한다.

### 3.3 기본 구현 원칙

새 실행기는 기존 모듈 중 검증 가능한 작은 구성만 재사용한다. 상수 `L=512,H=48`가 박혀 있는 기존 함수에 다른 horizon을 억지로 넣지 않는다. `L,H,num_channels,latent_k,origin_index,mask,scale`를 명시적 인자로 전달하고 모든 출력 형상을 검사한다.

초기 구현은 단일 GPU 순차 큐다. CPU 다운로드·문서 생성은 병행할 수 있지만 정밀 GPU 타이밍 중 학습하지 않는다. 중단 후 optimizer·scheduler·random state·data permutation·예산 누적까지 일치하도록 재개한다.

## 4. 데이터 확대: 다운로드보다 계약이 먼저다

### 4.1 필수 자료군

| 자료군 | 실행할 실제 범위 | 기본 과거 L / 미래 H | 역할 |
|---|---|---|---|
| Beijing | 기존 검증된 11채널 안전 FIT | 512 / 48 | 기존 음성 결과 연결 |
| Jena | 기존 검증된 21채널 안전 FIT | 512 / 48 | 기존 양성 결과 연결 |
| Electricity | **321개 hourly 시계열 전체**인 표준 전처리 자산 우선 | 512 / 48 | 큰 유사 채널, 비용·K 곡선 |
| PeMS08 | **170 센서**의 확인된 traffic-flow 대상 전체 | 512 / 12 | 시간 지연·다수 센서 |
| M5 제한 패널 | 메타데이터로 정한 3개 점포×부서 패널, 각 최대128 상품 | 364 / 28 | 일별 수요, 개별 차이·간헐성 |
| Robin | 기존 17채널 안전 FIT | 512 / 48 | 짧은 기간 스트레스; 주 평균에서 별도 |

여기 L/H는 새 설계다. PeMS의 L512/H12는 TimeMixer 논문의 L96/H12와 다르고, M5 L364/H28·MSE 패널은 TSMixer의 L35/H28·분포 손실/전체 M5와 다르다. 원논문 점수와 직접 비교하지 않는다. 두 도메인 모두 표준 과제의 초점과 다르다는 점을 `DATA_CONTRACT`와 보고서에 적는다.

Electricity UCI 원본은 370고객·15분 간격이며, 321 hourly 전처리와 동일 파일이 아니다. 전처리 형태·시간 범위·단위·고객 선택·시간대/일광절약시간 처리·마스크를 확인한다. 임의 321컬럼 선택으로 표준 자료라고 부르지 않는다. Zenodo Electricity Hourly의 공식 설명을 시작점으로 삼되 실제 버전/파일/라이선스를 잠근다.

PeMS08은 공식 논문/저자 코드가 가리키는 자료를 우선 사용한다. 배열의 어느 축/필드가 flow인지 직접 확인한다. 170센서가 확인되지 않거나 이미 미래값으로 보간한 파생 파일밖에 없으면 구체적 제한을 기록한다. 0을 자동 결측으로 보지 않는다. 타임스탬프가 없다면 공식 간격 메타데이터에 근거한 인덱스임을 명시한다.

M5는 실제 사용권과 접근을 확인한다. 파일 내용 확인 전 `d_*` 등 내부 key가 존재한다고 코드에 가정하지 않는다. 점포·부서·상품의 실제 메타데이터 필드를 찾고 매핑을 기록한다. 미리 정한 안정적인 해시 순서로 3개 패널을 선택하고, 각 패널에서 최대128개 상품도 같은 메타데이터 기반 해시로 고른다. 관측값·TEST 성능·미래 매출로 선택하지 않는다. 전체 시계열을 동일 달력에 정렬한다. 표본 패널은 M5 전체 벤치마크가 아니며 3패널을 3개 독립 도메인으로 세지 않는다. 상품의 판매 0은 관측일 수 있다. 상품 출시 전 구간을 결측으로 처리할지는 출처의 의미와 사전 규칙으로 정한다.

### 4.2 자료 선택 실패 처리

1. 로컬 자산을 확인한다.
2. 없으면 공식·허가된 공개 소스에서 무료 취득을 시도한다.
3. 로그인·라이선스 동의가 필요하면 그 경로만 `BLOCKED_ACCESS_OR_LICENSE`로 둔다. 출처 불명 mirror로 대체하지 않는다.
4. 해당 자료 없이도 가능한 다른 자료·방법은 진행한다.
5. 자료군이 빠졌는데 “5도메인 완료”라고 하지 않는다. 새 데이터 없이 기존 두 자료만 반복한 경우 `BREADTH_NOT_COMPLETED`를 보고한다.

### 4.3 값·마스크·시간 계약

모든 자산에 다음을 기록한다: 출처 URL, 원본 파일 버전과 SHA256, 라이선스/재배포 허용 범위, time interval, 단위, 컬럼 순서, 관측 mask 정의, 전처리·결측 채움, 중복/역순 시점, 원본→학습 배열 계보, 과거 캠페인 노출 여부.

A-only 통계는 원 관측만으로 적합한다. 결측 입력은 창 내부의 과거 방향 채움, 창 시작 전 관측이 없으면 A평균으로 초기화하는 기존 원칙을 기본으로 한다. 미래값을 사용한 보간은 금지한다. mask가 없는 자료의 0/센티널 처리도 출처를 확인하고 데이터 단위로 고정한다.

상수 채널은 관측값이 있으면 출력·원 MSE에 남긴다. 표준편차 0은 스케일1을 사용하고 별도 표시한다. 불안정한 저분산 채널을 결과를 본 뒤 삭제하지 않는다. 무관측 채널의 상태와 active channel 수는 점수마다 기록한다.

## 5. 시간순 데이터 사용: 결합기까지 누출 없이

### 5.1 개발 자료와 최종 평가 자료를 구분

- 기존 Beijing/Jena/Robin의 이미 본 기간은 **개발 자료**다. 새 분할을 만들어도 `historically_unseen=false`를 유지한다.
- 새 전력·교통·M5에서도 과거 프로젝트/백본 학습 이력은 별도 필드다. `new_to_campaign`, `historically_unseen_to_researcher`, `pretraining_overlap_known`을 구분하며 모르면 unknown이다.
- 기존 저장소의 보호 TEST는 이번 문서만으로 봉인을 해제하지 않는다. 새 데이터의 이번 프로토콜 최종 구간은 사전 등록 후 마지막에 한 번 평가할 수 있다.

### 5.2 새 데이터 기본 시간 구간

[추정·설계] 전 시간 범위를 결과를 보기 전에 다음처럼 나눈다. 기존 보호 경계가 더 제한적이면 그 허용 범위 안에서만 적용한다.

| 이름 | 기본 범위 | 역할 |
|---|---|---|
| A_fit | 앞 50% | 기본 모델 학습 |
| A_stop | 다음 10% | 기본 모델의 학습률/체크포인트 선택 |
| M_cal | 다음 10% | 이미 고정된 기본 예측기의 **시간 외 예측**으로 결합기 적합 |
| V_select | 다음 10% | 가족/K/결합/최종 체크포인트 선택 |
| T_final | 마지막 20% | 모든 선택 잠금 후 최종 평가 |

기본 모델은 A_fit으로 학습하고 A_stop에서 epoch/학습률을 선택한 뒤, **고정된 epoch 수로 A_fit+A_stop에 재적합**한다. 이 재적합 모델을 M_cal 이후에 사용한다. scaler/PCA/모델 초기화도 해당 적합 범위로 다시 계산하고 이전 캐시를 무효화한다. A_stop을 재적합에 포함할 때 초기 탐색 점수와 재적합 점수를 섞지 않는다.

M_cal에는 부모 모델이 학습하지 않은 미래 예측만 들어가야 한다. 부모가 M_cal까지 학습한 체크포인트의 in-sample 잔차를 결합기 주 결과로 사용하지 않는다. 결합기 early stopping과 최종 구조 선택은 V_select만 사용한다. T_final은 가중치·K·seed·지표·그림 사례의 선택에 사용하지 않는다.

**동일 정보 대조:** 결합기가 M_cal 정답을 추가로 사용하므로 `G_past_control`, `G_refit_AplusM`, `LoRA_refit_AplusM`, `Raw_calibrated`도 등록한다. 전자는 같은 고정 G 위의 동용량 과거 보정이고, 두 refit은 동일 A+M 학습 허용 범위를 갖는 강한 실제 사용 대안이다. 이때 scorer의 A기준 표준화는 통일한다. 단순 Raw는 무학습 참고선이지 지도정보량이 같은 적응 기준선이 아니다.

### 5.3 창 경계

- origin o에서 입력은 `[o-L,o)`, 정답은 `[o,o+H)`다.
- 각 학습/선택/평가 창의 **정답 전체**가 그 구간 안에 있어야 한다.
- 다음 평가 창이 이미 관측된 이전 평가 구간의 값을 과거 입력으로 사용하는 것은 허용한다. 가중치 온라인 갱신은 허용하지 않는다.
- 학습 origin stride1, 공식 평가 기본 strideH. 마지막 끝점 강제 추가는 하지 않아 불필요한 마지막 쌍 중첩을 피한다. stride나 패널별 origin 제한은 결과 전 고정한다.
- 긴 L 비교는 같은 합법 origin 교집합에서 Raw512도 다시 계산한다.
- 여러 cutoff 재적합 결과는 서로 겹치는 시계열의 조건부 반복이지 독립 데이터가 아니다.

자료가 짧아서 구간당 한 horizon도 없으면 임의로 concat/반복하지 않는다. 해당 중첩 없는 결합·최종 검증을 BLOCKED로 남기고 기존 개발 진단을 별도 표시한다.

### 5.4 두 번째 cutoff

새 데이터의 첫 T_final을 보기 전, 기존 개발 범위에서 시점을 옮긴 두 번째 분할을 한 번 사전 등록한다. 이미 공개된 자료는 개발 cutoff로만 쓰고, 새 데이터의 실제 T_final을 다른 fold의 훈련에 흡수하지 않는다. 구조 선택의 전이 검사를 위해 E/P/D/G와 통계를 모두 다시 적합한다. 종료 후 모든 fold의 노출 구간 union을 출력한다.

## 6. K와 C를 넓히는 실험표

`C`는 실제 채널 수다. `latent_k`와 `lora_rank`는 다른 필드다. 코드에서 `r` 하나로 혼용하지 않는다.

### 6.1 작은 자료의 등록된 K

- Beijing C11: `{2,3,6,9,11}`
- Jena C21: `{5,6,11,16,21}`
- Robin C17: `{4,5,9,13,17}`; 주 확대보다 스트레스 부록 우선.

### 6.2 큰 자료

`K_grid = unique(ceil(C * q))`, `q ∈ {1/8,1/4,1/2,3/4,1}`.

- Electricity C321이면 `{41,81,161,241,321}`.
- PeMS08 C170이면 `{22,43,85,128,170}`.
- M5 각 패널은 실제 C를 확정한 뒤 같은 규칙을 사용한다.

첫 최소 커버리지는 `ceil(C/4),ceil(C/2),C`와 기존 작은 K를 우선 수행한다. 전체 캠페인 커버리지에는 1/8과3/4도 남기며 못 끝내면 미완료다. 인접 K가 비슷해 보인다는 이유만으로 등록된 최소 비교를 지우지 않는다.

K는 V_select로만 선택한다. PCA 설명분산은 A에서 별도 기록할 수 있지만 이것을 예측용 최적 K의 보증으로 쓰지 않는다. K=C는 무압축 적응 대조, K>C는 이번 범위 밖이다.

### 6.3 채널 수 C의 확장 분석

주 결과는 큰 자료 전체 채널이다. 채널 수에 따른 비용·정확도 분석은 메타데이터 ID의 고정 해시 순서로 중첩 부분집합을 만든다. Electricity는 C32/64/128/전체, PeMS는 C32/64/128/전체에서 B0/B2와 Raw/G 기준만 제한 비교한다. 작은 subset을 큰 자료 전체 결과로 대체하지 않는다. 부분집합은 동일 도메인의 반복이며 독립 데이터 수를 늘리지 않는다. subset별 scaler/PCA/학습도 그 subset 계약에 맞게 재적합한다.

## 7. 공통 예측 목표·학습·기준선

### 7.1 목표와 지표

주지표는 A통계로 표준화한 **관측 원채널 미래의 채널 macro MSE**다.

`MSE = mean_c[sum_(o,h) mask*(pred-y)^2 / observed_count_c]`.

모든 모델이 같은 실제 미래와 mask를 사용한다. 잠재 시계열 MSE, 과거 복원 오차, 다운샘플 정답 점수를 주 성능으로 대체하지 않는다. 미니배치 loss의 채널 가중치가 최종 macro 지표와 어떻게 대응하는지 수식·코드를 대조한다. gradient accumulation에서는 microbatch별 loss를 단순 평균하여 실제 effective batch와 다른 가중치가 되지 않게 한다.

MAE는 공동 보고한다. PeMS는 물리 단위 MAE/RMSE, M5는 물리 단위 MAE·WAPE·판매총량 편향·0/양수 구간을 별도 보고한다. 작은 M5 패널의 WAPE를 전체 M5 WRMSSE라고 부르지 않는다. WAPE 분모가0이면 NA와 coverage를 기록한다. 모든 값이 같은 단위가 아닌 센서 자료에서 원단위 MSE를 무조건 한 숫자로 합치지 않는다.

Bolt 기본 점예측은 기존 중앙값(quantile .5)을 유지한다. MSE에 적응한 모델과 사전학습 중앙값의 차이를 명시한다. 특히 M5의 0이 많은 분포에서 중앙값과 평균은 다른 목적일 수 있다. 9개 quantile 평균을 분포의 정확한 평균으로 부르지 않는다. Raw에 동일 정보량의 단순 보정을 붙인 결과도 유지한다. 목적을 바꾸는 새 확률분포 loss는 별도 승인 없는 본 캠페인에 추가하지 않는다.

### 7.2 기본 기준선

| 새 실행 레이블 | 구현 역할 |
|---|---|
| RAW_BOLT_SMALL | 원채널 그대로, 학습 없음 |
| CURRENT / SEASONAL_NAIVE | 현재값 유지 / 출처와 sampling period에 근거한 단순 반복 |
| DIRECT_G | 기존 channel-shared 작은 과거→미래 변화 예측기 |
| RAW_PLUS_G | 기존 합산 경로를 새 시간 외 calibration 계약에 맞춤 |
| G_PAST_CONTROL / G_REFIT_APLUSM | 결합의 추가 용량/추가 학습정보 대조 |
| LORA_R8 / LORA_REFIT_APLUSM | q/v 저랭크 적응, 구현 key를 직접 확인; 추론은 병합 parity 후 측정 |
| PCA_FIXED_BOLT | A PCA 고정 E/D의 표준 선형 압축 |
| ADAPTER_B0 | 시간 P 없음, 예측 목적 E/D 학습 |
| FRONT_B2 | 기존 TSMixer 원리 시간·피처 앞단 |
| FRONT_B4 | 기존 TimeMixer 원리 성분별 다중 해상도 앞단 |
| DIRECT_TSMIXER / DIRECT_TIMEMIXER | 같은 원리의 직접 예측, 원모델 재현/작은 변형 구별 |
| RAW_RIDGE | 단순 원자료 예측; 전체 ridge 차원을 새로 확인 |
| ADAPTS_BOLT_PORT | 공식 AdaPTS의 확인된 결정론 경로를 최소 이식; B0의 이름만 바꾸지 않음 |
| RAW_BOLT_MINI / RAW_BOLT_TINY | 적법한 모델 자산 취득이 가능할 때 실제 크기·컨텍스트 확인 후 비교 |

LoRA는 base copy에서 별도로 학습하고, 종료 후 원래 고정 F의 hash를 확인한다. 공유 F가 병합 가중치로 오염되면 이후 경로를 무효화한다. 병합 전/후 예측 차이는 사전 수치 tolerance로 검증한다.

앞단 B0/B2/B4는 가능한 동일 D 클래스·동일 현재값 anchor·동일 학습 목표를 사용한다. `K` 시계열만 F에 보내도 원 C 현재값과 G의 원 C 과거가 존재한다. 이를 “추론 전체가 K 정보만 사용”한다고 부르지 않는다.

### 7.3 학습량과 최적화

[추정·초기안] seed는 선별1개(92601), 비교 보강2개(92601,92602), 최종 후보3개(92601–92603)다. seed로 최고의 T 값을 고르지 않는다.

에포크는 **합법 A origin 전체를 한 번 노출**하는 것으로 정의한다. effective batch16을 기본으로 하고 필요한 microbatch/accumulation만 바꾼다. 학습 memory에 의해 batch를 줄일 때 비교군 노출·optimizer step 수가 달라지지 않도록 검산한다.

- 최소 커버리지: 0.25→1→3 epoch의 같은 permutation prefix로 모든 핵심 가족을 먼저 실행한다. 이 결과는 선별 결과이며 수렴을 보장하지 않는다.
- 보강: 동일 dataset/K의 핵심 대조를 10→30 epoch까지 확장, A_stop 기준 5 epoch 무개선 조기종료.
- 최종: 선택된 방법·대조는 최대100 epoch 및 동일 patience5. 자원 상한이 먼저 오면 UNDERTRAINED_OR_BUDGET_LIMIT으로 남긴다. 초기 체크포인트도 후보다.
- 학습률 기본은 외부 모델/G `{1e-3,1e-4}`, LoRA `{1e-4,1e-5}`. 각 방법의 A_stop 점수로 선택하고 후보수·GPU비용을 기록한다. 선별은 한 값, 보강에서 제한된 두 값 비교를 한다. 학습률이 다른 모델의 결과를 완전히 같은 튜닝 예산이라고 말하지 않는다.
- AdamW/weight decay0/gradient clipping1은 기존 연결점으로 시작한다. dropout/정규화를 추가하면 별도 설정으로 등록하고 관련 비교군에도 같은 탐색 기회를 준다.

마지막 체크포인트가 best이거나 A_stop이 지속 하강하면 “방법이 안 됨”이 아니라 미해결이다. 유한 gradient·full rank는 수렴 보증이 아니다. 원신호 rank·표현 변화·clipping 비율은 원인 진단이지 효과의 대체지표가 아니다.

100 epoch/patience5는 TSMixer의 학습 프로토콜에서 참고한 상한이지 이번 모든 모델의 최적 설정이 아니다. 기존 2400 update는 한 epoch 미만일 수 있었으므로, 이제 update와 epoch·고유 target 시점·관측수·반복 노출을 함께 기록한다.

## 7A. 블록 E0 — 기존 12개 학습 미해결의 처리 장부

기존 보고서에서 12개 `UNDERTRAINED_OR_UNRESOLVED`의 **정확한 ID**를 읽어 `UNRESOLVED_RESOLUTION.csv`에 모두 등록한다. 다른 실험을 많이 했다는 이유로 이 12개가 해결됐다고 하지 않는다.

각 ID마다 원 데이터/분할/정규화/모델/선택 점수/최종 checkpoint/optimizer/RNG의 사용 가능성을 확인한다. 실제 재개 상태가 있으면 같은 계약에서 이어서 학습한다. 원 sampling schedule이 2,400에서 끝났다면 연장 구간의 permutation을 새로 고정하고 그 차이를 기록한다. optimizer가 없으면 완전한 이어 학습이라고 하지 말고 새 시작 조건으로 등록한다.

[추정·연장안] 4,800 → 9,600 → 19,200 update를 상한 내에서 순차 확인하며 원 B 점수로 체크포인트를 선택한다. 같은 질문에 필요한 비교군도 동일 최종 예산에서 확인한다. 미해결 경로만 더 튜닝하고 대조군의 낡은 점수와 공정한 최종 비교라고 말하지 않는다. 이는 옛 평가 구간을 이미 본 **개발 후속**이며 새 확증 결과가 아니다.

상태는 `RESOLVED_EARLY_STOP`, `STILL_IMPROVING`, `RESTARTED_NOT_EXACT_RESUME`, `BLOCKED_MISSING_STATE`, `DEFERRED_BUDGET` 등으로 구분한다. 한 숫자 상한에 도달했다고 수렴을 보장하지 않는다. 이전 시차 제거 실험을 앞으로 사용하지 않아 추가 학습을 강등한다면, 제거 이유와 그에 따라 더는 주장하지 않을 내용을 적고 “해결”이 아닌 `DEPRIORITIZED_NOT_RESOLVED`로 표시한다.

우선순위는 현재 주장을 바꾸는 B0/B2/G와 그 대조군이다. 그러나 12개 모두의 재개 가능성·실행/미실행·결론은 빠짐없이 장부에 있어야 한다. 전체 비교를 새 데이터로 재적합하는 E1–E4와 이 개발 연장을 중복된 새로운 방법으로 세지 않는다.

## 8. 블록 E1 — 데이터·K·학습의 기본 지도

### 필수 최소 실행

5개 주 자료군(패널은 별도 task)에서 Raw, DIRECT_G, LORA_R8, PCA_FIXED_BOLT, B0, B2를 먼저 실행한다. B0/B2는 K1/4,K1/2,C와 기존 작은 K를 포함한다. B4는 각 자료의 K1/4,C에서 최소한 한 번 실행한다. 전체 K_grid의 B0/B2 비교를 그다음 채운다.

신규 자료에서도 V_select에서 성능이 나쁘다는 이유로 그 도메인의 다른 원리 최소 비교를 막지 않는다. 모델 수를 늘리기 위해 동일 checkpoint를 여러 이름으로 세지 않는다.

### 분석

- 자료별 K–MSE/MAE 곡선, 같은 K B2−B0, C 유지 B2−B0, Raw/G/LoRA 참고선.
- 학습곡선과 초기값 대비 변화. Raw보다 좋아진 전부를 P의 효과로 세지 않는다.
- 매 K의 실제 F streams, E/D/P/G 파라미터, full pipeline 시간·메모리.
- 고차원 subset 효과는 전체 채널 결과와 별도.
- 데이터·패널별 원점수와 Raw/G 대비 비율을 제공한다. 서로 다른 horizon/단위 MSE를 무작정 하나로 평균하지 않는다.

## 9. 블록 E2 — G를 보존하는 결합과 추가 정보의 검증

### 9.1 등록할 부모 모델

각 자료에서 B0/B2의 K1/4,K1/2,C 및 B4 K1/4와 DIRECT_G를 사용한다. 부모는 A_fit/A_stop로만 선택·재적합하고 M_cal/V_select/T_final의 미래를 학습하지 않는다. 저장 예측의 contract·cohort·scaler·model hash를 대조하고 정렬한다.

### 9.2 단순 결합: 반드시 먼저 수행

`g`는 현재값까지 포함한 G-only 미래 예측, `b`는 압축-Bolt의 원채널 미래 예측이다.

`pred = g + alpha*(b-g)`, `0<=alpha<=1`.

M_cal에서 하나의 전역 alpha를 직접 적합한다. 채널 macro MSE 가중치 `w_(o,h,c)=mask/(C_active*N_c)`를 사용한다:

`alpha* = clip(-Σ w(g-y)(b-g) / Σ w(b-g)^2, 0, 1)`.

분모가0이면 alpha0으로 둔다. float64로 계산하고 조밀한 alpha-grid 계산과 대조하는 단위검사를 한다. 여기서의 개선은 적합 결과일 뿐이므로 실제 판정은 V_select와 최종 T_final에서 한다.

단순 대조는 alpha0(G-only), alpha1(Bolt route), alpha0.5(equal ensemble), M_cal에서 정한 alpha다. G+Raw도 같은 방식으로 비교하여 압축 표현 고유의 이득인지 확인한다.

오차 상관만으로 보완성을 확정하지 않는다. M_cal에서 `E[(g-y)(b-g)]`와 `E[(b-g)^2]`, 고정 alpha의 V 성능을 함께 기록한다. 음성 결과라도 E3/E4 최소 파일럿을 막는 gate로 사용하지 않는다.

### 9.3 예측값 조건부 보완: 새로운 실험

현재 `b+G(X)` 대신 `g + H_theta(X, g, b-g)`를 비교한다. G와 b는 고정한다. H는 각 채널의 `과거-current`, `g-current`, `b-g`를 입력받는 channel-shared hidden32 작은 네트워크로 시작하고, 마지막 출력층을0으로 초기화해 초기 예측이 정확히 g가 되게 한다. 최종 실제 미래 오차로 H를 M_cal에서 적합하고 V에서 checkpoint를 선택한다.

동일 정보/용량 대조:

- `G_PAST_CONTROL`: 같은 H 구조와 입력 폭에서 마지막 `b-g` 자리를 **이미 주어진 과거의 최근 H개 current-centered 값**으로 대체. 활성 파라미터를 그대로 두어 가짜 padding을 피한다. 이는 같은 과거 정보의 중복 특징이다.
- `G_REFIT_APLUSM`: G 전체를 동일한 A+M 허용 자료로 다시 학습하는 강한 직접 예측 대안.
- `G+Raw conditioned`: b 대신 Raw 예측을 쓰는 같은 구조.
- `B0 conditioned`: 같은 K의 시간 앞단 없는 부모로 비교.
- 위 전역 alpha 결합과 기존 `b+G(X)`도 새 calibration 계약에서 비교.

출력이0이어서 초기 G를 보존한다는 사실은 **학습 후 비악화 보장**이 아니다. 학습 중 파라미터 변화·입력 gradient·부모 hash를 검사한다. 좋은 T에서만 보정 강도를 조절하지 않는다.

### 9.4 선택적 제한 후속

첫 조건부 H에만 특이 개선이 있다면 하나의 출력 gate 변형을 추가할 수 있다. 과거와 두 예측의 차이만으로 alpha를 정하며 미래 실제 오차를 입력으로 쓰지 않는다. K 선택이나 Bolt 실행 생략까지 포함한 런타임 라우팅은 이번 기본 범위 밖이다. Bolt를 이미 호출한 뒤 gate가0이 되었다고 Bolt 비용이 사라진 것으로 세지 않는다.

**필수 범위:** 모든 접근 가능한 주 자료에서 전역 결합, 최소 B2 K1/4의 조건부 H, G_past/Raw/B0 대조를 실행한다. 양성 신호가 없다는 이유로 조건부 H 최소 비교를 생략하지 않는다.

## 10. 블록 E3 — 시간 의미를 유지하는 지연 압축

### 가설

현재 창 전체 혼합 표현은 관측 창이 달라지면 같은 과거 시점의 표현이 달라질 수 있다. 이는 미래 정답 누출은 아니지만, 고정 예측기가 다음 잠재값을 예측하기에 불리할 가능성이 있다. 현재 데이터에서 원인으로 확정하지 않는다.

### 새 후보와 시간 경계

`z_t = Σ_(ell in LAGS) x_(t-ell) W_ell`, `W_ell: C×K`.

첫 lag set은 `{0,1,2,4,8,16,32}`로 고정한다. 원자료 단위의 시간이 데이터별로 다르다는 것을 설명한다. 전 시점에 같은 가중치가 쓰이는 선형 지연 압축으로 시작한다. `W_0`를 A-only PCA(또는 K=C 항등), 나머지를0으로 초기화한다. 이렇게 모든 가지가 영원히 gradient를 못 받는 zero chain을 만들지 않는다.

입력은 동일 `[o-L,o)`만 허용한다. 왼쪽 부족한 lag 값은 사전 고정한 A-평균/0 정규화 padding을 사용하고 그 범위를 metadata로 남긴다. 몰래 L+32의 과거를 더 주지 않는다. interior prefix test는 padding 영향을 받는 최초32행을 제외하여 수행한다. 이 padding과 시간 간격을 예측 구간까지 가상 정답으로 연장하지 않는다.

D와 현재값 anchor는 기존 대조와 같은 클래스다. 신규 encoder의미·prefix 구조가 다른 점을 명시하고 최종 원미래 MSE로 실제 Bolt를 통과해 학습한다.

### 대조

1. B0의 시점별 선형 압축.
2. 기존 B2의 창 전체 시간·피처 혼합.
3. 지연 압축.
4. 같은7개 가중치·같은 전체 temporal span을 갖는 창 내부 양방향 지연 대조 `{-16,-8,-4,0,4,8,16}`. 입력 창 밖은 고정 padding만 사용한다. 미래 정답은 쓰지 않는다. 이 비교도 위치·경계조건 차이가 있으므로 순수 인과성 증명은 아니다.

파라미터·forward FLOPs·F streams를 기록한다. B0와 지연 압축의 차이를 전부 “causality 효과”라고 하지 않는다. K와 D를 바꾸지 않은 대조에서 학습·노출을 맞춘다.

### 검증과 범위

- 입력의 t 이후 **과거 행**을 바꿔도 interior `z_<=t`가 변하지 않아야 한다(새 causal 후보).
- 예측 origin 이후 미래를 바꿔도 출력은 변하지 않아야 한다(모든 후보; 합성 입력으로 검사).
- 같은 과거 prefix를 서로 다른 끝점의 창으로 표현했을 때 interior 차이를 측정한다. B2가 다르다는 것은 예상 가능한 특성이지 누출 확정이 아니다.
- 최소 Beijing+PeMS08에서 K1/4,K1/2,C 비교. PeMS 접근 불가이면 미완료로 남기되 Electricity에서 보조 대조를 실행한다.
- 이후 나머지 주 자료로 확대. 무개선도 보고한다.

## 11. 블록 E4 — 하나의 체크포인트로 여러 K 사용

### 필수 최소 파일럿

기존 논의에서 후순위였지만 사용자가 전체를 요청했으므로, **Beijing과 Electricity에서 최소 파일럿을 실행할 항목**으로 포함한다. 기존 독립 K 모델이 Raw를 이겨야만 실행하는 양성 gate를 만들지 않는다. 다만 충분 학습·전체 자료 확대는 V 결과와 남은 예산에 따라 좁힐 수 있다.

### 방법

B0와 B2 중 사전 V 규칙으로 정한 한 가족에 대해 최대 Kmax=C의 E/D를 한 번 만들고, `{ceil(C/4),ceil(C/2),C}`의 prefix K를 번갈아 학습한다. 모든 K의 실제 원채널 미래 MSE를 동일 비중으로 최적화한다. 현재값 anchor는 동일하다.

예산 절약 구현은 미니배치마다 K를 순환하거나 균등 추출한다. K별 update 수·감독 노출·F stream-steps를 기록한다. 전체 K를 동시에 평가하는 loss를 쓰면 그 추가 F연산/메모리를 학습 비용에 포함한다.

F는 latent row별 독립 추론이므로 Fmax 후 자르기와 Fk 호출의 parity가 검증된 경우만 해당 재사용을 허용한다. 추론 시간 측정 때는 **실제로 K행만 F에 전달**해야 한다. Kmax를 전부 예측한 뒤 잘라서 K비용이라 하지 않는다.

7시차 D의 prefix 자르기 주의: `D.reshape(7,Kmax,C)[:, :K, :]` 후 flatten한다. `D[:7*K]`처럼 lag 블록과 latent 순서를 섞는 구현을 금지한다. E의 앞 K column·z의 앞 K stream·F 출력·D slicing을 같은 registry로 관리한다.

### 비교

- K별 독립 학습 모델.
- 최대 K로만 학습한 모델을 그대로 prefix 절단한 경로(별도 학습 없음; 진단).
- 다중 K 공동학습 모델.
- nested PCA prefix(학습 없는 표준 압축).

동일 optimizer step 수, 동일 전체 학습 시간, 각 K에 대한 동일 노출은 동시에 같지 않을 수 있다. **동일 총 학습 예산 비교**와 **각 K 독립 모델의 총 학습 비용 비교**를 따로 보고한다. 하나의 checkpoint 장점은 저장량·전환 비용으로 측정한다. K별 E/D를 따로 저장했다면 하나의 공유 모델이라고 부르지 않는다.

이 실험이 실패하면 순서화된 표현의 최적화/용량 공유 문제가 남았다고 보고한다. 차원 순서화의 타 분야 성공을 현재 TSFM 효과로 옮겨 적지 않는다.

## 12. 자율 실행 순서와 분기

### 공통 큐

`자산·계약 → 단위/수치 검사 → 전체 가족 최소 커버리지 → V 판단 → 충분 학습·동일 비교 → 선택 잠금 → T 추론 → 재채점·비용·자료 생성 → main 게시·원격 검증`.

E1 전부 끝날 때까지 E2–E4를 방치하지 않는다. 기본 부모 모델이 생긴 자료부터 최소 결합·지연·다중 K 파일럿을 실행한다. 첫 자원 구간을 한 모델의 미세 튜닝에 모두 쓰지 않는다.

### 각 실험 다음에 남길 최소 의사결정 기록

`질문 / 기존 근거 / 관찰 / 경쟁 설명 / 유지·보류·확대·중단 / 판단을 바꿀 증거 / 다음 experiment_id / 사용한 split`.

단순 산술/버그 수선 외에 새 loss·구조·데이터 계약이 필요하면 이 계획의 변형 범위인지 확인한다. 범위 밖이면 제안으로 남기고 실행 승인 없이 추가하지 않는다.

### 분기 예시

| 관찰(V 기준) | 다음 행동 |
|---|---|
| K1/2는 유지되나 K1/4가 손해 | 등록된 K3/4와1/8을 채우고 압축 경계 보고 |
| Raw보다 좋지만 B0와 비슷함 | P의 고유 기여를 보류하고 단순 경로 우선 |
| G-only가 전 조합보다 좋음 | 오류 보완성·조건부 H 최소 비교는 수행, 불필요한 gate확대는 중단 |
| alpha 결합만으로 충분 | 복잡 H의 최소 대조 후 동등하면 단순 결합 선택 |
| 조건부 H가 좋아짐 | 과거 동용량 H·G_refit·Raw-conditioned가 설명하는지 확인 |
| 지연 압축 개선 | 같은 K·D·학습량·boundary/양방향 대조 유지 후 다른 자료로 확인 |
| ordered K가 손해 | 각 K의 노출·공유 D slicing/실제F연산 검사; 단순 개별 K 유지 |
| 마지막 epoch까지 감소 | 미해결, 관련 비교군과 연장; 예산 소진시 partial |
| T에서 악화 | 보고하고 freeze; 같은 T를 보고 새방법 선택하지 않음 |

수치가 작다는 이유로 근거 없는 1% 합격선·신규성 점수를 만들지 않는다. 개선 방향·크기·실행 간 변동·원자료 수·비용을 제시하고 유지/조건부/보류/기각을 판단한다.

## 13. 분석해야 할 항목

1. **성능:** 자료/task/horizon/K/seed별 normalized MSE·MAE, 물리 단위 보조지표, Raw/G/B0/LoRA/표준포트 대비 절대·상대 차이. 모든 분모0/NA 처리와 coverage.
2. **추가 기여:** B2−B0, 결합−G_only, conditioned−past_control, conditioned−alpha, compressed−Raw-conditioned. 표본·선정·정보·노출이 맞는 쌍만 계산.
3. **학습:** 실제 epoch/updates/고유 origins·unique target times·observed targets/초기 CP/선택 CP/gradient/clipping/seed/learning rate/미해결 상태. 제한된 A probe 점수는 전체 A loss와 구분.
4. **시간·채널:** 가까운·중간·먼 horizon, 채널별 오차, 원시점별 손실. 1–12/13–24/25–48처럼 기존H48 구분은 H12/H28에 그대로 복사하지 말고 `ceil(H/4),ceil(H/2),H` 규칙으로 정의.
5. **패턴:** A의 과거만으로 정한 변동·계절·지연상관·간헐성의 낮음/높음/NA 그룹을 모두 제공. 원인이나 배포 selector로 바꾸지 않음. 채널 간 상관은 인과성·필요정보의 증거가 아님.
6. **대표 예측:** 결과 전에 고정한 시작점과, 사후 최선/최악 사례를 다른 표식으로 모두 제공. 실제 시간·물리 단위·관측 mask·Raw/G/제안·LoRA를 같이 그림. 사후사례는 설명이지 성능증거의 대체가 아님.
7. **표현 진단:** prefix 일관성, E/D rank·latent variance·D 고정 출력공간 진단은 필요한 실패 경로에만 수행. 원미래 정답을 쓰는 oracle은 `ORACLE_ONLY_NOT_DEPLOYABLE`로 분리.
8. **가변 K:** 하나의 체크포인트가 여러 K에서 실제 F input을 줄이는지, 개별 K 학습 대비 오차·총학습·저장량·전환시간.
9. **자료 확대:** 작은 이질적 센서와 많은 유사 채널에서 나타나는 차이를 제시하되, 도메인 이름만 원인으로 삼지 않음. domain과C가 동시에 바뀌므로 subset 대조와 한계를 함께 보고.

### 불확실성

seed는 최적화 변동, 시간 블록은 평가 시점 변동이다. 채널×창×seed를 독립 표본으로 쌓아 유의성을 과장하지 않는다. 기본은 모든 seed 원수치·평균·범위, 시간 블록별 paired 차이를 제시한다.

선택적 block bootstrap은 A에서 정한 주기와H를 고려해 block 길이를 잠근 뒤 적용하고 채널 벡터를 함께 재표집한다. block 수가 작으면 CI 계산을 강제하지 말고 기술통계로 남긴다. 두 모델의 동일 origin/mask를 paired로 유지한다. 최종 T에서 여러 후보 중 최선만 골라 같은 CI를 확증처럼 쓰지 않는다. 통계 검정 방식을 새로 도입할 때 원문/표준 근거를 `ANALYSIS_PROTOCOL`에 연결한다.

## 14. 실제 비용과 메모리

### 필수 측정

- 배치 1과 8에서 동일한 **CPU 물리 입력 → 채움/정규화 → P/E → F → D/G/H → 역변환 → CPU 반환/동기화**를 측정한다.
- 모델 로딩 시간과 steady-state 추론을 구분한다. 저장 예측을 읽는 시간을 모델 추론으로 세지 않는다.
- 경로별 fresh process에서 실제 필요한 모델만 상주시킨다. G-only 프로세스에 Bolt를 로드하지 않는다.
- `resident_weights`, `active_adapter`, `peak_allocated`, `peak_reserved`를 MiB로 기록한다. `peak-resident`를 순수 activation 메모리라고 단정하지 않는다.
- 학습 peak는 gradient-enabled P/E/D, LoRA, 직접 모델을 같은 effective batch에서 별도로 측정한다. 백본 가중치 고정과 activation 역전파 제거는 다르다.
- Raw의 row microbatch `{full,32,8,4}`를 실제 장치에서 비교하고 예측 parity를 검사한다. 최소 메모리 Raw와 가장 빠른 Raw를 각각 보여준다.
- LoRA는 별도 base copy에 병합한 경로를 주 비용 비교로 사용한다. 병합 전후 예측과 원래 F의 복구를 검사한다.
- warmup 10회, 최소 실측 20회, 방법 순서의 balanced rotation을 사용한다. 반복 횟수는 이번 운영 설계이며 여러 독립 배포 환경을 대표하지는 않는다.
- general session과 exclusive process 결과가 다르면 모두 유지하고 같은 scope끼리 비교한다. 빠른 scope만 고르지 않는다.

### 정확도–비용 판정

V에서 정한 선택과 입력 기반 비용으로 후보를 고른다. 최종 T를 보고 허용 오차를 새로 정하지 않는다. 실제 메모리 상한이 주어지지 않았다면 예산별 곡선을 제시하고, 임의의 150MiB 같은 숫자를 실제 배포 제약으로 주장하지 않는다.

K/C를 속도 비율로 쓰지 않는다. 조건부 H가 b를 사용하면 b를 계산하는 F 비용을 모두 포함한다. G-only가 더 정확하고 더 저렴하면 현재 조합이 그 비교에서 불리하다는 반례를 남긴다. 장치별 안정적인 비용 차이를 확인하지 못하면 조건부 관찰로 제한한다.

## 15. 검증: 문서 점검과 실제 재계산을 구분한다

### 15.1 검증 수준

| 종류 | 실제로 해야 할 일 |
|---|---|
| `DOCUMENT_AUDIT` | 파일·필수 필드·링크·hash 검사 |
| `PUBLIC_STATS_REAGGREGATION` | 공개 SSE/SAE/count로 전체 macro 점수를 다시 집계 |
| `PREDICTION_RESCORING` | 저장된 실제 예측·정답·mask에서 별도 코드로 오차를 다시 계산 |
| `CHECKPOINT_REPLAY` | 저장 checkpoint와 원입력을 사용해 실제 재추론 |
| `TRAINING_RERUN` | 독립적인 재학습; 하지 않았으면 0으로 표시 |

앞 수준만 통과했다고 뒤 수준까지 완료했다고 쓰지 않는다. 문서 감사 PASS는 충분 수렴·성능 개선·독립 일반화의 증거가 아니다.

### 15.2 필수 단위·불변성 검사

- 원시점 인덱스, horizon, 입력 길이, 채널 순서, mask, 정규화 계약.
- 합성 배열에서 예측 origin 이후 값 변경 시 입력·예측 불변, 관측 0과 결측 0 구분.
- A 통계가 M/V/T 값 변경에 불변인지 합성 자료로 검사. 모든 target 경계 위반은 0이어야 한다.
- B0 K=C 초기 상태의 의도한 Raw parity, H 초기 상태의 G parity, 고정 F의 가중치 불변.
- 지연 압축의 prefix 검사, D의 lag×K 축 slicing, 다중 K의 실제 F stream 수.
- LoRA 병합 전후 parity와 원래 F 복구.
- data/config/model fingerprint가 달라지면 cache 재사용 거부.
- checkpoint 재개 시 다음 minibatch, optimizer, scheduler, RNG가 일치하는 짧은 실제 시험.

### 15.3 별도 채점과 실패 주입

훈련 코드의 `score`를 그대로 호출하는 것만으로 독립 채점이라고 하지 않는다. float64 별도 구현으로 채널 macro MSE·MAE를 계산하고 SSE/SAE/count도 대조한다.

다음 오류를 넣었을 때 실제 검사기가 실패해야 한다: 정답 1행 shift, 채널 swap, 잘못된 mask, SHA 변조, 실패 경로 누락, Kmax 전부 호출 후 작은 K 비용이라고 표시, 점수표의 숫자 변조. 탐지되지 않으면 검사를 수정하고 다시 실행한다.

FP32 추론 parity의 초기 기준은 `atol=1e-5, rtol=1e-4`, float64 집계는 `atol=1e-9, rtol=1e-7`로 잠근다. 이는 성능 합격선이 아닌 [추정] 수치 한도다. 실패 후 tolerance만 완화하지 말고 원차이·원인·수정 근거를 남긴다. 다른 기준으로 재검사하면 첫 실패와 버전을 보존한다.

### 15.4 checkpoint 재추론

자료별 최종 Raw/G/B0/제안/LoRA는 V에서 미리 정한 첫·중간·마지막 origin과 고정 난수 추출 origin으로 replay한다. 일부만 검사했다면 subset 검증이라고 표시한다. T를 사용하는 재채점은 최종 선택 잠금 뒤에만 수행한다.

동봉한 `audit_public_bundle.py`는 아래 새 공개 스키마의 **집계 재검산 도구**다. 이는 실험 실행기나 원예측 검증기가 아니다. 계획 패키지 작성 과정에서 합성 macro 평균·점수 변조·중복 통계 탐지의 세 자체시험을 수행했다. 실제 연구 결과 검증으로 집계하지 않는다.

## 16. main에서 검토할 수 있어야 하는 산출물

아래 이름과 필드는 이번 캠페인용 **새 제안**이다. 기존 파일이나 key라고 가정하지 않는다.

| 파일 | 내용 |
|---|---|
| `README_KO.md` | 목표, 전체/부분 상태, 핵심 수치, 반례, 한계, 재개 경로 |
| `CAMPAIGN_PLAN.md`, `CONFIG_LOCK.json` | 확정 계약, 시각, hash, 권한, 예산 |
| `LITERATURE_MAPPING.md` | 원문 범위, 이식 차이, 재현 여부, 코드 pin |
| `DATA_CONTRACT.json` | 자산, 범위, 컬럼, mask, 전처리, 표본, 노출 이력 |
| `UNRESOLVED_RESOLUTION.csv` | 기존 12개 미해결의 실제 후속 상태 |
| `EXPERIMENT_REGISTRY.csv` | 예정된 실험, 질문, 의존성, 필수/확대 구분, 상태·이유 |
| `RUN_INDEX.csv` | 명령, 시각, source/config hash, seed, epoch, 업데이트, 노출, 선택 CP |
| `SCORES.csv` | 실제 수치가 있는 경로의 점수와 계약 |
| `PUBLIC_ERROR_STATS/` | origin×channel×horizon-band별 SSE/SAE/count |
| `COST.csv`, `COST_TRIALS/` | whole/stage/trial/순서/장치/scope/메모리 |
| `PREDICTION_MANIFEST.csv` | 배열 shape/dtype/hash, model/data/cohort hash, 저장·공개 범위 |
| `SELECTION_LOCK.json` | T를 열기 전의 가족·K·epoch·seed 집합·결합 계수·선정 코드 |
| `AUDIT_REPORT.json` | 검증 수준별 실제 수행 범위, 실패·경고, 연결 hash |
| `CLAIM_EVIDENCE.csv` | 주장 → 원수치 행/그림/source, 지지·부분·반박·미해결 |
| `DECISION_LOG.md` | 관찰, 경쟁 설명, 다음 실험을 선택한 이유 |
| `REPORT_KO.md`, `PROFESSOR_BRIEF.md` | 결과, 대표 반례, 발표 설명과 한계 |
| `RESUME_QUEUE.json` | 미완료 ID, 상태, checkpoint, 재개 명령, 추가 예상 비용 |
| `PUBLIC_MANIFEST.sha256` | 공개 allowlist와 hash; 자기 자신은 대상에서 제외 |
| `REPRODUCE.md` | 실제 단위검사·재채점·replay·재학습 명령을 구분 |

### 공개 표의 최소 스키마

`SCORES.csv`에는 다음 필드를 둔다. 다른 운영 필드를 추가할 수 있다.

```text
run_id,dataset,panel,split,cohort,length,horizon,num_channels,latent_k,
lora_rank,seed,point_type,mse,mae,n_observed,n_origins,
execution_status,training_status,validity_status,source_sha,config_sha
```

`PUBLIC_ERROR_STATS/*.csv` 또는 `*.csv.gz`에는 다음 필드를 둔다.

```text
run_id,split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count
```

통계는 해당 normalized 점수와 같은 스케일이어야 한다. `horizon_start/end`는 0-based half-open이며 각 origin×channel의 `[0,H)`를 누락·중복 없이 나눈다. 관측 0개인 band도 count0/SSE0/SAE0으로 남긴다. 전체 origin×channel coverage가 필요하다. 미실행·무효 경로는 registry에 남기되 numeric 점수표에 가짜 0점으로 넣지 않는다.

채널별 SSE/count를 먼저 계산하고 그 채널 MSE를 평균한다. 관측 수가 다른 origin별 MSE를 그냥 평균해 같은 지표라고 하지 않는다. 물리 단위 통계는 다른 파일·metric key로 분리한다.

UTF-8 flat CSV/TSV와 줄바꿈 있는 JSON을 사용한다. 큰 JSON 전체를 CSV 한 cell에 넣지 않는다. 상대 변화는 `(new/base-1)*100`으로 계산하며 분모 0이면 NA다. 실행·학습·수치 유효성·과학 판정·게시 상태는 별도 필드로 관리한다.

### 공개 검산 자료와 권한

검토자가 SCORES만 믿지 않고 공개 SSE/SAE/count로 집계 점수를 다시 계산할 수 있어야 한다. 최종 후보와 모든 강한 기준선의 통계를 함께 제공한다. 공개 통계 자체도 데이터 라이선스를 확인한다.

기본 공개물에는 원자료나 실제 정답 배열을 넣지 않는다. 정답을 포함한 subset 공개는 재배포 권한과 사용자의 별도 승인이 모두 있을 때만 한다. 예측 subset·작은 어댑터 checkpoint도 라이선스와 민감성 검사를 통과한 범위에만 제공한다. 공개할 수 없으면 hash·원자료 취득 경로·재생성 명령을 제공하고, 외부에서 가능한 검증이 집계 수준임을 명시한다. 익명화만으로 사내 자료의 공개가 허용된다고 추정하지 않는다.

허용된 작은 어댑터는 safetensors 같은 안전한 형식을 우선한다. 백본은 공식 자산 pin으로 재현하고 main에 복제하지 않는다. 개별 공개 파일20MiB, 신규 공개 증분250MiB는 이번 자체 운영 상한이다. 플랫폼의 공식 한계라고 설명하지 않는다. 초과하면 분할·압축·집계하고 `git add .`로 대형 캐시를 올리지 않는다.

### 그림과 발표 설명

실제 수치로 여섯 묶음만 우선 생성한다: 자료/K 정확도, 시간 P의 추가 효과, G 기준 결합의 추가 효과, 학습량·시간 이전, 정확도–시간–메모리, 대표 예측·가변 K.

각 그림은 원수치 행, 실행 ID, 그림 생성 코드 hash를 참조한다. `PROFESSOR_BRIEF`에는 슬라이드별 질문·그림·한 문장 주장·강한 반론의 위치를 기록한다. 실제 PPT 제작은 이번 실행 범위에 넣지 않는다.

## 17. 안전한 main 게시 절차

### 시작과 작업 격리

1. `git status --short`, 실제 원격, HEAD와 원격 main을 확인한다. credential이 있는 URL은 로그에서 가린다.
2. unrelated 변경이 있으면 자동 stash/reset하지 않는다. 별도 worktree나 격리 브랜치에서 자기 파일만 작업한다.
3. 원격 main이 설계 기준 SHA보다 앞서면 바뀐 내용을 읽는다. 같은 실험이 완료돼 있으면 계약을 확인하고 재사용한다.

### 게시 전

공개 allowlist를 직접 만들고 그 파일만 stage한다. 토큰·키·전화번호·개인 경로·상담 텍스트·원자료 유출을 검사한다. 숫자 무결성에 중요한 오류가 남은 행은 확정 수치로 게시하지 않는다. 오류 상태와 원인 보고는 보존할 수 있다. 음성 결과라는 이유로 유효한 행을 지우지 않는다.

`git diff --cached --check`와 staged 목록·크기·라이선스를 검사한다. 루트 `README.md`는 새 캠페인으로 가는 최신 링크만 최소 수정하고 옛 링크를 보존한다. 새 `LATEST_INFORMS_RESULTS.json`이 실제 결과 경로를 가리키도록 한다.

### 실제 push

아래는 사전 검사를 통과한 뒤 수행할 **명령의 형태**다. 확인되지 않은 경로를 그대로 실행하지 않는다.

```text
git fetch origin main
# origin/main이 publication HEAD의 ancestor인지 확인한다.
# 자기 공개 파일만 stage하고 검산한 뒤 commit한다.
git commit -m "Publish INFORMS broad TSFM validation: results and reproducibility"
git push origin HEAD:main
```

원격이 앞서면 새 변경을 확인하고 자기 미공개 commit만 정상 통합한다. 충돌이 실험 source나 데이터 계약에 영향을 주면 다시 검증한다. `--force`, `--force-with-lease`, `reset --hard`, `clean -fd`는 사용하지 않는다.

브랜치 보호나 인증으로 거부되면 정책을 우회하지 않는다. 작업 브랜치/PR에 보관할 수 있지만, main 게시 완료라고 하지 않는다. main merge 또는 인증 복구가 필요하다고 보고한다.

예산·데이터 문제로 전체를 끝내지 못했어도, 검산된 부분 결과와 정확한 재개 큐는 main에 게시할 수 있다. commit 메시지와 README에 `PARTIAL`을 표시한다.

### push 후 원격 재확인

`git ls-remote origin refs/heads/main`과 `git fetch` 후 원격 commit을 읽는다. publication commit이 현재 main tip 또는 ancestor인지, README/SCORES/AUDIT/MANIFEST/재개 큐가 있는지, 그 commit에서 공개 파일 hash가 맞는지 확인한다.

로컬 commit 성공을 push 성공으로 바꾸어 말하지 않는다. 원격 SHA를 확인한 영수증은 로컬 출력과 최종 응답에 남긴다. commit 파일 안에 그 commit 자신의 SHA를 넣는 순환을 만들지 않는다. 필요하면 별도 receipt commit이 앞선 결과 commit을 참조하도록 한다.

원격이 동시에 더 진행돼도 자신의 commit이 ancestor이고 결과 파일이 유지되면 게시 유실로 단정하지 않는다. 반대로 브랜치에만 남아 있고 main에는 없으면 게시 완료가 아니다.

## 18. CLI 최종 보고 형식

다음 정보를 한 번에 제공한다.

1. **실행 범위:** 실제 자료·패널·C·K·모델·seed·epoch·시도 수. 재사용·미실행·무효를 구분한다.
2. **핵심 수치:** Raw, B0, G-only, 같은 정보 대조, 병합 LoRA, 표준 포트 대비 주지표·MAE·전체 비용.
3. **과학 판정:** 시간 P, 압축 예산, 결합 보완성, 지연 일관성, 가변 K 각각 유지/조건부/보류/기각과 근거.
4. **반례·공백:** 실패 자료, 미수렴, 표본·mask·라이선스·노출 한계, 대조가 없는 주장.
5. **검증:** 공개 집계, 원예측 재채점, checkpoint replay, 재학습의 실제 대상과 횟수.
6. **게시:** 원격에서 확인한 main SHA, 고정된 결과 경로, 파일 hash 검사, 미게시 사유.
7. **재개:** 추가 예상 비용, 정확한 명령, 남은 필수 experiment_id.

“학습했다”, “잘됐다”, “PASS”, “main에 올렸다” 같은 한 줄로 끝내지 않는다. 성능이 좋지 않아도 현재 방법의 한계와 다음에 필요한 최소 확인을 분리해 보고한다.

## 19. 참고문헌·출처

URL은 원문·표준 동작을 확인할 시작점이다. 데이터 배열이나 특정 라이브러리 버전까지 열람했다는 뜻은 아니다. 실제 사용 전 코드·자산을 pin한다.

1. **TSMixer: An All-MLP Architecture for Time Series Forecasting (2023/TMLR)**. 사용자 첨부 `2303.06053v5(1).pdf`, §4–5, Appendix C. https://openreview.net/forum?id=wbpxTuXgm0
2. **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (2024/ICLR)**. 사용자 첨부 `2405.14616v1(1).pdf`, §3, Appendix A. https://github.com/kwuking/TimeMixer
3. **AdaPTS (2025/ICML)**. https://proceedings.mlr.press/v267/benechehab25a.html ; 공식 코드 https://github.com/abenechehab/AdaPTS
4. **The Combination of Forecasts (1969)**. https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103
5. **DeepGLO (2019/NeurIPS)**, §5.2. https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf
6. **Learning Ordered Representations with Nested Dropout (2014/ICML)**. https://proceedings.mlr.press/v32/rippel14.html
7. **Matryoshka Representation Learning (2022/NeurIPS)**, §3. https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html
8. **Lightweight Online Adaption for Time Series Foundation Model Forecasts (2025/ICML)**, ELF. https://proceedings.mlr.press/v267/lee25ag.html
9. **Forecasting: Principles and Practice — Time series cross-validation**. https://otexts.com/fpp3/tscv.html
10. **Electricity hourly** 자료 설명. https://zenodo.org/records/3889830 ; UCI 원본 https://archive.ics.uci.edu/dataset/321/electricity
11. **M5 공식 자료**. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data (실제 데이터 파일과 접근권은 이번 설계에서 미확인)
12. **PEFT LoRA merge 문서**. https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
13. **Git push 문서**. https://git-scm.com/docs/git-push
14. **기준 캠페인**. https://github.com/CanelE452/hierarchical-tsfm-peft/tree/c7a06e522e1cd8e100a3e2291a7f693dd3f7125f/research/temporal_frontend_campaign_20261015_v1

## 20. 완료표

| 요구 | 완료 증거 | 부족하면 |
|---|---|---|
| 데이터 확대 | 새 전력 전체·교통 전체·M5 등록 패널의 실제 학습/추론 | 범위 확대 미완료 |
| K 확대 | 등록 grid의 실제 점수·F stream 수·비용 곡선 | K 커버리지 미완료 |
| 학습 미해결 | 12개 장부와 비교군 포함 학습곡선·종료 상태 | 수렴 주장 보류 |
| 결합 수선 | alpha/H/Gpast/Raw/B0/Grefit의 같은 cohort 비교 | Bolt 고유 가치 보류 |
| 지연 압축 | 실제 학습·prefix 검사·시간 경계 대조 | 해당 원인 판정 불가 |
| 다중 K | 같은 checkpoint의 실제 여러 K 호출·독립 K 비교 | 가변 예산 기여 미검증 |
| 최종 일반화 | 잠금 뒤 T 평가·노출 이력·모든 선정 seed | 독립 일반화 주장 불가 |
| 수치 무결성 | 별도 재채점·실패 주입·checkpoint replay | 감사 범위 제한 |
| main 게시 | 원격 commit·파일 hash·latest 연결 | 로컬 완료/게시 미완료 |

빈칸을 문장이나 그림으로 감추지 않는다. 실행 여부와 연구 가치의 판정을 분리하고, 다음 의사결정에 필요한 최소 확인으로 연결한다.
