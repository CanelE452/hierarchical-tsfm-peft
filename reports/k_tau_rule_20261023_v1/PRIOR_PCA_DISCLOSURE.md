# 기존 PCA 산출물 공개

2026-10-23 사용자 수정 G1 기준의 읽기 전용 감사다. 새 스펙트럼·누적 설명분산·τ별 K는 계산하지 않았다.

[확인] Jena memory_budget와 WAVE_002 dataset-search의 research/reports/cache 텍스트에서 네 데이터 **A_fit 누적곡선 또는 τ별 K 산출물은 발견하지 못했다**. JSON의 검색 범위·제외 범위와 한계를 따른다. 전체 과거 세션이나 외부 노트북의 부재를 보증하지 않는다. `data.py:195`의 cumsum은 관측 mask 개수이며 PCA가 아니다.

[확인] 각 데이터의 대표 seed92601 selected bundle에서 기존 A60 normalization metadata를 직접 확인했다. 아래 값은 기존 K 하나의 scalar이며 A_fit 곡선이나 τ별 K가 아니다.

```text
dataset           K   fit_end  explained_variance
jena              3   12579    0.7253609957424675
ETTh1             1   10452    0.38007787195435083
ETTh2_safe17400    1   10440    0.41327229395691173
appliances        4   11841    0.7134194406486349
```

정확한 bundle 경로는 동반 JSON에 보존했다. 네 bundle 모두 `basis_initialization=A_fitted_centered_PCA`다. 기존 두 캠페인의 보고서·로그 metadata 검색에서는 설명분산 scalar 출력 필드를 찾지 못했다. 이 파일이 해당 네 기존 scalar를 공개한다. 다른 K·seed나 A_fit checkpoint의 scalar를 전수 조사하지 않았다.

[확인] 기존 내부 PCA 계산은 `research/informs_broad_validation_20261015_v1/data_contract.py:114–128`이고, `fit_base.py:103–107`은 scalar/basis/center를 normalization에 남긴다. `research/b2_revin_20261018_v1/training_jobs.py:314–326`이 이를 bundle에 저장한다. 전체 A_fit/A60 고유값의 별도 저장은 조사한 JSON/MD에서 발견되지 않았다. binary 파일의 전체 spectrum 부재를 전수 입증하지는 않았다.

[확인] `reports/incremental_latent_adaptation_20261019_v1/COVARIANCE_SPECTRUM.csv:1–2`에는 기존 Jena 누적 스펙트럼이 있다. 이는 **M_cal 내부 M_fit의 표준화한 과거 window 공분산**으로, 반복된 history 관측을 유지한다. `research/incremental_latent_adaptation_20261019_v1/probes.py:128–132`가 생산하며 `diagnostics.py:13–17,52`가 공개한다. 계약은 `PARENT_AND_DATA_CONTRACT.json:43,85,273`에 M_cal/M_fit 및 A60 부모 끝점을 기록한다. A_fit raw-row 규칙과 구간·표본구성이 다르고, 사용자가 명시한 허용 예외다. 이 스펙트럼에서 새 K를 유도하지 않았다.

수정 G1에서는 감사 범위 안의 중단 근거가 발견되지 않았다. 기존 내부 PCA 계산과 scalar는 공개하고 허용한다는 사용자 설명으로 기존 G1/G2의 충돌이 해소되었다. 원시 데이터, V/T·private prediction 배열, loader, 학습, eigensolver는 실행하거나 열지 않았다.
