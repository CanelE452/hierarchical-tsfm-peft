# Exposure Audit

[확인] 이전 PEFT 노출 장부와 fresh manifest를 읽어 BDG2/Bull Office 노출 상태만 확인했다.
보호 E1/E2 예측이나 지표는 계산하지 않았다.

- Ledger: `E:\CODING\proj\mltimeseries\results\peft_paper_closure_v1\data_exposure_ledger.csv`
- BDG2 rows in ledger: 2
- Bull Office exact rows in ledger: 0
- Fresh manifest Bull READY mention: True
- Stage A certified clean units zero in holdout audit: True
- Sealed final certified units zero in holdout audit: True

[판정] Bull Office는 이번 로컬 보호 후보로 봉인할 수 있지만, raw family가 이미 inspect된 BDG2이므로 완전 독립 holdout이나 독립 재현으로 부르지 않는다.

## Five-Repo Bull Search

[확인] 원자료·`.cache`·대용량 배열을 제외하고 `Bull_office`, `Bull Office`, 단어 경계 `Bull`을 검색했다. mltimeseries에서는 fresh 후보 준비와 관련 진단 기록만 확인됐고, covariate-trust-pilot 및 tsfm-peft-method-screen에서는 0건이었다. hierarchical-tsfm-peft의 hits는 현재 연구 폴더의 계획·계약·감사 기록이다. forecast-revision-peft는 로컬 checkout이 없어 이 머신에서는 미확인이다.

- mltimeseries: [확인] matches=42
  - `.\_docs\history\2026-09-12.md:19:[확인] `results/peft_paper_closure_v1/fresh_stage_a_candidate_manifest.{md,json}` 및 read-only 재현 script 추가. Household post-P1은 train48h 결측 창 때문에 BLOCKED; BDG2 Bull Office2016A는 target-blind READY. 별도 source hash/split geometry/각 target 관측률 검산과 원자료 재현 PASS. Fresh 학습·forecast0, pretraining overlap UNKNOWN.`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:211:      "status_reason": "[판정] The candidate has fixed target-blind source/period/series, no local PEFT target-label overlap for the selected Bull office meters, feasible L336/H48 train/V/E1/E2 splits, and passes aggregate plus per-origin availability gates.",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:222:        "selection_rule": "Exclude previously exposed Eagle/Lamb sites; prefer Office usage to stay close to prior BDG2 office diagnostics; iterate site_id alphabetically; choose the first site whose first four metadata-order office electricity meters pass target-blind availability/nonconstant gates. Bobcat was checked first and failed availability; Bull was the first passing site.",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:223:        "site_id": "Bull",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:266:          "Bull_office_Lilla",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:267:          "Bull_office_Hilton",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:268:          "Bull_office_Myron",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:269:          "Bull_office_Nicolas"`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:272:          "Bull_office_Lilla",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:273:          "Bull_office_Hilton"`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:276:          "Bull_office_Myron",`
  - `.\results\peft_paper_closure_v1\fresh_stage_a_candidate_manifest.json:277:          "Bull_office_Nicolas"`
- forecast-revision-peft: [미확인] local checkout missing at `E:\CODING\proj\forecast-revision-peft`
- hierarchical-tsfm-peft: [확인] matches=198
  - `.\research\tsfm_peft_development_20260926\data_contract.json:98:        "Bull_office_Lilla",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:99:        "Bull_office_Hilton",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:100:        "Bull_office_Myron",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:101:        "Bull_office_Nicolas",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:102:        "Bull_office_Mai",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:103:        "Bull_office_Anne",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:104:        "Bull_office_Rob",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:105:        "Bull_office_Chantel",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:106:        "Bull_office_Efren",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:107:        "Bull_office_Yvonne",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:108:        "Bull_office_Ella",`
  - `.\research\tsfm_peft_development_20260926\data_contract.json:109:        "Bull_office_Ivette",`
- covariate-trust-pilot: [확인] matches=0
- tsfm-peft-method-screen: [확인] matches=0

[판정] 검색 범위 안에서는 2026-09-12 이후 Bull Office 예측·학습·평가 결과 노출을 새로 확인하지 못했다. 다만 raw family inspect와 pretraining overlap UNKNOWN은 남아 있어 본 개발을 막지는 않되 독립 최종 확증으로 부르지 않는다.

## Sealed Bull Office IDs

```text
Bull_office_Lilla
Bull_office_Hilton
Bull_office_Myron
Bull_office_Nicolas
Bull_office_Mai
Bull_office_Anne
Bull_office_Rob
Bull_office_Chantel
Bull_office_Efren
Bull_office_Yvonne
Bull_office_Ella
Bull_office_Ivette
Bull_office_Marco
Bull_office_Claudia
Bull_office_Sally
Bull_office_Debbie
Bull_office_Trevor
```

## Train-Only Selection

Rule: Train-only observed fraction >= 0.95 and train nonconstant; no protected target outcome used.
Selected count: 16 / 17

```text
Bull_office_Lilla
Bull_office_Hilton
Bull_office_Myron
Bull_office_Nicolas
Bull_office_Mai
Bull_office_Anne
Bull_office_Rob
Bull_office_Chantel
Bull_office_Efren
Bull_office_Yvonne
Bull_office_Ella
Bull_office_Ivette
Bull_office_Marco
Bull_office_Claudia
Bull_office_Sally
Bull_office_Debbie
```
