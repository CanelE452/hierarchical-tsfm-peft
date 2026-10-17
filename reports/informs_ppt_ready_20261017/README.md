# INFORMS 발표용 결과 패키지

발표 본문에 사용할 기존 결과의 원수치, 비교 조건, 보고서, 생성 코드와 그림을 모았다. **PeMS C32/K8 B0·B2의 최종 성능은 미완료**이며, 이 패키지의 게시가 전체 실험이나 MASTER 완료를 뜻하지 않는다.

- [상세 결과 보고서](README_KO.md) · [슬라이드별 표·그림과 주장 범위](PPT_RESULT_MAP_KO.md) · [발표자 메모](PPT_SPEAKER_NOTES_KO.md)
- [본문 원정밀도 CSV](PPT_MAIN_TABLES.csv) · [보조 결과 CSV](PPT_ABLATION_TABLES.csv) · [주장과 근거](PPT_CLAIM_EVIDENCE_MATRIX.csv) · [미완료 36셀과 이유](PPT_PENDING_CELLS.csv)
- [원 run·데이터 계약·비용 출처와 SHA](SOURCE_BINDINGS.json) · [고정 사례 선택과 V 재채점 근거](QUALITATIVE_SELECTION.json)

Jena 두 seed에서는 B2의 MSE가 B0보다 낮았다. Jena의 두 full-history linear no-Bolt 대조보다도 낮았으나 Ridge·Direct G·새 CPU LoRA보다 정확하지 않았다. 선택된 학습량과 용량 차이, 반복 개발 V, 자료별 반례를 함께 보고한다. PeMS C170의 기존 대조와 C32의 미완료 비교를 구분한다.

## 발표에 넣을 그림

### 시간 문맥: Jena 두 seed

![Jena B0와 B2의 두 seed별 MSE·MAE](figures/JENA_B0_B2_TWO_SEEDS.svg)

### 동결 Bolt와 no-Bolt 대조

![Jena와 PeMS C170의 Bolt 및 직접 예측 대조](figures/BOLT_VS_NO_BOLT_TWO_DATASETS.svg)

### 강한 기준선과 현재 위치

![장치와 학습 조건을 표시한 기존 정확도 표](figures/MAIN_ACCURACY_TABLE.svg)

### 정확도와 실제 비용

![CPU2 B8 warm 전체 추론 시간과 MSE](figures/ACCURACY_COST_TRADEOFF.svg)

### 사전에 고정한 예측 사례

![첫·중앙·끝 origin과 채널의 고정 예측 사례 18패널](figures/QUALITATIVE_FORECASTS.svg)

곡선은 A60/refit 채널별 z-score 단위다. 사례는 오차를 보기 전에 origin·채널의 첫·중앙·끝으로 고정했다. PeMS C170 그림은 C32 B0/B2의 빈칸을 채우지 않는다. 장치·학습량 차이와 현재값 anchor 경로는 상세 보고서의 각주를 따른다.

## 생성 코드와 재현 조건

- [extract_qualitative_v.py](extract_qualitative_v.py): 기존 V 예측의 지문·FP64 재채점과 고정 사례 추출.
- [build_ppt_results.py](build_ppt_results.py): 결속된 기존 근거로 표와 SVG 생성.
- [verify_ppt_package.py](verify_ppt_package.py): 원수치·비용·누락 셀·상대 경로 검산.

세 코드는 기본 실행에서 입력을 읽거나 출력을 쓰지 않는다. 실제 생성에는 원 private evidence/inventory, 예측 배열, 정확 SHA, 유효한 실행 scope와 새 출력 경로가 필요하다. 원 데이터·NPZ·체크포인트·곡선 값 CSV는 공개하지 않는다. 원 scope의 마감은 끝났으므로 과거 명령을 그대로 실행해 새 실험이나 예산을 재개할 수 없다.

출처 경로를 완결하기 위해 기존 [PeMS B0 V manifest](../../research/informs_broad_validation_20261015_v1/predictions/primary_pems08_E1_B0_k43_s92601__e1/V_select/PREDICTION_MANIFEST.json)와 [PeMS CPU G V manifest](../../research/informs_broad_validation_20261015_v1/predictions/primary_pems08_E1_DIRECT_G_k0_s92601__cpu_fresh_e3/V_select/PREDICTION_MANIFEST.json)도 원 경로와 바이트 그대로 포함했다. 두 파일은 평가 조건·배열 형태·SHA·기존 V 집계값의 metadata이며 예측 배열 자체는 아니다. manifest의 cache 상대 경로는 private 재현 자산의 위치를 뜻한다.

## 생성 검산과 Git 게시 기록의 구분

[PACKAGE_VALIDATION.json](PACKAGE_VALIDATION.json)은 **게시 전 생성 시점**의 검산 영수증이다. 원점수 108셀·비용 86셀·V archive 8개의 재채점·SVG 5개를 검증했으며, 과학 실험 전체는 미완료로 기록했다. [그림 생성 영수증](FIGURE_BUILD_VALIDATION.json)도 같은 시점의 기록이다.

원 19개 파일과 영수증은 바이트 그대로 게시한다. 따라서 상세 보고서와 원 영수증의 `로컬 산출물`, `LOCAL_ONLY_NO_COMMIT_PUSH_THIS_REQUEST`는 당시 상태를 뜻한다. 이후 사용자의 명시 요청에 따른 Git 게시 여부는 이 페이지의 Git commit 이력과 원격 main으로 확인한다. 게시 기준 부모 commit은 `d8fe799e06fd4bb0703163c50861a7214ab2c3e8`이다.

[PUBLICATION_FILES.csv](PUBLICATION_FILES.csv)는 이번 게시 파일의 경로·바이트 수·SHA-256 목록이다. 목록 자체만 자기 해시에서 제외한다. 상대 링크와 그림 파일명을 대소문자까지 맞췄으며 저장소 루트 README에서도 이 페이지로 연결한다.
