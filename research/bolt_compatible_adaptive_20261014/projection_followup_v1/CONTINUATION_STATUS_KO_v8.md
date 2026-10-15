# 계속 실행 현황 v8 — 같은 RECON 초기값에서 D 고정 대조까지 완료

[확인] 종전 시간·자원 상한을 종료 기준으로 삼지 않고 계속 실행했다. 학습 후 출력 공간24cell의 독립 감사·보고, RECON 초기화2fit의 학습·감사·보고, D를 고정한 E-only2fit의 학습·감사·보고까지 끝났다. **Raw 정확도와 비용을 함께 유지하려는 연구 목표는 아직 미해결이다.** 다음 L7 matched-native6fit은 SOURCE/합성 준비 중이며 이 문서 시점 실제 학습·비용은0이다.

## 이번에 확인한 핵심

같은 Robin C17/K4·L512/H48·FIT256/TUNE64/노출 VAL30·원 scaler/origin/mask·현재 C anchor·L7 lag를 유지했다. 초기 PCA E/center와 기존 FIT 과거 RECON D32를 같게 맞춘 두 절차를 비교했다. 새 E-only2fit은 D/center를 고정했고 직전 공동 E/D2fit은 재사용했다. 각각600update·두 minibatch-order seed·고정 full-FIT 관측수 가중 MSE·AdamW.001/WD0/clip1·TUNE CP0/100/200/400/600 최소/이른 tie를 유지했다. 두 선정 잠금과 registry 뒤 DEV를 추론했다.

```text
조건                         TUNE MSE      DEV MSE       선정 CP
Raw Bolt 참고                .056951091   .340401887   없음
원 RECON CP0 참고            .102808326   .488093014   0
공동 E/D seed92601 재사용    .083935897   .534758375   400
공동 E/D seed92602 재사용    .083136591   .554825407   600
고정 D/E-only seed92601 신규 .100779781   .492266798   400
고정 D/E-only seed92602 신규 .100926120   .505250004   600
```

all-observed 채널 macro 정규화 MSE다. 원 RECON 참고는 기존 NumPy 캐시이며 새 Torch CP0 TUNE .102808326937와 등록 공차 내 차이가 있다. E-only와 공동 절차는 새 zero-update CP0 probe에서 같은 Torch 상태·4개 forward component가 byte-exact였다. E-only는 공동 학습보다 DEV2/2가 낮고 TUNE2/2가 높았다. 두 E-only 모두 원 RECON CP0와 Raw보다 DEV가 높았다. TUNE 순위와 DEV 순위가 반대여서 손실 감소만으로 해결을 판단하지 않는다. E-only clipping과 공동 E/D clipping의 범위도 다르므로 D 갱신만의 순수 인과효과라고 부르지 않는다.

![같은 초기값의 E-only와 공동 학습](reports/fixed_recon_d_e_adaptation_v1/figures/01_E_only_joint_errors.png)

## 남은 공간 제약과 예측 격차

아래는 같은 complete-C mask의 DEV다. D를 고정해도 E가 만든 known-past offset은 달라질 수 있다.

```text
선정 상태                   actual F     E(Y)         numeric min  Raw
공동 E/D seed92601          .534758375   .411841442   .359056671   .340401887
공동 E/D seed92602          .554825407   .437672702   .387814902   .340401887
고정 D/E-only seed92601     .492266798   .344458315   .306905621   .340401887
고정 D/E-only seed92602     .505250004   .356808128   .311177336   .340401887
```

[확인] E-only의 DEV minimum은 Raw보다 낮았지만 실제 F는 높았다. 반면 E-only TUNE minimum .059734017/.059579576은 같은 mask의 Raw .056951091보다 높았다. 원 PCA E/RECON D 참고의 DEV E(Y)는 .332944728, minimum .306885041이었다. 현재 표현과 구간에 따라 남는 제약이 다르다. 낮은 minimum은 과거 정보로 달성한 성능이나 독립 일반화 검증이 아니다.

minimum은 선정 D32·known-past offset·관측 출력 행·rcond1e-15의 retained numeric space에 조건부인 정규화 MSE 해다. 미래 미지수는 HK이며 H×7K가 아니다. 보편적인 모든 실수/FP32 미래의 하한이나 모든 표현의 불가능성을 주장하지 않는다. E(Y)는 실제 미래를 쓰는 비배포 진단이고 MAE·물리 오차는 MSE 해의 값이지 각각의 최소가 아니다. 원 false-bound flags/버린 성분을 보존했다. 새4cell은 학습 감사 중 두 불변 수학 경로로 새 계산한 것으로, 제3 외부 재생과 구분한다.

## 확인할 수 있는 산출물

- [D 고정 E-only 보고서·10CSV·3PNG3SVG](reports/fixed_recon_d_e_adaptation_v1/REPORT_KO.md)·[manifest](reports/fixed_recon_d_e_adaptation_v1/MANIFEST.json)
- [E-only 실제2fit/1200update](runs/fixed_recon_d_e_adaptation_v1/RESULT.json)·[독립 학습 검산/새 공간4cell](audits/FIXED_RECON_D_E_ADAPTATION_AUDIT_v1.json)·[독립 보고 PASS](audits/FIXED_RECON_D_E_REPORT_AUDIT_v1.json): 17파일·32상대링크·핵심 CSV 원필드·MD 숫자·3PNG3SVG 형식. ROOT가 PNG3개를 직접 확인했다.
- [RECON 초기화2fit 보고서·10CSV·3PNG3SVG](reports/recon_init_sensitivity_v1/REPORT_KO.md)·[학습 검산 PASS v2](audits/RECON_INIT_SENSITIVITY_AUDIT_v2.json)·[보고 PASS](audits/RECON_INIT_REPORT_AUDIT_v1.json)
- [보존된 RECON 감사 실패 v1](audits/RECON_INIT_SENSITIVITY_AUDIT_v1.json): 같은 path/SHA record의 bytes 항목 유무를 전체 dict로 비교한 메타데이터 오류였다. 원 실패/코드는 보존했고 v2에서 path/SHA 결속만 수정했다. 수치·학습·공차를 변경하지 않았다.
- [학습 후12상태/24출력 공간 보고서·7CSV·4PNG4SVG](reports/learned_output_space_v2/REPORT_KO.md)·[독립 과학 PASS](audits/LEARNED_OUTPUT_SPACE_AUDIT_v1.json)·[독립 보고 PASS](audits/LEARNED_OUTPUT_SPACE_REPORT_AUDIT_v1.json): v7에서 진행 중이던 별도 감사/보고가 완료됐다. 그림 겹침만 고친 v2의7CSV는 v1과 byte동일이다.
- [직전 F 입력 gradient4fit 및 같은 cohort 전체 비용](CONTINUATION_STATUS_KO_v7.md)

학습 감사는 저장 E/lag/D 산술·점수·CP·로그·순서·잠금을 검산했다. E-only의10CP와 선정 복원 시점에서 D/center bytes 불변, producer의1200개 per-step grad/flag 기록과 D norm0을 확인했다. F forward/backward·optimizer·매 update loss/bytes를 감사자가 독립 재생한 것은 아니다. 보고 검산도 과학 배열/모델을 재실행하지 않는다. 새 두 학습 묶음의 deployment 비용은 미측정이며 다른 cohort timing을 차용하지 않았다.

## 이어서 실행할 같은 디코더의 예측기 비교

[다음 계획](PLAN_MATCHED_L7_NATIVE_KO_v1.md)은 같은 frozen E/D 세 상태(원 RECON CP0와 E-only 선정 두 상태)에서 K별 독립 affine512→48 head를 두 순서 seed로 학습하는6fit/3600update다. CP0는 latent persistence이며 L7 known-past prefix 때문에 current-C 예측과 다르다. E/D·입력·손실·TUNE 선정 규칙을 고정하고 모든6잠금 뒤 DEV를 연다. 같은 CUDA/FP32/canonical32 표현 추출을 맞춘다. E0의 기존 실제 Torch Bolt latent를 현재 RECON D로 새 decode하는 것과 새로운 F forward를 구분한다. 예측 뒤 Native6/Bolt3/Raw1의 같은 물리 입력 whole-cost20셀100표본을 계획했다. 이6fit과 새 비용은 아직 미실행이다.

기존 native는 static/inverse/per-origin shrink K×C D였으며 현재7K×C L7 D의 matched fit은 아니었다. 기존 고정 미래 teacher LATENT_FUTURE_MSE는 [개발8fit](../../smallk_predictive_alignment_20261013/run_dev_v1/RESULT.json)과 [새 자료4fit](../../smallk_predictive_alignment_20261013/run_new_resolved_v1/RESULT.json)에서 이미 실행했다. q 이후 회귀와 현재 E(future)의 직접 일치를 혼동하지 않으며, 이 작업을 예측 목적 표현의 첫 시도로 부르지 않는다.

## 보존과 자원

보호된 TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT는 하지 않았다. HEAD da1114945f90b32017b3caf39b81538557af6fd1과 기존 tracked 변경을 재확인했다. 새 결과는 로컬이다. K streams+K 과거 buffer+현재 C anchor이므로 strict K-only가 아니다. 이미 노출된 DEV의 적응적 탐색이고 두 seed는 독립 자료가 아니다.

[자원 snapshot v8](audits/RESOURCE_SNAPSHOT_20261015_v8.json)은 E-only 보고 생성 뒤 마지막 독립 보고 감사 전의118terminal receipts/당시RUNNING0·중복 process group0·CPU user+system2,241.140625초·GPU process meterwall1,368.893959초·폴더2,427,279,221bytes다. legacy32/보호18SHA가 보존됐다. 이 continuation의 시점별 기록이며 전체 대화의 자원 합계가 아니다. 마지막 보고 감사와 뒤 SOURCE/합성 작업은 이 snapshot 이후이며 일부 SOURCE/fixture는 미계측이다. GPU meterwall을 CUDA kernel 시간으로 바꾸어 말하지 않는다. [상한 해제 권한](AUTHORIZATION_SCOPE_v1.json)에 따라 반복 승인 없이 같은 목적의 검산·다음 대조를 이어간다.
