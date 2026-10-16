# 미완료 상태와 재개 경계

## 별도 post-T CPU 작업 진행

[Jena4조건 검색 완료 보고서](postT_V_only_cpu_lag_20261016_v1/PROGRESS_E3_KO.md)의 새 ID들은 동일 K6·seed·손실·노출 일정에서 CPU로 새로 시작한 개발 비교다. 네 조건 모두3epoch 검색을 완료했으며, 선택은 B0/과거 지연3epoch와 B2/양방향 지연1epoch다. 두 조건의 새 A60 적합·V43 예측은 완료했고 나머지는 진행 중이다. [이전1epoch snapshot](postT_V_only_cpu_lag_20261016_v1/PROGRESS_E1_KO.md)과 구분하며, 기존 중단 학습을 재개하거나 원 선택 잠금을 해제하지 않았다. 기존 작업이 실행 중이면 새 중복 작업을 시작하지 않는다. 새 CLI도 기존 run/prediction 경로를 덮거나 재개하지 않는다.

네 조건 전체의 V/독립 검산/비용과 [고정 PCA6조건](postT_V_only_cpu_fixed_pca_20261016_v1/README_KO.md), [다중 K의 새 CPU 비교](postT_V_only_cpu_multik_20261016_v1/README_PREPARATION_KO.md)는 아직 미완료다. Raw/ridge/G의 같은 CPU 비용도 fresh6-origin 수치 parity 후 측정하며, 이전 replay cache를 새 CPU forward로 세지 않는다. 아래 원471행의 상태·미완료192·GPU3epoch 미달48은 그대로 보존한다. 이번 CPU 진행을 T 관측 전의 원 GPU 최소 실행이나 독립 일반화 완료로 바꾸어 세지 않는다.

**PARTIAL.** 필수 학습의 실제 미달과 재개 경계다. 정확한 ID·원 상태·이유·참조 명령은 [RESUME_QUEUE.json](RESUME_QUEUE.json), [RESUME_COMMANDS.ps1](RESUME_COMMANDS.ps1)에 보존한다. 이미 T를 보았으므로 학습 명령은 PRE_SELECTION_REFERENCE_ONLY이며 기존 잠금 아래 즉시 실행할 큐가 아니다. 새 development·분리된 평가 설계 및 resource audit이 먼저 필요하다.

## 등록된 학습·범위

동결471행 중 COMPLETE279/nonCOMPLETE192다. 미완료192는 PARTIAL_SCREEN025124+PENDING54+BLOCKED9+BUDGET3+LT3 2이며 GPU3epoch 미달48은192 안의 부분집합이라 더하지 않는다. 재개 reference union439는 원snapshot437과 빠졌던 selected partial prefix2를 포함하고, 미완료192+이미 COMPLETE인 reference247이다. 439를 추가 실행 수로 부르지 않는다. E1 neural56target과 CPU fresh9의 별도 장치 조건을 보존한다.

B0/B2 우선 K 최소28조건은 V가 있으나 등록 전체40조건 중 다음8개 V가 없다.

```text
primary_beijing_E1_B0_k9_s92601
primary_beijing_E1_B2_k9_s92601
primary_jena_E1_B0_k16_s92601
primary_jena_E1_B2_k16_s92601
primary_electricity_E1_B0_k241_s92601
primary_electricity_E1_B2_k241_s92601
primary_pems08_E1_B0_k128_s92601
primary_pems08_E1_B2_k128_s92601
```

neural dataset/K 최소56조건 중 GPU3epoch 검색 완료8개, CPU fresh를 별도로 포함17개다. GPU 기준48조건의3epoch가 남는다. selected refit epoch와 search 최대 epoch는 다르며 같은 artifact/state가 선택돼도 신규 search 수행을 지우지 않는다. CPU fresh와 GPU continuation을 같은 시도로 합치지 않는다.

Beijing shared/max 새3epoch parent는 budget으로 BASE/V가 완성되지 않았다. 오래된0.25 prefix V를 새3epoch 결과로 재표기하지 않는다. E3 주 비교 Beijing/PeMS12개는 초기 V가 있지만 대부분 충분 학습이 남았다. Jena/Electricity E3 등록12, channel subset24, Beijing E4 미완료3epoch 부모2+그 prefix4, 전체 K 미실행8을 exact ID의 비가산 부분집합으로 남긴다. 추가2seed, Electricity/PeMS secondary, subset 비교 계약·10/30/100epoch·추가 cutoff 같은 미등록 요구에는 새 ID/명령을 임의 생성하지 않는다. secondary Beijing/Jena12개를 main4 전체 시간 전이 검증으로 쓰지 않는다.

M5 3panel/등록9 placeholder는 실제 접근·라이선스 자산이 없어서 차단이다. 우회 미러·자동 약관 동의·없는 자산 추정으로 재개하지 않는다. 이 차단을 접근 가능한 다른 블록에 적용하지 않는다. Robin은 별도 짧은 stress다.

## E0의 별도 미해결

원12의 exact join은4800 endpoint 완료9·부분4291/4800 한 개·미시작2이고 추가 controls16은 모두 미시작이다. 전체28=완료9+부분1+미시작18이며 endpoint4800 미완료19다. 원12·controls16과9trajectory/55checkpoint 평가의 분모를 분리한다. 완료9도 수렴 완료가 아니며9600/19200 및 동등 대조가 끝난 상태가 아니다. 과거 RNG가 없는 경로를 exact continuation으로 부르지 않는다.

공식 B55 checkpoint 실제 추론·독립 scorer는 [registry b672…](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0780.json.gz)에서 완료됐다.9개 legacy ID의55 checkpoint 모두 원 B curve와 MATCH(atol1e-7/rtol0)했고 새 학습을 수행한 것은 아니다. B는 already-seen 개발 구간이며 C/protected TEST 평가가 아니다. 원 학습 endpoint 부족·장기 수렴·controls16의 미완료는 그대로다. E0 후속은 잠금된 신규 E1–E4/T의 선택으로 되먹이지 않는다.

## 비용과 검증

현재 core72는60 MEASURED/12 FAILED다. Jena Raw B8은4variant 모두 추론 성공했으나 원 parity gate가 실패했다. 원 tolerance와 실패 영수증을 유지하고 누락 timed 비용을 완료로 세지 않는다. Electricity 원8건은 반환 dict 저장 전 assertion으로 종료돼 당시 variant별 수치/OOM 세부 원인은 미확정이었다. [원 gate 확인 영수증](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0250.json.gz)에 당시 확인 범위를 보존했다. 이후 [같은 원 함수의 추가 실제 진단](diagnostics/electricity_raw_parity_report_20261016_v1/REPORT_KO.md)에서 B1/B8 각4variant가 모두 반환했고 OOM은 재현되지 않았다. full 자기비교 외 분할6조건은 원 공차를 통과하지 못했다. 이 재현 결과로 모든 역사적 실패의 직접 원인이나 특정 kernel을 확정하지 않는다. 기존 실패12건과 timed 비용 미완료 상태는 유지한다. 이 경로의 실패가 독립적인 다른 비용 경로를 중단시키는 이유는 아니다.

최종 K전환은20scope3600trial(원exact1440+fresh2160) 모두 resource-valid로 완료·독립 원수치 검산됐다. 이전 제한 실행166(165valid/1invalid)은 역사적 부분/실패로 보존하고3600에 더하지 않는다. 이 비용 완료가 shared/max 미완료 학습이나 학습된 순서화의 이득을 채우지 않는다. 정확한 deep-G Hpast fresh8은 미측정이고 Beijing 심화 LoRA 새 비용은169.6135초 잔여로 최소240초 admission을 통과하지 못했다.

checkpoint replay50은50PASS(41fresh+9reuse)이며 고유 replay70과 겹친다. 최종 whole 감사는 표준499=V405+T46+M48, 공개통계451, legacy B55/errors0/global true다. 제공 public entry451gzip, horizon1353band, core137/96/6850/274 및 K20/3600 검산도 완료했다. 이러한 평가·검산 완료는 mandatory coverage false와 미완료 학습을 바꾸지 않는다. T가 관찰된 뒤 모델/K/epoch/방법을 그 T에 맞춰 다시 고르지 않는다. 공개 allowlist·최종 링크/그림/복원·실제 remote main 확인은 게시 단계에 남는다. 실제52쌍 사후 기술 분해(전체52/block2990/horizon156/pattern624)는 완료됐지만 기존14쌍과 독립합산하지 않으며 미완료 학습·seed·시간 전이를 채우지 않는다. v3 V405/T46 그림과 [학습 보완9stem/18이미지](publication_derivatives/gzip_extra_aux_20261016_v2/files/item_0000.json.gz)는 생성됐고 Beijing V 범례 분리를 실제 확인했다. [extraaux2995 전수·독립·반복 원형 복원](publication_derivatives/gzip_extra_aux_20261016_v1/BUILD_RESULT.json)과 [추가77](publication_derivatives/gzip_extra_aux_20261016_v2/BUILD_RESULT.json)은 PASS다. 합3072파일의 [원바이트 복원 색인](RESTORE_INDEX_KO.md)과 [후속 실제 집계·포장 감사](FINAL_POSTPROCESS_AUDIT.json)는 실제 생성됐으며 원 AUDIT_REPORT의 당시 pending snapshot은 보존한다. 이 완료를 남은 학습·전그림 가독성·최종 전체 게시 검증 완료로 확대하지 않는다. 이 보고서 조립시점(게시전검사)에는 추가 commit/push가 미실행이었다.

## 실행 원칙

원 partial·checkpoint·receipt·source snapshot을 덮거나 종료하지 않는다. 진행 중 owner와 같은 GPU를 중복 점유하지 않는다. 재사용은 exact source/config/data/F/scaler/initialization/cohort와 artifact가 맞을 때만 한다. 새 ID·학습 recipe·정보 시점이 달라지면 기존 결과를 같은 조건으로 덮지 않는다.

선택이 locked된 뒤 남은 학습을 기존 T에 맞춰 이어서 모델/K를 바꾸지 않는다. 새 개발은 별도 등록·분리된 평가로 다루며 이미 본 T를 선택 자료로 재사용하지 않는다. 비용·시간·저장 한도는 실제 중앙 ledger와 원 승인 계약이 우선한다. 미측정 경로를 포함한 총 ETA나 추가 자원으로 전체 완료를 보장하지 않는다.

총 추가 GPU/CPU/wall 시간은 미측정이다. 저장 search/refit receipt의 seconds·updates는 해당 자료/호출의 관찰값이며 누적 update와 호출시간을 혼동하거나48조건에 일괄 외삽하지 않는다. 최초 게시 조립시점 GPU 누적은43030.3865/43200초였다. 이후 Electricity 원 parity 진단에62.677388429641724초를 추가 정산해 누적43093.06391596794초, 잔여106.93608403205872초다. [실제 CLOSED·상한 준수 영수증](diagnostics/electricity_raw_parity_prepare_20261016_v1/GPU_AUDIT.json)에 원 수치와 자식 종료 확인을 보존했다. 학습 최소 미달은 바뀌지 않았다.

## 게시 후 확인

첫 PARTIAL 게시 commit은 `61b85f9e17839ddfb49b35dfca97757b9e03d18d`이며 당시 원격 main과6042개 파일의 경로·Git blob을 대사했다. 대표 보고서·그림7개는 GitHub 실제 HTTP200과 SHA도 일치했다. 위의 게시 전 대기 문장은 이전 조립시점의 기록이다. 추가 parity 보고서는 학습·T 평가를 수행하지 않은 후속 진단이며, 원 source39·registry·selection lock과 실패8건은 변경하지 않았다.

완료된 분석 그림의 실제 공개 경로는 [PUBLIC_FIGURES_KO.md](PUBLIC_FIGURES_KO.md)를 따르며, 그림 존재를 남은 학습 완료의 근거로 세지 않는다.
