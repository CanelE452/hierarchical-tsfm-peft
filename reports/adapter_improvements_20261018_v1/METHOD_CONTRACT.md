# 방법·비교 계약

[확인] 계획·과학 source 식별자는 EXPERIMENT_PLAN.json에 봉인된다. 보고서 producer hash는 REPORT_STATE.json/FIGURE_INDEX.json에 따로 기록하며 보고서 편집은 과거 과학 cache identity를 바꾸지 않는다.

통계 schema: CHANNEL_HORIZON_STATS의 (row_id,horizon,channel_id)는 RUNS.row_id로 조인한다. SSE/SAE/count는 원 full precision 그대로이며 contextual 여부와 통계 source binding은 RUNS의 record_origin/statistics_source_file/statistics_source_sha256을 따른다. SAME_UPDATE와 COMPARISONS도 같은 짧은 row_id를 사용하고, 원 긴 ID는 RUNS.producer_row_id다. 원 ID 중복이나 조인 누락은 생성 오류이며 통계를 조용히 제외하지 않는다.

PAIRED_ORIGIN_EFFECTS는 공개 origin별 SSE/SAE/count에서 계산한 기술 통계다. canonical matching은 index 정렬을 증명하며 실제 시간 ID 대신 origin_index를 공개한다. 한 origin의 고정 채널 중 N_c=0이면 UNDEFINED이고, IID 독립성이나 p-value를 주장하지 않는다.

고정 경로는 X→P→E→동결 단변량 F→lag-major D→원채널 미래다. P 후보는 kernel9/25, width32, bias 없음, causal replicate padding, 마지막 1×1만 zero-init이며 1,152 params다. P_SPLIT은 trailing mean25의 잔차/추세, 대조는 X/2 두 branch다. CP0 identity 이후 학습된 P의 가역성·완전 정보 보존은 보장하지 않는다.

학습 stride1, 평가 strideH; seed92601/92602; CP0/.25/1/3. A_fit/A_stop으로 endpoint를 선택한 뒤 scaler/PCA/DCA를 fresh A60 fit한다. AdamW lr.001, weight decay0, clip1, effective16/micro4, F row128, FP32, checkpoint_inputsFalse다. V는 선택에 사용하지 않는다.

D_SHARED/D_HORIZON/D_LINEAR/D_GAIN은 같은 fixed P/E와 fitting RMS를 사용한다. linear/gain gamma={1,10,100}, lambda={.0001,.001,.01,.1,1}, degree1이다. 불균등 mask는 전체 채널 count N_c 정규화를 유지한다. head별 설정 선택이 다를 수 있다.

DCA는 공식 고정 source objective, T5/n_init1/PCA64 init, CPU FP64 L-BFGS-B maxiter200/maxfun1000/120초를 사용한다. A_fit/A60 scope를 분리하며 미수렴은 downstream 정상 arm을 허용하지 않는다. 저장 정렬 E32를 설치한 뒤 actual FP32 torch pinv로 D[0]을 만든다. deployment loader는 DCA를 최적화하지 않는다.

MATCHED는 같은 data/split/mask/scaler/origins/channel order/F snapshot/device/core precision/recipe를 증명하고 등록 intervention만 허용한다. learned checkpoint 전체 SHA가 같아야 한다는 뜻은 아니다. independent verifier의 실제 NPZ canonical fingerprint를 우선하며 다른 hash convention은 직접 동치로 취급하지 않는다. 공통 endpoint는 epoch/update/exposure/sampler도 같아야 한다.

선택-refit은 같은 선택 recipe에서 각자 선택된 epoch를 허용하고 MATCHED_RECIPE_SELECTED_REFIT로 표시한다. 공통 endpoint는 MATCHED_COMMON_ENDPOINT다. 누락된 필수 signature는 INCOMPLETE, 실제 차이는 CONTEXTUAL이다.

G-only는 원채널 own-history의 channel-shared L→hidden→H innovation + current anchor 직접 예측기다. F를 호출하지 않는다. B2는 TSMixer 아이디어 일부를 적용한 변형, B4/새 P는 TimeMixer 전체 재현이 아니다. 공식 출처와 차이는 METHOD_REFERENCES.md를 따른다.

비용은 physical+mask→fill→normalize→P/E/F/D/refiner→inverse→CPU의 whole path다. warm10/timed20, 별도 stage probe, CPU BLAS2/GPU B1/B8을 분리한다. p95는 20개 표본의 기술 통계다. cost_valid는 정확한 scored bundle reload와 full-V parity 통과 후에만 성립한다. cold/준비/검색/refit/평가/메모리 범위는 분리하며 미측정은 UNKNOWN이다.
