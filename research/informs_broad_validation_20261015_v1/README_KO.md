# 고정 TSFM의 다변량 적응 실험

**후속 CPU 학습 진행:** [Jena B0/B2/지연 인코더4조건의1 epoch 중간 보고서·곡선·코드](postT_V_only_cpu_lag_20261016_v1/PROGRESS_E1_KO.md). 새3epoch 검색과 이후 A60 적합·V·검산·비용은 아직 미완료다. 이 별도 post-T development는 아래 동결된 원471행의 완료 수를 변경하지 않는다.

**PARTIAL — 실행 결과 부분 공개·필수 학습 미완료.** 고정 Chronos-Bolt의 stream 수를 C에서 K로 줄일 때 시간 처리·채널 압축·예측 결합·지연·가변 K가 예측과 전체 비용에 무엇을 추가하는지 강한 native/Raw/LoRA 대안과 비교했다. 일부 결합 이득은 관찰됐지만 Beijing T에는 V의 개선이 유지되지 않았고 PeMS H_B2–LoRA T MSE 차이는약0.23%로 작다. 일반적 우월성·TSFM 유지의 필수성은 확정하지 못했다.

- [상세 결과·방법·다섯 축 판정](REPORT_KO.md) · [교수님 상담 핵심](PROFESSOR_BRIEF.md)
- [점수](SCORES.csv) · [비용](COST.csv) · [실행 범위](RUN_INDEX.csv) · [검산](AUDIT_REPORT.json) · [주장과 증거12행](CLAIM_EVIDENCE.csv)
- [최종 실제 그림 색인](PUBLIC_FIGURES_KO.md) · [재현/복원](REPRODUCE.md) · [미완료/재개 경계](RESUME.md)
- [후속 실제 집계·포장 감사](FINAL_POSTPROCESS_AUDIT.json) · [원바이트 복원 색인](RESTORE_INDEX_KO.md) — 원 AUDIT_REPORT의 당시 pending snapshot과 후속 완료를 구분한다.
- [최종 report snapshot](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_2072.json.gz)
- [원 whole 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0583.json.gz) · [제공 public entry 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0592.json.gz)
- [core 비용 원trial 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0290.json.gz) · [K20 비용 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0505.json.gz)
- [T46 영수증](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0698.json.gz) · [선택 잠금](SELECTION_LOCK.json) · [E0 B55 개발검산](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0780.json.gz)
- [보호 경계 위반·복구](publication_derivatives/final_md_20261016_v2/docs/doc_0019.md)
- [실제52쌍 보조표](publication_derivatives/gzip_extra_aux_20261016_v2/files/item_0012.json.gz) · [사후 기술 해석](publication_derivatives/final_md_20261016_v2/docs/doc_0065.md)

![V에서 같은 G를 고정한 정규화 MSE](diagnostics/deep_g_matched_figures_1792096000000/render_v2_1792096170672/same_deep_G_MSE.svg)

![잠금된 T의 main4 고정9역할 정규화 MSE](diagnostics/postfinal_T_comparison_figures_20261016_v1/snapshot_1792103119089196800/locked_T_MSE.svg)



실제 점수451행=V405+T46, 표준 배열499=V405+T46+M48, 공개 충분통계451와 legacy B55를 errors0로 검산했다. 선택 replay50(41fresh+9reuse)은 고유 replay70과 겹친다. core72는60실측/12실패, K전환20scope/3600trial은 완료다. fresh deep-G Hpast8은 미측정이고 Beijing 심화 LoRA 비용은 예산 admission을 통과하지 않아 숫자가 없다. GPU 누적43030.3865/43200초다. 실제 v3 그림은 V405/71stem·T46/52stem이며34/24CSV를 v2와 동일하게 유지했다. 원 T matched PENDING과 별도52쌍 사후 분해(원14와 독립합산0)를 구분한다. 모든 그림의 가독성 완료를 주장하지 않으며 [학습 보완9stem/18이미지](publication_derivatives/gzip_extra_aux_20261016_v2/files/item_0000.json.gz)와 최종 그림 색인은 실제 생성됐다. Beijing V 범례 분리를 실제 확인했고 전그림 시각검토 완료로 확대하지 않는다. 비용74행 전사 대사는 PASS, [aux2995 원바이트 전수·독립·반복 복원](publication_derivatives/gzip_extra_aux_20261016_v1/BUILD_RESULT.json)과 [추가77 복원](publication_derivatives/gzip_extra_aux_20261016_v2/BUILD_RESULT.json)도 PASS다. 두 묶음3072원파일36,797,468B→물리9,893,189B이고 최종 전체 게시 검증과 구분한다.

등록471행의 nonCOMPLETE192, neural56의 GPU3epoch 완료8/CPU fresh별도9, 전체 K32/40, E0 full28중 endpoint미완료19는 그대로 남는다. audit PASS·T46·K20 완료가 필수 학습을 채우지 않는다. 한 seed·한 T cutoff·미수렴/CP0와 지표 상충 때문에 보편 효과나 독립 일반화로 확대하지 않는다. 재개 참조439개는 미완료192와 이미 완료된 reference247의 합이며 미완료 수가 아니다.

실행 부분 완료 / 필수 학습 미완료 / 사용 가능한 수치 검산 PASS / 일반적 기여 미확정 / 원격 게시 여부는 이 문서를 포함한 Git 커밋으로 확인한다. 원 데이터·NPZ·weights·개인 경로는 공개하지 않고 집계 행·정밀도는 무손실 복원한다. 이 문서는 부분 공개 범위를 설명한다. 이 보고서 조립시점(게시전검사)에는 이번 산출물의 추가 commit/push가 미실행이었다. 최종 그림·상대 링크·원형 복원·용량·원격 main 확인은 게시 영수증을 따른다. 검증 전 remote SHA를 적거나 게시 완료로 표시하지 않는다.
