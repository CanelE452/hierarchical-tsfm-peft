# PeMS C32 B0/B2 최종 상태: 예산 종료, 새 성능 비교는 미완료

이번 두 fit은 search 도중 시간 상한에 도달했다. B0는847 update·13541 exposure·약1.611epoch, B2는846·13525·약1.609epoch다. 완료 endpoint는0/.25/1이며 event_000003의 목표3을3epoch 완료로 읽으면 안 된다.

최종 bundle·fresh refit·전체 V가 없어 후처리8개는 실제 미실행으로 보존됐다. 독립 감사는 새 accuracy0행·cost0행의 NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED이며 exit2다. 원 source48·입력SHA·종료 증거를 보존했다는 사실을 성능 감사 PASS로 바꾸지 않는다. 별도 diagnostics에는 partial search 중 F 입력의 유한·비영 기울기 callback B0 3386회/B2 3382회가 기록됐다. 이 구현 관측과 최종 정확도 검증은 구분한다. 두 원 checkpoint_000003.pt는 로컬에 보존됐고 실제 payload 복원·C32 재개 CLI는 미검증/미구현이다.

- [다섯 질문의 최종 판단·기존 원수치·조건·한계·재개 자원](PRIORITY_FINAL_JUDGMENT_KO_20261017_v2.md)
- [실제 종료 진행량·저장점·미실행 metadata](B0_B2_TERMINAL_PROGRESS_20261017_v2.json)
- [원 독립 감사 JSON](INDEPENDENT_PEMS_B0_B2_AUDIT_ROOT_v1.json)
- [B0 원 종료 기록](diagnostics/pems_c32_b0_k8_fit_root_v1/RESULT.json) · [B2 원 종료 기록](diagnostics/pems_c32_b2_k8_fit_root_v1/RESULT.json)
- [이미 완료된 Raw/G의 MD·비용](RAW_G_KO_v1.md) · [기존 Raw/G 그림](figures/RAW_G_FOUR_PANEL_v1.svg)

기존 Jena에서 B2는 B0보다 낮은 MSE였지만 Beijing K6에서는 악화됐다. 일부 고정표현 직접 head 대조에는 이득이 있었으나 Ridge/G/LoRA를 넘는 보편적 정확도 우위는 미확정이다. 예측 MSE 학습은 이미 적용됐으므로 모든 미해결을 재구성 목적 탓으로 설명하지 않는다.

이 보고는 이번 허용 경로의 실제 실행·예산 중단·검산 시도·미완료를 정리한다. 전체 MASTER는 paused·미완료이며 모든 실험이나 방법론의 성능 검증을 완료했다고 주장하지 않는다. 작성 기준 이전 원격 main은5b582c9a이며 이 파일 자체가 새 push의 성공 영수증은 아니다.
