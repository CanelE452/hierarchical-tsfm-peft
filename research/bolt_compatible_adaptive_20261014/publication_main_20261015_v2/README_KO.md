# 교수 미팅용 방법론·추가 실험 결과 색인

2026-10-15 추가 작업의 코드·보고서·원수치·시각자료를 연결한 게시 묶음이다. **방법론 후보와 검증 자료는 정리했지만, 작은 K가 Raw 정확도와 비용을 함께 만족하는 성공 방법은 아직 얻지 못했다.** 반복 노출된 개발 자료의 탐색 결과이며 독립 일반화·학술적 신규성을 확정하지 않는다.

가장 먼저 [교수 미팅용 방법론 메모](reports/professor_method_brief_v1/REPORT_KO.md)를 읽으면 된다. 연구 목적, TSFM 배경, 배포식, trainable/frozen, FIT 학습·TUNE 선택·DEV 채점과 과거 전용 배포, 대조군의 역할, 원수치·실측 비용, 실패 해석과 기여 한계를 한 문서에 모았다.

![고정 encoder·Bolt·기존 decoder와 학습되는 잔차 보정](reports/professor_method_brief_v1/figures_v3/method_spec.png)

## 지금 답할 수 있는 것

F의 미래 예측은 지정된 같은 공간·보정 class의 FLAT보다 12/12 대조에서 TUNE/DEV MSE가 낮았다. 그러나 PCA의 시간 ramp 보정은 네 조건 모두 자기 무보정 F보다 DEV가 높았고, 작은 K 보정8조건 중 원 Raw보다 DEV가 낮은 조건은 없었다. Beijing Raw 보정의 이득은 압축 고유 이득과 구분한다.

새 FIT 내부 시간순 진단은 A/B에서 각각11창을 purge해117/53/64 origin을 사용했다. PCA K8은 B 0.408280→0.406539이나 C 0.100626→0.103507, Raw도 B 0.328818→0.321778이나 C 0.074479→0.075802로 보정 이득이 이전되지 않았다. 기존 scaler/E/D가 full FIT에서 정해져 있고 C label·고정 특징도 이미 가용한 조건부 진단이다. 독립 holdout이나 전체 pipeline의 시간순 검증으로 부르지 않는다.

원/반복 전체 비용을 실제 측정했다. B8에서 Robin Raw 대 PCA K8은 원26.6482/18.2678, 반복26.0451/17.7897 ms/batch였다. B1의 일관된 절감은 없었다. K/C를 latency 절감률로 대체하지 않는다. 시간순 진단에서 바뀐 계수의 비용은 미측정이며 이 비용을 새 정확도에 붙이지 않는다.

## 상세 보고서와 원수치

- [교수 미팅용 메모·수식·도식·표14행 수치 연결](reports/professor_method_brief_v1/REPORT_KO.md).
- [시간순 보정 이전: 2fit·22후보·44 A/B·선정2 C,6CSV·3개 그림](reports/purged_readout_stability_v1/REPORT_KO.md).
- [STATIC/RAMP 보정: 24fit·264후보·528 FIT/TUNE·24 DEV, 원/반복 각각72비용셀·360표본,12CSV·4개 그림](reports/horizon_decoder_diagnostic_v1/REPORT_KO.md).
- [보정 방향과 크기의 오차항:72전체·3456h·216구간·52416 per-C,7CSV·4개 그림](reports/horizon_error_terms_v1/REPORT_KO.md).
- [예측 cross-moment·K 확대·동일 E/D native16fit·직접 ridge12fit·전체76비용셀,12CSV·4개 그림](reports/new_direction_comparison_v1/REPORT_KO.md).
- [ridge 규제 확장10경로·100새 FIT/TUNE 평가,2CSV·2개 그림](reports/ridge_regularization_extension_v1/REPORT_KO.md).
- [고정 Q·분위 요약·Q 학습6fit/3600update,6CSV·4개 그림](reports/methodology_diagnostics_v1/REPORT_KO.md).
- [고정 L7 잠재값 보정과 전체20비용셀,14CSV·4개 그림](reports/fixed_l7_latent_calibration_v1/REPORT_KO.md).

이8개 보고 폴더에는59CSV·PNG28개·SVG28개가 있다. 도식의 이전 두 버전도 보존했으므로28개 모두가 새로운 성능 결과 그림은 아니다. [이전 게시 묶음](../publication_main_20261015_v1/README_KO.md)의 실험·실패·수치 공간 진단도 유지한다.

## 검산·실패·마지막 조사

과학 검산은 선정 해·FP32 출력식·관측 mask·점수·선정 잠금·source와 자원 영수증을 확인한 범위다. F GPU·optimizer·타이머 전체를 별도로 재생하거나 새로운 holdout을 검증한 것으로 확대하지 않는다. 상세 보고서의 PASS 링크에서 범위를 확인할 수 있다.

비용감사v1의 이름 충돌 실패2개와 수정v2 PASS를 함께 보존했다. 실패 기록을 지우거나 수치를 바꿔 PASS로 만들지 않았다. 원문 작성 당시 ‘로컬’, ‘미실행’, ‘검산 중’ 표현은 해당 시점 기록이며 최신 완료 상태는 이 색인과 [마무리 상태](notes/METHOD_COMPLETION_STATUS_KO_v1.md)를 따른다.

- [학습 조건 조사](notes/LEARNING_REGIME_REVIEW_KO_v1.md): 실제 예측 목적 E/D 학습16fit·9600update 등과 데이터 범위를 확인했다. Robin 중첩 문제를 Beijing에도 그대로 적용할 수 없다.
- [TRAIN 자산 조사](notes/TRAIN_ASSET_REVIEW_KO_v1.md): 파일 존재·계약·가드만 확인했다. 실제 새 자료 역직렬화·표현 학습·추론은 하지 않았다.
- [조건부 다음 설계 검토](notes/NEXT_PHASE_LEARNING_DESIGN_REVIEW_KO_v1.md): 미실행 제안이다. 사용자의 마무리 요청 뒤 새 학습을 시작하지 않았다.

현재 continuation에서 보호 TEST를 새로 사용하지 않았다. 별도 과거 Electricity 조사 접근 incident는 원 보고와 교수 메모에서 구분해 공개하며, 전체 역사에 보호 접근이 없었다고 말하지 않는다.

## 코드·게시 검증·재현

- [실행 코드와 로컬 재현 자산](CODE_AND_REPRODUCTION_KO.md).
- [최종 원본/게시 사본 SHA와 링크 변환 기록](REPORT_COPY_MAP_v2.json), [첫 게시 초안 변환 기록](REPORT_COPY_MAP.json), [큰 JSON의 무손실 압축 계보](EVIDENCE_ARCHIVES.json).
- [판단에 직접 인용한 과거 텍스트·코드 8파일과 미게시 역사 의존성 안내](references/README_KO.md).
- [최종 게시 파일 manifest](FILES_v2.json), [최종 Git index·링크·이미지·형식·민감 패턴 검사](https://github.com/CanelE452/hierarchical-tsfm-peft/blob/main/research/bolt_compatible_adaptive_20261014/publication_main_20261015_v2/FINAL_INDEX_CHECKS.json).
- [초기 묶음 검사 PASS](LINK_AUDIT_v2.json), [초기 manifest](FILES.json), [자기 영수증 생성 전 링크1개 실패 기록](LINK_AUDIT.json). 실패를 보존하고 실제 영수증을 만든 뒤 다시 검사했다. 최종 검사는 README의 최종 연결과 최종 manifest를 다시 확인한다.

과학 원본은 유지하고 게시 사본의 상대 링크만 정리했다. CSV·PNG·SVG는 byte 동일 사본이다. 큰 결과 JSON의 `.gz` 링크는 다운로드 가능한 무손실 사본이며 표의 핵심 원수치는 Markdown과 CSV에서도 읽을 수 있다. 원데이터·NPZ·모델·가중치·checkpoint는 게시하지 않았다.
