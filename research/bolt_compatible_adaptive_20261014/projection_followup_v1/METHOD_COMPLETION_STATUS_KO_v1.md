# 교수 설명 자료 완료·Raw 대체 정확도는 미해결

2026-10-15 관측 상태다. [교수 미팅용 메모](reports/professor_method_brief_v1/REPORT_KO.md)에 목적·배포 수식·trainable/frozen·학습/선정/배포 입력·공정한 대조의 범위·원수치·실측 비용·실패 해석·선행과 기여 한계를 정리했다. 최신 도식은 [v3 PNG](reports/professor_method_brief_v1/figures_v3/method_spec.png)이며 예전 도식은 보존했다.

**완료한 것은 구현·비교·검산·설명 자료다. 작은 K가 Raw 정확도와 비용을 함께 만족하는 성공 알고리즘이나 독립 일반화·학술적 신규성의 확정은 아니다.** 새 결과를 좋게 만들려고 DEV로 새 winner를 선택하거나 성공 허용치를 사후 도입하지 않았다.

## 이번 추가 작업의 실제 상태

- [같은 E/D의 native·직접 ridge·전체 비용](reports/new_direction_comparison_v1/REPORT_KO.md): 실제 학습·추론·비용·과학/비용 검산·보고 완료.
- [ridge 규제 확장](reports/ridge_regularization_extension_v1/REPORT_KO.md): 실제10경로 확장·검산·보고 완료. 바뀐 계수의 새 배포 비용은 미측정이다.
- [STATIC/RAMP 잔차 보정](reports/horizon_decoder_diagnostic_v1/REPORT_KO.md): 24fit·264후보·528 FIT/TUNE·24선정 DEV와 별도 원/반복 각각72비용셀·360표본, 과학 검산과 수정된 비용감사v2 모두 PASS, 12CSV·4개 PNG/SVG 보고 완료.
- [선정 보정의 오차항](reports/horizon_error_terms_v1/REPORT_KO.md): 72전체·3456h·216구간·52416per-C 진단,7CSV·4개 PNG/SVG 보고 완료. 별도 외부 인과 실험은 아니다.
- [FIT 내부 시간순 이전](reports/purged_readout_stability_v1/REPORT_KO.md): 실제2fit·22후보·44 A/B·선정2 C, A/B 각각11창 purge 후117/53/64, 선정2fit 독립 수치 재풀이·저장 출력 검산 PASS,6CSV·3개 PNG/SVG 보고 완료. full-FIT upstream을 고정한 조건부 진단이며 독립 holdout이 아니다.
- [교수 메모 파일 검산](audits/PROFESSOR_METHOD_BRIEF_FILE_CHECK_v1.json) PASS. [수치 연결](reports/professor_method_brief_v1/NUMERIC_BINDINGS.json)은14행·표시38수치가 기존 JSON metadata와 일치함을 확인했다. 파일·표 검사는 과학 실험의 독립 재현을 대신하지 않는다.

Q·분위 요약·예측 cross-moment·기존 E/D 관련 내역은 교수 메모에서 각각의 실제 결과로 연결했다. 이 상태 문서는 예전 다른 캠페인의 모든 미완료 실험까지 완료했다고 뜻하지 않는다.

## 현재 내릴 판단

F의 미래가 지정된 FLAT/native보다 유리한 근거는 있다. B8에서는 작은 K의 전체 경로 비용 절감도 관측했다. 그러나 작은 K의 Raw 대비 정확도 격차는 남았고 B1 비용 이득은 일관되지 않는다. 예측 목적을 넣거나 D의 시간별 크기를 바꾸는 것만으로 문제를 해결하지 못했다.

새 시간순 진단에서도 PCA K8과 Raw 모두 B 이득이 C 악화로 바뀌었다. 이는 보정의 시간 이전 실패에 대한 조건부 반례다. 이 선형 fit의 반복 최적화 미완료를 가정할 필요는 없지만, 기존 신경망 학습의 모든 최적화 문제·공변량 변화·표현 한계 중 고유한 원인을 확정한 것은 아니다.

다음 강한 성능 주장에는 upstream을 포함한 적절한 시간순 적합·평가, 별도 검증 범위와 배포 채택 기준이 필요하다. 같은 exposed 자료에서 spline·동적 그래프·대조학습 등 복잡성을 자동으로 늘려 성공을 찾는 결론으로 넘어가지 않는다. 현재 연구 목표 자체를 달성 완료로 표시하지 않는다.

## 보존·자원·게시 상태

마지막 확인 HEAD는 `6f7e312a3b1f6eecb15b134239f2893816864aa4`다. 그 이후 이번 자료는 **로컬에만 있다**. 자동 commit/push, 보호 TEST 새 접근, 설치/다운로드, 큰 백본 학습, 기존 사용자 작업 종료·중단 실험 재개는 하지 않았다. 전체 역사에는 별도 보호 접근 incident가 있으며 교수 메모에서 현재 continuation의 범위와 구분했다.

[자원 snapshot v12](audits/RESOURCE_SNAPSHOT_20261015_v12.json)는 현재 continuation178개 terminal receipt, CPU process별 최대값 합3311.390625초, GPU process meter wall 합1879.8484532991424초, 실행 중 receipt0을 확인했다. 이전170개 receipt hash와 원18개 보호 source는 불변이다. source 준비·일부 테스트 등 비계측 작업과 예전 전체 캠페인을 포함한 총량은 아니다. GPU process wall은 GPU kernel 시간이 아니며 자원 합계를 추론 latency로 쓰지 않는다. 종전 시간 상한은 재도입하지 않았다.

원/반복 비용감사v1의 이름 충돌 실패와 수정v2 PASS, 교수 메모 표 검산의 행 접두어 매칭 실패 receipt를 보존했다. 검산기 수정으로 과학 출력·비용 표본·계수·공차를 바꾸지 않았다.
