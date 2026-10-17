# 원 multi-K 부모 재개 보고·시각화 입력 준비

현재는 PREP_ONLY다. `build_multik_resume_report.py`는 기본 호출에서 입력/출력/모델/자식 작업0으로 반환한다. 실제 N7 학습·추론·통계·비용이나 현재 프로세스를 조회하지 않았고, 실제 감사/보고/그림을 생성하지 않았다. 실행 중인 두 부모와 기존 파일을 변경하거나 종료하지 않는다. 작은 stdlib 가짜 메타데이터 테스트만 이 준비의 검증 대상이다.

## 실제 보고의 선행 조건

ROOT가 현재 auditor/builder SHA와 실제 새 감사 PASS SHA를 명시해야 한다. 감사 schema는 `INDEPENDENT_MULTIK_RESUME_AUDIT_v1`이며 parents2·rows6·accuracy6·cost_conditions12, source48/입력 보존과 전체 MASTER 미완료를 요구한다. 부분 종료나 실제 증거 누락이면 완성 보고를 거부한다. 실패/부분 완료는 ROOT가 성공·미실행·저장점·재개 조건을 별도 정직한 문서로 남긴다.

```text
build_multik_resume_report.py --execute --root-triggered
  --public-root <R> --cache-root <C>
  --audit-path <actual new N7 audit JSON> --audit-sha <actual SHA>
  --auditor-sha <reviewed current auditor SHA>
  --builder-sha <reviewed current builder SHA>
```

실제 출력은 `MULTIK_RESUME_RESULTS_v1.csv`, `MULTIK_RESUME_KO_v1.md`, `MULTIK_RESUME_BINDINGS_v1.json` 세 개다. 모두 CreateNew이며 기존 보고를 덮지 않는다. 이 준비에는 실제 audit/output SHA가 없다.

## 비교와 해석 경계

신규 표는 원 shared grid3/6/11·max-only grid11 두 부모의 K3/6/11 여섯 출력이다. 부모 refit2와 alias6을 구분한다. 원 BASE의 full_V_completed=false를 고치지 않고 실제 alias 감사로 판단한다. 원 search1868update/29880exposure는 재사용 이력이고 신규 search update/F call0, 원 refitCP0→3epoch는2262update/36189exposure다. 원 시간/누적F 기록을 별도 열로 보존하고 search 재구축 호출 시간을 과거 search 전체 wall로 바꾸지 않는다. 미저장 손실 진행량 UNKNOWN_NOT_LOGGED와 불완전한 중단 없는 bitwise 동일성 검증을 명시한다.

원정밀도 CSV는 신규6행과 재사용 참고4행을 `row_scope/accuracy_reused/cost_reused/reference_source`로 구분한다. 기존 nested PCA3와 N9 independent K6는 pinned CPU2/V43/20539 라벨 CSV·감사·bindings를 읽어 재사용한다. 이 builder가 참고 통계/모델/비용을 재실행하거나 새 감사한 것으로 세지 않는다. nested PCA의 basis-transpose D와 학습 lag7 D·현재값 anchor의 차이, shared/max PCA K11와 independent K11 identity 초기화 차이, 독립 K3/K11 및 Electricity·동등 예산·전체 E4 잔여를 보존한다. 같은 반복 개발 V와 한 seed의 검사 통과를 수렴·독립 일반화로 확대하지 않는다.

비용은 실제 B1/B8 한 batch 전체 ordinary timed20 median/p90이며 warm10·별도 intrusive stage20·모델/F pool cold load와 구분한다. raw N7 receipt에 없는 total F counter/proof SHA map/output dtype는 빈 값+NOT_RECORDED다. 공개50trial에서 독립 합산한 counter와 actual argv proof는 별도 근거다. Route memory dict/관측 guard RSS/F-only resident NA도 섞지 않는다.

## 시각화 준비 범위

보고 CSV와 bindings를 향후 그림의 단일 입력으로 사용한다. 준비할 경우 K3/6/11별 shared·max-only 두 선과 고정 PCA 참고를 같은 축에서 비교하고 independent K6는 그 한 점만 표시한다. 후보 패널은 MSE·MAE·B1/B8 whole median이다. 새 결과에 없는 K/seed·CI·p90 band·전환 비용을 그리지 않는다. K가 부모 모수/메모리를 같은 비율로 줄였다는 그림도 만들지 않는다.

이번 소유 범위는 보고 builder·작은 테스트·이 준비 문서까지이며 renderer 구현/실제 figure 생성은0이다. ROOT가 실제 감사/보고를 먼저 검토한 뒤 필요한 시각자료만 별도 준비할 수 있다. 그림 준비나 실패를 다른 독립 과학 경로의 중단 gate로 쓰지 않는다.
