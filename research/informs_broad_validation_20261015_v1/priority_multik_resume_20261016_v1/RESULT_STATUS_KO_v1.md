# Beijing multi-K: 현재 판단과 남은 검증

[확인] 이번 반복 개발 V에서는 **여러 K를 함께 학습한 부모가 최대 K만 학습한 뒤 잘라 쓰는 대조보다 K3/6/11 모두 MSE가 낮았다.** 고정 PCA보다 일관되게 좋아진 결과는 아니다. 원 shared/max 부모2개를 마지막 정상 refit CP0에서 재개했고 여섯 prefix 출력·전체 V43·별도 FP64·fresh CPU6·실제 CPU2 B1/B8 비용12조건·독립 검산·그림을 완료했다. 여섯 출력을 여섯 독립 학습으로 세지 않는다. 전체 MASTER와 충분 학습·독립 일반화는 미완료다.

![K별 오차와 실제 warm 전체 비용](figures/MULTIK_RESUME_v1.svg)

[원수치·학습 이력·제약](MULTIK_RESUME_KO_v1.md) · [원정밀도 CSV](MULTIK_RESUME_RESULTS_v1.csv) · [보고 결속](MULTIK_RESUME_BINDINGS_v1.json) · [독립 감사](INDEPENDENT_MULTIK_RESUME_AUDIT_ROOT_v1.json) · [그림 근거](MULTIK_RESUME_VISUALIZATION_KO_v1.md)

## 무엇이 확인됐는가

같은 Beijing C11/L512/H48·seed92601·반복 개발 V43/20,539 관측 라벨·동결 Bolt·학습 가능한 lag7 디코더/현재값 anchor 조건이다. shared는 K3/6/11을 학습했고 max-only는 K11만 학습했다. 최종 부모별로 실제 K개 잠재 시계열을 Bolt에 넣었다. 원 parent BASE의 full_V_completed=false는 그대로 두고 별도 prefix 예측과 검산 영수증으로 완료를 판단했다.

```text
K   shared MSE          max-only/prefix MSE  재사용 fixed PCA MSE
3   0.8758627372446505  0.9992618739139410    0.7979250373047883
6   0.7759337461842761  0.8933939212706146    0.7794565960483851
11  0.7374726715181666  0.7846963840141720    0.7517899596417101
```

여러 압축 수준을 직접 감독하는 것이 최대 K 학습 후 단순 prefix 절단보다 도움이 될 수 있다는 조건부 근거다. K6 shared와 고정 PCA의 차이는 0.0035228499이고 한 seed·반복 V에서의 작은 차이다. PCA는 basis transpose 디코더이고 learned 방법은 lag7 디코더/현재값 anchor이므로 이 표만으로 인코더 학습의 순수 이득을 분리할 수 없다. K3에서는 PCA가 더 낮은 MSE였다.

## 정확도와 비용에서 무엇을 주장할 수 있는가

Shared K3/K6/K11의 B8 warm whole median은 각각0.1599116998258978/0.3502740499097854/0.5715973498299718초다. 이는8개 입력 한 batch의 전체 시간이고8로 나눈 값이 아니다. 작은 K는 여기서 비용과 오차의 교환을 보였다. 최대 K11의 E/D 구조를 유지하므로 resident memory나 저장량이 K에 비례해 줄었다는 주장은 하지 않는다. cold Bolt/Route load, K 전환 비용·여러 독립 모델의 총 저장량·동등 총 학습 compute 비교는 미완료다.

비용은 다른 캠페인 모델 학습/추론과 겹치지 않은 단일 Route의 CPU2 측정이다. OS 전체의 CPU 격리나 affinity를 검증한 것은 아니다. ROOT의 소스 검토와 짧은 stdlib/PowerShell 합성 검사가 후처리 시간대에 있었으며, 작은 방법 간 시간 차이를 통계적으로 확정하지 않는다. warm10/ordinary timed20/별도 stage20·p90와 실제 원 trial은 원 보고/CSV에 보존한다. 기록되지 않은 F-only bytes, 원 cost receipt의 raw F 합계/proof map/output dtype은 NA로 남겼다.

## B2와 Bolt 유지 근거는 해결됐는가

이번 E4 부모는 B0 계열의 부차 비교다. 이번 결과로 주 후보 B2를 가변 K 방법으로 교체하지 않는다. 모든 표의 경로에 Bolt가 있어 Bolt를 직접 head·G·Ridge·LoRA보다 유지해야 한다는 새 근거가 되지는 않는다. 그 질문은 [Jena/PeMS 우선 검증](../priority_core_followup_20261016_v1/README_KO.md), [Jena 방법군](../priority_valid_family_20261016_v1/RESULT_STATUS_KO_v1.md), [Beijing B0/B2 대조](../priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md)의 비교 조건과 한계를 함께 따른다.

부모 둘 다 마지막 search3 epoch가 선택돼 UNDERTRAINED_OR_BUDGET_LIMIT이다. 같은 refit2262 update/36189 exposure라도 shared의 K별 노출과 F 연산 배분은 max-only와 다르다. 과거 search 재사용·이번 refit·알 수 없는 unsaved 손실을 구분했고, 동일 총 예산·수렴·초기화 효과를 통제했다고 말하지 않는다. 검사 PASS는 저장 예측과 재추론/원 trial 정합성의 확인이며 독립 일반화 검증이 아니다.

## 다음 유효 작업

원 예산과 마감 안에서 N5의 Jena AdaPTS 원리 선형 port와 Beijing B4 대조를 다음 명시 큐로 검토한다. 독립 K3/K11, 추가 자료/seed, 충분 학습, PeMS 부분채널 대조, Jena full-C 등 남은 축은 실제 실행 전이며 완료 수에 포함하지 않는다. M5 자산 부재/기존 공식 접근 실패는 그 경로의 제약이다. 기존 대형 큐와 보호 T 평가를 재개하지 않는다.

이번 science 완료와 main 게시 완료는 구분한다. 이 문서의 containing commit 및 별도 실제 원격 SHA/HTTP 영수증으로 게시를 확인한다. 과거 그림8개는 [원 바이트 보존 안내](../publication_derivatives/git_history_svg_20261017_v1/README_KO.md)를 따른다.

## CSV 파생 범위 열 정정

[확인] CSV의 B1/B8_derived_trial_F_scope 두 열은 exporter의 키 이름 불일치로 공란이다. builder가 derived_trial_scope 키를 쓰지만 헤더는 derived_trial_F_scope를 기대했다. 실제 파생 F counter 원수치·정확도·비용·그림은 이 문제의 영향을 받지 않았고, 합산 범위는 독립 감사의 derived_trial_F_counters.scope = INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS에 기록돼 있다. 이 공란을 원 cost receipt에 counter가 기록됐다는 뜻으로 해석하지 않는다. 이미 결속된 원 CSV/MD/bindings와 builder는 보존하고 이 정정으로 공개한다.
