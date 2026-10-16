# Jena CPU 비교: 3epoch 검색 완료 기록

**네 조건 모두 검색3epoch를 실제 완료했다. 최종 V 비교·검산·비용 완료 보고서는 아니다.** 각 조건의 완료 checkpoint에는 누적1863 update·29769 origin 노출이 있다. A_stop 규칙으로 B0/과거 지연은3epoch, B2/양방향 지연은1epoch를 선택했고, 그 epoch 수로 A60에서 새로 적합을 시작했다. 이후 B2/양방향 지연의 refit752 update·12020 노출과 전체 V43 예측 완료를 확인했다. 저장 점수의 독립 재채점·checkpoint 재추론은 대기 중이고, B0/과거 지연은 아직 적합 중이다. 선택1epoch와 검색3epoch 미완료를 구분한다. 기존 [1epoch snapshot](PROGRESS_E1_KO.md)은 당시 기록 그대로 보존한다.

## 확인하는 질문과 비교 조건

현재 시점의 채널 투영B0, 창 전체 시간 혼합B2, 과거 방향7지연, 창 내부 양방향7지연을 Jena의 L512/H48/C21/K6/seed92601에서 비교한다. 디코더·현재값 anchor·frozen Bolt-small·원 미래의 정규화 channel-macro MSE·A50/A_stop/A60 규칙과 검색 노출 일정을 맞췄다. 네 worker 모두 CPU2 thread로 실행했다. 원 계약은 [README](README_KO.md), [PLAN](PLAN.json), [등록](REGISTRY.csv)에 있다.

검색 후보와 선택 규칙은 같지만 선택된 refit 길이는1epoch 또는3epoch로 다르다. 인코더 용량·tap 위치·경계조건도 다르므로 같은 파라미터 수나 동일 refit 연산량의 비교는 아니다. 양방향 인코더 역시 origin 이전의 같은 입력 창만 사용하며, 이름에 있는 양방향을 미래 정답 사용으로 해석하지 않는다.

## 원수치와 선택 결과

```text
Encoder       A_stop@0     A_stop@0.25  A_stop@1     A_stop@3     Selected
B0            0.609146790  0.442987429   0.421947032  0.411867903  3
B2            0.609146790  0.447932764   0.402603278  0.427643366  1
LAG_CAUSAL    0.609146790  0.418885567   0.406966662  0.405919461  3
LAG_BIDIR     0.609146790  0.393911805   0.384033462  0.403813454  1

Epoch         0            0.25          1             3
Updates       0            156           622           1863
Exposures     0            2481          9923          29769
```

![완료된 검색 곡선과 A_stop 선택 endpoint](diagnostics/search_milestone_e3_root_20261016_v2/SEARCH_MILESTONE_E3.svg)

- [반올림하지 않은16행 CSV](diagnostics/search_milestone_e3_root_20261016_v2/SEARCH_CURVES_THROUGH_E3.csv).
- [event·private checkpoint SHA와 source/config/data/scaler 계약](diagnostics/search_milestone_e3_root_20261016_v2/EVIDENCE.json).
- [그림·CSV·렌더러 결합](diagnostics/search_milestone_e3_root_20261016_v2/FIGURE_BINDING.json), [집계 CSV만으로 다시 그리는 코드](render_search_milestone.py).

완료된 search checkpoint4개의 SHA를 전후 확인한 뒤 CPU에서 읽어 곡선과 원 strict-less-than 선택 규칙을 검산했다. 이 캡처에는 foundation/model 생성·forward·학습·Task/정답 배열 로딩이 없다. 그림의 PNG preview를 실제 열어 확인했고, 밀접한 초기 두 tick의 겹침을 고친 별도v2 그림을 사용했다. 먼저 만든v1 산출물은 변경하지 않았으며 게시 대상에서 제외한다.

## 현재 결과가 말하는 것과 남는 판단

B0와 과거 지연은3epoch에서도 A_stop이 낮아졌고, B2와 양방향 지연은1epoch보다3epoch가 높았다. 따라서 마지막 epoch만 비교하거나 짧은0.25epoch의 순위를 최종 결론으로 쓰지 않는다. 이 곡선만으로 악화의 원인을 과적합·최적화 실패·표현의 구조적 한계 중 하나로 확정할 수 없다. 특히 B0/과거 지연의 선택점이 탐색 끝이라는 점은 충분 수렴의 근거가 아니다.

선택된 A_stop은 checkpoint 선택 자료의 결과다. native ridge·Raw·직접 G의 V 점수와 섞어 순위를 매기지 않는다. 다음 판단에는 fresh A60 모델의 전체 V43, 독립 FP64 재채점, 고정6 origin의 checkpoint 재추론이 필요하다. 비용도 학습과 다른 측정을 겹치지 않은 CPU2·batch1/8·warmup10/timed20/stage20 프로토콜로 확인한다. 원 GPU 지연시간과 새 CPU 지연시간을 직접 합쳐 속도 향상으로 제시하지 않는다.

기준선 Raw/ridge/G도 기존 출력과 artifact를 보존한 별도 CPU 재추론 경로를 준비했다. 원 replay cache는 실행 장치를 reference에 기록하지 않으므로 기존 cache hit를 새 CPU forward의 증거로 쓰지 않는다. 이 기록 작성시 실제 CPU 수치 parity와 새 기준선 비용 측정은 아직 실행하지 않았다. 성공하더라도6 origin의 수치 검증이며 전체 V 재실행·독립 일반화와 다르다.

## 캠페인과 게시 상태

이는 T 관측 이후의 별도 post-T 탐색적 V 전용 development다. 기존 승인 safeFIT 전체 역직렬화와 legacy 정규화의 노출 이력을 유지한다. 새 T 정답 구성·적합·선택·채점은 요청하지 않으며, 전체 thread에서 T bytes 접근이0이었다고 주장하지 않는다. 한 seed·반복 사용한 개발 자료이므로 독립 일반화나 불확실성 구간을 제시하지 않는다.

원 source39·registry·선택 잠금·SCORES/COST/T 결과를 보존했다. 이 네 CPU 검색으로 원471행의 미완료192나 GPU3epoch 미달48을 완료 처리하지 않는다. [고정 PCA6조건](../postT_V_only_cpu_fixed_pca_20261016_v1/README_KO.md), [다중 K 비교](../postT_V_only_cpu_multik_20261016_v1/README_PREPARATION_KO.md) 및 [원 RESUME](../RESUME.md)의 범위를 구분한다. 고정 PCA의 반복 평가를 신경망 epoch 학습으로 세지 않는다.

이 문서와 그림의 존재만으로 원격 게시 완료를 주장하지 않는다. 실제 push 이후 main SHA·원격 경로·그림/링크 검증을 별도로 기록한다.
