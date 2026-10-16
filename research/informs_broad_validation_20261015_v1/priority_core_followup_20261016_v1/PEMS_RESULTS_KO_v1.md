# PeMS08 고정 B2 표현의 전체 과거 직접 head 비교

**[확인] 정확한 기존 B2 표현·디코더를 고정한 K43 직접 head는 MSE 0.12062288970242162로 부모 B2+Bolt의 0.10425969886874974보다 높았다. 대신 동일 CPU 전체 추론 비용은 훨씬 낮았다.** 기존 Raw·LoRA와 3epoch G는 B2보다 낮은 MSE·MAE를 보이므로 전체 정확도 우위는 확보하지 못했다. Jena 추가 seed 두 조건은 이 snapshot 시점에 실행 중이며 완료로 세지 않는다.

[원수치](PEMS_RESULTS_v1.csv), [봉인 계획](PEMS_FIXED_HEAD_PLAN.json), [완료·자원 반환](PEMS_EXECUTION_CLOSURE_ROOT_v1.json), [기존 baseline 대사](PEMS_BASELINE_REUSE.json), [독립 검산](INDEPENDENT_PEMS_REVIEW.json)이 근거다. 본문 표는 반올림 표시이며 CSV가 원정밀도다.

같은 primary V148, 과거512/미래12, C170/K43, seed92601, A60 scaler94e8c71c…69c60, 301,920 관측 정답과 같은 target/mask/origin/channel을 사용했다. 실제 원자료 FP64 C-order 배열의 앞14,284행만 수치 변환해 feature0를 사용했고, 보호 T 이후 수치는 파싱하지 않았다. 원 full-data 지문은 상속 metadata로 별도 표기한다. 전체 압축 파일의 opaque 바이트 SHA/ZIP I/O를 수치 파싱0과 혼동하지 않는다.

```text
방법                                 정규화 MSE         정규화 MAE
B0                                   0.1053385337397075  0.2126966734626624
B2_PARENT                            0.1042596988687497  0.2058417942879392
RAW                                  0.0953549334407595  0.1963338292536118
RIDGE                                0.2289959431467351  0.2325320321186678
G_MATCHED_ONE_EPOCH_REFERENCE        0.1060504404528483  0.1962265583186836
LORA                                 0.08472487374298254  0.1817426053892815
G_CPU_FRESH_THREE_EPOCH_REFERENCE    0.09938846978422584  0.1890542792341157
FIXED_B2_NO_BOLT_HEAD                0.1206228897024216  0.2181754324941234
```

이 부모는 `primary_pems08_E1_B2_k43_s92601__e1`의 제한 학습 결과다. A50 검색0/.25/1epoch, 선택1, A60 fresh refit1epoch이며 3epoch 학습·수렴 완료가 아니다. B0의1epoch MSE 0.10533853373970749보다 B2가 약1.02% 낮지만, 다른 dataset/seed까지 입증한 결과가 아니다. G의1epoch와 CPU fresh3epoch를 별도 행으로 남기며 유리한 한 행만 고르지 않았다.

직접 대조는 부모의 최종 P/E/D와 scaler를 정확히 고정하고 **전체 Z[512,43]**를 채널 공유 Linear(512,12)에 입력한다. 같은 지연 D와 현재값 anchor를 유지하고 head6,156파라미터만 A60 고정3epoch, 1,911 update/30,570 origin 노출로 학습했다. 미래 MSE로 학습했으며 재구성 목표의 실패가 아니다. 기존 표현이 이미 본 A_stop을 미노출 head 선택 자료로 쓰지 않았고 M_cal 정답을 추가하지 않았다. 추가 최적화28.012초·fit+V wall32.996초/CPU60.125초이며 부모 표현을 얻은 학습 비용은 별도다.

```text
방법                                 CPU B1 median 초/배치    CPU B8 median 초/배치
B0                                   NOT_MEASURED             NOT_MEASURED
B2_PARENT                            0.28220830019563437      2.2950869500637054
RAW                                  NOT_MEASURED             NOT_MEASURED
RIDGE                                0.0003635501489043236    0.005099399946630001
G_MATCHED_ONE_EPOCH_REFERENCE        NOT_MEASURED             NOT_MEASURED
LORA                                 NOT_MEASURED             NOT_MEASURED
G_CPU_FRESH_THREE_EPOCH_REFERENCE    NOT_MEASURED             NOT_MEASURED
FIXED_B2_NO_BOLT_HEAD                0.0007654998917132616    0.006547349970787764
```

부모와 head는 같은 physical 입력·mask·원점의 CPU2, 10warm/20timed/별도20stage, 원 physical 전처리·실제 P/E·예측·D·FP64 역변환 전체 경로다. 부모는 K43 streams와 row128을 사용한다. 새 head 비용은 cached Z/forecast를 사용하지 않았다. 부모의 새 CPU6(4+2) 재추론은 원 GPU V 저장 예측과 공차1e-5/1e-4에서 일치(maxabs2.86102294921875e-6, 실제258streams/3calls)했고 이를 통과한 후 비용을 측정했다. 전체 CPU V148를 다시 추론한 것으로 확대하지 않는다. Ridge는 기존 bound CPU 기록을 재사용하며 구 영수증에 thread 필드가 없고 pinned 실행 source의2threads 설정을 근거로 한다. Raw/G/LoRA의 옛 GPU 비용을 CPU 열로 옮기지 않았다.

새 head의 full V148 저장 예측 FP64 재점수와 충분통계 재집계, 별도 CPU6 재추론을 수행했다. full V 검산과 subset 재추론은 서로 다른 범위다. 부모 full V 점수는 기존 exact 감사에서 재사용한다. 모든 PeMS 학습·재추론·비용 worker를 종료하고 자원 반환을 확인한 뒤 Jena seed92602 B0/B2 학습을 시작했다. 두 seed만으로 유의성 검정이나 독립 일반화를 주장하지 않는다.

이번 PeMS 비교는 Bolt용 표현과 고정 D 아래의 조건부 선형 예측기 진단이다. PeMS에서 같은 구조를 별도 학습한 R5b는 실행하지 않았고, 부모1epoch와 추가 head3epoch의 정보·학습량 차이를 숨기지 않는다. 따라서 Bolt의 보편적 필요성·압축 방식의 한계·충분 수렴을 확정하지 않는다. B2 후보와 기존 T 잠금은 그대로이며 새 그래프·G/H 결합·모든K 큐·PPT/PDF를 추가하지 않았다.
