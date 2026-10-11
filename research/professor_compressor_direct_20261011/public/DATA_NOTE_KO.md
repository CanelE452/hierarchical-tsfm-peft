> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `2258aa198187aef630dceb6c77ce90394abb78d3627fb8d0ef77db39a4a695a0`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# 데이터 계약 메모

[확인] 이 데이터 레이어는 `runtime_v18.load_data(dataset, include_test=False)`만 호출한다. TEST archive는 열거나 해시하지 않았고, 새 실험의 개발 평가는 기존 TRAIN/VAL 구간에 한정된다.

[확인] 다운스트림 결정은 GTConvAE/TS2Vec 표현이 TSFM 미래 예측 뒤 원공간 복원에 쓸 만한지 보는 것이다. 그래서 예전 PCA/LEVEL의 K, D8, 잔차 G 조건은 데이터 계약에 넣지 않았다.

```text
dataset  T      C   train_obs  val_target_obs  rec_windows  train_F_origins  val_origins
robin    3312   17    100.00%        100.00% 12          12               30
jena     35135  21    100.00%         99.99% 201         201              371
```

[확인] 시계열 순서는 두 자료 모두 strictly increasing이고 중복 timestamp는 0개다. 분할은 시간 순서를 유지하며 TRAIN 뒤에 VAL이 온다.

[확인] 원자료 물리값은 v18의 mean/std로 역변환했고, 새 입력 `x`는 TRAIN observed 값으로 다시 계산한 mean/std를 사용해 표준화했다. 결측 입력은 미래를 보지 않는 causal forward-fill로 채웠고, 정답 metric에는 `mask`의 observed 위치만 쓰도록 보존했다.

[확인] graph는 TRAIN 구간에서 두 채널이 모두 관측된 시점만 사용한 절대 Pearson 상관이다. diagonal은 0이고, self term과 중복되지 않게 별도 처리하도록 남겼다. 물리 topology가 아니라 TRAIN-only 추정 graph다.

[확인] TS2Vec용 배열은 `[1, T_train, C]`이며 unobserved 위치는 NaN으로 되돌릴 수 있다. GTConvAE 재구성 창은 TRAIN 내부에서 길이 512가 완전히 들어가는 start를 stride 128로 만들었다.

[확인] 전체 검산 상태: PASS. 상세 hash와 count는 `DATA_CHECKS.json`에 있다.
