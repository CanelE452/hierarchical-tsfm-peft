# 원 채널 일부와 혼합 투영의 Bolt 호환성 대조

상태: 다음 최소 대조의 사전 계획. 실제 새로운 subset 선정·F inference·학습·비용은 아직 0이다. 앞 shrinkage/matched-native 묶음의 실행과 독립 검산·보고를 먼저 마무리하고 이 대조를 별도 새 경로로 수행한다. 원 소스·결과·원 선정 잠금을 수정하거나 원 개발 점수로 K/방법/채널을 고르지 않는다.

## 목적과 반증

[확인] 정적 복원 공간의 한계가 있었고, 과거 조건부 D는 일부 오차를 줄였지만 추론 적합 비용을 더했다. 선정된 D에 미래 E(Y)를 주면 원 채널 Bolt보다 낮은 오차의 여지가 있었으나, 실제 Bolt·동일 D로 재학습한 affine native가 그 수준에 도달하지 못했다. 이는 decoder 공간만이 전부라는 설명을 약화하지만 과거 정보의 예측 가능성이나 모든 학습기의 한계를 증명하지 않는다.

[추정] 원 채널들을 섞은 잠재 시계열이 frozen Bolt에 어려운 입력일 수 있다. 단순한 원 채널 subset을 같은 작은 K로 넣는 비교가 현재의 혼합 신호와 예측기의 호환성을 확인하는 최소 대조다. 이 비교 자체로 혼합이 실패의 유일 원인이라고 확정하지 않는다. Subset도 정보 손실을 만들고, FIT의 과거 복원 적합성이 미래 예측 적합성을 보장하지 않는다.

목적 트리는 `작은 K에서 원 C 예측 정확도와 전체 비용을 유지할 수 있는가 → 입력 좌표 변화와 복원 제약을 나누는가 → 원 채널 subset과 혼합 공간을 같은 고정 FIT decoder 규칙으로 비교하는가`다. 큰 graph/contrastive 모델이나 데이터·주제를 추가하는 계획이 아니다. 조사한 기존 `_docs/history/2026-10-14.md` 및 현재 캠페인 Markdown에는 이 exact subset 대조를 찾지 못했지만, 저장소 전체에서 첫 연구라고 주장하지 않는다. 기존 latent-teacher ridge·actual-Y 학습의 존재를 보존한다.

## 원 조건과 선택

Robin C17/K4, Beijing Aotizhongxin C11/K2, L512/H48, 원 FIT256/TUNE64 및 기노출 Robin VAL30/Beijing EVAL146, 원 mean/std·origin·true observed mask·causal fill·frozen Bolt를 유지한다. 보호 TEST나 새로운 평가 데이터는 사용하지 않는다.

Subset은 자료마다 FIT의 저장된 **과거**만으로 한 번 고정한다. 채널의 원 index 순서에서 greedy하게 K개를 고른다. 각 후보의 입력은 이미 causal-fill한 정규화 과거이고, 재구성 target은 실제 관측된 과거 C 값만 사용한다. 각 C의 관측 subset에서 predictor/response를 중심화한 least squares로 재구성 정규화 MSE를 구한 뒤, 관측이 있는 C를 균등 평균한다. 그 값이 가장 작은 후보를 추가하며 exact tie는 작은 원 channel index를 사용한다. TUNE/DEV/미래 정답/Raw-F forecast 점수는 채널 선정에 넣지 않는다. 중복되는 FIT window를 독립 표본으로 부르지 않는다.

Least squares slope의 singular relative cutoff는 사전 1e-4다. Gram 행렬의 eigen cutoff를 쓸 경우 대응하는 1e-8을 명시하고 직접 centered lstsq와 독립 검산해야 한다. 관측 0/1 또는 centered variance0에 대한 slope·intercept fallback과 미학습 채널을 명시한다. 같은 greedy 규칙을 검증 가능한 합성 fixture로 확인하고, 실제 후보 점수·전 단계 선택·최종 index/채널명을 저장한다. Greedy가 최적 subset을 보장한다고 주장하지 않는다.

원 normalized raw channel의 값과 순서를 유지하기 위해 subset encoder는 고정 one-hot E32, center0이다. Whitening이나 horizon별 채널 변경을 하지 않는다. E는 시점별 과거 전용 선형 encoder이고 선택은 FIT에서만 고정되므로 각 평가 origin의 미래 정보와 무관하다.

## 동일 복원 규칙

세 E 공간은 기존 PCA, 기존 FIT-target, 새 subset이다. 기존 두 E는 source/state SHA로 재사용한다. 세 공간 모두 FIT **관측 과거**에서만 centered truncated least squares affine decoder를 같은 cutoff1e-4로 적합한다. Predictor는 실제 해당 E의 저장된/실행한 FP32 과거 latent이고, target은 C별 true-observed 과거다. FIT-target E 자체는 이미 FIT 실제 미래로 학습했으므로 세 E의 감독 정보가 모두 같다고 설명하지 않는다.

배포는 `currentC + (predictedZfuture - lastZ) @ D32`다. Affine bias는 innovation 식에서 상쇄된다. D는 FIT 고정이라 평가 origin의 C 과거로 새로운 계수를 적합·전송하지 않는다. 입력 K streams와 current-C anchor를 사용하며 strict K-only는 아니다. 기존 per-origin shrinkage D나 기존 pinv D와 same decoder라는 주장은 하지 않는다. 비교되는 것은 **같은 decoder fitting rule**이다.

Bolt NumPy와 native Torch32 numeric backend의 실제 latent 과거를 보존한다. 양 backend별 FIT decoder 적합을 분리하고 수치 차이를 기록한다. 기존 E/F cache는 exact lineage일 때만 재사용한다. 원 채널 subset의 frozen F 미래는 실제로 새로 추론해야 하며, 기존 full-C forecast에서 subset을 잘라낸 것을 새 실제 K-F inference로 세지 않는다. 두 결과의 차이는 원 baseline과 subset 예측의 수치 비교로만 기록한다.

새 static D32가 기존 native와 byte-identical하지 않으면 기존 head를 matched 대조로 재사용하지 않는다. 계획상 자료2×E3×seed2 최대12개 fresh zero-init native head를 같은 actual-C macro MSE·AdamW .001/WD0/clip1/batch4·600updates·CP0/100/200/400/600로 학습하고 TUNE MSE/earliest exact tie로 고른다. 반복 시행을 독립 generalization 반복으로 세지 않는다. 모든 선택 잠금이 저장된 후 새 DEV 예측을 한다. 일부 실패는 해당 경로에 기록하고 가능한 다른 경로를 계속한다.

## 진단·비용·해석

관측 과거 복원, 실제 FIT/TUNE/DEV full-observed 정규화 MSE/MAE·채널별/원단위 오차, 같은 complete-C 출력공간 minimum 및 E(Y) 진단을 분리한다. 미래 oracle는 fit·subset·head·decoder·branch 선택에 쓰지 않는다. 어떤 E의 D라도 그 D32를 대상으로 새 진단을 계산하고 옛 static/global rank-K/per-origin floor를 가져오지 않는다. FP32 rounding allowance와 real-space-only minimum을 유지한다.

실제 비용은 causal physical fill/FIT normalize→E→실제 frozen F 또는 선택 native→D→physical CPU return/sync를 포함한다. Static D의 training 비용은 신규 학습 비용에 따로 기록한다. Per-origin D 적합을 제거했으므로 이전 shrinkage latency를 비용으로 재사용하지 않는다. 새로운 같은 시행에 원 Raw 대조를 포함한다. 전체 비용 행렬·반복·순서 균형은 actual 실행 전에 별도 cost source/plan으로 고정한다. Shared RSS/VRAM을 방법 고유 메모리로 해석하지 않는다.

[판단] 같은 데이터·K·과거 전용 배포·고정 Bolt·전체 비용이라는 원 목적을 지속하는 최소 baseline이다. 최신 직접 지시의 자원 상한 해제를 따르며 새 개별 인간 승인이 있었다고 주장하지 않는다. 원 승인·실패·사용량은 보존한다. 큰 backbone 학습·설치·TEST·기존 중단 작업 재개/종료·Git 게시·논문/PPT는 포함하지 않는다. 현재 실행 원수치가 없으므로 성공이나 완료를 주장하지 않는다.
