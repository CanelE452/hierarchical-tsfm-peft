# Runtime / R2 controller 소스 리뷰

작성: 2026-10-15. 담당 범위는 소스 읽기와 이 문서 작성뿐이다. `runtime.py`, `campaign.py`, `data.py`를 import하거나 실행하지 않았고, 실제 자료·F·GPU를 로드하지 않았다. 현재 실행 중인 루트 R2 학습과 경쟁하는 실험도 시작하지 않았다. 소스 수정은 하지 않았다.

리뷰한 파일 SHA256:

```text
runtime.py   a11e7eb6b1c4dc043ff36489ec51688d57b3102a6a2f6cd6666b597986c4e07a
campaign.py  2e92e1ea5231018d232c685ac327e7cfbd360968454769e23270d075fdd15ec3
data.py      274289f01e46780c2795f949d11ebe3dbb886b4dac5dc45ca3b8a5e2ec5eae52
```

이 파일들은 루트가 별도로 보관할 실제 실행 소스 snapshot과 대조해야 한다. 아래 `[확인]`은 해당 소스의 호출·인덱싱 흐름을 끝까지 추적했다는 뜻이다. 실제 데이터/학습/예측의 실행 검증을 뜻하지 않는다.

## 수정 반영을 확인한 지적

**R2 연장 gate의 한쪽 조건 누락:** 최초 소스는 `selected_update == endpoint`일 때만 연장했다. 승인 문서는 마지막 CP 최선과 B 곡선의 지속 하강을 각각 연장 조건으로 둔다. 따라서 과거 CP가 최저이나 최근 CP200→400→600이 계속 하강하는 경우를 놓칠 수 있었다. 루트에 즉시 전달했다.

[확인] 현재 `campaign.py:76`의 `still_decreasing`은 최근 B checkpoint 세 개의 엄격한 감소를 판정한다. `:242`는 마지막 CP 최선 **또는** 이 조건이면 동일 dataset/r/seed의 B0–B4 그룹을 함께 연장한다. `:177`도 2400 endpoint에서 같은 조건을 `UNDERTRAINED_OR_UNRESOLVED` 판정에 사용한다. 600→1200→2400 범위를 넘는 추가 연장은 없다. 최근 세 CP라는 정의는 이번 구현에서 사전에 고정한 운영적 기준이며 수렴 증명은 아니다.

## 남는 소스상 주의점

1. **[P2, 확인] 프로브 peak 메모리는 모델별 peak가 아니다.** `campaign.py:215`는 `max_memory_allocated/reserved`를 읽지만 각 모델 전에 peak를 초기화하지 않는다. 이전 모델의 최대치가 다음 행에 남는다. 이 수치를 모델 간 메모리 우위나 최종 COST로 해석하지 말고, 프로브 프로세스의 누적 peak로 표시해야 한다. 최종 비용 단계의 별도 process/동일 장치·dtype·출력 조건 측정은 별도로 필요하다.

2. **[P2, 확인] 연장 실패 또는 dataset 로드 실패는 core 전체를 중단한다.** 최초 600-update `train`만 `:233`의 예외 처리 안에 있다. 연장 `train`(`:247`)과 `dataset_windows`(`:225`)의 예외는 최상위까지 전파된다. 이미 완성된 checkpoint/selected 파일은 보존되지만 최종 `R2_SELECTIONS.json`은 끝에서만 저장되어 생성되지 않을 수 있다. 실패 시 루트의 별도 복구 driver는 이미 완료한 파일을 검증·재사용하고, 기존 파일을 덮어쓰지 않은 새 시도 ID로 미완료 경로를 이어야 한다. 단순 `core` 재실행은 `x/xb` 모드의 기존 cp0/fit/failure 경로와 충돌한다.

3. **[P2, 확인] 자원 실패와 방법 무효가 같은 예외 상태에 들어간다.** `:237`은 모든 최초-fit 예외를 `INVALID`로 기록한다. CUDA OOM/자원 상한 문제도 이 경로에 포함된다. 실제 그런 실패가 발생하면 원 실패 기록은 보존하고 최종 RUN_INDEX/보고서에 자원 원인을 구분해야 한다. 이 경우를 방법의 `NO_GAIN`/`HARMFUL` 근거로 사용하면 안 된다. 현재 리뷰에서는 실제 실패 발생 여부를 조사하지 않았다.

4. **[확인, 후속 단계 조건] C 잠금은 이 두 파일 자체에서 강제하지 않는다.** `assess`는 임의 split을 받을 수 있지만 현재 `probe/core/train`의 점수 호출은 A/B뿐이다. 따라서 현 코드에서 C 점수 누출 호출은 발견하지 못했다. 앞으로 추가할 C driver는 모든 후보/후속 선택 lock과 비용 완료를 검사한 다음에만 C를 점수화해야 한다. 이 리뷰는 아직 작성 중인 R1/R3/R4/비용/C driver의 안전성 검증을 대신하지 않는다.

5. **[확인, 검증 범위] 실제 Raw parity의 실행 증거는 이 리뷰에 없다.** B0/r=C 초기식은 아래처럼 Raw와 일치하지만, 새 로더·normalizer·실제 F를 포함한 배치 parity는 루트의 별도 실제 receipt로 확인해야 한다. 소스의 수학적 항등만으로 실제 batch parity가 끝났다고 쓰면 안 된다.

## 문제를 발견하지 못한 핵심 흐름

**데이터 API와 과거 입력:** [확인] `load_dataset → make_fold → Windows`의 key/인자 연결은 현재 `data.py` 반환과 맞는다. `fit_statistics`는 fold A bounds만 사용한다. `Windows.get`은 `origin + [-L,0)` 인덱스만 history로 만들고, 그 시점까지의 마지막 관측 `previous[t]`를 사용한다. window 시작 전 관측은 0(A 평균의 정규화값)으로 초기화한다. 미래 `[origin,origin+H)`의 값과 mask는 별도 `y/mask` 반환에만 들어가며 model 입력에 전달되지 않는다. `previous`를 전체 허용 FIT에 계산해도 각 과거 t의 누적 인덱스는 t 이후 행에 의존하지 않는다. 반환 자산의 전체 배열은 GPU에 올라가므로 C가 물리적으로 메모리에 존재하지 않는다는 주장은 할 수 없다.

**A-only PCA와 초기 E/D:** [확인] `Windows.basis`는 A 행의 인과 채움값에만 PCA를 적합한다. r=C는 center=0/E=I다. 최초 D는 lag0에만 `pinv(E)`를 두고 다른 lag는 0이다. B0/r=C의 초기 출력은 `x_last + (f - z_last) = f`이므로 Raw와 수학적으로 일치한다. 작은-r는 현재 원 C 값에 잠재 변화량을 더한다.

**Lag 좌표:** [확인] `lag_features`의 horizon index h=0..H-1와 lag ℓ에 대한 값은 `joined[L-ℓ+h] - z[L-1-ℓ]`다. 원 시간으로는 `z_hat(origin+h-ℓ) - z(origin-1-ℓ)`이며, h<ℓ일 때 첫 항은 관측 history에서, h≥ℓ이면 F의 예측에서 온다. 현재값 보존과 origin 기준 lag 차이가 일관되며 실제 미래 Y를 참조하지 않는다. P가 전체 과거 창을 혼합하므로 각 과거 표현의 prefix causality는 주장하지 않는다.

**Mask 손실과 점수:** [확인] target 미관측 값은 0 placeholder이며 원 `observed` mask로 SSE/절대오차에서 제외한다. 평가 점수는 채널별 관측수로 나눈 뒤 활성 채널을 동일 가중 평균한다. 학습은 모든 합법 A origin의 반복 target 노출을 포함한 고정 채널별 분모를 쓰고 `N_A / batch`를 곱한다. 균등 origin 표집일 때 전체-A 채널 평균 목적의 minibatch 추정량이며, batch마다 관측이 적은 채널을 임의로 평균에서 제거하지 않는다. 실제 관측수/finite 상태의 데이터 검증은 별도이다.

**F gradient와 median:** [확인] Adapter의 `bolt(z)`는 `no_grad` 안에 있지 않아 입력/E/P gradient 경로가 유지된다. Bolt weight는 optimizer에서 제외하고 `requires_grad=False`다. 설치된 `chronos_bolt.py`의 forward는 이미 inverse-normalized `quantile_preds[B,Q,H_native]`를 반환하며 wrapper는 q=.5와 앞 H48을 선택한다. 추가 inverse 변환을 중복 적용하지 않는다. 새 `SafeInstanceNorm`의 forward 식은 설치된 InstanceNorm과 같고 0 variance의 sqrt backward만 유한한 0으로 정의한다. 실제 Raw/기존 경로 parity나 gradient 값은 실행 receipt로 확인해야 한다.

**Checkpoint와 연장 상태:** [확인] CP0가 후보에 포함되고 B MSE 최소, 동점이면 이른 update를 선택한다. A 평가는 사전 고정 32-origin 진단이며 선정 키에는 들어가지 않는다. 연장은 선택 checkpoint에서 시작하지 않고 별도 보관한 마지막 상태와 optimizer를 복원하여 동일 2400-step schedule의 다음 prefix를 이어 간다. 각 확장 구간 끝에 모든 CP 중 같은 B 기준으로 다시 선정한다. checkpoint `.pt`는 CPU state_dict이므로 현재 `weights_only=True` load 형식과 맞는다. 이 흐름에 C 참조는 없다.

최종 판정: **현재 리뷰한 R2 소스 흐름에서 신규 미래 정답 누출·lag off-by-one·mask 분모 오류 또는 즉시 실행을 막는 데이터 API 불일치는 발견하지 못했다.** 연장 gate 지적의 소스 반영은 확인했다. 남는 항목은 프로브 해석, 중단 후 복구/실패 분류, 후속 C 잠금과 실제 parity의 실행 증거다. 실제 작업의 성공·수렴·성능·전체 캠페인 완료를 판정한 리뷰가 아니다.
