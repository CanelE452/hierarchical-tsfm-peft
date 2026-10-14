# D1 변환 순서 probe 준비

상태: **PREPARATION_ONLY**. [확인] 코드·JSON 메타데이터를 읽고 새 순수 수학 helper와 합성 CPU 테스트를 작성했다. 이전 scientific NPZ/CP, 실제 모델·GPU·보호 TEST는 열거나 실행하지 않았다. Bi의 검산·비용·보고가 완료된 뒤 root가 실제 실행으로 전환한다.

목적은 작은 출력 rank의 근사 가능성과 투영된 과거를 F가 예측하는 능력을 구분하는 것이다. 원 목적의 두 가지 진단 경로는 (a) 표현 예산 부족을 확인해 K/readout 검토에 근거를 주고, (b) RAW 출력은 근사되는데 F(XB)가 달라지는지 확인해 좌표·입력 분포 문제의 단서를 주는 것이다. [추정] 어느 경로가 실제 실패를 설명하는지는 아직 unknown이다. full-C 회전의 차이는 정보 손실 없이도 발생 가능한 반례이며 실제 자료의 원인을 대신하지 않는다.

## 고정 수식과 후보

`delta = FIT RAW forecast - exact preprocessed FIT x_last`만 basis 적합에 사용한다. 실제 미래 Y, 미래 mask, 학습된 Beijing full-C teacher basis는 basis 적합 인자가 아니다. FP64 `delta.T @ delta / n`과 `(delta-mu).T @ (delta-mu) / n`의 eigensystem을 따로 계산하고, 각 eigenvector의 최대 절댓값 좌표가 양수가 되도록 부호를 고정한다. 동일 NumPy/LAPACK 조건에서 결정적이며 중복 eigenvalue 내부 회전의 플랫폼 간 동일성을 주장하지 않는다.

각 estimator의 `max(1,floor(C/4))`, `floor(C/2)`, `C`를 전부 저장한다. Robin은 4/8/17, Beijing은 2/5/11이다. full-C는 회전 진단이며 압축 성과가 아니다. 별도 `identity_C`를 먼저 실행해 같은 F의 저장 RAW parity를 확인한다. 기존 RAW와 다르면 그 출력·실패를 보존하고 비교를 중단한다. 기본 normalized tolerance는 atol/rtol 각각 2e-6으로 명시적으로 저장된다.

```text
through-origin projected RAW = x_last + delta B B^T
centered projected RAW       = x_last + mu + (delta-mu) B B^T
query E(X)                    = X B                 (입력 mean 이동 없음)
through-origin F decode       = x_last + (F(XB)-x_last B) B^T
centered F decode             = through decode + mu_perp
mu_perp                       = mu - mu B B^T
```

runtime E/decode는 FP32 B를 사용한다. basis와 별도 RAW projection reference는 FP64로 저장하여 수치 오차를 분리한다. 모든 후보·split에서 과거 복원 오차, projected RAW 대비 F(projected past) 차이, 실제 Y의 observed channel-macro normalized/physical MSE·MAE·채널별 값과 관측수를 남긴다. 원단위 오차는 정규화 오차에 저장 std를 곱해 계산하며 자료 간 이질 단위를 pooling하지 않는다. 이 선형 B 진단은 기존 nonlinear D의 보편적 하한이 아니다. 기존 nonlinear D 재적합에는 절대 z_F/z_last가 필요하며 이 helper의 delta와 혼합하지 않는다.

## 인터페이스와 저장 schema

[확인: 코드] `projected_past_probe.py`는 import 시 기존 engine/data를 import하거나 assets를 열지 않는다. `run_probe`는 caller가 이미 검증한 배열과 predictor callback을 받는다. 실제 GPU의 callback은 `make_bolt_predictor(original_bolt)`이며 모델 생성·normalizer 교체·source/model verification은 root runner의 책임이다.

```text
splits: fit/tune/val 또는 fit/tune/이미 노출된 eval; test 허용 안 함
각 split:
  history             FP32 [N,512,C]    정확한 기존 normalized past
  raw_forecast        FP32 [N,48,C]     기존 RAW baseline
  target              FP32 [N,48,C]     기존 평가 노출 범위만
  observed            bool [N,48,C]
  history_observed    bool [N,512,C]
  origins             int64 [N]
공통 mean/std         저장된 FP32 [C]; 재계산값으로 대체하지 않음

INPUT_<split>.npz: 원 입력·RAW·target/masks·origins·last·mean/std
<candidate>__basis.npz: B64/B32, delta_mean, mu_perp, eigenvalues
<candidate>__<split>.npz:
  projected_raw_forecast, projected_raw_reference_fp64,
  projected_past_forecast, latent_past, latent_last, latent_future,
  raw_last, reconstructed_history
RESULT.json: 모든 후보·split 지표, caller provenance,
             배열 dtype/shape/SHA, 파일 SHA, 경로별 실패와 partial 목록
```

새 output만 허용한다. 성공·실패 모두 exclusive writes를 쓰고 partial artifacts를 보존한다. `actual=True`는 L512/H48, 새 followup cache output, caller의 기존 Bi `rt.Limit`을 요구하며 프로세스 생성부터 300초와 RSS8GiB를 추가로 협력 점검한다. 예산은 새로 만들거나 초기화하지 않는다. root가 외부 300초 watchdog과 shared Limit terminal receipt를 관리한다. Limit의 기존 campaign/storage/VRAM gate가 막으면 중단하며 cap을 수정하지 않는다.

## 실행 연결과 준비 검증

```powershell
& ./.venv/Scripts/python.exe -m pytest -q research/smallk_failure_followup_20261014/test_projected_past_probe.py
& ./.venv/Scripts/python.exe research/smallk_failure_followup_20261014/projected_past_probe.py --describe
```

[확인] 14개 합성 CPU 테스트가 통과했다(exit0). through/centered fit, 선형 공유 시간 예측기의 교환, centered mu_perp, full-C 비선형 반례, identity, 실제 미래/mask와 basis의 분리, channel-macro 단위 분리, 모든 후보 저장, exclusive output, partial/path-specific 실패, TEST·unmetered actual 차단을 확인했다. CLI 설명과 py_compile도 exit0이다. 독립 소스 리뷰가 지적한 basis-fit 실패의 잘못된 단계/경로 기록을 수정했으며, 회귀 테스트의 실패→통과를 확인했다. 검토한 D1 수식·후보 보존 범위에서는 추가 actionable finding이 없었다. 이 결과는 실제 forecasting 성능 검증이 아니다.

남은 runner 연결은 **기존 창의 정확 재구성과 검증**이다. Robin `common.load_data(...sizes=(256,64,None))`의 원 physical/mask·origin을 사용하되 원 RAW/contract 저장 mean/std로 override하고 `ex.preprocess`를 호출해야 한다. Beijing은 기존 repair FIT/TUNE payload와 `load_evaluation_inputs`의 이미 저장된 EVAL normalized history를 사용한다. 첫 EVAL 개봉 API를 재실행하지 않는다. 모든 split의 RAW origin/stats 및 normalized past SHA·FIT last를 검증하고, 원 source/RAW/MAIN CP·model snapshot·설치 Chronos SHA를 확인한 manifest를 `provenance`로 전달한다. `run_probe` 자체는 이러한 caller 검증을 독립 감사했다고 주장하지 않는다.

실제 실행 전에 root의 새 source snapshot에는 helper/tests/이 문서 및 아래 원 의존 source를 포함한다. [확인: 이번 source-only 해시] engine `05f2cf63b3f7f6a01bf722c51771455be9c73c21ab301d7eb950b3dd58bab817`, controls `324c75a65a001ac0cae30610abb148499718cbad838a69e0012df4a85bd83393`, newdata `ccef24b64c9beef0e6e31f9bc49dbad4e8e6ad57a6ce7c99973ccbce691de3bd`, newdata_repair `8cc75ffc4c86329939ae5155192afa03494f7bfff70fcf1ab2946c8c13f83900`, experiment `b2dd127caa3bfec96556b6de56eb803960c58fb92fcbfea211c6fb06ea13a9d3`, common `1bb0dc92a37d3bb3af525c2e72b62ab52430237ac4653086da6bf361a8c353c2`, Bi runtime `c8852f0811ce76f6ea70b3f97e014b3794e2250a94bf6f0cc02d17977b0a7d1a`. 이번 준비에서는 source snapshot 복사본을 생성하지 않았다.

새 학습·rank 선택·독립 holdout 주장은 하지 않는다. 원 source/결과와 Git을 변경하지 않았다.
