# 짧은 search 학습의 엄격한 연장

[확인] `learning_extension.extend_fit`은 완료된 부모의 A50 search 상태를 **별도로 사전 등록한 새 ID**에 복제한 뒤 같은 학습을 이어 간다. 예전 A60 refit은 복사하지 않는다. 연장된 A_stop 선택 epoch로 A60 scaler·PCA·모델을 새로 초기화하여 refit한다. 이 문서는 실행기 구현과 합성 CPU 검증을 설명하며 실제 과학 실험이 완료되었다는 보고가 아니다.

```python
from learning_extension import extend_fit

receipt = extend_fit(
    parent_run_id, new_run_id, [.25, 1., 3.], task,
    cache_root, public_root, foundation, deadline,
)
```

부모가 `[.25]`이면 새 schedule은 `[.25, 1.]` 또는 `[.25, 1., 3.]`이어야 한다. `[1., 3.]`로 .25 경계를 삭제하면 작은 마지막 batch의 위치가 바뀌므로 거부한다. 기존 경계 사이에 새로운 경계를 삽입하는 것도 거부한다. LR·seed·effective/micro batch·K/grid·family·fold·F를 비롯한 모든 기존 spec 값은 유지하며 새 ID·전체 epoch schedule·stage label·부모 증빙만 바뀐다. Registry 등록은 호출자가 먼저 수행해야 하며 이 모듈에는 등록 기능이 없다.

## 검증과 보존

- 부모의 `RUN_CONTRACT.json`, 공개 `SOURCE_CONTRACT.json`, 선택 bundle SHA·spec, 공개 search receipt를 검산한다. 현 `fit_base`가 사용하는 6개 소스 파일이 부모 기록과 모두 같아야 한다.
- 현 task/F로 원 `RUN_CONTRACT`를 재계산한다. 실제 A50 통계·PCA 초기화로 scaler/initialization SHA를 재계산하여 마지막 search checkpoint와 대조한다. AdamW param group의 LR·weight decay 등을 새 동일 optimizer와 대조하며, legal origins로 permutation sampler를 재생하여 순서·위치·exposures·NumPy RNG를 검산한다.
- 부모는 완료된 search이어야 한다. 등록된 모든 경계와 exposures·update 수가 맞아야 하며 model/optimizer/sampler/torch RNG/CUDA RNG가 모두 있어야 한다. 학습 파라미터가 없는 고정 PCA checkpoint는 이 API로 연장하지 않는다.
- 마지막 search checkpoint를 새 ID의 `search/checkpoint_000000.pt`로 복제한다. contract만 교체하고 나머지 모든 payload의 구조·tensor/array 원바이트 SHA를 전후 대조한다. 원 파일 SHA, old/new contract, 부모 fit 파일 목록 SHA, 복제 파일 SHA를 `MIGRATION_PROVENANCE.json`에 남긴다.
- 부모의 선택 artifact·search/refit·계약·영수증은 함수 전후 SHA가 같아야 한다. 다른 ID의 기존 파일이 있거나 불완전한 migration만 있으면 덮어쓰지 않고 실패한다. 동일 migration이 확인된 새 ID의 deadline 중단은 같은 호출로 재개할 수 있다.
- 새 ID가 완료되었더라도 원 부모 SHA·현재 소스·spec·복제 checkpoint를 다시 검산한 뒤에만 엄격하게 재사용한다. `LEARNING_EXTENSION_RECEIPT.json`은 최초 완료 때 한 번 작성한다.

`SOURCE_PATCH_LAG_C.json`의 자동 예외는 구현하지 않았다. 초기 Beijing 3건처럼 부모 소스가 다르면 명확히 거부한다. 해당 조건은 현 소스로 새 ID에서 scratch 실행해야 한다. 근거가 충분하지 않은 호환성 추정으로 예외를 넓히지 않는다.

## 숫자의 의미

기본 receipt의 `search_updates`와 `F_logical_streams`에는 이어받은 search 이력이 포함되어 있다. `extension_accounting`에는 다음을 별도로 남긴다.

- `inherited_search_updates`, `inclusive_search_updates`, `new_search_updates`
- `new_refit_updates`, `new_total_updates`, `preserved_parent_refit_updates`
- inherited/inclusive/new search F streams, 새 refit F streams, `new_total_F_logical_streams`

원래 짧은 A60 refit 비용은 새 실행 비용에 다시 합치지 않는다. 실제 process wall·CPU/GPU peak는 외부 resource ledger와 원 engine의 측정을 이용해야 하며 이 helper는 과거 시간이 신규 시간이라고 보고하지 않는다. 검증 중 통계/PCA·모델 초기화 비용은 실제 호출의 일부이고 F forward는 발생하지 않는다. M/V/T 예측이나 재채점은 하지 않는다.

## 진단 한계

기존 engine은 deadline이 training segment 중간에 걸리면 partial segment의 clipping count·gradient/loss 요약을 반환하지 않는다. optimizer·RNG·sampler checkpoint로 학습은 이어 갈 수 있지만 이 진단 누락을 복구했다고 주장할 수 없다. 부모 또는 새 ID의 `BUDGET_LIMIT` event가 하나라도 있으면 완료 후에도 `clipping_diagnostics_complete=false`로 보수적으로 남긴다. source live 변경을 피하기 위해 engine은 수정하지 않았다.

부모의 마지막 search 상태에 기존 외부 서명이 없는 경우, 이 helper는 관측한 원 checkpoint 바이트를 고정하고 계약·일관성 검사를 수행한다. 최초 migration 이전에 공격자가 원 checkpoint와 모든 연관 기록을 일관되게 다시 쓴 경우까지 원래 학습 결과와 같다고 증명하지 않는다. 새 migration 이후에는 고정된 SHA에 대한 변조 검사가 가능하다.

## 합성 CPU 검증

```text
python -X utf8 -m unittest test_learning_extension -v
```

[확인] 구현 전 모듈 부재로 6개 검사 RED를 확인했고, 구현 후 확장한 9개 검사가 통과했다. 실제 GPU·F 모델은 실행하지 않았고 원 campaign core 6개도 수정하지 않았다.

1. 동일 `.25 → 1 → 3` 경계의 scratch와 연장 search 마지막 모델·AdamW·sampler·torch/CUDA RNG·선택 상태가 bit-exact이며 새 A60 refit도 bit-exact이다.
2. 선택 epoch와 A_stop MSE가 같고 부모 원 파일 SHA가 유지된다.
3. 엄격한 완료 ID 재사용, 다른 schedule 거부, 외부 기존 ID 거부를 확인한다.
4. scaler·initialization·optimizer LR·sampler·source·spec LR·data·F identity 변조를 거부한다.
5. deadline 중단 후 재개의 선택 모델이 scratch와 같으며 진단 누락 flag가 유지된다.
6. 합성 frozen F를 둔 B0 연장과 scratch가 같고 inherited/new F 비용이 분리된다.

이 결과는 continuation 구현의 합성 검증이다. 실제 데이터에서 더 긴 학습이 개선되는지, 수렴했는지, 독립적인 일반화가 성립하는지는 새 ID의 실제 학습·추론·검산 결과로 별도 판단해야 한다.


## 최종 평가 자원 배분

`deepen.py`의 GPU 누적 한도43200초는 유지하고 내부 reserve만3600초에서9000초로 높였다. 긴 단일 학습 child가 최종 재추론·비용·T 시간을 모두 쓰지 않게 하는 실행 배분 변경이다. 모델·학습 손실·데이터·선택 기준은 바뀌지 않는다. 중단 시 부분 학습과 미실행 항목을 그대로 기록하며, 이 변경으로 충분 학습이나 수렴을 보장하지 않는다.
