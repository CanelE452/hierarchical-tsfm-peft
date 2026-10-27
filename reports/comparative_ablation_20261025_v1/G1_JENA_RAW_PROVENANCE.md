# G1 Jena RAW foundation provenance

[확인] `PASS_COMPOUND_F_STATE_BINDING`이다. 이전 `UNRESOLVED_HISTORICAL_F_STATE`는 생성 시의 F 지문 불일치가 아니라, 복합 지문을 풀지 않은 이전 검증기의 기본값이었다. 기존 기록을 수정하지 않고 원본 manifest와 생성 코드로 증명했다.

[확인] 원본 manifest: `research/informs_broad_validation_20261015_v1/publication_derivatives/gzip_extra_aux_20261016_v1/files/item_1569.json.gz`.
압축 SHA256: `5411bddab320734cdd65a7b717376528d472c7a0a6c15efae3728218c205f6d5`.
연결된 RAW V 배열: `asset://.cache/informs_broad_validation_20261015_v1/predictions/primary_jena_E1_RAW_BOLT_SMALL_k0_s92601/V_select.npz`.
실제 파일 SHA256: `e78689d787f50c375d8ff221f9ef174ccac75da1a48719db4f280bd5031be462`. 원본 manifest 및 VERIFIED_ROWS의 결속 값과 모두 같다. 배열은 로드하지 않고 파일 바이트만 해시했다.

[확인] `prediction_cache.py:171–176`의 `model_sha`는 F 상태 자체의 해시가 아니라 baseline/F/scaler/data 네 항목의 복합 SHA256이다. `fit_base.py:42–47`은 여기에 들어가는 F 지문을 `initial_hash` 또는 실제 module state hash에서 가져온다. `training.py:11–12`의 동일한 canonical JSON 규칙으로 다음 세 값을 재계산했고, 모두 원본과 바이트 단위로 일치했다.

```text
model_sha   19793fd032c67655ab9ec559d1122a7c32b62366d99100c99a4e559e915b0392
config_sha  430698c8119d32d3d61502510931d384cd2a8129dbec5e36c94ed46c0d3621af
source_sha  c29a0f4d0f2979bd9d3de1db096413f2baecbf23a1429350148b553d09d5f384
F_SHA       c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874
```

[확인] 생성 코드 7개(prediction_cache.py, fit_base.py, data_contract.py, training.py, evaluation.py, networks.py, gpu_runtime.py)의 현재 SHA가 manifest의 역사적 SHA와 전부 같다. 정확한 기대값/실제값은 동반 JSON의 `producer_source_comparisons`에 저장했다. 입력 10개에 대해 작업 전후 SHA도 동일하다.

[확인] 이전 verifier는 `verification.py:437–440`에서 원본 복합 지문을 `prediction_model_label_sha256`로 옮기고, `:445`에서 model_sha 기본값을 unresolved로 채운다. 따라서 이 기본값만으로 F 불일치를 판단할 수 없다. 기존 `diagnostics/fullC_fixed_basis_diagnostic_20261016_v1/audit_fixed_basis.py:288–291`도 같은 방식으로 Raw의 동일 F/scaler/data 결속을 검증했다.

[확인] 저장 지표 교차 검산: 원본 MSE 0.8491941389472357, VERIFIED_ROWS MSE 0.8491941389472368이며 차이는 1.1102230246251565e-15다. MAE는 둘 다 0.4813883402485757이다. 이 보고서에서는 배열 지표를 재계산하지 않았으며, 별도 G1 배열 검산을 대체하지 않는다.

[확인] 이 증거로 역사적 RAW F 지문 공백은 해결되므로 해당 항목 해결을 위한 재추론은 필요 없다. 현재 로드한 F의 지문 확인과 나머지 게이트는 별도다. 학습·추론·T 접근·기존 파일 수정은 하지 않았다.

재현: 프로젝트 루트에서 아래 코드를 실행한다. JSON에 저장된 완전한 payload를 사용하여 모델/설정/소스 복합 지문과 모든 입력 파일을 다시 검산한다. 출력 파일을 만들지 않는다.

```python
from pathlib import Path
import hashlib, json
p = Path('reports/comparative_ablation_20261025_v1/G1_JENA_RAW_PROVENANCE.json')
a = json.loads(p.read_text(encoding='utf-8'))
for name, item in a['compound_hashes'].items():
    payload = json.dumps(item['payload'], sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    actual = hashlib.sha256(payload).hexdigest()
    assert actual == item['expected'] == item['actual'], name
for path, expected in a['input_hashes_before'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert all(a['checks'].values())
print('PASS: historical RAW F-state compound binding')
```
