# 검증 도구 공개 사본

이 폴더의 도구4개·기존 안내2개·초기394검사 백업은 원래 .cache 파일과 byte 단위로 같은 공개 사본이다.
BYTE_COPY_MANIFEST.json에 원위치와 SHA256을 기록했다. 등록된 연구 코드11개와 학습 조건은 변경하지 않았다.

도구는 원래 .cache 위치를 기준으로 저장소 경로를 계산한다. 저장소 루트에서 아래 절차로 원위치에 복사한 뒤 사용한다.
기존 파일의 내용이 다르면 중단하며 덮어쓰지 않는다.

```python
from pathlib import Path
import json
import shutil

source = Path('reports/k_tau_rule_20261023_v1/validation_tools')
manifest = json.loads((source/'BYTE_COPY_MANIFEST.json').read_text(encoding='utf8'))
for item in manifest['files']:
    current = source/item['file']
    target = Path(item['original'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == current.read_bytes(), str(target)
    else:
        shutil.copyfile(current, target)
```

공개 표와 공개 scalar 영수증만으로 하는 수치 검산:

```powershell
python -B .cache/k_tau_rule_20261023_v1/verify_final_report.py --replay --replay-dir .cache/k_tau_rule_20261023_v1/report_replay_new
```

replay 출력 폴더는 새 이름이어야 한다. 이 명령은 모델·PCA·학습을 실행하지 않으며 FINAL_REPORT_VALIDATION.json을 갱신한다.

나머지 도구는 기능별 전제가 다르다.

- audit_actual_contracts.py: 원래 저장소의 이전 실험 metadata와 canonical cache SEARCH_RESULT.json 등 봉인 입력이 필요하다. 저장 prediction·모델은 열지 않는다. ACTUAL_CONTRACT_AUDIT.json을 갱신한다.
- verify_preservation.py: 최초 PRE_HASHES에 기록한 원 E: checkout HEAD·Git 상태와 선택151개 파일을 비교한다. 다른 clone에서는 같은 보존 판정을 기대할 수 없다. FINAL_HASH_VALIDATION.json을 갱신한다.
- render_figure_layout_001.py: 기존 attempt/output 폴더가 있으면 덮어쓰지 않고 거부한다. 기존 시도 기록을 삭제해 재실행하지 않는다. 동작과 정확한 위치 변경은 함께 둔 figure_layout_001_REPRODUCTION.md에 있다.

이 도구들을 일괄 재실행하는 절차는 제공하지 않는다. 공개 표의 수치 재검산과 원 실행환경·파일 보존 재검증을 구분한다.
ZIP은 앞서 전달한 결과 스냅샷 그대로 보존하며, 이 게시용 안내와 초기394검사 백업은 ZIP 생성 이후 별도 공개했다.
