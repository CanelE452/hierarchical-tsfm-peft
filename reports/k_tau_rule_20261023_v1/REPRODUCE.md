# 실행 기록과 결과 재검산

작업 위치는 `C:/Users/User/.codex/worktrees/adapter-diagnostics-v2/hierarchical-tsfm-peft`다.
사전등록 커밋은 `480c2211af3fd195916645b13ba952084a7b4eb9`이며, 커밋과 실제 스펙트럼 생성 시각의 관계는
`PREREGISTRATION_COMMIT.json`, `SPECTRUM_VALIDATION.json`, `SPECTRUM_INDEPENDENT_AUDIT.json`에 있다.

원래 실행 순서는 다음과 같다. 이미 봉인된 산출물이 있는 namespace에서 spectrum/plan을 새로 계산하거나 덮어쓰지 않는다.

```powershell
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -m research.k_tau_rule_20261023_v1.execute recheck
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -m research.k_tau_rule_20261023_v1.execute spectrum
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -m research.k_tau_rule_20261023_v1.execute plan
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -m research.k_tau_rule_20261023_v1.execute train
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -m research.k_tau_rule_20261023_v1.execute report
```

`train`은 단일 연속 controller로 실행한다. 중단 후 재입장은 저장된 controller receipt와
`BUDGET_PROFILE_UPDATE.json`의 실제 시간으로 먼저 예산을 재계산해야 한다.
기존 완료 run이 cache hit일 때 호출 시간을 새 학습 시간으로 오인하지 않는다.
고정 clock을 초기화하지 않는다.

아래 명령은 공개 표만으로 새 폴더에 보고서와 그림을 다시 만든다. 학습·PCA·모델·원자료·T 접근은 없다.
최종 `DECISION.json` 생성 이후 저장소 루트에서 실행한다. 재실행하려면 출력 폴더 이름을 바꾼다.

```powershell
@'
from pathlib import Path
import json
import shutil
from research.k_tau_rule_20261023_v1.analysis import build_report

source = Path('reports/k_tau_rule_20261023_v1')
target = Path('.cache/k_tau_rule_20261023_v1/report_replay_manual_001')
target.mkdir(parents=True, exist_ok=False)
for name in ('RUNS.csv', 'K_TAU.csv', 'BASELINE_RECALCULATION.csv', 'RUN_PLAN.json'):
    shutil.copyfile(source/name, target/name)
decision = json.loads((source/'DECISION.json').read_text(encoding='utf8'))
build_report(target, stop_reason=decision['stop_reason'], elapsed_seconds=decision['elapsed_seconds'])
print(target.resolve())
'@ | & 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 -
```

집계의 독립 검산 기록은 `FINAL_REPORT_VALIDATION.json`, 실제 학습 조건의 대조 기록은
`ACTUAL_CONTRACT_AUDIT.json`, 기존 151개 파일과 원 E: checkout 상태의 보존 확인은
`FINAL_HASH_VALIDATION.json`이다. 마지막 파일은 그 목록의 범위만 검증하며 전체 저장소나 원자료의 전수 hash가 아니다.

로컬 private 검증기가 있는 현재 작업 폴더에서는 다음 명령도 사용할 수 있다.

```powershell
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 .cache/k_tau_rule_20261023_v1/audit_actual_contracts.py
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 .cache/k_tau_rule_20261023_v1/verify_final_report.py --replay
& 'E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe' -B -X utf8 .cache/k_tau_rule_20261023_v1/verify_preservation.py
```

기존 학습 코어가 실행마다 저장 prediction을 FP64로 검산한다. 독립 공개 표 검증기는 그 scalar receipt와
봉인 연결을 확인하며 prediction 배열을 다시 열어 계산했다고 주장하지 않는다.
초기 G1 중단 보고서는 `attempts/initial_G1_stop_001/RESULTS_KO.md`에 보존했다.

최종 K 곡선은 ETTh2 패널의 K_tau 주석을 axes(.02,.03)에서 (.02,.10)으로 옮겼다.
수치·축·다른 패널·BOXplot은 바꾸지 않았다. 원본 그림은 attempts/figure_layout_001에 보존했다.
FIGURE_LAYOUT_001.json에 변경 전후 SHA와 위치가 있다. 표 기반 기본 replay는 동결 renderer의
원래 주석 위치를 재현한다. 최종 배치용 local wrapper는
.cache/k_tau_rule_20261023_v1/render_figure_layout_001.py이고, 해당 폴더의
figure_layout_001_REPRODUCTION.md에 재생성 방법이 있다.


검증기 원형의 공개 사본과 원래 .cache 위치로 복원하는 방법은 `validation_tools/README.md`에 있다.
