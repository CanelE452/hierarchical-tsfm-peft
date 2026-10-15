# 별도 프로세스 비용 측정의 Windows 실행 경로 수선

학습·R1·일반 비용 측정 뒤, 첫 `exclusive_all` 시도에서 Windows 가상환경 launcher 부모가 다른 exclusive 작업으로 감지됐다. 실제 자식은 `exclusive()` 첫 경쟁 검사에서 거부됐으며, 모델 로드·정밀 비용 trial 이전의 실행 오류다. 방법의 성능 실패가 아니다.

- [실패 영수증](EXCLUSIVE_LAUNCHER_FAILURE.json): 당시 새 queue의 PID·부모·argv와 C 미채점 상태를 보존했다. 해당 새 실패 queue만 종료했다. 기존 Trading32/Anaconda 작업은 종료하지 않았다.
- [잔여 터미널 오류 출력](EXCLUSIVE_LAUNCHER_FAILURE_LOG.txt): 도구에서 마지막으로 회수한 원 출력을 저장했다. 이 파일을 모든 이전 터미널 출력의 완전한 사본이라고 주장하지 않는다.
- 종료 전 범위 검사 두 번은 console descendant/검사 명령 자신을 포함해 assertion에서 정지했다. 그 두 검사에서는 프로세스를 종료하지 않았다. Python 이름·정확한 script argv·phase를 한정한 세 번째 검사에서 새 queue만 종료했다.
- [실행환경 확인](DIRECT_INTERPRETER_PARITY.json): 원 가상환경의 실제 worker와 같은 `mlts/python.exe`를 직접 호출하고 기존 `.venv/Lib/site-packages`를 `PYTHONPATH`로 지정했다. torch/numpy/chronos/transformers/psutil/sklearn의 실제 파일 경로와 버전, CUDA 사용 가능 상태를 확인했다. 새 설치는 없다.

다시 실행한 명령은 아래와 같다. 기존 `cost_and_score.py`와 경쟁 작업 검사는 수정하지 않았다. 선정 모델·통계·일반 비용·과학 소스와 잠금 SHA도 유지했다.

```powershell
$env:PYTHONPATH='E:\CODING\proj\hierarchical-tsfm-peft\.venv\Lib\site-packages'
& 'C:/Users/User/anaconda3/envs/mlts/python.exe' -X utf8 research/temporal_frontend_campaign_20261015_v1/cost_and_score.py exclusive_all
```

패키지 경로·버전 검사는 모든 수치가 비트 단위로 같다는 검증이 아니다. 실제 재실행 상태는 `cost_exclusive/`의 원 trial, `COST_LOCK.json`, `COST.csv`로 확인한다. 첫 실패를 성공 측정으로 세거나 기존 결과를 덮어쓰지 않는다. C를 보거나 모델을 다시 선택한 수선이 아니다.
