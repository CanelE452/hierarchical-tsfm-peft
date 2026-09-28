# V5 status

[확인] 계산·비교·비용·검산·게시 완료. LEVEL을 Hog 개발 구성으로 조건부 유지하며, 예비 FULL과 K 변경은 실자료 미실행이다. 결과 commit `fb03b892dbf9cc41eae1e44081c789abcea24fa7`을 기존 `origin/main`에 push했고 `git ls-remote`로 실제 일치를 확인했다. 이 게시 완료 기록은 뒤따르는 문서 commit에 포함한다.

- Hog MSE 3.220290→2.666809(−17.187%); matched RAW 2.949584보다−9.587%. LoRA2.683560 대비 점추정은 작고 조건부 구간은0을 포함한다. Hog B기간·MAE 손해는 남는다.
- Bull0.727659→0.708616이지만 matched RAW0.697731보다 전체 MSE가 높다. Electricity0.165850→0.167458로 악화했다. 모든 자료에 같은 수정을 교체 적용하지 않는다.
- 같은 회차 Hog batch4 LEVEL RES353.725원점/초·225.859MiB allocated, 병합 LoRA167.330·328.676. batch1과 모든 블록 drift를 보존했다. 기존 RES 대비 속도 개선이나 환경 전반의 안정적 속도비는 주장하지 않는다.
- 모든 평가는 노출 개발 자료이며, seed 손실 평균이다. 독립 확증·예측 ensemble·논문 신규성 완료가 아니다.

```text
Budget                        Actual             Limit
Real-data attempts            12                 24
  Hog / Bull / Electricity      4 / 4 / 4
  Failed / restart / extension  0 / 0 / 0
Synthetic optimizer sessions   1 (10 updates)      6 (<=10/session)
GPU occupied seconds        1295.975           14400
GPU occupied minutes          21.600             240
CPU check seconds             10.635*            900
New storage                 ~23 MiB*           5120 MiB
New downloads / conflict wait  0 / 0
```

*게시 전 snapshot이다. 정확한 최종 값은 `budget_final.json`과 `ledger.json`을 따른다.

상한은 자동 증액하지 않았고 사용하지 않은 예산을 채울 실험은 하지 않는다. 부모 GPU job과 내부 fit 시간을 이중 합산하지 않았다. TRAIN PCA 준비는 부모 job 시간에 포함되지만 별도 시간을 계측하지 못한 한계가 있다(`preparation_accounting.json`).

실행 이력: 기존 예측 진단→LEVEL 선택→Hog4fit→두 이전 자료8fit→적합 종료→Hog 대응 비용→저장 결과 검산→그림 표시 배치 수리. 각 단계의 판단은 `decision_initial.json`, `decision_extension.json`, `decision_pre_cost.json`에 보존했다. 진단/합성 검사 소스의 실행 후 추가분은 `repairs/`에서 당시 실행 byte/hash와 구분했고 미실행 assert를 PASS로 세지 않았다.

게시 전 정적 검사 첫 회는 Windows 기본 cp949로 UTF-8 JSON을 읽다가 실패했다. 원장에 실패 시간과 traceback을 보존하고 UTF-8을 명시한 두 번째 검사에서 통과했다. 학습·예측·채점에는 영향이 없으며 fit 재시도로 세지 않는다(`publication_checks.json`).

재개 시 원장·PID·게시 상태를 먼저 확인한다. 적합은 닫혔으며 새 학습/예측을 중복 시작하지 않는다. 다음 연구의 단일 권고는 고정 LEVEL/matched RAW를 미사용 집단에서 확인할지 결정하는 것이며 이번 승인 범위에서 실행하지 않는다.
