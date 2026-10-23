# FIG_K_CURVE 주석 위치 수정 기록

실행 명령:

```powershell
E:\CODING\proj\hierarchical-tsfm-peft\.venv\Scripts\python.exe -B .cache/k_tau_rule_20261023_v1/render_figure_layout_001.py
```

동결된 `analysis._figures`를 공개 SUMMARY/RUNS/BASELINE_RECALCULATION scalar로 렌더링했다.
`Axes.text`를 임시로 감싸 제목이 `C  ETTh2_safe17400`이고 본문이 `K_tau = 4`,
원래 좌표가 `(0.02,0.03)`인 호출 하나만 `(0.02,0.10)`으로 옮겼다.
K_CURVE SVG 저장 직후 렌더링을 끝내 BOXplot 생성 단계에는 진입하지 않았다.

원본 PNG/SVG는 `reports/k_tau_rule_20261023_v1/attempts/figure_layout_001/`에 보존했다.
개선본은 `.cache/k_tau_rule_20261023_v1/figure_layout_001/`에서 만든 뒤 canonical 두 파일만 교체했다.
원본·개선본 SHA와 보호한 파일 SHA는 `reports/k_tau_rule_20261023_v1/FIGURE_LAYOUT_001.json`에 있다.

개선한 canonical PNG를 직접 열어 panel C의 문구가 Raw 점선 위 빈 공간에 놓여 겹침이 사라진 것을 확인했다.
통계·표·판정·BOXplot·등록된 소스는 변경하지 않았고 원자료/모델/PCA/학습에는 접근하지 않았다.
이 래퍼는 같은 attempt 경로가 있으면 덮어쓰지 않고 중단한다.
