#!/usr/bin/env python
"""REPORT_KO.md and FINAL_DECISION.md in the order required by contract section 11.
The STABILITY table comes first, as the user asked."""
from __future__ import annotations
import csv, time
import numpy as np
import ga


def rows(name):
    p = ga.OUT/name
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def fmt(x, nd=3):
    try: v = float(x)
    except (TypeError, ValueError): return str(x) if x not in (None, '') else '—'
    return '—' if not np.isfinite(v) else f'{v:.{nd}f}'


def table(head, body):
    return '\n'.join(['| '+' | '.join(str(h) for h in head)+' |',
                      '| '+' | '.join(['---']*len(head))+' |'] +
                     ['| '+' | '.join(str(c) for c in r)+' |' for r in body])


def verdict(stab):
    live = [r for r in stab if r['verdict'] != 'INFEASIBLE']
    controls = [r for r in live if r['role'] == 'control']
    candidates = [r for r in live if r['role'] == 'candidate']
    if any('CONTROL_FAIL' in r['verdict'] for r in controls):
        bad = [r['config'] for r in controls if 'CONTROL_FAIL' in r['verdict']]
        return 'ATLAS_CONTROL_FAIL', ('음성 대조에서 우리 파이프라인이 격차를 만들어 냈다: '
                                      + ', '.join(bad) + '. 학습 모델 쪽으로 기운 파이프라인이라 결론을 내지 않는다.')
    stable = [r for r in candidates if r['verdict'].startswith('STABLE_GAP')]
    if stable:
        return 'ATLAS_STABLE_GAP', ('두 기간 모두 격차가 유지된 설정: ' + ', '.join(r['config'] for r in stable)
                                    + '. 위치 분석과 난이도 가설까지 쓰고 멈춘다.')
    return 'ATLAS_NO_STABLE_GAP', ('후보 중 두 기간 모두 격차가 유지된 설정이 없다. GIFT-Eval에서 격차 우선 경로는 '
                                   '여기서 종료하고, 다음은 사전학습 분포 밖 자료로 Stage 1을 다시 하는 것이다.')


MEANING = {
    'STOP_DEBUG': 'E0(공식 점수 재현) 실패. 과학적 판정 없음.',
    'ATLAS_CONTROL_FAIL': '음성 대조 설정에서 우리 파이프라인이 격차를 만들어 냄. 학습 모델 쪽으로 기운 파이프라인 — 결론 없음.',
    'ATLAS_NO_STABLE_GAP': '후보 중 두 기간 모두 격차가 유지된 설정이 없음. GIFT-Eval에서 격차 우선 경로 종료. '
                           '다음은 사전학습 분포 밖 데이터로 Stage 1을 다시 하는 것.',
    'ATLAS_STABLE_GAP': '하나 이상 설정에서 두 기간 모두 격차 유지. 위치 분석과 난이도 가설 문서까지 작성하고 멈춤.',
}


def build():
    audit = ga.read_json(ga.OUT/'PREMISE_AUDIT.json')
    e0 = ga.read_json(ga.OUT/'E0_REPORT.json')
    stab = rows('STABILITY.csv')
    gaps = rows('GAP_TABLE.csv')
    seal = ga.read_json(ga.OUT/'SEAL.json') if (ga.OUT/'SEAL.json').exists() else {}
    ver = ga.read_json(ga.OUT/'VERIFY_RECOMPUTE.json') if (ga.OUT/'VERIFY_RECOMPUTE.json').exists() else {}
    led = ga.read_json(ga.CACHE/'fit_ledger.json') if (ga.CACHE/'fit_ledger.json').exists() else {}
    token, why = verdict(stab) if stab else ('STOP_DEBUG', 'STABILITY.csv 가 없다')

    L = ['# 격차 지도 Stage 2 — 보고서', '']
    # 1. STABILITY first (user request), then the token
    body = []
    for r in stab:
        body.append([r['id'], r['config'], r['role'],
                     fmt(r['G_P1'], 4) if r['G_P1'] != '' else '—',
                     r['ci_P1'] or '—',
                     fmt(r['G_P2'], 4) if r['G_P2'] != '' else '—',
                     r['ci_P2'] or '—',
                     r['ach_P1'] or '—', r['ach_P2'] or '—', r['verdict']])
    L += ['## 1. STABILITY', '',
          table(['id', '설정', '역할', 'G_P1', 'P1 CI', 'G_P2', 'P2 CI', 'P1 ACH', 'P2 ACH', '판정'], body), '',
          '격차 G = (CRPS_F0 − CRPS_ACH)/CRPS_F0. 양수면 학습이 도움이 된 것이다. '
          f"기준은 두 기간 모두 G ≥ 0.10 이고 부트스트랩 CI 하한 > 0 (봉인 전 고정).", '']

    L += ['## 2. 판정', '', f'`{token}` — {MEANING[token]}', '', f'근거: {why}', '']

    # 3. planner premises that were wrong
    q1b = audit.get('Q1b_context_length', {})
    L += ['## 3. 계획자 전제 중 틀린 것', '',
          f'- [확인] Chronos-2의 기본 context 길이는 {q1b.get("model_default_context")}이다'
          f' (`config.json`의 `chronos_config.context_length`). 계약 §5.2는 가능성 조건을 '
          f'"context 길이 + 5·H"로 쓰는데, 이 값을 그대로 넣으면 가능한 설정이 '
          f'{q1b.get("feasible_strict")} 뿐이라 §4의 후보 목록과 모순된다. 실용값 '
          f'{q1b.get("practical_context_used")}를 써서 {q1b.get("feasible_practical")}를 실행했고, 두 해석을 모두 기록했다.',
          '- [확인] 공식 노트북이 쓰는 `predict_batches_jointly` 인자는 chronos-forecasting 2.3.2에 그 이름으로 없다. '
          '`cross_learning`의 폐기 예정 별칭이고 내부에서 그대로 매핑된다 (`chronos/chronos2/pipeline.py:570-577`). 의미는 같다.',
          '- [확인] 계약 §2가 지목한 `gift_eval` 패키지는 설치하면 numpy를 1.26으로, datasets를 2.17로 내려 이 환경을 깬다. '
          '패키지 대신 공식 로더 파일을 그대로 가져다 썼다 (sha256 '
          f'{audit.get("Q1_official_geometry", {}).get("loader_sha256", "")[:16]}).',
          '- [확인] Toto arm은 실행하지 않았다. 가중치는 공개지만 추론 패키지가 torch를 2.7로, numpy를 1.26으로 되돌린다. '
          '계약 Q6이 허용한 대로 생략하고 기록했다(원래 보고 전용).',
          '- [확인] statsforecast 2.1.1은 공식 래퍼가 분위수 0.5에 대해 만드는 level 0을 거부한다. level 0은 점예측과 같아서 '
          '그 항만 모델 점예측 열에서 가져오도록 고쳤다. 분위수 값은 바뀌지 않는다(E0b가 리더보드와 0.00% 일치).',
          '', f'- [확인] 계약 §5.2 가능성 조건으로 7개 중 3개가 제외됐다: '
          f'{", ".join(audit["Q2_feasibility"]["infeasible"])}. §8이 대체 설정 추가를 금지하므로 그대로 뒀다. '
          'Stage 1에서 "견고"로 꼽힌 후보 셋 중 둘이 여기서 빠진다.', '']

    # 4. E0
    e0a = e0['E0a_official_reproduction']['per_config']; e0b = e0['E0b_seasonal_naive']['per_config']
    body = []
    for cid in e0a:
        a, b = e0a[cid], e0b[cid]
        body.append([cid, a['key'], fmt(a['crps'], 5), fmt(a['leaderboard_crps'], 5),
                     f"{a['relative_difference']*100:.3f}%", fmt(b['crps'], 5),
                     fmt(b['leaderboard_crps'], 5), f"{b['relative_difference']*100:.3f}%"])
    L += ['## 4. E0 — 공식 점수 재현', '',
          table(['id', '설정', 'F0 CRPS', '리더보드', '상대차', 'SNAIVE CRPS', '리더보드', '상대차'], body), '',
          f"E0a 기준 2%, E0b 기준 5%. 최대 상대차는 각각 "
          f"{max(v['relative_difference'] for v in e0a.values())*100:.4f}% 와 "
          f"{max(v['relative_difference'] for v in e0b.values())*100:.4f}% 다. 판정 {e0.get('verdict')}.", '',
          '설정은 공식 노트북 그대로다 — 분위수 0.1–0.9, cross-learning, float32, batch 100, 창 하나씩 예측. '
          '누설 감사(E0c)는 기간별 학습·검증·선택 구간을 인덱스로 적어 `E0_REPORT.json`에 남겼다.', '']

    # 5. our reference versus the leaderboard reference
    import csv as _csv
    # the stage 1 file has two columns called 'dataset'; DictReader keeps the short one, so the
    # configuration key has to be read from column 0 of the raw rows
    stage1 = {}
    with (ga.HERE/'stage1_gift_eval_gap.csv').open(encoding='utf-8') as f:
        raw = list(_csv.reader(f))
    head = raw[0]
    for row in raw[1:]:
        if not row: continue
        stage1[row[0]] = {head[i]: row[i] for i in range(1, len(row))}
    body = []
    for cfg in ga.CONFIGS:
        ours = [r for r in gaps if r['id'] == cfg['id'] and r['period'] == 'P2' and r['arm'].startswith('DL-')]
        if not ours: continue
        best = min(ours, key=lambda r: fnum(r['crps']))
        s1 = stage1.get(cfg['key'])
        f0 = next((fnum(r['crps']) for r in gaps if r['id'] == cfg['id'] and r['period'] == 'P2'
                   and r['arm'] == 'F0'), float('nan'))
        board_best = f0*(1 - fnum(s1['gDL_crps'])) if s1 else float('nan')
        body.append([cfg['id'], cfg['key'], best['arm'], fmt(best['crps'], 5), fmt(board_best, 5),
                     (s1 or {}).get('gDL_argmin_crps', '—'),
                     fmt(fnum(best['crps'])/board_best - 1, 3) if np.isfinite(board_best) else '—'])
    if body:
        L += ['## 5. 우리 참조 대 리더보드 참조 (계약 §9 첫째)', '',
              table(['id', '설정', '우리 최선 DL', 'P2 CRPS', '리더보드 최선 DL', '그 모델', '상대차'], body), '',
              '리더보드 값은 Stage 1 의 gDL_crps 로 되돌린 것이다(F0 CRPS × (1 − gDL)). 상대차가 0에 가까우면 '
              '우리 딥러닝 참조가 리더보드 수준이라는 뜻이고, 크게 양수면 계약 §9 첫째대로 "격차가 없다"가 아니라 '
              '"우리 참조가 약하다"로 읽어야 한다. 하이퍼파라미터가 공개돼 있지 않아 같은 조건은 아니다.', '',
              '- [확인] C1(bitbrains_rnd/5T/medium) 에서 우리 DL 은 리더보드 최선과 거의 같은 자리에 선다. '
              '즉 P2 에서 리더보드가 보고한 격차 자체는 재현된다. 문제는 그 격차가 바로 앞 기간(P1)에서 사라진다는 것이다.', '']
    # 5b. what the validation-only selection cost
    body = []
    for cfg in ga.CONFIGS:
        for period in ('P1', 'P2'):
            rows_cp = [r for r in gaps if r['id'] == cfg['id'] and r['period'] == period]
            if not rows_cp: continue
            f0 = next((fnum(r['crps']) for r in rows_cp if r['arm'] == 'F0'), float('nan'))
            trained = [r for r in rows_cp if r['arm'] not in ('F0', 'SNAIVE')]
            if not trained or not np.isfinite(f0): continue
            ach = next((r for r in trained if str(r['is_ach']) == 'True'), None)
            best = min(trained, key=lambda r: fnum(r['crps']))
            if ach is None: continue
            body.append([cfg['id'], period, ach['arm'], fmt(ga.gap(f0, fnum(ach['crps'])), 4),
                         best['arm'], fmt(ga.gap(f0, fnum(best['crps'])), 4)])
    if body:
        L += ['## 5b. 검증으로만 고른 결과 (평가 창으로 골랐다면)', '',
              table(['id', '기간', '검증이 고른 arm', '그 격차', '평가 창 최선 arm', '그 격차'], body), '',
              '계약 §5.3 은 선택을 검증 블록으로만 하게 한다. 오른쪽 두 열은 만약 평가 창으로 골랐다면 나왔을 값이라 '
              '판정에 쓰지 않는다 — 다만 격차의 상당 부분이 "어느 arm 을 고르느냐"에 달려 있고 그 선택이 '
              '기간을 건너 옮겨가지 않는다는 것을 보여 준다.', '']

    # 6. resources and deviations
    total_s = sum(v.get('seconds', 0) for v in led.values())
    peaks = [v.get('peak_vram_gib') for v in led.values() if v.get('peak_vram_gib')]
    L += ['## 6. 하지 않은 것, 이탈, 자원', '',
          '- 새 PEFT 모듈은 만들지도 학습하지도 않았다(계약 §8). 학습은 §5.3의 LORA·FULL·DL 뿐이다.',
          '- Stage 3(난이도 검증)은 이번 계약 범위가 아니다.',
          f'- 학습 {len(led)}회, 합계 {total_s/60:.0f}분, peak VRAM {max(peaks) if peaks else float("nan"):.2f} GiB.',
          '- [추정] 미해결: N1(창 18 × 500계열, 추론 context 8192)에서 arm 별 평가 소요가 F0 909초 / FULL 636초 / '
          'LORA 2816초였고, LORA 는 두 기간 모두 정확히 2816초였다. OOM 재시도는 0건이고, chronos 가 어댑터를 '
          '로드할 때 이미 merge_and_unload 를 부르므로(`chronos/chronos2/pipeline.py:1177`) LORA 의 모델 구조는 '
          'F0·FULL 과 같다. 원인을 특정하지 못했다. 예측값과 판정에는 영향이 없고 wall-clock 만 늘어난다.',
          '- 이탈: (1) `gift_eval` 패키지 대신 공식 로더 파일 사용 (2) `gluonts`·`neuralforecast` 설치 '
          '(3) statsforecast level 0 등가 처리 (4) TOTO arm 생략 (5) §5.2의 context 길이를 실용값으로 해석.',
          f'- 봉인: SEAL.json {seal.get("date", "—")} (selection digest {str(seal.get("selection_digest", ""))[:12]}). '
          '평가 창 채점은 봉인 뒤에만 했고, 채점 스크립트는 봉인 해시가 다르면 실행을 거부한다.',
          f'- 독립 재계산: 최대 차이 {fmt(ver.get("max_abs_difference"), 12)} '
          f'({ver.get("n_checks", 0)}개 항목, 기준 1e-9, {"PASS" if ver.get("pass_") else "FAIL"}).', '',
          '## 7. 그림', '']
    figdir = ga.OUT/'figures'
    titles = {'fig1_stability_gap': '설정별 P1·P2 격차',
              'fig2_horizon_curve': '예측 단계별 격차 (L1)',
              'fig3_interval_coverage': '80% 구간 포함률과 폭 (L6)'}
    shown = [k for k in titles if (figdir/f'{k}.png').exists()]
    if shown:
        for k in shown: L += [f'![{titles[k]}](figures/{k}.png)', '']
        L += ['캡션 전문은 [CAPTIONS.md](CAPTIONS.md), 그림 안의 모든 좌표는 '
              '[FIGURE_VALUES.csv](FIGURE_VALUES.csv) 에 있다.', '']
    else:
        L += ['그림이 생성되지 않았다. 사유는 `CAPTIONS.md` 또는 실행 로그를 보라.', '']
    L += ['## 8. 산출물', '',
          '[전제 감사](PREMISE_AUDIT.json) · [E0](E0_REPORT.json) · [기간](PERIODS.json) · [봉인](SEAL.json) · '
          '[격차 표](GAP_TABLE.csv) · [안정성](STABILITY.csv) · [재계산](VERIFY_RECOMPUTE.json) · '
          '[위치 분석](LOCALIZATION/) · [난이도 가설](GAP_HYPOTHESES.md)', '']

    (ga.OUT/'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    (ga.OUT/'FINAL_DECISION.md').write_text(
        f'# {token}\n\n{MEANING[token]}\n\n근거: {why}\n\n생성 {time.strftime("%Y-%m-%d %H:%M:%S")}\n',
        encoding='utf-8')
    print('REPORT_KO.md written; verdict', token)
    return token


if __name__ == '__main__':
    build()
