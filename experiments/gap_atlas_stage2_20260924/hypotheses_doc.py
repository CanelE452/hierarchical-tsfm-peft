#!/usr/bin/env python
"""GAP_HYPOTHESES.md (contract 6 S2c): at most three difficulty hypotheses read off the L1-L7 tables.

Each hypothesis gets an evidence table, a measure that could be computed from the training segment
alone, and the strongest competing explanation. No method proposals — the contract forbids them.
Only configurations with a STABLE_GAP verdict are written up; otherwise the file records why not.
"""
from __future__ import annotations
import csv
import numpy as np
import ga

L = ga.OUT/'LOCALIZATION'


def rows(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def table(head, body):
    return '\n'.join(['| '+' | '.join(str(h) for h in head)+' |',
                      '| '+' | '.join(['---']*len(head))+' |'] +
                     ['| '+' | '.join(str(c) for c in r)+' |' for r in body])


def candidates_for(cid, period='P2'):
    """Rank the localization axes by how concentrated the gain is, and return the top three."""
    tag = f'{cid}_{period}'
    out = []

    horizon = rows(L/f'L1_horizon_{tag}.csv')
    if horizon:
        g = np.array([fnum(r['gap']) for r in horizon])
        half = len(g)//2
        early, late = float(np.nanmean(g[:half])), float(np.nanmean(g[half:]))
        out.append(dict(axis='horizon', spread=abs(late-early),
                        detail=f'앞 절반 평균 격차 {early:+.4f}, 뒤 절반 {late:+.4f}',
                        head=['구간', '평균 격차'],
                        body=[['앞 절반 (1..%d)' % half, f'{early:+.4f}'],
                              ['뒤 절반 (%d..%d)' % (half+1, len(g)), f'{late:+.4f}']],
                        measure='학습 구간에서 예측 길이별 조건부 분산 증가율 (기간 밖 자료 불필요)',
                        rival='예측 길이가 길수록 참조 모델이 평균회귀로 이득을 보는 것일 수 있다 — '
                              '같은 길이에서 SNAIVE 격차와 비교해 구분한다.'))

    ter = rows(L/f'L3b_terciles_{tag}.csv')
    if ter:
        best, spread = None, -1
        for attr in sorted({r['attribute'] for r in ter}):
            sub = {r['tercile']: r for r in ter if r['attribute'] == attr}
            if 'top' not in sub or 'bottom' not in sub: continue
            d = abs(fnum(sub['top']['median_gap']) - fnum(sub['bottom']['median_gap']))
            if d > spread: best, spread = (attr, sub), d
        if best:
            attr, sub = best
            out.append(dict(axis=f'series attribute: {attr}', spread=spread,
                            detail=f"상위 삼분위 중앙 격차 {fnum(sub['top']['median_gap']):+.4f}, "
                                   f"하위 {fnum(sub['bottom']['median_gap']):+.4f}",
                            head=['삼분위', '계열 수', '중앙 격차', '총 이득 중 비중'],
                            body=[[t, sub[t]['series'], f"{fnum(sub[t]['median_gap']):+.4f}",
                                   f"{fnum(sub[t]['share_of_total_gain']):.3f}"]
                                  for t in ('bottom', 'middle', 'top') if t in sub],
                            measure=f'{attr} 는 학습 구간만으로 계산된다 (정의는 localize.py)',
                            rival='그 속성이 난이도가 아니라 규모(정규화 분모)와 상관돼 비율 지표를 흔드는 것일 수 있다 — '
                                  '절대 이득 비중과 함께 읽는다.'))

    reg = rows(L/f'L5_regime_{tag}.csv')
    if reg:
        g = {r['regime']: fnum(r['gap']) for r in reg}
        sp = max(g.values()) - min(g.values()) if g else 0
        out.append(dict(axis='regime', spread=sp,
                        detail=', '.join(f'{k} {v:+.4f}' for k, v in g.items()),
                        head=['국면', '점 수', '격차'],
                        body=[[r['regime'], r['points'], f"{fnum(r['gap']):+.4f}"] for r in reg],
                        measure='학습 구간의 버스트 비율·유휴 비율 (직전 하루 95분위수 기준)',
                        rival='국면 자체가 아니라 국면이 몰린 특정 계열의 효과일 수 있다 — L3 계열 분해와 대조한다.'))

    cal = rows(L/f'L6_calibration_{tag}.csv')
    if cal:
        f0 = next((r for r in cal if r['arm'] == 'F0'), None)
        ach = next((r for r in cal if r['arm'] != 'F0'), None)
        if f0 and ach:
            d = abs(fnum(ach['empirical_coverage']) - fnum(f0['empirical_coverage']))
            out.append(dict(axis='interval calibration', spread=d,
                            detail=f"포함률 F0 {fnum(f0['empirical_coverage']):.3f} -> "
                                   f"{ach['arm']} {fnum(ach['empirical_coverage']):.3f} (명목 0.80)",
                            head=['arm', '80% 구간 포함률', '평균 폭 / 평균 |y|'],
                            body=[[r['arm'], f"{fnum(r['empirical_coverage']):.3f}",
                                   f"{fnum(r['mean_width_over_mean_abs_y']):.3f}"] for r in cal],
                            measure='검증 구간의 포함률 편차 (학습 없이 잴 수 있다)',
                            rival='학습 없는 사후 보정(conformal 등)으로 닫히는 문제라면 PEFT 주제가 아니다 — '
                                  '계약 §9 넷째가 지적한 경쟁 설명이고, 다음 단계에서 반드시 대조로 넣어야 한다.'))
    out.sort(key=lambda d: -d['spread'])
    return out[:3]


def build():
    stab = rows(ga.OUT/'STABILITY.csv')
    stable = [r for r in stab if r['verdict'].startswith('STABLE_GAP') and r['role'] == 'candidate']
    lines = ['# 난이도 가설 (격차 지도 Stage 2)', '',
             '계약 §6 S2c. 격차가 가장 몰린 자리 최대 3개를 가설로 적는다. 방법 제안은 쓰지 않는다.', '']
    if not stable:
        lines += ['## 해당 없음', '',
                  '두 기간 모두 격차가 유지된 후보가 없어 위치 분석을 하지 않았다. '
                  'STABILITY.csv 의 판정은 다음과 같다.', '',
                  table(['id', '설정', '역할', '판정', 'G_P1', 'G_P2'],
                        [[r['id'], r['config'], r['role'], r['verdict'], r['G_P1'], r['G_P2']] for r in stab]), '',
                  '가설을 쓰려면 먼저 재현되는 격차가 있어야 한다. 이 단계에서는 없었다.']
        (ga.OUT/'GAP_HYPOTHESES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('GAP_HYPOTHESES.md written (no stable gap)')
        return
    for r in stable:
        lines += [f"## {r['id']} {r['config']}", '',
                  f"P1 격차 {r['G_P1']} {r['ci_P1']}, P2 격차 {r['G_P2']} {r['ci_P2']}. "
                  f"도달 수준 arm: P1 {r['ach_P1']}, P2 {r['ach_P2']}.", '']
        for i, h in enumerate(candidates_for(r['id']), 1):
            lines += [f"### [가설 {i}] {h['axis']}", '', h['detail'], '',
                      table(h['head'], h['body']), '',
                      f"- 학습 구간만으로 잴 지표: {h['measure']}",
                      f"- 가장 강한 경쟁 설명: {h['rival']}", '']
    (ga.OUT/'GAP_HYPOTHESES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('GAP_HYPOTHESES.md written for', [r['id'] for r in stable])


if __name__ == '__main__':
    build()
