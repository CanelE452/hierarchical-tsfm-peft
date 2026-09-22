"""Scoring, cost ledgers, decision and REPORT_KO.md. Reads only saved artifacts; never trains."""
from __future__ import annotations
import csv, math, subprocess, sys, time
import numpy as np
from .common import *
from .common import rc
from .decision import (CAND, FIT_PROCEDURE, TRANSFER_SELECTION, DIAGNOSTIC, cost_of, prefix_costs, timing, decide,
                       sensitivity_total)


def _load():
    panel = read_panel(CACHE/'panel.npz')
    levels = np.asarray(read_json(GAP_RESULTS/'jena'/'MODEL_RECEIPT.json')['levels'])
    fits = {(a, s): read_json(OUT/'fits'/f'{a}_{s}.json') for s in CFG['seeds'] for a in ARMS}
    sel = read_json(OUT/'MODEL_SELECTION.json'); man = read_json(OUT/'PREDICTIONS_MANIFEST.json')
    if not man.get('all_saved_before_scoring') or man['selection_sha256'] != sha(OUT/'MODEL_SELECTION.json'):
        raise Blocked('PREDICTIONS_NOT_SEALED')
    for p, h in man['predictions'].items():
        if sha(REPO/p) != h: raise Blocked('PREDICTION_CHANGED_BEFORE_SCORING')
    return panel, levels, fits, sel


def build_tables():
    panel, levels, fits, sel = _load()
    pb, pe = read_json(OUT/'PREFLIGHT_BASE.json'), read_json(OUT/'PREFLIGHT_EMULATOR.json')
    write_json(OUT/'PREFLIGHT.json', dict(status='PASS' if pb['status'] == pe['status'] == 'PASS' else 'FAIL',
        original_model=pb, emulator_path=pe, note='contract name; merges PREFLIGHT_BASE.json and PREFLIGHT_EMULATOR.json'))
    y = panel.targets('test'); pred = CACHE/'test_predictions'; seeds = CFG['seeds']
    labels = panel.timestamps[panel.origins['test']]//int(CFG['bootstrap_block_days']*86400*10**9)
    per, score_rows, origin_rows, channel_rows = {}, [], [], []

    def add(label, arm, seed, variant, step):
        q = np.load(pred/f'{label}.npz')['q']
        np.testing.assert_array_equal(np.load(pred/f'{label}.npz')['origins'], panel.origins['test'])
        m, p = score(q, y, panel.sigma, levels); per[(arm, seed, variant)] = p
        chk = scalar_primary(q[:2, :2], y[:2, :2], panel.sigma[:2], levels)
        if not np.isclose(chk, score(q[:2, :2], y[:2, :2], panel.sigma[:2], levels)[0]['primary'], rtol=1e-10, atol=1e-12):
            raise Blocked('INDEPENDENT_METRIC_REPLAY_FAILED')
        score_rows.append(dict(arm=arm, seed=seed, variant=variant, step=step, **{k: v for k, v in m.items() if k != 'per_channel'}))
        for i, o in enumerate(panel.origins['test']):
            origin_rows.append(dict(arm=arm, seed=seed, variant=variant, origin=int(o),
                                    timestamp=str(np.datetime64(int(panel.timestamps[o]), 'ns')), primary=float(p[i].mean())))
        for c, v in zip(panel.columns, m['per_channel']):
            channel_rows.append(dict(arm=arm, seed=seed, variant=variant, channel=c, primary=v))
    add('F0', 'F0', 0, 'reference_f0', 0)
    for s in seeds:
        for a in ARMS:
            add(f'{a}_{s}', a, s, 'selected', fits[(a, s)]['selected_step'])
        add(f'E_EMLOC_{s}_NAIVE', 'E_EMLOC', s, 'naive_transfer_same_step', fits[('E_EMLOC', s)]['selected_step'])
    prior = [r for r in csv.DictReader((GAP_RESULTS/'jena'/'SCORES.csv').open(encoding='utf-8'))
             if r['arm'] == SET['head_reference_arm'] and r['variant'] == 'selected']
    for r in prior:
        score_rows.append(dict(arm='HEAD_PRIOR_RUN', seed=int(r['seed']), variant='reference_prior_run', step=int(r['step']),
                               **{k: float(r[k]) for k in ('primary', 'nmae', 'raw_mae', 'nrmse', 'coverage80', 'width80', 'crossing')}))
    write_csv(OUT/'SCORES.csv', score_rows); write_csv(OUT/'ORIGIN_SCORES.csv', origin_rows); write_csv(OUT/'CHANNEL_SCORES.csv', channel_rows)

    def arr(arm, variant='selected'):
        return np.stack([per[(arm, s, variant)] for s in seeds])
    pairs = [(CAND, a) for a in ARMS if a != CAND] + [(a, 'F_FULL') for a in ARMS if a not in ('F_FULL', CAND)]
    effects = []
    for a, b in pairs:
        bs = paired_bootstrap(arr(a), arr(b), CFG['bootstrap_draws'], CFG['bootstrap_block_days'], CFG['bootstrap_seed'], labels=labels)
        effects.append(dict(candidate=a, baseline=b, mean_gain_pct=bs['gain_pct'], ci_low=bs['ci95'][0], ci_high=bs['ci95'][1],
                            seed1_gain_pct=bs['seed_gains'][0], seed2_gain_pct=bs['seed_gains'][1], blocks=bs['blocks']))
    bs = paired_bootstrap(arr('E_EMLOC'), arr('E_EMLOC', 'naive_transfer_same_step'), CFG['bootstrap_draws'],
                          CFG['bootstrap_block_days'], CFG['bootstrap_seed'], labels=labels)
    effects.append(dict(candidate='E_EMLOC', baseline='E_EMLOC_NAIVE_SAME_STEP', mean_gain_pct=bs['gain_pct'], ci_low=bs['ci95'][0],
                        ci_high=bs['ci95'][1], seed1_gain_pct=bs['seed_gains'][0], seed2_gain_pct=bs['seed_gains'][1], blocks=bs['blocks']))
    f0 = per[('F0', 0, 'reference_f0')]
    for a in ARMS:
        bs = paired_bootstrap(arr(a), np.stack([f0, f0]), CFG['bootstrap_draws'], CFG['bootstrap_block_days'], CFG['bootstrap_seed'], labels=labels)
        effects.append(dict(candidate=a, baseline='F0', mean_gain_pct=bs['gain_pct'], ci_low=bs['ci95'][0], ci_high=bs['ci95'][1],
                            seed1_gain_pct=bs['seed_gains'][0], seed2_gain_pct=bs['seed_gains'][1], blocks=bs['blocks']))
    write_csv(OUT/'SEED_EFFECTS.csv', effects)

    test_primary = {(r['arm'], r['seed']): r['primary'] for r in score_rows if r['variant'] == 'selected'}
    resources, components, curves, transfer_rows, ttt = [], [], [], [], []
    target = sel['time_quality_target']['target']
    for (a, s), f in fits.items():
        c = cost_of(f); tm = timing(f); pc = prefix_costs(f)
        resources.append(dict(arm=a, seed=s, selected_step=f['selected_step'], test_primary=test_primary[(a, s)],
            total_s=c['total_s'], preparation_s=c['preparation_s'], fit_procedure_s=c['fit_procedure_s'],
            transfer_selection_s=c['transfer_selection_s'], diagnostic_excluded_s=c['diagnostic_s'],
            total_s_sensitivity_used_train_cache_only=sensitivity_total(f, len(panel.origins['train'])),
            conservative_reload_s=f['parts'].get('emulator_load', 0.) + f['parts'].get('full_model_load_transfer', 0.) if a in E_ARMS else 0.,
            unique_train_origins=f['unique_train_origins'],
            main_updates_s=f['parts'].get('main_updates', 0.), full_process_wall_s=f['full_process_wall_s'], **tm,
            train_peak_allocated_mib=f['train_peak']['peak_allocated_bytes']/2**20,
            train_peak_reserved_mib=f['train_peak']['peak_reserved_bytes']/2**20,
            trainable_parameters=f['trainable_parameters'], lora_modules=f['lora_modules'], adapter_bytes=f['adapter_bytes'],
            rss_mib=f['rss_bytes']/2**20))
        for k, v in f['charged_preparation'].items():
            components.append(dict(ledger='cold_start_charged', arm=a, seed=s, group='preparation', component=k, seconds=v))
        for k, v in f['parts'].items():
            grp = 'fit_procedure' if k in FIT_PROCEDURE else 'transfer_selection' if k in TRANSFER_SELECTION else 'diagnostic_not_charged'
            components.append(dict(ledger='cold_start_charged' if grp != 'diagnostic_not_charged' else 'diagnostic',
                                   arm=a, seed=s, group=grp, component=k, seconds=v))
        for r in f['curves']:
            curves.append(dict(arm=a, seed=s, step=r['step'], val_primary=r['val_primary'], val_nmae=r['val_nmae'],
                               emulator_val_primary=r.get('emulator_val_primary'), naive_val_primary=r.get('naive_val_primary'),
                               prefix_cost_s=pc[r['step']], selected=r['step'] == f['selected_step']))
            if a in E_ARMS:
                transfer_rows.append(dict(arm=a, seed=s, step=r['step'], emulator_training_val_primary=r['emulator_val_primary'],
                    transferred_val_primary=r['val_primary'], transfer_gap=r['val_primary'] - r['emulator_val_primary'],
                    train_vs_actual_sigma_rmse=r['train_vs_actual_sigma_rmse'], naive_val_primary=r.get('naive_val_primary')))
        hit = next((r for r in sorted(f['curves'], key=lambda z: z['step']) if r['val_primary'] <= target), None)
        ttt.append(dict(arm=a, seed=s, target=target, status='NOT_REACHED' if hit is None else
                        ('NO_ADAPTATION_NEEDED_AT_THIS_TARGET' if hit['step'] == 0 else 'REACHED_AT_RECORDED_CHECKPOINT'),
                        step=None if hit is None else hit['step'], prefix_cost_upper_s=None if hit is None else pc[hit['step']]))
    for stage in ('prepare', 'compress', 'calibrate', 'caches'):
        for k, v in read_json(CACHE/'stages'/f'{stage}.json')['parts'].items():
            components.append(dict(ledger='experiment_actual_once', arm='SHARED', seed=0, group=stage, component=k, seconds=v))
    for (a, s), f in fits.items():
        for k, v in f['parts'].items():
            components.append(dict(ledger='experiment_actual_once', arm=a, seed=s, group='fit', component=k, seconds=v))
    for k, v in read_json(OUT/'PREDICTIONS_MANIFEST.json')['evaluation_seconds'].items():
        components.append(dict(ledger='evaluation_not_adaptation', arm='ALL', seed=0, group='test', component=k, seconds=v))
    write_csv(OUT/'RESOURCES.csv', resources); write_csv(OUT/'COST_COMPONENTS.csv', components)
    write_csv(OUT/'CURVES.csv', curves); write_csv(OUT/'TRANSFER_GAP.csv', transfer_rows); write_csv(OUT/'TIME_TO_TARGET.csv', ttt)
    decision = decide(fits, test_primary, resources, read_json(OUT/'PREFLIGHT_EMULATOR.json')['checks']['emloc'])
    decision['created_utc'] = time.time()
    write_json(OUT/'DECISION.json', decision)
    return decision


def run_figures():
    script = HERE/'figures.py'
    if not script.exists(): raise Blocked('FIGURE_SCRIPT_MISSING')
    rc_ = subprocess.run([sys.executable, str(script)], cwd=REPO).returncode
    need = ['fig1_transferred_quality', 'fig2_total_cost_frontier', 'fig3_cost_breakdown', 'fig4_transfer_mismatch', 'fig5_response_ablation']
    missing = [n for n in need for ext in ('png', 'svg', 'pdf') if not (OUT/'figures'/f'{n}.{ext}').exists()]
    if rc_ or missing or not (OUT/'FIGURE_VALUES.csv').exists() or not (OUT/'CAPTIONS.md').exists():
        raise Blocked(f'FIGURES_INCOMPLETE exit={rc_} missing={missing}')


def _csv(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8')))


def _f(x, nd=4):
    return '—' if x is None or x == '' else f'{float(x):.{nd}f}'


def _pct(x, nd=2):
    return '—' if x is None else f'{float(x):+.{nd}f}%'


def table(rows, cols, heads=None):
    heads = heads or cols
    return '\n'.join(['| '+' | '.join(heads)+' |', '| '+' | '.join(['---']*len(cols))+' |'] +
                     ['| '+' | '.join(str(r.get(c, '')) for c in cols)+' |' for r in rows])


LABEL = {'F_FULL': 'F_FULL (원래 모델 전체 LoRA+)', 'F_TOP3': 'F_TOP3 (마지막 3 block + 출력 LoRA+)',
         'E_EMLOC': 'E_EMLOC (대리모델 학습 + EMLoC 보정 이식)', 'E_DELTA': 'E_DELTA (출발 예측 보존, 보정 없음)',
         'E_VALUE_DELTA': 'E_VALUE_DELTA (출력 값 보정 16회)', 'E_RESPONSE_DELTA': 'E_RESPONSE_DELTA (값 + 반응 보정 16회, 후보)'}


def write_markdown(d):
    seeds = CFG['seeds']; b = d['base_outcome']
    scores = _csv('SCORES.csv'); eff = _csv('SEED_EFFECTS.csv'); res = _csv('RESOURCES.csv')
    gap_rows = _csv('TRANSFER_GAP.csv'); cal = _csv('CALIBRATION_RESPONSE.csv'); ttt = _csv('TIME_TO_TARGET.csv')
    sel = read_json(OUT/'MODEL_SELECTION.json'); emu = read_json(OUT/'EMULATOR_MANIFEST.json')
    calm = read_json(OUT/'CALIBRATION_MANIFEST.json'); pre = read_json(OUT/'PREFLIGHT_EMULATOR.json')['checks']
    ver_updates = sum(1 for _ in (OUT/'UPDATE_LEDGER.jsonl').open()) // 2
    E = {(r['candidate'], r['baseline']): r for r in eff}
    sc = {(r['arm'], int(r['seed'])): r for r in scores if r['variant'] == 'selected'}
    rs = {(r['arm'], int(r['seed'])): r for r in res}
    mean = lambda arm, key, src: float(np.mean([float(src[(arm, s)][key]) for s in seeds]))
    status = d['status']
    # 1. five-sentence summary
    q_loss = b['relative_quality_loss_pct']; saved = b['total_time_saved_pct']; rgain = b['response_gain_pct']
    dom = ', '.join(d['base_outcome']['dominating_controls']) or '없음'
    s1 = ('예상은 대리모델에서 LoRA를 학습해 원래 Chronos-2로 옮기면 원래 품질을 유지하면서 준비비를 포함한 적응시간이 줄고, '
          '반응 보정이 VALUE 보정보다 이식 후 오차를 더 낮출 것이라는 것이었다.')
    s2 = f'이번 판정은 `{status}`다' + (f' (원판정 `{b["status"]}`; {"; ".join(d["reasons"])})' if d['reasons'] else '') + '.'
    s3 = (f'후보 E_RESPONSE_DELTA의 원래 모델 TEST 오차는 F_FULL보다 평균 {q_loss:+.2f}% '
          f'({"허용 0.5% 이내" if b["quality_ok"] else "허용 0.5% 초과"})다.')
    s4 = (f'준비비와 128회 선택 절차 전체를 포함한 총시간은 F_FULL 대비 {saved:+.1f}% 절감이다(음수는 더 오래 걸림). '
          f'요구 기준은 20% 절감이다.')
    std_ok = ', '.join(d['standard_controls_meeting_goal']) or '없음'
    s5 = (f'E_VALUE_DELTA 대비 오차 개선은 {rgain:+.2f}%(기준 0.3%, 두 seed 같은 방향 필요)이고, '
          f'후보보다 오차와 총시간이 모두 같거나 낮은 대조군은 {dom}, 품질·시간 목표를 단독으로 충족한 대조군은 {std_ok}이다.')
    L = [f'# Jena: 예측 변화 보존 대리모델 PEFT — 제한된 Go/No-go', '',
         '[확인] 이 보고서는 저장된 예측·선택·비용 기록에서 자동 작성됐다. 모든 품질 수치는 원래 Chronos-2에 LoRA를 이식한 뒤의 값이다.', '',
         '## 1. 예상했던 효과와 이번 판정', '', ' '.join([s1, s2, s3, s4, s5]), '']
    # 2. setting
    rows = []
    for a in ARMS:
        r0 = rs[(a, seeds[0])]
        rows.append(dict(arm=LABEL[a], modules=r0['lora_modules'], params=r0['trainable_parameters'],
                         emulator=EMULATOR_OF.get(a, '원래 모델'), adapter_kib=f'{int(r0["adapter_bytes"])/1024:.0f}'))
    L += ['## 2. 고정 문제·정보 권한·비교군', '',
          f'- 자료: Jena 2024, 10분 간격 21계열, 과거 1152(8일) → 미래 144(1일). 합법 원점 TRAIN/VAL/TEST = '
          f'{read_json(OUT/"DATA_MANIFEST.json")["origins"]}. 이전 감사와 값·scale·원점이 모두 일치했다.',
          '- 원래 모델: amazon/chronos-2 revision 29ec3766…, FP32 master + BF16 autocast, gradient checkpointing.',
          f'- 대리모델: encoder Linear {len(emu["modules"])}개를 activation-aware SVD(rank 192/307)로 분해. '
          f'텐서 용량 {emu["emulator_tensor_bytes"]/2**20:.0f} MiB (원래 {emu["full_model_parameter_bytes"]/2**20:.0f} MiB).',
          f'- 공통: seed {seeds}, LoRA rank 8 / alpha 16, LoRA+ (A 1e-5, B 1.6e-4), fit당 128 update, 체크포인트 0/16/32/64/128. '
          f'실제 main update {ver_updates}회, calibration 32회.',
          '- 선택: 모든 체크포인트를 원래 모델로 이식한 VAL 최소. 대리모델 점수는 선택에 쓰지 않았다.', '',
          table(rows, ['arm', 'modules', 'params', 'emulator', 'adapter_kib'], ['군', 'LoRA 모듈', '학습 파라미터', '학습 모델', '배포 adapter (KiB)']), '']
    # 3. quality
    def vs_full(a):
        return '—' if a == 'F_FULL' else _pct(E[(a, 'F_FULL')]['mean_gain_pct'])
    q = []
    for a in ARMS:
        q.append(dict(arm=a, s1=_f(sc[(a, seeds[0])]['primary']), s2=_f(sc[(a, seeds[1])]['primary']),
                      mean=_f(mean(a, 'primary', sc)), nmae=_f(mean(a, 'nmae', sc)), cov=_f(mean(a, 'coverage80', sc), 3),
                      cross=_f(mean(a, 'crossing', sc), 4), steps='/'.join(sc[(a, s)]['step'] for s in seeds), vs_full=vs_full(a)))
    ref = [r for r in scores if r['variant'] in ('reference_f0', 'reference_prior_run')]
    L += ['## 3. 원래 TSFM에 이식한 실제 예측 품질', '',
          '주지표는 원단위 분위수 예측을 TRAIN 표준편차로 나눈 평균 2-pinball이며 낮을수록 좋다. `F_FULL 대비`의 양수는 F_FULL보다 오차가 낮다는 뜻이다.', '',
          table(q, ['arm', 's1', 's2', 'mean', 'vs_full', 'nmae', 'cov', 'cross', 'steps'],
                ['군', f'seed {seeds[0]}', f'seed {seeds[1]}', '평균', 'F_FULL 대비', 'nMAE', 'coverage80', 'crossing', '선택 step']), '',
          '참조(같은 TEST·같은 지표, 판정에는 쓰지 않음): ' + '; '.join(
              f'{r["arm"]} seed {r["seed"]} = {float(r["primary"]):.4f}' for r in ref) +
          '. HEAD_PRIOR_RUN은 이전 실험(다른 seed·다른 실행)의 출력 보정 결과다.', '',
          '![원래 모델로 이식한 TEST 점수](figures/fig1_transferred_quality.png)', '',
          table([dict(c=r['candidate'], b=r['baseline'], m=_pct(r['mean_gain_pct']), ci=f'[{float(r["ci_low"]):+.2f}, {float(r["ci_high"]):+.2f}]',
                      s1=_pct(r['seed1_gain_pct']), s2=_pct(r['seed2_gain_pct'])) for r in eff],
                ['c', 'b', 'm', 'ci', 's1', 's2'], ['비교 군', '기준 군', '평균 오차 감소', '7일 block bootstrap 95%', f'seed {seeds[0]}', f'seed {seeds[1]}']), '']
    # 4. cost
    crow = []
    for a in ARMS:
        crow.append(dict(arm=a, total=f'{mean(a, "total_s", rs):.1f}', sens=f'{mean(a, "total_s_sensitivity_used_train_cache_only", rs):.1f}',
                         t1=f'{float(rs[(a, seeds[0])]["total_s"]):.1f}',
                         t2=f'{float(rs[(a, seeds[1])]["total_s"]):.1f}', prep=f'{mean(a, "preparation_s", rs):.1f}',
                         proc=f'{mean(a, "fit_procedure_s", rs):.1f}', tr=f'{mean(a, "transfer_selection_s", rs):.1f}',
                         upd=f'{mean(a, "main_updates_s", rs):.1f}', tput=f'{mean(a, "throughput_steps_per_s", rs):.2f}',
                         cv='/'.join(f'{float(rs[(a, s)]["active_step_cv"]):.2f}' for s in seeds),
                         mem=f'{max(float(rs[(a, s)]["train_peak_allocated_mib"]) for s in seeds):.0f}'))
    par = [dict(arm=p['arm'], e=_f(p['mean_test_primary']), t=f'{p["mean_total_s"]:.1f}', po='예' if p['pareto_optimal'] else '아니오',
                by=', '.join(p['dominated_by']) or '—') for p in d['pareto']]
    L += ['## 4. 준비비 포함 총시간과 비용 성분', '',
          '총시간은 한 사용자가 그 군을 처음부터 적용할 때의 비용이다. 자료 준비, 모델 로드, activation/SVD, probe 교사 예측·gamma 보정, F0/E0 cache, '
          '128 update 전부, 다섯 체크포인트의 이식·보정·원래 모델 VAL, 최종 선택·export를 포함한다. 최선 체크포인트를 미리 안 것처럼 계산하지 않았다. '
          '공유 준비물도 군마다 전체를 청구했다. 진단 전용 계산(대리모델 VAL, 보정 전 EMLoC 이식)은 뺐다.', '',
          table(crow, ['arm', 'total', 't1', 't2', 'prep', 'proc', 'tr', 'upd', 'tput', 'cv', 'mem', 'sens'],
                ['군', '총시간 평균(s)', f'seed {seeds[0]}', f'seed {seeds[1]}', '준비(s)', '학습·VAL 절차(s)', '이식·선택(s)', '128 update(s)',
                 'update/s', 'step CV', '학습 peak (MiB)', '민감도 총시간(s)']), '',
          '민감도 총시간은 보조값이다. 계약 청구액에서 VAL cache(청구하지 않는 대리모델 진단에만 쓰임)와, 학습 schedule이 쓰지 않은 TRAIN 원점 몫의 cache를 뺐다. '
          'TRAIN 몫은 원점 수 비례로 추정했다 [추정]. 판정은 계약 청구액으로만 한다. '
          f'E 군의 fit 안 대리모델 재로드와 이식용 원래 모델 재로드(보수적 이중 청구)는 평균 '
          f'{np.mean([float(r["conservative_reload_s"]) for r in res if r["arm"] in E_ARMS]):.1f}초다.', '',
          '![준비비 포함 총시간과 원래 모델 TEST 오차](figures/fig2_total_cost_frontier.png)', '',
          '![비용 성분](figures/fig3_cost_breakdown.png)', '',
          table(par, ['arm', 'e', 't', 'po', 'by'], ['군', '평균 TEST 오차', '평균 총시간(s)', 'Pareto 최적', '지배하는 군']), '',
          f'보조 목표 도달(사후 목표 = 1.005 × F_FULL 두 seed 최선 VAL 평균 = {sel["time_quality_target"]["target"]:.5f}; F0 VAL '
          f'{sel["time_quality_target"]["f0_val"]:.5f}, F0 달성 여부 {sel["time_quality_target"]["f0_meets_target"]}). 첫 도달 비용은 기록된 체크포인트의 상한이며 학습 종료 규칙이 아니다.', '',
          table([dict(a=r['arm'], s=r['seed'], st=r['status'], k=r['step'] or '—', c=_f(r['prefix_cost_upper_s'], 1)) for r in ttt],
                ['a', 's', 'st', 'k', 'c'], ['군', 'seed', '상태', 'step', '도달 비용 상한(s)']), '']
    full_e, full_t = mean('F_FULL', 'primary', sc), mean('F_FULL', 'total_s', rs)
    std = []
    for a in ('F_TOP3', 'E_EMLOC'):
        e_, t_ = mean(a, 'primary', sc), mean(a, 'total_s', rs)
        ok_q = e_ <= full_e*(1+CFG['quality_relative_margin']); ok_t = t_ <= full_t*(1-CFG['minimum_total_time_saving'])
        std.append(f'- {a}: F_FULL 대비 오차 {100*(e_/full_e-1):+.2f}%, 총시간 {100*(1-t_/full_t):+.1f}% 절감. '
                   f'품질 기준 {"충족" if ok_q else "미충족"}, 시간 기준 {"충족" if ok_t else "미충족"}.')
    L += ['### EMLoC와 부분 층 LoRA로 충분한가', '', *std,
          f'- 사전 판정표: 후보가 GO가 아닐 때 대조군이 F_FULL 대비 품질 0.5% 이내(평균과 두 seed)와 총시간 20% 이상 절감(평균, 두 seed 모두 더 빠름)을 '
          f'함께 충족하면 `GO_STANDARD_ONLY`다. 충족한 대조군: {std_ok}. 후보를 지배한 대조군: {dom}.',
          f'- EMLoC 대조 상태: `{d["emloc_baseline"]}`. 공식 EMLoC는 attention과 MLP 모두에 LoRA를 두고 보정하지만, 이 계약의 LoRA 지도는 '
          'attention q/k/v/o와 출력 head뿐이다. 그래서 압축된 MLP wi/wo의 오차는 EMLoC 보정이 고칠 수 없다(대리모델 3 delta 군은 출발 예측 보존으로 0차 오차를 없앤다).', '']
    # 5. transfer gap
    gsel = []
    for a in E_ARMS:
        for s in seeds:
            st = int(sc[(a, s)]['step']); r = next(x for x in gap_rows if x['arm'] == a and int(x['seed']) == s and int(x['step']) == st)
            gsel.append(dict(arm=a, seed=s, step=st, emu=_f(r['emulator_training_val_primary']), act=_f(r['transferred_val_primary']),
                             gap=_f(r['transfer_gap']), rmse=_f(r['train_vs_actual_sigma_rmse']), naive=_f(r['naive_val_primary'])))
    naive = E[('E_EMLOC', 'E_EMLOC_NAIVE_SAME_STEP')]
    L += ['## 5. 대리모델 점수와 실제 이식 후 점수의 차이', '',
          '대리모델 쪽 점수는 학습에 쓰인 예측(E_EMLOC는 대리모델 예측, delta 계열은 F0 + 대리모델 변화분)의 VAL 오차다. 실제 점수는 같은 LoRA를 원래 모델에 옮긴 VAL 오차다. '
          '둘의 차이는 진단이며 선택에 쓰지 않았다.', '',
          table(gsel, ['arm', 'seed', 'step', 'emu', 'act', 'gap', 'rmse', 'naive'],
                ['군', 'seed', '선택 step', '대리모델 쪽 VAL', '이식 후 VAL', '차이(이식−대리)', '예측 차이 RMSE/σ', '보정 전 EMLoC VAL']), '',
          f'EMLoC 보정의 TEST 효과(같은 step, 보정 전 대비 오차 감소): {_pct(naive["mean_gain_pct"])} '
          f'(seed {_pct(naive["seed1_gain_pct"])} / {_pct(naive["seed2_gain_pct"])}).', '',
          '![대리모델 학습용 예측과 이식 후 예측](figures/fig4_transfer_mismatch.png)', '']
    # 6. response vs value
    def cmean(variant, label, key):
        v = [float(r[key]) for r in cal if r['variant'] == variant and r['label'] == label]
        return float(np.mean(v)) if v else None
    crow2 = [dict(v=v, lv=_f(cmean(v, 'heldout', 'L_value')), lr=_f(cmean(v, 'heldout', 'L_response')),
                  lvc=_f(cmean(v, 'calibration', 'L_value')), lrc=_f(cmean(v, 'calibration', 'L_response')),
                  tgt=f'{cmean(v, "heldout", "response_target_mse"):.3g}') for v in ('svd', 'value', 'response')]
    rv = E[(CAND, 'E_VALUE_DELTA')]; rd = E[(CAND, 'E_DELTA')]
    L += ['## 6. VALUE 보정 대비 RESPONSE 보정의 추가 효과', '',
          '보정 대리 지표(낮을수록 원래 모델에 가까움)는 L_value = 출력 오차 / 보정 전 출력 오차, L_response = 반응 오차 / 반응 크기다. '
          '검사용 8원점은 보정에 쓰지 않았고 probe seed도 다르다.', '',
          table(crow2, ['v', 'lvc', 'lrc', 'lv', 'lr', 'tgt'],
                ['대리모델', 'L_value (보정 원점)', 'L_response (보정 원점)', 'L_value (검사 원점)', 'L_response (검사 원점)', '반응 크기 MSE/σ² (검사)']), '',
          f'gamma 통계: VALUE {calm["stats"]["VALUE"]["gamma_mean_abs"]:.4f} (경계 도달 {100*calm["stats"]["VALUE"]["fraction_at_bound"]:.1f}%), '
          f'RESPONSE {calm["stats"]["RESPONSE"]["gamma_mean_abs"]:.4f} (경계 도달 {100*calm["stats"]["RESPONSE"]["fraction_at_bound"]:.1f}%), 평균 |gamma|.', '',
          f'실제 원래 모델 TEST에서 E_RESPONSE_DELTA의 오차 감소: E_VALUE_DELTA 대비 {_pct(rv["mean_gain_pct"])} '
          f'(seed {_pct(rv["seed1_gain_pct"])} / {_pct(rv["seed2_gain_pct"])}, CI [{float(rv["ci_low"]):+.2f}, {float(rv["ci_high"]):+.2f}]), '
          f'E_DELTA 대비 {_pct(rd["mean_gain_pct"])}. 대리 지표의 개선은 예측 성공의 증거가 아니다.', '',
          ('불일치: 보정 뒤 L_value가 1보다 크다(VALUE 검사 원점 ' + _f(cmean('value', 'heldout', 'L_value'), 3) + ', RESPONSE ' + _f(cmean('response', 'heldout', 'L_value'), 3) +
           '). 즉 계약의 고정 recipe(AdamW lr 1e-2, 원점 하나씩 16 update)로 한 gamma 보정이 대리모델 출력을 원래 모델에서 오히려 더 멀어지게 했다. '
           '반응 오차 L_response도 보정 전보다 ' + ('나빠졌다' if (cmean('response', 'heldout', 'L_response') or 0) > (cmean('svd', 'heldout', 'L_response') or 0) else '줄었다') +
           '. 계약에 따라 lr과 update 수는 바꾸지 않았다. 원인으로는 Adam 초기 step이 25,800개 gamma를 모두 약 ±lr씩 움직인 과도 이동을 추정한다 [추정].')
          if (cmean('value', 'heldout', 'L_value') or 0) > 1 else '보정 뒤 검사 원점 L_value가 1 이하로 줄었다.', '',
          '1e-4 probe의 FP32 반응 크기(MSE/σ² ≈ 3×10⁻⁹)는 L_response 분모의 ε(1e-8)보다 작다. 그래서 그 probe가 쓰인 보정 step(원점 j mod 4 ∈ {0, 2})에서는 반응 항이 약해진다. ε은 계약 고정값이다.', '',
          '![VALUE 대비 RESPONSE: 반응 대리 지표와 실제 TEST](figures/fig5_response_ablation.png)', '']
    # 7. decision and limits
    crit = [dict(c='1) 후보 품질: F_FULL 대비 평균·두 seed 모두 0.5% 이내 손해', v=str(b['quality_ok'] and not d['quality_failure'])),
            dict(c='2) 총시간: 평균 20% 이상 절감, 두 seed 모두 더 빠름', v=str(b['total_time_ok'])),
            dict(c='3) E_VALUE_DELTA보다 평균 0.3% 이상 낮은 오차, 두 seed 같은 방향', v=str(b['response_added_value'])),
            dict(c='4) 더 정확하면서 비용도 낮은 대조군 없음', v=str(not b['dominating_controls'])),
            dict(c='(참고) 표준 대조군의 단독 목표 충족 → GO_STANDARD_ONLY', v=std_ok),
            dict(c='5) 실모델·이식·자료 무결성과 시간 신뢰도', v=str(not d['timing_hold']) + (' (TIMING_HOLD)' if d['timing_hold'] else ''))]
    lim = [f'- 한 자료(Jena), 한 모델, 두 main seed. 두 seed는 같은 압축·같은 calibration을 공유하므로 calibration 모집단 반복이 아니다.',
           '- 고정 LR·rank·압축비·128 update의 짧은 적응 설정이다. HPO나 수렴 최적 학습률이 아니다.',
           f'- 예산 경계 표시(128에서 선택되고 마지막 구간 VAL 개선 0.2% 초과): {d["limited_budget"]}.',
           f'- 시간 신뢰도: step CV 0.2 초과 {d["noisy_fits"] or "없음"}, seed 간 throughput 차이 {({k: round(v, 3) for k, v in d["seed_throughput_gap"].items()})}.',
           f'- EMLoC 대조는 Chronos로 옮긴 핵심 수식이다(`{d["emloc_baseline"]}`). 공식 함수와 FP32 parity를 확인했지만 원논문 VLM 결과의 재현은 아니다. 보정은 FP32로 체크포인트마다 적용했다.',
           '- 반응 보정 forward(교사와 대리모델)는 FP32로 실행했다. BF16에서는 1e-4/1e-3 probe의 반응이 반올림 잡음으로 측정됐기 때문이다(사전검사 기록). 본학습은 계약대로 BF16이다.',
           '- 이 계약의 판정표에서 EMLoC 대조는 항상 LIMITED_BASELINE(MLP 보정 불가)이므로, 후보가 받을 수 있는 최선의 판정은 HOLD_NO_AUTO_RESCUE다(산술 GO도 보류). 산술 원판정은 base_outcome에 따로 남겼다.',
           '- 두 seed 간 차이가 0.3% 기준과 같은 크기일 수 있고(이전 LoRA+ 두 seed TEST 차이 0.445%), gamma는 lr 1e-2 × 16 update로 [−log 2, log 2] 경계에 거의 닿지 않을 것으로 결과 전에 예상했다.',
           '- 이 PC는 Windows 데스크톱 GPU라 다른 앱의 간헐적 GPU 사용이 시간 잡음을 만들 수 있다. 잡음은 재실행으로 숨기지 않았다.',
           '- Jena TEST는 이전 개발에 쓰인 자료다. 독립 확증이 아니다.',
           '- 최종 추론은 모든 군이 원래 모델 + LoRA이므로 추론 가속을 주장하지 않는다.',
           '- 이 비교가 끝난 뒤 추가 진단, 다른 자료(ETTm2 등), 후속 모델을 자동으로 실행하지 않았다.']
    L += ['## 7. 판정과 한계', '', f'판정: `{status}`', '',
          table(crit, ['c', 'v'], ['사전 기준', '충족']), '',
          f'원판정 `{b["status"]}`: 품질 손해 {q_loss:+.3f}%, 총시간 절감 {saved:+.2f}%, VALUE 대비 {rgain:+.3f}%, 지배 대조군 {dom}.'
          + (f' 보정 규칙 적용: {"; ".join(d["reasons"])}.' if d['reasons'] else ''), '',
          '[판단] 판정 기준(0.5% / 20% / 0.3%)은 결과 전에 고정한 연구 투자 선별값이다. 통계적 동등성이나 논문 PASS가 아니다. '
          'NO_GO여도 모든 대리모델 PEFT가 불가능하다는 뜻이 아니고, 현재 구현의 한계로 남긴다.', '', *lim, '',
          '## 8. 재검산 자료', '',
          '[자료](DATA_MANIFEST.json) · [모듈 지도](MODULE_MAP.json) · [대리모델](EMULATOR_MANIFEST.json) · [보정](CALIBRATION_MANIFEST.json) · '
          '[probe](PROBE_MANIFEST.json) · [사전검사](PREFLIGHT.json) · [사전검사(원래)](PREFLIGHT_BASE.json) · [사전검사(대리)](PREFLIGHT_EMULATOR.json) · [비용 기준](COST_BASIS.json)', '',
          '[선택 봉인](MODEL_SELECTION.json) · [예측 hash](PREDICTIONS_MANIFEST.json) · [점수](SCORES.csv) · [seed 효과](SEED_EFFECTS.csv) · '
          '[자원](RESOURCES.csv) · [비용 성분](COST_COMPONENTS.csv) · [이식 차이](TRANSFER_GAP.csv) · [보정 반응](CALIBRATION_RESPONSE.csv) · '
          '[학습곡선](CURVES.csv) · [판정](DECISION.json) · [검산](VERIFICATION.json) · [그림 값](FIGURE_VALUES.csv) · [캡션](CAPTIONS.md)', '',
          '원자료, 모델·대리모델 가중치, LoRA 체크포인트, 예측 배열은 로컬 `.cache`에만 있다. GitHub의 표만으로 전체를 재생할 수 있다고 주장하지 않는다.', '']
    (OUT/'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    (OUT/'FINAL_DECISION.md').write_text(f'# {status}\n\n' + json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build():
    decision = build_tables()
    run_figures()
    write_markdown(decision)
    return decision
