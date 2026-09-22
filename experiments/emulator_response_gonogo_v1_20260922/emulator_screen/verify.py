"""Independent CPU verification of saved artifacts. Writes VERIFICATION.json; PASS only when every check holds."""
from __future__ import annotations
import csv, math, re, time
import numpy as np
import torch
from .common import *
from .common import rc
from . import decision


def _rows(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8')))


def independent_status(srows, resources, emloc):
    """Second implementation of the pre-registered table from CSV values only (not calling decision/reference_core)."""
    seeds = CFG['seeds']; m = CFG['quality_relative_margin']; sv = CFG['minimum_total_time_saving']; mg = CFG['minimum_response_gain_over_value']
    err = {(a, s): float(srows[(a, str(s), 'selected')]['primary']) for a in ARMS for s in seeds}
    tot = {(r['arm'], int(r['seed'])): float(r['total_s']) for r in resources}
    me = lambda a: sum(err[(a, s)] for s in seeds)/len(seeds); mt = lambda a: sum(tot[(a, s)] for s in seeds)/len(seeds)
    F, C = 'F_FULL', CFG['candidate']; controls = ['F_TOP3', 'E_EMLOC', 'E_DELTA', 'E_VALUE_DELTA']
    def meets(a):
        return (me(a) <= me(F)*(1+m) and mt(a) <= mt(F)*(1-sv) and
                all(err[(a, s)] <= (1+m)*err[(F, s)] and tot[(a, s)] < tot[(F, s)] for s in seeds))
    keep = me(C) <= me(F)*(1+m); save = mt(C) <= mt(F)*(1-sv)
    both_q = all(err[(C, s)] <= (1+m)*err[(F, s)] for s in seeds); both_t = all(tot[(C, s)] < tot[(F, s)] for s in seeds)
    ev = me('E_VALUE_DELTA'); added = (ev - me(C))/ev >= mg and all(err[(C, s)] < err[('E_VALUE_DELTA', s)] for s in seeds)
    dom = [a for a in controls if me(a) <= me(C) and mt(a) <= mt(C)]
    if dom: base = 'GO_STANDARD_ONLY' if any(me(a) <= me(F)*(1+m) and mt(a) <= mt(F)*(1-sv) for a in dom) else 'NO_GO_CURRENT_IMPLEMENTATION'
    elif keep and save and added and both_q and both_t: base = 'GO_METHOD_SCREEN'
    elif not keep or not save: base = 'NO_GO_CURRENT_IMPLEMENTATION'
    else: base = 'HOLD_NO_AUTO_RESCUE'
    parity = emloc['official_vs_port_scaling1_fp32_rel'] <= SET['official_emloc_fp32_rel_tol'] and emloc['active_subspace_identity_rel'] <= 1e-10
    std = [a for a in controls if meets(a)]
    if not parity: status = 'BLOCKED_BASELINE'
    elif base == 'GO_METHOD_SCREEN': status = 'HOLD_NO_AUTO_RESCUE'          # EMLoC is a LIMITED_BASELINE under the contract map
    elif std: status = 'GO_STANDARD_ONLY'
    elif base == 'GO_STANDARD_ONLY': status = 'HOLD_NO_AUTO_RESCUE'
    else: status = base
    cv = [float(r['active_step_cv']) for r in resources]
    tp = {}
    for r in resources: tp.setdefault(r['arm'], []).append(float(r['throughput_steps_per_s']))
    hold = any(v > SET['timing_cv_limit'] for v in cv) or any(abs(v[0]-v[1])/min(v) > SET['seed_throughput_rel_limit'] for v in tp.values())
    if hold and status in ('GO_METHOD_SCREEN', 'GO_STANDARD_ONLY'): status = 'HOLD_NO_AUTO_RESCUE'
    return status


def verify():
    assert_seal(); seal = read_json(OUT/'SOURCE_SEAL.json'); checks = {}
    changed_reporting = {k: dict(sealed=seal['reporting_code'].get(k), current=v)
                         for k, v in code_hashes(REPORTING_CODE).items() if seal['reporting_code'].get(k) != v}
    # budgets
    main = require_complete_ledger(OUT/'UPDATE_LEDGER.jsonl', CFG['max_main_updates'])
    calib = require_complete_ledger(OUT/'CALIBRATION_LEDGER.jsonl', CFG['max_calibration_updates'])
    smoke = require_complete_ledger(OUT/'SMOKE_LEDGER.jsonl', CFG['max_smoke_updates'])
    total = main['updates'] + calib['updates'] + smoke['updates']
    assert total <= CFG['max_optimizer_updates'], total
    keys = {json.loads(l)['fit'] for l in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()}
    assert keys == {f'{a}_{s}' for s in CFG['seeds'] for a in ARMS}, keys
    checks['budget'] = dict(main=main, calibration=calib, smoke=smoke, total=total, cap=CFG['max_optimizer_updates'])
    for n in ('PREFLIGHT_BASE.json', 'PREFLIGHT_EMULATOR.json'): assert read_json(OUT/n)['status'] == 'PASS', n
    # fits
    meta = read_json(CACHE/'emulator'/'meta.json'); canon = meta['canonical']; tops = set(meta['top3'])
    inits = {s: torch.load(CACHE/'init'/f'seed_{s}.pt', weights_only=True) for s in CFG['seeds']}
    for s in CFG['seeds']:
        assert sha(CACHE/'init'/f'seed_{s}.pt') == read_json(OUT/'PREFLIGHT_BASE.json')['init_files'][str(s)]['sha256']
    fits = {}
    for s in CFG['seeds']:
        sched = set()
        for a in ARMS:
            f = read_json(OUT/'fits'/f'{a}_{s}.json'); fits[(a, s)] = f
            assert f['complete'] and f['frozen_unchanged'] and f['changed_trainable_tensors'] > 0, (a, s)
            assert len(f['trace']) == CFG['main_steps'] and f['trace'][-1]['step'] == CFG['main_steps']
            assert [r['step'] for r in f['curves']] == CFG['checkpoints']
            assert f['selected_step'] == choose_checkpoint(f['curves'])['step']
            work = REPO/f['checkpoint_dir']
            for r in f['curves']:
                assert sha(work/f'step_{r["step"]}.pt') == r['state_sha256']
                assert math.isfinite(r['val_primary'])
            assert sha(work/'selected_export.pt') == f['selected_export_sha256']
            names = tops if a == 'F_TOP3' else set(canon)
            expect = {k: v for k, v in inits[s].items() if k.rsplit('.', 2)[0] in names}
            assert f['initial_lora_sha256'] == tensor_digest(expect), ('init', a, s)
            sched.add(f['schedule_sha256'])
            if a in DELTA_ARMS or a == 'E_EMLOC':
                assert f['transfer']['backbone_unchanged'] and f['transfer']['hash_checked']
        assert len(sched) == 1, 'minibatch schedule differs between arms'
    checks['fits'] = dict(count=len(fits), selection_recomputed=True, init_and_schedule_matched=True, checkpoints_hashed=True)
    # selection and predictions
    sel = read_json(OUT/'MODEL_SELECTION.json')
    for fn, h in sel['fit_hashes'].items(): assert sha(OUT/'fits'/fn) == h, fn
    man = read_json(OUT/'PREDICTIONS_MANIFEST.json')
    assert man['all_saved_before_scoring'] and man['selection_sha256'] == sha(OUT/'MODEL_SELECTION.json')
    for p, h in man['predictions'].items(): assert sha(REPO/p) == h, p
    # scores recomputed independently of SCORES.csv
    panel = read_panel(CACHE/'panel.npz'); y = panel.targets('test')
    levels = np.asarray(read_json(GAP_RESULTS/'jena'/'MODEL_RECEIPT.json')['levels'])
    srows = {(r['arm'], r['seed'], r['variant']): r for r in _rows('SCORES.csv')}
    for (a, s), f in fits.items():
        q = np.load(CACHE/'test_predictions'/f'{a}_{s}.npz')['q']
        m, _ = score(q, y, panel.sigma, levels)
        assert math.isclose(m['primary'], float(srows[(a, str(s), 'selected')]['primary']), rel_tol=1e-12)
        assert int(srows[(a, str(s), 'selected')]['step']) == f['selected_step']
        assert np.isclose(scalar_primary(q[:3, :3], y[:3, :3], panel.sigma[:3], levels),
                          score(q[:3, :3], y[:3, :3], panel.sigma[:3], levels)[0]['primary'], rtol=1e-10)
    checks['scores'] = dict(recomputed_from_saved_predictions=True, scalar_loop_replay=True)
    # effects = ratio of means over seeds (not mean of ratios)
    mean_p = lambda a, v='selected': np.mean([float(srows[(a, str(s), v)]['primary']) for s in CFG['seeds']])
    for r in _rows('SEED_EFFECTS.csv'):
        a, b = r['candidate'], r['baseline']
        if b == 'E_EMLOC_NAIVE_SAME_STEP': bm = mean_p('E_EMLOC', 'naive_transfer_same_step')
        elif b == 'F0': bm = float(srows[('F0', '0', 'reference_f0')]['primary'])
        else: bm = mean_p(b)
        assert np.isclose(float(r['mean_gain_pct']), gain(mean_p(a), bm), rtol=1e-9, atol=1e-9), (a, b)
    checks['effects'] = dict(ratio_of_means=True)
    # costs: totals equal the sum of charged components
    comp = _rows('COST_COMPONENTS.csv')
    for r in _rows('RESOURCES.csv'):
        parts = [float(c['seconds']) for c in comp if c['ledger'] == 'cold_start_charged' and c['arm'] == r['arm'] and c['seed'] == r['seed']]
        assert math.isclose(sum(parts), float(r['total_s']), rel_tol=1e-9), (r['arm'], r['seed'])
        assert math.isclose(float(r['total_s']), decision.cost_of(fits[(r['arm'], int(r['seed']))])['total_s'], rel_tol=1e-12)
    checks['costs'] = dict(total_equals_components=True, diagnostics_excluded=True)
    # decision reproducible
    d = read_json(OUT/'DECISION.json')
    tp = {(a, s): float(srows[(a, str(s), 'selected')]['primary']) for (a, s) in fits}
    emloc = read_json(OUT/'PREFLIGHT_EMULATOR.json')['checks']['emloc']
    again = decision.decide(fits, tp, [dict(r, active_step_cv=float(r['active_step_cv']), throughput_steps_per_s=float(r['throughput_steps_per_s']),
                                            seed=int(r['seed'])) for r in _rows('RESOURCES.csv')], emloc)
    assert again['status'] == d['status'] and again['base_outcome'] == d['base_outcome'], (again['status'], d['status'])
    independent = independent_status(srows, _rows('RESOURCES.csv'), emloc)
    assert independent == d['status'], (independent, d['status'])
    checks['decision'] = dict(status=d['status'], recomputed=True, independent_table_replay=independent)
    # report, figures, links
    md = (OUT/'REPORT_KO.md').read_text(encoding='utf-8')
    links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', md)
    assert len(links) >= 5
    for l in links: assert (OUT/l).is_file(), l
    for l in re.findall(r'\]\(([^)#]+\.(?:json|csv|md))\)', md):
        if l != 'VERIFICATION.json': assert (OUT/l).is_file(), l
    for n in ('fig1_transferred_quality', 'fig2_total_cost_frontier', 'fig3_cost_breakdown', 'fig4_transfer_mismatch', 'fig5_response_ablation'):
        for ext in ('png', 'svg', 'pdf'): assert (OUT/'figures'/f'{n}.{ext}').is_file(), (n, ext)
    assert (OUT/'FIGURE_VALUES.csv').is_file() and (OUT/'CAPTIONS.md').is_file()
    assert f'`{d["status"]}`' in md
    checks['report'] = dict(image_links=len(links), figures=15, status_in_report=True)
    rec = dict(status='PASS', checks=checks, reporting_code_changed_after_seal=changed_reporting,
               scope='saved artifacts + CPU recomputation; GPU parity checks are recorded in PREFLIGHT_*.json',
               independent_new_machine_replication=False, utc=time.time())
    write_json(OUT/'VERIFICATION.json', rec)
    return rec
