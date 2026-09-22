"""Pre-registered cost accounting and decision table (sealed with the training code).

Decision table (PROTOCOL_KO.md section 9, fixed before any fit):
  1. EMLoC port parity failed                         -> BLOCKED_BASELINE
  2. candidate passes reference_core.outcome as GO_METHOD_SCREEN
       and EMLoC baseline is only LIMITED              -> HOLD_NO_AUTO_RESCUE (LIMITED_BASELINE)
       else                                            -> GO_METHOD_SCREEN
  3. else some control (F_TOP3, E_EMLOC, E_DELTA, E_VALUE_DELTA) keeps F_FULL quality within 0.5% on average and in
     both seeds AND saves >=20% total time on average and is faster in both seeds -> GO_STANDARD_ONLY
  4. else reference_core.outcome status (NO_GO_CURRENT_IMPLEMENTATION or HOLD_NO_AUTO_RESCUE); an outcome
     GO_STANDARD_ONLY that fails the per-seed test in 3 becomes HOLD_NO_AUTO_RESCUE
  5. timing unreliable (any fit trimmed step CV > 0.2 or an arm's seed throughput gap > 20%) turns only a speed-based
     GO (GO_METHOD_SCREEN or GO_STANDARD_ONLY) into HOLD_NO_AUTO_RESCUE (TIMING_HOLD). NO_GO is never rescued.
"""
from __future__ import annotations
import math, time
import numpy as np
from .common import CFG, SET, ARMS, OUT, read_json, gain, rc

CAND = CFG['candidate']
CONTROLS = {'F_TOP3': 'F_TOP3', 'E_EMLOC': 'E_EMLOC', 'E_DELTA': 'E_DELTA', 'VALUE_DELTA': 'E_VALUE_DELTA'}
FIT_PROCEDURE = ['main_updates', 'checkpoint_io', 'val_eval', 'full_model_load', 'emulator_load', 'full_model_load_transfer']
TRANSFER_SELECTION = ['transfer', 'selection_export']
DIAGNOSTIC = ['diag_emulator_val', 'diag_naive_transfer']


def cost_of(f):
    parts = f['parts']; prep = float(sum(f['charged_preparation'].values()))
    proc = float(sum(parts.get(k, 0.) for k in FIT_PROCEDURE))
    trans = float(sum(parts.get(k, 0.) for k in TRANSFER_SELECTION))
    return dict(preparation_s=prep, fit_procedure_s=proc, transfer_selection_s=trans, total_s=prep+proc+trans,
                diagnostic_s=float(sum(parts.get(k, 0.) for k in DIAGNOSTIC)))


def sensitivity_total(f, n_train):
    """Auxiliary only: contract total minus VAL caches (diagnostic-only) and the unused share of TRAIN caches.
    The unused share is estimated proportionally to origin counts [estimate]."""
    c = cost_of(f); ch = f['charged_preparation']; used = f['unique_train_origins']/n_train
    val = sum(v for k, v in ch.items() if k.startswith('caches:') and k.endswith('_val'))
    train = sum(v for k, v in ch.items() if k.startswith('caches:') and k.endswith('_train'))
    return c['total_s'] - val - train*(1-used)


def prefix_costs(f):
    """Auxiliary: cost to reach each recorded checkpoint (NOT the cost of knowing it is best)."""
    c = cost_of(f); parts = f['parts']
    loads = sum(parts.get(k, 0.) for k in ('full_model_load', 'emulator_load', 'full_model_load_transfer'))
    out, acc = {}, 0.
    for r in f['curves']:
        acc += r.get('val_eval_s', 0.) + r.get('transfer_s', 0.)
        out[r['step']] = c['preparation_s'] + loads + r['updates_s_so_far'] + r['checkpoint_io_s_so_far'] + acc
    return out


def timing(f):
    t = np.asarray([r['active_step_s'] for r in f['trace']]); tr = t[SET['timing_trim_steps']:]
    return dict(active_step_mean_s=float(t.mean()), active_step_cv=float(tr.std()/tr.mean()),
                throughput_steps_per_s=float(len(t)/t.sum()))


def meets_standard(pts_arm, pts_full):
    m = CFG['quality_relative_margin']; s = CFG['minimum_total_time_saving']
    fe = np.mean([p.test_error for p in pts_full]); ft = np.mean([p.total_seconds for p in pts_full])
    ae = np.mean([p.test_error for p in pts_arm]); at = np.mean([p.total_seconds for p in pts_arm])
    full = {p.seed: p for p in pts_full}
    return bool(ae <= fe*(1+m) and at <= ft*(1-s) and
                all(p.test_error <= (1+m)*full[p.seed].test_error and p.total_seconds < full[p.seed].total_seconds for p in pts_arm))


def decide(fits, test_primary, resources, emloc_checks):
    seeds = CFG['seeds']
    pts = {a: [rc.FixedBudgetResult(a, s, fits[(a, s)]['selected_step'], test_primary[(a, s)],
                                    cost_of(fits[(a, s)])['fit_procedure_s'], cost_of(fits[(a, s)])['preparation_s'],
                                    cost_of(fits[(a, s)])['transfer_selection_s']) for s in seeds] for a in ARMS}
    for a in ARMS:
        for p in pts[a]:
            if not math.isclose(p.total_seconds, cost_of(fits[(a, p.seed)])['total_s'], rel_tol=1e-12): raise ValueError('COST_SUM_MISMATCH')
    base = rc.outcome(pts['F_FULL'], pts[CAND], {k: pts[v] for k, v in CONTROLS.items()},
                      min_speedup=CFG['minimum_total_time_saving'], quality_margin=CFG['quality_relative_margin'],
                      min_response_gain=CFG['minimum_response_gain_over_value'])
    standard_ok = [v for v in CONTROLS.values() if meets_standard(pts[v], pts['F_FULL'])]
    noisy = [dict(arm=r['arm'], seed=r['seed'], cv=r['active_step_cv']) for r in resources if r['active_step_cv'] > SET['timing_cv_limit']]
    tput = {}
    for r in resources: tput.setdefault(r['arm'], []).append(r['throughput_steps_per_s'])
    seed_gap = {a: abs(v[0]-v[1])/min(v) for a, v in tput.items()}
    timing_hold = bool(noisy or any(g > SET['seed_throughput_rel_limit'] for g in seed_gap.values()))
    parity = (emloc_checks['official_vs_port_scaling1_fp32_rel'] <= SET['official_emloc_fp32_rel_tol']
              and emloc_checks['active_subspace_identity_rel'] <= 1e-10)
    emloc = 'LIMITED_BASELINE' if parity else 'BLOCKED_BASELINE'
    reasons = []
    if not parity:
        status = 'BLOCKED_BASELINE'; reasons.append('EMLoC port parity not established')
    elif base['status'] == 'GO_METHOD_SCREEN':
        status = 'GO_METHOD_SCREEN'
        if emloc == 'LIMITED_BASELINE':
            status = 'HOLD_NO_AUTO_RESCUE'; reasons.append('LIMITED_BASELINE: EMLoC cannot correct compressed MLP layers under the contract LoRA map')
    elif standard_ok:
        status = 'GO_STANDARD_ONLY'; reasons.append(f'standard control(s) meet quality and time: {standard_ok}')
    elif base['status'] == 'GO_STANDARD_ONLY':
        status = 'HOLD_NO_AUTO_RESCUE'; reasons.append('dominating control meets mean criteria but not the per-seed test')
    else:
        status = base['status']
    if timing_hold and status in ('GO_METHOD_SCREEN', 'GO_STANDARD_ONLY'):
        reasons.append(f'TIMING_HOLD: speed-based {status} withheld'); status = 'HOLD_NO_AUTO_RESCUE'
    by = {p.seed: p for p in pts[CAND]}; fb = {p.seed: p for p in pts['F_FULL']}
    quality_fail = (not base['quality_ok']) or any(by[s].test_error > (1+CFG['quality_relative_margin'])*fb[s].test_error for s in seeds)
    limited = {}
    for a in ARMS:
        for s in seeds:
            f = fits[(a, s)]; v = {r['step']: r['val_primary'] for r in f['curves']}
            limited[f'{a}_{s}'] = bool(f['selected_step'] == CFG['checkpoints'][-1] and gain(v[128], v[64]) > 0.2)
    mean = lambda a, k: float(np.mean([getattr(p, k) for p in pts[a]]))
    pareto = []
    for a in ARMS:
        e, t = mean(a, 'test_error'), mean(a, 'total_seconds')
        dom = [b for b in ARMS if b != a and mean(b, 'test_error') <= e and mean(b, 'total_seconds') <= t
               and (mean(b, 'test_error') < e or mean(b, 'total_seconds') < t)]
        pareto.append(dict(arm=a, mean_test_primary=e, mean_total_s=t, pareto_optimal=not dom, dominated_by=dom,
                           test_by_seed=[p.test_error for p in pts[a]], total_by_seed=[p.total_seconds for p in pts[a]],
                           meets_standard_criteria=meets_standard(pts[a], pts['F_FULL']) if a != 'F_FULL' else None))
    return dict(status=status, base_outcome=base, reasons=reasons, standard_controls_meeting_goal=standard_ok,
                timing_hold=timing_hold, noisy_fits=noisy, seed_throughput_gap=seed_gap, quality_failure=quality_fail,
                emloc_baseline=emloc, limited_budget=limited, pareto=pareto, candidate=CAND,
                thresholds=dict(quality_margin=CFG['quality_relative_margin'], min_total_time_saving=CFG['minimum_total_time_saving'],
                                min_response_gain=CFG['minimum_response_gain_over_value'], timing_cv=SET['timing_cv_limit'],
                                seed_throughput=SET['seed_throughput_rel_limit']),
                scope='Jena only, two main seeds sharing one compression and one calibration per kind, fixed 128 updates; '
                      'not significance/equivalence, not paper PASS', automatic_follow_up=False)
