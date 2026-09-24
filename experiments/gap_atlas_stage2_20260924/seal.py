#!/usr/bin/env python
"""SEAL.json (contract section 6): configuration list, period boundaries, arm settings, selection
rules and thresholds. Evaluation windows may be scored only after this file exists, and score.py
refuses to run if the sealed digest no longer matches."""
from __future__ import annotations
import time
import ga
import arms
from periods import Period


def selection_digest():
    """Everything the evaluation must not depend on: the selections and the fits behind them."""
    sel = ga.read_json(ga.CACHE/'selection.json')
    led = ga.read_json(ga.CACHE/'fit_ledger.json')
    trimmed = {k: {kk: (vv if not isinstance(vv, dict) else
                        {k3: vv[k3] for k3 in ('arm', 'val_crps', 'selected_checkpoint', 'selected_path',
                                               'val_scores', 'candidates') if k3 in vv})
                   for kk, vv in v.items()} for k, v in sel.items()}
    return ga.digest(dict(selection=trimmed,
                          fits={k: {kk: v[kk] for kk in ('steps', 'mode', 'period', 'config')}
                                for k, v in led.items()}))


def main():
    path = ga.OUT/'SEAL.json'
    if path.exists():
        print('SEAL.json already exists; refusing to re-seal'); return
    e0 = ga.read_json(ga.OUT/'E0_REPORT.json')
    if e0.get('verdict') != 'E0_PASS':
        raise SystemExit(f"REFUSED: E0 verdict is {e0.get('verdict')}")
    periods = ga.read_json(ga.OUT/'PERIODS.json')
    sel = ga.read_json(ga.CACHE/'selection.json')
    steps, fits = arms.optimizer_updates()
    seal = dict(
        utc=time.time(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
        contract_sha256=ga.sha(ga.HERE/'00_MASTER_CLI_gap_atlas_stage2_20260924.txt'),
        loader_sha256=ga.sha(ga.HERE/'gift_eval_data.py'),
        configurations=[dict(id=c['id'], key=c['key'], role=c['role'],
                             feasible=ga.geometry(c)['feasible']) for c in ga.CONFIGS],
        periods={k: {p: v[p] for p in ('P1', 'P2') if p in v} for k, v in periods['per_config'].items()},
        arms=dict(
            F0='Chronos-2 zero-shot, official notebook settings (quantiles 0.1-0.9, cross-learning, '
               'float32, batch 100, one rolling window at a time)',
            LORA=dict(finetune_mode='lora', lora_config='package default', learning_rate=1e-5,
                      num_steps=ga.FIT_STEPS, batch_size=32, validation_inputs=None,
                      save_strategy='steps', save_steps=ga.CKPT_EVERY, seed=0),
            FULL=dict(finetune_mode='full', learning_rate=1e-6, num_steps=ga.FIT_STEPS,
                      batch_size=32, validation_inputs=None, save_strategy='steps',
                      save_steps=ga.CKPT_EVERY, seed=0),
            DL=dict(library='neuralforecast', models=['PatchTST', 'NHITS'],
                    loss='MQLoss(quantiles 0.1-0.9)', early_stopping='validation block, patience 3',
                    note='library defaults except h, input_size, loss, max_steps'),
            SNAIVE='statsforecast SeasonalNaive through the official GIFT-Eval wrapper (report only)',
            TOTO='skipped, see PREMISE_AUDIT.json Q6'),
        selection=dict(checkpoints=arms.CKPTS,
                       rule='lowest validation-block CRPS; ACH is the best of LORA, FULL and DL',
                       chosen={k: {'ACH': v['ACH']['arm'],
                                   'LORA_ckpt': v.get('LORA', {}).get('selected_checkpoint'),
                                   'FULL_ckpt': v.get('FULL', {}).get('selected_checkpoint')}
                               for k, v in sel.items()}),
        thresholds=dict(gap=0.10, control_gap=0.10, ci_rule='bootstrap lower bound > 0',
                        bootstrap_draws=2000, bootstrap_seed=20260924,
                        e0a_tolerance=0.02, e0b_tolerance=0.05),
        metric=dict(crps='gluonts MeanWeightedSumQuantileLoss over quantiles 0.1-0.9',
                    reported_from='per-series terms recomputed in float64; the gluonts float32 '
                                  'aggregate is reported beside it and agrees to about 7e-09',
                    gap='G = (CRPS_F0 - CRPS_ACH)/CRPS_F0'),
        budget=dict(fits=fits, optimizer_steps=steps),
        selection_digest=selection_digest(),
        note='TEST windows are scored only after this file exists')
    ga.write_json(path, seal)
    print('SEAL written; selection digest', seal['selection_digest'][:16], '| fits', fits, '| steps', steps)


if __name__ == '__main__':
    main()
