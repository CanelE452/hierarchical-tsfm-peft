#!/usr/bin/env python
"""Q1-Q6 of contract section 3 plus the section 10 executor gate, written to PREMISE_AUDIT.json.
Stops nothing by itself; E0 decides. Every entry records what was actually observed."""
from __future__ import annotations
import importlib.metadata as md
import json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import ga


def q1_official_geometry():
    """Official loader window count, prediction length and stride, read per configuration."""
    rows = {c['id']: ga.geometry(c) for c in ga.CONFIGS}
    for r in rows.values():
        r['stride'] = r['prediction_length']          # generate_instances(distance=prediction_length)
    return dict(status='CONFIRMED', per_config=rows,
                source='gift_eval_data.py Dataset.prediction_length/.windows/.test_data '
                       '(vendored from SalesforceAIResearch/gift-eval src/gift_eval/data.py)',
                loader_sha256=ga.sha(ga.HERE/'gift_eval_data.py'),
                rule='windows = clip(ceil(0.1*min_series_length/H), 1, 20); test stride = H')


def q2_feasibility():
    rows = {c['id']: ga.geometry(c) for c in ga.CONFIGS}
    infeasible = [k for k, v in rows.items() if not v['feasible']]
    return dict(status='PARTIAL' if infeasible else 'CONFIRMED',
                infeasible=infeasible,
                per_config={k: {kk: v[kk] for kk in ('key', 'role', 'prediction_length', 'windows',
                                                     'validation_windows', 'min_series_length',
                                                     'train_length_P1', 'train_length_P2',
                                                     'required_train_length', 'feasible')}
                            for k, v in rows.items()},
                rule='contract 5.2: P1 training segment >= the fit context (2048) + 5*H',
                consequence='contract 3 Q2 and 8: an INFEASIBLE configuration is excluded and is not '
                            'replaced by another one')


def q1b_context_length():
    """Contract 5.2 measures feasibility against 'the Chronos-2 context length' without a number."""
    rows = {c['id']: ga.geometry(c) for c in ga.CONFIGS}
    return dict(status='CONTRACT_AMBIGUOUS',
                model_default_context=ga.CONTEXT_MODEL_DEFAULT,
                source='amazon/chronos-2 config.json chronos_config.context_length',
                practical_context_used=ga.CONTEXT_FIT,
                feasible_practical=[k for k, v in rows.items() if v['feasible']],
                feasible_strict=[k for k, v in rows.items() if v['feasible_strict']],
                measured='bench_context.py, 12 optimizer steps per setting at the contract batch size 32: '
                         'C4 (7 variates) 0.47 s/step at 512, 0.40 at 2048, 2.60 at 8192 with a 12.62 GiB '
                         'peak; C1 (2 variates) 0.38 / 0.41 / 3.32 s/step with a 9.21 GiB peak',
                decision='fits run at context 2048 and the section 5.2 rule uses the same number. '
                         'Fitting at the model default 8192 peaks at 12.62 GiB on this 12 GiB card and '
                         'thrashes, so it is not runnable at the batch size the contract fixes; '
                         'contract Q4 allows changing the compute implementation and recording it. '
                         'The feasible set at 2048 is the same as at 512, so this choice does not '
                         'select a favourable subset.',
                effect='F0 always predicts at the package default, unchanged from the official notebook. '
                       'For LORA and FULL the prediction context (2048 or 8192) is chosen on the '
                       'validation block and recorded in SEAL.json.')


def q3_environment():
    import torch
    pkgs = {}
    for name in ('torch', 'chronos-forecasting', 'transformers', 'peft', 'accelerate', 'gluonts',
                 'neuralforecast', 'pytorch-lightning', 'datasets', 'numpy', 'pandas', 'scipy',
                 'pyarrow', 'toolz', 'python-dotenv', 'matplotlib'):
        try: pkgs[name] = md.version(name)
        except md.PackageNotFoundError: pkgs[name] = 'MISSING'
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'
    vram = torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0.0
    try:
        commit = subprocess.run(['git', '-C', str(ga.ROOT), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        commit = 'unknown'
    return dict(status='CONFIRMED', platform=platform.platform(), python=sys.version.split()[0],
                gpu=gpu, vram_gib=round(float(vram), 2), cuda=getattr(torch.version, 'cuda', None),
                packages=pkgs, repo=str(ga.ROOT), repo_commit=commit,
                model=ga.MODEL, revision=ga.REVISION,
                gift_eval_package='NOT INSTALLED — salesforce-gift-eval pins numpy~=1.26, '
                                  'datasets~=2.17, scipy~=1.11 and would downgrade this environment; '
                                  'the loader module is vendored instead',
                note='the contract does not assume an environment; these are the observed values')


def q4_full_finetune_smoke(steps=5, batch_size=32):
    """Does finetune_mode='full' run at batch 32 on this GPU? Smoke only; updates are recorded."""
    import torch
    cfg = ga.BY_ID['N3']                               # smallest univariate config
    ds = ga.dataset(cfg)
    series = []
    for i, entry in enumerate(ds.training_dataset):
        series.append(torch.tensor(np.asarray(entry['target'], dtype=np.float32)))
        if i >= 63: break
    pipe = ga.pipeline(dtype='float32')
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    err = None
    try:
        torch.manual_seed(0)
        pipe.fit(series, prediction_length=48, finetune_mode='full', context_length=512,
                 learning_rate=1e-6, num_steps=steps, batch_size=batch_size,
                 validation_inputs=None, output_dir=str(ga.SCRATCH/'q4_smoke'),
                 save_strategy='no', seed=0)
        ok = True
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, f'OutOfMemoryError: {str(e)[:200]}'
    except Exception as e:
        ok, err = False, f'{type(e).__name__}: {str(e)[:200]}'
    peak = torch.cuda.max_memory_allocated()/2**30
    secs = time.perf_counter()-t0
    del pipe; torch.cuda.empty_cache()
    return dict(status='CONFIRMED' if ok else 'FAILED', ran=ok, error=err,
                batch_size=batch_size, steps=steps, seconds=round(secs, 1),
                peak_vram_gib=round(float(peak), 2), series=len(series),
                note='smoke only; these 5 steps are counted in the optimizer-update ledger')


def q5_dl_library():
    try:
        import neuralforecast
        from neuralforecast.models import PatchTST, NHITS
        from neuralforecast.losses.pytorch import MQLoss
        return dict(status='CONFIRMED', library='neuralforecast', version=neuralforecast.__version__,
                    models=['PatchTST', 'NHITS'], loss='MQLoss(level from quantiles 0.1-0.9)',
                    defaults='library defaults except input_size, h, max_steps, early stopping on the '
                             'validation block; recorded per fit in GAP_TABLE.csv',
                    lightning=md.version('pytorch-lightning'))
    except Exception as e:
        return dict(status='BLOCKED_DL_LIB', error=f'{type(e).__name__}: {e}')


def q6_toto():
    return dict(status='BLOCKED_TOTO_ENV', available_weights='Datadog/Toto-2.0-2.5B (public, not gated)',
                blocker='the inference package toto-ts 0.2.0 resolves torch 2.7.0, numpy 1.26.4, '
                        'datasets 2.17.1, gluonts 0.16.2 and would downgrade this environment '
                        '(pip install --dry-run)',
                decision='contract 3 Q6: the TOTO arm is skipped and recorded; it was report-only')


def gate():
    """Contract section 10, checked before execution."""
    checks = {}
    checks['1_tools_exist'] = dict(
        verdict='PARTIAL',
        detail='the GIFT-Eval data, the official loader, gluonts and a deep-learning library were all '
               'absent from this machine. Data downloaded from Salesforce/GiftEval, loader vendored, '
               'gluonts 0.17.0 + neuralforecast 3.2.2 installed (additive, no existing package changed). '
               'The notebook argument `predict_batches_jointly` does not exist under that name in '
               'chronos-forecasting 2.3.2; it is the deprecated alias of `cross_learning` '
               '(chronos/chronos2/pipeline.py:570-577), so the semantics are unchanged.')
    checks['2_numbers_match'] = dict(verdict='DEFERRED', detail='E0a and E0b decide')
    checks['3_no_leakage'] = dict(verdict='DEFERRED', detail='E0c checks the indices')
    outputs = [p for p in ('GAP_TABLE.csv', 'STABILITY.csv', 'E0_REPORT.json', 'SEAL.json')
               if (ga.OUT/p).exists()]
    checks['4_outputs_absent'] = dict(verdict='PASS' if not outputs else 'FAIL', existing=outputs)
    checks['5_denominator'] = dict(verdict='DEFERRED', detail='checked per configuration when the gap '
                                                              'is formed; ga.gap returns nan at noise level')
    return checks


def main():
    audit = dict(utc=time.time(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
                 contract_sha256=ga.sha(ga.HERE/'00_MASTER_CLI_gap_atlas_stage2_20260924.txt'),
                 stage1_csv_sha256=ga.sha(ga.HERE/'stage1_gift_eval_gap.csv'),
                 stage1_script_sha256=ga.sha(ga.HERE/'stage1_gap_mining.py'))
    audit['gate_section_10'] = gate()
    audit['Q1_official_geometry'] = q1_official_geometry()
    audit['Q1b_context_length'] = q1b_context_length()
    audit['Q2_feasibility'] = q2_feasibility()
    audit['Q3_environment'] = q3_environment()
    audit['Q5_dl_library'] = q5_dl_library()
    audit['Q6_toto'] = q6_toto()
    prev = ga.read_json(ga.OUT/'PREMISE_AUDIT.json') if (ga.OUT/'PREMISE_AUDIT.json').exists() else {}
    if prev.get('Q4_full_finetune', {}).get('status') == 'CONFIRMED':
        audit['Q4_full_finetune'] = prev['Q4_full_finetune']      # smoke already ran; do not spend updates again
        print('Q4 reused from the previous audit run', flush=True)
    else:
        print('Q1/Q2/Q3/Q5/Q6 done; running the Q4 full-finetune smoke', flush=True)
        audit['Q4_full_finetune'] = q4_full_finetune_smoke()
    blocked = [k for k, v in audit.items() if isinstance(v, dict) and str(v.get('status', '')).startswith(('BLOCKED', 'FAILED'))]
    audit['verdict'] = 'PREMISES_OK' if not blocked else 'PREMISES_OK_EXCEPT_' + '_'.join(blocked)
    ga.write_json(ga.OUT/'PREMISE_AUDIT.json', audit)
    print('verdict:', audit['verdict'])
    print('infeasible configs:', audit['Q2_feasibility']['infeasible'])


if __name__ == '__main__':
    main()
