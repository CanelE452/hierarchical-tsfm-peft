"""Synthetic checks for registration, chronology, and immutable selection."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.comparative_ablation_20261025_v1 import run_one as runner
from research.comparative_ablation_20261025_v1 import runtime


def job(method='A1', *, extended=False):
    return dict(job_id=f'CA_{method}_jena_K6_s92601' + ('_ext10' if extended else ''),
                method_id=method, dataset='jena', K=6, seed=92601, extended=extended)


@pytest.mark.parametrize('changes', [dict(seed=92600), dict(K=3), dict(dataset='unregistered'),
    dict(job_id='different'), dict(extended=1), dict(method_id='VAE')])
def test_job_rejects_unregistered_scientific_axes(changes):
    with pytest.raises(ValueError):
        runner.validate_job(dict(job(), **changes))


def test_fixed_pca_and_extension_schedules():
    assert runner.schedule(job('C1')) == []
    assert runner.schedule(job('C2')) == [.25, 1., 3., 6.]
    assert runner.schedule(job('C3', extended=True)) == [.25, 1., 3., 6., 10.]
    assert runner.schedule(job('A5')) == [.25, 1., 3.]
    with pytest.raises(ValueError):
        runner.validate_job(job('C1', extended=True))
    with pytest.raises(ValueError):
        runner.validate_job(job('OURS'))


def test_spec_records_actual_initialization():
    assert runner.build_spec(job('C2'))['init_kind'] == 'PYTORCH_DEFAULT'
    assert runner.build_spec(job('C3'))['init_kind'] == 'PYTORCH_DEFAULT'
    assert runner.build_spec(job('A3'))['init_kind'] == 'QR'
    for method in ('C1', 'A1', 'A2', 'A4', 'A5'):
        assert runner.build_spec(job(method))['init_kind'] == 'PCA'


def curve_result():
    return dict(curves=[dict(epoch=e, astop_mse=m) for e, m in [(0., 2.), (.25, 1.), (1., 1.), (3., 3.)]],
                summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.))


def test_selection_exact_tie_uses_earlier_epoch_and_complete_curve():
    assert runner.choose_epoch(curve_result(), [.25, 1., 3.]) == (.25, 1.)
    for kind in ('later_tie', 'incomplete', 'nonfinite'):
        value = curve_result()
        if kind == 'later_tie':
            value['summary']['selected_epoch'] = 1.
        elif kind == 'incomplete':
            value['curves'].pop()
        else:
            value['curves'][1]['astop_mse'] = float('nan')
        with pytest.raises(ValueError):
            runner.choose_epoch(value, [.25, 1., 3.])


def registration_fixture(monkeypatch):
    payloads = {'PREREGISTRATION.json': b'protocol', 'PREREGISTRATION.md': b'markdown', 'RUN_PLAN.json': b'run plan'}
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    records = {'RUN_PLAN.json': dict(jobs=[job()], extension_jobs=[]),
               'PREREGISTRATION.json': dict(registered_sources={'custom.py': 'source-hash'}),
               'PREREGISTRATION_COMMIT.json': dict(commit='commit123',
                   preregistration_sha256=digests['PREREGISTRATION.json'],
                   preregistration_md_sha256=digests['PREREGISTRATION.md'],
                   run_plan_sha256=digests['RUN_PLAN.json'])}
    monkeypatch.setattr(runtime, 'read', lambda path: records[Path(path).name])
    monkeypatch.setattr(runtime, 'sha', lambda path: digests.get(Path(path).name, 'plan-hash'))
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'custom.py': 'source-hash'})
    monkeypatch.setattr(runner.subprocess, 'check_output',
                        lambda args, **kwargs: payloads[args[-1].split('/')[-1]])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    return records, payloads


def test_registry_requires_exact_job_committed_bytes_and_custom_sources(monkeypatch):
    records, payloads = registration_fixture(monkeypatch)
    assert runner.validate_registration(job())['commit'] == 'commit123'
    records['RUN_PLAN.json']['jobs'] = []
    with pytest.raises(ValueError, match='absent'):
        runner.validate_registration(job())
    records['RUN_PLAN.json']['jobs'] = [job()]
    records['PREREGISTRATION.json']['registered_sources']['custom.py'] = 'changed'
    with pytest.raises(ValueError, match='implementation'):
        runner.validate_registration(job())
    records['PREREGISTRATION.json']['registered_sources']['custom.py'] = 'source-hash'
    payloads['PREREGISTRATION.md'] = b'changed in commit'
    with pytest.raises(ValueError, match='commit does not contain'):
        runner.validate_registration(job())


@pytest.mark.parametrize('method', ['A1', 'C1', 'C3'])
def test_registered_boundaries_and_seal_before_refit_and_V(tmp_path, monkeypatch, method):
    monkeypatch.setattr(runtime, 'REPORT', tmp_path / 'reports')
    monkeypatch.setattr(runtime, 'CACHE', tmp_path / 'cache')
    public, cache = runtime.REPORT / 'attempt', runtime.CACHE / 'attempt'
    public.mkdir(parents=True)
    cache.mkdir(parents=True)
    events = []
    selected_job = job(method)
    spec = runner.build_spec(selected_job)
    toy = torch.nn.Linear(2, 2)
    norm = dict(mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    bounds = dict(A_fit=(0, 10482), A_stop=(10482, 12579), M_cal=(12579, 14675),
                  V_select=(14675, 16772), T_final=(16772, 20965))
    task = dict(physical=np.zeros((16772, 2), np.float32), split_ranges=bounds, L=512, H=48,
                columns=['a', 'b'], array_sha256={}, data_sha256='data', dataset='jena', _campaign_access={})

    def stage(specification, data, endpoint, stop, epochs, *unused):
        if stop is not None:
            events.append(('search', endpoint, stop))
            result = curve_result()
            if method == 'C3':
                result['curves'].append(dict(epoch=6., astop_mse=4.))
        else:
            assert (public / 'SELECTION.json').exists()
            events.append(('refit', endpoint, stop))
            assert epochs == [.25]
            result = dict(summary=dict(status='SEGMENT_COMPLETE'))
        result['summary'].update(seconds=1., exposures=1, updates=1)
        result.update(model=toy.state_dict(), normalization=norm)
        return result

    def predict(model, normalization, data, foundation, specification, path, modules, *, deterministic=False):
        assert (public / 'SELECTION.json').exists()
        assert runtime.read(public / 'SELECTION.json')['V_used'] is False
        events.append(('V_diagnostic' if deterministic else 'V', bounds['V_select']))
        path.write_bytes(b'fake V arrays')
        return dict(MSE=1., MAE=1., prediction_path=str(path), prediction_sha256=runtime.sha(path))

    fake_fit = SimpleNamespace(_source_contract=lambda: ('core', {'fit.py': 'hash'}), _stage=stage,
        create_model=lambda *a: (toy, norm), _torch_new=lambda path, value: path.write_bytes(b'fake bundle'))
    fake_training = SimpleNamespace(fingerprint=lambda value: 'fingerprint', cpu_state=lambda model: model.state_dict())
    monkeypatch.setattr(runner, 'predict_saved', predict)
    result = runner.run_stages(selected_job, task, SimpleNamespace(initial_hash='f'),
        dict(fit=fake_fit, training=fake_training), spec, cache, public, 1000.)
    assert result['selection_sealed_before_V']
    if method == 'C1':
        assert events == [('V', bounds['V_select'])]
        assert result['row']['search_seconds'] == result['row']['refit_seconds'] == 0
    else:
        assert events[:2] == [('search', 10482, bounds['A_stop']), ('refit', 12579, None)]
        assert events[2] == ('V', bounds['V_select'])
    assert bool(result['deterministic_diagnostic']) == (method == 'C3')
    seal = public / 'SELECTION.json'
    original_digest = runtime.sha(seal)
    seal.write_text('{}')
    with pytest.raises(ValueError):
        runner.check_selection(seal, original_digest, selected_job)
