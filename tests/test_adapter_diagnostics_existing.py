import copy
import json
import time

import numpy as np
import pytest

from research.adapter_diagnostics_20261017_v2 import existing_jobs as jobs


def test_fp64_score_uses_observed_channel_macro_and_promotes_before_subtraction():
    prediction = np.array([[[1e8, 4], [1, 10]]], dtype=np.float32)
    target = np.array([[[1, 2], [np.nan, 4]]], dtype=np.float32)
    mask = np.array([[[True, True], [False, True]]])
    score = jobs.score_arrays(prediction, target, mask)
    assert score['channel_counts'] == [1, 2]
    assert score['channel_sse'][0] == (float(prediction[0, 0, 0]) - 1) ** 2
    assert score['MSE'] == ((1e8-1)**2 + (4+36)/2) / 2
    assert score['MAE'] == ((1e8-1) + (2+6)/2) / 2


def test_fp64_score_rejects_non_boolean_mask_and_unobserved_channel():
    a = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match='boolean'):
        jobs.score_arrays(a, a, np.ones_like(a))
    with pytest.raises(ValueError, match='observed'):
        jobs.score_arrays(a, a, np.zeros_like(a, dtype=bool))


def test_saved_archive_requires_exact_origins_target_mask_and_channel_order():
    values = dict(pred_normalized=np.zeros((2, 3, 2), np.float32),
                  target_normalized=np.zeros((2, 3, 2), np.float32),
                  observed=np.ones((2, 3, 2), bool), origins=np.array([10, 13]),
                  channel_ids=np.array(['a', 'b']))
    jobs.validate_cohort(values, copy.deepcopy(values))
    for name in ('origins', 'target_normalized', 'observed', 'channel_ids'):
        altered = copy.deepcopy(values)
        if name == 'channel_ids':
            altered[name] = altered[name][::-1]
        else:
            altered[name].flat[0] += 1
            if name == 'observed':
                altered[name].flat[0] = False
        with pytest.raises(ValueError, match=name):
            jobs.validate_cohort(values, altered)


def test_completed_receipt_reuse_rejects_scientific_identity_or_artifact_change(tmp_path):
    ctx = dict(report=tmp_path/'report', cache=tmp_path/'cache',
               repo=tmp_path/'repo', asset=tmp_path/'asset', deadline=time.time()+30, plan={})
    ctx['report'].mkdir()
    source = ctx['report']/'result.json'
    source.write_text('{"complete": true}', encoding='utf-8')
    job = dict(job_id='A02_s92601', question_id='A02', seed=92601,
               scientific_identity={'data': 'cohort1', 'reader': 'fp64'})
    original = dict(status='COMPLETE', reason='verified', artifact_paths=[str(source)], result={})
    jobs.save_receipt(job, ctx, original)
    reused = jobs.completed_receipt(job, ctx)
    assert reused['status'] == 'REUSED_VERIFIED'
    changed = dict(job, scientific_identity={'data': 'cohort2', 'reader': 'fp64'})
    with pytest.raises(ValueError, match='identity'):
        jobs.completed_receipt(changed, ctx)
    source.write_text('{"complete": false}', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact'):
        jobs.completed_receipt(job, ctx)


def test_failed_receipt_is_not_completed_and_complete_requires_artifacts(tmp_path):
    ctx = dict(report=tmp_path)
    job = dict(job_id='x', scientific_identity='one')
    jobs.save_receipt(job, ctx, dict(status='BLOCKED_MISSING_ASSET', reason='absent', artifact_paths=[]))
    assert jobs.completed_receipt(job, ctx) is None
    with pytest.raises(ValueError, match='artifact'):
        jobs.save_receipt(job, ctx, dict(status='COMPLETE', reason='', artifact_paths=[]))


def test_cost_registration_is_original_pems_pair_batch_routes():
    result = jobs.registered_cost_conditions()
    assert {(v['family'], v['batch']) for v in result} == {
        ('B0', 1), ('B0', 8), ('B2', 1), ('B2', 8)}
    assert all(v['dataset_cohort'] == 'pems08_C32' for v in result)
    assert all(v['warmups'] == 10 and v['timed_trials'] == 20 for v in result)


def test_endpoint_indices_never_alias_selected_a60_or_wrong_split():
    assert jobs.endpoint_spec(dict(seed=92601, family='B2_CAP_POINTWISE', endpoint_index=2,
                                   split='V_select')) == dict(
        epoch=1.0, updates=622, exposures=9923, metric_scale='canonical_A60', stride=48)
    with pytest.raises(ValueError):
        jobs.endpoint_spec(dict(seed=92601, family='B2_CAP_POINTWISE', endpoint_index=4, split='V_select'))
    with pytest.raises(ValueError):
        jobs.endpoint_spec(dict(seed=92601, family='B2_CAP_POINTWISE', endpoint_index=2, split='T_final'))


def test_rss_sampler_reports_observed_peak_and_scope_without_claiming_continuous_peak():
    values = iter([7, 11, 9])
    sampler = jobs.RSSSampler(lambda: next(values))
    sampler.sample()
    sampler.sample()
    summary = sampler.stop()
    assert summary['process_RSS_sampled_peak_bytes'] == 11
    assert summary['RSS_sample_count'] == 3
    assert summary['RSS_sampling_interval_seconds'] == .005
    assert summary['RSS_peak_is_sampled_not_continuous']
    assert 'excludes cold model load' in summary['RSS_scope']


def test_cuda_branch_conversion_preserves_cpu_progress_and_changes_only_branch_identity():
    original = dict(model={'weight': [1.]}, optimizer={'state': {'one': [2.]}},
                    sampler={'position': 17, 'rng': {'state': 3}}, torch_rng=[4, 5], cuda_rng=[],
                    contract=dict(source_sha='source', config_sha='old', data_sha='data',
                                  model_sha='weights', scaler_sha='scale'), updates=9,
                    extra={'completed_epochs': [0., .25, 1.]})
    snapshot = copy.deepcopy(original)
    spec = dict(device='cuda', family='B0', seed=92601, latent_k=8)
    converted, lineage = jobs.cuda_branch_payload(original, spec, 'terminal', 'asset://old.pt', [[6, 7]])
    assert original == snapshot
    assert converted['model'] == original['model']
    assert converted['optimizer'] == original['optimizer']
    assert converted['sampler'] == original['sampler']
    assert converted['torch_rng'] == original['torch_rng']
    assert converted['cuda_rng'] == [[6, 7]]
    assert converted['updates'] == 9
    assert converted['contract']['config_sha'] != 'old'
    assert converted['contract']['scaler_sha'] == 'scale'
    assert lineage['continuation_kind'] == 'CPU_TO_CUDA_NUMERICAL_BRANCH'
    assert not lineage['historical_bitwise_equivalence_claimed']
    again, _ = jobs.cuda_branch_payload(original, spec, 'terminal', 'asset://old.pt', [[6, 7]])
    assert converted['contract'] == again['contract']


def test_cuda_endpoint_parity_uses_fixed_tolerance_and_rejects_bad_predictions():
    from research.adapter_diagnostics_20261017_v2.gpu_endpoints import parity_statistics
    reference = np.ones((1, 48, 21), np.float32)
    close = parity_statistics(reference, reference+np.float32(1e-6))
    assert close['status'] == 'PASS'
    assert close['atol'] == 1e-5 and close['rtol'] == 1e-4
    failed = parity_statistics(reference, reference+.001)
    assert failed['status'] == 'FAILED'
    assert not failed['bitwise_equivalence_claimed']


def test_optimizer_check_allows_cpu_step_but_requires_cuda_moments():
    from types import SimpleNamespace
    class Parameter:
        device = 'cuda:0'
    parameter = Parameter()
    optimizer = SimpleNamespace(param_groups=[{'params': [parameter]}], state={parameter: {
        'step': SimpleNamespace(device='cpu'), 'exp_avg': SimpleNamespace(device='cuda:0'),
        'exp_avg_sq': SimpleNamespace(device='cuda:0')}})
    assert jobs._optimizer_devices(optimizer)['non_step_optimizer_tensors_checked'] == 2
    optimizer.state[parameter]['exp_avg_sq'].device = 'cpu'
    with pytest.raises(ValueError, match='device'):
        jobs._optimizer_devices(optimizer)


def test_disposable_cpu_profile_update_cannot_mutate_source_optimizer_bundle():
    import torch
    fingerprint=jobs.resume_payload_fingerprint
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter], lr=.001)
    parameter.grad=torch.tensor([.5]); optimizer.step(); optimizer.zero_grad()
    original=dict(model={'weight':parameter.detach().clone()}, optimizer=optimizer.state_dict(),
                  sampler={'position':17}, torch_rng=torch.get_rng_state(), cuda_rng=[],
                  contract={'config_sha':'old'},extra={})
    before=fingerprint(original)
    payload=jobs.disposable_resume_payload(original,'cpu')
    other=torch.nn.Parameter(original['model']['weight'].clone())
    audit=torch.optim.AdamW([other],lr=.001)
    audit.load_state_dict(payload['optimizer'])
    other.grad=torch.tensor([.5]); audit.step()
    assert fingerprint(original)==before
    assert fingerprint(payload['optimizer'])!=fingerprint(original['optimizer'])
    assert jobs.disposable_resume_payload(original,'cpu')['optimizer']['state'][0]['step'].item()==1
