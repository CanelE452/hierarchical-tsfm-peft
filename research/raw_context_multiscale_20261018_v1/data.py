"""Decode the existing permitted prefixes without hashing protected tails."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np

from .common import read, sha


PRIMITIVE = 'priority_b2_validation_20261016_v1/safe_jena_prefix.py'
PEMS = 'priority_core_followup_20261016_v1/pems_fixed_head_runner.py'
SUBSET = 'priority_pems_subset_20261016_v1/subset_minimum_runner.py'


def _module(path, label):
    spec = importlib.util.spec_from_file_location('raw_context_prefix_'+label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hashes(task, data, *, jena=False):
    result = {key: data.array_sha256(task[key]) for key in ('physical', 'observed', 'times')}
    result['columns'] = (hashlib.sha256(json.dumps(task['columns'], ensure_ascii=False).encode()).hexdigest()
                         if jena else data.array_sha256(np.asarray(task['columns'], dtype=str)))
    return result


def bounded_task(ctx, dataset):
    campaign = ctx['modules']['campaign']
    primitive = _module(campaign/PRIMITIVE, 'primitive')
    expected = read(ctx['repo']/'reports/b2_revin_20261018_v1/DATA_CONTRACT.json')['tasks'][dataset]
    if dataset == 'jena':
        old = read(campaign/'DATA_CONTRACT.json')['assets']['jena']
        source = ctx['asset']/primitive.SOURCE_REL
        metadata = ctx['asset']/primitive.CONTRACT_REL
        if source.stat().st_size != primitive.SOURCE_BYTES or sha(metadata) != primitive.CONTRACT_SHA:
            raise ValueError('Jena source size or ancestry contract changed')
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        values, members = {}, []
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate source members')
            for key in ('x', 'finite', 'times', 'mean', 'std', 'columns'):
                values[key], evidence = primitive.read_member(
                    archive, key+'.npy', 16772 if key in ('x', 'finite', 'times') else None)
                members.append(evidence)
        physical = (values['x']*values['std']+values['mean']).astype(np.float32)
        observed = values['finite'].copy()
        if observed.dtype != np.bool_:
            raise ValueError('Original Jena mask must remain bool')
        physical[~observed] = np.nan
        task = dict(name='jena', dataset='jena', physical=physical, observed=observed,
                    times=values['times'], columns=values['columns'].tolist(), L=512, H=48,
                    interval_seconds=600, period=144, data_sha256=old['data_sha256'],
                    array_sha256=copy.deepcopy(old['array_sha256']), metadata=copy.deepcopy(old['metadata']))
        actual = _hashes(task, ctx['modules']['data'], jena=True)
        prior = read(ctx['repo']/'reports/adapter_diagnostics_20261017_v2/B02/jena/RESULT.json')
        access = prior['access'] if 'access' in prior else next(
            value for value in prior.values() if isinstance(value, dict) and 'materialized_prefix_array_sha256' in value)
        if actual != access['materialized_prefix_array_sha256']:
            raise ValueError('Actual Jena prefix differs from the sealed previous prefix')
        historical_source_sha = primitive.SOURCE_SHA
    elif dataset == 'pems32':
        subset = _module(campaign/SUBSET, 'subset')
        pems = _module(campaign/PEMS, 'pems')
        plan = read(campaign/'priority_pems_subset_20261016_v1/PLAN.json')
        subset.require_sealed(plan)
        for relative, digest in plan['dependencies'].items():
            if sha(campaign/relative) != digest:
                raise ValueError('Sealed subset dependency changed: '+relative)
        old = plan['parent_task_contract']
        meta = old['metadata']
        source = ctx['modules']['old_cache']/meta['source_reference']
        if source.stat().st_size != meta['source_bytes']:
            raise ValueError('PeMS source size changed')
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        with zipfile.ZipFile(source) as archive:
            if archive.namelist() != ['data.npy']:
                raise ValueError('The original single PeMS source member is required')
            info = archive.getinfo('data.npy')
            with archive.open(info) as stream:
                raw, member = pems.prefix_payload(stream, info.file_size, primitive, rows=14284)
        physical = raw[:, :, 0].astype(np.float32)
        observed = np.isfinite(physical)
        physical[~observed] = np.nan
        parent = dict(name='pems08', dataset='pems08', physical=physical, observed=observed,
                      times=np.arange(14284, dtype=np.int64)*300*1000000000,
                      columns=old['columns'], L=512, H=12, interval_seconds=300, period=288,
                      data_sha256=old['data_sha256'], array_sha256=old['array_sha256'], metadata=meta)
        task = subset.subset_task(parent, plan, dict(np=np, data=ctx['modules']['data']))
        actual = _hashes(task, ctx['modules']['data'])
        if actual != expected['array_sha256']:
            raise ValueError('Actual PeMS subset prefix differs from the sealed cohort')
        members = [member]
        historical_source_sha = meta['source_sha256']
    else:
        raise ValueError('Only the registered Jena/PeMS32 cohorts are admitted')
    if before != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError('Source file changed while preparing its permitted prefix')
    if not np.isfinite(task['physical'][task['observed']]).all():
        raise ValueError('Observed physical values must be finite')
    if not np.all(np.diff(task['times']) == task['interval_seconds']*1000000000):
        raise ValueError('Original time grid changed')
    task.update(dataset=dataset, split_ranges=copy.deepcopy(expected['split_ranges']),
                materialized_rows=len(task['physical']), original_total_rows=expected['N'],
                materialized_prefix_array_sha256=actual)
    receipt = dict(dataset=dataset, materialized_rows=len(task['physical']),
                   actual_prefix_array_sha256=actual, inherited_data_sha256=task['data_sha256'],
                   historical_source_sha256_not_recomputed=historical_source_sha,
                   source_bytes=source.stat().st_size, members=members,
                   protected_T_numeric_values_parsed=0, full_source_hash_recomputed=False,
                   physical_IO_scope='Bounded numeric prefix; ZIP may buffer opaque additional bytes. No whole raw-source hash or numeric tail decoding.',
                   legacy_normalization_ancestry='Inherited original Jena cache scaling remains a limitation; current scalers fit only A_fit/A60.')
    return task, receipt
