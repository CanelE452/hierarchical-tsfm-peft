"""Sealed jobs, atomic receipts, and persistent finite campaign accounting."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


STATES = {'PLANNED', 'RUNNING', 'COMPLETE', 'REUSED_VERIFIED', 'FAILED',
          'BLOCKED_MISSING_ASSET', 'BLOCKED_ACCESS_OR_LICENSE', 'SKIPPED_BUDGET',
          'PARTIAL_CHECKPOINTED', 'NOT_RUN_DEPENDENCY'}
SUCCESS = {'COMPLETE', 'REUSED_VERIFIED'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_plan(plan):
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    return dict(body, plan_sha256=digest(body))


def validate_plan(plan, repo=None):
    if seal_plan(plan)['plan_sha256'] != plan.get('plan_sha256'):
        raise ValueError('Sealed plan identity mismatch')
    ids = [j['job_id'] for j in plan['jobs']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate job ID')
    seen = set()
    for j in plan['jobs']:
        if not set(j.get('dependency_ids', [])).issubset(seen):
            raise ValueError('Dependency missing or graph is not topologically ordered')
        seen.add(j['job_id'])
    if repo is not None:
        for name, expected in plan['source_hashes'].items():
            if sha(Path(repo) / name) != expected:
                raise ValueError('Sealed source changed: ' + name)


def campaign_ledger(path, now=None):
    path = Path(path)
    if path.exists():
        return read(path)
    now = time.time() if now is None else now
    value = dict(campaign='adapter_diagnostics_20261017_v2', started_unix=now,
                 limits=dict(wall_seconds=21600, gpu_seconds=14400, wall_reserve=4320,
                             gpu_reserve=2880, download_bytes=5 * 1024**3,
                             cache_bytes=30 * 1024**3, cpu_workers=1, blas_threads=2),
                 gpu_seconds=0.0, cpu_worker_seconds=0.0, download_reserved_bytes=0,
                 lineage='informs_b2_why_where_20261017 STOPPED; prior usage is preserved')
    atomic_json(path, value)
    return value


def dispatch_allowed(ledger, expected_wall, expected_gpu, now=None, final=False):
    now = time.time() if now is None else now
    limits = ledger['limits']
    wall = limits['wall_seconds'] - (now - ledger['started_unix'])
    gpu = limits['gpu_seconds'] - ledger['gpu_seconds']
    return (expected_wall <= wall - (0 if final else limits['wall_reserve']) and
            expected_gpu <= gpu - (0 if final else limits['gpu_reserve']))


def finish_job(path, identity, result, artifact_root):
    result = dict(result)
    if result['status'] not in STATES:
        raise ValueError('Unregistered job state')
    artifacts = {}
    root = Path(artifact_root).resolve()
    for item in result.pop('artifact_paths', []):
        p = Path(item).resolve()
        relative = p.relative_to(root).as_posix()
        if not p.is_file() or p.is_symlink():
            raise ValueError('Missing or indirect artifact: ' + relative)
        artifacts[relative] = sha(p)
    if result['status'] in SUCCESS and not artifacts:
        raise ValueError('Completion requires actual artifacts')
    result.update(scientific_identity=identity, artifact_hashes=artifacts, finished_at=utc())
    atomic_json(path, result)
    return result


def bound_references_valid(value, private_roots):
    if isinstance(value, list):
        return all(bound_references_valid(item, private_roots) for item in value)
    if not isinstance(value, dict):
        return True
    reference = value.get('reference', value.get('path'))
    if isinstance(reference, str) and '://' in reference and 'sha256' in value:
        scheme, relative = reference.split('://',1)
        if scheme in ('cache','asset','repo','report'):
            if scheme not in private_roots:
                return False
            root = Path(private_roots[scheme]).resolve()
            target = (root/relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha(target)!=value['sha256']:
                return False
    return all(bound_references_valid(item, private_roots) for item in value.values())


def reusable(path, identity, artifact_root, private_roots=None):
    if not Path(path).exists():
        return False
    value = read(path)
    if value.get('scientific_identity') != identity or value.get('status') not in SUCCESS:
        return False
    root = Path(artifact_root).resolve()
    artifacts = value.get('artifact_hashes', {})
    if not artifacts:
        return False
    for name, expected in artifacts.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != expected:
            return False
        if p.suffix=='.json':
            if not bound_references_valid(read(p), private_roots or {}):
                return False
    return True


def planned_jobs(source_hashes):
    jobs = []
    def add(identifier, question, family, cohort='jena', seed=None, k=6,
            device='cpu', dependencies=(), seconds=60, gpu=0, **extra):
        body = dict(job_id=identifier, question_id=question, family=family,
                    dataset_cohort=cohort, cohort=cohort, seed=seed, K=k,
                    backbone_revision='772f3d25d38aec6d914c8949dab4462e2d46f5d8',
                    dependency_ids=list(dependencies), device=device,
                    expected_cost=dict(wall_seconds=seconds, gpu_seconds=gpu,
                                       status='PRE_PROFILE_CONSERVATIVE_ESTIMATE'), **extra)
        if extra.get('backbone')=='moirai2':
            body['backbone_revision']='30f43ff08c8494f4943ae1521e9d4e94a0fbb389'
        elif extra.get('backbone')=='chronos2':
            body['backbone_revision']='ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a'
        namespace_files={
            'A01':{'data_audit.py'},'A04':{'data_audit.py'},
            'A02':{'existing_jobs.py'},'A03':{'existing_jobs.py','gpu_endpoints.py'},
            'B01':{'existing_jobs.py'},'B02':{'ridge_jobs.py','data_audit.py'},
            'B03':{'new_jobs.py','existing_jobs.py'},'C00':{'existing_jobs.py'},
            'C01':{'existing_jobs.py'},'C02':{'new_jobs.py','existing_jobs.py'},
            'D01':{'backbones.py','backbone_jobs.py','functional_scaler.py'},
            'D02':{'backbones.py','backbone_jobs.py','functional_scaler.py','new_jobs.py','existing_jobs.py'},
            'D03':{'backbones.py','backbone_jobs.py','existing_jobs.py','data_audit.py'},
            'E01':{'existing_jobs.py'},'E02':{'costs.py','existing_jobs.py'},
            'E03':{'backbone_costs.py','backbones.py','functional_scaler.py','backbone_jobs.py','existing_jobs.py','costs.py'}}
        relevant={name:value for name,value in source_hashes.items()
                  if ('research/adapter_diagnostics_20261017_v2/' not in name or
                      Path(name).name in namespace_files[question]) and
                     (not name.endswith('BACKBONE_SOURCES.json') or question.startswith('D') or question=='E03')}
        body['scientific_source_hashes']=relevant
        body['scientific_identity'] = digest(dict(recipe=body, sources=relevant))
        body.update(status='PLANNED', started_at=None, finished_at=None,
                    consumed_resource=None, artifact_hashes={}, reason=None)
        jobs.append(body)
    add('A01_data', 'A01', 'DATA_AUDIT', cohort='all', seconds=30)
    add('A04_solver', 'A04', 'RIDGE_METRIC_AUDIT', cohort='synthetic', seconds=10)
    for seed in (92601, 92602):
        add(f'A02_cap_s{seed}', 'A02', 'B2_CAP_POINTWISE', seed=seed,
            dependencies=['A01_data', 'A04_solver'], seconds=10)
    for family in ('B0', 'B2', 'B2_CAP_POINTWISE'):
        for seed in (92601, 92602):
            for endpoint in range(4):
                for split in ('A_fit', 'A_stop', 'V_select'):
                    add(f'A03_{family}_s{seed}_e{endpoint}_{split}', 'A03', family,
                        seed=seed, endpoint_index=endpoint, split=split,
                        dependencies=['A01_data'], device='cuda', seconds=90,gpu=90,
                        evaluation_branch='COMMON_CUDA_REPLAY_ALL_72; original 48 CPU records retained separately')
    for family in ('B0', 'B2', 'B2_CAP_POINTWISE'):
        for seed in (92601, 92602):
            add(f'B01_{family}_s{seed}', 'B01', family, seed=seed,
                dependencies=['A01_data', 'A04_solver'], seconds=900)
    for cohort, k in (('jena', 6), ('pems32', 8)):
        add(f'B02_{cohort}', 'B02', 'OWN_VS_CROSS_CHANNEL_RIDGE', cohort=cohort,
            k=k, dependencies=['A01_data', 'A04_solver'], seconds=300)
    for name in ('moirai2', 'chronos2'):
        add(f'D01_{name}', 'D01', name, backbone=name, cohort='synthetic',
            device='cuda', seconds=300, gpu=300)
    for seed in (92601, 92602):
        add(f'B03_s{seed}', 'B03', 'SHARED_VS_HORIZON', seed=seed,
            dependencies=['A01_data', 'A04_solver'], device='cuda', seconds=900, gpu=600)
    add('C00_pems_pair_profile', 'C00', 'REGISTERED_PAIR_PROFILE', cohort='pems32', k=8,
        dependencies=['A01_data'], seconds=60)
    for family in ('B0', 'B2'):
        add(f'C01_pems32_{family}', 'C01', family, cohort='pems32', seed=92601,
            k=8, device='cpu', dependencies=['A01_data','C00_pems_pair_profile'], seconds=1800)
    add('C00_pems_gpu_pair_profile','C00','DEVICE_CONTINUATION_PAIR_PROFILE',cohort='pems32',k=8,
        device='cuda',dependencies=['A01_data'],seconds=90,gpu=90)
    for family in ('B0','B2'):
        add(f'C01_pems32_cuda_{family}','C01',family,cohort='pems32',seed=92601,k=8,
            device='cuda',dependencies=['A01_data','C00_pems_gpu_pair_profile'],seconds=1800,gpu=1800,
            continuation='CPU_to_CUDA_device_branch_not_bitwise_CPU_resume')
    for seed in (92601, 92602):
        add(f'C02_s{seed}', 'C02', 'PCA_VS_RANDOM_QR', seed=seed, device='cuda',
            dependencies=['A01_data'], seconds=1800, gpu=1800)
    for seed in (92601, 92602):
        add(f'D02_moirai_s{seed}', 'D02', 'MOIRAI_MATCHED_TRIO', seed=seed,
            backbone='moirai2', device='cuda', dependencies=['A01_data', 'D01_moirai2'],
            seconds=1800, gpu=1800)
    for cohort, k in (('jena', 21), ('pems08', 170)):
        add(f'D03_{cohort}', 'D03', 'CHRONOS2_NATIVE_RAW', cohort=cohort, k=k,
            backbone='chronos2', device='cuda', dependencies=['A01_data', 'D01_chronos2'],
            seconds=120, gpu=120)
    for family in ('B0', 'B2'):
        for batch in (1, 8):
            add(f'E01_pems32_{family}_b{batch}', 'E01', family, cohort='pems32', seed=92601,
                k=8, batch=batch, dependencies=[f'C01_pems32_cuda_{family}'], seconds=120)
    add('E01_jena_matched', 'E02', 'JENA_MATCHED_COST', dependencies=['A02_cap_s92601'], seconds=600)
    add('E03_jena_gpu','E03','NATIVE_BACKBONE_MATCHED_COST',device='cuda',seconds=300,gpu=300,
        dependencies=['A01_data'])
    return jobs
