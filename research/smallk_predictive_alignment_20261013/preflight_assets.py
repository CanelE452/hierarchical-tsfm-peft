"""Read-only asset checks and one new, immutable preflight receipt."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/forecast_generalization_study_20261013'))
import experiment as ex
import psutil
import torch


def command(arguments):
    return subprocess.check_output(arguments, cwd=ROOT, text=True, encoding='utf-8').strip()


def asset(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': ex.sha(path)}


def main():
    started = time.perf_counter()
    destination = HERE / 'PREFLIGHT.json'
    if destination.exists():
        raise FileExistsError('Immutable preflight already exists')
    protected = ex.access_guard()
    record = {
        'status': 'ASSETS_CONFIRMED_NOT_EXPERIMENT_COMPLETE',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'started_unix': 1791890784,
        'started_unix_source': 'Actual active goal createdAt, returned by get_goal',
        'actual_head': command(['git', 'rev-parse', 'HEAD']),
        'branch': command(['git', 'branch', '--show-current']),
        'tracked_status': command(['git', 'status', '--short', '--untracked-files=no']),
        'protected_TEST_paths_guarded_not_opened': protected,
        'plan': asset(HERE / 'PLAN_KO.md'),
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda_build': torch.version.cuda,
        'gpu_inventory': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
        'free_disk_bytes': shutil.disk_usage(ROOT).free,
        'python_processes': [],
        'legacy_data': {},
        'old_checkpoint_assets': [],
        'preserve_files': [],
        'new_data_downloaded': False,
        'new_data_paths_expected_not_claimed_present': ['UCI374 Appliances', 'UCI501 Aotizhongxin'],
        'resource_caps': {'gpu_process_wall_seconds': 7200, 'cpu_only_wall_seconds': 7200,
                          'campaign_wall_seconds': 28800, 'storage_bytes': 12 * 1024**3,
                          'process_ram_or_allocated_vram_bytes': 8 * 1024**3},
    }
    assert record['actual_head'] == '95585946f99da9b50cbdadcadd7441ccbf6ff2e5'
    for process in psutil.process_iter(['pid', 'name', 'create_time']):
        if 'python' in (process.info['name'] or '').lower():
            record['python_processes'].append(process.info)
    for dataset in ('jena', 'robin'):
        data = ex.load_data(dataset, sizes=(256, 64, None))
        old_contract = ROOT / f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json'
        record['legacy_data'][dataset] = {
            'contract': asset(old_contract), 'source': asset(data['source']),
            'rows': len(data['physical']), 'C': len(data['columns']), 'columns': data['columns'],
            'bounds': {s: data[s + '_bounds'] for s in ('fit', 'tune', 'val')},
            'origin_counts': {s: len(o) for s, o in data['origins'].items()},
            'origin_hashes': {s: ex.ahash(o) for s, o in data['origins'].items()},
            'mean_sha256': ex.ahash(data['mean']), 'std_sha256': ex.ahash(data['std']),
            'physical_sha256': ex.ahash(data['physical']), 'mask_sha256': ex.ahash(data['mask']),
            'VAL_role': 'Previously repeated development diagnostic; not independent generalization',
        }
    resolved_path = ROOT / 'research/forecast_generalization_study_20261013/run_resolved_v1/RESULT.json'
    resolved = ex.read(resolved_path)
    assert resolved['status'] == 'COMPLETE'
    for dataset, channels in (('jena', 21), ('robin', 17)):
        for seed in (92601, 92602):
            for kind, k in (('PCA', channels // 4), ('IDENTITY', channels)):
                cases = [c for c in resolved['cases'] if c['dataset'] == dataset and c.get('seed') == seed
                         and c.get('objective') == 'PRED' and c.get('config') == 'BASE'
                         and c.get('length') == 512 and c.get('receptive_field') == 127
                         and c.get('k') == k and c.get('initialization', {}).get('kind') == kind]
                assert len(cases) == 1, (dataset, seed, kind, len(cases))
                case = cases[0]
                checkpoint = next(c for c in case['training']['checkpoints'] if c['update'] == case['selected_update'])
                actual = asset(checkpoint['path'])
                assert actual['sha256'] == checkpoint['sha256']
                record['old_checkpoint_assets'].append({**actual, 'dataset': dataset, 'seed': seed,
                                                        'kind': kind, 'K': k, 'update': case['selected_update']})
    snapshot = ex.previous.BOLT_DIRECTORY
    record['bolt_snapshot'] = {'directory': snapshot.relative_to(ROOT).as_posix(),
                               'files': [asset(p) for p in sorted(snapshot.rglob('*')) if p.is_file()]}
    assert record['bolt_snapshot']['files']
    # Preserve the entire published prior study and the unrelated tracked edit.
    paths = [p for p in (ROOT / 'research/forecast_generalization_study_20261013').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT / 'research/forecast_component_study_20261013/component_experiment.py',
              ROOT / 'research/past_only_repr_diagnosis_20261012/common.py',
              ROOT / '_docs/history/2026-10-11.md']
    paths += [ROOT / item['path'] for item in record['old_checkpoint_assets']]
    record['preserve_files'] = [asset(p) for p in sorted(set(paths))]
    record['read_only_check_wall_seconds'] = time.perf_counter() - started
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(ex.jsonable(record), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': record['status'], 'HEAD': record['actual_head'],
                      'preserved_files': len(record['preserve_files']),
                      'checkpoint_assets': len(record['old_checkpoint_assets']),
                      'seconds': record['read_only_check_wall_seconds'],
                      'PREFLIGHT_sha256': ex.sha(destination)}))


if __name__ == '__main__':
    main()
