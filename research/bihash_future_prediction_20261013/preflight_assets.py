"""Read actual assets and preserve old source/result fingerprints before new GPU work."""
from pathlib import Path
import subprocess
import sys
import time

import psutil
import torch
import runtime as rt


def command(*args):
    return subprocess.check_output(args, cwd=rt.ROOT, text=True, encoding='utf-8', errors='replace').strip()


def artifact(path):
    path = Path(path)
    return {'path': path.absolute().relative_to(rt.ROOT).as_posix(),
            'resolved_target': path.resolve().relative_to(rt.ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': rt.sha(path)}


def main():
    limit = rt.Limit('preflight_v2', gpu=False)
    try:
        from data import access_guard
        protected = access_guard()
        folders = [rt.ROOT / 'research/smallk_predictive_alignment_20261013',
                   rt.ROOT / '.cache/smallk_predictive_alignment_20261013',
                   rt.ROOT / 'research/bihash_future_alignment_20261013',
                   rt.ROOT / '.cache/bihash_future_alignment_20261013']
        preserved = []
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    preserved.append(artifact(path))
            limit.check()
        for relative in ('research/forecast_component_study_20261013/component_experiment.py',
            'research/forecast_generalization_study_20261013/experiment.py',
            'research/forecast_generalization_study_20261013/diagnose_constant_gradient.py',
            'research/past_only_repr_diagnosis_20261012/common.py', '_docs/history/2026-10-11.md'):
            preserved.append(artifact(rt.ROOT / relative))
        model = rt.ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
        if not model.is_dir(): raise FileNotFoundError('ACTUAL_BOLT_SNAPSHOT_MISSING')
        model_files = [artifact(path) for path in sorted(model.rglob('*')) if path.is_file()]
        if not any(p['path'].endswith('.safetensors') for p in model_files):
            raise FileNotFoundError('ACTUAL_BOLT_WEIGHTS_MISSING')
        legacy = {}
        for dataset in ('jena', 'robin'):
            path = rt.ROOT / f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json'
            contract = rt.read(path)
            source = rt.ROOT / contract['trainval']['path']
            if rt.sha(source) != contract['trainval']['sha256']:
                raise ValueError('LEGACY_SOURCE_SHA_CHANGED: ' + dataset)
            legacy[dataset] = {'contract': artifact(path), 'TRAIN_DEVVAL_source': artifact(source),
                               'protected_TEST_path_metadata_only': str((rt.ROOT / contract['test']['path']).resolve())}
        processes = []
        for process in psutil.process_iter(['pid', 'name', 'create_time']):
            if (process.info['name'] or '').lower() in ('python.exe', 'pythonw.exe'):
                processes.append({**process.info, 'is_preflight': process.pid == psutil.Process().pid})
        git_status = command('git', 'status', '--porcelain=v1', '--untracked-files=no')
        untracked = command('git', 'ls-files', '--others', '--exclude-standard')
        payload = {'status': 'ACTUAL_ASSETS_CHECKED_NOT_EXPERIMENT_COMPLETE', 'created_unix': time.time(),
            'goal_started_unix': rt.GOAL_STARTED_UNIX, 'actual_HEAD': command('git', 'rev-parse', 'HEAD'),
            'branch': command('git', 'branch', '--show-current'), 'tracked_dirty': git_status.splitlines(),
            'untracked_count': len(untracked.splitlines()), 'untracked_list_sha256': __import__('hashlib').sha256(untracked.encode()).hexdigest(),
            'python': sys.version, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'GPU_actual': command('nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'),
            'disk_free_bytes': __import__('shutil').disk_usage(rt.ROOT).free,
            'python_processes': processes, 'protected_TEST_paths_guarded_not_opened': protected,
            'legacy': legacy, 'bolt_snapshot': {'path': model.relative_to(rt.ROOT).as_posix(), 'files': model_files},
            'preserved_existing_artifacts': preserved, 'preserved_count': len(preserved),
            'PLAN': artifact(rt.HERE / 'PLAN_KO.md'), 'source_notes': artifact(rt.HERE / 'SOURCE_NOTES_KO.md'),
            'pilot': {'receipt_actual_sum_seconds': rt.PILOT_MEASURED_GPU_SECONDS,
                'fixture_unmetered': True, 'budget_reserve_seconds_not_measured': rt.PILOT_UNMETERED_RESERVE_SECONDS,
                'AUDIT': artifact(rt.HERE / 'PILOT_AUDIT_v1.json'), 'counts_as_primary_B_joint_training': False},
            'limits': {'GPU_process_wall_seconds': rt.GPU_SECONDS, 'CPU_process_wall_seconds': rt.CPU_SECONDS,
                       'campaign_wall_seconds': rt.CAMPAIGN_SECONDS, 'new_storage_bytes': rt.STORAGE_BYTES,
                       'process_RSS_allocated_VRAM_bytes': rt.PROCESS_BYTES},
            'restrictions': ['no protected TEST', 'no original stopped campaign resume', 'no overwrites',
                             'no large backbone training', 'no automatic commit/push']}
        rt.write(rt.HERE / 'PREFLIGHT.json', payload)
        print(rt.json_line({'PREFLIGHT': artifact(rt.HERE / 'PREFLIGHT.json'), 'preserved': len(preserved),
                           'HEAD': payload['actual_HEAD'], 'GPU': payload['GPU_actual'], 'python_processes': processes}), flush=True)
        limit.close()
    except Exception as error:
        limit.close('FAILED_PRESERVED', error)
        raise


if __name__ == '__main__': main()
