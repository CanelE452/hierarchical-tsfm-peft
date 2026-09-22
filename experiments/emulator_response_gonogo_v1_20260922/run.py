#!/usr/bin/env python
"""Single entry point. Stages run in fresh subprocesses; completion is judged by stage receipts, not exit codes."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import argparse, importlib.metadata, platform, re, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulator_screen.common import *

GPU_STAGES = ['prepare', 'preflight_base', 'compress', 'preflight_emulator', 'calibrate', 'caches', 'cost_basis']
POST_STAGES = ['select', 'test']


def gpu_busy():
    """Another pure-compute GPU process (Type C) is present. Windows WDDM lists desktop C+G clients too."""
    smi = run_command(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], check=False)
    if smi.returncode: return []
    others = [r for r in smi.stdout.splitlines() if r.strip() and r.split(',')[0].strip() != str(os.getpid())]
    if os.name == 'nt':
        table = run_command(['nvidia-smi'], check=False).stdout
        compute = {m.group(1) for m in re.finditer(r'^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+C\s', table, re.M)}
        others = [r for r in others if r.split(',')[0].strip() in compute]
    return others


def seal():
    if (OUT/'SOURCE_SEAL.json').exists():
        assert_seal(); return
    for n in ['PROTOCOL_KO.md', 'MASTER_CLI.txt', 'reference_core.py', 'config.json']:
        if not (HERE/n).exists(): raise Blocked('MISSING_CONTRACT_FILE '+n)
    manifest = read_json(HERE/'BUNDLE_MANIFEST.json')['files']
    bundle = {n: sha(HERE/('config.json' if n == 'config_proposal.json' else ('tests/'+n if n.startswith('test_') else n))) == v['sha256']
              for n, v in manifest.items()}
    if not all(bundle.values()): raise Blocked(f'BUNDLE_FILE_CHANGED {bundle}')
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    configure_runtime(DEVICE, BASECFG['precision'])
    head = run_command(['git', 'rev-parse', 'HEAD'], cwd=REPO, check=False).stdout.strip()
    packages = {}
    for name in ['torch', 'chronos-forecasting', 'transformers', 'peft', 'numpy', 'pandas', 'scipy', 'matplotlib', 'huggingface-hub']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    write_json(OUT/'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages=packages, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), bf16=torch.cuda.is_bf16_supported(),
        existing_head=head, reference_code_commit=CFG['reference_code_commit'], reference_latest_commit=CFG['reference_latest_commit'],
        env=dict(PYTHONUTF8=os.environ.get('PYTHONUTF8'), CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        nvidia_smi=run_command(['nvidia-smi'], check=False).stdout, authenticity='REAL_MODEL_RUN', created_utc=time.time()))
    (OUT/'requirements-lock.txt').write_text(run_command([sys.executable, '-m', 'pip', 'freeze'], check=False).stdout, encoding='utf-8')
    write_json(OUT/'CONFIG.json', dict(config=CFG, settings=SET, base_config_used_for_data_and_model=BASECFG))
    write_json(OUT/'SOURCE_SEAL.json', dict(training_code=code_hashes(TRAINING_CODE), reporting_code=code_hashes(REPORTING_CODE),
        docs=code_hashes(['PROTOCOL_KO.md', 'MASTER_CLI.txt', 'README_KO.md', 'tests/test_reference_core.py']),
        reused=reused_hashes(), config_hash=digest_object(CFG), settings_hash=digest_object(SET), bundle_files_match=bundle,
        git_head_at_seal=head, created_utc=time.time()))


def run_stage(name, extra=()):
    cmd = [sys.executable, str(Path(__file__).resolve()), 'stage', '--name', name, *extra]
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['seal', 'all', 'stage', 'report', 'verify'])
    p.add_argument('--name'); p.add_argument('--arm'); p.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.command == 'seal':
        seal(); print('SEALED'); return
    if a.command == 'stage':
        configure_runtime(DEVICE, BASECFG['precision'])
        from emulator_screen import stages
        label = a.name if a.name != 'fit' else f'fit_{a.arm}_{a.seed}'
        try:
            if a.name == 'fit':
                stages.stage_fit(a.arm, a.seed)
            else:
                getattr(stages, 'stage_'+a.name)()
        except BaseException as e:
            write_json(CACHE/'stages'/f'{label}.error.json', dict(error=repr(e), traceback=traceback.format_exc(), utc=time.time()))
            raise
        return
    if a.command == 'report':
        from emulator_screen import report
        report.build(); return
    if a.command == 'verify':
        from emulator_screen import verify
        print(verify.verify()['status']); return
    # all: seal -> stages -> 12 fits -> selection -> TEST. Report/figures/verify/publish are separate commands.
    lock = CACHE/'CONTROLLER.lock'; CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open('x') as f: f.write(str(os.getpid()))
    except FileExistsError:
        raise Blocked('CONTROLLER_LOCK_EXISTS: do not delete a live lock')
    try:
        busy = gpu_busy()
        if busy: raise Blocked(f'GPU_BUSY: {busy}; not terminating or waiting')
        seal()
        for name in GPU_STAGES:
            if name != 'cost_basis' and stage_done(name): continue
            if name == 'cost_basis' and (OUT/'COST_BASIS.json').exists(): continue
            rc_ = run_stage(name)
            done = (OUT/'COST_BASIS.json').exists() if name == 'cost_basis' else stage_done(name)
            if not done: raise Blocked(f'STAGE_FAILED {name} (exit {rc_}); inspect the log; no automatic rescue')
        for i, seed in enumerate(CFG['seeds']):
            arms = ARMS if i == 0 else list(reversed(ARMS))
            for arm in arms:
                fp = OUT/'fits'/f'{arm}_{seed}.json'
                if fp.exists() and read_json(fp).get('complete'): continue
                busy = gpu_busy()
                if busy: raise Blocked(f'GPU_BUSY before {arm}/{seed}: {busy}')
                rc_ = run_stage('fit', ['--arm', arm, '--seed', str(seed)])
                if not (fp.exists() and read_json(fp).get('complete')):
                    raise Blocked(f'FIT_FAILED {arm}/{seed} (exit {rc_}); partial work preserved; no automatic replay')
        for name in POST_STAGES:
            rc_ = run_stage(name)
            done = (OUT/'MODEL_SELECTION.json').exists() if name == 'select' else (OUT/'PREDICTIONS_MANIFEST.json').exists()
            if not done: raise Blocked(f'STAGE_FAILED {name} (exit {rc_})')
        print('ALL_TRAINING_AND_TEST_PREDICTIONS_COMPLETE', flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except Blocked as e:
        write_json(OUT/'ERROR.json', dict(error=str(e), traceback=traceback.format_exc(), utc=time.time()))
        print(f'BLOCKED: {e}', file=sys.stderr); raise SystemExit(2)
