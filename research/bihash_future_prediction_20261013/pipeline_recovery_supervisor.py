"""Fill the missing fresh-data branch after the live SML attempt terminates.

The existing attempt and its supervisor are observed, never killed or restarted.
The completed SML conditions are reused only through a separately audited merge.
"""
from pathlib import Path
import argparse
import os
import time

import psutil
import runtime as rt
import pipeline_supervisor as orchestration


def wait_process(pid, birth, script, required_arguments, folder):
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - birth) > .001:
            raise RuntimeError('OBSERVED_PID_IDENTITY_CHANGED')
        arguments = process.cmdline()
        if not any(Path(arg).name == script for arg in arguments) or not all(arg in arguments for arg in required_arguments):
            raise RuntimeError('OBSERVED_PROCESS_ARGUMENTS_CHANGED')
    except psutil.NoSuchProcess:
        orchestration.emit({'observed_process_missing': pid, 'terminal_artifact_required': True})
        return
    started, last = time.time(), -60.
    while process.is_running():
        try:
            if abs(process.create_time() - birth) > .001:
                raise RuntimeError('OBSERVED_PID_REUSED')
            if process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        if (folder / 'HALT_BEFORE_NEXT_PHASE').exists():
            raise RuntimeError('USER_REQUESTED_QUEUE_HALT_PRESERVE_RUNNING_JOB')
        elapsed = time.time() - started
        if elapsed - last >= 45:
            orchestration.emit({'verified_wait': script, 'pid': pid, 'birth': birth,
                                'elapsed_seconds': elapsed, 'supervisor_pid': os.getpid()})
            last = elapsed
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh-pid', required=True, type=int)
    parser.add_argument('--fresh-created', required=True, type=float)
    parser.add_argument('--old-supervisor-pid', required=True, type=int)
    parser.add_argument('--old-supervisor-created', required=True, type=float)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('INVALID_FRESH_TAG')
    folder = rt.HERE / ('pipeline_' + args.tag)
    folder.mkdir(parents=True, exist_ok=False)
    stages = []
    output = {'started_unix': time.time(), 'supervisor_pid': os.getpid(), 'stages': stages,
              'supervisor_source_sha256': rt.sha(__file__),
              'observed_fresh_pid': args.fresh_pid, 'observed_fresh_created': args.fresh_created,
              'no_existing_process_restart_or_termination': True, 'goal_complete': False}
    method = rt.HERE / 'METHOD_LOCK.json'
    dev = rt.HERE / 'run_dev_v1/RESULT.json'
    original = rt.HERE / 'run_fresh_v1/RESULT.json'
    subset = rt.HERE / 'run_household_recovery_v1/RESULT.json'
    merged = rt.HERE / 'run_fresh_merged_v1/RESULT.json'
    probes = rt.HERE / 'run_probes_v1/RESULT.json'
    evaluation = rt.HERE / 'run_evaluate_v1/RESULT.json'
    cost = rt.HERE / 'run_cost_v1/RESULT.json'
    repair = rt.HERE / 'ingest_repairs_v1/household_download_v1_TERMINAL.json'
    status, error = 'COMPLETE_GPU_PHASES_AWAITING_INDEPENDENT_AUDIT_AND_REPORT', None

    def launch(label, arguments):
        if orchestration.launch(label, arguments, folder, stages):
            raise RuntimeError('CHILD_FAILED_PRESERVE: ' + label)

    try:
        orchestration.require_complete(dev, 'dev')
        lock_sha = rt.sha(method)
        output['METHOD_LOCK_sha256'] = lock_sha
        if not rt.read(method).get('locked'):
            raise RuntimeError('METHOD_LOCK_NOT_LOCKED')
        if rt.read(repair).get('status') != 'COMPLETE':
            raise RuntimeError('DOWNLOAD_REPAIR_NOT_TERMINAL_COMPLETE')
        output['data_repair'] = {'path': repair.relative_to(rt.ROOT).as_posix(), 'sha256': rt.sha(repair)}
        for path in (subset, merged, probes, evaluation, cost):
            if path.parent.exists():
                raise FileExistsError('FUTURE_TAG_ALREADY_EXISTS: ' + str(path.parent))
        wait_process(args.fresh_pid, args.fresh_created, 'run.py', ('fresh', 'fresh_v1'), folder)
        wait_process(args.old_supervisor_pid, args.old_supervisor_created, 'pipeline_supervisor.py', ('continuation_v1',), folder)
        old_queue_path = rt.HERE / 'pipeline_continuation_v1/RESULT.json'
        old_queue = rt.read(old_queue_path)
        if old_queue.get('status') != 'FAILED_PRESERVED' or old_queue.get('error') != 'FRESH_TERMINAL_FAILED_PRESERVE_PATH_RESULTS':
            raise RuntimeError('ORIGINAL_PIPELINE_NOT_EXPECTED_TERMINAL')
        value = rt.read(original)
        failures = value.get('failed_paths', [])
        if value.get('stage') != 'fresh' or value.get('status') != 'FAILED_PRESERVED':
            raise RuntimeError('ORIGINAL_FRESH_NOT_EXPECTED_TERMINAL')
        if any(value.get(key) for key in ('error', 'error_type', 'traceback')):
            raise RuntimeError('ORIGINAL_GLOBAL_FAILURE_REQUIRES_ANALYSIS')
        if not value.get('backbone_before_sha256') or value.get('backbone_before_sha256') != value.get('backbone_after_sha256'):
            raise RuntimeError('ORIGINAL_BACKBONE_NOT_VERIFIED_UNCHANGED')
        expected_sources = rt.read(dev)['source_hashes']
        if value.get('source_hashes') != expected_sources:
            raise RuntimeError('ORIGINAL_SOURCE_SEAL_DIFFERS_FROM_DEV')
        for relative, expected_sha in expected_sources.items():
            if rt.sha(rt.ROOT / relative) != expected_sha:
                raise RuntimeError('CURRENT_FROZEN_SOURCE_CHANGED: ' + relative)
        if len(failures) != 1 or failures[0].get('dataset') != 'uci235_household_hourly' or failures[0].get('stage') != 'data_prepare':
            raise RuntimeError('ADDITIONAL_UNRESOLVED_FAILURES_REQUIRE_ANALYSIS')
        if len(value.get('cases', [])) != 36 or {c['dataset'] for c in value['cases']} != {'uci274_sml2010'}:
            raise RuntimeError('SML_COMPLETED_SCOPE_NOT_36')
        if rt.sha(method) != lock_sha:
            raise RuntimeError('METHOD_LOCK_CHANGED_DURING_WAIT')
        launch('household_recovery_v1', ['run_fresh_subset.py', '--dataset', 'uci235_household_hourly',
               '--tag', 'household_recovery_v1', '--method-lock', str(method), '--data-repair-result', str(repair)])
        orchestration.require_complete(subset, 'fresh')
        launch('fresh_merge_v1', ['merge_fresh_results.py', '--producer-result', str(original),
               '--producer-result', str(subset), '--dev-result', str(dev), '--method-lock', str(method),
               '--tag', 'fresh_merged_v1'])
        orchestration.require_complete(merged, 'fresh')
        diagnostic = ['hash_memory_diagnostics.py', '--result', str(dev), '--result', str(merged),
                      '--tag', 'future_keys_full_v1', '--expected-cases', '174']
        # Preserve a diagnostic failure while continuing independent science paths.
        orchestration.launch('future_keys_full_v1', diagnostic, folder, stages)
        common = ['--dev-result', str(dev), '--fresh-result', str(merged), '--method-lock', str(method)]
        launch('probes_v1', ['run.py', '--stage', 'probes', '--tag', 'probes_v1', *common])
        orchestration.require_complete(probes, 'probes')
        launch('evaluate_v1', ['run.py', '--stage', 'evaluate', '--tag', 'evaluate_v1', *common,
                              '--probe-result', str(probes)])
        orchestration.require_complete(evaluation, 'evaluate')
        launch('cost_v1', ['run.py', '--stage', 'cost', '--tag', 'cost_v1', *common,
                          '--probe-result', str(probes), '--evaluation-result', str(evaluation)])
        orchestration.require_complete(cost, 'cost')
        launch('deployment_outputs_v1', ['verify_deployment_outputs.py', *common, '--probe-result', str(probes),
               '--evaluation-result', str(evaluation), '--cost-result', str(cost), '--tag', 'outputs_v1'])
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc)
        output['failure_type'] = type(exc).__name__
    output.update(status=status, error=error, finished_unix=time.time(),
                  supervisor_CPU_seconds=sum(psutil.Process().cpu_times()[:2]),
                  resource_note='Waiting is orchestration, not GPU work. Child terminal receipts are counted once.')
    rt.write(folder / 'RESULT.json', output)
    orchestration.emit({'pipeline_terminal': str(folder / 'RESULT.json'), 'status': status,
                        'error': error, 'goal_complete': False})
    return 0 if error is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
