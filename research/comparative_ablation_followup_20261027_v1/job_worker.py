"""Admit one follow-up job, then call the unmodified registered runner."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from . import common


GATE_EXIT = 20
BUDGET_EXIT = 21
JOB_FIELDS = {'job_id', 'method_id', 'dataset', 'K', 'seed', 'extended'}


def validate_spec(job):
    if set(job) != JOB_FIELDS or sum(job == item for item in common.selected_jobs()) != 1:
        raise ValueError('Job differs from the exact six-field follow-up scope')
    return job


def assert_protected_T(result, dataset):
    access = result['task_access']
    if dataset == 'jena':
        members = access.get('members', [])
        if (access.get('new_T_numeric_values_parsed') != 0
                or {row['member'] for row in members} != {'x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy'}
                or any(row.get('tail_numeric_values_parsed') != 0 for row in members)):
            raise ValueError('Jena bounded-loader proof does not establish zero protected T numeric parsing')
    elif access.get('T_numeric_values_parsed') != 0:
        raise ValueError('Protected T numeric values were parsed or bounded-loader proof is missing')


def end_time():
    cutoff = datetime.fromisoformat(common.LAST_START).timestamp()
    path = common.REPORT / 'EXECUTION_CLOCK.json'
    if path.exists():
        clock = common.read(path)
        if clock['limit_seconds'] != common.LIMIT_SECONDS or clock['latest_start'] != common.LAST_START:
            raise ValueError('Follow-up clock limits changed')
        cutoff = min(cutoff, clock['training_started_unix'] + common.LIMIT_SECONDS)
    return cutoff


def run(job, attempt=1, verify_only=False):
    # Install before importing even the registered runtime, whose imports must
    # never create bytecode or modify historical campaign files.
    sys.dont_write_bytecode = True
    common.install_write_guard()
    from .controller import CampaignLock
    with CampaignLock(common.CACHE / 'worker.lock'):
        validate_spec(job)
        if attempt not in (1, 2):
            raise ValueError('Only one retry is registered')
        admission = common.validate_admission()
        if admission.get('status') != 'PASS':
            raise ValueError('Follow-up admission gate did not pass')
        common.guard_preserved()
        from research.comparative_ablation_20261025_v1 import run_one
        result_path = common.OLD_REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
        if verify_only:
            if not result_path.is_file():
                raise ValueError('Verification cannot create a missing result')
            result = run_one.run_job(job, attempt=attempt)
            if result['status'] != 'COMPLETE':
                raise ValueError('Existing result is not COMPLETE')
            assert_protected_T(result, job['dataset'])
            common.guard_preserved()
            return dict(status='VERIFIED_COMPLETE', job_id=job['job_id'], result_status=result['status'])
        if result_path.exists():
            raise ValueError('Existing COMPLETE must use verification without a training admission')
        if time.time() >= end_time():
            return dict(status='NOT_RUN_BUDGET', job_id=job['job_id'], exit_code=BUDGET_EXIT)
        for root in (common.OLD_REPORT, common.OLD_CACHE):
            if (root / 'jobs' / job['job_id'] / f'attempt_{attempt:03d}').exists():
                raise ValueError('Attempt directory already exists; refusing to repeat it')
        path = common.REPORT / 'EXECUTION_CLOCK.json'
        started = time.time()
        if started >= end_time():
            return dict(status='NOT_RUN_BUDGET', job_id=job['job_id'], exit_code=BUDGET_EXIT)
        if not path.exists():
            common.write_new(path, dict(training_started_unix=started,
                started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
                first_job_id=job['job_id'], limit_seconds=common.LIMIT_SECONDS,
                latest_start=common.LAST_START, admission=admission))
        receipt_path = common.CACHE / 'worker_receipts' / f'{job["job_id"]}_attempt{attempt}.json'
        receipt = dict(job_id=job['job_id'], attempt=attempt, admitted_unix=started,
                       status='RUNNING', exit_code=1)
        common.write_new(common.CACHE / 'worker_admissions' / f'{job["job_id"]}_attempt{attempt}.json',
                         dict(job=job, attempt=attempt, admitted_unix=started, pid=os.getpid()))
        try:
            # In particular, do not pass the campaign cutoff as the per-job
            # deadline. The original default permits the current job to finish.
            result = run_one.run_job(job, attempt=attempt)
            if result['status'] != 'COMPLETE' or result['job'] != job:
                raise ValueError('Registered runner returned an invalid COMPLETE result')
            assert_protected_T(result, job['dataset'])
            common.guard_preserved()
            receipt.update(status='COMPLETE', exit_code=0)
        except BaseException as exc:
            receipt.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            try:
                common.guard_preserved()
            except BaseException as preservation_error:
                receipt.update(status='STOPPED', exit_code=GATE_EXIT,
                               preservation_error=str(preservation_error))
            if isinstance(exc, (AssertionError, ValueError, FileExistsError)):
                receipt.update(status='STOPPED', exit_code=GATE_EXIT)
        finally:
            receipt.update(finished_unix=time.time(), wall_seconds=time.time() - started)
            common.write_new(receipt_path, receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(common.read(args.job), args.attempt, args.verify_only)
    except BaseException as exc:
        result = dict(status='STOPPED', exit_code=GATE_EXIT, error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result.get('exit_code', 0)


if __name__ == '__main__':
    raise SystemExit(main())
