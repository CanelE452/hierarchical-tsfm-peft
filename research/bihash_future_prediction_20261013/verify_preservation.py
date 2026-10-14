"""Recheck the preflight's old artifacts without opening protected TEST data."""
from pathlib import Path
import argparse
import subprocess
import time

import runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('INVALID_FRESH_TAG')
    folder = rt.HERE / 'decision_checks_v1' / ('preservation_' + args.tag)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    limit = rt.Limit('preservation_' + args.tag, gpu=False)
    preflight_path = rt.HERE / 'PREFLIGHT.json'
    preflight_sha = rt.sha(preflight_path)
    preflight = rt.read(preflight_path)
    protected = {Path(p).resolve() for p in preflight['protected_TEST_paths_guarded_not_opened']}
    result = {'scope': 'Only existing artifacts named by this study preflight; not study completion.',
              'started_unix': started, 'preflight_sha256': preflight_sha,
              'validator_source_sha256': rt.sha(__file__), 'failures': [],
              'checked_artifacts': 0, 'checked_bytes': 0,
              'protected_TEST_opened': False, 'goal_complete': False}
    status = 'COMPLETE'
    error = None
    try:
        actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=rt.ROOT, text=True).strip()
        result['actual_HEAD'] = actual_head
        if actual_head != preflight['actual_HEAD']:
            result['failures'].append({'item': 'HEAD', 'expected': preflight['actual_HEAD'], 'actual': actual_head})
        entries = preflight['preserved_existing_artifacts']
        result['expected_artifacts'] = len(entries)
        for index, entry in enumerate(entries):
            path = rt.ROOT / entry['path']
            if path.resolve() in protected:
                raise RuntimeError('PROTECTED_TEST_IN_PRESERVATION_LIST')
            if not path.is_file():
                result['failures'].append({'path': entry['path'], 'reason': 'MISSING'})
                continue
            actual_sha = rt.sha(path)
            actual_bytes = path.stat().st_size
            result['checked_artifacts'] += 1
            result['checked_bytes'] += actual_bytes
            if actual_sha != entry['sha256'] or actual_bytes != entry['bytes']:
                result['failures'].append({'path': entry['path'], 'reason': 'CHANGED',
                                           'expected_sha256': entry['sha256'], 'actual_sha256': actual_sha,
                                           'expected_bytes': entry['bytes'], 'actual_bytes': actual_bytes})
            if index % 250 == 0:
                limit.check()
                print(rt.json_line({'preserved_checked': result['checked_artifacts'],
                                    'expected': len(entries)}), flush=True)
        if rt.sha(preflight_path) != preflight_sha:
            raise RuntimeError('PREFLIGHT_CHANGED_DURING_CHECK')
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc)
    result.update(status='PASS' if not result['failures'] and error is None else 'FAIL',
                  execution_status=status, error=error, finished_unix=time.time(),
                  resource_receipt=limit.close(status, error))
    rt.write(folder / 'RESULT.json', result)
    print(rt.json_line({'preservation_result': str(folder / 'RESULT.json'),
                        'status': result['status'], 'checked_artifacts': result['checked_artifacts'],
                        'failures': len(result['failures'])}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
