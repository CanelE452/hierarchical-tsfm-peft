"""Verify diagnostic provenance and existing-curve replay without model execution."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V9 = ROOT / 'research/tsfm_peft_staged_p_v9_20260929'
CACHE = ROOT / '.cache/tsfm_peft_followup_diagnosis_20260929'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    receipt_path = HERE / 'verification_receipt.json'
    if receipt_path.exists():
        raise RuntimeError('Preserve completed verification')
    started = time.perf_counter()
    receipt = {'status': 'running', 'category': 'cpu_check', 'reserved_seconds': 60,
               'fit_attempts': 0, 'optimizer_updates': 0, 'source_sha256': digest(__file__)}
    write(receipt_path, receipt)
    try:
        diagnostic = read(HERE / 'diagnostic.json')
        replay = read(HERE / 'replay01_receipt.json')
        assert replay['status'] == 'complete'
        assert digest(CACHE / 'executed_replay01.py') == replay['source_sha256']
        assert digest(HERE / 'PLAN.md') == replay['plan_sha256']
        assert digest(diagnostic['data_path']) == diagnostic['data_sha256']
        assert len(diagnostic['rows']) == 12
        assert not diagnostic['test_access'] and diagnostic['fit_attempts'] == diagnostic['optimizer_updates'] == 0
        output_rows = []
        for row in diagnostic['rows']:
            run = V9 / 'runs' / row['run']
            result = read(run / 'result.json')
            curve = read(run / 'curve.json')
            assert digest(run / 'result.json') == row['source_result_sha256']
            assert digest(row['checkpoint']) == row['checkpoint_sha256']
            assert digest(row['prediction']['path']) == row['prediction']['sha256']
            assert digest(row['probe_source']) == row['probe_sha256']
            assert row['alpha'] == 1
            epoch = row['epoch']
            expected = next(c for c in curve if c['epoch'] == epoch)
            assert abs(row['scores']['val']['mse'] - expected['val_mse']) <= 1e-8
            assert abs(row['scores']['val']['mae'] - expected['val_mae']) <= 1e-8
            if row['checkpoint_role'] == 'final':
                assert epoch == curve[-1]['epoch'] == result['epochs_completed']
            assert row['scores']['train_probe']['origins'] == 96
            assert row['scores']['val']['origins'] == 30
            output_rows.append({'family': row['family'], 'seed': row['seed'], 'state': row['checkpoint_role'],
                                'epoch': epoch, 'train_probe_mse': row['scores']['train_probe']['mse'],
                                'val_mse': row['scores']['val']['mse']})
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        assert subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip() == ''
        with (HERE / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        used = sum(p.stat().st_size for base in (HERE, CACHE) for p in base.rglob('*') if p.is_file())
        assert used < 100 * 1024**2
        verification = {'status': 'PASS', 'new_inference_or_scoring': False, 'rows_verified': 12,
                        'final_checkpoint_matches_existing_last_curve': True, 'tracked_files_changed': False,
                        'executed_source': str(CACHE / 'executed_replay01.py'), 'executed_source_sha256': replay['source_sha256'],
                        'post_execution_change': 'Added guard against overwriting existing diagnostic output; no rerun.',
                        'diagnostic_sha256': digest(HERE / 'diagnostic.json'), 'storage_bytes': used,
                        'gpu_seconds': replay['elapsed_s'], 'fits': 0, 'optimizer_updates': 0,
                        'commit_push_performed': False}
        write(HERE / 'final_verification.json', verification)
        print(json.dumps(verification), flush=True)
        receipt.update(status='complete', elapsed_s=time.perf_counter() - started)
        assert receipt['elapsed_s'] < 60
    except BaseException as error:
        receipt.update(status='failed', elapsed_s=time.perf_counter() - started, error=repr(error))
        raise
    finally:
        write(receipt_path, receipt)
