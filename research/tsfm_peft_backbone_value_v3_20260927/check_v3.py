"""Reserve an entire CPU synthetic session before importing or executing tests."""
import argparse
import ast
from runtime import HERE, Job, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    with Job(args.name, category='cpu_check', reserve_seconds=60, synthetic=True) as job:
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        from test_model import run_checks
        result = run_checks()
        result['ledger_id'] = job.id
        save_json(HERE / (args.name + '.json'), result)
        job.heartbeat(result['status'], result['optimizer_updates'])
        print(result)
        if result['status'] != 'pass':
            raise RuntimeError('CPU synthetic checks failed; preserve results before repair')


if __name__ == '__main__':
    main()
