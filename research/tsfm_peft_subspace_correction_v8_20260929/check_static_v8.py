"""Reserve one bounded CPU static-check session, without importing model code."""
import argparse
import ast
from runtime_v8 import HERE, Job, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
with Job(args.name, category='cpu_check', reserve_seconds=30):
    parsed = []
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed.append(path.name)
    save_json(HERE / (args.name + '.json'), {'parsed': parsed, 'model_execution': False})
    print(parsed, flush=True)
