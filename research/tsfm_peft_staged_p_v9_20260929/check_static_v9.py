"""One reserved syntax/provenance check session; no model imports or inference."""
import argparse
import ast
import json
import subprocess

from runtime_v9 import HERE, ROOT, Job, digest, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
with Job(args.name, category='cpu_check', reserve_seconds=30):
    parsed = []
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed.append(path.name)
    preservation_path = HERE / 'preservation.json'
    if not preservation_path.exists():
        paths = subprocess.run(['git', 'ls-files', 'research/tsfm_peft_*'], cwd=ROOT,
                               text=True, capture_output=True, check=True).stdout.splitlines()
        paths = [p for p in paths if not p.startswith(HERE.relative_to(ROOT).as_posix() + '/')]
        protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        save_json(preservation_path, {'base_commit': protocol['base_commit'],
                  'tracked_files': {p: digest(ROOT / p) for p in paths},
                  'scope': 'Existing tracked tsfm_peft research artifacts before v9 model execution'})
    preserved = json.loads(preservation_path.read_text(encoding='utf-8'))
    for path, expected in preserved['tracked_files'].items():
        assert digest(ROOT / path) == expected, 'Preserved file changed: ' + path
    result = {'parsed': parsed, 'model_execution': False,
              'preserved_files_checked': len(preserved['tracked_files'])}
    save_json(HERE / (args.name + '.json'), result)
    print(json.dumps(result), flush=True)
