"""Verify and reference existing v7 arrays and predictions; no model execution."""
import json
from pathlib import Path
import subprocess
from runtime_v8 import HERE, ROOT, Job, digest, save_json


with Job('prepare_reuse01', category='data_prepare', reserve_seconds=120) as job:
    previous = ROOT / 'research/tsfm_peft_practical_controls_v7_20260928'
    if (HERE / 'reuse_manifest.json').exists():
        raise RuntimeError('Preserve the original reuse manifest')
    records = {}
    for dataset in ('robin', 'jena'):
        source = previous / f'data_contract_{dataset}.json'
        contract = json.loads(source.read_text(encoding='utf-8'))
        for part in ('trainval', 'test'):
            entry = contract[part]
            path = Path(entry['path'])
            if not path.is_absolute():
                path = ROOT / path
            assert digest(path) == entry['sha256']
        save_json(HERE / source.name, contract)
        evaluation = previous / f'{dataset}_eval01.json'
        records[dataset] = {'contract': str(source), 'contract_sha256': digest(source),
                            'evaluation': str(evaluation), 'evaluation_sha256': digest(evaluation),
                            'scope': 'Previously exposed development data; no independent TEST claim'}
        job.heartbeat(f'{dataset} existing array hashes verified', 0)
    names = subprocess.check_output(['git', 'ls-files', 'research/tsfm_peft_*'], cwd=ROOT, text=True).splitlines()
    names = [name for name in names if not name.startswith(HERE.relative_to(ROOT).as_posix() + '/')]
    preserved = {name: digest(ROOT / name) for name in names if (ROOT / name).is_file()}
    save_json(HERE / 'preservation.json', {'base_commit': '0c9a08f22a9f01820859aa0e8648fc1601a21444',
                                         'tracked_files': preserved})
    save_json(HERE / 'reuse_manifest.json', {'datasets': records, 'no_new_data_download': True,
                                          'no_backbone_raw_or_array_copy': True})
    job.heartbeat('Data reuse and previous-research preservation recorded', 0)
