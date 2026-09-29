"""Bind rounded document values to the completed result JSON; no model execution."""
import argparse
import json
from pathlib import Path

from runtime_v9 import HERE, ROOT, Job, digest, save_json


def receipt(path):
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': digest(path)}


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
with Job(args.name, category='cpu_analysis', reserve_seconds=30):
    target = HERE / 'numeric_claims.json'
    if target.exists():
        raise FileExistsError('Preserve the existing claim manifest')
    source = HERE / 'report_values.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    artifacts = [HERE / 'TOPIC_DECISION.md', HERE / 'METHOD_UPDATE.md', ROOT / 'README.md']
    claims = []

    def add_if_rendered(path, keys, value, fmt, multiplier=1):
        rendered = format(value * multiplier, fmt)
        if rendered in path.read_text(encoding='utf-8'):
            claims.append({'label': '/'.join(map(str, keys)) + '@' + path.name,
                           'source': receipt(source), 'json_path': keys, 'value': value,
                           'artifact': receipt(path)['path'], 'rendered': rendered,
                           'format_spec': fmt, 'multiplier': multiplier})

    for artifact in artifacts:
        for dataset, values in report['datasets'].items():
            for role, periods in values['roles'].items():
                for period, point in periods.items():
                    for metric in ('mse', 'mae'):
                        add_if_rendered(artifact, ['datasets', dataset, 'roles', role, period, metric], point[metric], '.6f')
            for role, refs in values['combined_changes'].items():
                for ref, change in refs.items():
                    key = ['datasets', dataset, 'combined_changes', role, ref, 'relative_mse_change_pct']
                    value = change['relative_mse_change_pct']
                    if value:
                        add_if_rendered(artifact, key, value, '.2f')
                        if value < 0:
                            add_if_rendered(artifact, key, value, '.2f', -1)
            for role, periods in values['space_scores'].items():
                for period, point in periods.items():
                    for metric in ('p_retained_space_mse', 'q_residual_space_mse'):
                        add_if_rendered(artifact, ['datasets', dataset, 'space_scores', role, period, metric], point[metric], '.6f')
        for dataset, result in report['cost']['results'].items():
            for index, group in enumerate(result['summary']['groups']):
                for metric, scale in [('mean_seed_median_ms', 1), ('mean_seed_median_throughput', 1),
                                      ('mean_seed_median_peak_allocated_bytes', 1 / 2**20)]:
                    add_if_rendered(artifact, ['cost', 'results', dataset, 'summary', 'groups', index, metric],
                                    group[metric], '.2f', scale)
    assert all(any(c['artifact'] == receipt(path)['path'] for c in claims) for path in artifacts)
    save_json(target, {'artifacts': [receipt(p) for p in artifacts], 'sources': [receipt(source)],
                      'claims': claims, 'scope': 'Registered rounded accuracy/change/space/cost values that occur in the documents; not exhaustive prose verification'})
    print(json.dumps({'claims': len(claims), 'artifact': str(target)}), flush=True)
