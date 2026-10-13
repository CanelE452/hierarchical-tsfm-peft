"""Re-measure completed deployment paths after competing study processes finish."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import traceback
import experiment as ex
from repair_experiment import RetryLimit, configure


class EffectiveRidge(ex.nn.Module):
    def __init__(self, state, channels):
        super().__init__()
        self.register_buffer('weight', ex.tensor(state['weight']))
        self.register_buffer('bias', ex.tensor(state['bias']))
        self.channels = channels

    def forward(self, features):
        return (features@self.weight+self.bias).reshape(len(features), ex.H, self.channels)


def resolve(full_tag='full_v1', repair_tag='repair_v1'):
    full_path = ex.HERE/('run_'+full_tag)/'RESULT.json'
    repair_path = ex.HERE/('run_'+repair_tag)/'RESULT.json'
    full, repair = ex.read(full_path), ex.read(repair_path)
    if full['status'] not in ('PARTIAL_FAILED_PATHS', 'COMPLETE') or repair['status'] != 'COMPLETE':
        raise RuntimeError('Completed source/repair required for clean-cost measurement')
    resolved = {(c['dataset'], c['label']): {'case': c, 'source_path': full_path} for c in full['cases']}
    for case in repair['cases']:
        resolved[(case['dataset'], case['label'])] = {'case': case, 'source_path': repair_path}
    for dataset, check in repair['configuration_checks'].items():
        selected = check['chosen_config']
        resolved = {key: value for key, value in resolved.items() if not
            (key[0]==dataset and value['case']['stage']=='G2' and value['case']['config'] != selected)}
    records = [resolved[k] for k in sorted(resolved)]
    count = sum(len(r['case']['methods']) for r in records)
    if len(records) != 70 or count != 236:
        raise RuntimeError(f'RESOLVED_CASE_METHOD_COVERAGE {len(records)} {count}')
    return records, {'full': {'path': str(full_path.relative_to(ex.ROOT)), 'sha256': ex.sha(full_path)},
                     'repair': {'path': str(repair_path.relative_to(ex.ROOT)), 'sha256': ex.sha(repair_path)}}


def models(case):
    channels = case['methods'][0]['scores']['val']['normalized']['per_channel_mse']
    channels = len(channels)
    if case['label'] == 'RAW':
        artifact = case['selected_models']
        assert ex.sha(ex.ROOT/artifact['path']) == artifact['sha256']
        saved = ex.torch.load(ex.ROOT/artifact['path'], map_location='cpu', weights_only=True)
        self_state = saved['self_ridge']
        self_model = ex.SelfRidge(ex.np.zeros(tuple(self_state['weights'].shape)), ex.np.zeros(tuple(self_state['biases'].shape)))
        self_model.load_state_dict(self_state)
        return {'codec': None, 'raw_ridge': EffectiveRidge(saved['raw_bolt_ridge'], channels),
                'only_ridge': EffectiveRidge(saved['ridge_only'], channels), 'self_ridge': self_model.eval()}, artifact
    k, length, receptive_field = case['k'], case['length'], case['receptive_field']
    codec = ex.Codec(ex.np.zeros(channels, ex.np.float32), ex.np.zeros((channels, k), ex.np.float32), case['seed'], receptive_field).to(ex.DEVICE)
    if 'selected_models' in case:
        artifact = case['selected_models']
        assert ex.sha(ex.ROOT/artifact['path']) == artifact['sha256']
        saved = ex.torch.load(ex.ROOT/artifact['path'], map_location='cpu', weights_only=True)
        codec.load_state_dict(saved['codec'])
        native = ex.NativeShared(length).to(ex.DEVICE)
        native.load_state_dict(saved['native'])
        decoder = ex.nn.Linear(k, channels).to(ex.DEVICE)
        decoder.load_state_dict(saved['common_decoder'])
        result = {'codec': codec, 'native': native.eval().requires_grad_(False),
            'raw_ridge': EffectiveRidge(saved['raw_ridge'], channels),
            'native_raw_ridge': EffectiveRidge(saved['native_raw_ridge'], channels),
            'common_decoder': decoder.eval().requires_grad_(False)}
    else:
        artifact = case['reused_checkpoint'] if case['stage']=='G0' else next(cp for cp in case['training']['checkpoints'] if cp['update']==case['selected_update'])
        assert ex.sha(ex.ROOT/artifact['path']) == artifact['sha256']
        codec.load_state_dict(ex.torch.load(ex.ROOT/artifact['path'], map_location='cpu', weights_only=True)['codec'])
        result = {'codec': codec}
    codec.eval().requires_grad_(False)
    return result, {'path': artifact['path'], 'sha256': artifact['sha256']}


def measure_method(data, case, method, loaded, bolt, limit):
    codec = loaded['codec']
    name = method['method']
    length = case['length']
    kwargs = {}
    if name in ('BOLT', 'BOLT_RAW_R', 'BOLT_COMMON_D'):
        forecaster = bolt
    elif name in ('NATIVE', 'NATIVE_RAW_R'):
        forecaster = loaded['native']
    elif name == 'PERSISTENCE':
        forecaster = ex.previous.persist
    elif name in ('RAW_BOLT', 'RAW_BOLT_RAW_R'):
        forecaster = bolt
    elif name in ('RAW_R_ONLY', 'RAW_SELF_LAG'):
        forecaster = None
    else:
        raise ValueError('Unexpected method '+name)
    if name in ('BOLT_RAW_R', 'RAW_BOLT_RAW_R'):
        kwargs['ridge'] = loaded['raw_ridge']
    elif name == 'NATIVE_RAW_R':
        kwargs['ridge'] = loaded['native_raw_ridge']
    elif name == 'RAW_R_ONLY':
        kwargs['ridge'] = loaded['only_ridge']
    elif name == 'RAW_SELF_LAG':
        kwargs['self_model'] = loaded['self_ridge']
    elif name == 'BOLT_COMMON_D':
        kwargs['decoder'] = loaded['common_decoder']
    artifact = method['prediction']
    assert ex.sha(ex.ROOT/artifact['path']) == artifact['sha256']
    with ex.np.load(ex.ROOT/artifact['path'], allow_pickle=False) as saved:
        prediction = saved['val_prediction']
    return ex.latency(data, prediction, codec, forecaster, length, limit, **kwargs)


def run(tag, full_tag, repair_tag):
    configure()
    records, sources = resolve(full_tag, repair_tag)
    protected = ex.access_guard()
    output = ex.HERE/('run_'+tag)
    output.mkdir(exist_ok=False)
    limit = RetryLimit(tag)
    nvidia = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True)
    result = {'status': 'RUNNING', 'source_runs': sources, 'records': [], 'resolved_case_count': len(records),
        'resolved_method_count': 236, 'protected_test_guarded': protected, 'new_training_updates': 0,
        'nvidia_smi_before_model_load': {'exit_code': nvidia.returncode, 'stdout': nvidia.stdout, 'stderr': nvidia.stderr},
        'source_hashes': {str(p.relative_to(ex.ROOT)): ex.sha(p) for p in (Path(__file__), ex.HERE/'experiment.py', ex.HERE/'repair_experiment.py')},
        'purpose': 'Main cost record after supplemental study training ended; original costs preserved'}
    try:
        competing_python = [line for line in nvidia.stdout.splitlines() if 'python' in line.lower() and not line.lstrip().startswith(str(os.getpid())+',')]
        result['competing_python_compute_processes_before_load'] = competing_python
        if nvidia.returncode != 0 or competing_python:
            raise RuntimeError('CLEAN_COST_GPU_AVAILABILITY_OR_COMPETING_PYTHON_PROCESS')
        snapshots = output/'sources'
        snapshots.mkdir()
        result['source_snapshots'] = []
        for path, digest in result['source_hashes'].items():
            snapshot = snapshots/Path(path).name
            with snapshot.open('xb') as f:
                f.write((ex.ROOT/path).read_bytes())
            assert ex.sha(snapshot) == digest
            result['source_snapshots'].append({'source': path, 'snapshot': str(snapshot.relative_to(ex.ROOT)), 'sha256': digest})
        load_start = time.perf_counter()
        bolt = ex.previous.FrozenBolt()
        result['bolt_load_wall_seconds'] = time.perf_counter()-load_start
        result['bolt_initial_hash'] = bolt.initial_hash
        data = {dataset: ex.load_data(dataset, sizes=(256, 64, None)) for dataset in ('jena', 'robin')}
        for resolved in records:
            case = resolved['case']
            loaded, model_artifact = models(case)
            for method in case['methods']:
                timings = measure_method(data[case['dataset']], case, method, loaded, bolt, limit)
                result['records'].append({'dataset': case['dataset'], 'label': case['label'], 'method': method['method'],
                    'source_run': str(resolved['source_path'].relative_to(ex.ROOT)), 'model_artifact': model_artifact,
                    'science_prediction': method['prediction'], 'length': case['length'], 'k': case['k'], 'latency': timings})
            print(json.dumps({'completed_clean_cost_case': case['label'], 'dataset': case['dataset'],
                              'completed_methods': len(result['records'])}), flush=True)
            del loaded
        if ex.previous.state_hash(bolt.model) != bolt.initial_hash:
            raise AssertionError('CLEAN_COST_BACKBONE_CHANGED')
        result['bolt_final_hash'] = ex.previous.state_hash(bolt.model)
        for path, digest in result['source_hashes'].items():
            assert ex.sha(ex.ROOT/path) == digest
        result['source_hashes_unchanged_at_completion'] = True
        result['status'] = 'COMPLETE'
        limit.check()
        limit.close('COMPLETE')
        result['resources'] = limit.record
        ex.write_new(output/'RESULT.json', result)
        print(json.dumps({'status': 'COMPLETE', 'records': len(result['records']), 'resources': limit.record}), flush=True)
    except BaseException as error:
        result.update(status='FAILED_PRESERVED', error=str(error), traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED', str(error))
        result['resources'] = limit.record
        ex.write_new(output/'FAILURE.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--full-tag', default='full_v1')
    parser.add_argument('--repair-tag', default='repair_v1')
    args = parser.parse_args()
    for value in (args.tag, args.full_tag, args.repair_tag):
        if not value.replace('_', '').isalnum():
            raise ValueError('Alphanumeric/underscore tags required')
    run(args.tag, args.full_tag, args.repair_tag)
