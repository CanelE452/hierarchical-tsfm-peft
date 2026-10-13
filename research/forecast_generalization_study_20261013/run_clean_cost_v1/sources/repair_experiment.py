"""Re-run only failed new-campaign paths, preserving the immutable first run."""
from pathlib import Path
import argparse
import json
import os
import time
import traceback
import experiment as ex
import diagnose_constant_gradient as gradient_fix


class RetryLimit(ex.Limit):
    def __init__(self, tag):
        self.external_receipts = []
        for path in (ex.ROOT/'research/forecast_metric_alignment_20261013').glob('run_*/RESOURCE_USAGE.json'):
            value = ex.read(path)
            if value['status'] not in ('COMPLETE', 'FAILED_PRESERVED'):
                raise RuntimeError('EXTERNAL_RESOURCE_RECEIPT_STILL_ACTIVE')
            self.external_receipts.append({'path': str(path.relative_to(ex.ROOT)), 'sha256': ex.sha(path),
                                           'wall_seconds': value['wall_seconds'], 'status': value['status']})
        self.external_storage = sum(p.stat().st_size for root in (ex.ROOT/'research/forecast_metric_alignment_20261013',
            ex.ROOT/'.cache/forecast_metric_alignment_20261013') for p in root.rglob('*') if p.is_file())
        super().__init__(tag)

    def check(self):
        super().check()
        self.record['external_completed_gpu_process_receipts'] = self.external_receipts
        self.record['external_completed_gpu_process_wall_seconds'] = sum(v['wall_seconds'] for v in self.external_receipts)
        self.record['all_new_campaign_gpu_process_wall_seconds'] = self.record['cumulative_gpu_process_wall_seconds']+self.record['external_completed_gpu_process_wall_seconds']
        self.record['external_new_storage_bytes'] = self.external_storage
        self.record['all_new_storage_bytes'] = self.record['new_cache_bytes']+self.record['new_research_bytes']+self.external_storage
        self.flush()
        if self.record['all_new_campaign_gpu_process_wall_seconds'] >= 2700:
            raise RuntimeError('ALL_NEW_CAMPAIGN_GPU_PROCESS_WALL_LIMIT_2700S')
        if self.record['all_new_storage_bytes'] > 4*1024**3:
            raise RuntimeError('ALL_NEW_CAMPAIGN_STORAGE_LIMIT_4GIB')

    def flush(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2)+'\n', encoding='utf-8')
        for attempt in range(20):
            try:
                temporary.replace(self.path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)


def identity_gpu_fixture(data, history, seed, bolt):
    ids = ex.np.random.default_rng(seed).integers(0, 256, size=(600, 4))[0]
    origins = data['origins']['fit'][ids]
    target = ex.tensor(ex.np.stack([data['x'][o:o+ex.H] for o in origins]))
    observed = ex.boolean(ex.np.stack([data['mask'][o:o+ex.H] for o in origins]))
    mean, basis, _ = ex.initialization(data, len(data['columns']), 'IDENTITY')
    codec = ex.Codec(mean, basis, seed).to(ex.DEVICE)
    old_normalizer = bolt.model.instance_norm
    x = ex.tensor(history['fit'][ids]).requires_grad_(True)
    original_prediction = bolt(x)
    original_loss = ex.previous.macro_loss(original_prediction, target, observed)
    original_gradient = ex.torch.autograd.grad(original_loss, x)[0]
    gradient_fix.replace_normalizer(bolt)
    x = ex.tensor(history['fit'][ids]).requires_grad_(True)
    prediction = codec.decode(bolt(codec.encode(x)))
    loss = ex.previous.macro_loss(prediction, target, observed)
    loss.backward()
    record = {'dataset': data['dataset'], 'seed': seed, 'device': 'CUDA float32', 'first_batch_indices': ids.tolist(),
        'original_gradient_finite': bool(ex.torch.isfinite(original_gradient).all()),
        'repaired_input_gradient_finite': bool(ex.torch.isfinite(x.grad).all()),
        'repaired_encoder_decoder_gradient_finite': all(p.grad is not None and bool(ex.torch.isfinite(p.grad).all()) for p in codec.parameters()),
        'forward_max_abs_vs_original': float((prediction.detach()-original_prediction.detach()).abs().max()),
        'original_loss': float(original_loss.detach()), 'repaired_loss': float(loss.detach())}
    assert not record['original_gradient_finite'] and record['repaired_input_gradient_finite']
    assert record['repaired_encoder_decoder_gradient_finite'] and record['forward_max_abs_vs_original']==0
    initial = ex.previous.clone_state(codec)
    ex.torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.)
    ex.torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=0.).step()
    record['actual_encoder_update_max_abs'] = max(float((p.detach().cpu()-initial[n]).abs().max()) for n, p in codec.named_parameters() if not n.startswith('decode.'))
    record['actual_decoder_update_max_abs'] = max(float((p.detach().cpu()-initial[n]).abs().max()) for n, p in codec.named_parameters() if n.startswith('decode.'))
    codec.load_state_dict(initial)
    bolt.model.instance_norm = old_normalizer
    record['temporary_update_restored'] = True
    record['backbone_hash_unchanged'] = ex.previous.state_hash(bolt.model)==bolt.initial_hash
    assert record['backbone_hash_unchanged'] and min(record['actual_encoder_update_max_abs'],record['actual_decoder_update_max_abs'])>0
    return record


def configure():
    ex.torch.set_num_threads(2)
    ex.torch.set_num_interop_threads(1)
    ex.torch.backends.cuda.matmul.allow_tf32 = False
    ex.torch.backends.cudnn.allow_tf32 = False
    ex.torch.backends.cudnn.benchmark = False
    ex.torch.backends.cudnn.deterministic = True


def select_config(cases):
    values = {}
    for config in ex.CONFIGS:
        selected = [next(cp for cp in c['training']['checkpoints'] if cp['update']==c['selected_update'])['forecast']['tune']['normalized']['mse']
                    for c in cases if c['stage']=='G1' and c['objective']=='PRED' and c['config']==config]
        if len(selected) != 2:
            raise ValueError('G1_CONFIGURATION_REMAINS_INCOMPLETE: '+config)
        values[config] = float(ex.np.mean(selected))
    minimum = min(values.values())
    candidate = next(c for c in ex.CONFIGS if values[c] <= minimum+1e-12)
    improvement = (values['BASE']-values[candidate])/values['BASE']
    chosen = candidate if improvement >= .01 else 'BASE'
    return {'chosen_config': chosen, 'seed_mean_tune_normalized_forecast_mse': values, 'candidate_minimum': candidate,
            'candidate_relative_improvement_vs_BASE': improvement, 'criterion': 'Same predeclared TUNE-only1%gate and1e-12configtie; VAL not used'}


def run(tag, source_tag):
    configure()
    protected = ex.access_guard()
    source_path = ex.HERE/('run_'+source_tag)/'RESULT.json'
    source = ex.read(source_path)
    if source['status'] not in ('PARTIAL_FAILED_PATHS', 'COMPLETE'):
        raise ValueError('Source run must have finished before repair')
    output = ex.HERE/('run_'+tag)
    output.mkdir(exist_ok=False)
    folder = ex.CACHE/('run_'+tag)
    folder.mkdir(parents=True, exist_ok=False)
    limit = RetryLimit(tag)
    source_files = (Path(__file__), ex.HERE/'diagnose_constant_gradient.py', ex.HERE/'experiment.py', ex.HERE/'test_experiment.py', ex.HERE/'PLAN_KO.md')
    summary = {'status': 'RUNNING', 'tag': tag, 'source_run': {'path': str(source_path.relative_to(ex.ROOT)), 'sha256': ex.sha(source_path)}, 'gradient_cuda_fixtures': [],
        'source_hashes': {str(p.relative_to(ex.ROOT)): ex.sha(p) for p in source_files}, 'cases': [], 'configuration_checks': {}, 'attempted_specs': [],
        'protected_test_guarded': protected,
        'changed_execution_behavior': 'Resource ledger replace retries20 times0.05s. Only identity paths with documented constant-context gradient failure use zero derivative convention at sqrt(variance=0); all forward values, positive-variance gradients, weights and selection remain identical.',
        'gradient_diagnostic': {'path': str((ex.HERE/'run_gradient_diagnosis_v1/RESULT.json').relative_to(ex.ROOT)),
                                'sha256': ex.sha(ex.HERE/'run_gradient_diagnosis_v1/RESULT.json')},
        'source_failed_paths': source.get('failed_paths', []), 'scope': 'New first-run failed paths only; not a resumed earlier research campaign'}
    try:
        snapshots = output/'sources'
        snapshots.mkdir()
        for path, digest in summary['source_hashes'].items():
            with (snapshots/Path(path).name).open('xb') as f:
                f.write((ex.ROOT/path).read_bytes())
            assert ex.sha(snapshots/Path(path).name) == digest
        load_start = time.perf_counter()
        bolt = ex.previous.FrozenBolt()
        original_normalizer = bolt.model.instance_norm
        summary['bolt_load_wall_seconds'] = time.perf_counter()-load_start
        summary['bolt_initial_hash'] = bolt.initial_hash
        datasets = {}
        for failure in source.get('failed_paths', []):
            dataset = failure['dataset']
            if dataset not in datasets:
                data = ex.load_data(dataset, sizes=(256, 64, None))
                history, _, _, _ = ex.arrays(data, 512)
                datasets[dataset] = (data, history)
                (output/dataset).mkdir()
                (folder/dataset).mkdir()
            data, history = datasets[dataset]
            use_gradient_fix = failure['spec']['init']=='IDENTITY' and failure['error']=='NONFINITE_TRAINING_GRADIENT'
            if use_gradient_fix:
                summary['gradient_cuda_fixtures'].append(identity_gpu_fixture(data, history, failure['spec']['seed'], bolt))
                gradient_fix.replace_normalizer(bolt)
            else:
                bolt.model.instance_norm = original_normalizer
            summary['attempted_specs'].append({'dataset': dataset, **failure['spec']})
            repaired = ex.safe_fit_case(data, failure['spec'], bolt, output/dataset, folder/dataset, limit, history, summary)
            for case in repaired:
                case['repair_of'] = {'failed_source_run': summary['source_run'], 'failed_spec': failure['spec'],
                    'note': 'Fresh initialization/full600updates; no continuation or overwritten source artifact'}
                case['zero_variance_backward_convention'] = use_gradient_fix
                case['inference_normalizer_forward_unchanged'] = True
            summary['cases'].extend(repaired)
            bolt.model.instance_norm = original_normalizer
        for dataset in ('jena', 'robin'):
            original = [c for c in source['cases'] if c['dataset']==dataset]
            repaired = [c for c in summary['cases'] if c['dataset']==dataset]
            selection = select_config(original+repaired)
            previous_selection = source['stage2_configuration_selection'][dataset]
            selection['source_selection'] = previous_selection
            selection['configuration_unchanged'] = selection['chosen_config'] == previous_selection['chosen_config']
            summary['configuration_checks'][dataset] = selection
            if not selection['configuration_unchanged']:
                data, history = datasets.get(dataset, (None, None))
                if data is None:
                    data = ex.load_data(dataset, sizes=(256, 64, None))
                    history, _, _, _ = ex.arrays(data, 512)
                    datasets[dataset] = (data, history)
                    (output/dataset).mkdir()
                    (folder/dataset).mkdir()
                quarter = data['x'].shape[1]//4
                for seed in ex.SEEDS:
                    for k, length, rf, initial in ((data['x'].shape[1]//2, 512, 127, 'PCA'),
                        (data['x'].shape[1], 512, 127, 'PCA'), (data['x'].shape[1], 512, 127, 'IDENTITY'),
                        (quarter, 128, 127, 'PCA'), (quarter, 512, 31, 'PCA')):
                        spec = {'stage': 'G2', 'objective': 'PRED', 'config': selection['chosen_config'],
                                'k': k, 'length': length, 'rf': rf, 'init': initial, 'seed': seed}
                        summary['attempted_specs'].append({'dataset': dataset, **spec})
                        summary['cases'].extend(ex.safe_fit_case(data, spec, bolt, output/dataset, folder/dataset, limit, history, summary))
        if ex.previous.state_hash(bolt.model) != bolt.initial_hash:
            raise AssertionError('FROZEN_BOLT_CHANGED')
        summary['bolt_final_hash'] = ex.previous.state_hash(bolt.model)
        for path, digest in summary['source_hashes'].items():
            assert ex.sha(ex.ROOT/path) == digest
        summary['source_hashes_unchanged_at_completion'] = True
        summary['status'] = 'PARTIAL_FAILED_PATHS' if summary.get('failed_paths') else 'COMPLETE'
        limit.check()
        limit.close(summary['status'])
        summary['resources'] = limit.record
        summary['method_count'] = sum(len(c['methods']) for c in summary['cases'])
        summary['new_training_runs'] = len({(c['dataset'], c['seed'], c['objective'], c['config'], c['k'], c['length'],
            c['receptive_field'], c['initialization']['kind']) for c in summary['cases']})
        ex.write_new(output/'RESULT.json', summary)
        print(json.dumps({'status': summary['status'], 'method_count': summary['method_count'], 'resources': limit.record}), flush=True)
    except BaseException as error:
        summary.update(status='FAILED_PRESERVED', error=str(error), traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED', str(error))
        summary['resources'] = limit.record
        ex.write_new(output/'FAILURE.json', summary)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--source-tag', default='full_v1')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum() or not args.source_tag.replace('_', '').isalnum():
        raise ValueError('Alphanumeric/underscore tags required')
    run(args.tag, args.source_tag)
