"""Historical pre-T task and source identities; no PCA or foundation execution."""
import importlib.metadata
from pathlib import Path
import platform
import re
import sys

from research.adapter_diagnostics_20261017_v2 import existing_jobs as oldexisting
from research.b2_dataset_extension_20261018_v1 import train as core
from research.b2_dataset_search_20261019_v1 import train as search
from research.b2_revin_20261018_v1.contracts import read, sha


NAME = 'k_tau_rule_20261023_v1'
SEARCH_NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
BROAD = 'research/informs_broad_validation_20261015_v1'


def _required_hash(record, key):
    value = record.get(key)
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Missing or invalid historical ' + key)
    return value


def _validate_prefix_config(config, dataset, identity):
    _required_hash(identity, 'data_sha256')
    expected_prefix = _required_hash(identity, 'prefix_sha256')
    if (config.get('dataset') != dataset or config.get('prefix_only_approved') is not True
            or (config.get('L'), config.get('H')) != (512, 48)
            or config.get('prefix_sha256') != expected_prefix):
        raise ValueError('Approved historical prefix identity changed')
    ranges = config.get('split_ranges', {})
    previous = 0
    for key in core.SPLITS:
        bounds = ranges.get(key)
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) is not int for v in bounds)
                or bounds[0] != previous or bounds[1] <= bounds[0]):
            raise ValueError('Historical pre-T split boundaries changed')
        previous = bounds[1]
    if (type(config.get('prefix_rows')) is not int
            or config['prefix_rows'] != ranges['V_select'][1]
            or config['prefix_rows'] != ranges['T_final'][0]):
        raise ValueError('Numeric prefix must stop exactly before T')


def _validate_task(task, dataset, identity):
    expected = _required_hash(identity, 'data_sha256')
    if (task.get('dataset') != dataset or task.get('data_sha256') != expected
            or (task.get('L'), task.get('H')) != (512, 48)):
        raise ValueError('Loaded task differs from historical data identity')
    ranges = task['split_ranges']
    rows = task['physical'].shape[0]
    if (rows != ranges['V_select'][1] or rows != ranges['T_final'][0]
            or task['observed'].shape != task['physical'].shape
            or task['physical'].shape[1] != len(task['columns'])):
        raise ValueError('Loaded task exceeds or differs from exact pre-T prefix')
    if 'split_ranges' in identity:
        for key, expected_bounds in identity['split_ranges'].items():
            if tuple(ranges[key]) != tuple(expected_bounds):
                raise ValueError('Loaded historical split changed: ' + key)
    if 'array_sha256' in identity and task.get('array_sha256') != identity['array_sha256']:
        raise ValueError('Loaded historical array identities changed')


def _historical_records(dataset):
    if dataset == 'jena':
        identity_path = REPO / 'reports/b2_revin_20261018_v1/DATA_CONTRACT.json'
        identity = read(identity_path)['tasks']['jena']
        source_path = REPO / ('reports/memory_budget_20261018_v1/runs/'
            'B2_jena_K3_OFF_s92601/cap3/SOURCE_CONTRACT.json')
    else:
        directory = REPO / 'reports' / SEARCH_NAME / 'datasets' / dataset
        identity_path = directory / 'TRAIN_IDENTITY.json'
        identity = read(identity_path)
        k = 4 if dataset == 'appliances' else 1
        source_path = directory / 'runs' / f'B2_{dataset}_K{k}_OFF_s92601' / 'cap3/SOURCE_CONTRACT.json'
    _required_hash(identity, 'data_sha256')
    return identity_path, identity, source_path, read(source_path)


def _source_hashes(source_record):
    historical = source_record.get('sources', source_record.get('files', {}))
    core_names = {name: value for name, value in historical.items() if '/' not in name}
    if len(core_names) != 6:
        raise ValueError('Historical source contract must name exactly six core files')
    expected = {(BROAD + '/' + name if '/' not in name else name): value
                for name, value in historical.items()}
    current = {}
    for relative in set(expected) | set(search.DEPENDENCIES):
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO.resolve()):
            raise ValueError('Scientific source escapes repository')
        actual = sha(path)
        if relative in expected and actual != expected[relative]:
            raise ValueError('Historical producer source changed: ' + relative)
        current[relative] = actual
    for path in sorted((REPO / 'research' / NAME).glob('*.py')):
        if not path.name.startswith('test_'):
            current[path.relative_to(REPO).as_posix()] = sha(path)
    return current, core_names


def build_context(dataset, deadline, asset_root=Path('E:/CODING/proj/hierarchical-tsfm-peft'), device='cuda'):
    if dataset not in DATASETS:
        raise ValueError('Unregistered dataset: ' + str(dataset))
    if device not in ('cuda', 'cpu'):
        raise ValueError('Registered device must be cuda or cpu')
    asset = Path(asset_root).resolve()
    identity_path, identity, source_path, source_record = _historical_records(dataset)
    sources, core_names = _source_hashes(source_record)
    report_root = REPO / 'reports' / NAME
    protocol = report_root / 'PREREGISTRATION.json'
    ctx = dict(repo=REPO, asset=asset, cache=REPO / '.cache' / NAME / 'datasets' / dataset,
        report=report_root / 'datasets' / dataset, deadline=float(deadline), device=device,
        scientific_source_hashes=sources,
        protocol_sha256=sha(protocol) if protocol.is_file() else None,
        training_allowed=protocol.is_file(), historical_identity=identity,
        historical_identity_path=identity_path, historical_source_record=source_record,
        historical_source_path=source_path, historical_core6=core_names)
    if dataset != 'jena':
        root = REPO / '.cache' / SEARCH_NAME / 'data' / dataset
        config_path, prefix_path = root / 'DATA_DIAGNOSTIC.json', root / 'PREFIX_PHYSICAL.npz'
        _required_hash(identity, 'input_config_sha256')
        if sha(config_path) != identity['input_config_sha256']:
            raise ValueError('Historical input config bytes changed')
        config = read(config_path)
        _validate_prefix_config(config, dataset, identity)
        if sha(prefix_path) != identity['prefix_sha256']:
            raise ValueError('Historical pre-T prefix bytes changed')
        config, task = core.load_task(config_path, prefix_path)
        ctx.update(config=config, config_path=config_path, prefix_path=prefix_path,
            task_access=dict(numeric_scope='historical prefix only; ends before T', T_numeric_values_parsed=0,
                             prefix_sha256=identity['prefix_sha256']))
        modules = oldexisting._modules(ctx)
    else:
        modules = oldexisting._modules(ctx)
        task = oldexisting._jena(ctx, modules)
        ctx['task_access'] = dict(T_numeric_values_parsed=0,
            numeric_scope='safe_jena_prefix bounded decoder: only rows [0,16772)',
            opaque_IO_scope='Existing loader hashes the original complete compressed archive and may discard opaque tails; no separate T file is opened.',
            identity_provenance=task.get('data_identity_provenance'),
            materialized_prefix_sha256=task.get('materialized_prefix_sha256'))
    _validate_task(task, dataset, identity)
    model_dir = (asset / modules['legacy'].MODEL_REL).resolve()
    if not model_dir.is_relative_to(asset) or not model_dir.is_dir():
        raise ValueError('Historical local foundation asset directory is missing or outside asset root')
    ctx.update(task=task, modules=modules, model_dir=model_dir)
    return ctx


def fingerprint_context(ctx):
    """Metadata only; actual foundation state SHA is audited by the caller."""
    task, modules = ctx['task'], ctx['modules']
    record, expected = ctx['historical_source_record'], ctx['historical_identity']
    core6 = {name: dict(expected=value, actual=sha(Path(ctx['repo']) / BROAD / name))
             for name, value in ctx['historical_core6'].items()}
    for item in core6.values():
        item['matches'] = item['actual'] == item['expected']
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'chronos-forecasting', 'transformers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment_path = Path(ctx['repo']) / 'reports' / (
        'memory_budget_20261018_v1' if task['dataset'] == 'jena' else SEARCH_NAME) / 'ENVIRONMENT.json'
    environment_reference = read(environment_path)
    return dict(dataset=task['dataset'], data_sha256=task['data_sha256'],
        historical_data_sha256=expected['data_sha256'], data_sha_matches=task['data_sha256'] == expected['data_sha256'],
        array_sha256=task.get('array_sha256'), materialized_rows=task['physical'].shape[0],
        materialized_prefix_sha256=task.get('materialized_prefix_sha256'), split_ranges=task['split_ranges'],
        task_access=ctx['task_access'], identity_record=str(ctx['historical_identity_path']),
        core6=core6, scientific_source_hashes=ctx['scientific_source_hashes'],
        historical_model_state_sha256=record.get('model_sha'),
        legacy_expected_model_state_sha256=modules['legacy'].F_SHA,
        actual_model_state_sha256=None, actual_model_state_audit='CALLER_REQUIRED_NO_MODEL_LOADED',
        model_dir=str(ctx['model_dir']),
        model_asset_files=[dict(path=p.relative_to(ctx['model_dir']).as_posix(), bytes=p.stat().st_size)
                           for p in sorted(Path(ctx['model_dir']).rglob('*')) if p.is_file()],
        environment=dict(python=platform.python_version(), executable=sys.executable,
            platform=platform.platform(), packages=packages, cuda_build=modules['torch'].version.cuda,
            cuda_initialized=modules['torch'].cuda.is_initialized(), gpu_identity='CALLER_AUDIT_REQUIRED'),
        historical_environment=environment_reference, protocol_sha256=ctx['protocol_sha256'],
        training_allowed=ctx['training_allowed'])
