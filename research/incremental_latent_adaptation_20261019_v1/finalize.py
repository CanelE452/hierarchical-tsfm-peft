"""CPU-only completion evidence and training-cost packaging; never opens predictions."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from . import common


C_METHODS = ('SEQ_BLOCKS', 'JOINT_FULL', 'JOINT_PREFIX')
NA = 'NA'
CALL_REASON = 'Legacy FrozenBolt has row_calls/physical_calls but no calls attribute; raw zero is an unavailable-counter sentinel, not measured zero calls.'
MEMORY_SCOPE = 'Shared foundation remains resident during dataset-group training worker; peak is not standalone no-F training memory. Standalone no-F training peak is NA. Fresh-process deployment memory is reported separately in COST.csv.'


def _read(path):
    return common.read(path) if Path(path).is_file() else {}


def _csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def _value(value):
    return NA if value is None else value


def _reference(path):
    path = Path(path)
    for root, prefix in ((common.REPORT, 'report://'), (common.CACHE, 'cache://'), (common.REPO, 'repo://')):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path)


def _status(record):
    return record.get('status', 'PENDING') if record else 'PENDING'


def _counter_interpretation():
    entries = []
    for path in sorted((common.REPORT/'C').glob('*/*.json')):
        value = common.read(path)
        backup = common.REPORT/'attempts/counter_metadata_001'/path.relative_to(common.REPORT)
        changed = []

        def visit(node, location=''):
            if isinstance(node, dict):
                for key, child in list(node.items()):
                    child_path = location+'.'+key if location else key
                    if key in ('foundation_counts_all_work', 'foundation_counts') and isinstance(child, dict):
                        if child.get('calls') == 0:
                            child.update(calls=None, calls_raw_value=0, calls_unavailable_reason=CALL_REASON)
                            changed.append(child_path+'.calls')
                        elif child.get('calls') is None and child.get('calls_raw_value') == 0:
                            changed.append(child_path+'.calls')
                    visit(child, child_path)
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    visit(child, f'{location}[{index}]')

        visit(value)
        if not changed:
            continue
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, backup)
        if common.read(path) != value:
            common.atomic(path, value)
        entries.append(dict(public_file=_reference(path), original_file=_reference(backup),
            original_sha256=common.sha(backup), interpreted_sha256=common.sha(path), fields=changed))
    result = dict(status='COMPLETE', unavailable_field='legacy foundation calls',
        reason=CALL_REASON, raw_value=0, canonical_value=None, files=entries,
        private_artifacts_modified=False,
        valid_counters='Engine training_F_forward_rows/backward_rows/physical_rows remain valid. Fresh-process inference F_calls are independently measured in COST.csv; they do not reconstruct total training calls.')
    common.atomic(common.REPORT/'COUNTER_INTERPRETATION.json', result)
    return result


def _ledger(refresh):
    path = common.REPORT/'RESOURCE_LEDGER.json'
    if not refresh and path.is_file():
        return common.read(path)
    value = common.ledger()
    bounded = [row for row in value.get('sessions', [])
               if 'conservative upper bound' in str(row.get('time_accounting', '')).lower()]
    active = [row for row in value.get('sessions', []) if row.get('active')]
    value['GPU_time_precision'] = ('Includes conservative upper-bound component; total is not entirely exact'
        if bounded else 'Observed owned-process intervals; running sessions are snapshot intervals' if active
        else 'Observed owned-process intervals')
    value['GPU_conservative_upper_bound_component_seconds'] = sum(
        row['end_unix']-row['start_unix'] for row in bounded if row.get('gpu_owned'))
    value['conservative_session_ids'] = [row['session_id'] for row in bounded]
    value['active_process_count'] = len(active)
    value['training_cost_scope'] = ('GPU-owned process wall includes cold initialization, full-Mcal data preparation, '
        'cache generation, evaluation, checkpoint I/O and worker idle; per-stage training timers are narrower and must not be summed with this wall total.')
    common.atomic(path, value)
    return value


def _training_row(group, dataset, method, seed, phase, block, record, evidence, cap=None, selected_prefix=None):
    row = dict(section='training', group=group, dataset=dataset, method=method, seed=seed,
        phase=phase, block=block, status=_status(record), canonical=True,
        cap=_value(cap), selected_epoch=_value(record.get('selected_epoch')),
        selected_prefix=_value(selected_prefix), actual_epochs=_value(record.get('actual_epochs')),
        updates=_value(record.get('updates')), exposures=_value(record.get('exposures')),
        epoch_zero_refit=phase=='refit' and record.get('selected_epoch')==0,
        evidence=_reference(evidence), evidence_exists=Path(evidence).is_file())
    cost = dict(group=group, dataset=dataset, method=method, seed=seed, phase=phase, block=block,
        status=row['status'], cost_scope='new_campaign_engine_stage; cumulative result, not additive to process wall',
        selected_epoch=row['selected_epoch'], cap=row['cap'], updates=row['updates'], exposures=row['exposures'],
        training_seconds=_value(record.get('training_seconds')),
        evaluation_seconds=_value(record.get('evaluation_seconds')),
        checkpoint_seconds=_value(record.get('checkpoint_seconds')),
        training_F_forward_rows=_value(record.get('training_F_forward_rows')),
        training_F_backward_rows=_value(record.get('training_F_backward_rows')),
        training_F_physical_rows=_value(record.get('training_F_physical_rows')),
        training_F_calls=NA, training_F_calls_reason='Engine records rows; total physical calls not instrumented',
        source=_reference(evidence), source_sha256=common.sha(evidence) if Path(evidence).is_file() else NA,
        memory_scope=MEMORY_SCOPE if group=='B' else 'No standalone training peak memory receipt; see process wall and fresh-process deployment costs')
    return row, cost


def _training(admission):
    rows, costs = [], []
    configurations = 0
    for dataset in common.DATASETS:
        for method in common.METHODS:
            for seed in common.SEEDS:
                configurations += 1
                directory = common.CACHE/'runs'/dataset/f'{method}_s{seed}'
                search = _read(directory/'search/RESULT.json')
                cap = max((point.get('epoch', 0) for point in search.get('curves', [])), default=None)
                for phase in ('search', 'refit'):
                    path = directory/phase/'RESULT.json'
                    row, cost = _training_row('B', dataset, method, seed, phase, 1, _read(path), path, cap)
                    receipts = sorted((common.REPORT/'runs'/dataset/f'{method}_s{seed}').glob(phase+'_cap*.json'))
                    if receipts:
                        receipt_path = max(receipts, key=lambda p: common.read(p).get('cap', 0))
                        receipt = common.read(receipt_path)
                        cost.update(peak_allocated_MiB=_value(receipt.get('peak_allocated_MiB')),
                            peak_reserved_MiB=_value(receipt.get('peak_reserved_MiB')),
                            memory_source=_reference(receipt_path), last_worker_method_seconds=_value(receipt.get('worker_method_seconds')),
                            last_worker_scope='Latest phase/cap invocation only; cumulative engine times and process wall have different scopes')
                    rows.append(row); costs.append(cost)
        admitted = admission.get('datasets', {}).get(dataset, {}).get('status', 'PENDING')
        for method in C_METHODS:
            for seed in common.SEEDS:
                if admitted != 'ADMITTED':
                    rows.append(dict(section='C_configuration', group='C', dataset=dataset, method=method,
                        seed=seed, status=admitted, canonical=True, configured_internal_stages=0,
                        evidence='report://C_ADMISSION.json', note='Admission outcome; no attempted training and no training failure'))
                    continue
                configurations += 1
                path = common.CACHE/'C'/dataset/f'{method}_s{seed}'/'RESULT.json'
                result = _read(path)
                count = 2 if method=='SEQ_BLOCKS' else 1
                costs.append(dict(group='C', dataset=dataset, method=method, seed=seed,
                    phase='all_C_work_current_invocation', status=_status(result),
                    whole_invocation_seconds=_value(result.get('seconds')), training_F_calls=NA,
                    training_F_calls_reason=CALL_REASON, source=_reference(path),
                    cost_scope='C-stage invocation includes prefix/checkpoint evaluation, search, fresh refit and stage1 caches; overlapping phase timers are not additive'))
                rows.append(dict(section='C_configuration', group='C', dataset=dataset, method=method,
                    seed=seed, status=_status(result), canonical=True, configured_internal_stages=count*2,
                    internal_stages_per_phase=count, cap=_value(result.get('cap')),
                    selected_epoch=json.dumps(result.get('selected_epoch')), selected_prefix=_value(result.get('selected_prefix')),
                    evidence=_reference(path), note='SEQ has two sequential blocks; joint methods train both blocks in one stage'))
                for phase, key in (('search','searches'), ('refit','refits')):
                    records = result.get(key, [])
                    for index in range(count):
                        record = records[index] if index<len(records) else {}
                        row, cost = _training_row('C', dataset, method, seed, phase, index+1,
                            record, path, result.get('cap'), result.get('selected_prefix'))
                        rows.append(row); costs.append(cost)
                for index, receipt in enumerate(result.get('stage1_caches', [])):
                    costs.append(dict(group='C', dataset=dataset, method=method, seed=seed,
                        phase='stage1_cache_generation', block=1, status='COMPLETE',
                        cost_scope='Original cache generation receipt; not additive to full C worker seconds',
                        cache_seconds=_value(receipt.get('seconds')), cache_reload_seconds=_value(receipt.get('reload_seconds')),
                        cache_reused=receipt.get('reused', False), training_F_calls=NA,
                        training_F_calls_reason=CALL_REASON, cache_F_rows=receipt.get('foundation_counts', {}).get('streams', NA),
                        cache_F_physical_rows=receipt.get('foundation_counts', {}).get('physical_streams', NA),
                        source=_reference(path)+f'#stage1_caches[{index}]'))
    for path in sorted((common.REPORT/'runs').glob('*/READOUT_ONLY_s*/*_cap*.json')):
        receipt = common.read(path)
        costs.append(dict(group='B', dataset=receipt.get('dataset'), method='READOUT_ONLY', seed=receipt.get('seed'),
            phase='readout_cache_generation_'+str(receipt.get('phase')), cap=receipt.get('cap'),
            status=receipt.get('status'), cache_seconds=_value(receipt.get('cache_generation_seconds')),
            peak_allocated_MiB=_value(receipt.get('peak_allocated_MiB')),
            peak_reserved_MiB=_value(receipt.get('peak_reserved_MiB')), memory_scope=MEMORY_SCOPE,
            cost_scope='Full-Mcal feature cache regenerated per worker invocation; distinct phase/cap receipts, not just latest READOUT_CACHE.json',
            source=_reference(path), source_sha256=common.sha(path)))
    return rows, costs, configurations


def _probes_and_parents():
    rows, costs = [], []
    parent_path = common.REPORT/'PARENT_AND_DATA_CONTRACT.json'
    contract = _read(parent_path)
    parents = contract.get('parents', [])
    rows.append(dict(section='A_reused_parents', status='COMPLETE' if contract.get('status')=='PASS' else 'PENDING',
        canonical=True, neural_parent_count=len(parents), parent_prediction_cache_count=contract.get('parent_cache_count', NA),
        new_parent_training_count=0, evidence=_reference(parent_path),
        note='G shared seed per dataset, S two seeds per dataset; original Raw cache is also reused'))
    for dataset in common.DATASETS:
        path = common.CACHE/'probes'/dataset/'RESULT.json'
        selected = _read(path).get('selected', {})
        for method in ('ZERO', 'MEAN', 'OWN', 'ALL'):
            record = selected.get(method, {})
            complete = bool(record) and (path.parent/(method+'.npz')).is_file()
            rows.append(dict(section='A_probe', dataset=dataset, method=method,
                status='COMPLETE' if complete else 'PENDING', canonical=True, evidence=_reference(path),
                new_neural_training_count=0, selected_coefficient_fit_and_refit_count=0 if method=='ZERO' else 2,
                penalty_candidates=4 if method in ('OWN','ALL') else 0))
            costs.append(dict(group='A', dataset=dataset, method=method, phase='coefficient_fit_select_refit',
                status='COMPLETE' if complete else 'PENDING',
                fit_refit_seconds=_value(record.get('fit_refit_seconds')), factorization_count=_value(record.get('factorization_count')),
                cost_scope='Combined probe fit, M_select penalty selection and full-Mcal refit timer; includes candidate fits', source=_reference(path)))
        for seed in common.SEEDS:
            path = common.CACHE/'probes'/dataset/f'alpha_s{seed}.json'
            record = _read(path)
            complete = 'fit_alpha' in record and 'refit_alpha' in record and len(record.get('rows', []))==4
            rows.append(dict(section='A_alpha', dataset=dataset, method='ALPHA', seed=seed,
                status='COMPLETE' if complete else 'PENDING', canonical=True, evidence=_reference(path),
                coefficient_settings=4, fitted_scalar_count=2, new_neural_training_count=0,
                note='0/0.5/1 fixed plus one fitted alpha; fitted scalar freshly refit on full Mcal'))
            costs.append(dict(group='A', dataset=dataset, method='ALPHA', seed=seed, phase='scalar_fit_and_refit',
                status='COMPLETE' if complete else 'PENDING', fit_refit_seconds=NA,
                unavailable_reason='Alpha receipt contains coefficients and row counts but no isolated fit/refit timer; included in process wall', source=_reference(path)))
    summary_path = common.REPO/'reports/memory_budget_20261018_v1/B_TRAINING_SUMMARY.json'
    summary = _read(summary_path).get('rows', [])
    for parent in parents:
        artifact_sha = parent.get('mirror', {}).get('sha256')
        matched = next((row for row in summary if artifact_sha and row.get('bundle_binding', {}).get('sha256')==artifact_sha), None)
        base = dict(group='A60_REUSED_PARENT', dataset=parent.get('dataset'), method=parent.get('role'), seed=parent.get('seed'),
            run_id=parent.get('run_id'), status='REUSED', artifact_sha256=artifact_sha,
            selected_epoch=parent.get('selected_epoch'), cap=parent.get('training_cap'),
            cost_scope='Historical cost of exact artifact; excluded from new campaign resource-clock total')
        binding = parent.get('search_result', {})
        reference = binding.get('reference', '')
        search_path = common.REPO/reference.removeprefix('repo://') if reference.startswith('repo://') else None
        receipt_path = None
        if search_path is not None:
            relative = search_path.relative_to(common.REPO/'.cache')
            receipt_path = common.REPO/'reports'/relative.parent/'BASE_FIT_RECEIPT.json'
        receipt = _read(receipt_path) if receipt_path else {}
        if receipt.get('artifact_sha256') != artifact_sha:
            receipt = {}
        for phase in ('search', 'refit'):
            row = dict(base, phase='historical_'+phase, training_seconds=NA, updates=NA, exposures=NA)
            if matched and isinstance(matched.get(phase+'_seconds'), (float, int)):
                row.update(training_seconds=matched[phase+'_seconds'], source=_reference(summary_path),
                    source_sha256=common.sha(summary_path), timing_scope='Exact artifact-matched historical public phase seconds; includes original trainer phase scope')
            elif phase=='search' and search_path is not None and search_path.is_file() and common.sha(search_path)==binding.get('sha256'):
                metadata = common.read(search_path)
                row.update(recorded_search_session_seconds=_value(metadata.get('summary', {}).get('seconds')),
                    source=_reference(search_path), source_sha256=binding.get('sha256'),
                    unavailable_reason='Bound search receipt may time only latest extension session; full cumulative phase time is unavailable')
            else:
                row.update(source=parent.get('source', {}).get('reference', NA),
                    unavailable_reason='Exact parent artifact is reused; no artifact-bound historical phase timing receipt found. Not estimated from latency.')
            for key in ('updates', 'exposures'):
                row[key] = receipt.get(phase+'_'+key, NA)
            if receipt:
                row['counts_source'] = _reference(receipt_path)
            costs.append(row)
    costs.append(dict(group='INITIALIZATION_AND_DATA', phase='cold_load_and_full_Mcal_preparation',
        status='INCLUDED_IN_PROCESS_WALL', isolated_seconds=NA,
        unavailable_reason='Training workers did not isolate foundation cold load and full-Mcal data/window preparation. Process ledger includes both; deployment cold-load timings have a different scope.',
        source='report://RESOURCE_LEDGER.json'))
    return rows, costs


def _workflow(admission):
    rows = []

    def add(name, complete, evidence, **values):
        rows.append(dict(section='workflow', stage=name, status='COMPLETE' if complete else 'PENDING',
            canonical=True, evidence=evidence, **values))

    add('registration', (common.REPORT/'EXPERIMENT_PLAN.json').is_file(), 'report://EXPERIMENT_PLAN.json')
    for dataset in common.DATASETS:
        expected = 23 + (18 if admission.get('datasets', {}).get(dataset, {}).get('status')=='ADMITTED' else 0)
        files = list((common.CACHE/'final').glob(dataset+'_*/V.npz'))
        bindings = sum(path.with_name('V.binding.json').is_file() for path in files)
        complete = len(files)==expected and bindings==expected and (common.CACHE/f'FINAL_MANIFEST_{dataset}.json').is_file()
        add('final_accuracy_'+dataset, complete, f'cache://FINAL_MANIFEST_{dataset}.json',
            expected_count=expected, actual_prediction_files=len(files), actual_binding_files=bindings,
            note='File inventory only; this packaging helper never opens, hashes or scores V arrays')
    add('final_accuracy_manifest', (common.CACHE/'FINAL_MANIFEST.json').is_file() and (common.REPORT/'RUNS.csv').is_file(),
        'cache://FINAL_MANIFEST.json; report://RUNS.csv')
    request = _read(common.REPORT/'COST_REQUESTS.json')
    expected_ids = {row['cost_id'] for row in request.get('requests', [])}
    costs = _csv(common.REPORT/'COST.csv')
    trials = _csv(common.REPORT/'COST_TRIALS.csv')
    phases = {}
    for row in trials:
        key = row.get('cost_id'), row.get('phase')
        phases[key] = phases.get(key, 0)+1
    cost_ids = {row.get('cost_id') for row in costs}
    expected = request.get('expected_count', len(expected_ids))
    complete = bool(expected_ids) and expected==len(expected_ids)==len(costs) and cost_ids==expected_ids
    complete = complete and all(row.get('parity_status')=='PASS' and str(row.get('fresh_process')).lower()=='true' for row in costs)
    complete = complete and all(phases.get((key,'timed'))==20 and phases.get((key,'warmup'))==10 for key in expected_ids)
    complete = complete and len(trials)==len(expected_ids)*30
    add('fresh_process_costs', complete, 'report://COST_REQUESTS.json; report://COST.csv; report://COST_TRIALS.csv',
        expected_count=expected or NA, actual_cost_rows=len(costs), actual_trial_rows=len(trials),
        note='Every requested B1/B8 process requires full-V parity and ten warmup plus twenty timed trials')
    for name in ('TRAINABLE_STATE_VALIDATION', 'CUDA_RESUME_VALIDATION', 'NUMERICAL_VALIDATION'):
        path = common.REPORT/(name+'.json')
        proof = _read(path)
        add(name.lower(), proof.get('status')=='PASS', _reference(path))
    manifest_path = common.REPORT/'FIGURE_MANIFEST.json'
    manifest = _read(manifest_path)
    figures = manifest.get('figures', [])
    required = ('README_KO.md','RESULT_ANALYSIS_KO.md','CLAIM_STATUS.md','REPRODUCE.md','FIGURE_POINTS.csv')
    complete = manifest.get('status')=='COMPLETE' and bool(figures) and all((common.REPORT/name).is_file() for name in required)
    complete = complete and all((common.REPORT/row['file']).is_file() for row in figures)
    add('report', complete, _reference(manifest_path), expected_artifacts=len(required),
        actual_artifacts=sum((common.REPORT/name).is_file() for name in required), figure_count=len(figures))
    replay = _read(common.REPORT/'REPLAY_VALIDATION.json')
    add('report_replay', replay.get('status')=='PASS', 'report://REPLAY_VALIDATION.json')
    audit = _read(common.REPORT/'VALIDATION_REPORT.json')
    rows.append(dict(section='workflow', stage='final_self_audit', status='COMPLETE' if audit.get('status')=='PASS' else 'PENDING',
        canonical=False, evidence='report://VALIDATION_REPORT.json',
        note='Self-audit bookkeeping is excluded from canonical completion rows to avoid verifier bootstrap recursion; final queue still requires actual PASS'))
    return rows


def build_status(refresh_ledger=True, asset_root=None):
    """Write packaging evidence without training, loading weights, or reading V arrays."""
    common.REPORT.mkdir(parents=True, exist_ok=True)
    counters = _counter_interpretation()
    ledger = _ledger(refresh_ledger)
    admission = _read(common.REPORT/'C_ADMISSION.json')
    rows, costs, configurations = _training(admission)
    probe_rows, probe_costs = _probes_and_parents()
    rows.extend(probe_rows); costs.extend(probe_costs)
    rows.extend(_workflow(admission))
    training = [row for row in rows if row['section']=='training']
    completed = sum(row['status']=='COMPLETE' for row in training)
    zero = sum(row['status']=='COMPLETE' and row['epoch_zero_refit'] for row in training)
    pending = [row for row in rows if row['status'] not in ('COMPLETE', 'NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE')]
    result = dict(overall_status='PENDING' if pending else 'COMPLETE',
        new_neural_configurations=configurations, configured_training_stages=len(training),
        completed_training_stages=completed, epoch_zero_refit_stages=zero,
        counter_interpretation=counters, GPU_time_precision=ledger.get('GPU_time_precision'),
        numeric_V_arrays_read=0, model_or_checkpoint_files_loaded=0)
    rows.append(dict(section='summary', status=result['overall_status'], canonical=False,
        new_neural_configurations=configurations, configured_training_stages=len(training),
        completed_training_stages=completed, epoch_zero_refit_stages=zero,
        note='Configured stages include epoch-zero refits; those have zero optimizer work and remain separate'))
    common.csv_write(common.REPORT/'EXECUTION_STATUS.csv', rows)
    common.csv_write(common.REPORT/'TRAINING_COST.csv', costs)
    start = _read(common.REPORT/'EXECUTION_START.json')
    python = start.get('python', 'python')
    asset_root = Path(asset_root) if asset_root else (Path(python).parents[2] if len(Path(python).parents)>=3 else common.REPO)
    module = 'research.'+common.NAME
    base = f'& "{python}" -X utf8 -m {module}'
    commands = [f'{base}.run {action} --asset-root "{asset_root}"' for action in ('register','audit')]
    commands += [f'{base}.run probes --asset-root "{asset_root}" --dataset {dataset}' for dataset in common.DATASETS]
    commands += [f'{base}.run {action} --asset-root "{asset_root}"' for action in
                 ('train-B','finish-training','final-accuracy','measure')]
    commands += [f'{base}.verify --manifest "{common.CACHE / "FINAL_MANIFEST.json"}" --outdir "{common.REPORT}"',
                 f'{base}.report --report-dir "{common.REPORT}"',
                 f'{base}.report --report-dir "{common.REPORT}" --verify-replay']
    text = ['# 실행·재개 상태', '',
        f'새 신경망 구성 {configurations}개, search/refit 내부 단계 {len(training)}개 중 {completed}개 완료. epoch 0 refit {zero}개는 최적화 없는 별도 단계다.', '',
        '실험·보고·검산 범위의 미완료 없음.' if not pending else '다음 실제 증거가 아직 필요하다:', '']
    for row in pending:
        label = '/'.join(str(row.get(key, '')) for key in ('section','dataset','method','seed','phase','block','stage') if row.get(key) is not None)
        text.append(f'- {label}: {row["status"]} ({row.get("evidence", NA)})')
    text += ['', '명령은 저장된 봉인과 완료 파일을 확인하며 재개한다. 예산 종료로 남은 단계는 새 결과로 간주하지 않는다.', '', '```powershell', *commands, '```', '',
        '정리 순서: 비용까지 완료 → build_status()로 원장·공개 카운터 해석 고정 → report 및 report --verify-replay → verify → build_status(refresh_ledger=False)로 상태만 갱신. 최종 검산 PASS 후 다시 verify하여 포장 상태를 확인한다.', '',
        '훈련 단계 시간과 프로세스 전체 시간은 범위가 겹쳐 합산하지 않는다. 과거 A60 부모 비용은 재사용 비용이며 신규 자원 한도와 구분한다. F가 없는 방법도 학습 그룹 프로세스에는 공통 F가 상주했으므로 standalone 훈련 메모리는 N/A다.', '',
        str(ledger.get('GPU_time_precision', 'GPU clock precision unavailable')), '']
    (common.REPORT/'RESUME_QUEUE.md').write_text('\n'.join(text), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-refresh-ledger', action='store_true')
    parser.add_argument('--asset-root')
    args = parser.parse_args()
    result = build_status(not args.no_refresh_ledger, args.asset_root)
    print(json.dumps({key:value for key,value in result.items() if key!='counter_interpretation'}, ensure_ascii=False))


if __name__=='__main__':
    main()
