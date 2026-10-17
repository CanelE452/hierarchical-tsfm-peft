"""Read-only audit of existing shared, independent, posthoc and nested K evidence."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import atomic_json, binding, read, sha


UNKNOWN = 'UNKNOWN'
BROAD = 'research/informs_broad_validation_20261015_v1'


def lag_major_prefix(decoder, maximum_k, k):
    values = np.asarray(decoder)
    if not 1 <= k <= maximum_k or values.ndim not in (2,3):
        raise ValueError('Invalid prefix K or decoder rank')
    if values.ndim == 2:
        if values.shape[0] != 7*maximum_k:
            raise ValueError('Flattened decoder must contain seven complete lag blocks')
        values = values.reshape(7,maximum_k,values.shape[-1])
    if values.shape[:2] != (7,maximum_k):
        raise ValueError('Decoder must have lag-major [7,Kmax,C] axes')
    return values[:, :k, :].reshape(7*k, values.shape[-1])


def _variant(spec, parent):
    name = parent.upper()
    if 'MAXK' in name or 'MAX_ONLY' in name:
        return 'MAX_ONLY_TRAINED_PARENT'
    if len(spec.get('k_grid', [])) > 1:
        return 'SHARED_MULTI_K_PARENT'
    return 'INDEPENDENT_K_PARENT'


def stage_rows(source, receipt, *, stage, parent, completion=None):
    spec = source.get('spec', {})
    grid = spec.get('k_grid') or ([spec['latent_k']] if 'latent_k' in spec else [])
    if not grid:
        return []
    total = receipt.get('exposures', UNKNOWN)
    exposure = receipt.get('k_exposures', {})
    if exposure and (any(int(k) not in grid for k in exposure) or any(int(v)<0 for v in exposure.values())):
        raise ValueError('Logged K exposure keys/values contradict source grid')
    if exposure and total != UNKNOWN and sum(exposure.values()) != total:
        raise ValueError('Logged per-K exposure sum differs from total origins')
    result = []
    for k in grid:
        n = exposure.get(str(k), exposure.get(k, UNKNOWN))
        fraction = n/total if n != UNKNOWN and total != UNKNOWN and total else UNKNOWN
        result.append(dict(parent=parent, dataset=spec.get('dataset', UNKNOWN), family=spec.get('family',UNKNOWN),
            seed=spec.get('seed',UNKNOWN), evidence_family=_variant(spec,parent), stage=stage,
            selected_epoch=receipt.get('selected_epoch', UNKNOWN), total_updates=receipt.get('updates',UNKNOWN),
            total_origin_exposures=total, K=int(k), K_origin_exposures=n,
            K_origin_exposures_evidence='LOGGED' if n != UNKNOWN else UNKNOWN,
            actual_F_rows=receipt.get('F_training_logical_streams', UNKNOWN),
            actual_F_rows_scope='whole stage training logical rows, all K' if 'F_training_logical_streams' in receipt else UNKNOWN,
            per_K_F_rows=int(k)*n if n != UNKNOWN else UNKNOWN,
            per_K_F_rows_evidence='DERIVED_K_TIMES_LOGGED_ORIGIN_EXPOSURES' if n != UNKNOWN else UNKNOWN,
            direct_K_exposure_fraction=fraction, direct_K_exposure_fraction_evidence='DERIVED_FROM_LOGGED_EXPOSURES' if fraction != UNKNOWN else UNKNOWN,
            completion=completion or receipt.get('status',UNKNOWN), checkpoint_bytes=UNKNOWN, switch_time=UNKNOWN,
            source_sha256=source.get('source_sha',UNKNOWN), config_sha256=source.get('config_sha',UNKNOWN),
            data_sha256=source.get('data_sha',UNKNOWN), foundation_sha256=source.get('model_sha',UNKNOWN),
            evidence_scope='historical logged evidence; no new training or invented exposure counts'))
    return result


def audit_sources(repo, ctx=None):
    root = Path(repo).resolve()
    broad = root/BROAD
    rows, sources, parents, private = [], {}, {}, []
    def register(path):
        relative = path.resolve().relative_to(root).as_posix()
        sources[relative] = sha(path)
        return relative
    for path in sorted(broad.rglob('SOURCE_CONTRACT.json')):
        source = read(path)
        spec = source.get('spec', {})
        parent = spec.get('run_id', path.parent.name)
        grid = spec.get('k_grid', [])
        ordered = any(marker in parent.upper() for marker in ('ORDERED_MULTI_K','MAXK_PCA','INDEPENDENT'))
        standard_independent = spec.get('dataset') in ('beijing','electricity') and spec.get('family') in ('B0','B2') and 'E1_' in parent
        if len(grid) < 2 and not ordered and not standard_independent:
            continue
        reference = register(path)
        parents[parent] = dict(source=source, path=path)
        base_receipt = path.parent/'BASE_FIT_RECEIPT.json'
        base = read(base_receipt) if base_receipt.is_file() else {}
        if base:
            register(base_receipt)
        for stage in ('search','refit'):
            stage_dir = path.parent/stage
            receipt_path = stage_dir/'RECEIPT.json'
            complete = receipt_path.is_file()
            if not complete:
                events = sorted(stage_dir.glob('event_*.json'))
                receipt_path = events[-1] if events else None
            if receipt_path is None:
                receipt = {}
                evidence = UNKNOWN
            else:
                receipt = read(receipt_path)
                evidence = register(receipt_path)
            if receipt.get('selected_epoch') is None and base.get('selected_epoch') is not None:
                receipt = dict(receipt, selected_epoch=base['selected_epoch'])
            stage_values = stage_rows(source, receipt, stage=stage, parent=parent,
                completion=receipt.get('status',UNKNOWN) if complete else ('PARTIAL_EVENT_ONLY' if receipt_path else 'NO_STAGE_EVIDENCE'))
            checkpoint_paths = []
            if ctx is not None and 'asset' in ctx:
                relative = path.parent.relative_to(broad)
                private_run = Path(ctx['asset'])/'.cache/informs_broad_validation_20261015_v1'/relative
                checkpoint_paths = sorted((private_run/stage).glob('checkpoint_*.pt'))
                if stage == 'refit' and (private_run/'selected_bundle.pt').is_file():
                    checkpoint_paths.append(private_run/'selected_bundle.pt')
            for value in stage_values:
                value.update(source_reference=reference, stage_evidence_reference=evidence,
                             selected_parent_completion=base.get('status',UNKNOWN))
                if checkpoint_paths:
                    selected = checkpoint_paths[-1]
                    bound = binding(selected,ctx)
                    private.append(bound)
                    value.update(checkpoint_bytes=selected.stat().st_size,checkpoint_sha256=bound['sha256'],
                                 checkpoint_reference=bound['reference'],checkpoint_bytes_evidence='ACTUAL_HASHED_FILE_BYTES')
                rows.append(value)
    # Forecast aliases establish artifact sharing and exact active-K inference,
    # but never establish unlogged training exposure counts.
    for path in sorted(broad.rglob('PREDICTION_MANIFEST.json')):
        if not any(marker in path.as_posix().lower() for marker in ('ordered','multik','nested','posthoc','post_max')):
            continue
        value = read(path)
        metadata = value.get('metadata', {})
        parent = metadata.get('parent_run_id', metadata.get('parent_run', UNKNOWN))
        k = metadata.get('inference_latent_k',metadata.get('inference_K',metadata.get('latent_k',metadata.get('K',UNKNOWN))))
        variant = metadata.get('prefix_variant',metadata.get('variant',UNKNOWN))
        reference = register(path)
        rows.append(dict(parent=parent,stage='V_SELECT_ALIAS',dataset=value.get('dataset',UNKNOWN),
            evidence_family=str(variant),run_id=value.get('run_id',UNKNOWN), K=k,
            selected_epoch=UNKNOWN,total_updates=UNKNOWN,K_origin_exposures=UNKNOWN,actual_F_rows=metadata.get('actual_F_streams',UNKNOWN),
            direct_K_exposure_fraction=UNKNOWN,completion=value.get('status','MANIFEST_PRESENT_NO_STATUS'),checkpoint_bytes=UNKNOWN,switch_time=UNKNOWN,
            source_reference=reference, artifact_sha256=metadata.get('parent_artifact_sha256',metadata.get('artifact_sha256',UNKNOWN)),
            evidence_scope='prediction manifest only; training exposures not inferred'))
    transition, scopes = [], {}
    for path in sorted(broad.rglob('BATCH_LAUNCHER_RECEIPT.json')):
        if 'multik_transition' not in path.as_posix():
            continue
        value = read(path)
        reference = register(path)
        for item in value.get('scope_ledger', []):
            scopes[item.get('scope_id')] = item
            transition.append(dict(dataset=item.get('dataset',UNKNOWN),batch_size=item.get('batch',UNKNOWN),
                strategy=item.get('strategy',UNKNOWN),completion=item.get('status',UNKNOWN),
                exact_minimum_protocol_complete=item.get('exact_minimum_protocol_complete',False),
                actual_numeric_trials=item.get('actual_numeric_trials',UNKNOWN),
                requested_numeric_trials=item.get('requested_numeric_trials',UNKNOWN),
                switch_time=UNKNOWN,source_reference=reference,
                note='Launcher completion is measured evidence; missing numeric switching latency remains UNKNOWN'))
    # Published lossless transition bundles retain the actual timed records even
    # when the original high-file-count package was removed before publication.
    for path in sorted(broad.glob('diagnostics/publication_multik_lossless_*/bundle_*/files/*/*/RESULT.json.gz')):
        raw = gzip.decompress(path.read_bytes())
        value = json.loads(raw)
        reference = register(path)
        scope_id = path.parents[1].name
        expected = scopes.get(scope_id,{}).get('result_files_sha256',{}).get(path.parent.name+'/RESULT.json')
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Published transition original result differs from launcher hash')
        actual = [item for item in value.get('rows',[]) if item.get('phase') == 'timed']
        pairs = sorted({(item['from_K'],item['to_K']) for item in actual})
        for origin_k,target_k in pairs:
            group = [item for item in actual if item['from_K']==origin_k and item['to_K']==target_k]
            switch = [item['switch_total_seconds'] for item in group if item.get('switch_total_seconds') is not None]
            whole = [item['transition_plus_physical_seconds'] for item in group if item.get('transition_plus_physical_seconds') is not None]
            streams = sorted({item['F_streams'] for item in group if 'F_streams' in item})
            batch = value.get('batch',UNKNOWN)
            expected_rows = batch*target_k if isinstance(batch,int) else UNKNOWN
            stream_parity = bool(streams and expected_rows != UNKNOWN and streams == [expected_rows])
            storage = value.get('storage',{})
            accuracy = value.get('accuracy_bindings',{}).get(str(target_k),{})
            route_spec = accuracy.get('accuracy_route_spec',accuracy.get('cost_route_spec',{}))
            parent_artifact = route_spec.get('artifact_cache_relative_path',UNKNOWN)
            parent = Path(parent_artifact).parent.name if parent_artifact != UNKNOWN else UNKNOWN
            rows.append(dict(parent=parent, stage='K_SWITCH',dataset=value.get('dataset',UNKNOWN),
                evidence_family=value.get('strategy',UNKNOWN), from_K=origin_k,K=target_k,batch_size=batch,
                selected_epoch=UNKNOWN,total_updates=UNKNOWN,K_origin_exposures=UNKNOWN,direct_K_exposure_fraction=UNKNOWN,
                completion=value.get('status',UNKNOWN), exact_minimum_protocol_complete=value.get('exact_minimum_protocol_complete',False),
                actual_F_rows=streams[0] if len(streams)==1 else UNKNOWN,actual_F_rows_scope='per timed physical prediction',
                actual_F_rows_evidence='LOGGED_TIMED_TRIAL_COUNTERS',actual_K_row_parity=stream_parity,
                checkpoint_bytes=storage.get('actual_artifact_bytes',UNKNOWN),
                checkpoint_bytes_evidence='HISTORICAL_LOGGED_ACTUAL_FILE_BYTES',
                checkpoint_bytes_scope='all strategy artifacts; common foundation excluded',
                switch_time=float(np.median(switch)) if switch else UNKNOWN,
                switch_time_evidence='MEDIAN_OF_HISTORICAL_TIMED_TRIALS' if switch else UNKNOWN,
                switch_time_scope='switch_total_seconds auxiliary probe; seconds; overlaps guarded whole',
                transition_plus_physical_median_seconds=float(np.median(whole)) if whole else UNKNOWN,
                timed_trials=len(group), source_reference=reference, original_result_sha256=hashlib.sha256(raw).hexdigest(),
                evidence_scope='historical actual transition records; no new timing or training'))
    return dict(status='COMPLETE', rows=rows, transitions=transition, source_hashes=sources,
                private_artifacts=private, new_training_jobs=0,new_F_calls=0,
                missing_log_policy='UNKNOWN; updates never substitute for K-origin exposures',
                prefix_rule='D.reshape(7,Kmax,C)[:, :K, :].reshape(7*K,C)',
                context='Original ORDERED and MULTIK_TRANSITION protocols; existing repeated-V and post-T history retained')


def execute_job(job, ctx):
    result = audit_sources(ctx['repo'],ctx)
    output = Path(ctx['report'])/'MULTIK_AUDIT.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    rows = result['rows']
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['parent','stage','K','completion']
    with output.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    evidence = Path(ctx['report'])/'jobs'/(job['job_id']+'.json')
    for name in ('ORDERED_PROTOCOL_KO.md','MULTIK_TRANSITION_PROTOCOL_KO.md'):
        source = Path(ctx['repo'])/BROAD/name
        if source.is_file():
            result['source_hashes'][source.relative_to(Path(ctx['repo'])).as_posix()] = sha(source)
    atomic_json(evidence,result)
    notes = Path(ctx['report'])/'MULTIK_AUDIT_KO.md'
    notes.write_text('# 기존 다중 K 근거 감사\n\n'
        f'[확인] 기존 공개 근거에서 {len(rows)}개 stage/K 또는 예측 alias 행을 읽었다. 신규 학습과 F 호출은 0이다.\n\n'
        'K별 origin 노출은 실제 k_exposures만 사용한다. 누락 로그, checkpoint bytes, 전환 시간은 UNKNOWN으로 남긴다. '
        'K×기록 노출로 얻은 행수와 직접 기록된 전체 F training rows는 구분한다. 같은 update 수는 같은 K별 노출을 뜻하지 않는다.\n\n'
        '공유 parent, 독립 K, Kmax 전용 parent의 사후 prefix, nested PCA는 서로 다른 근거다. '
        '과거 T 사용 및 반복 V 개발 평가 이력을 독립 holdout 검증으로 바꾸지 않는다.\n\n'
        'D prefix는 reshape(7,Kmax,C)[:, :K, :]이며 flatten된 앞 7K행 절단을 사용하지 않는다. '
        '이번 감사는 새로운 작은 K 비용 측정이나 신규 다중 K 정확도 실험이 아니다.\n',encoding='utf-8')
    return dict(status='COMPLETE',reason=None,artifact_paths=[str(output),str(evidence),str(notes)],result=result)
