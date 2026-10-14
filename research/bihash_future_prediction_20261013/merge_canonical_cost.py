"""Canonical 296 CUDA core/raw-Bolt rows plus all 240 CPU Ridge rows."""
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
import runtime as rt
import run as ru
import batch_science_reference as bs
import cpu_ridge_deployment as cpu


def key(row):return bs.row_key(row)


def combine_rows(cuda_rows,cpu_rows,expected,cuda_expected=296,cpu_expected=240):
    if len(cuda_rows)!=cuda_expected or len(cpu_rows)!=cpu_expected:raise ValueError('CANONICAL_BACKEND_COUNTS_NOT_EXACT')
    if any(r['route'] not in ('CORE','RAW_BOLT') or not r['parity_pass'] for r in cuda_rows):raise ValueError('CANONICAL_CUDA_ROUTE_OR_GATE_INVALID')
    if any(r['route'] not in cpu.RIDGE_ROUTES or not r['parity_pass'] or r.get('deployment_implementation')!=cpu.IMPLEMENTATION_ID for r in cpu_rows):
        raise ValueError('CANONICAL_CPU_ROUTE_IMPLEMENTATION_OR_GATE_INVALID')
    result=list(cuda_rows)+list(cpu_rows);keys=[key(r) for r in result]
    if len(set(keys))!=len(result) or set(keys)!=expected:raise ValueError('CANONICAL_EXACT_UNIQUE_ROUTE_BATCH_COVERAGE_FAILED')
    return result


def verify_rows(rows):
    for r in rows:
        seconds=np.asarray(r['seconds'],np.float64)
        if r['warmup']!=2 or r['repeats']!=7 or seconds.shape!=(7,) or not np.isfinite(seconds).all() or np.any(seconds<=0):
            raise ValueError('CANONICAL_TIMING_NOT_POSITIVE_SEVEN_REPEAT')
        if r['physical_parity_atol']!=.002 or r['physical_parity_rtol']!=2e-5:raise ValueError('CANONICAL_PHYSICAL_TOLERANCE_CHANGED')
        if not np.isclose(r['median_seconds'],np.median(seconds),rtol=1e-12,atol=0) or not np.isclose(r['median_seconds_per_window'],np.median(seconds)/r['batch'],rtol=1e-12,atol=0):
            raise ValueError('CANONICAL_MEDIAN_NOT_SOURCE_TIMINGS')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda-cost-result',type=Path,required=True);parser.add_argument('--cpu-cost-result',type=Path,required=True)
    parser.add_argument('--method-lock',type=Path,required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('CANONICAL_INVALID_EXCLUSIVE_TAG')
    folder=rt.HERE/'canonical_cost_v1'/args.tag;folder.mkdir(parents=True,exist_ok=False)
    with (folder/'SOURCE_AT_RUN.py').open('xb') as h:h.write(Path(__file__).read_bytes())
    limit=rt.Limit(args.tag,gpu=False);terminal,error='COMPLETE',None
    output={'status':'INCOMPLETE','stage':'cost','tag':args.tag,'started_unix':time.time(),'records':[],
        'timing_producers':[],'failed_paths':[],'merge_training_updates':0,'merge_GPU_used':False,
        'EVAL_target_arrays_opened':False,'protected_TEST_opened':False}
    try:
        cuda=rt.read(args.cuda_cost_result);heads=rt.read(args.cpu_cost_result);lock=ru.check_method_sources(args.method_lock)
        if cuda['status']!='FAILED_PRESERVED' or len(cuda['records'])!=536 or heads['status']!='COMPLETE':raise ValueError('CANONICAL_PRODUCER_STATUS_OR_COVERAGE')
        if cuda['source_hashes']!=lock['source_hashes'] or heads['source_hashes']!=lock['source_hashes']:raise ValueError('CANONICAL_SCIENTIFIC_SOURCE16_MISMATCH')
        if cuda['method_lock']['sha256']!=rt.sha(args.method_lock) or heads['method_lock']['sha256']!=rt.sha(args.method_lock):raise ValueError('CANONICAL_METHOD_LOCK_MISMATCH')
        cuda_ref={'path':str(args.cuda_cost_result.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(args.cuda_cost_result)}
        cpu_ref={'path':str(args.cpu_cost_result.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(args.cpu_cost_result)}
        values={}
        for name in ('dev','fresh','probe','evaluation'):
            ref=heads['source_results'][name];values[name]=rt.read(bs.verify_artifact(ref))
            if cuda['source_results'][name]['sha256']!=ref['sha256']:raise ValueError('CANONICAL_SOURCE_RESULTS_NOT_IDENTICAL')
        routes,aliases=bs.build_routes(values['dev'],values['fresh'],values['probe'],values['evaluation'],lock)
        expected={(r['dataset'],r['identity'],r['route'],b) for r in routes for b in (1,8)}
        selected_cuda=[r for r in cuda['records'] if r['route'] in ('CORE','RAW_BOLT')]
        records=combine_rows(selected_cuda,heads['records'],expected);verify_rows(records)
        cuda_producers={tuple(p['key']):p for p in cuda['timing_producers']}
        if len(cuda_producers)!=536:raise ValueError('CANONICAL_ORIGINAL_CUDA_TIMING_LINEAGE_NOT_UNIQUE')
        for row in selected_cuda:
            lineage=cuda_producers[key(row)]
            output['timing_producers'].append({**lineage,'key':list(key(row)),
                'deployment_implementation':'original_cuda_fp32','canonical_selected_from_producer':cuda_ref,
                'source_row_index':cuda['records'].index(row),'source_row_object_unchanged':True})
        for index,row in enumerate(heads['records']):
            bs.verify_artifact(row['actual_output']);bs.verify_artifact(row['matched_science_reference']['artifact'])
            if not row['matched_science_to_original_bulk_normalized']['gate_pass'] or not all(p['parity_pass'] for p in row['all_seven_matched_physical_comparisons']):
                raise ValueError('CANONICAL_CPU_CRITICAL_GATE_NOT_PASSED')
            output['timing_producers'].append({'key':list(key(row)),'deployment_implementation':cpu.IMPLEMENTATION_ID,
                'source_result':cpu_ref,'resource_receipt':heads['resource_receipt'],'source_row_index':index,
                'source_row_object_unchanged':True,'reference_kind':row['reference_kind']})
        receipts=cuda['timing_resource_receipts_counted_once']+[heads['resource_receipt']]
        unique={str((rt.ROOT/r['path']).resolve()):r for r in receipts}
        if len(unique)!=3:raise ValueError('CANONICAL_ORIGINAL_AND_NEW_GPU_RECEIPTS_NOT_THREE_UNIQUE')
        for receipt in unique.values():
            body=rt.read(bs.verify_artifact(receipt))
            if not body['gpu'] or body['wall_seconds']<=0:raise ValueError('CANONICAL_GPU_RECEIPT_INVALID')
        if heads['backbone_before_sha256']!=heads['backbone_after_sha256'] or heads['backbone_before_sha256']!=cuda['backbone_before_sha256'] or cuda['backbone_before_sha256']!=cuda['backbone_after_sha256']:
            raise ValueError('CANONICAL_BACKBONE_SHA_MISMATCH')
        ru.source_check(lock['source_hashes']);bs.verify_artifact(cuda_ref);bs.verify_artifact(cpu_ref)
        output.update(status='COMPLETE',source_hashes=lock['source_hashes'],method_lock=heads['method_lock'],
            global_model_lock=heads['global_model_lock'],source_results=heads['source_results'],records=records,aliases=aliases,
            expected_rows=536,canonical_paths=268,cuda_rows_reused=296,cpu_Ridge_rows_uniform=240,
            exact_route_batch_coverage=True,backend_uniformity_gate=True,original_objects_unchanged=True,
            producers=[{'source_result':cuda_ref,'original_status':cuda['status'],'selected_rows':296},
                {'source_result':cpu_ref,'original_status':heads['status'],'selected_rows':240}],
            original_cuda_536_failure_preserved={'source_result':cuda_ref,'status':cuda['status'],
                'failed_paths':cuda['failed_paths'],'failed_row_count':sum(not r['parity_pass'] for r in cuda['records']),
                'original_404_failed_cost':cuda['original_failed_attempt'],'status_or_objects_rewritten':False},
            timing_resource_receipts_counted_once=list(unique.values()),
            backend_contract={'CUDA_routes':['CORE','RAW_BOLT'],'CPU_routes':list(cpu.RIDGE_ROUTES),
                'CPU_implementation_id':cpu.IMPLEMENTATION_ID,'CPU_BLAS_threads':2,
                'all_Ridge_routes_use_CPU_not_only_previously_failed_rows':True,
                'reference_batch_protocol':bs.PROTOCOL['reference_kind'],'physical_atol':.002,'physical_rtol':2e-5,
                'bulk_accuracy_or_model_selection_changed':False,'old_CUDA_BLAS_threads_equal_new_CPU_threads_claimed':False},
            backbone_before_sha256=heads['backbone_before_sha256'],backbone_after_sha256=heads['backbone_after_sha256'],
            helper_scripts=heads['helper_scripts']+[{'path':str(Path(__file__).resolve().relative_to(rt.ROOT)),'sha256':rt.sha(__file__),
                'snapshot_path':str((folder/'SOURCE_AT_RUN.py').relative_to(rt.ROOT)),'snapshot_sha256':rt.sha(folder/'SOURCE_AT_RUN.py')}],
            resource_accounting_note='Original CUDA failure and recovery GPU receipts each once, plus new CPU-head GPU process once; this merge has only its own CPU receipt')
        limit.check()
    except Exception as exc:
        terminal='BUDGET_HELD_PRESERVED' if isinstance(exc,rt.BudgetHeld) else 'FAILED_PRESERVED';error=str(exc)
        output.update(status=terminal,error=error,traceback=traceback.format_exc())
    output['finished_unix']=time.time();output['resource_receipt']=limit.close(terminal,error)
    rt.write(folder/'RESULT.json',output)
    print(rt.json_line({'canonical_cost_result':str(folder/'RESULT.json'),'status':output['status'],'rows':len(output['records'])}),flush=True)
    if terminal!='COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
