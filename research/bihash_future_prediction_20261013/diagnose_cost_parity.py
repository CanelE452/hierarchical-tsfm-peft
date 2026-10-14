"""Small actual GPU reproduction of the preserved SML cost parity failure.

The benchmark algorithm and its comparator are not changed. Only past EVAL
inputs and saved prediction/stat/origin members are loaded, never EVAL targets.
"""
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
import torch
import runtime as rt
import model as bm
import benchmark as bench
import train as tr

DATASET = 'uci274_sml2010'
LABEL = 'B_COSINE_w0_s92601'


def cpu_fp32_inverse(normalized, mean, std):
    product = np.multiply(normalized.astype(np.float32), std.astype(np.float32))
    return np.add(product, mean.astype(np.float32))


def exact_fp64_inverse(normalized, mean, std):
    return normalized.astype(np.float64)*std.astype(np.float64)+mean.astype(np.float64)


def compare_normalized_reinference(actual, reference):
    error = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    allowed = 1e-5+2e-6*np.abs(reference.astype(np.float64))
    return {'gate_pass': bool(np.all(error <= allowed)), 'atol': 1e-5, 'rtol': 2e-6,
            'max_abs_gap': float(error.max()), 'failed_cell_count': int(np.count_nonzero(error > allowed)),
            'target_derived_MSE_bound_measured': False}


def compare_outputs(normalized, physical, science, mean, std, columns):
    cpu_same = cpu_fp32_inverse(normalized, mean, std)
    exact_same = exact_fp64_inverse(normalized, mean, std)
    cpu_science = cpu_fp32_inverse(science, mean, std)
    exact_science = exact_fp64_inverse(science, mean, std)
    original_error = np.abs(physical-cpu_science)
    original_allowed = 2e-3+2e-5*np.abs(cpu_science)
    failed = original_error > original_allowed
    norm_delta = normalized.astype(np.float64)-science.astype(np.float64)
    inverse_rounding = physical.astype(np.float64)-exact_same
    science_rounding = cpu_science.astype(np.float64)-exact_science
    decomposition = norm_delta*std.astype(np.float64)+inverse_rounding-science_rounding
    actual_difference = physical.astype(np.float64)-cpu_science.astype(np.float64)
    failed_cells = []
    for b, h, c in np.argwhere(failed):
        product = float(normalized[b, h, c])*float(std[c])
        exact_value = float(exact_same[b, h, c])
        cancellation = (abs(product)+abs(float(mean[c])))/abs(exact_value) if exact_value else None
        failed_cells.append({'window_index': int(b), 'horizon_1based': int(h)+1, 'channel_index': int(c), 'channel': columns[c],
            'normalized_actual': float(normalized[b, h, c]), 'normalized_science': float(science[b, h, c]),
            'normalized_delta': float(norm_delta[b, h, c]), 'mean': float(mean[c]), 'std': float(std[c]),
            'physical_GPU': float(physical[b, h, c]), 'science_physical_CPU_FP32': float(cpu_science[b, h, c]),
            'same_norm_physical_CPU_FP32': float(cpu_same[b, h, c]), 'same_norm_exact_FP64': exact_value,
            'physical_abs_gap': float(original_error[b, h, c]), 'original_allowed_abs_gap': float(original_allowed[b, h, c]),
            'normalized_delta_times_std': float(norm_delta[b, h, c]*float(std[c])),
            'GPU_inverse_rounding_vs_exact': float(inverse_rounding[b, h, c]),
            'CPU_science_inverse_rounding_vs_exact': float(science_rounding[b, h, c]),
            'physical_result_FP32_ULP': float(abs(np.spacing(cpu_science[b, h, c]))),
            'FP32_multiply_term_ULP': float(abs(np.spacing(np.float32(product)))),
            'FP32_mean_ULP': float(abs(np.spacing(mean[c]))),
            'relative_cancellation_factor': cancellation, 'exact_physical_zero': exact_value == 0.})
    summary = {'original_physical_gate_pass': bool(not failed.any()), 'original_physical_atol': .002, 'original_physical_rtol': 2e-5,
               'failed_cell_count': int(failed.sum()), 'failed_channels': sorted({columns[c] for _, _, c in np.argwhere(failed)}),
               'physical_max_abs_vs_science': float(original_error.max()),
               'preinverse_normalized_max_abs_vs_science': float(np.abs(norm_delta).max()),
               'normalized_roundtrip_max_abs_vs_science': float(np.abs((physical-mean)/std-science).max()),
               'GPU_physical_vs_CPU_separate_FP32_same_norm_max_abs': float(np.abs(physical-cpu_same).max()),
               'GPU_physical_vs_exact_FP64_same_norm_max_abs': float(np.abs(inverse_rounding).max()),
               'CPU_science_FP32_vs_exact_FP64_max_abs': float(np.abs(science_rounding).max()),
               'same_norm_CPU_FP32_physical_gate_pass': bool(np.all(np.abs(cpu_same-cpu_science) <= original_allowed)),
               'both_exact_FP64_inverse_original_physical_gate_pass': bool(np.all(np.abs(exact_same-exact_science) <= .002+2e-5*np.abs(exact_science))),
               'error_decomposition_residual_max_abs': float(np.abs(decomposition-actual_difference).max()),
               'failed_cells': failed_cells}
    arrays = {'normalized_preinverse': normalized, 'physical_gpu': physical, 'science_normalized': science,
              'same_norm_physical_CPU_separate_FP32': cpu_same, 'same_norm_physical_exact_FP64': exact_same,
              'science_physical_CPU_FP32': cpu_science, 'science_physical_exact_FP64': exact_science,
              'original_physical_failed_mask': failed, 'original_physical_allowed_error': original_allowed,
              'normalized_delta_times_std_FP64': norm_delta*std.astype(np.float64),
              'GPU_inverse_rounding_vs_exact_FP64': inverse_rounding,
              'CPU_science_inverse_rounding_vs_exact_FP64': science_rounding,
              'error_decomposition_FP64': decomposition}
    return summary, arrays


def verify(artifact):
    path = rt.ROOT / artifact['path']
    if rt.sha(path) != artifact['sha256']: raise ValueError('DIAGNOSTIC_ASSET_SHA_MISMATCH: ' + str(path))
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', default='cost_parity_actual_v1')
    parser.add_argument('--batch-matched-core-reference', action='store_true')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum(): raise ValueError('INVALID_EXCLUSIVE_DIAGNOSTIC_TAG')
    folder = rt.HERE / 'cost_parity_diagnostics_v1' / args.tag; folder.mkdir(parents=True, exist_ok=False)
    with (folder/'SOURCE_AT_RUN.py').open('xb') as handle: handle.write(Path(__file__).read_bytes())
    limit = rt.Limit(args.tag, gpu=True); diagnostic_started = time.perf_counter()
    wall_cap = 60 if args.batch_matched_core_reference else 120
    output = {'status': 'INCOMPLETE', 'tag': args.tag, 'dataset': DATASET, 'label': LABEL, 'batch_results': [],
              'training_updates': 0, 'algorithm_or_comparator_changed': False, 'EVAL_target_arrays_opened': False,
              'FIT_TUNE_dataset_loader_used': False, 'protected_TEST_opened': False,
              'new_GPU_process_wall_cap_seconds': wall_cap,
              'batch_matched_core_reference': args.batch_matched_core_reference,
              'evidence_scope': 'Separate diagnostic inference/readout only; original 404 timings and FAILED_PRESERVED receipt remain unchanged.'}
    terminal, error = 'COMPLETE', None
    try:
        lock = rt.read(rt.HERE/'METHOD_LOCK.json'); sources = lock['source_hashes']
        for path, digest in sources.items():
            if rt.sha(rt.ROOT/path) != digest: raise ValueError('SCIENTIFIC_SOURCE_CHANGED_BEFORE_DIAGNOSTIC')
        snapshot = rt.source_snapshot(args.tag)
        if snapshot != sources: raise ValueError('DIAGNOSTIC_SNAPSHOT_NOT_SCIENTIFIC_SOURCE16')
        cost_path = rt.HERE/'run_cost_v1/RESULT.json'; fresh_path = rt.HERE/'run_fresh_merged_v1/RESULT.json'; eval_path = rt.HERE/'run_evaluate_v1/RESULT.json'
        original_cost, fresh, evaluation = [rt.read(p) for p in (cost_path, fresh_path, eval_path)]
        if original_cost['status'] != 'FAILED_PRESERVED' or fresh['status'] != 'COMPLETE' or evaluation['status'] != 'COMPLETE': raise ValueError('WRONG_AUTHORITATIVE_TERMINAL_INPUT')
        case = next(c for c in fresh['cases'] if c['dataset'] == DATASET and c['label'] == LABEL)
        record = next(c for c in evaluation['core_evaluation'] if c['dataset'] == DATASET and c['source_label'] == LABEL)
        science_ref = next(c['prediction'] for c in record['entries'] if c['variant'] == 'regular')
        cp_ref = case['training']['selected_checkpoint']; cp_path = verify(cp_ref); science_path = verify(science_ref)
        contract = fresh['contracts'][DATASET]; past_ref = contract['payloads']['eval_past_input']; past_path = verify(past_ref)
        with np.load(past_path, allow_pickle=False) as z:
            raw, observed, history, origins, mean, std = [z[key].copy() for key in ('physical_history', 'history_observed', 'history', 'origins', 'mean', 'std')]
        # Do not read eval_target or eval_observed members from this forecast NPZ.
        with np.load(science_path, allow_pickle=False) as z:
            science, science_origins, science_mean, science_std = [z[key].copy() for key in ('eval_prediction', 'eval_origins', 'mean', 'std')]
        exact_contract = {'mean_exact': np.array_equal(mean, science_mean), 'std_exact': np.array_equal(std, science_std), 'origins_exact': np.array_equal(origins, science_origins)}
        if not all(exact_contract.values()): raise ValueError('PAST_SCIENCE_STAT_OR_ORIGIN_MISMATCH')
        pack = {'dataset': DATASET, 'mean': mean, 'std': std}
        preprocessed = bench.preprocess(pack, raw[:8], observed[:8])
        exact_contract['CPU_preprocess_vs_declared_past_history_exact'] = np.array_equal(preprocessed, history[:8])
        exact_contract['CPU_preprocess_vs_declared_past_max_abs'] = float(np.abs(preprocessed-history[:8]).max())
        if not exact_contract['CPU_preprocess_vs_declared_past_history_exact']: raise ValueError('ACTUAL_PREPROCESS_NOT_DECLARED_PAST_HISTORY')
        output.update(input_equality=exact_contract, scientific_source_hashes=sources,
                      source_snapshot_manifest={'path': str((rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json').relative_to(rt.ROOT)), 'sha256': rt.sha(rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json')},
                      source_checkpoint=cp_ref, science_prediction=science_ref, EVAL_past_only=past_ref,
                      original_failed_cost={'path': str(cost_path.relative_to(rt.ROOT)), 'sha256': rt.sha(cost_path), 'status': original_cost['status'], 'successful_timing_rows_preserved': len(original_cost['records'])},
                      source_fresh={'path':str(fresh_path.relative_to(rt.ROOT)),'sha256':rt.sha(fresh_path)},
                      source_evaluation={'path':str(eval_path.relative_to(rt.ROOT)),'sha256':rt.sha(eval_path)},
                      accessed_past_NPZ_members=['physical_history','history_observed','history','origins','mean','std'],
                      accessed_science_NPZ_members=['eval_prediction','eval_origins','mean','std'],
                      columns=contract['columns'], selected_update=case['training']['selected_update'])
        bm.en.configure()
        forecaster = bm.en.ex.previous.FrozenBolt(); bm.en.replace_normalizer(forecaster)
        output['backbone_before_sha256'] = rt.state_hash(forecaster.model)
        if output['backbone_before_sha256'] != evaluation['backbone_before_sha256']: raise ValueError('WRONG_FROZEN_BACKBONE')
        stored = torch.load(cp_path, map_location='cpu', weights_only=True); state = stored['model']
        basis = state['codec.fixed_basis'].numpy().copy()
        model = bm.FutureBiHash(np.zeros(len(mean),np.float32),basis,case['seed'],case['method'],state['bank_delta'].numpy(),state['bank_observed'].numpy(),state['bank_origins'].numpy()).to('cuda')
        model.load_state_dict(state); model.eval().requires_grad_(False)
        before_model = rt.state_hash(model)
        if before_model != case['training']['selected_hash'] or not bool(model.memory_is_frozen): raise ValueError('SELECTED_STATE_OR_CACHED_BANK_NOT_EXACT')
        output['selected_model_before_sha256'] = before_model
        hook_calls = []
        def capture(module, inputs, result):
            hook_calls.append({'normalized':result.detach().cpu().numpy().copy(),
                               'input':inputs[0].detach().cpu().numpy().copy(),'origins':inputs[1].detach().cpu().numpy().copy()})
        handle = model.register_forward_hook(capture)
        batch_arrays = {}
        try:
            with torch.no_grad():
                matched_reference = None
                if args.batch_matched_core_reference:
                    matched_reference = tr.predict(model, history[:1], origins[:1], forecaster, limit, batch=1)
                    captured = hook_calls[-1]
                    if not np.array_equal(captured['input'], history[:1]) or not np.array_equal(captured['origins'], origins[:1]):
                        raise ValueError('INDEPENDENT_SCIENCE_INPUT_OR_ORIGIN_NOT_EXACT')
                    if rt.state_hash(model) != before_model: raise ValueError('SCIENCE_PREDICT_MUTATED_SELECTED_STATE')
                    output['matched_science_reference'] = {
                        'API': 'train.predict(model, normalized_history[:1], origins[:1], forecaster, limit, batch=1)',
                        'independent_of_benchmark_deployment': True, 'actual_forward_calls': 1,
                        'selected_model_state_unchanged': True,
                        'normalized_vs_saved_bulk': compare_normalized_reinference(matched_reference, science[:1]),
                        'artifact': rt.save_npz(folder/'B1_MATCHED_SCIENCE_REFERENCE.npz',
                            matched_science_normalized=matched_reference,
                            matched_science_physical_CPU_FP32=cpu_fp32_inverse(matched_reference, mean, std),
                            original_bulk_science_normalized=science[:1],
                            original_bulk_science_physical_CPU_FP32=cpu_fp32_inverse(science[:1], mean, std),
                            actual_input_normalized=history[:1],actual_origins=origins[:1],mean=mean,std=std)}
                    if limit.check()['elapsed_seconds'] > wall_cap: raise rt.BudgetHeld(f'DIAGNOSTIC_NEW_PROCESS_WALL_{wall_cap}SECONDS')
                batches = (1,) if args.batch_matched_core_reference else (1,8)
                calls = 1 if args.batch_matched_core_reference else 9
                for batch in batches:
                    physical_calls=[]; normalized_calls=[]
                    for call in range(calls):
                        actual = bench.deployment(pack,raw[:batch],observed[:batch],origins[:batch],'CORE',forecaster,model=model)
                        torch.cuda.synchronize(); captured=hook_calls[-1]
                        if not np.array_equal(captured['input'],history[:batch]) or not np.array_equal(captured['origins'],origins[:batch]): raise ValueError('ACTUAL_GPU_INPUT_OR_ORIGIN_NOT_EXACT')
                        physical_calls.append(actual); normalized_calls.append(captured['normalized'])
                        accounting = limit.check()
                        if accounting['elapsed_seconds'] > wall_cap: raise rt.BudgetHeld(f'DIAGNOSTIC_NEW_PROCESS_WALL_{wall_cap}SECONDS')
                    physical, normalized = physical_calls[-1], normalized_calls[-1]
                    reference = matched_reference if args.batch_matched_core_reference else science[:batch]
                    summary, arrays = compare_outputs(normalized,physical,reference,mean,std,contract['columns'])
                    if args.batch_matched_core_reference:
                        bulk_summary, _ = compare_outputs(normalized,physical,science[:batch],mean,std,contract['columns'])
                        summary['original_saved_bulk_comparison'] = bulk_summary
                        summary['normalized_vs_independent_matched_science'] = compare_normalized_reinference(normalized, matched_reference)
                        arrays.update(original_bulk_science_normalized=science[:batch],
                                      original_bulk_science_physical_CPU_FP32=cpu_fp32_inverse(science[:batch],mean,std))
                    summary.update(batch=batch,actual_forward_calls=calls,warmup_calls=0 if args.batch_matched_core_reference else 2,
                                   repeat_calls=1 if args.batch_matched_core_reference else 7,
                                   normalized_repeat_max_abs=float(np.abs(np.stack(normalized_calls)-normalized_calls[-1]).max()),
                                   physical_repeat_max_abs=float(np.abs(np.stack(physical_calls)-physical_calls[-1]).max()))
                    arrays.update(all_normalized_calls=np.stack(normalized_calls),all_physical_calls=np.stack(physical_calls),
                                  raw_history=raw[:batch],history_observed=observed[:batch],actual_input_normalized=history[:batch],
                                  actual_origins=origins[:batch],mean=mean,std=std)
                    summary['actual_arrays']=rt.save_npz(folder/f'B{batch}_ACTUAL.npz',**arrays)
                    output['batch_results'].append(summary); batch_arrays[batch]=(normalized.copy(),physical.copy())
        finally:
            handle.remove()
        if not args.batch_matched_core_reference:
            output['batch1_vs_batch8_first_window']={'normalized_max_abs':float(np.abs(batch_arrays[1][0]-batch_arrays[8][0][:1]).max()),
                                                 'physical_max_abs':float(np.abs(batch_arrays[1][1]-batch_arrays[8][1][:1]).max())}
        output['backbone_after_sha256']=rt.state_hash(forecaster.model); output['selected_model_after_sha256']=rt.state_hash(model)
        output['backbone_unchanged']=output['backbone_after_sha256']==output['backbone_before_sha256']
        output['selected_model_unchanged']=output['selected_model_after_sha256']==before_model
        output['original_CP_and_science_and_past_SHA_unchanged']=all(rt.sha(path)==ref['sha256'] for path,ref in ((cp_path,cp_ref),(science_path,science_ref),(past_path,past_ref)))
        output['scientific16_unchanged']=all(rt.sha(rt.ROOT/path)==digest for path,digest in sources.items())
        output['original_failed_cost_SHA_unchanged']=rt.sha(cost_path)==output['original_failed_cost']['sha256']
        if not all(output[key] for key in ('backbone_unchanged','selected_model_unchanged','original_CP_and_science_and_past_SHA_unchanged','scientific16_unchanged','original_failed_cost_SHA_unchanged')): raise ValueError('READ_ONLY_ASSET_OR_STATE_CHANGED')
        if limit.check()['elapsed_seconds'] > wall_cap: raise rt.BudgetHeld(f'DIAGNOSTIC_NEW_PROCESS_WALL_{wall_cap}SECONDS')
        output['status']='COMPLETE_DIAGNOSTIC_ORIGINAL_FAILURE_PRESERVED'
        output['interpretation_limits']=['Only this selected SML case and first1/8 past windows were inferred; no current target was read.',
            'Forward hook adds output/readout synchronization. Diagnostic runtime is not mixed into original timing statistics.',
            'No comparator, inverse algorithm, selection, weights or training source has been repaired. This diagnostic must precede any new numerical contract.',
            'A normalized roundtrip error also contains inverse rounding; it is separated from the captured preinverse model output here.']
    except Exception as exc:
        terminal='BUDGET_HELD_PRESERVED' if isinstance(exc,rt.BudgetHeld) else 'FAILED_PRESERVED'; error=str(exc)
        output.update(status=terminal,error=error,traceback=traceback.format_exc())
    output['resource_receipt']=limit.close(terminal,error)
    rt.write(folder/'RESULT.json',output)
    lines=['# 비용 parity 실패의 실제 수치 진단\n\n',f'상태: {output["status"]}. 원 비용 실패와 원 404개 timing은 그대로 보존했다. 모델/비교 기준 수정이나 학습은 수행하지 않았다.\n\n']
    if output.get('batch_results'):
        if args.batch_matched_core_reference:
            note = output['matched_science_reference']['normalized_vs_saved_bulk']
            lines.append(f'원 `train.predict`에 선언된 normalized past 1개를 넣어 batch1 참조를 한 번 만들고, 원 `benchmark.deployment`의 raw past 배포 경로를 한 번 실행했다. 물리 단위 원 gate .002/2e-5는 그대로다. 직접 참조와 원 bulk science의 normalized 차이는 {note["max_abs_gap"]:.10g}, 기존 재추론 1e-5/2e-6 gate는 {note["gate_pass"]}다. target를 읽지 않았으므로 target 기반 MSE 변화 상한은 측정하지 않았다. 별도 두 번의 실제 forward이며 비용 timing 재측정이 아니다.\n\n')
        else:
            lines.append('동일 selected CP와 frozen Bolt, 원 benchmark.deployment 경로로 batch1/8 각각 warm2+repeat7을 추론했다. forward hook에서 역정규화 전 실제 normalized 출력과 입력을 저장했다. EVAL past와 science의 prediction/mean/std/origins만 읽었고 target/observed 회원은 읽지 않았다.\n\n')
        for summary in output['batch_results']:
            if 'original_saved_bulk_comparison' in summary:
                bulk = summary['original_saved_bulk_comparison']
                lines.append(f'원 bulk 참조를 유지하면 physical gate {bulk["original_physical_gate_pass"]}, gap {bulk["physical_max_abs_vs_science"]:.10g}, 실패 cell {bulk["failed_cell_count"]}다. 아래는 동일 batch1의 독립 `train.predict` 참조와의 비교다.\n\n')
            lines.append(f'## Batch {summary["batch"]}\n\n원 physical gate: {summary["original_physical_gate_pass"]}, 실패 cell {summary["failed_cell_count"]}, 채널 {summary["failed_channels"]}. physical 최대 gap {summary["physical_max_abs_vs_science"]:.10g}, pre-inverse normalized 최대 gap {summary["preinverse_normalized_max_abs_vs_science"]:.10g}.\n\n동일 normalized 값의 GPU inverse와 CPU separate FP32 inverse gap {summary["GPU_physical_vs_CPU_separate_FP32_same_norm_max_abs"]:.10g}, GPU inverse와 exact FP64 inverse gap {summary["GPU_physical_vs_exact_FP64_same_norm_max_abs"]:.10g}. 두 값을 모두 exact FP64로 inverse해도 원 physical gate 통과 여부: {summary["both_exact_FP64_inverse_original_physical_gate_pass"]}.\n\n')
            for cell in summary['failed_cells'][:12]:
                lines.append(f'- h{cell["horizon_1based"]}, `{cell["channel"]}`: std {cell["std"]:.10g}, normalized delta {cell["normalized_delta"]:.10g}, delta×std {cell["normalized_delta_times_std"]:.10g}, physical gap {cell["physical_abs_gap"]:.10g}, 허용치 {cell["original_allowed_abs_gap"]:.10g}, multiply ULP {cell["FP32_multiply_term_ULP"]:.10g}.\n')
        lines.append('\n실패 재현 recipe: `B1_ACTUAL.npz`의 physical_gpu와 science_physical_CPU_FP32를 원 식 `abs(actual-reference) <= .002 + 2e-5*abs(reference)`로 비교한다. normalized_preinverse와science_normalized를별도로대조하고 error_decomposition_FP64가 delta×std + 각 inverse rounding 차이와 physical gap을 분해하는지 검사한다. 실제 target는 필요하지 않다.\n')
    lines.append('\n이 문서는 비교기의 수리 제안이나 tolerance 확대가 아니다. 입력·CP·F·통계·반복성·batch 차이·inverse 반올림을 확인한 근거를 먼저 남긴 것이다.\n')
    with (folder/'REPORT_KO.md').open('x',encoding='utf-8') as handle: handle.write(''.join(lines))
    print(rt.json_line({'diagnostic_result':str(folder/'RESULT.json'),'status':output['status'],'batches':len(output['batch_results'])}),flush=True)
    if terminal!='COMPLETE': raise SystemExit(2)


if __name__=='__main__':main()
