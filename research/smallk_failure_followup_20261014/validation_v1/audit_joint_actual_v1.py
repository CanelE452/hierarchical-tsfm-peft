"""CPU-only completed-joint-run audit; never load or execute F."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
from followup_runtime import FollowupLimit
from audit_D1_actual_v1 import sha, read, resolve, array_sha, score
from audit_D2_actual_v1 import state_hash

OUT = HERE / 'JOINT_ARRAY_AUDIT_v1.json'
MD = HERE / 'JOINT_ARRAY_AUDIT_v1.md'
SOURCE_SHA = '352cb68c499a84ce9e5e32b95f4adfb2af96c89b226da28eec44630a3b26d21f'
LOCK_SHA = '43902321ae0a2caec166b050171f6fa164165dba727cf4d8bfabafd356e4e101'
CHECKPOINTS = (0, 100, 200, 400, 600)


def objective_value(y, raw, objective):
    return y if objective == 'Y' else raw if objective == 'RAW' else .5 * y + .5 * raw


def recompute_scores(prediction, target, teacher, observed, std):
    return {'normalized_actual_Y': score(prediction, target, observed),
            'physical_actual_Y': score(prediction, target, observed, std),
            'normalized_RAW_teacher': score(prediction, teacher, observed)}


def main():
    if OUT.exists() or MD.exists():
        raise FileExistsError('preserve original audit; use a new tag')
    expected = [(dataset, short, seed, objective) for dataset, short in
                (('robin', 'robin'), ('beijing_aotizhongxin', 'beijing_aotizhongxin'))
                for seed in (92601, 92602) for objective in ('Y', 'RAW', 'MIXED')]
    folders = {f'joint_{short}_{objective.lower()}_s{seed}_v2': ROOT / '.cache' / STUDY.name / 'joint_objectives' /
               f'joint_{short}_{objective.lower()}_s{seed}_v2' for dataset, short, seed, objective in expected}
    # Do not instantiate a resource process or read CP/NPZ from unfinished fits.
    for tag, folder in folders.items():
        if not (folder / 'RESULT.json').is_file() or read(folder / 'RESULT.json')['status'] != 'COMPLETE':
            raise ValueError('ALL_TWELVE_TERMINAL_COMPLETE_REQUIRED:' + tag)
    result = {'status': 'RUNNING', 'checks': 0, 'failed_count': 0, 'failures': [], 'conditions': [],
              'artifacts': [], 'F_forward_replayed': False, 'GPU_initialized': False,
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'curve_claim_scope': '600 recorded updates/finite values/objective arithmetic; per-update predictions not stored, so not independently re-forwarded',
              'physical_interpretation': 'per-channel native units; no heterogeneous-unit superiority claim',
              'checkpoints_recomputed': 0, 'selected_outputs_recomputed': 0}
    limit = None
    started = time.perf_counter()
    try:
        limit = FollowupLimit('joint_independent_audit_v1', gpu=False)
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        import torch
        torch.set_num_threads(2)
        def check(name, passed, details=None):
            result['checks'] += 1
            if not bool(passed):
                result['failed_count'] += 1
                result['failures'].append({'check': name, 'details': details})
        def exact(name, actual, expected_value):
            check(name, actual.dtype == expected_value.dtype and actual.shape == expected_value.shape
                  and np.array_equal(actual, expected_value, equal_nan=True))
        def close(name, actual, expected_value, atol=1e-11, rtol=1e-11):
            if isinstance(actual, list):
                check(name, len(actual) == len(expected_value) and all(
                    a is None and b is None or a is not None and b is not None and np.isclose(a, b, atol=atol, rtol=rtol)
                    for a, b in zip(actual, expected_value)))
            else:
                check(name, bool(np.isclose(actual, expected_value, atol=atol, rtol=rtol)))
        def metric_check(name, actual, declared):
            for scale in actual:
                for metric, value in actual[scale].items():
                    close(name + ':' + scale + ':' + metric, value, declared[scale][metric])
        def checked_npz(record):
            path = resolve(record['path'])
            before = sha(path)
            check(str(path) + ':file_sha', before == record['sha256'])
            with np.load(path, allow_pickle=False) as opened:
                arrays = {name: opened[name].copy() for name in opened.files}
            if 'arrays' in record:
                check(str(path) + ':array_keys', set(arrays) == set(record['arrays']))
                for name, value in arrays.items():
                    expected_value = record['arrays'][name]
                    check(str(path) + ':' + name + ':byte_sha', array_sha(value) == expected_value['sha256'])
                    check(str(path) + ':' + name + ':dtype_shape', str(value.dtype) == expected_value['dtype']
                          and list(value.shape) == expected_value['shape'])
            after = sha(path)
            check(str(path) + ':unchanged', after == before)
            result['artifacts'].append({'path': str(path.relative_to(ROOT)), 'sha256_before': before, 'sha256_after': after})
            limit.check()
            return arrays
        def checked_cp(record):
            path = resolve(record['path'])
            before = sha(path)
            check(str(path) + ':file_sha', before == record['sha256'])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            check(str(path) + ':unchanged', sha(path) == before)
            if 'state_sha_old_algorithm' in record:
                check(str(path) + ':old_state_sha', state_hash(payload['codec']) == record['state_sha_old_algorithm'])
                check(str(path) + ':Bi_state_sha', state_hash(payload['codec'], bi=True) == record['state_sha_Bi_algorithm'])
            limit.check()
            return payload
        check('frozen_source_SHA', sha(STUDY / 'run_joint_objectives_v2.py') == SOURCE_SHA)
        check('frozen_method_lock_SHA', sha(STUDY / 'METHOD_LOCK_JOINT_v2.md') == LOCK_SHA)
        groups = {}
        for dataset, short, seed, objective in expected:
            tag = f'joint_{short}_{objective.lower()}_s{seed}_v2'
            folder = folders[tag]
            producer = read(folder / 'RESULT.json')
            check(tag + ':condition', producer['dataset'] == dataset and producer['seed'] == seed and producer['objective'] == objective)
            check(tag + ':F_frozen', producer['F_before'] == producer['F_after'] and producer['F_frozen_pass']
                  and producer['F_has_no_parameter_gradients'])
            check(tag + ':scope', producer['independent_holdout'] is False and producer['protected_TEST_accessed'] is False
                  and producer['old_campaign_resumed'] is False and producer['local_process_wall_limit_seconds'] == 150)
            check(tag + ':recipe', producer['recipe']['new_updates'] == 600 and producer['recipe']['batch_size'] == 4
                  and producer['recipe']['lr'] == .001 and producer['recipe']['weight_decay'] == 0.
                  and producer['recipe']['clip_norm'] == 1. and producer['recipe']['checkpoints'] == list(CHECKPOINTS))
            for record in producer['source_snapshots']:
                check(tag + ':snapshot:' + record['path'], sha(resolve(record['snapshot_path'])) == record['snapshot_sha256'] == record['sha256'])
            original_cp = checked_cp(producer['original_MAIN_checkpoint']['checkpoint'])
            initial = original_cp['codec']
            check(tag + ':initial_original_state', state_hash(initial) == producer['initial_codec_hash']
                  == producer['original_MAIN_checkpoint']['selected_codec_hash_old_algorithm'])
            check(tag + ':initial_Bi_state', state_hash(initial, bi=True) == producer['initial_codec_Bi_hash'])
            bridgepath = resolve(producer['bridge_provenance']['input_bridge']['path'])
            bridge = read(bridgepath)
            check(tag + ':bridge_sha', sha(bridgepath) == producer['bridge_provenance']['input_bridge']['sha256'])
            inputs = {}
            for record in bridge['artifacts']:
                if record['dataset'] == dataset:
                    inputs[record['split']] = checked_npz(record)
            original_raw = checked_npz(producer['original_RAW_teacher_source'])
            original_main = checked_npz(producer['original_MAIN_prediction_source'])
            for split, values in inputs.items():
                for origin_label, container in [('RAW', original_raw), ('MAIN', original_main)]:
                    for suffix, field in [('target', 'target'), ('observed', 'observed'), ('origins', 'origins')]:
                        exact(tag + ':original_' + origin_label + ':' + split + ':' + suffix, container[split + '_' + suffix], values[field])
                    exact(tag + ':original_' + origin_label + ':' + split + ':mean', container['mean'], values['mean'])
                    exact(tag + ':original_' + origin_label + ':' + split + ':std', container['std'], values['std'])
                exact(tag + ':original_RAW:' + split + ':cached_teacher', original_raw[split + '_prediction'], values['raw_forecast'])
                metric_check(tag + ':original_MAIN:' + split + ':scores', recompute_scores(original_main[split + '_prediction'],
                             values['target'], values['raw_forecast'], values['observed'], values['std']), producer['original_MAIN_scores'][split])
                metric_check(tag + ':original_RAW:' + split + ':scores', recompute_scores(original_raw[split + '_prediction'],
                             values['target'], values['raw_forecast'], values['observed'], values['std']), producer['original_RAW_scores'][split])
            teacher = checked_npz(producer['FIT_teacher_arrays'])
            for saved, source in [('cached_RAW_forecast', 'raw_forecast'), ('actual_Y', 'target'),
                                   ('actual_Y_observed', 'observed'), ('origins', 'origins'), ('mean', 'mean'), ('std', 'std')]:
                exact(tag + ':FIT_teacher:' + saved, teacher[saved], inputs['fit'][source])
            fit_observed = teacher['actual_Y_observed']
            finite_contract = {
                'actual_Y_observed_all_finite': bool(np.isfinite(teacher['actual_Y'][fit_observed]).all()),
                'actual_Y_unobserved_all_finite': bool(np.isfinite(teacher['actual_Y'][~fit_observed]).all()),
                'actual_Y_unobserved_nonfinite_count': int((~np.isfinite(teacher['actual_Y'][~fit_observed])).sum()),
                'cached_RAW_teacher_all_finite': bool(np.isfinite(teacher['cached_RAW_forecast']).all()),
                'fit_per_channel_observed_counts': fit_observed.sum((0, 1)).tolist(),
                'zero_observed_fit_channels': np.flatnonzero(fit_observed.sum((0, 1)) == 0).tolist()}
            for name, value in finite_contract.items():
                check(tag + ':FIT_finite_contract:' + name, value == producer['FIT_target_finite_contract'][name])
            check(tag + ':teacher_historical_cost_unknown', producer['cached_teacher_cost']['historical_teacher_generation_compute_wall_seconds'] is None
                  and producer['cached_teacher_cost']['historical_teacher_generation_GPU_process_wall_seconds'] is None)
            order = checked_npz(producer['training_order'])['indices']
            expected_order = np.random.default_rng(seed).integers(0, len(inputs['fit']['history']), size=(600, 4))
            exact(tag + ':same_seed_training_order', order, expected_order)
            check(tag + ':order_hash', array_sha(order) == producer['training_order_sha256'])
            group = groups.setdefault((dataset, seed), [])
            group.append({'objective': objective, 'initial': producer['initial_codec_hash'],
                          'original_CP_sha': producer['original_MAIN_checkpoint']['checkpoint']['sha256'],
                          'order_sha': array_sha(order)})
            curve = producer['curve']
            check(tag + ':600_recorded_updates', len(curve) == 600 and [row['new_update'] for row in curve] == list(range(1, 601)))
            curve_values = np.asarray([[row[key] for key in ('objective_loss', 'actual_Y_loss', 'cached_RAW_teacher_loss', 'gradient_norm_before_clip')]
                                       for row in curve], np.float64)
            check(tag + ':finite_curve', bool(np.isfinite(curve_values).all()) and bool((curve_values[:, 3] >= 0).all()))
            wanted = objective_value(curve_values[:, 1], curve_values[:, 2], objective)
            check(tag + ':curve_objective_arithmetic', bool(np.allclose(curve_values[:, 0], wanted, atol=2e-6, rtol=2e-6)))
            check(tag + ':five_gradient_diagnostics', [row['new_update'] for row in producer['gradient_diagnostics']] == [0, 99, 199, 399, 599])
            for row in producer['gradient_diagnostics']:
                check(tag + ':gradient_record:' + str(row['new_update']), row['frozen_F_input_gradient_finite']
                      and np.isfinite(row['frozen_F_input_gradient_norm'])
                      and all(item['finite'] and np.isfinite(item['norm']) for item in row['E_D_gradients'].values()))
            check(tag + ':learning_proof', producer['learning_executed']['full_learning_proof_pass']
                  and producer['learning_executed']['optimizer_updates'] == 600
                  and producer['learning_executed']['E_updated'] and producer['learning_executed']['D_updated']
                  and producer['learning_executed']['frozen_F_input_gradient_nonzero_diagnostic_count'] > 0)
            check(tag + ':five_checkpoints', [row['new_update'] for row in producer['checkpoints']] == list(CHECKPOINTS))
            states, tune_values, cp_rows = {}, [], []
            for record in producer['checkpoints']:
                update = record['new_update']
                payload = checked_cp(record['checkpoint'])
                check(tag + ':checkpoint_metadata:' + str(update), payload['new_update'] == update and payload['seed'] == seed
                      and payload['objective'] == objective and payload['new_joint_comparison'])
                states[update] = payload['codec']
                if update == 0:
                    check(tag + ':checkpoint0_equals_original', state_hash(payload['codec']) == state_hash(initial))
                predictions = checked_npz(record['FIT_TUNE_predictions'])
                recomputed = {}
                for split in ('fit', 'tune'):
                    values = inputs[split]
                    for suffix, name in [('target', 'target'), ('observed', 'observed'), ('origins', 'origins'), ('RAW_teacher', 'raw_forecast')]:
                        exact(tag + ':' + str(update) + ':' + split + ':' + suffix, predictions[split + '_' + suffix], values[name])
                    exact(tag + ':' + str(update) + ':' + split + ':mean', predictions['mean'], values['mean'])
                    exact(tag + ':' + str(update) + ':' + split + ':std', predictions['std'], values['std'])
                    if update == 0:
                        check(tag + ':' + split + ':initial_original_MAIN_parity', bool(np.allclose(
                            predictions[split + '_prediction'], original_main[split + '_prediction'], atol=1e-5, rtol=2e-6)))
                    metrics = recompute_scores(predictions[split + '_prediction'], values['target'], values['raw_forecast'], values['observed'], values['std'])
                    metric_check(tag + ':' + str(update) + ':' + split + ':scores', metrics, record['scores'][split])
                    recomputed[split] = metrics
                actual_tune = recomputed['tune']['normalized_actual_Y']['mse']
                close(tag + ':' + str(update) + ':selection_metric_actual_Y', actual_tune, record['tune_actual_Y_mse'])
                tune_values.append(actual_tune)
                cp_rows.append({'new_update': update, 'actual_Y_FIT_MSE': recomputed['fit']['normalized_actual_Y']['mse'],
                                'RAW_teacher_FIT_MSE': recomputed['fit']['normalized_RAW_teacher']['mse'],
                                'objective_FIT_loss': objective_value(recomputed['fit']['normalized_actual_Y']['mse'], recomputed['fit']['normalized_RAW_teacher']['mse'], objective),
                                'actual_Y_TUNE_MSE': actual_tune})
                result['checkpoints_recomputed'] += 1
            check(tag + ':final_u600_hash', state_hash(states[600]) == producer['final_u600_codec_hash'])
            check(tag + ':fixed_basis_not_updated', torch.equal(initial['fixed_basis'], states[600]['fixed_basis']))
            actual_changes = {'E': 0, 'D': 0}
            for name, value in states[600].items():
                if name == 'fixed_basis':
                    continue
                group_name = 'D' if name.startswith('codec.decode.') or name.startswith('residual.') else 'E'
                changed = not torch.equal(value, initial[name])
                actual_changes[group_name] += int(changed)
                declared = next(row for row in producer['new_update_parameter_proof'][group_name]['parameters'] if row['name'] == name)
                delta = value.numpy().astype(np.float64) - initial[name].numpy().astype(np.float64)
                check(tag + ':parameter_change:' + name, declared['changed'] == changed)
                close(tag + ':parameter_max:' + name, float(np.abs(delta).max()), declared['max_abs'])
                close(tag + ':parameter_L2:' + name, float(np.linalg.norm(delta)), declared['l2'])
            for group_name in ('E', 'D'):
                check(tag + ':E_D_actual_changed_count:' + group_name, actual_changes[group_name] > 0
                      and actual_changes[group_name] == producer['new_update_parameter_proof'][group_name]['changed_parameter_tensors'])
            selected = min(range(5), key=lambda index: tune_values[index])
            selected_update = CHECKPOINTS[selected]
            check(tag + ':actual_Y_TUNE_earliest_selection', producer['selected_index'] == selected and producer['selected_new_update'] == selected_update)
            check(tag + ':selected_hash', producer['selected_codec_hash'] == state_hash(states[selected_update])
                  == producer['selected_checkpoint']['state_sha_old_algorithm'])
            check(tag + ':selected0_label', producer['selected_is_initial_original_MAIN'] == (selected_update == 0))
            selected_scores = {}
            for split, record in producer['selected_inference'].items():
                selected_arrays = checked_npz(record['artifact'])
                values = inputs[split]
                for name in ('target', 'observed', 'origins', 'mean', 'std'):
                    exact(tag + ':selected:' + split + ':' + name, selected_arrays[name], values[name])
                exact(tag + ':selected:' + split + ':RAW_teacher', selected_arrays['original_RAW_forecast'], values['raw_forecast'])
                exact(tag + ':selected:' + split + ':original_MAIN', selected_arrays['original_MAIN_forecast'], original_main[split + '_prediction'])
                metrics = recompute_scores(selected_arrays['prediction'], values['target'], values['raw_forecast'], values['observed'], values['std'])
                metric_check(tag + ':selected:' + split + ':scores', metrics, record['scores'])
                selected_scores[split] = metrics
                result['selected_outputs_recomputed'] += 1
            result['conditions'].append({'dataset': dataset, 'seed': seed, 'objective': objective, 'tag': tag,
                'producer_sha256': sha(folder / 'RESULT.json'), 'selected_new_update': selected_update,
                'selected_is_original': selected_update == 0, 'checkpoint_losses': cp_rows, 'selected_scores': selected_scores,
                'original_MAIN_scores': producer['original_MAIN_scores'], 'original_RAW_scores': producer['original_RAW_scores'],
                'u600_changed_parameters': actual_changes, 'guard_cadence': producer['guard_cadence'],
                'FIT_target_finite_contract': finite_contract,
                'masked_NaN_fix_empirical_failure_cause_evidence': False})
            limit.check()
        for identity, group in groups.items():
            check(str(identity) + ':three_objectives_same_initial_CP_order', {row['objective'] for row in group} == {'Y', 'RAW', 'MIXED'}
                  and len({row['initial'] for row in group}) == len({row['original_CP_sha'] for row in group}) == len({row['order_sha'] for row in group}) == 1)
        check('all12_60_36_coverage', len(result['conditions']) == 12 and result['checkpoints_recomputed'] == 60 and result['selected_outputs_recomputed'] == 36)
        result['GPU_initialized'] = torch.cuda.is_initialized()
        check('no_CUDA_initialized', not result['GPU_initialized'])
        result['status'] = 'PASS_WITHOUT_F_OR_PER_UPDATE_FORWARD_REPLAY' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['wall_seconds_before_terminal'] = time.perf_counter() - started
        if limit is not None:
            receipt = limit.close('COMPLETE' if result['status'].startswith('PASS') else 'FAILED_PRESERVED', result.get('failure'))
            result['shared_resource_receipt'] = receipt
            if receipt.get('terminal_status', 'COMPLETE') != 'COMPLETE':
                result.update(status=receipt['terminal_status'], terminal_error=receipt.get('terminal_error'))
        result['audit_source_sha256'] = sha(__file__)
        with OUT.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        lines = ['# Joint E/D 목적 비교의 독립 CPU 검산', '',
                 f"상태: {result['status']}; 검사 {result['checks']}개, 실패 {result['failed_count']}개.", '',
                 '저장60checkpoint·FIT/TUNE 예측과36selected 출력을 재채점하고, 원 CP/동일 order·FIT teacher mask·실제 Y TUNE 선택·u600 E/D 변화·selected hash를 검산했다.',
                 '600curve 기록은 coverage·유한성·목적 손실 조합을 확인했다. 매 update 출력/중간 CP가 없어 그 값 전체를 독립 재-forward했다고 쓰지 않는다. F도 새로 실행하지 않았다.',
                 'RAW 목적도 Y로 학습·선택된 초기 CP와 실제 Y TUNE 선택을 사용한다. 무감독/label-free 또는 독립 holdout 연구가 아니다.', '',
                 'FIT 관측/비관측 finite 여부·비관측 NaN 수는 실제 저장 teacher contract로 검산했다. masked-NaN backward 보정은 코드 견고성 확보이며, 실제 NaN이 없는 조건의 예측 부진이나 local cap 종료 원인으로 주장하지 않는다. GPU process wall에는 CPU 자원 검사·입출력도 포함되며 GPU kernel 학습 시간과 같지 않다.', '',
                 '```text', 'dataset              seed    objective  select     MSE          MAE']
        for row in result['conditions']:
            split = 'val' if row['dataset'] == 'robin' else 'eval'
            metric = row['selected_scores'][split]['normalized_actual_Y']
            lines.append(f"{row['dataset']:<21}{row['seed']:<8}{row['objective']:<11}{row['selected_new_update']:<11}{metric['mse']:<13.8f}{metric['mae']:.8f}")
        lines.extend(['```', '', 'select0은 원 MAIN 유지다. 실제600학습 실행과 새 선택 모델의 성능 개선은 다른 주장으로 구분한다. 세 목적을 VAL/EVAL 결과로 사후 선별하지 않는다.', '',
                      '[원수치·검사·출처](JOINT_ARRAY_AUDIT_v1.json) · [감사 코드](audit_joint_actual_v1.py)', ''])
        with MD.open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'failed_count': result['failed_count'],
                       'failure': result.get('failure'), 'output': str(OUT)}, ensure_ascii=False))
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
