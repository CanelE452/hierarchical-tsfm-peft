"""Bounded TRAIN/VAL audit of fixed-Q removal; never fits a prediction model."""
import argparse
import gc
import hashlib
import inspect
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch

from runtime_v12 import HERE, OLD, Job, artifact, import_file, read_json, save_json


DATASETS = ('robin', 'jena', 'hog')
SEEDS = (92601, 92602)
TOLERANCE = {'gradient_atol': 1e-6, 'gradient_rtol': 1e-4,
             'relative_norm_floor': 1e-12, 'gram_maxabs': 1e-5,
             'meaning': 'Fixed FP32 diagnostic tolerance, not proof of trajectory equivalence'}


def old_api():
    runtime = import_file(OLD / 'runtime_v11.py', 'v12_equivalence_old_runtime')
    model = import_file(OLD / 'model_v11.py', 'v12_equivalence_old_model')
    return runtime, model


def verified(path, expected):
    receipt = artifact(Path(path))
    if receipt['sha256'] != expected:
        raise RuntimeError('Source bytes changed: ' + str(path))
    return receipt


def array_hash(value):
    return hashlib.sha256(np.asarray(value).astype('<i8').tobytes()).hexdigest()


def mask_summary(data, split):
    origins = np.asarray(data[split + '_origins'], dtype=np.int64)
    finite = np.asarray(data['finite'], dtype=bool)
    prefix = np.vstack((np.zeros((1, finite.shape[1]), dtype=np.int64),
                        np.cumsum(finite, axis=0, dtype=np.int64)))
    counts = prefix[origins + 48] - prefix[origins]
    complete = (counts == 48).all(axis=1)
    totals = counts.sum(axis=0)
    result = {
        'origins': len(origins), 'origin_sha256': array_hash(origins),
        'target_occurrences': int(len(origins) * 48 * finite.shape[1]),
        'observed_target_occurrences': int(totals.sum()),
        'missing_target_occurrences': int(len(origins) * 48 * finite.shape[1] - totals.sum()),
        'channel_observed_counts': totals.tolist(),
        'fully_observed_origins': int(complete.sum()),
        'origins_with_missing_targets': int((~complete).sum()),
        'all_targets_observed': bool(complete.all()),
        'all_channel_counts_equal': bool(np.all(totals == totals[0])),
        'scope': 'All existing split origin targets, including repeated overlapping targets; not independent samples',
    }
    return result, origins[complete], origins[~complete]


def batches_for(data):
    train, complete, missing = mask_summary(data, 'train')
    val, _, _ = mask_summary(data, 'val')
    batches, absent = {}, {}
    if len(complete) >= 4:
        batches['complete'] = complete[:4].tolist()
    else:
        absent['complete'] = 'Fewer than four fully observed TRAIN origins; no artificial mask substitution'
    if len(missing):
        chosen = missing[:4].tolist()
        chosen += [int(o) for o in data['train_origins'] if int(o) not in chosen][:4 - len(chosen)]
        if len(chosen) != 4:
            raise RuntimeError('Approved batch4 cannot be formed')
        batches['missing'] = sorted(chosen)
    else:
        absent['missing'] = 'No missing target among the full TRAIN origin population'
    return {'train': train, 'val': val, 'paired_batches': batches, 'unavailable_batches': absent,
            'batch_selection': 'Chronological first four complete / first four missing origins; fill with earliest remaining origins if needed',
            'test_accessed': False}


def scheduler_replay(values, initial_lr):
    """The installed min/rel plateau rule with the exact v11 defaults and call order."""
    best, bad, lr = math.inf, 0, float(initial_lr)
    before, after, improvement = [], [], []
    for epoch, value in enumerate(values):
        before.append(lr)
        better = None
        if epoch:
            better = float(value) < best * (1 - 1e-4)
            if better:
                best, bad = float(value), 0
            else:
                bad += 1
            if bad > 2:
                reduced = max(lr * .5, 0.)
                if lr - reduced > 1e-8:
                    lr = reduced
                bad = 0
        after.append(lr)
        improvement.append(better)
    return {'lr_before_epoch': before, 'lr_after_val': after,
            'scheduler_improvement': improvement,
            'best_checkpoint_epoch': int(np.argmin(values)),
            'optimizer_updates': 0}


def scheduler_rows(offsets=None):
    rows = []
    protocol = read_json(OLD / 'protocol.json')
    specs = protocol['neural_fits']
    if protocol['fits'] != specs:
        raise RuntimeError('The v11 neural fit list and execution alias disagree')
    for spec in specs:
        if spec['family'] != 'a_p_lora_qfixed':
            continue
        path = OLD / 'runs' / spec['id'] / 'curve.json'
        curve = read_json(path)
        if [r['epoch'] for r in curve] != list(range(len(curve))):
            raise RuntimeError('Unexpected curve epoch order')
        values = [r['val_mse'] for r in curve]
        replay = scheduler_replay(values, spec['lr'])
        actual_lr = [r['lr'] for r in curve]
        if not np.allclose(actual_lr, replay['lr_before_epoch'], atol=1e-15, rtol=0):
            raise RuntimeError('Scalar replay does not match the actual saved scheduler history')
        shift = float(np.median(values))
        illustrative = scheduler_replay([v + shift for v in values], spec['lr'])
        row = {'id': spec['id'], 'dataset': spec['dataset'], 'seed': spec['seed'],
               'curve': artifact(path), 'val_sequence': values, 'saved_lr': actual_lr,
               'actual_replay': replay, 'matches_actual_saved_lr': True,
               'illustrative_positive_offset': shift, 'illustrative_replay': illustrative,
               'illustrative_lr_differs': illustrative['lr_after_val'] != replay['lr_after_val'],
               'illustration_is_not_actual_no_q_training': True}
        if offsets is not None:
            item = offsets[spec['dataset']][str(spec['seed'])]
            row['fixed_parent_val_offset'] = item
            if item['conditional_algebra_applicable']:
                main_values = [v - item['l_full_minus_l_main_constant'] for v in values]
                shifted = scheduler_replay(main_values, spec['lr'])
                row.update(conditional_main_loss_sequence=main_values,
                           conditional_same_trajectory_main_replay=shifted,
                           conditional_lr_differs=shifted['lr_after_val'] != replay['lr_after_val'],
                           interpretation='Counterfactual metrics on the saved A trajectory under exact P/Q orthogonality; not an independently optimized no-Q trajectory')
        rows.append(row)
    if len(rows) != 12:
        raise RuntimeError('Expected all 12 existing A LR/seed runs')
    return rows


def synthetic_math():
    u = torch.tensor([1., 1.], dtype=torch.float64) / math.sqrt(2)
    q = torch.tensor([.7, -.7], dtype=torch.float64)
    y = torch.tensor([.1, -.3], dtype=torch.float64)
    rows = {}
    for name, weights in [('complete_uniform', torch.tensor([.5, .5], dtype=torch.float64)),
                          ('masked_first_channel', torch.tensor([1., 0.], dtype=torch.float64))]:
        theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        b = theta * u
        plus = torch.sum(weights * (b + q - y).square())
        main = torch.sum(weights * (b - y).square())
        gp, = torch.autograd.grad(plus, theta, retain_graph=True)
        gm, = torch.autograd.grad(main, theta)
        expected = float(2 * torch.sum(u * weights * q))
        difference = float(gp - gm)
        if abs(difference - expected) > 1e-14:
            raise RuntimeError('FP64 linear gradient identity failed')
        rows[name] = {'gradient_with_q': float(gp), 'gradient_main': float(gm),
                      'difference': difference, 'formula_2_Jt_Wq': expected,
                      'weights': weights.tolist(), 'optimizer_updates': 0}
    if abs(rows['complete_uniform']['difference']) > 1e-14 or abs(rows['masked_first_channel']['difference']) < .1:
        raise RuntimeError('Synthetic complete/masked distinction was not observed')
    toy = [1., 1., .99985, .99985, .99985, .99985]
    original, shifted = scheduler_replay(toy, .001), scheduler_replay([v + 1 for v in toy], .001)
    if original['lr_after_val'] == shifted['lr_after_val']:
        raise RuntimeError('The predeclared scheduler counterexample failed')
    return {'dtype': 'float64', 'linear_examples': rows,
            'scheduler_counterexample': {'sequence': toy, 'constant_offset': 1.,
                                         'original': original, 'shifted': shifted},
            'optimizer_objects_created': 0, 'optimizer_updates': 0}


def cpu_stage(job):
    runtime, _ = old_api()
    output = HERE / 'equivalence_mask_policy01.json'
    if output.exists():
        raise RuntimeError('CPU audit exists; preserve it before any documented repair')
    data_rows = {}
    for dataset in DATASETS:
        data = runtime.load_data(dataset, include_test=False)
        item = batches_for(data)
        u = np.asarray(data['basis'], dtype=np.float64)
        item.update(trainval=artifact(data['_path']),
                    gram_maxabs=float(np.max(np.abs(u.T @ u - np.eye(u.shape[1])))),
                    basis_shape=list(u.shape))
        data_rows[dataset] = item
        job.heartbeat(dataset + ' full TRAIN/VAL mask contract inspected')
    source = inspect.getsource(torch.optim.lr_scheduler.ReduceLROnPlateau)
    payload = {'datasets': data_rows, 'tolerance': TOLERANCE,
               'synthetic': synthetic_math(), 'scheduler_rows': scheduler_rows(),
               'installed_scheduler_source': artifact(inspect.getsourcefile(torch.optim.lr_scheduler.ReduceLROnPlateau)),
               'scheduler_class_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'scheduler_contract': {'mode': 'min', 'factor': .5, 'patience': 2, 'threshold': 1e-4,
                                      'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0, 'eps': 1e-8,
                                      'step0': 'checkpoint candidate, no scheduler.step call'},
               'optimizer_updates': 0, 'test_accessed': False}
    save_json(output, payload)


def gradient_stats(parameters, with_q, main):
    all_plus, all_main, per_parameter = [], [], {}
    for (name, parameter), gp, gm in zip(parameters, with_q, main):
        plus = np.zeros(parameter.numel(), dtype=np.float64) if gp is None else gp.detach().cpu().double().numpy().ravel()
        baseline = np.zeros(parameter.numel(), dtype=np.float64) if gm is None else gm.detach().cpu().double().numpy().ravel()
        if not np.isfinite(plus).all() or not np.isfinite(baseline).all():
            raise RuntimeError('Nonfinite paired gradient')
        difference = np.abs(plus - baseline)
        tol = TOLERANCE['gradient_atol'] + TOLERANCE['gradient_rtol'] * np.maximum(np.abs(plus), np.abs(baseline))
        per_parameter[name] = {'numel': parameter.numel(), 'with_q_none': gp is None, 'main_none': gm is None,
                               'with_q_nonzero': int(np.count_nonzero(plus)), 'main_nonzero': int(np.count_nonzero(baseline)),
                               'difference_nonzero': int(np.count_nonzero(difference)),
                               'max_absolute_difference': float(difference.max()),
                               'elements_outside_diagnostic_tolerance': int(np.count_nonzero(difference > tol))}
        all_plus.append(plus)
        all_main.append(baseline)
    plus, baseline = np.concatenate(all_plus), np.concatenate(all_main)
    difference = plus - baseline
    norm_p, norm_m, norm_d = (float(np.linalg.norm(v)) for v in (plus, baseline, difference))
    return {'with_q_l2': norm_p, 'main_l2': norm_m, 'difference_l2': norm_d,
            'stabilized_relative_difference_l2': norm_d / max(norm_p, norm_m, TOLERANCE['relative_norm_floor']),
            'max_absolute_difference': float(np.abs(difference).max()),
            'with_q_nonzero': int(np.count_nonzero(plus)), 'main_nonzero': int(np.count_nonzero(baseline)),
            'with_q_zero': int(np.count_nonzero(plus == 0)), 'main_zero': int(np.count_nonzero(baseline == 0)),
            'difference_nonzero': int(np.count_nonzero(difference)), 'registered_trainable_scalars': len(plus),
            'elements_outside_diagnostic_tolerance': sum(v['elements_outside_diagnostic_tolerance'] for v in per_parameter.values()),
            'per_parameter': per_parameter}


def backward_pair(model, data, origins, runtime):
    x, y, mask = runtime.tensors(data, origins, device='cuda')
    model.train()
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n for n, _ in parameters):
        raise RuntimeError('Only the original A LoRA parameters may require gradients')
    main, q = model.components(x)
    full_loss = runtime.masked_macro_loss(main + q, y, mask)
    main_loss = runtime.masked_macro_loss(main, y, mask)
    values = [p for _, p in parameters]
    gp = torch.autograd.grad(full_loss, values, retain_graph=True, allow_unused=True)
    gm = torch.autograd.grad(main_loss, values, allow_unused=True)
    summary = gradient_stats(parameters, gp, gm)
    with torch.no_grad():
        u = model.basis0
        count = mask.bool().sum((0, 1))
        valid = count > 0
        weights = mask.to(torch.float64) / count.clamp_min(1).double() / valid.sum().double()
        weighted_projection = (weights * q.double()) @ u.double()
        summary.update(full_loss=float(full_loss), main_loss=float(main_loss),
                       loss_difference=float(full_loss - main_loss),
                       fixed_q_requires_grad=bool(q.requires_grad),
                       channel_observed_counts=count.cpu().tolist(),
                       all_targets_observed=bool(mask.all()),
                       q_projected_on_p_maxabs=float((q @ u).abs().max()),
                       main_outside_p_maxabs=float((main - (main @ u) @ u.T).abs().max()),
                       weighted_q_projected_on_p_maxabs=float(weighted_projection.abs().max()),
                       gram_maxabs=float((u.T @ u - torch.eye(u.shape[1], device=u.device)).abs().max()),
                       backbone_training=bool(model.backbone.training),
                       precision=str(main.dtype), optimizer_updates=0)
    if q.requires_grad or model.backbone.training or not torch.isfinite(full_loss) or not torch.isfinite(main_loss):
        raise RuntimeError('The restored A no longer follows its fixed-Q deterministic-forward contract')
    return summary


@torch.no_grad()
def fixed_val_offset(model, data, masks, runtime, job):
    if not masks['val']['all_targets_observed']:
        return {'conditional_algebra_applicable': False,
                'reason': 'Actual VAL contains missing targets; masked full/main difference need not be a phi-independent constant',
                'test_accessed': False}
    total, count, leakage = 0., 0, 0.
    model.eval()
    for start in range(0, len(data['val_origins']), 4):
        job.check_limits()
        x, y, mask = runtime.tensors(data, data['val_origins'][start:start + 4], device='cuda')
        if not bool(mask.all()):
            raise RuntimeError('VAL mask changed after the contract audit')
        q = model.q_prediction(x).double()
        u = model.basis0.double()
        yq = y.double() - (y.double() @ u) @ u.T
        total += float(((q - yq).square() - yq.square()).sum())
        count += q.numel()
        leakage = max(leakage, float((q @ u).abs().max()))
    return {'conditional_algebra_applicable': True, 'l_full_minus_l_main_constant': total / count,
            'definition': 'mean(||q-YQ||^2 - ||YQ||^2) in fixed standardized coordinates',
            'actual_fixed_parent_and_actual_val_targets': True, 'q_in_p_maxabs': leakage,
            'precision': 'FP32 parent Q forward then FP64 algebra',
            'finite_precision_note': 'Exact constant identity assumes exact orthogonality; measured FP32 P/Q leakage is retained',
            'test_accessed': False}


def gpu_stage(job):
    runtime, module = old_api()
    masks = read_json(HERE / 'equivalence_mask_policy01.json')['datasets']
    selected = read_json(OLD / 'selected.json')['selected']
    journal_path = HERE / 'equivalence_paired_backward_ledger.json'
    journal = read_json(journal_path) if journal_path.exists() else {'maximum_pairs': 24, 'pairs': [], 'optimizer_updates': 0}
    offset_path = HERE / 'equivalence_fixed_val_offsets01.json'
    offsets = read_json(offset_path) if offset_path.exists() else {d: {} for d in DATASETS}
    for dataset in DATASETS:
        data = runtime.load_data(dataset, include_test=False)
        selected_a = selected[dataset]['a_p_lora_qfixed']
        for index, seed in enumerate(SEEDS):
            run = read_json(OLD / 'runs' / selected_a['run_ids'][index] / 'result.json')
            if run['spec']['seed'] != seed or run['spec']['dataset'] != dataset:
                raise RuntimeError('Selected parent/seed order mismatch')
            best_receipt = verified(run['checkpoint'], selected_a['checkpoint_sha256'][index])
            initial_receipt = verified(run['initial_checkpoint'], run['initial_checkpoint_sha256'])
            initial = torch.load(initial_receipt['path'], map_location='cpu', weights_only=False)
            best = torch.load(best_receipt['path'], map_location='cpu', weights_only=False)
            if initial['sources'] != best['sources'] or initial['spec'] != best['spec']:
                raise RuntimeError('Initial and selected A do not share parent/spec')
            model = module.restore_model(best_receipt['path'], device='cuda')
            frozen_hash = model.frozen_state_digest()
            for state_name, state, receipt in [('initial', initial['state'], initial_receipt),
                                                ('selected', best['state'], best_receipt)]:
                model.restore_adapter(state)
                if model.frozen_state_digest() != frozen_hash:
                    raise RuntimeError('State swap changed frozen LEVEL, basis, or original backbone')
                for batch_type, origins in masks[dataset]['paired_batches'].items():
                    pair_id = f'{dataset}_{seed}_{state_name}_{batch_type}'
                    previous = [p for p in journal['pairs'] if p['id'] == pair_id]
                    if previous:
                        if len(previous) == 1 and previous[0]['status'] == 'complete':
                            continue
                        raise RuntimeError('An incomplete/failed pair is preserved; a documented bounded repair is required')
                    if len(journal['pairs']) >= 24:
                        raise RuntimeError('The 24 paired-backward reservation cap is exhausted')
                    record = {'id': pair_id, 'dataset': dataset, 'seed': seed, 'state': state_name,
                              'batch_type': batch_type, 'origins': origins, 'origin_sha256': array_hash(origins),
                              'checkpoint': receipt, 'status': 'reserved', 'optimizer_updates': 0,
                              'two_backward_calls_one_shared_forward': True}
                    journal['pairs'].append(record)
                    save_json(journal_path, journal)
                    job.check_limits()
                    job.heartbeat(pair_id + ' paired backward, optimizer updates 0')
                    try:
                        record['result'] = backward_pair(model, data, origins, runtime)
                        record['status'] = 'complete'
                    except BaseException:
                        record.update(status='failed', error=traceback.format_exc())
                        save_json(journal_path, journal)
                        raise
                    save_json(journal_path, journal)
                    print(json.dumps({'pair': pair_id, 'max_gradient_difference': record['result']['max_absolute_difference'],
                                      'relative_l2': record['result']['stabilized_relative_difference_l2']}), flush=True)
            if str(seed) not in offsets[dataset]:
                offsets[dataset][str(seed)] = fixed_val_offset(model, data, masks[dataset], runtime, job)
                save_json(offset_path, offsets)
            if frozen_hash != model.frozen_state_digest():
                raise RuntimeError('Frozen state changed during the no-update audit')
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('autograd.grad unexpectedly accumulated parameter gradients')
            del model, initial, best
            gc.collect()
            torch.cuda.empty_cache()


def report_stage():
    masks = read_json(HERE / 'equivalence_mask_policy01.json')
    pairs = read_json(HERE / 'equivalence_paired_backward_ledger.json')
    offsets = read_json(HERE / 'equivalence_fixed_val_offsets01.json')
    if any(p['status'] != 'complete' for p in pairs['pairs']):
        raise RuntimeError('Not every reserved pair completed')
    expected_pairs = sum(len(masks['datasets'][d]['paired_batches']) * 4 for d in DATASETS)
    if len(pairs['pairs']) != expected_pairs or expected_pairs > 24:
        raise RuntimeError('The fixed audit matrix is incomplete')
    replay = scheduler_rows(offsets)
    save_json(HERE / 'equivalence_scheduler_replay01.json', {'rows': replay, 'optimizer_updates': 0, 'test_accessed': False})
    decisions = {}
    for dataset in DATASETS:
        ds_pairs = [p for p in pairs['pairs'] if p['dataset'] == dataset]
        actual_masks = masks['datasets'][dataset]
        decisions[dataset] = {
            'branch': 'E1_same_checkpoint_claim_only', 'no_q_fit': False, 'additional_fit_count': 0,
            'reason': 'The fixed-A Q contribution claim compares outputs at the same trained P. Independently optimizing no-Q answers a different policy question; this report does not claim to beat that comparator or claim it is redundant.',
            'actual_train_complete': actual_masks['train']['all_targets_observed'],
            'actual_val_complete': actual_masks['val']['all_targets_observed'],
            'limited_pairs': len(ds_pairs),
            'any_pair_outside_diagnostic_tolerance': any(p['result']['elements_outside_diagnostic_tolerance'] for p in ds_pairs),
            'maximum_gradient_difference': max(p['result']['max_absolute_difference'] for p in ds_pairs),
            'full_training_policy_equivalence': 'not_established',
            'independently_optimized_no_q_comparator': 'not_measured; needed before any claim against that independently selected training recipe',
            'test_error_used_for_decision': False,
        }
    save_json(HERE / 'NO_Q_RUN_DECISION.json', {
        'scope': 'Existing Robin/Jena/Hog only; confirmation-unit decisions require their own TRAIN/VAL contract',
        'decision_basis': 'Predeclared same-checkpoint claim scope plus TRAIN/VAL mask, gradient, and scheduler audit; no TEST score inputs',
        'datasets': decisions, 'new_confirmation_units': 'not_decided: approved exact data contract unavailable',
        'optimizer_updates': 0, 'fit_count': 0})
    summary = {
        'paired_backward_count': len(pairs['pairs']), 'maximum_paired_backward_count': 24,
        'optimizer_updates': 0, 'test_accessed': False, 'tolerance': TOLERANCE,
        'inputs': {name: artifact(HERE / name) for name in
                   ('equivalence_mask_policy01.json', 'equivalence_paired_backward_ledger.json',
                    'equivalence_fixed_val_offsets01.json', 'equivalence_scheduler_replay01.json', 'NO_Q_RUN_DECISION.json')},
        'claim': 'Conditional P-gradient identity is distinct from finite-mask behavior and independently selected training-policy equivalence',
        'full_training_policy_equivalence_established': False,
    }
    save_json(HERE / 'equivalence_audit01.json', summary)
    lines = ['# Fixed-Q learning-equivalence audit', '',
             '[확인] 기존 TRAIN/VAL만 사용했다. TEST 점수나 mask는 분기 입력이 아니다. 실제 optimizer update와 신규 fit은 0이다.', '',
             '완전관측·동일가중 squared loss와 정확한 고정 직교 P/Q에서는 Q 유무 손실 차이가 LoRA에 대한 상수다. 실제 masked macro에서는 gradient 차이가 2 J_b^T W q이며, 채널별 관측수와 mask가 이 직교성을 깨뜨릴 수 있다.', '',
             f"제한된 실제 paired backward {len(pairs['pairs'])}개를 수행했다. 최대 24개이며 미존재 결측 batch를 합성해 실자료처럼 보충하지 않았다. 각 pair는 동일 forward graph의 with-Q/main 손실을 두 번 미분했으며 가중치를 갱신하지 않았다.", '',
             '허용오차는 FP32 elementwise atol 1e-6 + rtol 1e-4 × max(|g_full|, |g_main|)로 실행 전에 고정했다. 작은 차이나 통과는 전체 학습 궤적 동등성 증명이 아니다.', '']
    for dataset in DATASETS:
        d = decisions[dataset]
        n = masks['datasets'][dataset]
        lines += [f"- {dataset}: TRAIN 결측 target occurrence {n['train']['missing_target_occurrences']}, VAL {n['val']['missing_target_occurrences']}; paired backward {d['limited_pairs']}; 최대 gradient 절대차 {d['maximum_gradient_difference']:.8g}. 동등성 판정은 전체 궤적에 확대하지 않는다."]
    lines += ['', '설치된 ReduceLROnPlateau min/rel 규칙과 v11 호출 순서를 scalar replay하여 기존 12개 A curve의 실제 LR과 대조했다. FP64 작은 선형 예제는 완전관측 일치와 비등방 mask 반례를 구분한다. 상대 threshold scheduler에는 상수 offset 반례가 존재한다.', '',
              '완전관측 VAL에서는 동일 부모의 Q forward와 실제 VAL 정답으로 계산한 조건부 상수항도 기록했다. 이 항으로 기존 A loss sequence를 이동한 replay는 같은 A 궤적에서의 반사실적 scheduler 점검이다. 독립 no-Q 적합의 실제 궤적·최선 epoch·성능을 재구성한 결과가 아니다. 결측 VAL에서는 이 상수 이동을 실제 no-Q sequence로 제공하지 않는다.', '',
              '기존 세 자료는 E1의 **same-checkpoint 출력 기여로 주장을 제한하는 분기**를 선택한다. 별도 NO_Q_DIRECT_TRAIN은 이번 좁은 출력 기여 판단에 필수적이지 않아 적합하지 않는다. 이는 수학적 중복 판정이 아니다. 독립 no-Q 학습·선택 정책을 이겼다는 주장과 전체 정책 동등성은 미확인으로 남긴다. 새 확인 단위의 TRAIN/VAL 계약은 별도로 검토해야 한다.', '',
              '수치 원본: `equivalence_audit01.json`, `equivalence_mask_policy01.json`, `equivalence_paired_backward_ledger.json`, `equivalence_fixed_val_offsets01.json`, `equivalence_scheduler_replay01.json`, `NO_Q_RUN_DECISION.json`. 자체 검사이며 독립 재현이 아니다.', '']
    (HERE / 'TRAINING_EQUIVALENCE.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('cpu', 'gpu', 'report'), required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    category = 'gpu' if args.stage == 'gpu' else 'cpu_analysis'
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'TRAIN/VAL fixed-Q learning-equivalence audit',
                       'stage': args.stage, 'optimizer_updates': 0, 'test_access': False}) as job:
        if args.stage == 'cpu':
            cpu_stage(job)
        elif args.stage == 'gpu':
            gpu_stage(job)
        else:
            report_stage()


if __name__ == '__main__':
    main()
