"""CPU-only numerical investigation of algebraically cancelled decoder biases.

Saved FIT residuals and actual saved decoder weights are used. The forecast and
anchor K-dimensional latents are synthetic; this is not a Bolt forward replay.
All writes are exclusive and restricted to the new diagnostic directory.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CASE = HERE / 'run_dev_v1/jena/NO_ALIGN_w0_s92601.json'
BIAS_KEYS = ('codec.codec.decode.bias', 'codec.residual.2.bias')
KEYS = {
    'linear.weight': 'codec.codec.decode.weight', 'linear.bias': BIAS_KEYS[0],
    'residual.0.weight': 'codec.residual.0.weight', 'residual.0.bias': 'codec.residual.0.bias',
    'residual.2.weight': 'codec.residual.2.weight', 'residual.2.bias': BIAS_KEYS[1],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def checked(artifact):
    path = ROOT / artifact['path']
    if sha(path) != artifact['sha256']: raise ValueError('SOURCE_SHA_MISMATCH: ' + str(path))
    return path


def artifact(path):
    return {'path': Path(path).resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def save_npz(path, **arrays):
    with Path(path).open('xb') as handle: np.savez_compressed(handle, **arrays)
    return artifact(path)


class Decoder(nn.Module):
    def __init__(self, state, dtype):
        super().__init__()
        channels, k = state['codec.codec.decode.weight'].shape
        hidden = state['codec.residual.0.weight'].shape[0]
        self.linear = nn.Linear(k, channels, dtype=dtype)
        self.residual = nn.Sequential(nn.Linear(k, hidden, dtype=dtype), nn.GELU(), nn.Linear(hidden, channels, dtype=dtype))
        self.load_state_dict({name: state[key].detach().to(dtype).clone() for name, key in KEYS.items()})

    def forward(self, z):
        return self.linear(z) + self.residual(z)


def bias_free_difference(decoder, future, anchor, last):
    linear_delta = F.linear(future - anchor, decoder.linear.weight, None)
    first = decoder.residual[0]
    future_hidden = F.gelu(F.linear(future, first.weight, first.bias))
    anchor_hidden = F.gelu(F.linear(anchor, first.weight, first.bias))
    residual_delta = F.linear(future_hidden - anchor_hidden, decoder.residual[2].weight, None)
    return last + linear_delta + residual_delta


def macro_loss(prediction, target, observed):
    count = observed.sum((0, 1))
    desired = torch.where(observed, target, prediction.detach())
    error = (prediction - desired).square().sum((0, 1))
    valid = count > 0
    return (error / count.clamp_min(1))[valid].mean()


def optimizer_replay(decoder, checkpoint, restored):
    temporary = copy.deepcopy(decoder)
    named = dict(temporary.named_parameters())
    for name, parameter in decoder.named_parameters():
        named[name].grad = None if parameter.grad is None else parameter.grad.detach().clone()
    optimizer = torch.optim.AdamW(temporary.parameters(), lr=.001, weight_decay=0., eps=1e-8)
    if restored:
        # All registered parameter names in this frozen source end in weight/bias,
        # except the explicitly named scalar alpha. Buffers are excluded.
        names = [name for name in checkpoint['model'] if name.endswith(('.weight', '.bias')) or name == 'alpha_logit']
        identifiers = checkpoint['optimizer']['param_groups'][0]['params']
        if len(names) != len(identifiers): raise ValueError('SAVED_OPTIMIZER_PARAMETER_ORDER_COVERAGE')
        indexed = dict(zip(names, identifiers))
        for local_name, parameter in temporary.named_parameters():
            state = copy.deepcopy(checkpoint['optimizer']['state'].get(indexed[KEYS[local_name]], {}))
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    if state[key].shape != parameter.shape: raise ValueError('SAVED_OPTIMIZER_PARAMETER_SHAPE')
                    state[key] = state[key].to(dtype=parameter.dtype)
            optimizer.state[parameter] = state
    before = {name: parameter.detach().clone() for name, parameter in temporary.named_parameters()}
    first_moment = {name: optimizer.state[parameter].get('exp_avg', torch.zeros_like(parameter)).detach().clone()
                    for name, parameter in temporary.named_parameters()}
    optimizer.step()
    arrays, summary = {}, {}
    for name in ('linear.bias', 'residual.2.bias'):
        key = name.replace('.', '_')
        parameter = named[name]
        delta = parameter.detach() - before[name]
        arrays.update({key + '_before': before[name].numpy(), key + '_after': parameter.detach().numpy(),
                       key + '_step_delta': delta.numpy(), key + '_moment_before': first_moment[name].numpy(),
                       key + '_moment_after': optimizer.state[parameter].get('exp_avg', torch.zeros_like(parameter)).detach().numpy()})
        summary[name] = {'changed_scalar_count': int((delta != 0).sum()), 'max_abs_parameter_change': float(delta.abs().max()),
                         'first_moment_before_max_abs': float(first_moment[name].abs().max()),
                         'step_number_after': float(optimizer.state[parameter].get('step', torch.zeros(())))}
    return arrays, summary


def run_row(checkpoint, dtype, future_np, anchor_np, last_np, target_np, observed_np, saved_np, mode, form):
    decoder = Decoder(checkpoint['model'], dtype)
    future, anchor, last, target, saved = [torch.as_tensor(v, dtype=dtype) for v in (future_np, anchor_np, last_np, target_np, saved_np)]
    observed = torch.as_tensor(observed_np, dtype=torch.bool)
    prediction = last + decoder(future) - decoder(anchor) if form == 'original' else bias_free_difference(decoder, future, anchor, last)
    forward_value = prediction.detach().clone()
    if mode == 'saved_FIT_residual_Jacobian_replay':
        # Exact saved output values with the anchored expression's derivative.
        # This is a derivative diagnostic, not an actual E/F forward replay.
        prediction = saved + (prediction - prediction.detach())
    loss = macro_loss(prediction, target, observed)
    upstream = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    loss.backward()
    flat = upstream.reshape(-1, upstream.shape[-1]).sum(0)
    nested = upstream.sum(1).sum(0)
    reduction_delta = flat - nested
    arrays = {'prediction': prediction.detach().numpy(), 'synthetic_latent_forward_prediction': forward_value.numpy(),
              'upstream_prediction_gradient': upstream.detach().numpy(), 'flat_BH_reduction': flat.detach().numpy(),
              'nested_H_then_B_reduction': nested.detach().numpy(), 'reduction_difference': reduction_delta.detach().numpy()}
    summary = {'dtype': str(dtype), 'loss': float(loss.detach()), 'mode': mode, 'form': form,
               'flat_minus_nested_max_abs': float(reduction_delta.abs().max()),
               'nonzero_reduction_difference_channels': int((reduction_delta != 0).sum()), 'bias_gradients': {}, 'adamw_steps': {}}
    for name in ('linear.bias', 'residual.2.bias'):
        parameter = dict(decoder.named_parameters())[name]
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
        arrays[name.replace('.', '_') + '_gradient'] = gradient.numpy()
        summary['bias_gradients'][name] = {'gradient_is_none': parameter.grad is None, 'nonzero_scalar_count': int((gradient != 0).sum()),
            'max_abs': float(gradient.abs().max()), 'same_as_flat_minus_nested_exact': bool(torch.equal(gradient, reduction_delta)),
            'gap_from_flat_minus_nested_max_abs': float((gradient - reduction_delta).abs().max())}
    for restored in (False, True):
        update_arrays, updates = optimizer_replay(decoder, checkpoint, restored)
        label = 'restored_saved_moments' if restored else 'fresh_zero_moments'
        arrays.update({label + '_' + key: value for key, value in update_arrays.items()})
        summary['adamw_steps'][label] = updates
    summary['optimizer_scope'] = 'One disposable decoder-only AdamW step, lr=.001, wd=0, eps=1e-8, unclipped diagnostic gradients. Not a training continuation.'
    summary['restored_moments_note'] = 'Saved moments may change bias even with zero new gradient; FP64 casts saved FP32 parameters and moments.'
    return summary, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--tag', default='reproduction_v1')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started = time.perf_counter(); process = psutil.Process()
    folder = HERE / 'decision_checks_v1/decoder_bias_v1' / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    output = {'status': 'INCOMPLETE', 'rows': [], 'protected_TEST_opened': False, 'EVAL_opened': False,
              'GPU_executed': False, 'training_source_changed': False, 'checkpoint_modified': False,
              'scope': 'Numerical decoder/reduction reproduction using actual CP0/600 and actual saved FIT residuals, synthetic K latents; not a Bolt replay or forecast improvement experiment.'}
    try:
        case = read(args.case)
        if case['dataset'] != 'jena' or case['method'] != 'NO_ALIGN' or case['training']['updates'] != 600:
            raise ValueError('ONLY_COMPLETED_JENA_NO_ALIGN_600_ALLOWED')
        source_paths = [args.case] + [HERE / name for name in ('model.py', 'train.py', 'data.py', 'run.py', 'benchmark.py', 'probes.py', 'runtime.py')]
        checkpoints = {u: next(c for c in case['training']['checkpoints'] if c['update'] == u) for u in (0, 600)}
        checkpoint_paths = {u: checked(c) for u, c in checkpoints.items()}
        prediction_path = checked(case['predictions'])
        contract_path = ROOT / 'research/level_matched_learned_ed_v18_20261006/data_contract_jena.json'
        contract = read(contract_path)
        source_path = checked(contract['trainval'])
        source_paths += list(checkpoint_paths.values()) + [prediction_path, contract_path, source_path]
        initial_hashes = {str(path.resolve().relative_to(ROOT)): sha(path) for path in source_paths}
        states = {u: torch.load(path, map_location='cpu', weights_only=True) for u, path in checkpoint_paths.items()}
        for u, state in states.items():
            if state['update'] != u: raise ValueError('ACTUAL_CHECKPOINT_UPDATE_MISMATCH')
        with np.load(prediction_path, allow_pickle=False) as z:
            indices = np.asarray(case['training']['batch_indices'][0], np.int64)
            prediction = z['fit_prediction'][indices].copy(); target = z['fit_target'][indices].copy()
            observed = z['fit_observed'][indices].copy(); origins = z['fit_origins'][indices].copy()
            mean, std = z['mean'].copy(), z['std'].copy()
        with np.load(source_path, allow_pickle=False) as z:
            old_std = np.where(z['std'] > 1e-6, z['std'], 1.).astype(np.float32)
            physical = (z['x'].astype(np.float32) * old_std + z['mean'].astype(np.float32)).astype(np.float32)
            finite = z['finite'].astype(bool)
        raw = np.stack([physical[o-512:o] for o in origins])
        history_observed = np.stack([finite[o-512:o] for o in origins])
        last = np.empty((8, 1, len(mean)), np.float32)
        for b in range(8):
            filled = mean.copy()
            for t in range(512): filled = np.where(history_observed[b, t], raw[b, t], filled)
            last[b, 0] = (filled - mean) / std
        k = states[0]['model']['codec.codec.decode.weight'].shape[1]
        generator = np.random.default_rng(92601042)
        anchor = generator.normal(0., .7, (8, 1, k)).astype(np.float32)
        future = (anchor + generator.normal(0., .7, (8, 48, k)).astype(np.float32)).astype(np.float32)
        output.update(source_artifacts={'case': artifact(args.case), 'checkpoint_u0': checkpoints[0], 'checkpoint_u600': checkpoints[600],
                      'FIT_prediction_target': case['predictions'], 'legacy_TRAIN_DEVVAL': contract['trainval']},
                      actual_prediction_selected_update=case['training']['selected_update'], source_hashes=initial_hashes,
                      synthetic_latent_provenance={'generator': 'numpy default_rng', 'seed': 92601042, 'K': int(k),
                         'anchor': 'N(0,.7), shape8x1xK', 'future': 'anchor + N(0,.7), shape8x48xK', 'actual_E_or_F_latents': False},
                      actual_FIT_indices=indices.tolist(), actual_FIT_origins=origins.tolist(), mathematical_cancelled_bias_count=2*len(mean))
        output['actual_saved_bias_changes_u0_to_u600'] = {name: {
            'changed_scalar_count': int((states[0]['model'][name] != states[600]['model'][name]).sum()),
            'max_abs': float((states[600]['model'][name] - states[0]['model'][name]).abs().max())} for name in BIAS_KEYS}
        output['input_artifact'] = save_npz(folder / 'INPUTS.npz', actual_FIT_prediction=prediction, actual_FIT_target=target,
            actual_FIT_observed=observed, actual_FIT_origins=origins, actual_raw_history=raw,
            actual_history_observed=history_observed, actual_normalized_last=last, mean=mean, std=std,
            synthetic_future_latent=future, synthetic_anchor_latent=anchor)
        for update, state in states.items():
            for dtype in (torch.float32, torch.float64):
                for batch in (1, 8):
                    for mode in ('synthetic_latent_forward', 'saved_FIT_residual_Jacobian_replay'):
                        pair = []
                        for form in ('original', 'bias_free'):
                            summary, arrays = run_row(state, dtype, future[:batch], anchor[:batch], last[:batch], target[:batch],
                                                      observed[:batch], prediction[:batch], mode, form)
                            label = f'u{update}_{str(dtype).split(".")[-1]}_B{batch}_{mode}_{form}'
                            summary.update(update=update, batch=batch, arrays=save_npz(folder / (label + '.npz'), **arrays))
                            output['rows'].append(summary); pair.append((summary, arrays))
                        original, alternate = pair
                        output_gap = float(np.abs(original[1]['synthetic_latent_forward_prediction'] - alternate[1]['synthetic_latent_forward_prediction']).max())
                        parity = {'original_vs_bias_free_synthetic_forward_max_abs': output_gap,
                                  'loss_abs_gap': abs(original[0]['loss'] - alternate[0]['loss'])}
                        original[0]['bias_free_parity'] = alternate[0]['bias_free_parity'] = parity
                        if process.memory_info().rss > 2*1024**3: raise RuntimeError('CPU_RSS_OVER_2GIB')
                        if time.perf_counter() - started > 90: raise RuntimeError('CPU_DIAGNOSTIC_OVER_90SECONDS')
        output['source_hashes_unchanged'] = all(sha(ROOT / path) == value for path, value in initial_hashes.items())
        if not output['source_hashes_unchanged']: raise ValueError('READ_ONLY_SOURCE_OR_CHECKPOINT_CHANGED')
        output['status'] = 'COMPLETE'
    except Exception as exc:
        output.update(status='FAILED_PRESERVED', error=str(exc))
    output['resource_receipt'] = {'gpu': False, 'wall_seconds': time.perf_counter()-started, 'sampled_final_rss_bytes': process.memory_info().rss,
                                'torch_threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
                                'receipt_scope': 'This standalone CPU diagnostic; not included in original GPU timing.'}
    output['interpretation_limits'] = [
        'The exact-real-arithmetic final-bias derivative is zero. FP32 stored gradients/optimizer states may be nonzero.',
        'Synthetic CPU latent/reduction reproduction does not identify the exact kernel or gradients of historical GPU steps.',
        'Saved FIT residual Jacobian replay preserves saved prediction loss gradients, but does not reconstruct actual E/F latents.',
        'A one-step fresh or restored decoder-only AdamW diagnostic does not explain the entire observed 600-step bias change.',
        'Parameter cancellation is an algebraic capacity count, not a claim that FP32 parameters are unchanged or cost-free.',
        'No conclusion about forecasting failure causation or benefit of bias removal follows from this diagnostic.']
    write(folder / 'RESULT.json', output)
    report = [f'# 취소되는 디코더 bias 수치 진단\n\n상태: {output["status"]}\n',
              '실제 Jena NO_ALIGN의 CP0/600 디코더와 저장된 FIT 예측·정답을 사용했다. K 차원 미래/마지막 latent는 명시적으로 만든 synthetic 값이며 Chronos-Bolt의 실제 출력이 아니다.\n',
              '원 식 `x_last + D(future) - D(last)`에서 두 최종 affine bias의 정확한 미분은 0이다. 미래 B×H 행의 bias 합산과 broadcast 경로의 H→B 합산 순서 차이를 실제 저장 FIT residual gradient와 함께 기록했다.\n',
              '진단용 bias-free 식은 선형 분기의 `W(future-last)`와 비선형 hidden 차이의 최종 선형 변환을 사용한다. 이 식을 실제 학습에 적용하거나 체크포인트를 수정하지 않았다.\n']
    if 'actual_saved_bias_changes_u0_to_u600' in output:
        for name, value in output['actual_saved_bias_changes_u0_to_u600'].items():
            report.append(f'- `{name}`: 변경 {value["changed_scalar_count"]}개, 최대 절대 차이 {value["max_abs"]:.10g}\n')
        for dtype in ('torch.float32', 'torch.float64'):
            rows = [r for r in output['rows'] if r['dtype'] == dtype and r['form'] == 'original']
            if rows:
                largest = max(v['max_abs'] for r in rows for v in r['bias_gradients'].values())
                step = max(v['max_abs_parameter_change'] for r in rows for v in r['adamw_steps']['fresh_zero_moments'].values())
                parity = max(r['bias_free_parity']['original_vs_bias_free_synthetic_forward_max_abs'] for r in rows)
                report.append(f'\n{dtype}: original bias gradient 최대 {largest:.10g}, fresh AdamW 한 step 최대 변화 {step:.10g}, bias-free synthetic 출력 최대 차이 {parity:.10g}.\n')
    report.append('\n이 현상의 재현은 표현학습 실패 원인의 확정이나 bias 제거의 예측 성능 개선 증거가 아니다. 실제 과거 GPU 학습의 정확한 gradient tensor는 저장되지 않았으므로 이번 CPU 재현과 분리한다. AdamW restored 조건은 기존 momentum 자체로 갱신될 수도 있다.\n')
    with (folder / 'REPORT_KO.md').open('x', encoding='utf-8') as handle: handle.write(''.join(report))
    print(json.dumps({'status': output['status'], 'result': str(folder / 'RESULT.json'), 'rows': len(output['rows']), 'cpu_seconds': output['resource_receipt']['wall_seconds']}, ensure_ascii=True), flush=True)
    if output['status'] != 'COMPLETE': raise SystemExit(2)


if __name__ == '__main__': main()
