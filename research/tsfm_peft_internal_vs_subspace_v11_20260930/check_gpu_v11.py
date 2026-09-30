"""Reserved actual-Bolt forward/backward checks; zero optimizer updates."""
import argparse
import gc
import time
import traceback

import numpy as np
import torch

from model_v11 import (DATASETS, build_model, deployment_copy, restore_legacy_level,
                       state_digest)
from runtime_v11 import HERE, Job, load_data, save_json, sources_for, tensors
from train_v11 import configure, initial_for


def parity(left, right):
    error = (left - right).abs()
    allowed = 1e-5 + 1e-4 * right.abs()
    record = {'max_absolute': float(error.max()),
              'max_relative': float((error / right.abs().clamp_min(1e-12)).max()),
              'violating_elements': int((error > allowed).sum()), 'elements': error.numel()}
    assert record['violating_elements'] == 0, record
    return record


def check(job, output):
    records = []
    for dataset, (channels, latent) in DATASETS.items():
        data = load_data(dataset)
        x, _, _ = tensors(data, data['train_origins'][:4])
        for seed in (92601, 92602):
            sources = sources_for(dataset, seed, require_direct=True)
            common = dict(dataset=dataset, channels=channels, latent=latent, seed=seed, lr=1e-4)
            row = {'dataset': dataset, 'seed': seed, 'checks': {}}
            records.append(row)
            save_json(output, {'status': 'running', 'records': records, 'optimizer_updates': 0})
            a_spec = dict(common, family='a_p_lora_qfixed')
            initial, _ = initial_for(a_spec, data)
            a = build_model(a_spec, data['basis'], initial, 'cuda', sources)
            legacy = restore_legacy_level(sources['level_checkpoint'], 'cuda').eval()
            with torch.no_grad():
                row['checks']['a_equals_legacy_parent'] = parity(a(x), legacy(x))
                q_before = a.q_prediction(x).clone()
            del legacy
            a.zero_grad(set_to_none=True)
            a(x).square().mean().backward()
            grads = {n: p.grad for n, p in a.named_parameters() if p.requires_grad}
            assert all(g is None or torch.isfinite(g).all() for g in grads.values())
            assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads.values())
            assert all(p.grad is None for p in a.parameters() if not p.requires_grad)
            row['checks']['lora_gradient_active_tensor_count'] = sum(g is not None and bool(torch.count_nonzero(g)) for g in grads.values())
            with torch.no_grad():
                row['checks']['q_unchanged_without_update'] = parity(a.q_prediction(x), q_before)
                deployed = deployment_copy(a)
                row['checks']['a_merge'] = parity(deployed(x), a(x))
                deployed.chunk_rows = latent
                row['checks']['a_chunk_order'] = parity(deployed(x), a(x))
            del grads, a, deployed, q_before
            gc.collect()
            torch.cuda.empty_cache()
            full_spec = dict(common, family='full_lora_mse')
            full = build_model(full_spec, data['basis'], initial, 'cuda', sources)
            f0 = build_model(dict(common, family='f0'), data['basis'], device='cuda')
            with torch.no_grad():
                row['checks']['full_initial_f0'] = parity(full(x), f0(x))
            del full, f0
            gc.collect()
            torch.cuda.empty_cache()
            b_spec = dict(common, family='b_learned_u')
            b_initial, _ = initial_for(b_spec, data)
            b = build_model(b_spec, data['basis'], b_initial, 'cuda', sources)
            fixed = build_model(dict(common, family='b_fixed_u'), data['basis'], device='cuda', sources=sources)
            with torch.no_grad():
                row['checks']['b_initial_fixed'] = parity(b(x), fixed(x))
                row['checks']['u_initial_pca'] = parity(b.U, b.basis0)
                gram = float((b.U.T @ b.U - torch.eye(latent, device='cuda')).abs().max())
                assert gram <= 1e-5
                row['checks']['gram_max_error'] = gram
                b.set_gamma(torch.zeros(latent))
                row['checks']['gamma_zero_is_s'] = parity(b(x), b.direct_prediction(x))
                b.set_gamma(torch.ones(latent))
                features = b.mixture_features(x)
                decomposition = features['latent'] @ b.U.T + features['S'] - (features['S'] @ b.U) @ b.U.T
                row['checks']['gamma_identity_decomposition'] = parity(b(x), decomposition)
            del fixed, features, decomposition
            for label, context in [('train_context', x), ('constant', torch.zeros_like(x)),
                                   ('near_constant', torch.ones_like(x) + torch.randn_like(x) * 1e-5)]:
                with torch.no_grad():
                    normalizer = b.backbone.instance_norm
                    original_normalizer = type(normalizer)(eps=normalizer.eps, use_arcsinh=normalizer.use_arcsinh)
                    latent_context = (context @ b.U).transpose(1, 2).reshape(-1, 512)
                    repaired_value, repaired_stats = normalizer(latent_context)
                    original_value, original_stats = original_normalizer(latent_context)
                    assert torch.equal(repaired_value, original_value)
                    assert all(torch.equal(a, c) for a, c in zip(repaired_stats, original_stats))
                    row['checks']['native_normalization_forward_exact_' + label] = True
                b.zero_grad(set_to_none=True)
                prediction = b(context)
                assert torch.isfinite(prediction).all(), label
                prediction.square().mean().backward()
                gradient = b.directions.parametrizations.weight.original.grad
                check_row = {'finite_output': True, 'finite_u_gradient': bool(torch.isfinite(gradient).all()),
                             'nonzero_gradient_scalars': int(torch.count_nonzero(gradient))}
                row['checks']['b_gradient_' + label] = check_row
                save_json(output, {'status': 'running', 'records': records, 'optimizer_updates': 0})
                assert check_row['finite_u_gradient'], (dataset, seed, label, check_row)
                if label == 'train_context':
                    assert check_row['nonzero_gradient_scalars'] > 0
                assert all(p.grad is None for p in b.parameters() if not p.requires_grad)
            with torch.no_grad():
                deployed = deployment_copy(b)
                row['checks']['b_export'] = parity(deployed(x), b(x))
            del b, deployed, prediction, gradient
            gc.collect()
            torch.cuda.empty_cache()
            job.heartbeat(f'actual Bolt parity/backward {dataset}/{seed}; no optimizer', 0)
            save_json(output, {'status': 'running', 'records': records, 'optimizer_updates': 0})
        del x
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    configure()
    output = HERE / 'checks' / (args.job + '.json')
    if output.exists():
        raise RuntimeError('Preserve existing check result')
    started = time.perf_counter()
    with Job(args.job, category='gpu', reserve_seconds=600,
             metadata={'purpose': 'actual Bolt gradient and parent/initial/merge/export parity', 'optimizer_updates': 0}) as job:
        try:
            records = check(job, output)
            save_json(output, {'status': 'complete', 'records': records, 'optimizer_updates': 0,
                               'elapsed_s': time.perf_counter() - started})
        except BaseException:
            from runtime_v11 import read_json
            partial = read_json(output) if output.exists() else {}
            partial.update(status='failed', error=traceback.format_exc(), elapsed_s=time.perf_counter() - started)
            save_json(output, partial)
            raise


if __name__ == '__main__':
    main()
