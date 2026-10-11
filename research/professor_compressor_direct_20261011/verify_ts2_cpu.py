"""Small CPU-only algorithm, source, restoration and export contract checks."""
from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import torch

from common_direct import CACHE, HERE, Resources, digest, state_digest, write_json
from ts2_adapter import (ReconstructionDecoder, create_ts2, export_last_causal,
                         export_offline, fit_default_iterations, freeze_encoder,
                         model_configuration, official_class, vendor_provenance)


def main():
    torch.set_num_threads(2)
    resources = Resources()
    records = []
    run_id = str(time.time_ns())
    output = HERE / f'TS2_CPU_VERIFICATION_{run_id}.json'
    checkpoint_dir = CACHE / 'cpu_checks' / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    def check(name, condition, **evidence):
        records.append({'name': name, 'passed': bool(condition), **evidence})
        if not condition:
            raise AssertionError(name)

    try:
        with resources.stage('TS2_source_import_CPU_fixture', gpu=False):
            original = {key: sys.modules.get(key) for key in ('models', 'utils', 'models.losses')}
            sentinels = {key: types.ModuleType(key) for key in original}
            sys.modules.update(sentinels)
            try:
                official_class()
                check('absolute_import_bindings_restored', all(sys.modules[key] is value for key, value in sentinels.items()))
            finally:
                for key, value in original.items():
                    if value is None:
                        sys.modules.pop(key, None)
                    else:
                        sys.modules[key] = value
            source = vendor_provenance()
            check('all_seven_upstream_sources_unmodified', len(source['files']) == 7 and all(p['matches'] for p in source['files']), files=source['files'])
            for channels in (17, 21):
                for dim in (16, 320):
                    model = create_ts2(channels, dim, 92601, device='cpu')
                    config = model_configuration(model)
                    check(f'normal_size_C{channels}_d{dim}', config['hidden_dims'] == 64 and config['depth'] == 10 and config['output_dims'] == dim and config['lr'] == .001 and config['batch_size'] == 8 and config['max_train_length'] == 3000, configuration=config)
                    decoder = ReconstructionDecoder(dim, channels)
                    shape = tuple(decoder(torch.zeros(2, 23, dim)).shape)
                    check(f'new_decoder_shape_linear_output_C{channels}_d{dim}', shape == (2, 23, channels) and isinstance(decoder.layers[-1], torch.nn.Linear), shape=shape, parameters=sum(p.numel() for p in decoder.parameters()))
            check('official_default_fit_iteration_rule', fit_default_iterations(np.zeros((1, 100000, 1))) == 200 and fit_default_iterations(np.zeros((1, 100001, 1))) == 600)

        with resources.stage('TS2_actual_fit_save_load_CPU_fixture', gpu=False):
            rng = np.random.default_rng(92601)
            for dim in (16, 320):
                history = rng.normal(size=(2, 24, 3)).astype(np.float32)
                history[0, 4, 1] = np.nan
                history[1, 11, :] = np.nan
                losses = []
                model = create_ts2(3, dim, 92601, device='cpu',
                                    after_iter_callback=lambda net, loss: losses.append({'iteration': net.n_iters, 'loss': loss}),
                                    batch_size=2)
                before = state_digest(model._net)
                fit_losses = model.fit(history.copy(), n_iters=3, verbose=False)
                after = state_digest(model._net)
                check(f'actual_original_fit_updates_swa_d{dim}', model.n_iters == 3 and len(losses) == 3 and before != after and int(model.net.n_averaged) == 4 and all(np.isfinite(p['loss']) for p in losses), iterations=model.n_iters, iteration_losses=losses, epoch_losses=fit_losses, online_before=before, online_after=after, swa_n_averaged=int(model.net.n_averaged))
                offline = export_offline(model, history, batch_size=2)
                check(f'missing_values_encode_finite_d{dim}', offline.shape == (2, 24, dim) and np.isfinite(offline).all())
                checkpoint = checkpoint_dir / f'official_swa_d{dim}.pt'
                model.save(str(checkpoint))
                restored = create_ts2(3, dim, 92602, device='cpu')
                restored.load(str(checkpoint))
                actual = export_offline(restored, history, batch_size=2)
                check(f'official_save_load_bitwise_d{dim}', np.array_equal(offline, actual), checkpoint=str(checkpoint), sha256=digest(checkpoint), max_abs_error=float(np.max(np.abs(offline-actual))), note='Official checkpoint stores SWA inference state; not optimizer/online resume state.')

                for length in (17, 213):
                    values = rng.normal(size=(1, length, 3)).astype(np.float32)
                    expected = model.encode(values.copy(), causal=True, sliding_length=1,
                                            sliding_padding=200, batch_size=16)[:, -1]
                    fast = export_last_causal(model, values, batch_size=16)
                    difference = float(np.max(np.abs(expected-fast)))
                    check(f'official_causal_last_roll_equivalence_d{dim}_L{length}', np.allclose(expected, fast, atol=2e-5, rtol=2e-5), max_abs_error=difference, official_arguments={'causal': True, 'sliding_length': 1, 'sliding_padding': 200}, specialized_window=201)

                origin = 31
                complete = rng.normal(size=(2, origin+9, 3)).astype(np.float32)
                perturbed = complete.copy()
                perturbed[:, origin:] = rng.normal(100., 30., size=perturbed[:, origin:].shape)
                first = export_offline(model, complete[:, :origin], batch_size=2)
                second = export_offline(model, perturbed[:, :origin], batch_size=2)
                causal_first = export_last_causal(model, complete[:, :origin], batch_size=2)
                causal_second = export_last_causal(model, perturbed[:, :origin], batch_size=2)
                check(f'origin_future_perturbation_export_contract_d{dim}', np.array_equal(first, second) and np.array_equal(causal_first, causal_second), origin=origin, future_points_modified=9, scope='Adapter past-only export. Actual E/F/D future perturbation verified separately by runner.')
                freeze_encoder(model)
                check(f'freeze_both_online_and_swa_d{dim}', not any(p.requires_grad for net in (model._net, model.net) for p in net.parameters()))
                resources.check(stage='TS2_CPU_fixture', dimension=dim)
        check('CPU_only_no_CUDA_initialized', not torch.cuda.is_initialized())
        status = 'PASS'
    except Exception as error:
        status = 'FAILED'
        records.append({'name': 'exception', 'passed': False, 'error': repr(error)})
    resources.flush()
    write_json(output, {'status': status, 'scope': 'Synthetic implementation fixture only; no actual data or scientific forecast run.', 'records': records, 'source_sha256': {'adapter': digest(HERE/'ts2_adapter.py'), 'verification': digest(Path(__file__))}, 'resource_receipt': str(resources.path), 'GPU_jobs': 0, 'actual_data_fit': 0, 'protected_TEST_reads': 0})
    print(json.dumps({'status': status, 'checks': len(records), 'output': str(output)}, ensure_ascii=False), flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
