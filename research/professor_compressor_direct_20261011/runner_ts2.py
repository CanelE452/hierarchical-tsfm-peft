"""Actual normal-size TS2Vec fit, new decoder, frozen F and native ridge study."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

from common_direct import (CACHE, HERE, ROOT, Resources, array_digest, digest,
                           masked_loss, observed_metrics, state_digest,
                           tensor_state, write_json)
from data_io import load_data, make_windows, rec_windows, ts2_train_array
from foundation import Foundation
from run_support import benchmark_path, save_case
from ts2_adapter import (RIDGE_ALPHAS, VENDOR, ReconstructionDecoder, create_ts2,
                         export_last_causal, export_offline, fit_default_iterations,
                         freeze_encoder, model_configuration, vendor_provenance)


@torch.no_grad()
def predict_decoder(decoder, features, batch_size=32):
    values = []
    decoder.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start+batch_size]).cuda()
        values.append(decoder(batch).cpu().numpy())
    return np.concatenate(values)


def fit_decoder(decoder, train_features, train_target, train_mask, val_features,
                val_target, val_mask, resources, folder, label, seed,
                learning_rate, max_epochs, patience, include_initial=False):
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    best_score, stale, best_epoch, updates = float('inf'), 0, None, 0
    trace = []
    best_path = folder / f'{label}_best.pt'
    initial_hash = state_digest(decoder)
    initial_score = None
    if include_initial:
        initial_prediction = predict_decoder(decoder, val_features)
        initial_metrics = observed_metrics(initial_prediction, val_target, val_mask)
        initial_score = initial_metrics['mse']
        best_score, best_epoch = initial_score, 0
        trace.append({'epoch': 0, 'train_observed_mse': None,
                      'val_macro_mse': initial_score,
                      'val_macro_mae': initial_metrics['mae'],
                      'updates': 0, 'improved': True,
                      'candidate': 'unchanged reconstruction decoder before calibration'})
        torch.save({'state_dict': tensor_state(decoder), 'epoch': 0,
                    'val_macro_mse': initial_score, 'optimizer_updates': 0}, best_path)
        write_json(folder / f'{label}_curve.json', trace)
        resources.check(stage=label, epoch=0, max_epochs=max_epochs,
                        optimizer_updates=0, initial_val_macro_mse=initial_score)
        print(f'{label} epoch=0 updates=0 val={initial_score:.6f} initial_checkpoint_candidate', flush=True)
    for epoch in range(1, max_epochs+1):
        order = rng.permutation(len(train_features))
        decoder.train()
        loss_sum, count = 0., 0
        for start in range(0, len(order), 8):
            indices = order[start:start+8]
            z = torch.from_numpy(train_features[indices]).cuda()
            target = torch.from_numpy(train_target[indices]).cuda()
            mask = torch.from_numpy(train_mask[indices]).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(decoder(z), target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{label} non-finite observed reconstruction/forecast loss')
            loss.backward()
            optimizer.step()
            observations = int(mask.sum())
            loss_sum += float(loss.detach()) * observations
            count += observations
            updates += 1
            resources.check(stage=label, epoch=epoch, max_epochs=max_epochs,
                            optimizer_updates=updates)
        val_pred = predict_decoder(decoder, val_features)
        metrics = observed_metrics(val_pred, val_target, val_mask)
        score = metrics['mse']
        improved = score < best_score
        trace.append({'epoch': epoch, 'train_observed_mse': loss_sum/max(count, 1),
                      'val_macro_mse': score, 'val_macro_mae': metrics['mae'],
                      'updates': updates, 'improved': improved})
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({'state_dict': tensor_state(decoder), 'epoch': epoch,
                        'val_macro_mse': score, 'optimizer_updates': updates}, best_path)
        else:
            stale += 1
        write_json(folder / f'{label}_curve.json', trace)
        print(f'{label} epoch={epoch} updates={updates} train={trace[-1]["train_observed_mse"]:.6f} val={score:.6f}', flush=True)
        if stale >= patience:
            break
    final_path = folder / f'{label}_final.pt'
    torch.save({'state_dict': tensor_state(decoder), 'epoch': epoch,
                'optimizer_updates': updates}, final_path)
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    decoder.load_state_dict(best['state_dict'])
    decoder.eval()
    return {'optimizer': 'Adam', 'learning_rate': learning_rate, 'batch_size_windows': 8,
            'max_epochs': max_epochs, 'patience': patience, 'epochs': epoch,
            'optimizer_updates': updates, 'best_epoch': best_epoch,
            'best_val_macro_mse': best_score,
            'checkpoint_selection': 'repeated development VAL, predeclared MSE/patience; first strict minimum',
            'initial_checkpoint_candidate_included': include_initial,
            'initial_val_macro_mse': initial_score,
            'candidate_epoch_range': [0 if include_initial else 1, epoch],
            'patience_rule': 'consecutive completed epochs without strict improvement over the current best candidate',
            'selected_optimizer_updates': best['optimizer_updates'],
            'training_loss': 'sum squared error / original observed entry count',
            'initial_state_sha256': initial_hash, 'selected_state_sha256': state_digest(decoder),
            'initial_to_selected_changed': initial_hash != state_digest(decoder),
            'best_checkpoint': best_path, 'best_checkpoint_sha256': digest(best_path),
            'final_checkpoint': final_path, 'final_checkpoint_sha256': digest(final_path),
            'learning_curve': folder / f'{label}_curve.json', 'training_windows': len(train_features)}


def fit_observed_ridge(train_features, train_target, train_mask,
                       val_features, val_target, val_mask):
    """Official intercept/alpha criterion, grouped by original observed masks."""
    x = np.asarray(train_features, dtype=np.float64)
    xv = np.asarray(val_features, dtype=np.float64)
    y = train_target.reshape(len(train_target), -1).astype(np.float64)
    ym = train_mask.reshape(len(train_mask), -1)
    yv = val_target.reshape(len(val_target), -1).astype(np.float64)
    yvm = val_mask.reshape(len(val_mask), -1)
    groups = {}
    for output in range(y.shape[1]):
        groups.setdefault(ym[:, output].tobytes(), []).append(output)
    decompositions = []
    unavailable = []
    for outputs in groups.values():
        valid = ym[:, outputs[0]]
        if not valid.any():
            unavailable.extend(outputs)
            continue
        xx, yy = x[valid], y[valid][:, outputs]
        xm, ymean = xx.mean(0), yy.mean(0)
        centered = xx-xm
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered)
        transformed = eigenvectors.T@(centered.T@(yy-ymean))
        decompositions.append((outputs, xm, ymean, eigenvalues, eigenvectors, transformed))
    scores, best = [], None
    for alpha in RIDGE_ALPHAS:
        coefficient = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
        intercept = np.zeros(y.shape[1], dtype=np.float64)
        for outputs, xm, ymean, eigenvalues, eigenvectors, transformed in decompositions:
            block = eigenvectors@(transformed/(eigenvalues[:, None]+alpha))
            coefficient[:, outputs] = block
            intercept[outputs] = ymean-xm@block
        prediction = xv@coefficient+intercept
        error = (prediction-yv)[yvm]
        score = float(np.sqrt(np.square(error).mean())+np.abs(error).mean())
        scores.append({'alpha': alpha, 'validation_observed_rmse_plus_mae': score})
        if best is None or score < best[0]:
            best = score, alpha, coefficient, intercept, prediction
    return best[2], best[3], best[4].reshape(val_target.shape).astype(np.float32), {
        'alpha': best[1], 'alpha_grid': RIDGE_ALPHAS, 'scores': scores,
        'criterion': 'sqrt(observed entry MSE) + observed entry MAE, first argmin',
        'fit_intercept': True, 'training_origins': len(x), 'mask_pattern_groups': len(groups),
        'unobserved_training_outputs': unavailable,
        'implementation': 'centered sklearn Ridge objective via mask-group Gram eigendecomposition',
        'deviation': 'Missing labels excluded using original mask; repeated development VAL selects alpha.'}


def verify_ridge_math():
    rng = np.random.default_rng(92801)
    x = rng.normal(size=(17, 9))
    y = rng.normal(size=(17, 6))
    xv = rng.normal(size=(7, 9))
    yv = rng.normal(size=(7, 6))
    mask = np.ones_like(y, dtype=bool)
    mask[::3, :2] = False
    mask[1::4, 2:4] = False
    val_mask = np.ones_like(yv, dtype=bool)
    val_mask[::2, 1] = False
    coefficient, intercept, prediction, record = fit_observed_ridge(x, y, mask, xv, yv, val_mask)
    scores = []
    selected = None
    for alpha in RIDGE_ALPHAS:
        pred = np.empty_like(yv)
        for column in range(y.shape[1]):
            valid = mask[:, column]
            model = Ridge(alpha=alpha).fit(x[valid], y[valid, column])
            pred[:, column] = model.predict(xv)
        error = (pred-yv)[val_mask]
        scores.append(float(np.sqrt(np.square(error).mean())+np.abs(error).mean()))
        if alpha == record['alpha']:
            selected = pred
    error = float(np.max(np.abs(selected-prediction)))
    score_error = float(np.max(np.abs(np.array(scores)-np.array([r['validation_observed_rmse_plus_mae'] for r in record['scores']]))))
    if error > 1e-6 or score_error > 1e-9 or RIDGE_ALPHAS[int(np.argmin(scores))] != record['alpha']:
        raise RuntimeError('Grouped observed Ridge differs from independent sklearn calculation')
    return {'status': 'PASS', 'selected_alpha': record['alpha'],
            'prediction_max_abs_error': error, 'all_alpha_score_max_abs_error': score_error,
            'fixture': '17 train x9,6 outputs,3 original mask patterns,13 alphas,independent sklearn'}


def native_features(model, data, origins, resources, label, batch_size=256):
    output = []
    elapsed_first = []
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start+batch_size]
        past = np.stack([data['x'][int(origin)-201:int(origin)] for origin in batch_origins])
        torch.cuda.synchronize()
        started = time.perf_counter()
        output.append(export_last_causal(model, past, batch_size=batch_size))
        torch.cuda.synchronize()
        duration = time.perf_counter()-started
        elapsed_first.append(duration)
        if len(elapsed_first) == min(10, int(np.ceil(len(origins)/batch_size))):
            estimated = float(np.mean(elapsed_first)*np.ceil(len(origins)/batch_size))
            print(f'{label} measured_batches={len(elapsed_first)} estimate_total_seconds={estimated:.1f}', flush=True)
        resources.check(stage=label, completed_origins=min(start+batch_size, len(origins)),
                        total_origins=len(origins))
    return np.concatenate(output), {'batch_size': batch_size,
            'actual_batches': len(elapsed_first), 'batch_seconds': elapsed_first,
            'total_seconds': sum(elapsed_first), 'origin_count': len(origins),
            'last_window_length': 201, 'official_causal_rolling_specialization': True}


def run(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    job_id = f'{args.dataset}_TS2_d{args.dim}_s{args.seed}'
    if args.retry_tag:
        job_id += f'_retry_{args.retry_tag}'
    job = {'id': job_id, 'dataset': args.dataset, 'seed': args.seed,
           'family': 'TS2', 'dim': args.dim, 'started_unix': time.time()}
    folder = CACHE / 'runs' / job_id
    result_dir = HERE / 'runs' / job_id
    if folder.exists() or result_dir.exists():
        raise FileExistsError(f'Preserve existing run; select a new retry tag: {job_id}')
    folder.mkdir(parents=True)
    resources = Resources()
    curve = []
    model = None
    try:
        with resources.stage('load_TRAIN_VAL_and_normal_encoder', gpu=False, job=job_id):
            data = load_data(args.dataset)
            train = ts2_train_array(data, nan_for_unobserved=True)
            source = vendor_provenance()
            expected_iterations = fit_default_iterations(train)
            write_json(folder / 'initial_protocol.json', {
                'job': job, 'data_contract_hash': data['contract_hash'],
                'TRAIN_shape': train.shape, 'expected_official_iterations': expected_iterations,
                'contiguous_TRAIN_only': True, 'input_missing': 'official NaN masking',
                'hidden_dims': 64, 'depth': 10, 'output_dims': args.dim,
                'batch_size': args.encoder_batch, 'max_train_length': 3000,
                'lr': .001, 'source': source, 'protected_TEST_new_access': 0})

        def callback(current, loss):
            curve.append({'iteration': current.n_iters, 'loss': loss,
                          'swa_n_averaged': int(current.net.n_averaged), 'unix': time.time()})
            resources.check(stage='official_TS2Vec_fit', job=job_id,
                            iteration=current.n_iters, expected_iterations=expected_iterations,
                            loss=loss)
            if current.n_iters % 25 == 0 or current.n_iters == expected_iterations:
                write_json(folder / 'encoder_curve.json', curve)
                current.save(str(folder / f'encoder_swa_iteration{current.n_iters}.pt'))
                print(f'{job_id} encoder updates={current.n_iters}/{expected_iterations} loss={loss:.6f}', flush=True)

        with resources.stage('official_TS2Vec_representation_fit', gpu=True, job=job_id):
            model = create_ts2(data['x'].shape[1], args.dim, args.seed, device='cuda',
                               after_iter_callback=callback, batch_size=args.encoder_batch)
            initial_state = state_digest(model._net)
            epoch_curve = model.fit(train.copy(), verbose=False)
            if model.n_iters != expected_iterations:
                raise RuntimeError('Official default iteration budget was not completed')
            encoder_path = folder / 'encoder_swa.pt'
            model.save(str(encoder_path))
            encoder_config = model_configuration(model)
            write_json(folder / 'encoder_curve.json', curve)
            encoder_training = {'configuration': encoder_config, 'TRAIN_shape': train.shape,
                    'TRAIN_scalar_count': train.size, 'epoch_loss': epoch_curve,
                    'iteration_curve': folder / 'encoder_curve.json',
                    'initial_online_state_sha256': initial_state,
                    'final_online_state_sha256': state_digest(model._net),
                    'inference_swa_state_sha256': state_digest(model.net),
                    'checkpoint': encoder_path, 'checkpoint_sha256': digest(encoder_path),
                    'sampling': 'official contiguous-series split/crop; no overlapping-window negative pool',
                    'seed_rule': 'official init_dl_program random seed, numpy seed+1, torch seed+2, CUDA seed+3',
                    'algorithm_modified': False, 'encoder_batch_adjustment': args.encoder_batch != 8}
            freeze_encoder(model)
            model._net.cpu()

        with resources.stage('offline_past_only_feature_export', gpu=True, job=job_id):
            rec_x, rec_mask = rec_windows(data, data['train_window_starts'])
            val_x, val_xmask = make_windows(data, data['val_origins'], masked=True)
            train_x = make_windows(data, data['train_forecast_origins'])
            train_y, train_ymask = make_windows(data, data['train_forecast_origins'], future=True, masked=True)
            val_y, val_ymask = make_windows(data, data['val_origins'], future=True, masked=True)
            rec_z = export_offline(model, rec_x, batch_size=8)
            val_z = export_offline(model, val_x, batch_size=8)
            train_z = export_offline(model, train_x, batch_size=8)
            resources.check(stage='features_exported', job=job_id, rec_windows=len(rec_x),
                            train_forecast_origins=len(train_x), val_origins=len(val_x))
            replica = create_ts2(data['x'].shape[1], args.dim, args.seed+100, device='cuda')
            replica.load(str(encoder_path))
            restored = export_offline(replica, val_x[:1], batch_size=1)
            restore_difference = float(np.max(np.abs(restored-val_z[:1])))
            if not np.allclose(restored, val_z[:1], atol=2e-5, rtol=2e-5):
                raise RuntimeError('Actual trained encoder save/load export mismatch')
            del replica

        torch.manual_seed(args.seed+31)
        decoder_a = ReconstructionDecoder(args.dim, data['x'].shape[1]).cuda()
        with resources.stage('TS2_new_reconstruction_decoder_fit', gpu=True, job=job_id):
            drec_training = fit_decoder(decoder_a, rec_z, rec_x, rec_mask,
                        val_z, val_x, val_xmask, resources, folder, 'reconstruction_decoder',
                        args.seed+32, 5e-4, 100, 15)
            reconstruction = predict_decoder(decoder_a, val_z)
        reconstruction_metrics = observed_metrics(reconstruction, val_x, val_xmask, data['std'])
        encoder_frozen_hash = state_digest(model.net)
        decoder_a_hash = state_digest(decoder_a)
        with resources.stage('actual_frozen_TSFM_TRAIN_VAL_latent_forecasts', gpu=True, job=job_id):
            foundation = Foundation(resources)
            pipeline_parity = foundation.verify_pipeline_parity()
            forecast_arrays = []
            for label, values in (('TRAIN', train_z), ('VAL', val_z)):
                predictions = []
                for start in range(0, len(values), 8):
                    latent = torch.from_numpy(values[start:start+8]).cuda()
                    predictions.append(foundation.forecast(latent, 48).cpu().numpy())
                    resources.check(stage='actual_F_latent_prediction', job=job_id,
                                    split=label, completed_origins=min(start+8, len(values)),
                                    total_origins=len(values))
                forecast_arrays.append(np.concatenate(predictions))
            train_zf, val_zf = forecast_arrays
            transfer = predict_decoder(decoder_a, val_zf)
            persist_z = np.repeat(val_z[:, -1:], 48, axis=1)
            persist_transfer = predict_decoder(decoder_a, persist_z)
            np.savez_compressed(folder / 'actual_F_latent_predictions.npz',
                                train_origins=data['train_forecast_origins'], train=train_zf,
                                val_origins=data['val_origins'], val=val_zf)

        decoder_b = copy.deepcopy(decoder_a)
        with resources.stage('same_native_TS2_decoder_forecast_calibration', gpu=True, job=job_id):
            calibration = fit_decoder(decoder_b, train_zf, train_y, train_ymask,
                    val_zf, val_y, val_ymask, resources, folder, 'forecast_calibrated_decoder',
                    args.seed+33, 1e-4, 50, 10, include_initial=True)
            calibrated = predict_decoder(decoder_b, val_zf)
            persist_calibrated = predict_decoder(decoder_b, persist_z)
            if state_digest(decoder_a) != decoder_a_hash:
                raise RuntimeError('Calibration changed original reconstruction decoder')

        with resources.stage('official_causal_last_representation_export', gpu=True, job=job_id):
            train_start, train_end = data['train_bounds']
            native_origins = np.arange(train_start+512, train_end-48+1, dtype=np.int64)
            native_train, native_train_timing = native_features(model, data, native_origins, resources, 'native_full_TRAIN_stride1')
            native_val, native_val_timing = native_features(model, data, data['val_origins'], resources, 'native_VAL')
            official_reference = model.encode(val_x[:1].copy(), causal=True, sliding_length=1,
                                            sliding_padding=200, batch_size=256)[:, -1]
            causal_difference = float(np.max(np.abs(official_reference-native_val[:1])))
            if not np.allclose(official_reference, native_val[:1], atol=2e-5, rtol=2e-5):
                raise RuntimeError('Actual trained normal encoder causal rolling specialization differs')
            np.savez_compressed(folder / 'native_causal_features.npz',
                                train=native_train, val=native_val, train_origins=native_origins,
                                val_origins=data['val_origins'])

        with resources.stage('original_forecasting_ridge_native_head', gpu=False, job=job_id):
            ridge_check = verify_ridge_math()
            native_y, native_mask = make_windows(data, native_origins, future=True, masked=True)
            coefficient, intercept, native_prediction, native_record = fit_observed_ridge(
                            native_train, native_y, native_mask, native_val, val_y, val_ymask)
            ridge_path = folder / 'native_ridge.npz'
            np.savez_compressed(ridge_path, coefficient=coefficient, intercept=intercept,
                                alpha=np.array(native_record['alpha']), train_origins=native_origins)
            native_record.update({'ridge_math_verification': ridge_check,
                        'checkpoint': ridge_path, 'checkpoint_sha256': digest(ridge_path),
                        'TRAIN_stride': 1, 'TRAIN_first_origin': int(native_origins[0]),
                        'TRAIN_last_origin': int(native_origins[-1]),
                        'minimum_past_required': 512, 'representation_effective_past': 201,
                        'last_causal_rolling_actual_max_abs_difference': causal_difference,
                        'train_export_timing': native_train_timing, 'val_export_timing': native_val_timing,
                        'native_forecasting_deviation': 'No calendar covariates, common TRAIN-only scaler and original observed masks; origins require full common L512 history.'})

        predictions = {'TRANSFER_F': transfer, 'CALIBRATED_F': calibrated,
                       'LATENT_PERSIST_TRANSFER': persist_transfer,
                       'LATENT_PERSIST_CALIBRATED': persist_calibrated,
                       'TS2_NATIVE_RIDGE': native_prediction}
        metrics = {name: observed_metrics(value, val_y, val_ymask, data['std'])
                   for name, value in predictions.items()}

        def infer_with(decoder):
            def inference(x):
                latent = export_offline(model, x.detach().cpu().numpy(), batch_size=8)
                latent_tensor = torch.from_numpy(latent).cuda()
                with torch.no_grad():
                    return decoder(foundation.forecast(latent_tensor, 48))
            return inference

        infer_a, infer_b = infer_with(decoder_a), infer_with(decoder_b)
        with resources.stage('actual_inference_origin_future_perturbation', gpu=True, job=job_id):
            origin = int(data['val_origins'][0])
            sequence = data['x'][origin-512:origin+48].copy()
            modified = sequence.copy()
            modified[512:] = np.random.default_rng(92699).normal(100, 30, modified[512:].shape)
            before = infer_a(torch.from_numpy(sequence[:512][None]).cuda()).cpu().numpy()
            after = infer_a(torch.from_numpy(modified[:512][None]).cuda()).cpu().numpy()
            future_unchanged = np.array_equal(before, after)
            if not future_unchanged:
                raise RuntimeError('Origin-after observations alter actual E/F/D inference')

        cost = benchmark_path(infer_a, data, resources, 'TRANSFER_F', transfer)
        cost += benchmark_path(infer_b, data, resources, 'CALIBRATED_F', calibrated)
        coefficient_gpu = torch.from_numpy(coefficient.astype(np.float32)).cuda()
        intercept_gpu = torch.from_numpy(intercept.astype(np.float32)).cuda()

        def infer_native(x):
            representation = export_last_causal(model, x.detach().cpu().numpy(), batch_size=256)
            features = torch.from_numpy(representation).cuda()
            return (features@coefficient_gpu+intercept_gpu).reshape(len(x), 48, data['x'].shape[1])

        cost += benchmark_path(infer_native, data, resources, 'TS2_NATIVE_RIDGE', native_prediction)
        foundation_check = foundation.verify_unchanged()
        frozen = state_digest(model.net) == encoder_frozen_hash
        if not frozen or not foundation_check['unchanged'] or not foundation_check['parameter_gradients_absent']:
            raise RuntimeError('Frozen encoder or Foundation weights changed during decoder fitting/inference')
        positions = np.unique(np.array([0, len(val_x)//2, len(val_x)-1], dtype=np.int64))
        diagnostic = {'indices': positions, 'past': val_x[positions],
                      'past_mask': val_xmask[positions], 'reconstruction': reconstruction[positions],
                      'latent_history': val_z[positions], 'latent_forecast': val_zf[positions],
                      'targets': val_y[positions], 'target_mask': val_ymask[positions],
                      'origins': data['val_origins'][positions]}
        training = {'encoder': encoder_training, 'decoder_reconstruction': drec_training,
                    'native_forecasting_head': native_record,
                    'representation_export': 'offline separate past-only window, 512 ticks',
                    'encoder_frozen_during_all_decoder_training': frozen,
                    'SWA_save_load_actual_max_abs_difference': restore_difference,
                    'actual_future_perturbation_unchanged': future_unchanged,
                    'foundation_pipeline_parity': pipeline_parity,
                    'foundation_frozen_verification': foundation_check}
        calibration.update({'encoder_and_F_frozen': True,
                            'input': 'actual frozen F TRAIN latent predictions, never E(encoded true future)',
                            'original_decoder_preserved': state_digest(decoder_a) == decoder_a_hash,
                            'extra_supervision': 'TRAIN future original-space observed labels',
                            'actual_F_cache': folder / 'actual_F_latent_predictions.npz'})
        shape = {'past': [512, data['x'].shape[1]], 'latent': [512, args.dim],
                 'latent_future': [48, args.dim], 'output': [48, data['x'].shape[1]],
                 'input_scalar_count_per_origin': 512*data['x'].shape[1],
                 'latent_scalar_count_per_origin': 512*args.dim,
                 'F_rows_per_origin': args.dim, 'F_context_length': 512,
                 'physical_future_horizon': 48, 'temporal_downsampling': 1,
                 'representation_category': 'expansion' if args.dim > data['x'].shape[1] else 'channel_dimensional_reduction',
                 'online_encoder_residence_during_inference': 'CPU frozen, separate full SWA encoder on CUDA',
                 'decoder_parameters': sum(p.numel() for p in decoder_a.parameters()),
                 'prototype_resident_decoder_copies': 2,
                 'deployment_transfer': 'official NumPy encode includes D2H input, H2D encoder, D2H repr and H2D F; costs include these transfers'}
        source_files = [HERE/'runner_ts2.py', HERE/'ts2_adapter.py', HERE/'data_io.py',
                        HERE/'common_direct.py', HERE/'foundation.py', HERE/'run_support.py']
        source_files += [VENDOR/item['path'] for item in source['files']]
        resources.flush()
        result = save_case(job, data, reconstruction_metrics, metrics, predictions, diagnostic,
                           training, calibration, shape, cost, resources, source_files)
        print(f'{job_id} COMPLETE methods={len(metrics)} iterations={model.n_iters}', flush=True)
        return result
    except Exception as error:
        if model is not None:
            try:
                model.save(str(folder / f'encoder_failure_partial_{time.time_ns()}.pt'))
            except Exception:
                pass
        resources.flush()
        write_json(result_dir / 'failure.json', {'status': 'incomplete', 'job': job,
                    'error': repr(error), 'actual_encoder_iterations': model.n_iters if model is not None else 0,
                    'encoder_curve': curve, 'resources': resources.value,
                    'resume': 'Preserve this attempt; new --retry-tag and, for CUDA OOM, --encoder-batch 4. Official SWA-only checkpoint is not optimizer resume.'})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('robin', 'jena'), required=True)
    parser.add_argument('--dim', type=int, choices=(16, 320), required=True)
    parser.add_argument('--seed', type=int, choices=(92601, 92602), required=True)
    parser.add_argument('--encoder-batch', type=int, default=8)
    parser.add_argument('--retry-tag')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
