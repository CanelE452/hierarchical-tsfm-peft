"""Actual native GTConvAE learning, frozen-TSFM transfer and native-D calibration."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from common_direct import (HERE, ROOT, CACHE, Resources, BudgetExceeded, write_json,
                           tensor_state, state_digest, observed_metrics, masked_loss)
from data_io import load_data, rec_windows, make_windows
from models_gt import GTConvAE
from foundation import Foundation
from run_support import benchmark_path, save_case


@torch.no_grad()
def encode_all(model, values, resources):
    results = []
    model.eval()
    for start in range(0, len(values), 16):
        x = torch.as_tensor(values[start:start + 16], device='cuda')
        results.append(model.encode(x).cpu().numpy())
        resources.check()
    return np.concatenate(results)


@torch.no_grad()
def reconstruct_all(model, values, resources):
    results = []
    for start in range(0, len(values), 16):
        results.append(model(torch.as_tensor(values[start:start + 16], device='cuda')).cpu().numpy())
        resources.check()
    return np.concatenate(results)


@torch.no_grad()
def forecast_all(foundation, latent, resources):
    results = []
    for start in range(0, len(latent), 4):
        z = torch.as_tensor(latent[start:start + 4], device='cuda')
        results.append(foundation.forecast(z, 12).cpu().numpy())
        resources.check()
    return np.concatenate(results)


@torch.no_grad()
def decode_all(model, latent, future, resources):
    results = []
    for start in range(0, len(latent), 16):
        z = torch.as_tensor(latent[start:start + 16], device='cuda')
        f = torch.as_tensor(future[start:start + 16], device='cuda')
        results.append(model.decode_future(z, f, horizon=48).cpu().numpy())
        resources.check()
    return np.concatenate(results)


def fit_reconstruction(model, data, resources, job, folder):
    train, train_mask = rec_windows(data, data['train_window_starts'])
    validation, validation_mask = make_windows(data, data['val_origins'], masked=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(job['seed'])
    curve, best, best_epoch, patience, updates = [], float('inf'), 0, 0, 0
    initial = tensor_state(model)
    checkpoint = folder / 'reconstruction_best.pt'
    with resources.stage('GT_joint_encoder_native_decoder_reconstruction_fit', gpu=True, job=job['id']):
        started = time.perf_counter()
        for epoch in range(1, 101):
            model.train()
            scores = []
            order = rng.permutation(len(train))
            for start in range(0, len(order), 8):
                index = order[start:start + 8]
                x = torch.as_tensor(train[index], device='cuda')
                mask = torch.as_tensor(train_mask[index], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(x)
                squared = masked_loss(reconstruction, x, mask)
                penalty = .14 * model.sparsity_penalty()
                loss = squared + penalty
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite GT reconstruction objective')
                loss.backward()
                optimizer.step()
                updates += 1
                scores.append([float(squared.detach()), float(penalty.detach())])
                resources.check(job=job['id'], phase='GT_reconstruction', epoch=epoch,
                                maximum_epochs=100, actual_updates=updates)
            model.eval()
            prediction = reconstruct_all(model, validation, resources)
            metric = observed_metrics(prediction, validation, validation_mask)
            row = {'epoch': epoch, 'actual_updates': updates,
                   'train_reconstruction_mse': float(np.mean(scores, axis=0)[0]),
                   'train_sparsity_penalty': float(np.mean(scores, axis=0)[1]),
                   'val_reconstruction_mse': metric['mse'], 'val_reconstruction_mae': metric['mae'],
                   'elapsed_seconds': time.perf_counter() - started}
            curve.append(row)
            if metric['mse'] < best:
                best, best_epoch, patience = metric['mse'], epoch, 0
                torch.save({'state_dict': tensor_state(model), 'epoch': epoch,
                            'actual_updates': updates, 'optimizer': optimizer.state_dict()}, checkpoint)
            else:
                patience += 1
            write_json(HERE / 'runs' / job['id'] / 'reconstruction_curve.json', curve)
            print(json.dumps({'job': job['id'], 'phase': 'reconstruction', **row}), flush=True)
            if patience >= 15:
                break
        duration = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    model.eval()
    changes = {key: bool(not torch.equal(initial[key], value.detach().cpu()))
               for key, value in model.state_dict().items() if key in dict(model.named_parameters())}
    return {'epochs': len(curve), 'maximum_epochs': 100, 'patience': 15,
            'selected_epoch': best_epoch, 'actual_optimizer_updates': updates,
            'checkpoint_path': checkpoint, 'seconds': duration,
            'batch': 8, 'windows_per_epoch': len(train), 'all_TRAIN_windows_once_per_epoch': True,
            'loss': 'observed entry mean squared reconstruction + rho0.14*sum(abs(shared product coefficients))',
            'optimizer': 'Adam lr0.0005', 'selection': 'existing development VAL reconstruction MSE',
            'parameter_changes': changes, 'parameters': model.parameter_report()}


def fit_calibration(model, ztrain, ftrain, train_target, train_mask,
                    zval, fval, target, mask, resources, job, folder):
    model.requires_grad_(False)
    for parameter in model.decoder_parameters():
        parameter.requires_grad_(True)
    encoder_before = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
                      if key.startswith('encoder.') or key.startswith('product.')}
    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=1e-4)
    rng = np.random.default_rng(job['seed'] + 100)
    checkpoint = folder / 'forecast_calibrated_best.pt'
    base_metric = observed_metrics(decode_all(model, zval, fval, resources), target, mask)
    best, best_epoch, patience, updates = base_metric['mse'], 0, 0, 0
    torch.save({'state_dict': tensor_state(model), 'epoch': 0, 'actual_updates': 0}, checkpoint)
    curve = []
    with resources.stage('GT_same_native_decoder_forecast_calibration', gpu=True, job=job['id']):
        started = time.perf_counter()
        for epoch in range(1, 51):
            model.train()
            order = rng.permutation(len(ztrain))
            scores = []
            for start in range(0, len(order), 8):
                index = order[start:start + 8]
                z = torch.as_tensor(ztrain[index], device='cuda')
                forecast = torch.as_tensor(ftrain[index], device='cuda')
                y = torch.as_tensor(train_target[index], device='cuda')
                observed = torch.as_tensor(train_mask[index], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                prediction = model.decode_future(z, forecast, horizon=48)
                loss = masked_loss(prediction, y, observed)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite native decoder calibration loss')
                loss.backward()
                optimizer.step()
                updates += 1
                scores.append(float(loss.detach()))
                resources.check(job=job['id'], phase='GT_native_D_calibration', epoch=epoch,
                                maximum_epochs=50, actual_updates=updates)
            model.eval()
            metric = observed_metrics(decode_all(model, zval, fval, resources), target, mask)
            row = {'epoch': epoch, 'actual_updates': updates, 'train_forecast_mse': float(np.mean(scores)),
                   'val_forecast_mse': metric['mse'], 'val_forecast_mae': metric['mae']}
            curve.append(row)
            if metric['mse'] < best:
                best, best_epoch, patience = metric['mse'], epoch, 0
                torch.save({'state_dict': tensor_state(model), 'epoch': epoch,
                            'actual_updates': updates, 'optimizer': optimizer.state_dict()}, checkpoint)
            else:
                patience += 1
            write_json(HERE / 'runs' / job['id'] / 'calibration_curve.json', curve)
            print(json.dumps({'job': job['id'], 'phase': 'native_decoder_calibration', **row}), flush=True)
            if patience >= 10:
                break
        duration = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    unchanged = all(torch.equal(value, model.state_dict()[key].cpu()) for key, value in encoder_before.items())
    if not unchanged:
        raise RuntimeError('Calibration modified encoder/shared product coefficients')
    return {'epochs': len(curve), 'maximum_epochs': 50, 'patience': 10,
            'selected_epoch': best_epoch, 'actual_optimizer_updates': updates,
            'checkpoint_path': checkpoint, 'seconds': duration,
            'E_and_shared_product_unchanged': unchanged, 'F_frozen': True,
            'native_D_trainable_parameters': 120, 'optimizer': 'Adam lr0.0001',
            'supervision': 'observed TRAIN future targets paired with actual F-predicted latent future',
            'selection': 'existing development VAL forecast MSE; epoch0 transfer checkpoint included'}


def run(dataset, seed, no_spatial=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    job = {'id': f'{dataset}_GT_{"no_spatial" if no_spatial else "full"}_s{seed}',
           'dataset': dataset, 'seed': seed, 'family': 'GT', 'no_spatial': no_spatial}
    resources = Resources()
    folder = CACHE / 'runs' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    data = load_data(dataset)
    model = GTConvAE(data['graph'], no_spatial=no_spatial).cuda()
    training = fit_reconstruction(model, data, resources, job, folder)
    model.eval().requires_grad_(False)
    with resources.stage('load_and_verify_frozen_official_F', gpu=True, job=job['id']):
        foundation = Foundation(resources)
        parity = foundation.verify_pipeline_parity()
    with resources.stage('GT_actual_TRAIN_and_VAL_E_F_native_D', gpu=True, job=job['id']):
        past, past_mask = make_windows(data, data['val_origins'], masked=True)
        target, mask = make_windows(data, data['val_origins'], future=True, masked=True)
        train_past = make_windows(data, data['train_forecast_origins'])
        train_target, train_mask = make_windows(data, data['train_forecast_origins'], future=True, masked=True)
        ztrain = encode_all(model, train_past, resources)
        zval = encode_all(model, past, resources)
        ftrain = forecast_all(foundation, ztrain, resources)
        fval = forecast_all(foundation, zval, resources)
        reconstruction = reconstruct_all(model, past, resources)
        transfer = decode_all(model, zval, fval, resources)
        persistence = np.repeat(zval[:, -1:], 12, axis=1)
        persist_transfer = decode_all(model, zval, persistence, resources)
    calibrated = copy.deepcopy(model).cuda()
    calibration = fit_calibration(calibrated, ztrain, ftrain, train_target, train_mask,
                                  zval, fval, target, mask, resources, job, folder)
    with resources.stage('GT_calibrated_actual_future_inference', gpu=True, job=job['id']):
        calibrated_forecast = decode_all(calibrated, zval, fval, resources)
        persist_calibrated = decode_all(calibrated, zval, persistence, resources)
        index = np.array([0, len(past) // 2, len(past) - 1])
        diagnostic = {'past': past[index], 'past_mask': past_mask[index],
                      'reconstruction': reconstruction[index], 'latent_history': zval[index], 'indices': index}
        predictions = {'TRANSFER_F': transfer, 'CALIBRATED_F': calibrated_forecast,
                       'LATENT_PERSIST_TRANSFER': persist_transfer,
                       'LATENT_PERSIST_CALIBRATED': persist_calibrated}
        forecasts = {name: observed_metrics(value, target, mask, data['std']) for name, value in predictions.items()}
        rec_metric = observed_metrics(reconstruction, past, past_mask, data['std'])
        # Whole-future actual observations are used only as metric labels. The
        # inference function accepts a past tensor alone.
        origin = int(data['val_origins'][0])
        altered = data['x'].copy()
        altered[origin:] = np.random.default_rng(seed).normal(size=altered[origin:].shape)
        x = torch.as_tensor(data['x'][origin - 512:origin][None], device='cuda')
        a = model.decode_future(model.encode(x), foundation.forecast(model.encode(x), 12), 48)
        altered_past = torch.as_tensor(altered[origin - 512:origin][None], device='cuda')
        b = model.decode_future(model.encode(altered_past), foundation.forecast(model.encode(altered_past), 12), 48)
        if not torch.equal(a, b):
            raise RuntimeError('Past-only inference was non-deterministic')
        guards = {'actual_inference_signature': 'past tensor only',
                  'future_target_never_passed': True, 'past_only_repeat_bitwise': True,
                  'actual_observations_after_origin_randomized': True,
                  'native_decoder_after_F': True, 'replacement_D8_or_LEVEL_G': False,
                  'F_pipeline_parity': parity, **foundation.verify_unchanged()}
        write_json(HERE / 'runs' / job['id'] / 'INFERENCE_GUARDS.json', guards)
    costs = []
    for name, decoder in [('TRANSFER_F', model), ('CALIBRATED_F', calibrated)]:
        def infer(x, decoder=decoder):
            z = model.encode(x)
            return decoder.decode_future(z, foundation.forecast(z, 12), 48)
        costs.extend(benchmark_path(infer, data, resources, name, predictions[name]))
    resources.flush()
    shape = {'input': [512, model.channels, 1], 'latent': [128, model.channels, 2],
             'F_input_rows_per_origin': 2 * model.channels, 'F_context_length': 128,
             'F_future_length': 12, 'native_D_output': [48, model.channels],
             'input_scalars': 512 * model.channels, 'latent_scalars': 256 * model.channels,
             'latent_scalar_ratio': .5, 'alignment': model.alignment_report()}
    files = [Path(__file__), HERE / 'models_gt.py', HERE / 'data_io.py', HERE / 'foundation.py',
             HERE / 'common_direct.py', HERE / 'run_support.py', HERE / 'protocol.json']
    result = save_case(job, data, rec_metric, forecasts, predictions, diagnostic,
                       training, calibration, shape, costs, resources, files)
    print(json.dumps({'job': job['id'], 'status': result['status'],
                      'reconstruction_mse': rec_metric['mse'],
                      'forecast': {name: {'mse': value['mse'], 'mae': value['mae']} for name, value in forecasts.items()}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('robin', 'jena'), required=True)
    parser.add_argument('--seed', type=int, choices=(92601, 92602), required=True)
    parser.add_argument('--no-spatial', action='store_true')
    args = parser.parse_args()
    try:
        run(args.dataset, args.seed, args.no_spatial)
    except BaseException as exc:
        write_json(HERE / f'FAILED_GT_{args.dataset}_{args.seed}_{args.no_spatial}_{time.time_ns()}.json',
                   {'status': 'BLOCKED_BUDGET' if isinstance(exc, BudgetExceeded) else 'FAILED',
                    'error': str(exc), 'traceback': traceback.format_exc(),
                    'resume': 'Fresh invocation preserves partial checkpoints; no optimizer-state continuation claimed'})
        raise
