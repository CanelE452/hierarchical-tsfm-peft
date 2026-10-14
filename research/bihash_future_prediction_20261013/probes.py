"""Frozen-representation readouts and matched raw forecasting controls."""
import time
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
import model as bm
import train as tr

ALPHAS = (.1, 1., 10., 100.)


def fit_grouped_ridge(fit_features, fit_target, fit_observed, tune_features, tune_target, tune_observed, limit=None):
    """Observed-target fits; shared FIT feature scaling, target-wise valid intercept."""
    started = time.perf_counter()
    feature_mean = np.asarray(fit_features, np.float64).mean(0)
    feature_std = np.asarray(fit_features, np.float64).std(0)
    feature_std[feature_std < 1e-6] = 1.
    x = (np.asarray(fit_features, np.float64) - feature_mean) / feature_std
    tune = (np.asarray(tune_features, np.float64) - feature_mean) / feature_std
    y = np.asarray(fit_target, np.float64).reshape(len(x), -1)
    mask = np.asarray(fit_observed, bool).reshape(y.shape)
    groups = {}
    for column in range(y.shape[1]): groups.setdefault(mask[:, column].tobytes(), []).append(column)
    candidates = []
    for alpha in ALPHAS:
        weights = np.zeros((x.shape[1], y.shape[1]), np.float64)
        biases = np.zeros(y.shape[1], np.float64)
        for columns in groups.values():
            valid = mask[:, columns[0]]
            if not valid.any(): continue
            xv = x[valid]
            xm = xv.mean(0); ym = y[valid][:, columns].mean(0)
            centered = xv - xm
            factor = cho_factor(centered @ centered.T + alpha * np.eye(len(centered)), lower=True, check_finite=False)
            dual = cho_solve(factor, y[valid][:, columns] - ym, check_finite=False)
            coefficient = centered.T @ dual
            weights[:, columns] = coefficient
            biases[columns] = ym - xm @ coefficient
            if limit is not None: limit.check()
        prediction = (tune @ weights + biases).reshape(tune_target.shape).astype(np.float32)
        metric = bm.en.ex.score(prediction, tune_target, tune_observed)
        candidates.append({'alpha': alpha, 'tune_mse': metric['mse'], 'weights': weights, 'biases': biases})
    selected = min(candidates, key=lambda c: (c['tune_mse'], c['alpha']))
    # Fold feature scaling into the effective affine layer used in inference.
    weight = selected['weights'] / feature_std[:, None]
    bias = selected['biases'] - feature_mean @ weight
    model = {'weight': weight.astype(np.float32), 'bias': bias.astype(np.float32), 'horizon': np.asarray(fit_target.shape[1], np.int64),
        'channels': np.asarray(fit_target.shape[-1], np.int64)}
    record = {'alpha': selected['alpha'], 'candidates': [{'alpha': c['alpha'], 'tune_norm_mse': c['tune_mse']} for c in candidates],
        'feature_dimensions': x.shape[1], 'mask_groups': len(groups), 'FIT_feature_mean': feature_mean.tolist(),
        'FIT_feature_std': feature_std.tolist(), 'regularizer': 'sum squared observed target errors + alpha times squared coefficient norm',
        'intercept': 'per-target FIT observed rows; all-unobserved targets use zero delta',
        'selection': 'TUNE observed normalized channel-macro MSE; smallest alpha exact tie',
        'wall_seconds': time.perf_counter() - started}
    return model, record


def ridge_predict(features, model):
    return (np.asarray(features, np.float32) @ model['weight'] + model['bias']).reshape(-1, model['horizon'], model['channels'])


def block_features(x):
    return np.concatenate((x.reshape(len(x), 32, 16, x.shape[-1]).mean(2), x[:, -1:]), axis=1).reshape(len(x), -1)


def representation_features(model, history, origins, enriched, limit=None):
    model.eval(); model.freeze_memory()
    device = next(model.parameters()).device
    result = []
    with torch.no_grad():
        for start in range(0, len(history), 8):
            x = torch.as_tensor(history[start:start + 8], dtype=torch.float32, device=device)
            if enriched:
                z, _ = model.encode_with_memory(x, torch.as_tensor(origins[start:start + 8], device=device))
            else:
                z = model.codec.encode(x)
            value = torch.cat((z.flatten(1), x[:, -1]), dim=1).cpu().numpy()
            result.append(value)
            if limit is not None: limit.check()
    return np.concatenate(result)


def fit_representation_readout(pack, case, model, folder, rt, limit):
    case_record = {'dataset': pack['dataset'], 'seed': case['seed'], 'source_method': case['method'],
        'source_checkpoint': case['training']['selected_checkpoint'], 'readouts': [],
        'limitation': 'Encoder and decoder previously optimized through Bolt; frozen readout does not establish predictor-independent encoder quality'}
    delta = {s: pack['target'][s] - pack['history'][s][:, -1:] for s in ('fit', 'tune')}
    fit_features = {s: representation_features(model, pack['history'][s], pack['origins'][s], False, limit) for s in ('fit', 'tune')}
    for enriched in (False, True):
        name = 'E_ENRICHED_RIDGE' if enriched else 'E_RAW_RIDGE'
        duplicate = enriched and not case['method'].startswith('B_')
        if duplicate:
            original = case_record['readouts'][0]
            case_record['readouts'].append({**original, 'method': name, 'duplicate_of': 'E_RAW_RIDGE', 'additional_fit': False})
            continue
        features = fit_features if not enriched else {s: representation_features(model, pack['history'][s], pack['origins'][s], True, limit) for s in ('fit', 'tune')}
        ridge, selection = fit_grouped_ridge(features['fit'], delta['fit'], pack['observed']['fit'], features['tune'], delta['tune'], pack['observed']['tune'], limit)
        prediction = {s: pack['history'][s][:, -1:] + ridge_predict(features[s], ridge) for s in features}
        artifact = rt.save_npz(folder / (name + '_weights.npz'), **ridge)
        case_record['readouts'].append({'method': name, 'additional_fit': True, 'model': artifact, 'selection': selection,
            'prediction': tr.prediction_artifact(rt, folder / (name + '_FIT_TUNE.npz'), pack, prediction),
            'scores': {s: tr.scores(pack, p, s) for s, p in prediction.items()},
            'features': 'flattened 512 K continuous history + known last C; intercept/FIT scaling; forecast delta with original D removed'})
    return case_record


def evaluate_readout(pack, split, case, model, readout, rt, limit):
    path = rt.ROOT / readout['model']['path']
    if rt.sha(path) != readout['model']['sha256']: raise ValueError('READOUT_MODEL_SHA')
    with np.load(path, allow_pickle=False) as z: ridge = {k: z[k] for k in z.files}
    features = representation_features(model, pack['history'][split], pack['origins'][split], readout['method'] == 'E_ENRICHED_RIDGE', limit)
    return pack['history'][split][:, -1:] + ridge_predict(features, ridge)


def fit_raw_controls(pack, forecaster, folder, rt, limit):
    predictions = {}
    for split in ('fit', 'tune'):
        rows = []
        with torch.no_grad():
            for begin in range(0, len(pack['history'][split]), 8):
                x = torch.as_tensor(pack['history'][split][begin:begin + 8], dtype=torch.float32, device='cuda')
                rows.append(forecaster(x).cpu().numpy()); limit.check()
        predictions[split] = np.concatenate(rows)
    controls = [{'method': 'RAW_BOLT', 'prediction': tr.prediction_artifact(rt, folder / 'RAW_BOLT_FIT_TUNE.npz', pack, predictions),
        'scores': {s: tr.scores(pack, p, s) for s, p in predictions.items()}}]
    features = {s: block_features(pack['history'][s]) for s in ('fit', 'tune')}
    for name in ('RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R'):
        base = predictions if name == 'RAW_BOLT_RAW_R' else {s: pack['history'][s][:, -1:] for s in ('fit', 'tune')}
        residual = {s: pack['target'][s] - base[s] for s in ('fit', 'tune')}
        ridge, selection = fit_grouped_ridge(features['fit'], residual['fit'], pack['observed']['fit'], features['tune'], residual['tune'], pack['observed']['tune'], limit)
        forecast = {s: base[s] + ridge_predict(features[s], ridge) for s in ('fit', 'tune')}
        controls.append({'method': name, 'model': rt.save_npz(folder / (name + '_weights.npz'), **ridge), 'selection': selection,
            'prediction': tr.prediction_artifact(rt, folder / (name + '_FIT_TUNE.npz'), pack, forecast),
            'scores': {s: tr.scores(pack, p, s) for s, p in forecast.items()},
            'features': '32 nonoverlapping historical block means + last C; normalized data; all channels jointly'})
    self_weights, self_bias = [], []
    self_predictions = {s: np.zeros_like(pack['target'][s]) for s in ('fit', 'tune')}
    self_selections = []
    c = len(pack['mean'])
    for channel in range(c):
        one = {s: block_features(pack['history'][s][:, :, channel:channel + 1]) for s in ('fit', 'tune')}
        delta = {s: pack['target'][s][:, :, channel:channel + 1] - pack['history'][s][:, -1:, channel:channel + 1] for s in ('fit', 'tune')}
        mask = {s: pack['observed'][s][:, :, channel:channel + 1] for s in ('fit', 'tune')}
        ridge, selection = fit_grouped_ridge(one['fit'], delta['fit'], mask['fit'], one['tune'], delta['tune'], mask['tune'], limit)
        self_weights.append(ridge['weight']); self_bias.append(ridge['bias']); self_selections.append(selection)
        for split in ('fit', 'tune'):
            self_predictions[split][:, :, channel:channel + 1] = pack['history'][split][:, -1:, channel:channel + 1] + ridge_predict(one[split], ridge)
    controls.append({'method': 'RAW_SELF33', 'model': rt.save_npz(folder / 'RAW_SELF33_weights.npz', weight=np.stack(self_weights), bias=np.stack(self_bias)),
        'channel_selections': self_selections, 'prediction': tr.prediction_artifact(rt, folder / 'RAW_SELF33_FIT_TUNE.npz', pack, self_predictions),
        'scores': {s: tr.scores(pack, p, s) for s, p in self_predictions.items()}, 'features': 'Each channel independent: 32 historical block means + last value; forecast delta'})
    return {'dataset': pack['dataset'], 'label': 'RAW_CONTROLS', 'methods': controls}


def evaluate_raw_control(pack, split, control, forecaster, rt, limit):
    name = control['method']
    history = pack['history'][split]
    raw = None
    if name in ('RAW_BOLT', 'RAW_BOLT_RAW_R'):
        rows = []
        with torch.no_grad():
            for begin in range(0, len(history), 8):
                rows.append(forecaster(torch.as_tensor(history[begin:begin + 8], dtype=torch.float32, device='cuda')).cpu().numpy()); limit.check()
        raw = np.concatenate(rows)
    if name == 'RAW_BOLT': return raw
    path = rt.ROOT / control['model']['path']
    if rt.sha(path) != control['model']['sha256']: raise ValueError('RAW_CONTROL_MODEL_SHA')
    with np.load(path, allow_pickle=False) as z: ridge = {k: z[k] for k in z.files}
    if name == 'RAW_SELF33':
        prediction = np.empty((len(history), 48, history.shape[-1]), np.float32)
        for channel in range(history.shape[-1]):
            one = {'weight': ridge['weight'][channel], 'bias': ridge['bias'][channel], 'horizon': 48, 'channels': 1}
            prediction[:, :, channel:channel + 1] = history[:, -1:, channel:channel + 1] + ridge_predict(block_features(history[:, :, channel:channel + 1]), one)
        return prediction
    ridge.update(horizon=48, channels=history.shape[-1])
    return (raw if raw is not None else history[:, -1:]) + ridge_predict(block_features(history), ridge)
