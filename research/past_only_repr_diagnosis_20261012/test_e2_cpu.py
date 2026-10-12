"""Small independent algebra/alignment tests; no real-data science or GPU."""
import time
import numpy as np
from threadpoolctl import threadpool_limits

from common import Budget, HERE, Ridge, access_guard, write
from e2 import feature_block, fit_candidates
from synthetic import make_record, noiseless_alignment


def main():
    access_guard()
    with threadpool_limits(limits=2), Budget(phase=1, gpu=False, label='E2_numerical_unit') as budget:
        checks = []
        alignment = noiseless_alignment()
        checks.append({'check': 'noiseless_direction_alignment', 'passed': alignment['status'] == 'PASS'})
        history = np.arange(2*512*6).reshape(2, 512, 6)
        for model in ('SELF_LAG', 'SELF_LAG_PLUS_CROSS_NOW', 'ALL_CHANNEL_LAG'):
            f = feature_block(history, model, 2)
            if model == 'SELF_LAG':
                expected = np.stack([history[:, -1-lag, 2] for lag in range(33)], 1)
            elif model == 'SELF_LAG_PLUS_CROSS_NOW':
                expected = np.concatenate([np.stack([history[:, -1-lag, 2] for lag in range(33)], 1), history[:, -1, [0, 1, 3, 4, 5]]], 1)
            else:
                expected = np.stack([history[:, -1-lag] for lag in range(33)], 1).reshape(2, -1)
            checks.append({'check': f'{model}_feature_order', 'passed': np.array_equal(f, expected)})
        rng = np.random.default_rng(831)
        for samples, features in ((24, 7), (12, 31)):
            x, y = rng.normal(size=(samples, features)), rng.normal(size=(samples, 3))
            xp = rng.normal(size=(5, features))
            for lam in (1e-4, 1., 1e4):
                fitted = Ridge(lam).fit(x, y)
                z = (x-x.mean(0))/np.maximum(x.std(0), 1e-12)
                coef = np.linalg.solve(z.T@z+samples*lam*np.eye(features), z.T@(y-y.mean(0)))
                expected = ((xp-x.mean(0))/np.maximum(x.std(0), 1e-12))@coef+y.mean(0)
                delta = float(np.max(np.abs(fitted.predict(xp)-expected)))
                checks.append({'check': 'primal_dual_prediction_equivalence', 'samples': samples, 'features': features,
                               'lambda': lam, 'max_absolute_error': delta, 'passed': delta < 1e-9})
        for condition in ('unrelated', 'fixed', 'changed'):
            data, _ = make_record(condition, 92610)
            checks.append({'check': f'{condition}_target_bounds', 'passed': all(
                np.all(oo >= data[f'{s}_bounds'][0]) and np.all(oo+48 <= data[f'{s}_bounds'][1])
                for s, oo in data['origins'].items())})
            noiseless, _ = make_record(condition, 92610, noise=False)
            checks.append({'check': f'{condition}_noiseless_normalization_finite',
                           'passed': np.isfinite(noiseless['x']).all()})
        history = {s: rng.normal(size=(n, 512, 3)) for s, n in (('fit', 16), ('tune', 7), ('val', 8))}
        target = {s: rng.normal(size=(len(x), 48, 3)) for s, x in history.items()}
        valid = {s: np.ones_like(y, dtype=bool) for s, y in target.items()}
        valid['fit'][:3, :, 1] = False
        valid['tune'][:2, :, 2] = False
        changed = {s: y.copy() for s, y in target.items()}
        changed['fit'][~valid['fit']] = 1e8
        for model in ('SELF_LAG', 'SELF_LAG_PLUS_CROSS_NOW', 'ALL_CHANNEL_LAG'):
            pred, info, candidates, _ = fit_candidates(history, target, valid, model, np.array([1., 3., 9.]), budget)
            pred_changed, _, _, _ = fit_candidates(history, changed, valid, model, np.array([1., 3., 9.]), budget)
            delta = float(np.max(np.abs(pred['val']-pred_changed['val'])))
            checks.append({'check': f'{model}_missing_fit_targets_excluded', 'max_absolute_error': delta,
                           'passed': delta == 0.})
            chosen = min(range(5), key=lambda i: info['lambda_grid'][i]['tune_physical_macro_mse'])
            checks.append({'check': f'{model}_TUNE_only_grid_selection',
                           'passed': chosen == info['selected_grid_index'] and np.array_equal(pred['val'], candidates[chosen]['val'])})
        if not all(c['passed'] for c in checks):
            raise AssertionError(checks)
        budget.check()
        path = HERE/'e2'/f'CPU_UNIT_{time.time_ns()}.json'
        write(path, {'status': 'PASS', 'checks': checks, 'budget_receipt': str(budget.path),
                     'scope': 'algebra/feature indexing/synthetic generation only, no real prediction improvement evidence'})
        print(path)


if __name__ == '__main__':
    main()
