"""A synthetic PCA/median counterexample; not a dataset performance result."""
import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
import runtime as rt


def weighted_quantile(values, probability, level=.5):
    order = np.argsort(values, kind='stable')
    return float(values[order[np.searchsorted(np.cumsum(probability[order]), level)]])


def score(point, outcomes, probability):
    errors = outcomes - point
    return {'mse': float((probability[:, None] * errors**2).sum(0).mean()),
            'mae': float((probability[:, None] * np.abs(errors)).sum(0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('INVALID_FRESH_TAG')
    folder = rt.HERE / 'decision_checks_v1' / ('quantile_projection_' + args.tag)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    limit = rt.Limit('quantile_projection_' + args.tag, gpu=False)
    probability = np.array([.15, .2, .2, .45], np.float64)
    outcomes = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]], np.float64)
    expectation = probability @ outcomes
    centered = outcomes - expectation
    covariance = centered.T @ (probability[:, None] * centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, -1]
    if basis[np.argmax(np.abs(basis))] < 0:
        basis = -basis
    projected = outcomes @ basis
    raw_median = np.array([weighted_quantile(outcomes[:, c], probability) for c in range(2)])
    projected_median = weighted_quantile(projected, probability)
    decoded_median = projected_median * basis
    projected_raw_median = float(raw_median @ basis)
    projection_of_raw_forecast = projected_raw_median * basis
    checks = {
        'probability_sums_to_one': bool(np.isclose(probability.sum(), 1.)),
        'basis_is_actual_top_PCA_eigenvector': bool(np.allclose(covariance @ basis, eigenvalues[-1] * basis)),
        'raw_median_unique_ten': bool(np.array_equal(raw_median, [10., 10.])),
        'projected_median_unique': bool(np.isclose(probability[projected < projected_median].sum(), .15)
                                      and np.isclose(probability[projected <= projected_median].sum(), .55)),
        'raw_median_forecast_represented_exactly_by_rank_one': bool(np.allclose(projection_of_raw_forecast, raw_median)),
        'median_and_PCA_projection_do_not_commute': bool(not np.isclose(projected_median, projected_raw_median)),
        'decoded_latent_median_is_five': bool(np.allclose(decoded_median, [5., 5.])),
        'conditional_mean_commutes_with_this_linear_projection': bool(np.isclose(probability @ projected, expectation @ basis)),
        'MSE_MAE_conflict_exists': bool(score(decoded_median, outcomes, probability)['mse'] < score(raw_median, outcomes, probability)['mse']
                                      and score(decoded_median, outcomes, probability)['mae'] > score(raw_median, outcomes, probability)['mae']),
    }
    wrapper = rt.ROOT / 'research/forecast_component_study_20261013/component_experiment.py'
    installed = Path(importlib.util.find_spec('chronos').origin).parent / 'chronos_bolt.py'
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'started_unix': started,
              'finished_unix': time.time(), 'checks': checks, 'outcomes': outcomes,
              'probability': probability, 'covariance': covariance, 'eigenvalues': eigenvalues,
              'PCA_basis': basis, 'raw_channel_medians': raw_median,
              'latent_median': projected_median, 'projection_of_raw_medians': projected_raw_median,
              'decoded_latent_median': decoded_median,
              'raw_median_error': score(raw_median, outcomes, probability),
              'decoded_latent_median_error': score(decoded_median, outcomes, probability),
              'raw_conditional_mean_error': score(expectation, outcomes, probability),
              'local_point_forecast_wrapper': {'path': str(wrapper.relative_to(rt.ROOT)), 'sha256': rt.sha(wrapper),
                                              'verified_source': 'FrozenBolt selects quantile .5, not conditional mean.'},
              'installed_chronos_source': {'path': str(installed), 'sha256': rt.sha(installed)},
              'validator_source_sha256': rt.sha(__file__),
              'scope': 'Synthetic exact distribution and top PCA basis; no actual dataset inference or model training.',
              'actual_dataset_cause_verified': False, 'fundamental_nonrecoverability_proved': False,
              'learning_or_decoder_correction_can_still_help': True, 'goal_complete': False}
    limit.check()
    result['resource_receipt'] = limit.close('COMPLETE')
    rt.write(folder / 'RESULT.json', result)
    report = '\n'.join([
        '# PCA와 분위수 예측의 변환 순서: 합성 반례', '',
        '[확인] 실제 로컬 FrozenBolt wrapper는 0.5 분위수를 선택한다. 아래는 실제 자료의 실험이 아닌 네 가지 미래 상태의 정확한 확률 계산이다.', '',
        '상태 (0,0)/(10,0)/(0,10)/(10,10)의 확률은 .15/.20/.20/.45다. 채널별 median은 (10,10)이고, 이 분포의 실제 최상위 PCA 방향은 (1,1)/sqrt(2)다.', '',
        'RAW median forecast는 rank1 공간으로 정확히 표현된다. 하지만 PCA로 투영한 미래 분포의 median을 선형 복원하면 (5,5)가 된다. 따라서 RAW forecast의 낮은 rank만으로 압축된 과거의 TSFM forecast가 같은 값을 낼 것이라고 가정할 수 없다.', '',
        '```text', '예측                         MSE       MAE',
        '원채널별 median (10,10)      35.00      3.50',
        '잠재 median 복원 (5,5)       25.00      5.00',
        '원채널별 mean (6.5,6.5)      22.75      4.55', '```', '',
        '[판단] 이 반례는 압축된 분위수의 복원이 MSE를 낮추면서 MAE를 높일 수 있음을 보여준다. 실제 Robin/Beijing/Appliances의 원인을 확정하지 않는다. 학습된 decoder가 이를 보정할 가능성도 있으므로 방법의 보편적 한계나 복원 불가능성을 입증한 것은 아니다.', '',
        '후속 D1에서는 RAW forecast 투영과 투영된 과거의 실제 Bolt forecast를 분리한다. 평균/분위수 목적의 차이는 실제 FIT/TUNE 선택과 동일 baseline 비용을 맞춰 검증해야 한다.', '',
        '[원수치와 소스 SHA](RESULT.json) · [로컬 구현 근거](../../../forecast_component_study_20261013/component_experiment.py)', '',
    ])
    with (folder / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
        handle.write(report)
    print(rt.json_line({'result': str(folder / 'RESULT.json'), 'status': result['status'],
                        'checks': len(checks), 'real_dataset_opened': False}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
