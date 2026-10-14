"""SOURCE-ready terminal-JSON-only matched-readout report; no NPZ/model reads.

Actual report generation requires --execute and the fresh CPU runtime gate.
No JSON/source snapshots/output directory/matplotlib is opened before that gate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / 'report_matched_readout_v2'
DATASETS = ('robin', 'beijing_aotizhongxin')
SEEDS = (92601, 92602)
FINAL = {'robin': 'val', 'beijing_aotizhongxin': 'eval'}
DISPLAY = {'robin': 'Robin', 'beijing_aotizhongxin': 'Beijing Aotizhongxin'}
C_K = {'robin': (17, 4), 'beijing_aotizhongxin': (11, 2)}
NEW = ('RAW33C', 'E512K')
METHODS = ('RAW_BOLT', 'ORIGINAL_MAIN', 'E33K_REUSED', 'RAW33C', 'E512K')
LABELS = {'RAW_BOLT': 'RAW-F', 'ORIGINAL_MAIN': 'MAIN', 'E33K_REUSED': 'E33K', 'RAW33C': 'RAW33C', 'E512K': 'E512K'}
COLORS = {'RAW_BOLT': '#65758b', 'ORIGINAL_MAIN': '#a87924', 'E33K_REUSED': '#9370b2', 'RAW33C': '#278f8e', 'E512K': '#3374b6'}
ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10.)
FIGURES = ('01_eval_seed_dots', '02_fit_tune_final', '03_alpha_tune_grid',
           '04_capacity_rank', '05_channel_delta', '06_cpu_cost')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_inputs(producers, records, audit):
    expected = {(d, s) for d in DATASETS for s in SEEDS}
    keys = [(p['dataset'], p['seed']) for p in producers]
    if len(keys) != 4 or set(keys) != expected:
        raise ValueError('EXACT4_UNIQUE_MATCHED_CONDITIONS_REQUIRED')
    if audit.get('status') != 'PASS' or audit.get('failed_count') != 0 or len(audit.get('conditions', [])) != 4:
        raise ValueError('BOUND_INDEPENDENT_AUDIT_PASS_REQUIRED')
    audit_before, audit_after = audit.get('audit_source_before'), audit.get('audit_source_after')
    if not audit_before or audit_before != audit_after or sha(audit_before['path']) != audit_before['sha256']:
        raise ValueError('AUDIT_EXECUTION_SOURCE_SHA_BINDING_REQUIRED')
    audit_keys = [(r['dataset'], r['seed']) for r in audit['conditions']]
    if set(audit_keys) != expected:
        raise ValueError('AUDIT_EXACT4_UNIQUE_CONDITIONS_REQUIRED')
    for p, record in zip(producers, records):
        if p.get('status') != 'COMPLETE' or p.get('future_oracle_read') is not False or p.get('GPU_execution') is not False:
            raise ValueError('COMPLETE_PAST_ONLY_CPU_PRODUCER_REQUIRED')
        if p.get('independent_holdout') is not False or p.get('old_N2_retrained') is not False:
            raise ValueError('MATCHED_REUSE_SCOPE_FLAGS_REQUIRED')
        bound = next(r for r in audit['conditions'] if (r['dataset'], r['seed']) == (p['dataset'], p['seed']))
        if bound.get('status') != 'PASS' or bound['producer_sha256'].lower() != record['sha256']:
            raise ValueError('AUDIT_EXACT_PRODUCER_SHA_BINDING_REQUIRED')
        for route in ('RAW33C', 'E512K', 'E33K_REUSED'):
            row = p['families'][route]
            if row.get('status') != 'COMPLETE':
                raise ValueError('INCOMPLETE_ROUTE_RETAINED:' + route)
            if route in NEW:
                entries = row['alpha_grid']
                if tuple(e['alpha'] for e in entries) != ALPHAS:
                    raise ValueError('EXACT_SEVEN_ALPHA_REQUIRED')
                scores = [e['tune_actual_Y_mse'] for e in entries]
                if any(v is None or not math.isfinite(v) for v in scores):
                    raise ValueError('FINITE_TUNE_SELECTION_REQUIRED')
                index = min(range(7), key=lambda i: scores[i])
                if row['selected_index'] != index or row['selected_alpha'] != ALPHAS[index]:
                    raise ValueError('EARLIEST_ACTUAL_Y_TUNE_SELECTION_REQUIRED')
    return {'passed': True, 'conditions': 4, 'producer_SHAs_bound': True, 'audit_execution_source': audit_before,
            'new_candidates': sum(len(p['families'][r]['alpha_grid']) for p in producers for r in NEW)}


def method_scores(producer, method, split):
    return (producer['references'][split][method] if method in ('RAW_BOLT', 'ORIGINAL_MAIN')
            else producer['families'][method]['splits'][split])


def metric_rows(producers):
    rows = []
    for p in producers:
        for method in METHODS:
            for split in ('fit', 'tune', FINAL[p['dataset']]):
                score = method_scores(p, method, split)
                for scale in ('normalized', 'physical_per_channel_diagnostic'):
                    value = score[scale]
                    rows.append({'dataset': p['dataset'], 'seed': p['seed'], 'method': method,
                        'split': split, 'split_role': {'fit': 'training', 'tune': 'selection', 'val': 'reused_development', 'eval': 'exposed_evaluation'}[split],
                        'metric_scale': scale, 'primary_scale': scale == 'normalized',
                        'mse': value['mse'], 'mae': value['mae'], 'active_channels': value['active_channels'],
                        'observed_counts': value['observed_counts'], 'per_channel_mse': value['per_channel_mse'],
                        'per_channel_mae': value['per_channel_mae'], 'new_training': method in NEW,
                        'physical_macro_comparable_units_guaranteed': False})
    return rows


def alpha_rows(producers):
    rows = []
    for p in producers:
        for route in NEW:
            family = p['families'][route]
            for index, entry in enumerate(family['alpha_grid']):
                rows.append({'dataset': p['dataset'], 'seed': p['seed'], 'method': route,
                    'alpha': entry['alpha'], 'selected': index == family['selected_index'],
                    'fit_normalized_mse': entry['fit_scores']['normalized']['mse'],
                    'fit_normalized_mae': entry['fit_scores']['normalized']['mae'],
                    'tune_normalized_mse': entry['tune_scores']['normalized']['mse'],
                    'tune_normalized_mae': entry['tune_scores']['normalized']['mae'],
                    'fit_physical_diagnostic_mse': entry['fit_scores']['physical_per_channel_diagnostic']['mse'],
                    'fit_physical_diagnostic_mae': entry['fit_scores']['physical_per_channel_diagnostic']['mae'],
                    'tune_physical_diagnostic_mse': entry['tune_scores']['physical_per_channel_diagnostic']['mse'],
                    'tune_physical_diagnostic_mae': entry['tune_scores']['physical_per_channel_diagnostic']['mae'],
                    'fit_observed_counts': entry['fit_scores']['normalized']['observed_counts'],
                    'tune_observed_counts': entry['tune_scores']['normalized']['observed_counts'],
                    'model_sha256': entry['model_sha256'], 'apply_inference_save_wall_seconds': entry['FIT_apply_inference_save_wall_seconds']})
    return rows


def capacity_rows(producers):
    rows = []
    for p in producers:
        c, k = C_K[p['dataset']]
        for route in ('E33K_REUSED', 'RAW33C', 'E512K'):
            family = p['families'][route]
            old = route == 'E33K_REUSED'
            meta = family.get('selected_model_metadata', {})
            feature_count = 33*k if old else family['feature_state']['feature_count']
            rows.append({'dataset': p['dataset'], 'seed': p['seed'], 'method': route, 'C': c, 'K': k,
                'p': feature_count, 'registered_coefficients': feature_count*48*c,
                'registered_intercepts': 48*c, 'trained_coefficients': meta.get('trained_coefficient_count'),
                'rank_min': meta.get('design_rank_min'), 'rank_max': meta.get('design_rank_max'),
                'factorization_groups': family.get('preparation', {}).get('factorization_groups'),
                'selected_alpha': family['selected_alpha'], 'new_training': not old,
                'old_rank_not_read_from_NPZ': old, 'same_alpha_means_same_capacity': False})
    return rows


def cost_rows(producers, receipts):
    rows = []
    for p, receipt in zip(producers, receipts):
        for route in ('RAW33C', 'E512K', 'E33K_REUSED'):
            family = p['families'][route]
            rows.append({'dataset': p['dataset'], 'seed': p['seed'], 'method': route,
                'new_training': route in NEW,
                'new_FIT_factorization_wall_seconds': family.get('new_FIT_factorization_wall_seconds'),
                'family_including_apply_save_guards_wall_seconds': family.get('total_family_wall_seconds'),
                'original_historical_training_wall_seconds': family.get('historical_training_wall_seconds'),
                'producer_body_wall_seconds': p['body_wall_seconds'], 'producer_process_wall_seconds': receipt['wall_seconds'],
                'new_deployment_latency_measured': False, 'GPU_kernel_time': False,
                'old_training_unknown_not_zero': route == 'E33K_REUSED'})
    return rows


def summary_values(producers):
    summary = {}
    for dataset in DATASETS:
        subset = [p for p in producers if p['dataset'] == dataset]
        summary[dataset] = {method: {metric: statistics.mean(method_scores(p, method, FINAL[dataset])['normalized'][metric]
                                 for p in subset) for metric in ('mse', 'mae')} for method in METHODS}
    return summary


def figures(producers, capacities, costs, pyplot, output, save):
    def finish(fig, stem, note):
        fig.text(.01, -.035, note, fontsize=8, color='#444444')
        for suffix in ('png', 'svg'):
            save(fig, output/'figures'/(stem+'.'+suffix))
        pyplot.close(fig)
    fig, axes = pyplot.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row, dataset in enumerate(DATASETS):
        subset = [p for p in producers if p['dataset'] == dataset]
        for col, metric in enumerate(('mse', 'mae')):
            ax = axes[row, col]
            for index, method in enumerate(METHODS):
                values = [method_scores(p, method, FINAL[dataset])['normalized'][metric] for p in subset]
                ax.bar(index, statistics.mean(values), color=COLORS[method], alpha=.55, width=.6)
                ax.scatter([index-.1, index+.1], values, color=COLORS[method], edgecolors='black', zorder=3)
                ax.text(index, max(values), f'{statistics.mean(values):.4f}', ha='center', va='bottom', fontsize=9)
            ax.set_xticks(range(5), [LABELS[m] for m in METHODS])
            ax.set_title(DISPLAY[dataset]+' — '+FINAL[dataset].upper()+' '+metric.upper())
            ax.set_ylabel('Normalized channel-macro '+metric.upper()); ax.grid(axis='y', alpha=.2)
    finish(fig, FIGURES[0], 'Dots: original E seeds92601/92602. RAW-F/RAW33C repeat identical input; these are not independent datasets.')

    fig, axes = pyplot.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row, dataset in enumerate(DATASETS):
        subset = [p for p in producers if p['dataset'] == dataset]
        splits = ('fit', 'tune', FINAL[dataset])
        for col, metric in enumerate(('mse', 'mae')):
            ax = axes[row, col]
            for method in METHODS:
                values = [statistics.mean(method_scores(p, method, split)['normalized'][metric] for p in subset) for split in splits]
                ax.plot(range(3), values, '-o', label=LABELS[method], color=COLORS[method])
            ax.set_xticks(range(3), [s.upper() for s in splits]); ax.set_title(DISPLAY[dataset]+' '+metric.upper())
            ax.set_ylabel('Normalized channel-macro '+metric.upper()); ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=3)
    finish(fig, FIGURES[1], 'FIT trains coefficients; TUNE selects alpha; only selected models predict reused VAL / exposed EVAL.')

    fig, axes = pyplot.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for p, ax in zip(sorted(producers, key=lambda p:(DATASETS.index(p['dataset']), p['seed'])), axes.flat):
        for route in NEW:
            row = p['families'][route]
            values = [e['tune_actual_Y_mse'] for e in row['alpha_grid']]
            ax.plot(range(7), values, '-o', label=route, color=COLORS[route])
            ax.scatter(row['selected_index'], values[row['selected_index']], marker='*', s=160,
                       color=COLORS[route], edgecolors='black', zorder=4)
        ax.set_xticks(range(7), ['0', '1e-4', '.001', '.01', '.1', '1', '10'])
        ax.set_xlabel('Fixed alpha candidates'); ax.set_ylabel('TUNE normalized MSE')
        ax.set_yscale('log')
        ax.set_title(DISPLAY[p['dataset']]+' seed'+str(p['seed'])); ax.legend(); ax.grid(alpha=.2)
    finish(fig, FIGURES[2], 'Star: lowest actual-Y TUNE MSE, earliest exact tie. Same alpha does not guarantee matched regularization/capacity.')

    fig, axes = pyplot.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for index, dataset in enumerate(DATASETS):
        subset = [r for r in capacities if r['dataset'] == dataset]
        methods = ('E33K_REUSED', 'RAW33C', 'E512K')
        pvalues = [next(r['p'] for r in subset if r['method'] == m) for m in methods]
        coefs = [next(r['registered_coefficients'] for r in subset if r['method'] == m) for m in methods]
        axes[index, 0].bar(range(3), pvalues, color=[COLORS[m] for m in methods])
        axes[index, 0].set_xticks(range(3), [LABELS[m] for m in methods]); axes[index, 0].set_ylabel('Input feature count p')
        for i, value in enumerate(pvalues): axes[index, 0].text(i, value, str(value), ha='center', va='bottom')
        axes[index, 0].set_title(DISPLAY[dataset]+' input and effective rank')
        for i, method in enumerate(methods):
            ranks = [r['rank_max'] for r in subset if r['method'] == method and r['rank_max'] is not None]
            if ranks: axes[index, 0].scatter([i-.08, i+.08][:len(ranks)], ranks, marker='D', c='black', label='max mask-group rank' if i==1 else None)
        axes[index, 0].legend(fontsize=8)
        axes[index, 1].bar(range(3), coefs, color=[COLORS[m] for m in methods])
        axes[index, 1].set_xticks(range(3), [LABELS[m] for m in methods]); axes[index, 1].set_ylabel('Registered p x H x C coefficients')
        axes[index, 1].set_title(DISPLAY[dataset]+' readout capacity')
        for i, value in enumerate(coefs): axes[index, 1].text(i, value, f'{value:,}', ha='center', va='bottom', fontsize=9)
    finish(fig, FIGURES[3], 'Black diamonds: selected new-route max design rank. Old E33K rank is unknown in JSON; registered counts are not independent degrees of freedom.')

    fig, axes = pyplot.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for dataset, ax in zip(DATASETS, axes):
        matrix, labels = [], []
        for p in sorted([p for p in producers if p['dataset'] == dataset], key=lambda p:p['seed']):
            baseline = method_scores(p, 'E33K_REUSED', FINAL[dataset])['normalized']['per_channel_mse']
            for route in NEW:
                values = method_scores(p, route, FINAL[dataset])['normalized']['per_channel_mse']
                matrix.append([float('nan') if v is None or b is None else v-b for v,b in zip(values,baseline)])
                labels.append(route+' s'+str(p['seed']))
        finite = [abs(v) for row in matrix for v in row if math.isfinite(v)]
        bound = max(finite) if finite and max(finite)>0 else 1.
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
        ax.set_yticks(range(len(labels)), labels); ax.set_xticks(range(C_K[dataset][0]), [str(i) for i in range(C_K[dataset][0])])
        ax.set_xlabel('Original channel index'); ax.set_title(DISPLAY[dataset]+' final MSE delta vs E33K')
        fig.colorbar(im, ax=ax, label='New minus E33K normalized MSE')
    finish(fig, FIGURES[4], 'Negative blue cells improve over E33K; positive red cells worsen. Separate color ranges by dataset; masked channels are blank.')

    fig, axes = pyplot.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    new_costs = [r for r in costs if r['new_training']]
    xlabels = [DISPLAY[r['dataset']].split()[0]+' s'+str(r['seed'])+' '+r['method'] for r in new_costs]
    preparation = [r['new_FIT_factorization_wall_seconds'] for r in new_costs]
    remainder = [max(0., r['family_including_apply_save_guards_wall_seconds']-r['new_FIT_factorization_wall_seconds']) for r in new_costs]
    axes[0].bar(range(8), preparation, label='FIT preparation / factorization', color='#3374b6')
    axes[0].bar(range(8), remainder, bottom=preparation, label='Apply / scores / IO / guards', color='#c8d6e8')
    axes[0].set_xticks(range(8), xlabels, rotation=55, ha='right'); axes[0].set_ylabel('CPU family wall seconds'); axes[0].legend(fontsize=8)
    unique = [r for r in costs if r['method']=='RAW33C']
    axes[1].bar(range(4), [r['producer_process_wall_seconds'] for r in unique], color='#65758b')
    axes[1].set_xticks(range(4), [DISPLAY[r['dataset']].split()[0]+' s'+str(r['seed']) for r in unique], rotation=30, ha='right')
    axes[1].set_ylabel('Whole producer CPU process wall seconds')
    finish(fig, FIGURES[5], 'Process wall includes imports, provenance, IO and guards. Historical E/E33K training is reused/unknown here; no new deployment latency measurement.')


def write_KO_report(output, producers, audit, summary, capacities, costs):
    lines = ['# 원 C·K 표현·시간 요약을 같은 direct-C readout으로 비교', '',
        '[확인] 목적은 현재 고정 인코더가 만든 정보와 시간 요약의 조건부 예측 유용성을 좁히는 것이다. 미래 oracle 없이 원 과거 데이터와 cached 과거 latent만 사용했다.', '',
        'RAW33C는 원 C 과거의 32개16-point 평균과 마지막 값, E33K는 기존 N2의 동일 요약, E512K는 512개 latent 시점 전체를 사용한다. 세 경로 모두 actual-Y innovation 목표·current-C anchor·intercept·FIT 정규화·채널별 관측 수 기반 ridge를 사용했다. 입력 차원과 readout 용량은 서로 다르다.', '',
        'current-C anchor는 원 C 채널의 현재값 전체를 예측 경로에 제공한다. 따라서 세 경로는 모두 엄격한 K 전용 배포가 아니며, K 표현만으로 미래 전체 C를 복원했다는 해석은 적용하지 않는다.', '',
        '주지표는 정규화 observed-channel macro MSE이고 MAE도 함께 보고한다. 아래 평균은 원 E seed 두 개의 기술 통계이며 독립 데이터셋 반복이나 유의성 검정 결과가 아니다. RAW-F와 RAW33C는 같은 데이터의 결정적인 계산을 두 E seed 조건에서 반복했다.', '']
    lines += ['```text', 'dataset / method          normalized MSE      normalized MAE']
    for dataset in DATASETS:
        for method in METHODS:
            value = summary[dataset][method]
            lines.append(f'{DISPLAY[dataset].split()[0]:8s} {LABELS[method]:12s} {value["mse"]:16.8f} {value["mae"]:19.8f}')
    lines += ['```', '', '각 질문의 원수치와 판단:', '']
    for dataset in DATASETS:
        s = summary[dataset]
        raw_delta = s['RAW33C']['mse']-s['E33K_REUSED']['mse']
        full_delta = s['E512K']['mse']-s['E33K_REUSED']['mse']
        full_raw = s['E512K']['mse']-s['RAW33C']['mse']
        lines += [f'- {DISPLAY[dataset]}: RAW33C−E33K MSE는 `{raw_delta:+.8f}`, E512K−E33K는 `{full_delta:+.8f}`, E512K−RAW33C는 `{full_raw:+.8f}`이다. 음수는 앞 경로의 오차가 더 작다는 뜻이다. RAW33C 비교는 원 채널 요약 경로, E512K 비교는 더 많은 과거 latent와 더 큰 readout을 유지한 경로의 조건부 유용성을 보여준다. 이 수치만으로 encoder가 정보를 잃었다거나 압축이 불필요하다고 확정하지 않는다.']
    lines += ['', '다음 판단은 단독 encoder 효과보다 같은 조건에서 어떤 경로가 남는지를 기준으로 해야 한다. E512K가 더 좋아도 시간 정보 보존과 계수 용량 증가가 함께 변했고, 더 나빠도 현재 ridge와 표본·정규화·규제의 한계가 남는다. 대조학습·지연 그래프의 필요성이나 불필요성은 이 실험으로 판정하지 않는다.', '',
        '![평가 오차와 두 seed](figures/01_eval_seed_dots.png)', '[SVG](figures/01_eval_seed_dots.svg)', '',
        '![FIT TUNE 평가 차이](figures/02_fit_tune_final.png)', '[SVG](figures/02_fit_tune_final.svg)', '',
        '![고정 후보와 TUNE 선택](figures/03_alpha_tune_grid.png)', '[SVG](figures/03_alpha_tune_grid.svg)', '',
        '![입력과 계수 용량](figures/04_capacity_rank.png)', '[SVG](figures/04_capacity_rank.svg)', '',
        '![채널별 정규화 MSE 변화](figures/05_channel_delta.png)', '[SVG](figures/05_channel_delta.svg)', '',
        '![CPU 계수 추정과 전체 비용](figures/06_cpu_cost.png)', '[SVG](figures/06_cpu_cost.svg)', '',
        '비교 조건과 한계:', '',
        '- Robin C17/K4, Beijing Aotizhongxin C11/K2, 원 selected MAIN seeds92601/92602를 유지했다. L512/H48, FIT256/TUNE64, 원 FIT 통계·origin·Y/past mask·current-C anchor를 유지했다.',
        '- Robin VAL30은 반복 개발 자료이고 Beijing EVAL146은 이미 노출된 평가 자료다. 보호 TEST와 새 독립 holdout은 사용하지 않았다. 기존 E/D/F를 새로 load·forward·update하지 않았다.',
        '- 신규8 route fits에서 고정7 alpha의 56 후보를 계산하고 FIT/TUNE을 저장했다. 선택은 TUNE actual-Y MSE 최솟값, exact tie는 앞 alpha다. 선택 모델만 원 VAL/EVAL에 적용했다. E33K는 기존 계수와 결과를 검증해 재사용했다.',
        '- 고차원 solver는 동일 mask의 centered thin SVD를 공유한다. alpha0은 minimum-norm이며 intercept는 penalize하지 않는다. penalty는 alpha*Nc이고 horizon별 count가 아니다. FP32 Y−last → solver FP64 → innovation FP32 → current-C FP32 anchor 순서를 유지했다.',
        '- 같은 alpha도 입력 차원·전체 feature energy가 달라 같은 capacity나 규제 효과를 보장하지 않는다. 계수 수는 독립 자유도와 같지 않다. 기존 E33K rank는 JSON에서 확인할 수 없어 빈 값으로 남겼다.',
        '- 원단위 오차는 채널별 단위가 달라 진단 자료다. 이질 단위의 macro 값을 주순위로 사용하지 않는다. 관측0인 출력은 innovation0이며 미학습으로 표시한다.', '',
        f'독립 검산은 `{audit.get("status")}`, 실패 `{audit.get("failed_count")}`이며 새 후보 `{audit.get("new_candidate_rows_checked")}`, FIT/TUNE 후보 예측 `{audit.get("candidate_fit_tune_predictions_checked")}`, 선택 예측 `{audit.get("selected_predictions_checked")}`를 확인했다. 범위는 저장 배열·계수·통계·mask·선택·점수 재계산이다. E/F 재추론이나 SVD/lstsq 재학습을 독립 검산했다고 확대하지 않는다.', '',
        'provenance에 복사된 계획·고정 조건의 SOURCE_ONLY 및 승인 대기 문구는 실행 전 당시의 원문을 그대로 보존한 것이다. 현재 완료 여부는 별도 승인 기록, 고정 실행 manifest, 각 producer 결과, 독립 검산과 이 보고서 실행 결과에 근거한다.', '',
        f'실제 네 producer의 CPU process wall 합은 `{sum(r["producer_process_wall_seconds"] for r in costs if r["method"]=="RAW33C"):.4f}`초다. factorization/family 비용은 아래 CSV에 별도로 남겼다. 원 E 표현 생성·기존 E33K 학습 비용을 새 학습 비용이나 0초로 바꾸어 쓰지 않는다. 이 단계에서 새 GPU 추론이나 deployment latency는 측정하지 않았다.', '',
        '[모든 seed·split 정규화/원단위·채널별 점수](METRICS.csv) · [56 후보 선택 정보](ALPHA_GRID.csv) · [용량·rank](CAPACITY.csv) · [비용](COST.csv)', '',
        '[검산 입력 사본](provenance/inputs/INDEPENDENT_AUDIT.json) · [원 조건 결과 사본](provenance/inputs/) · [계획](provenance/sources/PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md) · [고정 조건](provenance/sources/METHOD_LOCK_MATCHED_READOUT_v1.md)', '',
        '[산출물 manifest](MANIFEST.json) · [보고서 실행 상태와 자원 receipt](RESULT.json)', '']
    with (output/'REPORT_KO.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def execute(args):
    import matched_readout_runtime_v1 as runtime
    tag = re.sub('[^a-z0-9_-]', '_', Path(args.out).name.lower())[:60]
    limit = runtime.ContinuationLimit('matched_report_'+tag, gpu=False)
    output = Path(args.out).resolve()
    result = {'status': 'RUNNING', 'terminal_JSON_only': True, 'NPZ_or_models_read': False,
        'actual_new_GPU_execution': False, 'protected_TEST_accessed': False,
        'authorization': limit.authorization_record, 'run_manifest': limit.run_manifest_record,
        'whole_goal_complete': False}
    initialized = False
    try:
        full = limit.checkfull
        full()
        producers, records = [], []
        for path in args.results:
            full()
            path = resolve(path)
            producers.append(read_json(path)); records.append({'role': 'producer', 'path': str(path), 'sha256': sha(path)})
        audit_path = resolve(args.audit)
        full(); audit = read_json(audit_path)
        audit_record = {'role': 'independent_audit', 'path': str(audit_path), 'sha256': sha(audit_path)}
        result['audit_gate'] = validate_inputs(producers, records, audit)
        for field in ('source_sha', 'source_sha256'):
            if field in audit:
                result['audit_gate'][field] = audit[field]
        receipts, receipt_records = [], []
        for p in producers:
            full()
            record = p['resource_receipt']; path = resolve(record['path'])
            if sha(path) != record['sha256']:
                raise ValueError('PRODUCER_RESOURCE_RECEIPT_SHA_MISMATCH')
            receipt = read_json(path)
            if receipt.get('status') != 'COMPLETE' or receipt.get('gpu') is not False:
                raise ValueError('COMPLETE_CPU_PRODUCER_RECEIPT_REQUIRED')
            receipts.append(receipt); receipt_records.append({'role': 'producer_receipt', 'path': str(path), 'sha256': sha(path)})
        full()
        output.mkdir(parents=True, exist_ok=False); initialized = True
        (output/'figures').mkdir(); (output/'provenance/inputs').mkdir(parents=True); (output/'provenance/sources').mkdir()
        copied = []
        for p, record, receipt_record in zip(producers, records, receipt_records):
            for source, name in ((record, p['dataset']+'_s'+str(p['seed'])+'.json'),
                                 (receipt_record, p['dataset']+'_s'+str(p['seed'])+'_RESOURCE.json')):
                full(); destination = output/'provenance/inputs'/name
                with Path(source['path']).open('rb') as src, destination.open('xb') as dst: shutil.copyfileobj(src, dst)
                if sha(destination) != source['sha256']: raise ValueError('INPUT_PROVENANCE_COPY_HASH')
                copied.append({**source, 'copy_relative': str(destination.relative_to(output)).replace('\\','/')}); full()
        for source, destination in ((audit_record, output/'provenance/inputs/INDEPENDENT_AUDIT.json'),
            (limit.authorization_record, output/'provenance/inputs/AUTHORIZATION.json'),
            (limit.run_manifest_record, output/'provenance/inputs/RUN_MANIFEST.json')):
            full()
            with Path(source['path']).open('rb') as src, destination.open('xb') as dst: shutil.copyfileobj(src, dst)
            if sha(destination) != source['sha256']: raise ValueError('APPROVAL_AUDIT_COPY_HASH')
            copied.append({**source, 'copy_relative': str(destination.relative_to(output)).replace('\\','/')}); full()
        for relative in ('make_matched_readout_report_v2.py', 'run_matched_readout_v1.py', 'matched_readout_math_v1.py',
            'matched_readout_contract_v1.py', 'matched_readout_runtime_v1.py', 'stage_separation_math_v1.py',
            'fast_storage_scan_v1.py', 'PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md', 'METHOD_LOCK_MATCHED_READOUT_v1.md',
            'validation_v1/audit_matched_readout_v1.py', 'validation_v1/audit_matched_readout_v2.py'):
            full(); source = HERE/relative; destination = output/'provenance/sources'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = sha(source)
            with source.open('rb') as src, destination.open('xb') as dst: shutil.copyfileobj(src, dst)
            if sha(destination) != digest: raise ValueError('REPORT_SOURCE_COPY_HASH')
            copied.append({'role': 'source', 'path': str(source), 'sha256': digest,
                           'copy_relative': str(destination.relative_to(output)).replace('\\','/')}); full()
        # The byte-identical plan's relative interpretation link needs a guide at
        # this copied location; the original/source-snapshot text stays unchanged.
        full()
        interpretation = HERE/'STAGE_SEPARATION_INTERPRETATION_KO_v1.md'
        interpretation_sha = sha(interpretation)
        guide = output/'provenance/sources/STAGE_SEPARATION_INTERPRETATION_KO_v1.md'
        with guide.open('x', encoding='utf-8') as handle:
            handle.write('# 원 해석 문서 연결 안내\n\n'
                '이 파일은 복사된 계획 문서의 상대 링크를 연결하기 위해 새로 만든 안내다. '
                '원 해석 문서의 바이트 사본이나 새로운 실험 결과가 아니다.\n\n'
                '[원 stage 해석 문서](../../../STAGE_SEPARATION_INTERPRETATION_KO_v1.md)\n\n'
                '원 문서 SHA256: `'+interpretation_sha+'`\n')
        result['source_link_guides'] = [{'path':str(guide.relative_to(output)).replace('\\','/'),
            'sha256':sha(guide), 'original_source':{'path':str(interpretation), 'sha256':interpretation_sha},
            'byte_identical_source_copy':False}]
        full()
        metrics, alphas, capacities, costs = metric_rows(producers), alpha_rows(producers), capacity_rows(producers), cost_rows(producers, receipts)
        for name, rows in (('METRICS.csv', metrics), ('ALPHA_GRID.csv', alphas), ('CAPACITY.csv', capacities), ('COST.csv', costs)):
            full()
            with (output/name).open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
                for row in rows: writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})
            full()
        full()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
        pyplot.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
        def save_figure(fig, path):
            full()
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            full()
        figures(producers, capacities, costs, pyplot, output, save_figure)
        summary = summary_values(producers)
        full(); write_KO_report(output, producers, audit, summary, capacities, costs); full()
        for record in copied:
            if sha(record['path']) != record['sha256'] or sha(output/record['copy_relative']) != record['sha256']:
                raise ValueError('REPORT_INPUT_OR_SOURCE_CHANGED')
        if sha(interpretation) != interpretation_sha:
            raise ValueError('LINK_GUIDE_ORIGINAL_SOURCE_CHANGED')
        artifacts = [{'path': str(path.relative_to(output)).replace('\\','/'), 'sha256': sha(path), 'bytes': path.stat().st_size}
                     for path in sorted(output.rglob('*')) if path.is_file()]
        manifest = {'status': 'COMPLETE', 'report_scope': 'terminal JSON and code only; no scientific array/model replay',
            'input_records': copied, 'artifact_records': artifacts, 'audit_gate': result['audit_gate'],
            'source_link_guides':result['source_link_guides'],
            'conditions': [{'dataset':p['dataset'], 'seed':p['seed']} for p in producers], 'summary': summary,
            'metrics_rows': len(metrics), 'new_alpha_rows': len(alphas), 'capacity_rows': len(capacities), 'cost_rows': len(costs),
            'figure_png_count': 6, 'figure_svg_count': 6, 'old_training_cost_unknown_not_zero': True,
            'physical_macro_units_not_comparable': True, 'independent_generalization_proven': False}
        full(); runtime.write_new(output/'MANIFEST.json', manifest); full()
        result.update(status='COMPLETE', manifest={'path':'MANIFEST.json', 'sha256':sha(output/'MANIFEST.json')},
                      figures=list(FIGURES), new_alpha_rows=len(alphas), summary=summary)
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type':type(error).__name__, 'message':str(error)})
    finally:
        receipt = limit.close(result['status'], result.get('failure')); result['resource_receipt'] = receipt
        if receipt.get('terminal_status') != 'COMPLETE':
            result.update(status=receipt['terminal_status'], resource_terminal_error=receipt.get('terminal_error'))
        if initialized: runtime.write_new(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true'); mode.add_argument('--execute', action='store_true')
    parser.add_argument('--results', nargs=4); parser.add_argument('--audit'); parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED', 'terminal_JSON_only':True,
            'figures':list(FIGURES), 'new_alpha_rows':56, 'canonical_auth_required':True})); return 0
    if not args.results or not args.audit: parser.error('execute requires exactly4 --results and --audit')
    result = execute(args); print(json.dumps({'status':result['status'], 'failure':result.get('failure')}))
    return 0 if result['status']=='COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
