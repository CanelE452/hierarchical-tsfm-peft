"""Capture completed search endpoints or redraw the public aggregate CSV only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
COLORS = {'B0': '#666666', 'B2': '#0072B2', 'LAG_CAUSAL': '#D55E00', 'LAG_BIDIR': '#009E73'}
LABELS = {'B0': 'B0: pointwise', 'B2': 'B2: full-window mixer',
          'LAG_CAUSAL': 'Past-directed lags', 'LAG_BIDIR': 'Within-window bidirectional lags'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def draw(csv_path, output, preview=None):
    with csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12 or {row['family'] for row in rows} != set(COLORS):
        raise ValueError('This progress snapshot requires four conditions and endpoints0/0.25/1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, (axis, info) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [2.25, 1]})
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.23, wspace=.20)
    for family, color in COLORS.items():
        selected = sorted((row for row in rows if row['family'] == family), key=lambda row: float(row['epoch']))
        if [float(row['epoch']) for row in selected] != [0, .25, 1] or [int(row['updates']) for row in selected] != [0, 156, 622]:
            raise ValueError('Captured learning endpoints differ')
        axis.plot([int(row['updates']) for row in selected], [float(row['astop_mse']) for row in selected],
                  marker='o', markersize=5, linewidth=2, color=color, label=LABELS[family])
    axis.set(xlabel='Completed optimizer updates (same exposure schedule)', ylabel='A_stop channel-macro MSE',
             xticks=[0, 156, 622], xticklabels=['0\ninitial', '156\n0.25 epoch', '622\n1 epoch'])
    axis.set_ylim(.36, .63)
    axis.grid(axis='y', alpha=.22)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper right', frameon=False, fontsize=9)
    info.axis('off')
    lines = ['MATCHED CPU DEVELOPMENT', '', 'Jena: L=512, H=48', 'C=21, K=6, seed=92601',
             'Frozen Bolt-small; same decoder', '4 simultaneous CPU workers', '2 threads per worker', '',
             'Completed: search through 1 epoch', 'Next: 3 epochs, fresh A60 refit', 'Full V / cost: pending', '',
             'A_stop at 1 epoch:']
    for family in COLORS:
        row = next(row for row in rows if row['family'] == family and float(row['epoch']) == 1)
        lines.append('%-12s %.6f' % ({'B0': 'B0', 'B2': 'B2', 'LAG_CAUSAL': 'Past lags', 'LAG_BIDIR': 'Bidir lags'}[family], float(row['astop_mse'])))
    info.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=9.2, linespacing=1.25)
    fig.suptitle('Search progress through 1 epoch — intermediate A_stop results', x=.07, ha='left', fontsize=15, fontweight='bold')
    fig.text(.07, .055, 'Post-T exploratory development. No new V score or independent holdout claim.\n'
             'Same K, seed, loss and exposure schedule; different encoder capacities/boundaries. One seed, no uncertainty interval.', fontsize=9)
    svg = output / 'SEARCH_PROGRESS_E1.svg'
    if svg.exists() or (preview is not None and preview.exists()):
        raise FileExistsError('Preserve previous figure bytes')
    fig.savefig(svg, metadata={'Date': None}, bbox_inches='tight')
    if preview is not None:
        fig.savefig(preview, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return dict(svg_sha256=sha(svg), public_csv_sha256=sha(csv_path), preview_sha256=sha(preview) if preview is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--plan-sha')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--input-csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('A new progress snapshot directory is required')
    if args.capture:
        if not all((args.public_root, args.cache_root, args.asset_root, args.plan_sha)) or args.input_csv:
            parser.error('Capture requires old roots/plan SHA and no input CSV')
        public, cache, asset = (path.resolve() for path in (args.public_root, args.cache_root, args.asset_root))
        if not output.is_relative_to(public / HERE.name / 'diagnostics'):
            raise ValueError('Capture output must stay under the separate CPU diagnostic root')
        spec = importlib.util.spec_from_file_location('search_progress_original_runner', HERE / 'cpu_lag_runner.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        plan = runner.metadata(public, cache, asset, args.plan_sha)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        import torch
        torch.set_num_threads(2)
        records, inputs = [], []
        for condition in plan['conditions']:
            run_id = condition['run_id']
            event = public / HERE.name / 'runs' / run_id / 'search/event_000002.json'
            checkpoint = cache / HERE.name / 'runs' / run_id / 'search/checkpoint_000002.pt'
            event_value = read(event)
            if (event_value['status'], event_value['epoch'], event_value['updates'], event_value['exposures']) != ('SEGMENT_COMPLETE', 1, 622, 9923):
                raise ValueError('The fixed 1-epoch milestone is not complete')
            before = sha(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (sha(checkpoint) != before or saved['updates'] != 622 or saved['contract'] != event_value['contract']
                    or saved['contract']['data_sha'] != plan['task_contract']['data_sha256']
                    or saved['contract']['model_sha'] != plan['foundation_tensor_sha256']
                    or saved['contract']['scaler_sha'] != plan['task_contract']['search_scaler_sha256']):
                raise ValueError('Captured search snapshot contract differs')
            curves = saved['extra']['curves']
            if [row['epoch'] for row in curves] != [0, .25, 1]:
                raise ValueError('Capture must not choose a different endpoint retrospectively')
            for row in curves:
                records.append(dict(run_id=run_id, family=condition['family'], **row))
            inputs.append(dict(run_id=run_id, event_public_relative_path=event.relative_to(public).as_posix(),
                event_sha256=sha(event), checkpoint_cache_relative_path=checkpoint.relative_to(cache).as_posix(),
                checkpoint_sha256=before, contract=saved['contract']))
        runner.metadata(public, cache, asset, args.plan_sha)
        output.mkdir(parents=True, exist_ok=False)
        csv_path = output / 'SEARCH_CURVES_THROUGH_E1.csv'
        with csv_path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        new(output / 'EVIDENCE.json', dict(status='CAPTURED_COMPLETED_SEARCH_ENDPOINTS_ONLY', inputs=inputs,
            plan_sha256=args.plan_sha, capture_source_sha256=sha(__file__), captured_unix=time.time(),
            actual_model_forward_or_training_in_capture=False, captured_weights_published=False,
            V_scoring_or_cost_protocol_complete=False, post_T_exploratory_development=True,
            original_sources_registry_selection_T_and_scores_preserved=True, original_GPU_gap_completed=False))
    else:
        if args.input_csv is None:
            parser.error('Supply a public aggregate --input-csv or request --capture')
        csv_path = args.input_csv.resolve()
        output.mkdir(parents=True, exist_ok=False)
    result = draw(csv_path, output, args.preview.resolve() if args.preview else None)
    new(output / 'FIGURE_BINDING.json', dict(status='RENDERED_PUBLIC_AGGREGATE_SEARCH_PROGRESS', **result,
        source_sha256=sha(__file__), one_seed=True, uncertainty_interval=False, final_V_result=False))
    print(json.dumps(dict(status='SEARCH_PROGRESS_CAPTURE_OR_RENDER_COMPLETE', output=str(output), **result)))


if __name__ == '__main__':
    main()
