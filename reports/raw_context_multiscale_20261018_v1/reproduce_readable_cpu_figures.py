"""Redraw the two audited CPU facets using public CSVs, preserving every source point."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile


NAMES = ('F3_NATIVE_ALL_jena_cpu_B1', 'F3_NATIVE_ALL_jena_cpu_B8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def snapshot(source):
    return {p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*')) if p.is_file()}


def load_points(source):
    manifest = read(source/'FIGURE_MANIFEST.json')
    inputs = read(source/'REPORT_INPUTS.json')
    if inputs.get('status') != 'FULL_AUDITED_PUBLIC_REPLAY':
        raise ValueError('Audited final public inputs required')
    for name in ('FIGURE_POINTS', 'COST'):
        if sha(source/(name+'.csv')) != inputs['tables'][name]:
            raise ValueError('Audited source table SHA changed: '+name)
    if manifest['point_table'] != dict(file='FIGURE_POINTS.csv', sha256=sha(source/'FIGURE_POINTS.csv')):
        raise ValueError('Figure manifest must bind exact point table')
    with (source/'FIGURE_POINTS.csv').open(newline='', encoding='utf-8-sig') as stream:
        points = [r for r in csv.DictReader(stream) if r['device'] == 'cpu']
    with (source/'COST.csv').open(newline='', encoding='utf-8-sig') as stream:
        costs = [r for r in csv.DictReader(stream) if r['device'] == 'cpu']
    if len(points) != 4 or len(costs) != 2 or {r['figure'] for r in points} != set(NAMES):
        raise ValueError('Exactly both CPU Ridge facets and their four metric points required')
    rows, evidence = [], []
    for name, batch in zip(NAMES, (1, 8)):
        selected = [r for r in points if r['figure'] == name]
        matching_costs = [r for r in costs if r['batch_size'] == str(batch)]
        facets = [r for r in manifest['all_native_facets'] if r['figure'] == name]
        figure = [r for r in manifest['figures'] if r['name'] == name]
        if (len(selected) != 2 or {r['metric'] for r in selected} != {'latency', 'RSS'}
                or len(matching_costs) != 1 or len(facets) != 1 or len(figure) != 1
                or figure[0]['route_point_rows'] != 2):
            raise ValueError('CPU figure/facet/point coverage differs')
        for extension in ('png', 'svg'):
            if sha(source/(name+'.'+extension)) != figure[0][extension+'_sha256']:
                raise ValueError('Original figure differs from immutable manifest')
        cost, facet = matching_costs[0], facets[0]
        cost_id = 'Ridge_jena_native_B'+str(batch)
        if (cost['cost_id'] != cost_id or cost['accuracy_run_id'] != 'Ridge_jena'
                or cost['dataset'] != 'jena' or cost['family'] != 'RAW_RIDGE'
                or cost['schedule'] != 'native' or cost['comparison_scope'] != 'CPU_RIDGE_CONTEXT'
                or cost['status'] != 'COMPLETE' or cost['cost_valid'] != 'True'
                or facet != dict(figure=name, dataset='jena', device='cpu', batch_size=batch,
                    run_id='Ridge_jena', schedule='native', seed=92601, cost_id=cost_id)):
            raise ValueError('Exact audited CPU Ridge identity required')
        by_metric = {r['metric']:r for r in selected}
        for point in selected:
            if (point['cost_id'] != cost_id or point['run_id'] != 'Ridge_jena'
                    or point['batch_size'] != str(batch) or point['dataset'] != 'jena'
                    or point['seed'] != '92601' or point['mode'] != 'Ridge'
                    or point['schedule'] != 'native' or point['view'] != 'all_native'
                    or point['label'] != 'Ridge s92601' or point['filled'] != 'True'
                    or point['y'] != cost['MSE']):
                raise ValueError('Every CPU source point must retain its exact identity and MSE')
        if (by_metric['latency']['x'] != cost['median_batch_ms']
                or by_metric['RSS']['x'] != cost['CPU_RSS_sampled_peak_MiB']):
            raise ValueError('Source coordinates differ from audited cost table')
        values = dict(MSE=cost['MSE'], median_batch_ms=cost['median_batch_ms'],
                      CPU_RSS_sampled_peak_MiB=cost['CPU_RSS_sampled_peak_MiB'])
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
            raise ValueError('Positive finite source coordinates required')
        rows.append(dict(figure=name, cost_id=cost_id, dataset='jena', device='cpu', batch_size=batch,
            mode='Ridge', run_id='Ridge_jena', seed=92601, schedule='native', **values))
        evidence.append(dict(original_figure=figure[0], manifest_facet=facet, original_point_rows=selected))
    return rows, evidence


def render(source, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('Output must be outside the immutable original source tree')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    before = snapshot(source)
    rows, evidence = load_points(source)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'svg.hashsalt':'readable-cpu-figures-001', 'svg.fonttype':'none'})
    layouts = []
    for row in rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        fig.subplots_adjust(left=.085, right=.98, bottom=.22, top=.72, wspace=.30)
        fig.suptitle('Jena | CPU Ridge | batch '+str(row['batch_size'])+'\nNative whole-path cost; seed 92601', fontsize=14, y=.96)
        for ax, metric, title, xlabel in zip(axes,
                ('median_batch_ms', 'CPU_RSS_sampled_peak_MiB'), ('Latency', 'CPU memory'),
                ('Whole-path median (ms/batch; log axis)', 'Sampled CPU RSS peak (MiB)')):
            ax.scatter(float(row[metric]), float(row['MSE']), marker='D', s=70,
                facecolors='#E69F00', edgecolors='#E69F00', label='Ridge (seed 92601)')
            if metric == 'median_batch_ms':
                ax.set_xscale('log')
            ax.set(title=title, xlabel=xlabel, ylabel='V channel-macro MSE')
            ax.grid(alpha=.18)
            ax.legend(loc='upper right', fontsize=9, frameon=False)
        fig.text(.5, .055, 'CPU context only | One measured route per panel | Same audited points and units',
                 ha='center', fontsize=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = list(fig.texts)
        for ax in axes:
            texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label, ax.xaxis.offsetText, ax.yaxis.offsetText])
            texts.extend(ax.get_legend().get_texts())
            for axis in (ax.xaxis, ax.yaxis):
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if low <= tick.get_loc() <= high:
                        texts.extend([tick.label1, tick.label2])
        text_bounds = [artist.get_window_extent(renderer) for artist in texts if artist.get_visible() and artist.get_text()]
        outer = fig.bbox
        if any(box.x0 < outer.x0-1 or box.y0 < outer.y0-1 or box.x1 > outer.x1+1 or box.y1 > outer.y1+1
               for box in text_bounds):
            raise ValueError('Text exceeds the figure canvas')
        layouts.append(dict(figure=row['figure'], text_within_canvas=True,
            panels=[dict(xscale=ax.get_xscale(), yscale=ax.get_yscale(),
                         xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())) for ax in axes]))
        for extension in ('png', 'svg'):
            fig.savefig(output/(row['figure']+'_READABLE.'+extension), dpi=180,
                metadata={'Software':'Readable CPU public replay'} if extension == 'png' else
                         {'Date':None, 'Creator':'Readable CPU public replay'})
        plt.close(fig)
    with (output/'SOURCE_POINTS.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    if snapshot(source) != before:
        raise ValueError('Original final source tree changed during public redraw')
    result = dict(status='PASS', figures=2, source_facets=2, original_metric_points=4,
        source_files_sha256=before, source_preserved_byte_exact=True,
        input_files={name:sha(source/name) for name in ('FIGURE_POINTS.csv', 'COST.csv', 'FIGURE_MANIFEST.json', 'REPORT_INPUTS.json')},
        helper_sha256=sha(Path(__file__)), source_point_identity='PASS_EXACT_CSV_STRINGS_AND_MANIFEST_IDS',
        source_point_evidence=evidence, layouts=layouts,
        scope='Readable supplement only; original14 figures/16CSV unchanged; V development metrics; no CPU/GPU ratio',
        source_points_sha256=sha(output/'SOURCE_POINTS.csv'),
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    with (output/'READABLE_CPU_FIGURES_VALIDATION.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = render(args.source_dir, args.output_dir)
    with tempfile.TemporaryDirectory(prefix='readable-cpu-replay-') as temporary:
        replay = Path(temporary)/'replay'
        render(args.source_dir, replay)
        files = sorted(p.name for p in args.output_dir.iterdir() if p.is_file())
        if any((args.output_dir/name).read_bytes() != (replay/name).read_bytes() for name in files):
            raise ValueError('Independent public replay differs byte-for-byte')
        proof = dict(status='PASS', independent_renders=2, byte_exact_files={name:sha(args.output_dir/name) for name in files},
                     original_source_preserved=True, GPU_execution=False, model_or_array_access=False)
    with (args.output_dir/'REPRODUCIBILITY_CHECK.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status=result['status'], figures=2, byte_exact_replay='PASS')))


if __name__ == '__main__':
    main()
