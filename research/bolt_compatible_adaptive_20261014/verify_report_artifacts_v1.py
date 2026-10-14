"""Independent report/CSV/link/image validation, without model or NPZ reads."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import re
import sys
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-directory',required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import adaptive_runtime_v1 as rt
    limit = rt.ContinuationLimit(args.tag,gpu=False)
    report = {'status':'RUNNING','scope':'Report/CSV/file/image metadata only; no NPZ/model/timer replay'}
    output = (ROOT/args.output).resolve()
    error = None
    try:
        folder = (ROOT/args.report_directory).resolve()
        assert folder.is_relative_to(HERE) and output.is_relative_to(HERE) and not output.exists()
        manifest = rt.read(folder/'MANIFEST.json')
        assert manifest['status'] == 'COMPLETE'
        bound = []
        def check_records(value):
            if isinstance(value,dict):
                if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                    path = Path(value['path'])
                    if not path.is_absolute():
                        path = ROOT/path
                    assert path.is_file() and rt.sha(path) == value['sha256'], str(path)
                    bound.append(str(path))
                for child in value.values():
                    check_records(child)
            elif isinstance(value,list):
                for child in value:
                    check_records(child)
        check_records(manifest)
        producers = [rt.read(HERE/'runs'/name/'RESULT.json') for name in ('first_robin_v1','first_beijing_v1')]
        dense = rt.read(HERE/'runs/expanded_robin_v2/RESULT.json')
        counts = {}
        metric_cells = 0
        def verify_metric_table(name,cohort,expected_rows):
            nonlocal metric_cells
            with (folder/'tables'/name).open(encoding='utf-8-sig',newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == expected_rows
            unique = set()
            for row in rows:
                p = next(p for p in cohort if p['dataset'] == row['dataset'])
                method,split,scale = row['method'],row['split'],row['metric_scale']
                identity = (p['dataset'],method,split,scale)
                assert identity not in unique
                unique.add(identity)
                if method == 'RAW_BOLT':
                    values = p['references'][split]['RAW_BOLT'][scale]
                elif method in ('PERSIST','MEAN'):
                    values = p['reference_shared_controls'][method][split][scale]
                else:
                    values = p['families'][method]['splits'][split][scale]
                for key in ('mse','mae'):
                    assert math.isclose(float(row[key]),values[key],rel_tol=2e-12,abs_tol=2e-12),identity
                    metric_cells += 1
                for key in ('per_channel_mse','per_channel_mae','observed_counts'):
                    saved = json.loads(row[key])
                    assert len(saved) == len(values[key])
                    assert all(math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12) for a,b in zip(saved,values[key])),identity
                    metric_cells += len(saved)
                assert int(row['active_channels']) == values['active_channels']
                assert row['independent_holdout'] == 'False'
                metric_cells += 2
            counts[name] = len(rows)
        verify_metric_table('COMPARISON.csv',producers,72)
        verify_metric_table('EXPANDED_COMPARISON.csv',[dense],12)
        for name,cohort,n in (('POOL_SELECTION.csv',producers,54),('EXPANDED_POOL_SELECTION.csv',[dense],18)):
            with (folder/'tables'/name).open(encoding='utf-8-sig',newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == n
            for row in rows:
                p = next(p for p in cohort if p['dataset'] == row['dataset'])
                family = p['families'][row['method']]
                index = int(row['candidate_index'])
                entry = family['pool'][index]
                assert row['candidate_kind'] == entry['kind']
                assert (row['selected'] == 'True') == (index == family['selected_index'])
                for split in ('fit','tune'):
                    for key in ('mse','mae'):
                        assert math.isclose(float(row[split+'_'+key]),entry[split+'_scores']['normalized'][key],rel_tol=2e-12,abs_tol=2e-12)
                        metric_cells += 1
            counts[name] = len(rows)
        for name,cohort,n in (('CAPACITY.csv',producers,6),('EXPANDED_CAPACITY.csv',[dense],2)):
            with (folder/'tables'/name).open(encoding='utf-8-sig',newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == n
            for row in rows:
                p = next(p for p in cohort if p['dataset'] == row['dataset'])
                state = p['feature_states'][row['method']]
                assert int(row['channels']) == state['input_channels']
                assert int(row['features_per_output_channel']) == state['feature_count']
                assert int(row['potential_registered_coefficients']) == state['feature_count']*48*state['input_channels']
            counts[name] = len(rows)
        for name,cost_path,n in (('COST.csv','first_readout24_v2',24),('EXPANDED_COST.csv','expanded_robin4_v1',4)):
            cost = rt.read(HERE/'cost_runs'/cost_path/'RESULT.json')
            with (folder/'tables'/name).open(encoding='utf-8-sig',newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == n
            for row in rows:
                source = next(r for r in cost['rows'] if (r['dataset'],r['route'],r['batch']) == (row['dataset'],row['method'],int(row['batch'])))
                assert math.isclose(float(row['whole_median_ms_per_window']),1000*source['median_seconds_per_window'],rel_tol=2e-12,abs_tol=2e-12)
                metric_cells += 1
            counts[name] = len(rows)
        links = []
        for path in folder.glob('*.md'):
            content = path.read_text(encoding='utf-8')
            assert not re.search('[\u4e00-\u9fff]',content), 'Unexpected non-Korean CJK prose:'+str(path)
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
                target = target.strip().split(' "',1)[0].strip('<>')
                if target.startswith(('https://','http://','#')):
                    continue
                local = (path.parent/unquote(target.split('#',1)[0])).resolve()
                assert local.exists(), str(local)
                links.append(str(local))
        from PIL import Image
        pngs = sorted((folder/'figures').glob('*.png'))
        svgs = sorted((folder/'figures').glob('*.svg'))
        assert len(pngs) == len(svgs) == 8
        image_records = []
        for path in pngs:
            with Image.open(path) as im:
                width,height = im.size
                assert width >= 1000 and height >= 400
                im.verify()
            image_records.append({'path':str(path),'width':width,'height':height,'sha256':rt.sha(path)})
        for path in svgs:
            assert '<svg' in path.read_text(encoding='utf-8')[:2000]
        limit.checkfull()
        report.update(status='PASS',report_manifest_sha256=rt.sha(folder/'MANIFEST.json'),
                      declared_file_sha_bindings_checked=len(bound),csv_rows=counts,metric_cells_checked=metric_cells,
                      relative_links_checked=len(links),missing_links=[],PNG=image_records,SVG_count=len(svgs),
                      independently_opened_TEST=False,new_fit_or_F_forward=False)
    except Exception as failure:
        import traceback
        error = dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
        report.update(status='FAIL',failure=error)
    finally:
        terminal = limit.close('COMPLETE' if report['status'] == 'PASS' else 'FAILED_PRESERVED',error)
        report['resource_terminal'] = terminal
        if terminal['terminal_status'] != 'COMPLETE':
            report['status'] = 'FAIL'
        rt.write_new(output,report)
        print(json.dumps({'report_audit':str(output),'status':report['status'],'metric_cells':report.get('metric_cells_checked'),'failure':error}),flush=True)
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
