"""Replay portable measurement receipts and independently check raw trial CSVs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .analysis import export_receipts


def _quantile(values, probability):
    values=sorted(values)
    index=(len(values)-1)*probability
    low=math.floor(index); high=math.ceil(index)
    return values[low]+(values[high]-values[low])*(index-low)


def verify_csv(report_root):
    root=Path(report_root)
    with (root/'MEMORY_COST_TRIALS.csv').open(encoding='utf-8',newline='') as stream:
        trials=list(csv.DictReader(stream))
    with (root/'MEMORY_COST_SUMMARY.csv').open(encoding='utf-8',newline='') as stream:
        summaries=list(csv.DictReader(stream))
    verified=[]
    for row in summaries:
        if row['status']!='COMPLETE':
            continue
        selected=[value for value in trials if value['facet_id']==row['facet_id'] and value['stage']==row['stage']]
        timed=[value for value in selected if value['phase']=='timed']
        warm=[value for value in selected if value['phase']=='warmup']
        expected=(2,3) if row['stage']=='screen' else (10,20)
        if (len(warm),len(timed))!=expected:
            raise ValueError('Raw CSV repeat coverage differs: '+row['facet_id'])
        times=[float(value['whole_seconds'])*1000 for value in timed]
        batch=int(row['B']); median=statistics.median(times)
        statistics_to_check=dict(median_batch_ms=median,IQR_batch_ms=_quantile(times,.75)-_quantile(times,.25),
            p95_batch_ms=_quantile(times,.95),median_ms_per_origin=median/batch,origins_per_second=batch*1000/median)
        if row['device']=='cuda':
            statistics_to_check.update(peak_allocated_MiB=max(float(value['peak_allocated_MiB']) for value in timed),
                peak_reserved_MiB=max(float(value['peak_reserved_MiB']) for value in timed))
        for key,value in statistics_to_check.items():
            if not math.isclose(float(row[key]),value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Raw CSV independent statistic differs: '+key)
        if any(int(value['pid'])!=int(row['process_pid']) for value in selected):
            raise ValueError('CSV trial is not from the recorded isolated worker')
        verified.append((row['facet_id'],row['stage']))
    return dict(status='PASS',verified_facets=len(verified),raw_trial_rows=len(trials),
                accumulation='Python float64; independent statistics/linear quantiles from raw CSV')


def replay(report_root, output_root, figures=True):
    source=Path(report_root).resolve(); target=Path(output_root).resolve()
    if source==target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Replay destination must be separate from published evidence')
    if target.exists():
        raise ValueError('Replay destination must be a new directory; preserve existing files')
    paths=sorted((source/'receipts').glob('*.receipt.json'))
    if not paths:
        raise ValueError('Portable receipts required; no inference is run by replay')
    products=export_receipts(paths,target,make_figures=figures)
    comparisons=[]
    for name in products['files']:
        original=source/name; regenerated=target/name
        if not original.exists() or original.read_bytes()!=regenerated.read_bytes():
            raise ValueError('Public replay bytes differ: '+name)
        comparisons.append(dict(file=name,sha256=hashlib.sha256(regenerated.read_bytes()).hexdigest()))
    result=verify_csv(source)
    return dict(result,products=comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    result=replay(args.report_root,args.output,not args.no_figures)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
