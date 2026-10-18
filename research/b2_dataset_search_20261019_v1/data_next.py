"""Metadata-admitted, bounded native-resolution ETTm1 and full Solar prefixes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
from itertools import islice
import json
from pathlib import Path
import re

import numpy as np

from research.b2_dataset_search_20261019_v1 import data as base


NAME = 'b2_dataset_search_20261019_v1'
DATASETS = {
    'ETTm1': dict(rows=69680, columns=base.ETT_COLUMNS, interval_seconds=900, period=96,
        header=('date', *base.ETT_COLUMNS), historical_bounds=[0,41808,48776,55744,69680],
        first_timestamp='2016-07-01 00:00:00',exclusive_prefix_end='2018-02-01 16:00:00'),
    'solar_full137': dict(rows=52560, columns=tuple(str(i) for i in range(137)),
        interval_seconds=600, period=144),
}


def source_locations(asset_root,dataset):
    if dataset not in DATASETS:
        raise ValueError('Unsupported next dataset: '+str(dataset))
    siblings=Path(asset_root).resolve().parent
    project=siblings/'tsfm-peft-method-screen'
    if dataset=='ETTm1':
        return dict(source=siblings/'covariate-trust-pilot/data/ett_small/ETTm1.csv',project=project,
            metadata=project/'results/condition_studies_v1_20260916/N03/data_receipt.json',
            manifest=siblings/'covariate-trust-pilot/data/external_datasets_manifest.json',
            master_manifest=project/'results/condition_studies_v1_20260916/MASTER_MANIFEST.json')
    return dict(project=project,
        source=project/'.cache/continuation_lora_v1_20260922/data/solar_AL_7f402f185cc2435b5e66aed13a3b560ed142e023.txt.gz',
        metadata=project/'results/continuation_lora_v1_20260922/DATA_AUDIT.json')


def admit_source(asset_root,dataset,*,metadata_path,manifest_path=None,master_manifest_path=None):
    locations=source_locations(asset_root,dataset)
    spec=DATASETS[dataset]
    required=[metadata_path,manifest_path,master_manifest_path] if dataset=='ETTm1' else [metadata_path]
    if any(path is None for path in required):
        raise ValueError('Explicit source metadata inputs are required before numeric preparation')
    documents=[base._metadata(path,asset_root) for path in required]
    value=documents[0][0]
    n,channels=spec['rows'],len(spec['columns'])
    prefix=n*4//5
    if dataset=='ETTm1':
        files=documents[1][0].get('datasets',{}).get('ett_small',{}).get('files',[])
        records=[row for row in files if row.get('name')=='ETTm1.csv']
        studies=[row for row in documents[2][0].get('specs',[]) if row.get('id')=='N03']
        study=studies[0].get('data',{}) if len(studies)==1 else {}
        if (value.get('raw_shape')!=[n,channels] or value.get('period')!=spec['period']
                or value.get('bounds')!=spec['historical_bounds'] or spec['historical_bounds'][-2]!=prefix
                or study.get('source')!='ettm1' or study.get('frequency_minutes')!=spec['interval_seconds']//60
                or study.get('bounds')!=spec['historical_bounds'] or len(records)!=1
                or records[0].get('bytes')!=value.get('bytes') or records[0].get('sha256')!=value.get('sha256')):
            raise ValueError('ETTm1 full-shape, frequency, protection or dual metadata identity differs')
        size,digest=value.get('bytes'),value.get('sha256')
        protection=dict(historical_role='E_DISCOVERY',raw_prefix_exclusive_end=prefix,
            historical_bounds=spec['historical_bounds'],same_station_hourly_cutoff=prefix*spec['interval_seconds']//3600,
            scope='Same-station ETTh1 clock cutoff alignment; numeric equality between resolutions is not claimed')
    else:
        source=value.get('source',{})
        schema,aggregation,clock=(source.get(key,{}) for key in ('schema','aggregation','time_axis'))
        test=source.get('split_edges',{}).get('TEST',{})
        relative=locations['source'].relative_to(locations['project']).as_posix()
        if (schema.get('format')!='gzip csv matrix' or schema.get('raw_shape')!=[n,channels]
                or schema.get('timestamp_column') is not False or aggregation.get('rows_per_hour')!=6
                or aggregation.get('raw_rows')!=n or aggregation.get('hourly_rows')!=n//6
                or aggregation.get('invented_timestamps') is not False
                or clock.get('calendar_dates_invented') is not False or clock.get('timezone_invented') is not False
                or test.get('start')!=prefix//6 or test.get('end')!=n//6
                or str(source.get('raw_path','')).replace('\\','/')!=relative):
            raise ValueError('Solar full-shape, elapsed clock or historical TEST metadata contract differs')
        size,digest=source.get('raw_bytes'),source.get('raw_sha256')
        protection=dict(historical_role='TEST',historical_hourly_start=test['start'],rows_per_historical_hour=6,
            raw_prefix_exclusive_end=prefix,
            scope='Chronological native rows preceding the old hourly TEST; no aggregation is performed')
    if type(size) is not int or size<1 or locations['source'].stat().st_size!=size:
        raise ValueError('Raw source size differs from the admitted metadata')
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Historical source SHA metadata is required; do not hash the protected raw file')
    return dict(locations=locations,spec=spec,source_metadata=[item[1] for item in documents],
        source_bytes=size,historical_sha256=digest,protection=protection)


def _prefix(path,dataset,spec):
    count,channels=spec['rows']*4//5,len(spec['columns'])
    physical=np.empty((count,channels),np.float32)
    times=np.arange(count,dtype=np.int64)*spec['interval_seconds']*10**9
    opener=gzip.open if dataset=='solar_full137' else Path.open
    with opener(path,'rb') as handle:
        lines=base._ConsumedLines(handle)
        records=csv.reader(lines) if dataset=='ETTm1' else (line.rstrip('\r\n').split(',') for line in lines)
        if dataset=='ETTm1' and next(records,None)!=list(spec['header']):
            raise ValueError('ETTm1 header differs from all seven original physical columns')
        consumed=0
        for index,row in enumerate(islice(records,count)):
            if len(row)!=channels+(dataset=='ETTm1'):
                raise ValueError('Native source prefix record width differs from the full-channel contract')
            if dataset=='ETTm1':
                stamp=datetime.strptime(row[0].strip(),'%Y-%m-%d %H:%M:%S')
                times[index]=np.datetime64(stamp,'ns').astype(np.int64)
                row=row[1:]
            values=np.asarray([float(token.strip()) if token.strip() else np.nan for token in row],np.float32)
            values[~np.isfinite(values)]=np.nan
            physical[index]=values
            consumed+=1
        if consumed!=count:
            raise ValueError('Native source ended before the admitted prefix')
        provenance=dict(source_prefix_consumed_records=consumed,source_prefix_consumed_bytes=lines.bytes,
            source_prefix_consumed_sha256=lines.digest.hexdigest(),
            source_prefix_hash_scope='Only exactly floor(.8N) consumed source records, plus the ETTm1 header; protected records and full raw source are not hashed',
            source_prefix_hash_encoding='Decompressed consumed gzip bytes' if dataset=='solar_full137' else 'Consumed original CSV bytes')
    return physical,np.isfinite(physical),times,np.asarray(spec['columns']),provenance


def prepare_dataset(asset_root,dataset,output_root,*,metadata_path,manifest_path=None,master_manifest_path=None):
    if NAME not in Path(output_root).resolve().parts:
        raise ValueError('Prepared outputs must stay in the new search namespace')
    source_locations(asset_root,dataset)
    folder=Path(output_root)/dataset
    if folder.exists():
        raise FileExistsError('Prepared next dataset already exists: '+dataset)
    admitted=admit_source(asset_root,dataset,metadata_path=metadata_path,
        manifest_path=manifest_path,master_manifest_path=master_manifest_path)
    spec,locations=admitted['spec'],admitted['locations']
    physical,observed,times,columns,provenance=_prefix(locations['source'],dataset,spec)
    rows=spec['rows']*4//5
    if (physical.shape!=(rows,len(columns)) or observed.shape!=physical.shape
            or physical.dtype!=np.dtype('float32') or observed.dtype!=np.dtype('bool')
            or columns.tolist()!=list(spec['columns']) or times.shape!=(rows,) or times.dtype!=np.dtype('int64')
            or not np.array_equal(observed,np.isfinite(physical))
            or not np.all(np.diff(times)==spec['interval_seconds']*10**9)):
        raise ValueError('Prepared physical prefix, masks, full channel order or clock differs')
    elapsed=dataset=='solar_full137'
    step=spec['interval_seconds']*10**9
    first=0 if elapsed else np.datetime64(spec['first_timestamp'],'ns').astype(np.int64)
    cutoff=rows*step if elapsed else np.datetime64(spec['exclusive_prefix_end'],'ns').astype(np.int64)
    if times[0]!=first or times[-1]+step!=cutoff or first+rows*step!=cutoff:
        raise ValueError('Native source clock must end exclusively at its registered protected cutoff')
    ranges=base.split_ranges(spec['rows'])
    structure=base.previous.train_structure(physical,observed,columns.tolist(),ranges['A_fit'][1])
    roles=base.role_observability(physical,observed,columns.tolist(),ranges)
    issues=base.readiness_issues(roles)
    if structure['unobserved_columns']:
        issues.append(dict(role='A_fit_rows',unobserved_columns=structure['unobserved_columns']))
    metadata=dict(schema_version=1,status='NOT_READY_UNOBSERVED_CHANNEL' if issues else 'READY',dataset=dataset,
        C=len(columns),L=base.L,H=base.H,interval_seconds=spec['interval_seconds'],period=spec['period'],
        original_rows=spec['rows'],source_rows=spec['rows'],prefix_rows=rows,prefix_only_approved=True,
        candidate_training_admitted=not issues,columns=columns.tolist(),split_ranges=ranges,
        clock_kind='elapsed_nanoseconds' if elapsed else 'source_naive_datetime_nanoseconds',
        clock='Elapsed 600-second row coordinates only; no calendar dates or timezone exist in the source' if elapsed
            else 'Original naive 15-minute source labels preserved; timezone unspecified; no conversion or resampling',
        calendar_dates_invented=False,timezone_invented=False,
        prefix_clock=dict(elapsed_start_seconds=0,elapsed_exclusive_end_seconds=rows*spec['interval_seconds']) if elapsed
            else dict(first_timestamp=spec['first_timestamp'],exclusive_end_timestamp=spec['exclusive_prefix_end']),
        source_column_label_scope='Zero-based original matrix column positions; all137 retained' if elapsed
            else 'All seven original named physical columns in the exact CSV header order',
        source_metadata=admitted['source_metadata'],source_bytes=admitted['source_bytes'],
        source_reference='asset-parent://'+locations['source'].relative_to(Path(asset_root).resolve().parent).as_posix(),
        historical_full_raw_sha256_in_metadata_not_recomputed=admitted['historical_sha256'],source_full_file_hashed=False,
        source_identity_scope='Full N/C admitted from bound historical metadata; current source byte size matches; only consumed prefix content is hashed',
        protection_boundaries=admitted['protection'],train_structure=structure,role_observability=roles,readiness_issues=issues,
        qualification_scope='A_fit_and_A_stop_observability_only',
        column_selection='All original physical channels retained; old selected subsets ignored; no validation-based channel removal',
        normalization='No source scaler used. Consumer fits fresh A_fit/A60 statistics and resets causal history fill at each window',
        missingness='Nonfinite and empty source cells remain NaN with observed=false; targets are never filled',
        protected_numeric_access=dict(T_final=False,historical_protected_tail=False,old_EVAL=False,full_raw_file=False),
        development_exposure='Previously used source/family; adaptive development search is not independent test evidence',
        source_sha256=base.sha256_file(Path(__file__)),
        dependency_source_hashes={path.relative_to(Path(__file__).resolve().parents[2]).as_posix():base.sha256_file(path)
            for path in (Path(base.__file__).resolve(),Path(base.previous.__file__).resolve())},
        prefix_array_hashes={name:base.previous.array_sha256(value) for name,value in
            dict(physical=physical,observed=observed,times=times,columns=columns).items()},**provenance)
    folder.mkdir(parents=True,exist_ok=False)
    output=folder/'PREFIX_PHYSICAL.npz'
    with output.open('xb') as handle:
        np.savez_compressed(handle,physical=np.ascontiguousarray(physical),observed=np.ascontiguousarray(observed),
                            times=np.ascontiguousarray(times),columns=np.ascontiguousarray(columns))
    metadata.update(prefix_sha256=base.sha256_file(output),prefix_bytes=output.stat().st_size)
    with (folder/'DATA_DIAGNOSTIC.json').open('x',encoding='utf-8') as handle:
        json.dump(metadata,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')
    return metadata


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--dataset',choices=tuple(DATASETS),required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--master-manifest',type=Path)
    args=parser.parse_args(argv)
    result=prepare_dataset(args.asset_root,args.dataset,args.output_root,metadata_path=args.metadata,
        manifest_path=args.manifest,master_manifest_path=args.master_manifest)
    print(json.dumps({key:result[key] for key in ('dataset','status','C','prefix_rows','readiness_issues')},allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
