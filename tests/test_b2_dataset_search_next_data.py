import gzip
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    try:return importlib.import_module('research.b2_dataset_search_20261019_v1.data_next')
    except ModuleNotFoundError:pytest.fail('The bounded next-candidate loader is missing')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture(params=['ETTm1','solar_full137'])
def source(tmp_path,monkeypatch,request):
    c=api();dataset=request.param
    spec=dict(c.DATASETS[dataset],rows=3840)
    if dataset=='ETTm1':
        spec['historical_bounds']=[0,2304,2688,3072,3840]
        spec['last_timestamp']='2016-08-09 23:45:00'
        spec['first_timestamp']='2016-07-01 00:00:00'
        spec['exclusive_prefix_end']='2016-08-02 00:00:00'
    monkeypatch.setitem(c.DATASETS,dataset,spec)
    asset=tmp_path/'assets/hierarchical-tsfm-peft';asset.mkdir(parents=True)
    locations=c.source_locations(asset,dataset)
    path=locations['source'];path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if dataset=='solar_full137' else Path.open
    count=spec['rows']*4//5
    with opener(path,'wb') as handle:
        if dataset=='ETTm1':handle.write((','.join(spec['header'])+'\n').encode())
        for row in range(count):
            values=[str((row%17)+col/10) for col in range(len(spec['columns']))]
            if row==20:values[1]='nan'
            if dataset=='ETTm1':
                stamp=str(np.datetime64('2016-07-01T00:00:00')+np.timedelta64(row*900,'s')).replace('T',' ')
                values.insert(0,stamp)
            handle.write((','.join(values)+'\n').encode())
        handle.write(b'FORBIDDEN_PROTECTED_TAIL\n')
    size=path.stat().st_size
    if dataset=='ETTm1':
        metadata=dict(raw_shape=[3840,7],shape=[3840,4],bytes=size,sha256='a'*64,
                      period=96,bounds=spec['historical_bounds'],selected_columns=['0','1','2','3'])
        inventory=dict(datasets=dict(ett_small=dict(files=[dict(name='ETTm1.csv',bytes=size,sha256='a'*64)])))
        master=dict(specs=[dict(id='N03',data=dict(source='ettm1',frequency_minutes=15,bounds=spec['historical_bounds']))])
        write_json(locations['manifest'],inventory);write_json(locations['master_manifest'],master)
    else:
        metadata=dict(source=dict(raw_path=path.relative_to(locations['project']).as_posix(),raw_bytes=size,raw_sha256='a'*64,
            schema=dict(format='gzip csv matrix',raw_shape=[3840,137],timestamp_column=False,selected_columns=['36','10']),
            aggregation=dict(rows_per_hour=6,raw_rows=3840,hourly_rows=640,invented_timestamps=False),
            time_axis=dict(calendar_dates_invented=False,timezone_invented=False),
            split_edges=dict(TEST=dict(start=512,end=640))))
    write_json(locations['metadata'],metadata)
    monkeypatch.setattr(c.base.previous,'train_structure',lambda p,o,cols,end:dict(scope='A_fit_only',unobserved_columns=[]))
    return dict(c=c,dataset=dataset,spec=spec,path=path,asset=asset,locations=locations,
        output=tmp_path/c.NAME/'data',metadata=metadata)


def prepare(source,**overrides):
    loc=source['locations']
    kwargs=dict(metadata_path=loc['metadata'],manifest_path=loc.get('manifest'),
                master_manifest_path=loc.get('master_manifest'))
    kwargs.update(overrides)
    return source['c'].prepare_dataset(source['asset'],source['dataset'],source['output'],**kwargs)


def test_registered_full_shapes_and_protected_prefixes():
    c=api()
    assert (c.DATASETS['ETTm1']['rows'],len(c.DATASETS['ETTm1']['columns']),c.DATASETS['ETTm1']['interval_seconds'])==(69680,7,900)
    assert (c.DATASETS['solar_full137']['rows'],len(c.DATASETS['solar_full137']['columns']),c.DATASETS['solar_full137']['interval_seconds'])==(52560,137,600)
    assert c.DATASETS['ETTm1']['rows']*4//5==55744
    assert c.DATASETS['solar_full137']['rows']*4//5==42048


def test_only_prefix_records_are_consumed_and_saved_for_generic_consumer(source,monkeypatch):
    c=source['c'];consumed=[];original_open=Path.open;original_gzip=gzip.open
    class Guard:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):self.handle.__enter__();return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def readline(self,*args):
            value=self.handle.readline(*args)
            assert b'FORBIDDEN' not in value
            consumed.append(value);return value
    def safe_open(path,*args,**kwargs):
        handle=original_open(path,*args,**kwargs)
        return Guard(handle) if path==source['path'] else handle
    def safe_gzip(path,*args,**kwargs):return Guard(original_gzip(path,*args,**kwargs))
    monkeypatch.setattr(Path,'open',safe_open)
    monkeypatch.setattr(c.gzip,'open',safe_gzip)
    original_sha=c.base.sha256_file
    def safe_sha(path):
        assert Path(path)!=source['path'],'Protected full raw must not be hashed'
        return original_sha(path)
    monkeypatch.setattr(c.base,'sha256_file',safe_sha)
    result=prepare(source)
    assert result['status']=='READY' and result['prefix_rows']==3072
    assert result['C']==len(source['spec']['columns'])
    assert result['source_prefix_consumed_records']==3072
    assert len(consumed)==3072+(source['dataset']=='ETTm1')
    assert result['source_prefix_consumed_sha256']==hashlib.sha256(b''.join(consumed)).hexdigest()
    assert result['source_full_file_hashed'] is False
    assert result['historical_full_raw_sha256_in_metadata_not_recomputed']=='a'*64
    folder=source['output']/source['dataset']
    with np.load(folder/'PREFIX_PHYSICAL.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={'physical','observed','times','columns'}
        assert archive['physical'].shape==(3072,result['C']) and archive['physical'].dtype==np.float32
        assert np.isnan(archive['physical'][20,1]) and not archive['observed'][20,1]
        assert np.diff(archive['times']).tolist()==[source['spec']['interval_seconds']*10**9]*3071
        if source['dataset']=='solar_full137':
            assert archive['times'][0]==0 and archive['columns'].tolist()==[str(i) for i in range(137)]
            assert result['clock_kind']=='elapsed_nanoseconds' and result['calendar_dates_invented'] is False
    from research.b2_dataset_search_20261019_v1 import train
    config,task=train.load_task(folder/'DATA_DIAGNOSTIC.json',folder/'PREFIX_PHYSICAL.npz')
    assert task['split_ranges']['T_final']==(3072,3840)
    assert task['metadata']['T_access'] is False and config['prefix_only_approved'] is True
    assert str(source['asset']) not in json.dumps(result)


@pytest.mark.parametrize('changed',['N','C','size','sha','frequency','cutoff'])
def test_bad_metadata_blocks_numeric_open(source,monkeypatch,changed):
    c=source['c'];value=source['metadata'];loc=source['locations']
    if source['dataset']=='ETTm1':
        if changed=='N':value['raw_shape'][0]+=1
        elif changed=='C':value['raw_shape'][1]=4
        elif changed=='size':value['bytes']+=1
        elif changed=='sha':value['sha256']='b'*64
        elif changed=='frequency':value['period']=24
        else:value['bounds'][-2]+=1
    else:
        item=value['source']
        if changed=='N':item['schema']['raw_shape'][0]+=1
        elif changed=='C':item['schema']['raw_shape'][1]=8
        elif changed=='size':item['raw_bytes']+=1
        elif changed=='sha':item['raw_sha256']='invalid'
        elif changed=='frequency':item['aggregation']['rows_per_hour']=1
        else:item['split_edges']['TEST']['start']+=1
    write_json(loc['metadata'],value)
    original_open=Path.open
    def forbidden(path,*args,**kwargs):
        assert path!=source['path'],'Numeric source opened before metadata admission'
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',forbidden)
    monkeypatch.setattr(c.gzip,'open',lambda *args,**kwargs:pytest.fail('gzip opened before admission'))
    with pytest.raises(ValueError,match='metadata|identity|size|contract|SHA'):
        prepare(source)
    assert not source['output'].exists()


def test_explicit_metadata_is_required_without_source_open(source,monkeypatch):
    c=source['c']
    monkeypatch.setattr(c.base,'_csv_prefix',lambda *args:pytest.fail('Parser must not run'))
    monkeypatch.setattr(c.gzip,'open',lambda *args,**kwargs:pytest.fail('gzip must not open'))
    with pytest.raises(ValueError,match='metadata'):
        prepare(source,metadata_path=None)
    if source['dataset']=='ETTm1':
        with pytest.raises(ValueError,match='metadata'):
            prepare(source,master_manifest_path=None)


def test_missing_frequency_evidence_rejects_before_numeric_open(source,monkeypatch):
    c=source['c'];loc=source['locations']
    if source['dataset']=='ETTm1':
        path=loc['master_manifest'];value=json.loads(path.read_text())
        del value['specs'][0]['data']['frequency_minutes']
    else:
        path=loc['metadata'];value=json.loads(path.read_text())
        del value['source']['aggregation']['hourly_rows']
    write_json(path,value)
    monkeypatch.setattr(c,'_prefix',lambda *args:pytest.fail('Missing metadata must block parser'))
    with pytest.raises(ValueError,match='metadata'):
        prepare(source)


def test_cli_forwards_explicit_metadata_paths_without_preparation(tmp_path,monkeypatch,capsys):
    c=api();calls=[]
    def prepare(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(dataset='ETTm1',status='READY',C=7,prefix_rows=55744,readiness_issues=[])
    monkeypatch.setattr(c,'prepare_dataset',prepare)
    assert c.main(['--asset-root',str(tmp_path/'asset'),'--dataset','ETTm1',
        '--output-root',str(tmp_path/c.NAME/'data'),'--metadata','receipt.json',
        '--manifest','inventory.json','--master-manifest','master.json'])==0
    assert calls[0][1]==dict(metadata_path=Path('receipt.json'),manifest_path=Path('inventory.json'),
                              master_manifest_path=Path('master.json'))
    assert json.loads(capsys.readouterr().out)['prefix_rows']==55744


def test_existing_prefix_and_old_output_namespace_are_preserved(source):
    prepare(source)
    path=source['output']/source['dataset']/'PREFIX_PHYSICAL.npz';before=path.read_bytes()
    with pytest.raises(FileExistsError):prepare(source)
    assert path.read_bytes()==before
    source['output']=source['output'].parents[1]/'old_namespace'
    with pytest.raises(ValueError,match='namespace'):prepare(source)


def test_shifted_source_clock_cannot_move_the_protected_cutoff(source,monkeypatch):
    c=source['c'];original=c._prefix
    def shifted(*args):
        physical,observed,times,columns,provenance=original(*args)
        times+=source['spec']['interval_seconds']*10**9
        return physical,observed,times,columns,provenance
    monkeypatch.setattr(c,'_prefix',shifted)
    with pytest.raises(ValueError,match='clock|cutoff'):
        prepare(source)
    assert not source['output'].exists()
