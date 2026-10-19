import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from research.incremental_latent_adaptation_20261019_v1 import common,report


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def fixture_public(root):
    rows=[]
    for i,method in enumerate((*report.CORE,'S','OWN','ALL')):
        for seed in (92601,92602):
            rows.append(dict(run_id=method+str(seed),dataset='jena',method=method,seed=seed,prefix=12,split='V_select',
                             mse=1-i*.01+(seed-92601)*.002,mae=.5+i*.01,selected_epoch=0 if method=='G' else .25))
    # A mean row must not be reweighted as a third independent model.
    rows.append(dict(rows[0],seed='mean',mse=100,mae=100))
    write_csv(root/'RUNS.csv',rows)
    costs=[]
    for method in report.CORE:
        for batch in (1,8):
            costs.append(dict(dataset='jena',method=method,seed=92601,batch_size=batch,device='cuda',
                              median_batch_ms=2*batch,median_origin_ms=2,peak_allocated_MiB=200,peak_reserved_MiB=300))
    write_csv(root/'COST.csv',costs)
    write_csv(root/'COST_TRIALS.csv',[dict(phase='timed',whole_seconds=.002)])
    stats=[]
    for method,factor in (('G',1.),('S',1.25),('OWN',.9),('ALL',.8)):
        for o in range(4):
            for h in range(2):
                for c in range(2):
                    count=int(not (o==0 and h==0 and c==1))
                    stats.append(dict(run_id=method+'92601',dataset='jena',method=method,seed=92601,prefix=12,
                        split='V_select',origin=100+o,horizon=h,channel_index=c,channel_id='c'+str(c),
                        sse=factor*(h+c+1)*count,sae=np.sqrt(factor*(h+c+1))*count,count=count))
    with gzip.open(root/'ERROR_SUFFICIENT_STATS.csv.gz','wt',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    (root/'C_DECISIONS.json').write_text(json.dumps({'status':'SEALED_BEFORE_V'}),encoding='utf8')
    (root/'NUMERICAL_VALIDATION.json').write_text(json.dumps({'status':'PASS'}),encoding='utf8')
    (root/'VALIDATION_REPORT.json').write_text(json.dumps({'status':'PARTIAL'}),encoding='utf8')
    for method in common.METHODS:
        path=root/'runs'/'jena'/(method+'_s92601')/'search_cap3.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(seed=92601,selected_epoch=.25,curves=[
            dict(epoch=0,M_fit_mse=.9,M_select_mse=1.),dict(epoch=.25,M_fit_mse=.8,M_select_mse=.95),
            dict(epoch=1,M_fit_mse=.7,M_select_mse=.96),dict(epoch=3,M_fit_mse=.6,M_select_mse=.97)])),encoding='utf8')


def test_macro_is_equal_channel_not_equal_cell():
    values=[np.array([1.,0.,1.]),np.array([9.,0.,3.])]
    assert report._macro(values)==2
    assert report._macro([np.zeros(3)]) is None
    assert report._display(0)=='0'
    assert report._display(False)=='False'
    assert report._display([0,1])=='[0, 1]'
    assert report._display(None)=='N/A'


def test_mean_rows_are_excluded_and_seed_ranges_are_not_ci():
    rows=[dict(dataset='jena',method='RESIDUAL_F',seed=92601,mse=1.),
          dict(dataset='jena',method='RESIDUAL_F',seed=92602,mse=3.),
          dict(dataset='jena',method='RESIDUAL_F',seed='mean',mse=100.)]
    assert report._mean(report._actual(rows),'jena','RESIDUAL_F')==2


def test_public_cell_statistics_keep_signed_losses_and_mask_counts(tmp_path):
    fixture_public(tmp_path)
    stats=report._sufficient(tmp_path/'ERROR_SUFFICIENT_STATS.csv.gz')
    g=next(v for k,v in stats.items() if k[1]=='G')
    s=next(v for k,v in stats.items() if k[1]=='S')
    gd,sd=report._dimensions(g),report._dimensions(s)
    assert all(gd['channel'][c]-sd['channel'][c]<0 for c in gd['channel'])
    assert len(gd['time_block'])==4
    assert g['h'][(0,1)][2]==3


def test_canonical_curve_uses_max_completed_registered_cap_and_cp0(tmp_path):
    fixture_public(tmp_path)
    first=tmp_path/'runs'/'jena'/'RESIDUAL_F_s92601'/'search_cap3.json'
    contents=json.loads(first.read_text(encoding='utf8'))
    contents['curves'].append(dict(epoch=6,M_fit_mse=.5,M_select_mse=.98))
    first.with_name('search_cap6.json').write_text(json.dumps(contents),encoding='utf8')
    curves=[r for r in report._curves(tmp_path) if r['method']=='RESIDUAL_F']
    assert {r['cap'] for r in curves}=={6}
    assert min(r['epoch'] for r in curves)==0


def test_replay_preserves_numeric_points_all_signs_text_and_png_svg_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DATASETS',('jena',))
    fixture_public(tmp_path)
    manifest=report.build(tmp_path)
    assert manifest['status']=='COMPLETE'
    assert 'VALIDATION_REPORT.json' not in manifest['input_files']
    assert 'NUMERICAL_VALIDATION.json' in manifest['input_files']
    assert len(manifest['figures'])==14
    (tmp_path/'VALIDATION_REPORT.json').write_text(json.dumps({'status':'PASS','replay':'later proof'}),encoding='utf8')
    result=report.replay(tmp_path,tmp_path/'new_folder')
    assert result['status']=='PASS'
    assert result['values_and_text_identical'] and result['figure_bytes_identical']
    assert result['figure_manifest_sha256']==common.sha(tmp_path/'FIGURE_MANIFEST.json')
    points=list(csv.DictReader((tmp_path/'FIGURE_POINTS.csv').open(encoding='utf8')))
    assert any(r['metric']=='MSE_gain' and float(r['value'])<0 for r in points)
    assert any(r['metric']=='MSE_gain' and float(r['value'])>0 for r in points)
    assert any(r['metric']=='mse' and r['x']=='0.0' and r.get('split')=='M_select' for r in points)
    text=(tmp_path/'RESULT_ANALYSIS_KO.md').read_text(encoding='utf8')
    assert '신뢰구간이 아니다' in text
    assert 'N/A' in text
    assert '독립 TEST가 아니다' in text
    assert '|---' not in text


def test_missing_inputs_remain_partial_not_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DATASETS',())
    manifest=report.build(tmp_path)
    assert manifest['status']=='PARTIAL'
    assert 'Missing RUNS.csv' in manifest['warnings']
    assert 'N/A' in (tmp_path/'README_KO.md').read_text(encoding='utf8')
