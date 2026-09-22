from __future__ import annotations
import copy,time
from pathlib import Path
import numpy as np
import torch
from gap_screen.io import *
from gap_screen.data import Panel,save_panel
from gap_screen.runner import Context
from gap_screen import runner
from gap_screen import report
from gap_screen.verify import verify
from conftest import TinyBase


def test_complete_cpu_workflow(tmp_path,cfg,fake_peft,monkeypatch):
    """Train/choose/predict/report/verify 12 tiny trajectories; not a pretrained-model or real-data test."""
    torch.set_num_threads(1)
    repo=tmp_path/'repo';repo.mkdir();(repo/'.git').mkdir();code=Path(__file__).resolve().parents[1]
    ctx=Context(repo,code,cfg,device='cpu')
    write_json(ctx.out/'SOURCE_SEAL.json',{'code':code_manifest(code),'config_hash':digest_object(cfg)})
    write_json(ctx.out/'ENVIRONMENT.json',{'authenticity':'SYNTHETIC_CPU_TEST','note':'Not pretrained Chronos; test control flow only.'})
    torch.manual_seed(77);original=TinyBase()
    def loader(config,device,model_dir=None):return copy.deepcopy(original),.001
    monkeypatch.setattr(runner,'load_base',loader)
    monkeypatch.setattr(runner,'check_loraplus_reference',lambda m,c:{'test_double_only':True})
    for name in cfg['panels']:
        n=100;tt=np.arange(n);values=np.stack([np.sin(tt/5),np.cos(tt/7)],-1).astype('float32')
        oo={'train':np.arange(16,55,3),'val':np.arange(60,72,4),'test':np.arange(76,96,4)}
        sigma=values[:60].std(0).astype('float64');meta={'name':name,'values_sha256':__import__('hashlib').sha256(values.tobytes()).hexdigest(),
            'source_first':'2000-01-01','source_last':'2000-01-05','frequency_seconds':3600,'data_prepare_s':.002,
            'bounds':{'train':[16,60],'val':[60,76],'test':[76,100]}}
        pan=Panel(name,values,tt.astype('int64')*3600*10**9,sigma,['a','b'],16,8,4,oo,meta)
        result,cache=ctx.paths(name);save_panel(pan,cache/'panel.npz');write_json(result/'DATA_AUDIT.json',meta)
        runner.preflight_panel(ctx,name)
        for seed in cfg['seeds']:
            for arm in cfg['arms']:runner.fit(ctx,name,arm,seed)
        runner.seal_selection(ctx,name);runner.predict_test(ctx,name);report.render_panel(ctx,name)
    report.summary_report(ctx);v=verify(ctx)
    assert v['status']=='PASS' and v['main']['updates']==48 and v['smoke']['updates']==12
    for name in cfg['panels']:
        result,_=ctx.paths(name)
        assert len(list((result/'figures').glob('*.png')))==5
        assert 'SYNTHETIC' in (result/'REPORT_KO.md').read_text()
    # Preserve an explicitly synthetic render for author-side visual inspection only.
    import shutil,os
    dest=os.environ.get('GAP_TEST_OUTPUT')
    if dest:shutil.copytree(ctx.out,Path(dest),dirs_exist_ok=True)
