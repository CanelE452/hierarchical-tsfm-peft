from __future__ import annotations
from pathlib import Path
import json, re, time
import numpy as np
import torch
from .io import *
from .metrics import score, scalar_primary, choose_checkpoint, gain


def verify(ctx,require_all=True):
    ctx.assert_seal(); cfg=ctx.cfg; checked=[]; expected_updates=0; expected_smoke=0
    for name in cfg['panels']:
        result,cache=ctx.paths(name)
        if not (result/'REPORT_RECEIPT.json').exists():
            if require_all: raise Blocked('MISSING_COMPLETE_REPORT '+name)
            checked.append(dict(panel=name,status='NOT_COMPLETE'));continue
        p=ctx.panel(name); pre=read_json(result/'PREFLIGHT.json');assert pre['status']=='PASS'
        expected_smoke+=6; expected_updates+=len(cfg['arms'])*len(cfg['seeds'])*cfg['max_steps']
        manifest=read_json(result/'PREDICTIONS_MANIFEST.json'); seal=read_json(result/'SELECTION_SEAL.json')
        assert manifest['complete'] and manifest['all_saved_before_scoring']
        assert manifest['selection_sha256']==sha(result/'SELECTION_SEAL.json')
        for path,h in manifest['predictions'].items():assert sha(ctx.repo/path)==h,path
        y=p.targets('test'); levels=np.array(read_json(result/'MODEL_RECEIPT.json')['levels'])
        seedscores={}
        for seed in cfg['seeds']:
            schedules=[]; initials={}
            for arm in cfg['arms']:
                key=f'{name}_{arm}_{seed}'; f=read_json(result/'fits'/f'{key}.json')
                assert f['complete'] and f['frozen_unchanged'] and f['changed_trainable_tensors']>0
                assert len(f['trace'])==cfg['max_steps'] and f['trace'][-1]['step']==cfg['max_steps']
                assert f['selected_step']==choose_checkpoint(f['curves'])['step']
                schedules.append(f['train_schedule_sha256']);initials[arm]=f['initial_hash']
                for c in f['curves']:
                    path=ctx.repo/c['state_path'];assert sha(path)==c['state_sha256']
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    assert state['step']==c['step']
                    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
                    for v in state['optimizer']['state'].values():
                        for t in v.values():
                            if isinstance(t,torch.Tensor):assert torch.isfinite(t).all()
                step=f['selected_step']; q=np.load(cache/'test_predictions'/f'{key}_{step}.npz')['q']
                m,per=score(q,y,p.sigma,levels); seedscores[(arm,seed)]=m['primary']
                assert np.isclose(scalar_primary(q[:2,:2],y[:2,:2],p.sigma[:2],levels),score(q[:2,:2],y[:2,:2],p.sigma[:2],levels)[0]['primary'],rtol=1e-10)
            assert len(set(schedules))==1,'minibatch schedule differs'
            assert initials['LORA']==initials['LORAPLUS'],'LoRA initial state differs'
        # Recompute effects from the independent source scores, not from rounded report numbers.
        import csv
        for r in csv.DictReader((result/'EFFECTS.csv').open(encoding='utf-8')):
            a,b=r['candidate'],r['baseline']
            if b not in cfg['arms']:
                q=np.load(cache/'test_predictions'/(b+'.npz'))['q']; bm=score(q,y,p.sigma,levels)[0]['primary']
            else:bm=np.mean([seedscores[b,s] for s in cfg['seeds']])
            am=np.mean([seedscores[a,s] for s in cfg['seeds']])
            assert np.isclose(float(r['mean_gain_pct']),gain(am,bm),rtol=1e-9,atol=1e-9)
        # A report with broken image links is not complete.
        md=(result/'REPORT_KO.md').read_text(encoding='utf-8')
        image_links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
        assert len(image_links)>=5
        for link in image_links:assert (result/link).is_file(),link
        checked.append(dict(panel=name,status='PASS',fits=len(cfg['arms'])*len(cfg['seeds']),
            checkpoint_values=True,optimizer_states_finite=True,scalar_metrics=True,ratio_of_means=True,
            matched_schedules=True,matched_lora_initials=True,report_images_exist=True))
    main=require_complete_ledger(ctx.out/'UPDATE_LEDGER.jsonl',expected_updates)
    smoke=require_complete_ledger(ctx.out/'SMOKE_LEDGER.jsonl',expected_smoke)
    if require_all:assert len([r for r in checked if r['status']=='PASS'])==len(cfg['panels'])
    record=dict(status='PASS' if require_all else 'PARTIAL_CHECKED',panels=checked,main=main,smoke=smoke,
        timestamp=time.time(),source_unchanged=True,local_raw_predictions_verified=True,
        independent_new_machine_replication=False,scope='Saved artifacts + scalar CPU checks; GPU preflight recorded separately')
    write_json(ctx.out/'VERIFICATION.json',record)
    return record
