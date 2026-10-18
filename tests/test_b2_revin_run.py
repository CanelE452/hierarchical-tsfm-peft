from contextlib import contextmanager
from pathlib import Path
from research.b2_revin_20261018_v1 import run, contracts, training_jobs, audit
from research.adapter_improvements_20261018_v1 import audit as old_audit


def test_all_registered_search_and_normalization_sealed_before_any_new_v(tmp_path,monkeypatch):
    events=[]
    cache=tmp_path/'cache';report=tmp_path/'report'
    cache.mkdir();report.mkdir()
    (report/'IMPLEMENTATION_PLAN_KO.md').write_text('test',encoding='utf-8')
    cp=cache/'parent.pt';cp.write_bytes(b'fixture')
    ctx=dict(cache=cache,report=report,repo=tmp_path,asset=tmp_path,modules={},scientific_source_hashes={})
    def result(job,cap):
        return dict(status='COMPLETE',spec=dict(dataset=job['cohort'],seed=job['seed'],revin=job['revin']),
            curves=[dict(epoch=epoch,astop_mse=10-epoch-(.1 if job['revin'] else 0)) for epoch in (0,.25,1,3) if epoch<=cap]+
                   ([dict(epoch=6,astop_mse=4-(.1 if job['revin'] else 0))] if cap==6 else []),
            selected_epoch=cap,parent_checkpoint=dict(path=str(cp)))
    controls={contracts.job_id(name,False,seed):result(dict(cohort=name,seed=seed,revin=False),3)
              for name in contracts.COHORTS for seed in contracts.SEEDS}
    monkeypatch.setattr(audit,'run_audit',lambda ctx:(dict(controls=controls),{name:dict(H=1) for name in contracts.COHORTS}))
    class Foundation:
        def assert_frozen_unchanged(self):pass
    monkeypatch.setattr(old_audit,'load_foundation',lambda *args:Foundation())
    @contextmanager
    def fake_lease(*args):yield None
    monkeypatch.setattr(run,'lease',fake_lease)
    def search(job,ctx,epochs=contracts.INITIAL_EPOCHS,parent_checkpoint=None):
        cap=int(epochs[-1]);events.append(('search',job['cohort'],job['family'],cap))
        return result(job,cap)
    def refit(job,ctx,search_result,cap):
        events.append(('V',job['cohort'],job['family'],cap));return dict(status='COMPLETE')
    monkeypatch.setattr(training_jobs,'search',search)
    monkeypatch.setattr(training_jobs,'_refit_predict',refit)
    run.execute(ctx)
    first_v=next(i for i,event in enumerate(events) if event[0]=='V')
    assert all(i<first_v for i,event in enumerate(events) if event[0]=='search')
    assert len([event for event in events if event[0]=='search' and event[2]=='B0'])==2
