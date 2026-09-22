"""Publish safety tests use temporary files only. They never call a real Git remote."""
import importlib.util,json
from pathlib import Path
import pytest
from gap_screen.io import Blocked,write_json

P=Path(__file__).resolve().parents[1]/'publish.py'
spec=importlib.util.spec_from_file_location('gap_publisher_tests',P)
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)


def layout(tmp_path,status='PASS',auth='REAL_MODEL_RUN'):
    cfg={'experiment_id':'test_screen','max_single_artifact_bytes':100000}
    code=tmp_path/'experiments/test_screen';out=tmp_path/'results/test_screen'
    code.mkdir(parents=True);out.mkdir(parents=True)
    write_json(out/'VERIFICATION.json',{'status':status})
    write_json(out/'ENVIRONMENT.json',{'authenticity':auth})
    return cfg,code,out


def test_publish_refuses_incomplete(tmp_path,monkeypatch):
    cfg,code,out=layout(tmp_path,'INCOMPLETE')
    monkeypatch.setattr(pub,'git',lambda *a,**kw:pytest.fail('git must not be touched'))
    with pytest.raises(Blocked,match='VERIFICATION'):pub.publish(tmp_path,code,cfg)


def test_publish_refuses_synthetic(tmp_path,monkeypatch):
    cfg,code,out=layout(tmp_path,auth='SYNTHETIC_CPU_TEST')
    monkeypatch.setattr(pub,'git',lambda *a,**kw:pytest.fail('git must not be touched'))
    with pytest.raises(Blocked,match='SYNTHETIC'):pub.publish(tmp_path,code,cfg)


def test_publish_fresh_verification_precedes_git(tmp_path,monkeypatch):
    cfg,code,out=layout(tmp_path)
    import gap_screen.verify
    def stop(*a,**kw):raise Blocked('CHANGED_CHECKPOINT')
    monkeypatch.setattr(gap_screen.verify,'verify',stop)
    monkeypatch.setattr(pub,'git',lambda *a,**kw:pytest.fail('git must not be touched'))
    with pytest.raises(Blocked,match='CHANGED_CHECKPOINT'):pub.publish(tmp_path,code,cfg)


def test_publish_refuses_wrong_remote(tmp_path,monkeypatch):
    cfg,code,out=layout(tmp_path)
    import gap_screen.verify
    monkeypatch.setattr(gap_screen.verify,'verify',lambda *a,**kw:{'status':'PASS'})
    calls=[]
    def fake(repo,*a,**kw):
        calls.append(a)
        if a==('diff','--cached','--name-only'):return ''
        if a==('remote','get-url','origin'):return 'https://github.com/other/project.git'
        pytest.fail('unexpected git call')
    monkeypatch.setattr(pub,'git',fake)
    with pytest.raises(Blocked,match='WRONG_REMOTE'):pub.publish(tmp_path,code,cfg)
    assert not any('push' in a for a in calls)


def test_publish_refuses_unrelated_staging(tmp_path,monkeypatch):
    cfg,code,out=layout(tmp_path)
    import gap_screen.verify
    monkeypatch.setattr(gap_screen.verify,'verify',lambda *a,**kw:{'status':'PASS'})
    monkeypatch.setattr(pub,'git',lambda repo,*a,**kw:'unrelated.py' if a==('diff','--cached','--name-only') else pytest.fail('must stop before remote'))
    with pytest.raises(Blocked,match='STAGED'):pub.publish(tmp_path,code,cfg)
