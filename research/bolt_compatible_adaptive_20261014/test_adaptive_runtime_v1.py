import json
import time
import pytest
import adaptive_runtime_v1 as rt

def records(tmp_path,monkeypatch):
    plan = tmp_path/'plan.md'
    plan.write_text('fixed plan')
    authp, runp = tmp_path/'auth.json',tmp_path/'run.json'
    start = time.time()-1
    auth = dict(status='APPROVED',origin='human_user_response',response_identifier=rt.EXPECTED_REPLY_ID,
                user_response='계획과 자원 상한 승인',approved_plan_sha256=rt.sha(plan),
                runtime_source_sha256=rt.sha(rt.CORE_RUNTIME),approved_unix=start-1,limits=dict(rt.MAX_LIMITS))
    rt.write_new(authp,auth)
    run = dict(authorization_sha256=rt.sha(authp),runtime_source_sha256=rt.sha(rt.CORE_RUNTIME),
               started_unix=start,expires_unix=start+14400,baseline_storage_bytes=0,
               storage_roots=[str(tmp_path)],protected_records=[dict(path=str(plan),sha256=rt.sha(plan))])
    rt.write_new(runp,run)
    monkeypatch.setattr(rt,'AUTHORIZATION',authp)
    monkeypatch.setattr(rt,'RUN_MANIFEST',runp)
    monkeypatch.setattr(rt,'APPROVED_PLAN',plan)
    return authp,runp,auth,run

def test_real_shape_valid_authorization(tmp_path,monkeypatch):
    _,_,auth,run = records(tmp_path,monkeypatch)
    assert rt.authorized_records() == (auth,run)

@pytest.mark.parametrize('change',[{'user_response':'yes'}, {'response_identifier':'old'},
    {'status':'PENDING'}, {'runtime_source_sha256':'0'*64}])
def test_approval_or_source_substitution_rejected(tmp_path,monkeypatch,change):
    authp,runp,auth,run = records(tmp_path,monkeypatch)
    auth.update(change)
    authp.write_text(json.dumps(auth))
    run['authorization_sha256'] = rt.sha(authp)
    runp.write_text(json.dumps(run))
    with pytest.raises(rt.AuthorizationRequired):
        rt.authorized_records()

def test_cap_increase_rejected(tmp_path,monkeypatch):
    authp,runp,auth,run = records(tmp_path,monkeypatch)
    auth['limits']['cpu_process_wall_seconds'] += 1
    authp.write_text(json.dumps(auth))
    run['authorization_sha256'] = rt.sha(authp)
    runp.write_text(json.dumps(run))
    with pytest.raises(rt.AuthorizationRequired):
        rt.authorized_records()

def test_clock_extension_rejected(tmp_path,monkeypatch):
    _,runp,_,run = records(tmp_path,monkeypatch)
    run['expires_unix'] += 1
    runp.write_text(json.dumps(run))
    with pytest.raises(rt.AuthorizationRequired):
        rt.authorized_records()

def test_write_new_preserves_existing(tmp_path):
    p = tmp_path/'result.json'
    rt.write_new(p,{'old':1})
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        rt.write_new(p,{'new':2})
    assert p.read_bytes() == before

def test_storage_counts_payload(tmp_path):
    (tmp_path/'a').write_bytes(b'1'*101)
    sub = tmp_path/'sub'
    sub.mkdir()
    (sub/'b').write_bytes(b'2'*202)
    assert rt.storage_bytes([tmp_path]) == 303
