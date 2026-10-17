import subprocess
import sys
from research.adapter_improvements_20261018_v1 import run
from research.adapter_improvements_20261018_v1.contracts import DEFAULT_LIMITS,atomic_json,seal_job,seal_plan,sha


def test_default_is_source_only_without_loading_torch_or_creating_output(tmp_path):
    code="import sys; from research.adapter_improvements_20261018_v1.run import main; assert main([])==0; assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    assert "SOURCE_ONLY" in result.stdout


def test_group_rechecks_unfinished_pair_instead_of_cached_admission(tmp_path,monkeypatch):
    report=tmp_path/"report"
    jobs=[dict(job_id="candidate",kind="training",family="P_SPLIT",init_kind="PCA",group="pair"),
          dict(job_id="control",kind="training",family="P_LOCAL_CONTROL",init_kind="PCA",group="pair")]
    profile=dict(admitted=True,profiles=[dict(measurements=[dict(family="P_SPLIT",init_kind="PCA",estimated_complete_arm_seconds=4000),
                      dict(family="P_LOCAL_CONTROL",init_kind="PCA",estimated_complete_arm_seconds=4000)])])
    atomic_json(report/"profiles/PROFILE_pair.json",profile)
    ctx=dict(report=report,plan=dict(jobs=jobs),ledger=dict(started_unix=0,limits=DEFAULT_LIMITS,gpu_seconds=0))
    monkeypatch.setattr(run,"reusable",lambda *args:False)
    decision=run.group_admission(jobs[0],ctx,now=14000)
    assert decision["estimated_unfinished_seconds"]==8000
    assert not decision["admitted"]
    assert decision["unfinished_job_ids"]==["candidate","control"]


def test_finished_control_is_not_charged_again_on_partial_group_resume(tmp_path,monkeypatch):
    report=tmp_path/"report"
    jobs=[dict(job_id="candidate",kind="training",family="P_SPLIT",init_kind="PCA",group="pair"),
          dict(job_id="control",kind="training",family="P_LOCAL_CONTROL",init_kind="PCA",group="pair")]
    atomic_json(report/"profiles/PROFILE_pair.json",dict(profiles=[dict(measurements=[
        dict(family="P_SPLIT",init_kind="PCA",estimated_complete_arm_seconds=4000),
        dict(family="P_LOCAL_CONTROL",init_kind="PCA",estimated_complete_arm_seconds=4000)])]))
    monkeypatch.setattr(run,"reusable",lambda path,job,ctx:job["job_id"]=="control")
    decision=run.group_admission(jobs[0],dict(report=report,plan=dict(jobs=jobs),ledger=dict(started_unix=0,limits=DEFAULT_LIMITS,gpu_seconds=0)),now=10000)
    assert decision["estimated_unfinished_seconds"]==4000
    assert decision["admitted"]


def test_failed_scientific_identity_is_not_automatically_retried(tmp_path,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    job=seal_job(dict(job_id="failed",kind="reuse_pca",dependency_ids=[]),{})
    report=tmp_path/"report"
    atomic_json(report/"REUSE_AUDIT.json",dict(status="PASS"))
    plan=seal_plan(dict(source_hashes={},execution_control_sha256=sha(run.__file__),jobs=[job],
        campaign_started_unix=0,reuse_audit_sha256=sha(report/"REUSE_AUDIT.json")))
    plan_path=tmp_path/"plan.json"
    atomic_json(plan_path,plan)
    receipt=report/"receipts/failed.json"
    atomic_json(receipt,dict(status="FAILED",scientific_identity=job["scientific_identity"],reason="original numerical failure"))
    original=receipt.read_bytes()
    monkeypatch.setitem(sys.modules,"resources",SimpleNamespace())
    monkeypatch.setattr(run,"handler",lambda *a: (_ for _ in ()).throw(AssertionError("Unexpected retry")))
    run.execute(dict(repo=Path.cwd(),report=report,cache=tmp_path/"cache",ledger=dict(started_unix=0)),plan_path,worker=True)
    assert receipt.read_bytes()==original
