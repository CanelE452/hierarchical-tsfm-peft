"""CPU synthetic fixtures only: never inspect the running campaign's arrays."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.raw_context_multiscale_20261018_v1 import calibration, common, report
from research.raw_context_multiscale_mcal_diagnostics_20261019_v1 import run


def scaffold(tmp_path):
    ctx = dict(common.paths(tmp_path), asset=tmp_path/"asset")
    ctx["cache"].mkdir(parents=True)
    ctx["report"].mkdir(parents=True)
    sources = {}
    for index in range(13):
        path = tmp_path/f"research/frozen_primary/source{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"FIXTURE = {index}\n", encoding="utf8")
        sources[path.relative_to(tmp_path).as_posix()] = common.sha(path)
    common.atomic_json(ctx["report"]/"PRIMARY_PLAN.json", dict(source_files=sources))
    common.atomic_json(ctx["report"]/"SUPPLEMENT_PLAN.json", dict(synthetic=True))
    common.atomic_json(ctx["report"]/"INDEPENDENT_PLAN.json", dict(synthetic=True))
    common.atomic_json(ctx["cache"]/"CORE_EXECUTION_START.json", dict(started_unix=10., limits=dict(wall_seconds=3600.)))
    diagnostic = dict(split="M_cal", models=40, alpha_models=32, head_models=8, head_batch_size=64, device="cpu",
        new_F_inference=False, V_selection=False, new_V_forecasts=False, T_access=False,
        metrics=list(run.METRICS), tolerance=dict(metric_atol=1e-12, sufficient_stats_atol=1e-12, sufficient_stats_rtol=1e-13))
    scope = dict(status="REGISTERED_BEFORE_NEW_V", scientific_contract_changed=False, selection_changes=False,
        new_training_updates=0, diagnostic_scope=diagnostic, primary_source_files=sources,
        supplement_source_files={}, independent_source_files={},
        primary_plan_binding=common.binding(ctx["report"]/"PRIMARY_PLAN.json", ctx),
        supplement_plan_binding=common.binding(ctx["report"]/"SUPPLEMENT_PLAN.json", ctx),
        independent_plan_binding=common.binding(ctx["report"]/"INDEPENDENT_PLAN.json", ctx),
        original_execution_start_binding=common.binding(ctx["cache"]/"CORE_EXECUTION_START.json", ctx), original_deadline_unix=3610.)
    scope_path = ctx["report"]/"MCAL_FIT_DIAGNOSTIC_SCOPE_001.json"
    common.atomic_json(scope_path, scope)
    code = {}
    for filename in ("__init__.py", "run.py"):
        path = tmp_path/"research"/run.NAME/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC = True\n", encoding="utf8")
        code[path.relative_to(tmp_path).as_posix()] = common.sha(path)
    test = tmp_path/"tests/test_raw_context_multiscale_mcal_diagnostics.py"
    test.parent.mkdir(parents=True)
    test.write_text("SYNTHETIC = True\n", encoding="utf8")
    preflight_path = ctx["report"]/"MCAL_FINAL_FIT_PREFLIGHT_001.json"
    common.atomic_json(preflight_path, dict(status="PASS_SYNTHETIC_AND_FROZEN_CODE", scope_binding=common.binding(scope_path, ctx),
        source_files=code, test_source_files={test.relative_to(tmp_path).as_posix():common.sha(test)}, actual_evaluation_performed=False))
    common.atomic_json(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="COMPLETE", exit_code=0))
    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json"):
        common.atomic_json(ctx["report"]/name, dict(status="COMPLETE"))
    bind_audit(ctx, scope)
    (ctx["cache"]/"EVENTS.jsonl").write_text('{"stage":"preserved_primary_synthetic_event"}\n', encoding="utf8")
    return ctx, scope_path, preflight_path


def bind_audit(ctx, scope):
    common.atomic_json(ctx["report"]/"VALIDATION_REPORT.json", dict(status="PASS", models=56,
        manifest_bindings={**{name:common.binding(ctx["report"]/name, ctx) for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json")},
                           "registered_plan":scope["primary_plan_binding"]}))


def arrays():
    rng = np.random.default_rng(315)
    past = rng.normal(size=(131,8,2)).astype(np.float32)
    g = np.repeat(past[:, -1:],4,axis=1)
    b = g+.2
    target = g+.15
    mask = np.ones(g.shape,bool)
    mask[::3,:,1] = False
    return past,g,b,target,mask


def fit_statistics(task, end):
    physical = task["physical"][:end]
    observed = task["observed"][:end]
    mean = np.array([physical[:,c][observed[:,c]].mean() for c in range(physical.shape[1])],np.float32)
    std = np.ones_like(mean)
    return mean,std,None


def fixture_models(ctx):
    tasks, records = {}, []
    scaler = lambda mean,std:common.fingerprint(dict(mean=common.array_identity(mean),std=common.array_identity(std)))
    ctx["modules"] = dict(data=SimpleNamespace(fit_statistics=fit_statistics),fit=SimpleNamespace(scaler_fingerprint=scaler))
    for dataset in common.EXPECTED:
        physical = np.arange(80,dtype=np.float32).reshape(40,2)/10
        observed = np.ones(physical.shape,bool)
        observed[::5,1] = False
        task = dict(dataset=dataset,L=8,H=4,columns=["x","y"],physical=physical,observed=observed,
                    data_sha256="synthetic-"+dataset,split_ranges=dict(M_cal=(16,36)))
        tasks[dataset] = task
        origins = np.arange(16,33,dtype=np.int64)
        mean,std,_ = fit_statistics(task,16)
        rows = origins[:,None]+np.arange(4)[None]
        mask = observed[rows]
        target = np.where(mask,(physical[rows]-mean)/std,0).astype(np.float32)
        parents, cache = {}, {}
        for role,seed in (("Raw",None),("G",92601),("S",92601),("S",92602)):
            identifier=f"{role}_{dataset}_{seed}"
            spec=dict(family="RAW" if role=="Raw" else "DIRECT_G" if role=="G" else "B2",run_id=identifier)
            if role != "Raw": spec["artifact_sha256"]="model-"+identifier
            path=ctx["cache"]/"parent_fixtures"/(identifier+".npz")
            prediction=(np.where(mask,target,0)+(.3 if role=="Raw" else .1 if role=="G" else .2)).astype(np.float32)
            metadata=dict(route_spec=spec,split="M_cal",parent_fit_end=16,data_sha256=task["data_sha256"],scaler_sha256=scaler(mean,std))
            common.save_prediction(path,prediction=prediction,target=target,observed=mask,origins=origins,task=task,metadata=metadata)
            parents[role,seed]=dict(run_id=identifier,route_spec=spec)
            cache[role,seed]=dict(prediction=prediction,binding=common.binding(path,ctx))
        split=calibration.calibration_split(task)
        past=report.canonical_past(task,origins,mean,std)
        for seed in common.SEEDS:
            bp,bv=parents["S",seed],cache["S",seed]
            for role in ("Raw","G"):
                gp,gv=parents[role,None if role=="Raw" else 92601],cache[role,None if role=="Raw" else 92601]
                diagnostic=calibration.fit_alpha_diagnostic(gv["prediction"],bv["prediction"],target,mask,split["fit_indices"],split["select_indices"])
                for label,alpha in (("alpha0",0.),("alpha05",.5),("alpha1",1.),("fitted",diagnostic["refit_alpha"]["alpha"])):
                    identifier=f"alpha_{dataset}_{role}_{seed}_{label}"
                    records.append(dict(run_id=identifier,dataset=dataset,seed=seed,mode="alpha",base_role=role,alpha_label=label,
                        alpha=alpha,route_spec=dict(alpha=alpha),parent_g=gp,parent_b=bp,diagnostic=diagnostic,
                        parent_g_prediction_binding=gv["binding"],parent_b_prediction_binding=bv["binding"]))
            gp,gv=parents["G",92601],cache["G",92601]
            for control in (False,True):
                mode="past_control" if control else "conditional"
                identifier=f"head_{dataset}_{seed}_{mode}"
                head=calibration._new_head(8,4,seed)
                with torch.no_grad():head.output.bias.fill_(.05)
                state={key:value.detach().clone() for key,value in head.state_dict().items()}
                search=dict(selected_epoch=1.,curves=[dict(endpoint_epochs=0.,M_fit_mse=1.3,M_fit_mae=.9,M_select_mse=1.4,M_select_mae=1.),
                    dict(endpoint_epochs=1.,M_fit_mse=1.1,M_fit_mae=.8,M_select_mse=1.2,M_select_mae=.9)])
                receipt=dict(final_state_sha256=run.state_hash(state),selected_epoch=1.,mean_batch_loss=.77)
                contract=dict(dataset=dataset,seed=seed,control=control,data_sha256=task["data_sha256"],
                    scaler=dict(mean=common.array_identity(mean),std=common.array_identity(std)),
                    origins=common.array_identity(origins),past=common.array_identity(past),
                    calibration_source_sha256=common.sha(calibration.__file__),g=gv["binding"],b=None if control else bv["binding"])
                path=ctx["cache"]/"head_fixtures"/(identifier+".pt")
                path.parent.mkdir(exist_ok=True)
                torch.save(dict(state=state,head_config=dict(length=8,horizon=4,hidden=32),search=search,receipt=receipt,contract=contract),path)
                records.append(dict(run_id=identifier,dataset=dataset,seed=seed,mode=mode,parent_g=gp,parent_b=None if control else bp,
                    parent_g_prediction_binding=gv["binding"],parent_b_prediction_binding=None if control else bv["binding"],
                    route_spec=dict(head_state_sha256=common.sha(path)),head_binding=common.binding(path,ctx),search=search,refit=receipt))
    common.atomic_json(ctx["cache"]/"CALIBRATION_MODELS.json",records)
    scored=[dict(record,score=dict(mse=.987,mae=.654)) for record in records]
    scored.extend(dict(run_id=f"synthetic_body_{index}",mode="S") for index in range(16))
    common.atomic_json(ctx["report"]/"SCORES.json",dict(status="COMPLETE",arms=scored))
    return tasks,scored


def test_scalar_fsum_and_separate_numpy_stats_match_masked_equal_channel_metric():
    _,g,_,target,mask=arrays()
    target=target.copy();target[~mask]=np.nan
    actual=run.independent_statistics(g,target,mask)
    vectorized=common.statistics(g,target,mask)
    run.check_statistics(actual,vectorized,dict(metric_atol=1e-12,sufficient_stats_atol=1e-12,sufficient_stats_rtol=1e-13))
    assert actual["mse"]==pytest.approx(.15**2,abs=1e-8)
    bad=copy.deepcopy(vectorized);bad["horizon_channel_count"][0][0]+=1
    with pytest.raises(ValueError,match="counts differ"):run.check_statistics(actual,bad,dict(metric_atol=1e-12,sufficient_stats_atol=1e-12,sufficient_stats_rtol=1e-13))
    with pytest.raises(ValueError,match="bool"):run.independent_statistics(g,target,mask.astype(np.uint8))


@pytest.mark.parametrize("control",[False,True])
def test_cpu64_chunks_match_existing_predict_head_and_preserve_rng(control):
    past,g,b,_,_=arrays()
    head=calibration._new_head(8,4,92601)
    with torch.no_grad():head.output.weight.fill_(.03)
    state=head.state_dict()
    rng=torch.random.get_rng_state().clone()
    expected=calibration.predict_head(state,past,g,None if control else b,control=control,batch_size=64)
    actual=run.predict_head_bounded(state,past,g,None if control else b,control=control,deadline=None)
    np.testing.assert_array_equal(actual,expected)
    assert torch.equal(torch.random.get_rng_state(),rng)


def test_deadline_stops_before_any_head_forward(monkeypatch):
    past,g,b,_,_=arrays()
    monkeypatch.setattr(run,"time",lambda:100.)
    monkeypatch.setattr(calibration,"predict_head",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("head called")))
    with pytest.raises(TimeoutError,match="Original execution deadline"):
        run.predict_head_bounded({},past,g,b,control=False,deadline=99.)


def test_nonterminal_and_missing_preflight_fail_before_numpy_or_head_access(tmp_path,monkeypatch):
    ctx,scope,preflight=scaffold(tmp_path)
    monkeypatch.setattr(run,"time",lambda:100.)
    monkeypatch.setattr(np,"load",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("NPZ opened")))
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("head opened")))
    common.atomic_json(ctx["cache"]/"TERMINAL_CORE_002.json",dict(status="FAILED",exit_code=1))
    with pytest.raises(ValueError,match="Main002"):run.evaluate(ctx,scope,preflight)
    data=common.read(preflight);data["scope_binding"]["sha256"]="changed";common.atomic_json(preflight,data)
    with pytest.raises(ValueError,match="bound to this scope"):run.evaluate(ctx,scope,preflight)


def test_registered_source_change_and_deadline_change_fail_closed(tmp_path,monkeypatch):
    ctx,scope,preflight=scaffold(tmp_path)
    monkeypatch.setattr(run,"time",lambda:100.)
    data=common.read(scope)
    first=next(iter(data["primary_source_files"]))
    (ctx["repo"]/first).write_text("changed",encoding="utf8")
    with pytest.raises(ValueError,match="Frozen diagnostic dependency"):run.gates(ctx,scope,preflight)


def test_full40_evaluation_separates_fit_choice_batchloss_and_V_without_EVENTS(tmp_path,monkeypatch):
    ctx,scope,preflight=scaffold(tmp_path)
    tasks,scored=fixture_models(ctx)
    bind_audit(ctx,common.read(scope))
    before=(ctx["cache"]/"EVENTS.jsonl").read_bytes()
    monkeypatch.setattr(run,"time",lambda:100.)
    monkeypatch.setattr(common,"load_task",lambda context,dataset:tasks[dataset])
    result=run.evaluate(ctx,scope,preflight)
    assert result["status"]=="PASS" and result["models"]==40
    assert result["reused_parent_cache_count"]==8
    assert result["GPU_seconds_added"]==0 and result["primary_EVENTS_appended"] is False
    assert result["V_prefix_numeric_decoded_by_existing_bounded_loader"] is True
    assert result["new_V_forecasts"] is False and result["new_training_updates"]==0
    assert (ctx["cache"]/"EVENTS.jsonl").read_bytes()==before
    rows=report._read_csv(ctx["report"]/run.CSV_NAME)
    assert len(rows)==40 and sum(row["mode"]=="alpha" for row in rows)==32
    assert all(row["V_report_mse"]==.987 and row["V_report_mae"]==.654 for row in rows)
    assert all(all(key in row for key in run.METRICS) for row in rows)
    heads=[row for row in rows if row["mode"]!="alpha"]
    assert all(row["M_cal_mean_batch_loss_not_full_metric"]==.77 for row in heads)
    assert all(row["M_cal_final_model_mse"]!=.77 for row in heads)
    assert all(row["M_select_selected_model_mse"]==1.2 for row in heads)
    assert len(list((ctx["cache"]/"mcal_final_fit_diagnostics_001/predictions").glob("*.npz")))==40
    assert len(result["evidence"])==40
    again=run.evaluate(ctx,scope,preflight)
    assert again==result and (ctx["cache"]/"EVENTS.jsonl").read_bytes()==before


def test_parent_target_tamper_and_fresh_head_state_tamper_are_rejected(tmp_path,monkeypatch):
    ctx,_,_=scaffold(tmp_path)
    tasks,arms=fixture_models(ctx)
    task=tasks["jena"]
    alpha=next(arm for arm in arms if arm.get("mode")=="alpha" and arm["dataset"]=="jena")
    original=common.resolve_binding(alpha["parent_g_prediction_binding"],ctx)
    with np.load(original,allow_pickle=False) as archive:content={key:archive[key] for key in archive.files}
    content["target_normalized"]=content["target_normalized"].copy();content["target_normalized"][0,0,0]+=.1
    target=ctx["cache"]/"tampered.npz"
    with target.open("wb") as stream:np.savez(stream,**content)
    with pytest.raises(ValueError,match="canonical origins/targets"):
        run.load_parent(ctx,task,common.binding(target,ctx),alpha["parent_g"]["route_spec"],{},None)
    head=next(arm for arm in arms if arm.get("mode")=="conditional" and arm["dataset"]=="jena")
    path=common.resolve_binding(head["head_binding"],ctx)
    saved=torch.load(path,map_location="cpu",weights_only=False)
    saved["state"]["output.bias"][0]+=.1
    changed=ctx["cache"]/"tampered_head.pt";torch.save(saved,changed)
    head["head_binding"]=common.binding(changed,ctx)
    head["route_spec"]["head_state_sha256"]=head["head_binding"]["sha256"]
    with pytest.raises(ValueError,match="Fresh final head state"):
        run.final_prediction(ctx,task,head,{}, {},None)


def test_V_metrics_are_copied_only_and_state_selection_is_not_refitted(tmp_path,monkeypatch):
    ctx,scope,preflight=scaffold(tmp_path)
    tasks,_=fixture_models(ctx);bind_audit(ctx,common.read(scope))
    monkeypatch.setattr(run,"time",lambda:100.)
    monkeypatch.setattr(common,"load_task",lambda context,dataset:tasks[dataset])
    for name in ("head_search","head_refit","fit_alpha_diagnostic"):
        monkeypatch.setattr(calibration,name,lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("new fitting attempted")))
    result=run.evaluate(ctx,scope,preflight)
    assert result["new_F_inference"] is False and result["V_selection"] is False
    assert result["T_access"] is False
