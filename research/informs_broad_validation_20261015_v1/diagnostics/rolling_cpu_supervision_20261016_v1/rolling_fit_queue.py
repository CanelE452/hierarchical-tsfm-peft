"""Private two-slot rolling CPU queue. Default: no experiment or child access."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

DEADLINE=1792222550
GIB=1024**3
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
CORE_SHA="d7707ed7fcd4f0fb09ecbc12ee9caae24957726b97b20d2fd6ed0a1466a5fdca"
ADMISSION_SHA="85b4456415e6ced69444c6bdb6ffb04999ce45bc4ec0da8df41bb7a2f571e994"
GROUPS={
 "multik":dict(name="postT_V_only_cpu_multik_20261016_v1",runner="multik_runner.py",runner_sha="44cd5e3fed5f4ad8a2110c54c9118063c6838b68d9059dea46f6b33bc32be10a",
  plan_sha="2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec",registry_sha="7bb634cf469105c455d552ffedeadbfcb324c2803a95829891589c7fa10c5144",
  verifier="verify_multik_outputs.py",verifier_sha="0455361c307b7142cc15d5f820fef9cb827271b1df5669e1724f7ab5a9666238"),
 "required":dict(name="postT_V_only_cpu_required_minima_20261016_v1",runner="required_minima_runner.py",runner_sha="e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b",
  plan_sha="04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152",registry_sha="fc5f4012a6ced7a1849fcd6474f67d81b600dadaa0014f56df4cda4b8a140f29",
  verifier="verify_required_minima_outputs.py",verifier_sha="c48cfc2bb738798b8e316fdc86de6d160f954ed634ed0831c5427afdcb3616a5"),
 "family":dict(name="postT_V_only_cpu_family_minima_20261016_v1",runner="family_minima_runner.py",runner_sha="16e3a30d678b61ac49328ca1d3f31386bb03cbc4947f2c396b0fc1622258fbd9",
  plan_sha="e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10",registry_sha="631f294006c88a1487b9dc14f731beb4437279e8c3b8d8f1a26d79762b0ed6b2",
  verifier="verify_family_minima_outputs.py",verifier_sha="907488e3bf92e3ac13e90f0fc30c4dac09e241e0b51f7ba7c0b274e654dc45fa")}


class GlobalStop(RuntimeError):pass


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def safe(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in [path]+list(path.parents)):raise ValueError("Unsafe linked path")
    return path.resolve()


def read(path):return json.loads(safe(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def load(path,digest,name):
    if sha(path)!=digest:raise GlobalStop("Frozen dependency changed: "+Path(path).name)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def pin(pins,path,digest=None):
    path=safe(path);actual=sha(path)
    if digest is not None and actual!=digest:raise GlobalStop("Pinned file differs: "+path.name)
    if str(path) in pins and pins[str(path)]!=actual:raise GlobalStop("Conflicting pin")
    pins[str(path)]=actual


def command(config,group,script,mode,run_id,output,probe=None):
    entry=GROUPS[group];public,cache,asset=(config[k] for k in ("public_root","cache_root","asset_root"))
    result=[config["python"],"-X","utf8","-u","-B",str(Path(public)/entry["name"]/script),"--public-root",public,
        "--cache-root",cache,"--asset-root",asset,"--plan-sha",entry["plan_sha"],"--mode",mode,"--run-id",run_id,
        "--deadline",str(DEADLINE),"--output",str(output),"--execute"]
    if probe is not None:result.extend(["--probe-result",probe["file"],"--probe-result-sha",probe["sha256"]])
    return result


def single_spec(config,group,identity,argv,token):
    root=Path(config["cache_root"])/GROUPS[group]["name"]
    return dict(schema="CPU_GROUP_SUPERVISOR_v1",asset_root=config["asset_root"],cache_root=str(root),
        output_root=str(root/("supervisor_rolling_root_v1_"+token)),deadline_unix=DEADLINE,
        processes=[dict(id=identity,command=argv,cwd=config["cwd"],max_seconds=DEADLINE-time.time())],
        max_each_RSS=3*GIB,max_group_RSS=3*GIB,poll_seconds=2,terminate_grace_seconds=10,kill_grace_seconds=5)


def fit_order(plans):
    mk=plans["multik"]["fits"];req=plans["required"]["conditions"];fam=plans["family"]["conditions"]
    ordered=[("multik",mk[0]),("multik",mk[1]),("required",req[0]),("family",fam[0]),
        ("required",req[1]),("family",fam[1]),("required",req[2]),("family",fam[2]),
        ("multik",mk[2]),("multik",mk[3]),("family",fam[3]),("required",req[3]),
        ("required",req[4]),("required",req[5]),("multik",mk[4])]
    if len({row["run_id"] for _,row in ordered})!=15:raise ValueError("Fifteen unique fits required")
    return ordered


def make_config(public,cache,asset,cwd,python):
    public,cache,asset,cwd,python=map(safe,(public,cache,asset,cwd,python))
    config=dict(schema="POST_T_CPU_ROLLING_FITS_v1",public_root=str(public),cache_root=str(cache),asset_root=str(asset),cwd=str(cwd),python=str(python),
        deadline_unix=DEADLINE,maximum_concurrent_workers=2,per_worker_RSS_bytes=3*GIB,total_sampled_RSS_bound_bytes=6*GIB,
        storage_limit_bytes=30*GIB,admission_reserve_bytes=256*1024**2,CPU_threads_per_worker=2,
        ROOT_must_confirm_all_other_model_cost_and_audit_jobs_closed_before_execute=True,cost_measurement_included=False)
    pins={};plans={};runners={}
    guard=cache/"postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
    core=cache/"postT_cpu_queue_handoff_20261016_v1/cpu_sequence.py"
    admission=cache/"postT_cpu_aftercost_admission_20261016_v1/CLOSURE_ADMISSION.json"
    sequence=admission.with_name("PROBE_SEQUENCE_RESULT.json")
    for path,digest in ((guard,GUARD_SHA),(core,CORE_SHA),(admission,ADMISSION_SHA),(Path(__file__),None),(sequence,None)):pin(pins,path,digest)
    config.update(guard=str(guard),core=str(core),closure_admission=str(admission),probe_sequence=str(sequence))
    sequence_value=read(sequence)
    if sequence_value.get("all_groups_complete") is not True:raise GlobalStop("Actual probe sequence is not terminal")
    for key,entry in GROUPS.items():
        folder=public/entry["name"]
        for name,digest in ((entry["runner"],entry["runner_sha"]),(entry["verifier"],entry["verifier_sha"]),("PLAN.json",entry["plan_sha"]),("REGISTRY.csv",entry["registry_sha"])):pin(pins,folder/name,digest)
        for pattern in ("test_*.py","README_PREPARATION_KO.md"):
            for path in folder.glob(pattern):pin(pins,path)
        runner=load(folder/entry["runner"],entry["runner_sha"],"rolling_"+key)
        plan=runner.metadata(public,cache,asset,entry["plan_sha"]);runners[key]=runner;plans[key]=plan
        for fields in (plan["source39"],plan["immutable_public"],plan.get("upstream_recipe_metadata",{}),plan.get("dependencies",{})):
            for name,digest in fields.items():pin(pins,public/name,digest)
        pin(pins,asset/plan["instance_norm_source_relative_path"],plan["instance_norm_source_sha256"])
        for field in ("source_reference","shared_utility","task_validator","family_lock","task_contract_reference"):
            descriptor=plan.get(field)
            if descriptor:pin(pins,public/descriptor["relative_path"],descriptor["sha256"])
    mk=plans["multik"]
    for pathkey,shakey in (("shared_utility_relative_path","shared_utility_sha256"),("pca_runner_relative_path","pca_runner_sha256")):
        pin(pins,public/mk[pathkey],mk[shakey])
    old_fit_dir=cache/"postT_cpu_multik_preparation_20261016_v1/fit_specs_root_v1"
    proof=read(old_fit_dir/"PROOF.json");pin(pins,old_fit_dir/"PROOF.json")
    for path,digest in proof["probe_evidence"]["bindings"].items():pin(pins,path,digest)
    for descriptor in proof["source_pins"].values():pin(pins,descriptor["file"],descriptor["sha256"])
    pin(pins,cache/"postT_cpu_multik_preparation_20261016_v1/fit_group_spec_builder.py",proof["helper_sha256"])
    builder=load(cache/"postT_cpu_multik_preparation_20261016_v1/fit_group_spec_builder.py",proof["helper_sha256"],"rolling_probe_contract")
    guard_module=load(guard,GUARD_SHA,"rolling_guard_metadata_only")
    evidence=proof["probe_evidence"]["bindings"]
    probe_spec=next(Path(p) for p in evidence if Path(p).name=="SPEC.json")
    probe_guard=Path(read(probe_spec)["output_root"])/"RESULT.json"
    builder.validate_probes(dict(public=public,cache=cache,asset=asset,python=python,runner_path=public/GROUPS["multik"]["name"]/GROUPS["multik"]["runner"],
        guard=guard_module,plan=mk),probe_spec,evidence[str(probe_spec)],probe_guard,evidence[str(probe_guard)])
    fit_jobs=[]
    for index,(group,row) in enumerate(fit_order(plans),1):
        entry=GROUPS[group];run_id=row["run_id"]
        job=dict(id="fit_%02d"%index,group=group,kind="fit",run_id=run_id,recipe=row["spec"],ready=True)
        if group=="multik":
            original_index=next(i for i,value in enumerate(mk["fits"],1) if value["run_id"]==run_id)
            path=old_fit_dir/("SPEC_%02d.json"%original_index)
            digest=proof["specs"][path.name]["sha256"];pin(pins,path,digest)
            job["fit_spec_file"]=dict(file=str(path),sha256=digest)
            job["actual_probe_evidence"]=proof["probe_evidence"]["bindings"]
            job["outputs"]=[value for value in mk["outputs"] if value["parent_run_id"]==run_id]
        else:
            position=next(i for i,value in enumerate(plans[group]["conditions"],1) if value["run_id"]==run_id)
            path=public/entry["name"]/"diagnostics"/("probe_root_v1_%02d"%position)/"RESULT.json"
            probe=dict(file=str(path),sha256=sha(path)) if path.is_file() else None
            if probe is None:job.update(ready=False,not_ready_reason="ACTUAL_OWN_PROBE_RESULT_ABSENT")
            else:
                pin(pins,path,probe["sha256"])
                if group=="required":runners[group].validate_probe_proof(read(path),row,plans[group],entry["plan_sha"])
                else:runners[group].validate_probe_proof(read(path),row,plans[group]["task_contracts"][row["dataset"]],entry["plan_sha"],entry["runner_sha"])
            job["actual_probe_evidence"]=probe
            out=public/entry["name"]/"diagnostics"/("rolling_fit_root_v1_%02d"%index)
            job["fit_spec"]=single_spec(config,group,"FIT_%02d"%index,
                command(config,group,entry["runner"],"fit",run_id,out,probe),"fit_%02d"%index)
            job["outputs"]=[dict(run_id=run_id,kind="standard",latent_k=row.get("effective_F_width",row["latent_k"]))]
        fit_jobs.append(job)
    aux=[dict(id="nested_%d"%row["latent_k"],group="multik",kind="nested",run_id=row["run_id"],outputs=[row],ready=True,
        optimizer_updates=0,parent_fit_run_id=None,full_C_PCA_rotation_differs_from_K_equals_C_identity_training=True)
        for row in mk["outputs"] if row["kind"]=="nested"]
    config.update(units=aux+fit_jobs,expected_fits=15,expected_nested_fixed_outputs=3,expected_E4_V_outputs=12,expected_total_V_outputs=22,
        pins=pins,nested_V_creation_serialized_for_shared_artifact_xb_safety=True,
        replay_protocols=dict(multik=dict(batch_size=4,origins=6),required=dict(batch_size=6,origins=6),family=dict(batch_size=6,origins=6)))
    for group in sequence_value["groups"]:
        path=cache/group["spec_relative_path"];pin(pins,path,group["spec_sha256"])
        result=Path(read(path)["output_root"])/"RESULT.json";pin(pins,result,group["result_sha256"])
    verify_config(config)
    return config


def verify_config(config):
    if (config.get("schema")!="POST_T_CPU_ROLLING_FITS_v1" or config.get("maximum_concurrent_workers")!=2
            or config.get("deadline_unix")!=DEADLINE or config.get("expected_fits")!=15 or config.get("cost_measurement_included") is not False):
        raise GlobalStop("Rolling queue scope/resources differ")
    units=config["units"]
    if len(units)!=18 or len({row["id"] for row in units})!=18 or sum(row["kind"]=="fit" for row in units)!=15:raise GlobalStop("Exact 15 fits plus three fixed auxiliary units required")
    for path,digest in config["pins"].items():
        if sha(safe(path))!=digest:raise GlobalStop("Pinned input changed: "+Path(path).name)
    admission=read(config["closure_admission"])
    if (admission.get("status")!="ALL_THREE_PRIOR_QUEUES_AND_65_OBSERVED_GUARDS_CLOSED" or any(row["state"]!="DEAD" for row in admission["outer_owners"])):
        raise GlobalStop("Prior closure admission differs")
    return config


def metadata(path,digest):
    path=safe(path)
    if sha(path)!=digest:raise GlobalStop("Explicit immutable CONFIG SHA differs")
    return verify_config(read(path))


def rolling(units,work,check,event,stop=None):
    stop=stop or threading.Event();pending=iter(units);active={};rows=[];exhausted=False;global_error=None
    def invoke(unit):
        try:return work(unit)
        except BaseException:
            stop.set();raise
    with ThreadPoolExecutor(max_workers=2) as pool:
        while active or not exhausted:
            while not stop.is_set() and len(active)<2 and not exhausted:
                try:unit=next(pending)
                except StopIteration:exhausted=True;break
                try:check()
                except Exception as error:
                    global_error=type(error).__name__+": "+str(error);stop.set()
                    rows.append(dict(id=unit["id"],status="NOT_LAUNCHED_GLOBAL_STOP",reason=global_error));break
                active[pool.submit(invoke,unit)]=unit
                event(dict(event="SLOT_SUBMITTED",unit=unit["id"],active_slots=len(active)))
            if not active:break
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                unit=active.pop(future)
                try:row=future.result()
                except Exception as error:
                    row=dict(id=unit["id"],status="GLOBAL_STOP_PRESERVED",error_class=type(error).__name__)
                    global_error=type(error).__name__+": "+str(error);stop.set()
                rows.append(row);event(dict(event="WORKFLOW_TERMINAL",row=row,active_slots=len(active)))
        for unit in pending:rows.append(dict(id=unit["id"],status="NOT_LAUNCHED_GLOBAL_STOP",reason=global_error or "stop_event"))
    return dict(rows=rows,global_stop=stop.is_set(),global_error=global_error)


class Runtime:
    def __init__(self,path,digest,output):
        self.path,self.digest,self.output=Path(path),digest,Path(output)
        self.config=metadata(path,digest);self.stop=threading.Event();self.log_lock=threading.Lock();self.nested_lock=threading.Lock()
        self.guard=load(self.config["guard"],GUARD_SHA,"rolling_frozen_guard")
        self.core=load(self.config["core"],CORE_SHA,"rolling_closure_core")
        self.public,self.cache,self.asset=(Path(self.config[key]) for key in ("public_root","cache_root","asset_root"))

    def event(self,value):
        with self.log_lock:
            with (self.output/"PROGRESS.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(at_unix=time.time(),**value),ensure_ascii=False,allow_nan=False)+"\n")

    def check(self):
        if self.stop.is_set():raise GlobalStop("Global stop prevents another child launch")
        metadata(self.path,self.digest)
        if time.time()>=DEADLINE-20:raise GlobalStop("Original deadline reserve reached")

    def storage(self):
        used=sum(p.stat().st_size for root in (self.public,self.cache) for p in root.rglob("*") if p.is_file())
        reserve=self.config["admission_reserve_bytes"]
        if used+reserve>self.config["storage_limit_bytes"] or min(shutil.disk_usage(root).free for root in (self.public,self.cache))<reserve:
            self.stop.set();raise GlobalStop("Combined storage cap/free-space reserve exceeded")

    def prior_closure(self):
        admission=read(self.config["closure_admission"])
        for row in admission["outer_owners"]:
            if self.core.identity_state(row)!="DEAD":raise GlobalStop("Prior outer process identity is not dead")
        sequence=read(self.config["probe_sequence"])
        for group in sequence["groups"]:
            path=self.cache/group["spec_relative_path"];spec=read(path);result=read(Path(spec["output_root"])/"RESULT.json")
            status,_=self.core.closure_state(result,sha(path),GUARD_SHA,[p["id"] for p in spec["processes"]],self.core.identity_state)
            if status!="READY":raise GlobalStop("Prior actual probe tree is not READY")
        proof=read(self.cache/"postT_cpu_multik_preparation_20261016_v1/fit_specs_root_v1/PROOF.json")
        paths=proof["probe_evidence"]["bindings"]
        spec_path=next(Path(p) for p in paths if Path(p).name=="SPEC.json")
        spec=read(spec_path);result=read(Path(spec["output_root"])/"RESULT.json")
        status,_=self.core.closure_state(result,sha(spec_path),GUARD_SHA,[p["id"] for p in spec["processes"]],self.core.identity_state)
        if status!="READY":raise GlobalStop("Prior E4 probe tree is not READY")

    def perform(self,unit,label,spec=None):
        self.check();self.storage()
        stage=self.output/unit["id"]/label;stage.mkdir(parents=True,exist_ok=False)
        if spec is None:
            descriptor=unit["fit_spec_file"];path=Path(descriptor["file"]);spec=read(path);digest=descriptor["sha256"]
        else:path=stage/"SPEC.json";new(path,spec);digest=sha(path)
        if Path(spec["output_root"]).exists():raise GlobalStop("Owned guard output already exists; preserve it")
        if len(spec["processes"])!=1:raise GlobalStop("Only single-worker guard steps allowed")
        effective=self.guard.validate_spec(spec)
        self.event(dict(event="GUARD_START",unit=unit["id"],step=label,spec_file=str(path),spec_sha256=digest))
        result=self.guard.supervise(effective,spec_file_sha256=digest)
        result_path=Path(spec["output_root"])/"RESULT.json"
        if read(result_path)!=result:raise GlobalStop("Returned guard receipt differs from saved bytes")
        ids=[row["id"] for row in spec["processes"]]
        while True:
            status,detail=self.core.closure_state(result,digest,GUARD_SHA,ids,self.core.identity_state)
            if status=="READY":break
            if status=="UNSAFE" or time.time()>=DEADLINE:self.stop.set();raise GlobalStop("Owned guard closure is unknown")
            time.sleep(2)
        argv=spec["processes"][0]["command"];scientific=Path(argv[argv.index("--output")+1])/"RESULT.json"
        value=None;scientific_error=None;scientific_sha=None
        if scientific.is_file():
            try:
                scientific_sha=sha(scientific);value=read(scientific)
            except (OSError,UnicodeDecodeError,json.JSONDecodeError) as error:scientific_error=type(error).__name__
            if scientific_error is None and not isinstance(value,dict):scientific_error="SCIENTIFIC_RESULT_NOT_OBJECT"
            if scientific_error is None and (not isinstance(value.get("status"),str) or not value["status"]):scientific_error="SCIENTIFIC_RESULT_INVALID_STATUS"
        else:scientific_error="SCIENTIFIC_RESULT_MISSING_OR_NONREGULAR"
        successful=result["status"]=="COMPLETE" and result["processes"][0]["returncode"]==0 and scientific_error is None
        category=label.split("_",1)[0]
        if category=="fit":successful=successful and value.get("status") in {"COMPLETE_FRESH_CPU_FIT_NO_V_YET","COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"}
        if category=="V":successful=successful and value.get("status")=="COMPLETE_FULL_V_ONLY"
        if category in {"rescore","replay"}:
            successful=successful and value.get("status")=="PASS"
            if category=="replay":successful=successful and value.get("actual_checkpoint_forward_in_this_command") is True
        row=dict(step=label,guard_result=str(result_path),guard_result_sha256=sha(result_path),closed=True,success=successful,
            scientific_result=str(scientific),scientific_result_sha256=scientific_sha,scientific_result_error=scientific_error,
            scientific_status=value.get("status") if isinstance(value,dict) else None,closure=detail)
        new(stage/"TERMINAL.json",row);self.event(dict(event="GUARD_CLOSED",unit=unit["id"],**row))
        self.check()
        return row

    def step_spec(self,unit,label,run_id,script,mode):
        entry=GROUPS[unit["group"]];token=unit["id"]+"_"+label+"_"+hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
        output=self.public/entry["name"]/"diagnostics"/("rolling_root_v1_"+token)
        if output.exists():raise GlobalStop("Existing scientific output must be preserved")
        return single_spec(self.config,unit["group"],label.upper(),command(self.config,unit["group"],script,mode,run_id,output),token)

    def work(self,unit):
        steps=[];entry=GROUPS[unit["group"]]
        if not unit["ready"]:return dict(id=unit["id"],status="NOT_READY",reason=unit.get("not_ready_reason"))
        self.check()
        if unit["kind"]=="fit":
            runner=load(self.public/entry["name"]/entry["runner"],entry["runner_sha"],"rolling_fresh_check_"+unit["id"])
            try:runner.require_fresh(self.public,self.cache,"fit",unit["run_id"])
            except FileExistsError:return dict(id=unit["id"],status="EXISTING_PATH_PRESERVED_NO_RESUME")
            row=self.perform(unit,"fit",unit.get("fit_spec"));steps.append(row)
            if not row["success"]:return dict(id=unit["id"],status="LOCAL_FIT_FAILURE_PRESERVED",steps=steps)
        for output in unit["outputs"]:
            run_id=output["run_id"]
            if unit["group"]=="multik":
                mode="nested" if unit["kind"]=="nested" else "prefix"
                spec=self.step_spec(unit,"V",run_id,entry["runner"],mode)
                if mode=="nested":
                    with self.nested_lock:row=self.perform(unit,"V_"+run_id,spec)
                else:row=self.perform(unit,"V_"+run_id,spec)
                steps.append(row)
                if not row["success"]:continue
            for mode in ("rescore","replay"):
                spec=self.step_spec(unit,mode,run_id,entry["verifier"],mode)
                steps.append(self.perform(unit,mode+"_"+run_id,spec))
        return dict(id=unit["id"],kind=unit["kind"],status="ALL_STEPS_PASSED" if all(row["success"] for row in steps) else "LOCAL_OUTPUT_FAILURE_PRESERVED",steps=steps,
            fresh_CPU_verification_is_not_convergence_or_cost_measurement=True)


def execute(path,digest,output,root_triggered):
    if root_triggered is not True:raise ValueError("ROOT review and --root-triggered are required")
    runtime=Runtime(path,digest,output);output=safe(output)
    if output.exists() or not output.is_relative_to(Path(path).parent):raise FileExistsError("Fresh private rolling output below preparation directory required")
    runtime.prior_closure();runtime.check();runtime.storage();output.mkdir(parents=True,exist_ok=False)
    for entry in GROUPS.values():
        source=runtime.public/entry["name"]/"REGISTRY.csv";target=source.with_name("EXPERIMENT_REGISTRY.csv")
        safe(target)
        if not target.exists():
            with target.open("xb") as stream:stream.write(source.read_bytes())
        if sha(target)!=entry["registry_sha"]:raise GlobalStop("Exact separate registry alias differs")
    runtime.event(dict(event="ROOT_TRIGGERED_START",CONFIG_sha256=digest,actual_pid=os.getpid(),fits_planned=15,nested_fixed_planned=3))
    try:result=rolling(runtime.config["units"],runtime.work,runtime.check,runtime.event,runtime.stop)
    except Exception as error:
        with (output/"PRIVATE_ERROR_TRACE.txt").open("x",encoding="utf-8") as stream:traceback.print_exc(file=stream)
        result=dict(global_stop=True,global_error=type(error).__name__)
    result.update(status="GLOBAL_STOP_WITH_OWNED_GUARDS_WAITED" if result["global_stop"] else "REQUESTED_WORKFLOWS_TERMINAL_REVIEW_ROWS_FOR_FAILURES",
        CONFIG_sha256=digest,helper_sha256=sha(__file__),finished_unix=time.time(),scientific_success_or_convergence_claimed=False,
        original_registry_updated=False,new_T_scoring=False,cost_measurement_executed=False,
        group_RSS_scope="Two independent original 3GiB sampled guards; no common-instant sum or unobserved orphan absence proof")
    new(output/"RESULT.json",result)
    return 0 if not result["global_stop"] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config");parser.add_argument("--config-sha")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true");parser.add_argument("--output")
    args=parser.parse_args(argv)
    if args.execute:
        if not all((args.config,args.config_sha,args.output,args.root_triggered)):parser.error("Explicit CONFIG/SHA/fresh output/ROOT trigger required")
        return execute(args.config,args.config_sha,Path(args.output).absolute(),args.root_triggered)
    if args.config or args.config_sha:
        if not args.config or not args.config_sha:parser.error("CONFIG and exact SHA required together")
        metadata(args.config,args.config_sha)
    print(json.dumps(dict(status="METADATA_ONLY_ROLLING_QUEUE_NOT_EXECUTED",files_written=0,numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,actual_children_or_science=0)))
    return 0


if __name__=="__main__":raise SystemExit(main())
