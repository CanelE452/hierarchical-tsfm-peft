"""Explicit finite adapter-improvement campaign; the default performs no work."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .contracts import (NAME, REQUEST_SHA256, SUCCESS, admit_bundle, atomic_json,
                        binding, campaign_ledger, finish_job, read, reusable, seal_job,
                        seal_plan, select_jobs, sha, utc, validate_plan)

REPO = Path(__file__).resolve().parents[2]


def context(args):
    repo = args.repo.resolve()
    if args.asset_root is None:
        raise ValueError("--asset-root is required for explicit campaign operations")
    cache = (args.cache_root or repo/".cache"/NAME).resolve()
    report = (args.report_root or repo/"reports"/NAME).resolve()
    previous = repo/".cache/adapter_diagnostics_20261017_v2"
    cache.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    requested_plan=getattr(args,"plan",None) or report/"EXPERIMENT_PLAN.json"
    if Path(requested_plan).exists() and not (cache/"CAMPAIGN.json").exists():
        raise ValueError("An existing sealed plan requires its persistent original campaign ledger; resource usage cannot be reset")
    ledger = campaign_ledger(cache/"CAMPAIGN.json")
    ctx = dict(repo=repo,asset=args.asset_root.resolve(),cache=cache,cache_root=cache,
               report=report,previous_cache=previous,ledger=ledger,
               deadline=ledger["started_unix"]+ledger["limits"]["wall_seconds"])
    return ctx


def scientific_sources(repo):
    names = ["contracts.py","audit.py","reuse.py","decoder.py","decoder_jobs.py","frontends.py",
             "training_jobs.py","initialization.py","lift.py","lift_jobs.py","decoder_transfer.py","costs.py","multik_audit.py"]
    paths = [Path(repo)/"research"/NAME/name for name in names]
    base = Path(repo)/"research/informs_broad_validation_20261015_v1"
    paths += [base/name for name in ("networks.py","fit_base.py","training.py","data_contract.py",
               "gpu_runtime.py","evaluation.py","deployment.py","resources.py","DATA_CONTRACT.json")]
    paths += [Path(repo)/"research/temporal_frontend_campaign_20261015_v1/runtime.py",
              Path(repo)/"reports"/NAME/"OFFICIAL_SOURCE_MANIFEST.json"]
    return {path.relative_to(repo).as_posix():sha(path) for path in sorted(paths)}


def validate_context_plan(plan,ctx):
    validate_plan(plan,ctx["repo"])
    if plan["campaign_started_unix"]!=ctx["ledger"]["started_unix"]:
        raise ValueError("Resumed plan and persistent ledger start differ")
    if sha(__file__)!=plan["execution_control_sha256"]:
        raise ValueError("Sealed execution controller changed")
    if sha(ctx["report"]/"REUSE_AUDIT.json")!=plan["reuse_audit_sha256"]:
        raise ValueError("Sealed historical reuse audit changed")


def make_plan(ctx):
    destination = ctx["report"]/"EXPERIMENT_PLAN.json"
    if destination.exists():
        plan = read(destination)
        validate_context_plan(plan,ctx)
        return plan
    sources = scientific_sources(ctx["repo"])
    jobs = []
    def add(identifier,kind,cohort="jena",seed=None,family=None,dependencies=(),device="cpu",seconds=60,**extra):
        body = dict(job_id=identifier,kind=kind,cohort=cohort,seed=seed,family=family,
                    K=6 if cohort=="jena" else 8,device=device,dependency_ids=list(dependencies),
                    expected_cost=dict(wall_seconds=seconds,gpu_seconds=seconds if device=="cuda" else 0),**extra)
        jobs.append(seal_job(body,sources))
    for seed in (92601,92602):
        add(f"REUSE_PCA_JENA_s{seed}","reuse_pca",seed=seed,seconds=10)
    for seed in (92601,92602):
        add(f"D_JENA_s{seed}","decoder",seed=seed,seconds=600)
    for scope in ("A_fit","A60"):
        add("DCA_INIT_JENA_"+scope,"initialization",scope=scope,k=6,seconds=125)
    for seed in (92601,92602):
        group = f"JENA_P_s{seed}"
        add("PROFILE_"+group,"profile",seed=seed,device="cuda",seconds=45,group=group)
        for family in ("P_SPLIT","P_LOCAL_CONTROL"):
            add(f"{family}_JENA_s{seed}","training",seed=seed,family=family,init_kind="PCA",group=group,
                device="cuda",seconds=900,dependencies=["PROFILE_"+group])
        group = f"JENA_E_s{seed}"
        add("PROFILE_"+group,"profile",seed=seed,device="cuda",seconds=45,group=group,
            dependencies=["DCA_INIT_JENA_A_fit","DCA_INIT_JENA_A60"])
        add(f"B2_DCA_JENA_s{seed}","training",seed=seed,family="B2",init_kind="DCA",group=group,
            device="cuda",seconds=900,dependencies=["PROFILE_"+group,f"REUSE_PCA_JENA_s{seed}",
                                                   "DCA_INIT_JENA_A_fit","DCA_INIT_JENA_A60"])
    for scope in ("A_fit","A60"):
        add("DCA_INIT_PEMS32_"+scope,"initialization",cohort="pems32",scope=scope,k=8,seconds=125)
    for seed in (92601,92602):
        common_deps=[]
        if seed==92602:
            common_deps=[f"{family}_PEMS32_s92601" for family in ("P_SPLIT","P_LOCAL_CONTROL","B2_PCA","B2_DCA")]
        for factor,members in (("P",(("P_SPLIT","PCA","P_SPLIT"),("P_LOCAL_CONTROL","PCA","P_LOCAL_CONTROL"))),
                                ("E",(("B2","PCA","B2_PCA"),("B2","DCA","B2_DCA")))):
            group=f"PEMS32_{factor}_s{seed}"
            deps=common_deps+([] if factor=="P" else ["DCA_INIT_PEMS32_A_fit","DCA_INIT_PEMS32_A60"])
            add("PROFILE_"+group,"profile",cohort="pems32",seed=seed,device="cuda",seconds=60,group=group,
                dependencies=deps,conditional=seed==92602)
            for family,init,label in members:
                add(f"{label}_PEMS32_s{seed}","training",cohort="pems32",seed=seed,family=family,init_kind=init,
                    device="cuda",group=group,seconds=1200,dependencies=["PROFILE_"+group],conditional=seed==92602)
    fallback=ctx["previous_cache"]/"runs/pems_B2/search/checkpoint_000003.pt"
    fallback_binding=binding(fallback,ctx) if fallback.exists() else None
    add("D_PEMS32_s92601","decoder_transfer",cohort="pems32",seed=92601,device="cuda",seconds=60,
        dependencies=["D_JENA_s92601","D_JENA_s92602","P_SPLIT_PEMS32_s92601","P_LOCAL_CONTROL_PEMS32_s92601"],
        conditional=True,fallback_A_fit_checkpoint_binding=fallback_binding)
    core=[j["job_id"] for j in jobs if j["kind"]=="training" and (j["cohort"]=="jena" or j["seed"]==92601)]
    add("LIFT_JENA_s92601","lift_jobs",seed=92601,device="cuda",seconds=60,
        dependencies=core+["D_JENA_s92601","D_JENA_s92602"],conditional=True)
    add("MULTIK_AUDIT","multik_audit",cohort="existing",seconds=60)
    for cohort in ("jena","pems32"):
        for device in ("cpu","cuda"):
            add(f"COST_{cohort.upper()}_{device.upper()}","costs",cohort=cohort,device=device,seed=92601,seconds=600,batches=[1,8])
    plan = seal_plan(dict(campaign=NAME,created_utc=utc(),request_sha256=REQUEST_SHA256,
        parent_commit=read(ctx["report"]/"START_STATE.json")["parent_commit"],
        campaign_started_unix=ctx["ledger"]["started_unix"],
        source_hashes=sources,execution_control_sha256=sha(__file__),limits=ctx["ledger"]["limits"],
        reuse_audit_sha256=sha(ctx["report"]/"REUSE_AUDIT.json"),
        acceptance=dict(accuracy_tolerance="UNREGISTERED",speedup="UNREGISTERED",required_batch="UNREGISTERED"),
        recipe=dict(epochs=[.25,1,3],CP0="internal",seed_order=[92601,92602],lr=.001,
                    optimizer="AdamW",weight_decay=0,gradient_clip=1,effective_batch=16,microbatch=4,
                    F_row_microbatch=128,F_dtype="float32",checkpoint_inputs=False,
                    selection="A_fit -> exact A_stop MSE, earlier endpoint tie; fresh A60 refit; sealed before V",
                    train_stride=1,evaluation_stride="H",T_access=False,M_cal_selection=False),
        numerical_tolerances=dict(saved_FP64=dict(atol=1e-12,rtol=1e-12),
            FP32_route=dict(atol=1e-5,rtol=1e-4),legacy_FP32_reduction=dict(atol=1e-8,rtol=0),
            evidence="Existing pinned-environment numerical parity receipts and new synthetic augmented-solve probes; no accuracy pass threshold"),
        hypotheses=dict(D="Shared,full,linear,gain readout with common fitting data/PE and registered grids",
            P="Split versus identical-capacity local control",E="DCA versus matched PCA initialization only"),
        intervention_allowlist=["P architecture and its source/checkpoint", "E/D initialization and init source/cache", "D fit and selected head recipe"],
        decoder_grid=dict(lambda_values=[.0001,.001,.01,.1,1],gamma_values=[1,10,100],degree=1,settings_per_seed=40),
        optional_blocks=dict(LIFT=dict(status="REGISTERED_CONDITIONAL",seed=92601,cohort="jena",
            base="original B03 B2 A_fit selected epoch and fresh A60; same-base gain control",n_leaders=4,state_num=20,
            temperature_init=1.,optimizer="Adam",lr=.001,batch=128,max_epochs=100,patience=5,schedule="official type3",
            endpoints="CP0 and integer A_stop epochs; exact strict improvement, earlier tie; fresh selected-epoch A60",
            gain_lambda=[.0001,.001,.01,.1,1],gain_gamma=[1,10,100],gain_basis="centered unitRMS h; ownfit deltaRMS/channel; no intercept",
            gain_penalty="lambda/C sum_c(s_c^2*(alpha_c^2+gamma*beta_c^2))",max_pair_seconds=3600,
            admission="Only after all mandatory valid first-seed pairs and remaining cost/reserves; full100+100 worst schedule profile; no V-driven recipe change"),
            decoder_pems32=dict(status="REGISTERED_CONDITIONAL",seed=92601,
                arms=["D_SHARED","D_HORIZON","D_LINEAR","D_GAIN"],grid="same40 settings asJena",bands=[[1,4],[5,8],[9,12]],
                parent_preference="verified new freshCUDA B2_PCA seed92601; otherwise byte-pinned prior completed CUDA-continuation B2",
                admission="Whole four-head extraction/fit/verification including source hashes must fit budget; no best-Jena-arm-only transfer"),
            multik="Read-only existing evidence audit; no new multi-K training"),
        jobs=jobs))
    validate_plan(plan,ctx["repo"])
    if plan["campaign_started_unix"]!=ctx["ledger"]["started_unix"]:
        raise ValueError("Resumed plan and persistent ledger start differ")
    atomic_json(destination,plan)
    return plan


def _runtime_context(ctx):
    from .audit import load_foundation,load_task
    ctx["load_task"] = lambda cohort: load_task(ctx,cohort)
    ctx["task_loader"] = lambda context,modules,cohort: load_task(context,cohort,modules)
    ctx["foundation_loader"] = load_foundation
    return ctx


def group_admission(job,ctx,now=None):
    profile=read(ctx["report"]/"profiles"/("PROFILE_"+job["group"]+".json"))
    timings={(row["family"],row["init_kind"]):row["estimated_complete_arm_seconds"]
             for probe in profile["profiles"] for row in probe["measurements"]}
    unfinished=[]
    estimate=0.
    for member in ctx["plan"]["jobs"]:
        if member["kind"]!="training" or member.get("group")!=job["group"]:
            continue
        if reusable(ctx["report"]/"receipts"/(member["job_id"]+".json"),member,ctx):
            continue
        unfinished.append(member["job_id"])
        estimate+=timings[(member["family"],member.get("init_kind","PCA"))]
    decision=dict(group=job["group"],checked_utc=utc(),unfinished_job_ids=unfinished,
                  estimated_unfinished_seconds=estimate,
                  profile_sha256=sha(ctx["report"]/"profiles"/("PROFILE_"+job["group"]+".json")),
                  admitted=admit_bundle(ctx["ledger"],estimate,estimate,now=now))
    return decision


def fatal_family_failure(job,ctx):
    members=[j for j in ctx["plan"]["jobs"] if j["kind"]=="training" and j.get("group")==job.get("group")]
    signatures={(j["family"],j.get("init_kind","PCA")) for j in members}
    for prior in ctx["plan"]["jobs"]:
        if prior["kind"]!="training":
            continue
        same_group=prior.get("group")==job.get("group")
        source_family=(job["cohort"]=="pems32" and prior["cohort"]=="jena" and
                       (prior["family"],prior.get("init_kind","PCA")) in signatures)
        path=ctx["report"]/"receipts"/(prior["job_id"]+".json")
        if (same_group or source_family) and path.exists() and read(path)["status"] in {"FAILED","FAILED_NUMERICAL","BLOCKED_DATA_CONTRACT"}:
            return prior["job_id"]
    return None


def handler(job,ctx):
    kind=job["kind"]
    if kind=="initialization":
        from .initialization import execute_job
        result=execute_job(job,_runtime_context(ctx))
        if result.get("cache_binding") and result.get("cache_sha256"):
            result["private_artifacts"]=[dict(reference="cache://"+result["cache_binding"],sha256=result["cache_sha256"])]
        path=ctx["report"]/"jobs"/(job["job_id"]+".json")
        atomic_json(path,result)
        return dict(status=result["status"],reason=None if result["status"]=="COMPLETE" else "DCA fit did not converge under registered limits",artifact_paths=[str(path)],result=result)
    if kind=="profile":
        from .training_jobs import profile_pair
        members=[j for j in ctx["plan"]["jobs"] if j.get("group")==job["group"] and j["kind"]=="training"]
        probes=[]
        for pair in ([j for j in members if j["family"].startswith("P_")],
                     [j for j in members if j["family"]=="B2"]):
            if len(pair)==1:
                candidate=pair[0]
                pair=[dict(candidate,job_id=candidate["job_id"]+"_PCA_PROBE",init_kind="PCA"),candidate]
            if pair:
                probes.append(profile_pair(pair,_runtime_context(ctx)))
        estimate=sum(p["estimated_complete_pair_seconds"] for p in probes)
        ledger=campaign_ledger(ctx["cache"]/"CAMPAIGN.json")
        if ctx.get("lease"):
            ledger["gpu_seconds"]+=max(0.,time.time()-ctx["lease"].started_unix)
        admitted=admit_bundle(ledger,estimate,estimate)
        result=dict(status="COMPLETE",group=job["group"],profiles=probes,
                    conservative_complete_group_seconds=estimate,admitted=admitted,
                    note="Includes both candidate/control routes and verification; old PCA training need not be repeated")
        path=ctx["report"]/"profiles"/(job["job_id"]+".json")
        atomic_json(path,result)
        return dict(status="COMPLETE",reason=None,artifact_paths=[str(path)],result=result)
    if kind in ("training","decoder","reuse_pca"):
        name={"training":"training_jobs","decoder":"decoder_jobs","reuse_pca":"reuse"}[kind]
        return importlib.import_module("."+name,__package__).execute_job(job,_runtime_context(ctx))
    if kind in ("costs","multik_audit","lift_jobs","decoder_transfer"):
        if kind=="lift_jobs":
            limits=ctx["ledger"]["limits"]
            current=max(0.,time.time()-ctx["lease"].started_unix) if ctx.get("lease") else 0.
            ctx["lift_dependencies_complete"]=True
            ctx["lift_budget"]=dict(optional_seconds=3600,remaining_seconds=ctx["deadline"]-time.time(),
                reserve_seconds=limits["wall_reserve"],remaining_gpu_seconds=limits["gpu_seconds"]-ctx["ledger"]["gpu_seconds"]-current,
                gpu_reserve_seconds=limits["gpu_reserve"])
        return importlib.import_module("."+kind,__package__).execute_job(job,_runtime_context(ctx))
    raise ValueError("Unregistered handler kind")


def execute(ctx,plan_path,only=None,worker=False):
    plan=read(plan_path)
    validate_context_plan(plan,ctx)
    ctx["plan"]=plan
    jobs=select_jobs(plan,only)
    if not worker:
        for job in jobs:
            command=[sys.executable,"-m","research."+NAME+".run","--mode","execute","--worker",
                "--repo",str(ctx["repo"]),"--asset-root",str(ctx["asset"]),"--cache-root",str(ctx["cache"]),
                "--report-root",str(ctx["report"]),"--plan",str(Path(plan_path).resolve()),"--only",job["job_id"]]
            code=subprocess.run(command,cwd=ctx["repo"],check=False).returncode
            if code:
                raise RuntimeError("Isolated worker exited "+str(code)+": "+job["job_id"])
        return
    sys.path.insert(0,str(ctx["repo"]/"research/informs_broad_validation_20261015_v1"))
    import resources
    for job in jobs:
        path=ctx["report"]/"receipts"/(job["job_id"]+".json")
        if reusable(path,job,ctx):
            print(job["job_id"],"REUSED_VERIFIED",flush=True)
            continue
        if path.exists():
            old=read(path)
            if old.get("scientific_identity")!=job["scientific_identity"]:
                raise ValueError("Existing receipt has another scientific identity")
            if old["status"]=="RUNNING":
                try:
                    alive=abs(resources.psutil.Process(old["worker_pid"]).create_time()-old["worker_create_time"])<.001
                except (resources.psutil.NoSuchProcess,resources.psutil.AccessDenied):
                    alive=False
                if alive:
                    raise resources.ResourceBusy("Registered job has a verified live worker")
                checkpoints=list((ctx["cache"]/"runs"/job["job_id"]).glob("*/checkpoint_*.pt"))
                if not checkpoints:
                    archive=ctx["report"]/"attempts"/job["job_id"]/(sha(path)+".json")
                    atomic_json(archive,old)
                    finish_job(path,job,dict(status="FAILED",reason="Interrupted worker left no resumable checkpoint; new identity required",artifact_paths=[str(archive)]),ctx)
                    continue
            elif old["status"]!="PARTIAL_CHECKPOINTED":
                print(job["job_id"],"TERMINAL_PRESERVED",old["status"],flush=True)
                continue
            archive=ctx["report"]/"attempts"/job["job_id"]/(sha(path)+".json")
            if not archive.exists():
                atomic_json(archive,old)
        registered={member["job_id"]:member for member in plan["jobs"]}
        dependencies=[(ctx["report"]/"receipts"/(identifier+".json"),registered[identifier]) for identifier in job["dependency_ids"]]
        if any(not reusable(p,dependency,ctx) for p,dependency in dependencies):
            finish_job(path,job,dict(status="NOT_RUN_DEPENDENCY",reason="Required scope/control preparation incomplete",artifact_paths=[]),ctx)
            continue
        ledger=campaign_ledger(ctx["cache"]/"CAMPAIGN.json")
        ledger["gpu_seconds"]=resources.gpu_usage_seconds(ctx["cache"])
        ctx["ledger"]=ledger
        cost=job["expected_cost"]
        if job["kind"] in ("training","profile"):
            failure=fatal_family_failure(job,ctx)
            if failure:
                finish_job(path,job,dict(status="NOT_RUN_DEPENDENCY",reason="Numeric/contract family failure preserved: "+failure,artifact_paths=[]),ctx)
                continue
        if job["kind"]=="training":
            decision=group_admission(job,ctx)
            atomic_json(ctx["report"]/"admissions"/(job["job_id"]+".json"),decision)
            if not decision["admitted"]:
                finish_job(path,job,dict(status="SKIPPED_BUDGET",reason="Measured unfinished paired group no longer fits current reserves",artifact_paths=[]),ctx)
                continue
            cost=dict(wall_seconds=decision["estimated_unfinished_seconds"],gpu_seconds=decision["estimated_unfinished_seconds"])
        final=job["kind"] in ("costs","multik_audit")
        if not admit_bundle(ledger,cost["wall_seconds"],cost["gpu_seconds"],final=final):
            finish_job(path,job,dict(status="SKIPPED_BUDGET",reason="Job cannot finish before registered resource reserve",artifact_paths=[]),ctx)
            continue
        resources.check_storage_budget(ctx["cache"],max_bytes=ledger["limits"]["cache_bytes"])
        started,cpu=time.time(),time.process_time()
        atomic_json(path,dict(job_id=job["job_id"],scientific_identity=job["scientific_identity"],status="RUNNING",worker_pid=os.getpid(),worker_create_time=resources.psutil.Process().create_time(),started_utc=utc()))
        print(job["job_id"],"RUNNING",flush=True)
        ctx["job_deadline"]=ctx["deadline"]-(0 if final else ledger["limits"]["wall_reserve"])
        if job["kind"]=="decoder":
            family_timer=ctx["cache"]/"DECODER_FAMILY_START.json"
            if not family_timer.exists():
                atomic_json(family_timer,dict(started_unix=started))
            ctx["decoder_family_started_unix"]=read(family_timer)["started_unix"]
            ctx["job_deadline"]=min(ctx["job_deadline"],ctx["decoder_family_started_unix"]+1800)
        elif job["kind"] in ("decoder_transfer","lift_jobs"):
            ctx["job_deadline"]=min(ctx["job_deadline"],started+(1800 if job["kind"]=="decoder_transfer" else 3600))
        gpu=job["device"]=="cuda"
        try:
            if gpu:
                with resources.GPUReservation(ctx["cache"],total_gpu_seconds=ledger["limits"]["gpu_seconds"],
                        reserve_seconds=0 if final else ledger["limits"]["gpu_reserve"],wall_deadline=ctx["job_deadline"],purpose="audit" if final else "training",kind=job["job_id"]) as lease:
                    ctx["lease"]=lease
                    ctx["job_deadline"]=min(ctx["job_deadline"],lease.deadline)
                    result=handler(job,ctx)
                    lease.check()
            else:
                result=handler(job,ctx)
        except FileNotFoundError:
            result=dict(status="BLOCKED_ASSET",reason="Required bound local asset missing",artifact_paths=[])
            error=traceback.format_exc()
        except (TimeoutError,resources.BudgetExceeded):
            result=dict(status="PARTIAL_CHECKPOINTED",reason="Registered deadline reached; inspect saved stage/checkpoints",artifact_paths=[])
            error=traceback.format_exc()
        except Exception as failure:
            result=dict(status="FAILED",reason=type(failure).__name__,artifact_paths=[])
            error=traceback.format_exc()
        finally:
            ctx.pop("lease",None)
        if "error" in locals():
            directory=ctx["cache"]/"errors"
            directory.mkdir(exist_ok=True)
            (directory/(job["job_id"]+".txt")).write_text(error,encoding="utf-8")
            del error
        elapsed=time.time()-started
        result.update(started_unix=started,consumed_resource=dict(wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu,gpu_lease_seconds=elapsed if gpu else 0))
        finish_job(path,job,result,ctx)
        ledger=campaign_ledger(ctx["cache"]/"CAMPAIGN.json")
        ledger["gpu_seconds"]=resources.gpu_usage_seconds(ctx["cache"])
        ledger["cpu_worker_seconds"]+=0 if gpu else elapsed
        ledger["process_cpu_seconds"]+=result["consumed_resource"]["process_cpu_seconds"]
        atomic_json(ctx["cache"]/"CAMPAIGN.json",ledger)
        print(job["job_id"],result["status"],f"{elapsed:.2f}s",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["audit","plan","execute","report","verify","all"])
    parser.add_argument("--repo",type=Path,default=REPO)
    parser.add_argument("--asset-root",type=Path)
    parser.add_argument("--cache-root",type=Path)
    parser.add_argument("--report-root",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--only",nargs="+")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.mode is None:
        print("SOURCE_ONLY: choose --mode explicitly; no model, assets, downloads or reports opened")
        return 0
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="2"
    ctx=context(args)
    if args.mode in ("audit","all"):
        from .audit import audit
        audit(ctx)
    plan_path=args.plan or ctx["report"]/"EXPERIMENT_PLAN.json"
    if args.mode in ("plan","all"):
        if args.plan is not None and args.plan.exists():
            validate_plan(read(args.plan),ctx["repo"])
        else:
            plan=make_plan(ctx)
            print(plan["plan_sha256"],len(plan["jobs"]),"jobs")
    if args.mode in ("execute","all"):
        if args.mode=="execute" and args.plan is None:
            parser.error("--plan required for execute")
        execute(ctx,plan_path,args.only,args.worker)
    if args.mode in ("report","verify","all"):
        from .reporting import build_report,verify
        if args.mode in ("report","all"):
            build_report(ctx)
        if args.mode in ("verify","all"):
            result=verify(ctx)
            return 0 if result["status"]=="PASS" else 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
