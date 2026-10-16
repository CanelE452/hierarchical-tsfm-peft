"""Standard-library synthetic metadata and mocked execution only."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import extended_parent_e2 as bridge


class ExtendedParentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="extended_e2_toy_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"public"; self.root.mkdir()
        self.cache = Path(self.temp.name)/"private"; self.cache.mkdir()
        for name in bridge.CORE_FILES:
            shutil.copyfile(Path(bridge.__file__).with_name(name), self.root/name)
        self.assets = Path(self.temp.name)/"assets"; self.assets.mkdir()
        self.bounds = dict(A_fit=[0,1000], A_stop=[1000,1200], M_cal=[1200,1400], V_select=[1400,1600], T_final=[1600,2000])
        self.write(self.root/"DATA_CONTRACT.json", dict(assets=dict(beijing=dict(shape=[2000,11], L=512,H=48,
            data_sha256="toy_data",primary_split_ranges=self.bounds))))
        self.rows=[]
        for family in ("DIRECT_G","B0","B2"):
            self.parent(family,.25)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")

    def registry(self):
        with (self.root/"EXPERIMENT_REGISTRY.csv").open("w",encoding="utf-8",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)

    def parent(self,family,endpoint,chained=False):
        k=0 if family=="DIRECT_G" else 3
        base=f"primary_beijing_E1_{family}_k{k}_s92601"
        run=base if endpoint==.25 else base+("__e1__e3" if chained else "__e"+str(int(endpoint)))
        epochs=[e for e in (.25,1.,3.) if e<=endpoint]
        row=dict(experiment_id=run,dataset="beijing",panel="",fold="primary",block="E1",family=family,
            num_channels="11",latent_k=str(k),seed="92601",length="512",horizon="48",requirement="MANDATORY_MINIMUM",
            execution_status="COMPLETE_VALID",training_status="LIMITED_SCREEN",validity_status="NOT_CHECKED",
            scientific_status="UNASSESSED",publication_status="NOT_PUBLISHED",reason="toy",planned_epoch_endpoints="|".join(map(str,epochs)),resume_command="")
        self.rows.append(row);self.registry()
        spec=dict(run_id=run,dataset="beijing",family=family,fold="primary",role="base_parent",latent_k=k or 11,
            seed=92601,epochs=epochs,lr=.001,microbatch=4,effective_batch=16)
        source=dict(files=bridge.SOURCE_SIX,source_sha=bridge.fingerprint(bridge.SOURCE_SIX),spec=spec,
                    data_sha="toy_data",model_sha="NO_FOUNDATION" if family=="DIRECT_G" else "toy_frozen_F")
        artifact=self.cache/"runs"/run/"selected_bundle.pt";artifact.parent.mkdir(parents=True);artifact.write_bytes(b"synthetic opaque artifact, not model weights")
        receipt=dict(run_id=run,status="COMPLETE_VALID",family=family,epochs=epochs,selected_epoch=0.,search_fit_end=1000,
            refit_fit_end=1200,normalization_fit_end=1200,source_sha=source["source_sha"],
            artifact_cache_relative_path=artifact.relative_to(self.cache).as_posix(),artifact_sha256=bridge.file_sha(artifact))
        self.write(self.root/"runs"/run/"SOURCE_CONTRACT.json",source)
        self.write(self.root/"runs"/run/"BASE_FIT_RECEIPT.json",receipt)
        return run

    def extended(self):
        for family in ("DIRECT_G","B0","B2"):self.parent(family,1.)
        return bridge.build_plan(self.root,self.cache,datasets=["beijing"])

    def test_highest_common_requested_endpoint_and_chained_actual_ids(self):
        for family in ("DIRECT_G","B0","B2"):self.parent(family,1.)
        self.parent("DIRECT_G",3.,True)
        plan=bridge.build_plan(self.root,self.cache,datasets=["beijing"])
        self.assertEqual(plan["datasets"][0]["endpoint"],1.)
        for family in ("B0","B2"):self.parent(family,3.,True)
        item=bridge.build_plan(self.root,self.cache,datasets=["beijing"])["datasets"][0]
        self.assertEqual(item["endpoint"],3.)
        self.assertTrue(all(p["run_id"].endswith("__e1__e3") for p in item["parents"].values()))
        self.assertTrue(all(p["selected_epoch"]==0. for p in item["parents"].values()))

    def test_quarter_only_reuses_initial_E2_without_new_rows_or_fits(self):
        before={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        plan=bridge.build_plan(self.root,self.cache,datasets=["beijing"])
        self.assertEqual(plan["datasets"][0]["disposition"],"REUSE_INITIAL_E2_SKIP")
        self.assertEqual(plan["proposed_registry_rows"],[])
        self.assertEqual(plan["new_fit_count"],0)
        self.assertFalse(plan["datasets"][0]["initial_E2_completion_claimed"])
        self.assertEqual(before,{p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_AplusM_fold_source_and_refit_leakage_are_rejected(self):
        run=self.rows[0]["experiment_id"];sp=self.root/"runs"/run/"SOURCE_CONTRACT.json";rp=sp.with_name("BASE_FIT_RECEIPT.json")
        original=json.loads(sp.read_text());receipt=json.loads(rp.read_text())
        for mutate in (lambda s:s["spec"].update(role="AplusM_control"),lambda s:s["spec"].update(fold="secondary"),lambda s:s["files"].update({"training.py":"wrong"})):
            value=copy.deepcopy(original);mutate(value);self.write(sp,value)
            plan=bridge.build_plan(self.root,self.cache,datasets=["beijing"])
            self.assertEqual(plan["datasets"][0]["disposition"],"DEFERRED_SOURCE_PARENT")
            self.assertEqual(plan["new_fit_count"],0)
        self.write(sp,original);receipt["refit_fit_end"]=1400;self.write(rp,receipt)
        self.assertEqual(bridge.build_plan(self.root,self.cache,datasets=["beijing"])["datasets"][0]["disposition"],"DEFERRED_SOURCE_PARENT")

    def test_old_source_omission_does_not_hide_valid_higher_common_parent(self):
        run=self.rows[0]["experiment_id"];path=self.root/"runs"/run/"SOURCE_CONTRACT.json"
        source=json.loads(path.read_text());source["files"]["training.py"]="historical";self.write(path,source)
        plan=self.extended();item=plan["datasets"][0]
        self.assertEqual(item["endpoint"],1.)
        self.assertEqual(item["parent_omissions"][0]["run_id"],run)
        self.assertEqual(item["parent_omissions"][0]["actual_source_files"]["training.py"],"historical")
        self.assertFalse(item["parent_omissions"][0]["equivalence_to_current_source_claimed"])

    def test_explicit_new_parents_alpha_four_aliases_and_past_G(self):
        plan=self.extended();ds=plan["datasets"][0];routes=ds["routes"]
        self.assertEqual(len(routes),4);self.assertEqual(sum(len(r["registry_rows"]) for r in routes),7)
        self.assertEqual(len(plan["proposed_registry_rows"]),7)
        alpha=next(r for r in routes if r["spec"]["variant"]=="alpha_family")
        self.assertEqual(len(alpha["registry_rows"]),4)
        self.assertEqual({r["family"] for r in alpha["registry_rows"]},{"ALPHA0","ALPHA05","ALPHA1","ALPHA_FIT"})
        for route in routes:
            spec=route["spec"];self.assertEqual(spec["g_run_id"],ds["parents"]["DIRECT_G"]["run_id"])
            self.assertEqual(spec["epoch_endpoints"],[.25,1.,3.]);self.assertEqual(spec["learning_rate"],.001)
        past=next(r for r in routes if r["spec"]["variant"]=="past_control")
        self.assertEqual(past["spec"]["parent_run_id"],past["spec"]["g_run_id"])

    def test_existing_conflicting_scientific_identity_is_rejected(self):
        plan=self.extended();row=dict(plan["proposed_registry_rows"][0]);row["family"]="B4";self.rows.append(row);self.registry()
        with self.assertRaises(ValueError):bridge.build_plan(self.root,self.cache,datasets=["beijing"])

    def runtime(self,plan,fail_variant=None,busy=False):
        events=[];root,cache=self.root,self.cache;test=self;parents=plan["datasets"][0]["parents"]
        lookup={v["run_id"]:v for v in parents.values()}
        class Budget(RuntimeError):pass
        class Busy(RuntimeError):pass
        class Lease:
            lease_id="toy_lease";deadline=10**12
            def __init__(self,*args,**kw):self._entered=False;events.append(("reserve",kw))
            def __enter__(self):
                if busy:raise Busy("private-path must not be public")
                self._entered=True;events.append(("enter",));return self
            def check(self):events.append(("lease_check",))
            def __exit__(self,*args):
                finished=bridge.time.time()
                self._entered=False;test.write(cache/"gpu_receipts/toy_lease.json",dict(lease_id=self.lease_id,charged_process_seconds=2,status="CLOSED",
                    started_unix=finished-2,finished_unix=finished,deadline=self.deadline,deadline_overrun_seconds=max(0.,finished-self.deadline)));events.append(("exit",))
        def register(public,rows,reason):
            events.append(("register",len(rows)));existing={r["experiment_id"] for r in test.rows};added=[]
            for row in rows:
                if row["experiment_id"] not in existing:test.rows.append(dict(row));added.append(row)
            test.registry();return added
        reg=SimpleNamespace(require_pre_selection=lambda *a:bridge.require_unlocked(root),register=register)
        resources=SimpleNamespace(GPUReservation=Lease,BudgetExceeded=Budget,ResourceBusy=Busy)
        task=dict(name="beijing",data_sha256="toy_data",L=512,H=48,columns=list(map(str,range(11))),split_ranges=self.bounds)
        def load(ds):events.append(("load_task",ds));return task
        class Pool:
            def __init__(self,*a,**kw):events.append(("pool",kw))
            def predict(self,run,split="V_select",dense=False):
                events.append(("predict",run,split,dense));parent=lookup[run]
                return dict(metadata=dict(run_id=run,split=split,dense=dense,role="base_parent",fold="primary",data_sha256="toy_data",
                    parent_fit_end=1200,canonical_scaler_fit_end=1200,artifact_sha256=parent["artifact_sha256"],family=parent["family"],
                    model_sha="toy_selected_"+run,source_files=bridge.SOURCE_SIX,
                    cohort=f"beijing:primary:{split}:stride{1 if dense else 48}",reused_prediction_cache=False))
            def close(self):events.append(("close",))
        def prepare(run,*args,**kw):events.append(("prepare",run));return {"fake":run}
        def fit(spec,*args):
            events.append(("fit",spec["run_id"],spec["variant"]))
            if spec["variant"]==fail_variant:raise RuntimeError("PRIVATE_FAILURE_DETAIL")
            value=dict(run_id=spec["run_id"],execution_status="COMPLETE_VALID",training_status="NO_TRAINING" if spec["variant"]=="alpha_family" else "SCREENED_NOT_CONVERGENCE_PROVEN",updates=0 if spec["variant"]=="alpha_family" else 2)
            test.write(root/"combinations"/spec["run_id"]/"receipt.json",value);test.write(cache/"combinations"/spec["run_id"]/"spec.json",spec);return value
        def wrap(receipt,*args):
            events.append(("wrap",receipt["run_id"]))
            route=next(r for r in plan["datasets"][0]["routes"] if r["spec"]["run_id"]==receipt["run_id"])
            for row in route["registry_rows"]:
                path=root/"predictions"/row["experiment_id"]/"V_select/PREDICTION_MANIFEST.json"
                if not path.is_file():test.write(path,dict(run_id=row["experiment_id"],score=dict(mse=.1)))
            return []
        runtime=SimpleNamespace(data=SimpleNamespace(load_task=load,split_ranges=lambda *a:self.bounds,fit_statistics=lambda *a:([],[],[])),
            postfit=SimpleNamespace(ParentPool=Pool),combination=SimpleNamespace(fit_combination=fit,validate_completed_receipt=lambda *a:events.append(("validate_complete",))),
            audit=SimpleNamespace(prepare_combination_reference=prepare,wrap_combination_output=wrap))
        return events,resources,reg,runtime

    def execute(self,plan,**kwargs):
        events,resources,reg,runtime=self.runtime(plan,**kwargs)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        return result,events

    def test_mock_execution_registers_before_load_seals_before_four_fits_and_charges(self):
        plan=self.extended();result,events=self.execute(plan)
        self.assertEqual(result["fit_calls_attempted_now"],4)
        self.assertEqual(result["completed_new_fits"],4)
        self.assertEqual(result["completed_new_aliases"],7)
        self.assertEqual(result["new_alpha_family_fits"],1)
        kinds=[e[0] for e in events];self.assertLess(kinds.index("register"),kinds.index("load_task"))
        self.assertEqual(sum(e[0]=="predict" for e in events),6)
        flow=[e for e in events if e[0]!="lease_check"]
        for i,e in enumerate(flow):
            if e[0]=="fit":self.assertEqual(flow[i-1],("prepare",e[1]))
        self.assertTrue(result["central_lease_receipt_exists"])
        self.assertTrue(result["lease_receipt_cache_relative_path"].endswith("toy_lease.json"))
        self.assertFalse((self.root/"SCORES.csv").exists())

    def test_failed_head_preserves_unknown_training_and_independent_routes_continue(self):
        result,events=self.execute(self.extended(),fail_variant="past_control")
        self.assertEqual(result["fit_calls_attempted_now"],4)
        failed=next(r for r in result["routes"] if r["variant"]=="past_control")
        self.assertIsNone(failed["training_executed_now"])
        self.assertEqual(result["completed_new_fits"],3)
        self.assertNotIn("PRIVATE_FAILURE_DETAIL",json.dumps(result))
        self.assertTrue(any(e[0]=="fit" and e[2]=="alpha_family" for e in events))

    def test_T_lock_and_resource_busy_keep_public_receipt_without_runtime_import(self):
        plan=self.extended();self.write(self.root/"SELECTION_LOCK.json",dict(locked=True))
        with patch.object(bridge,"_load_runtime",side_effect=AssertionError("MODEL_IMPORT")) as heavy:
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        heavy.assert_not_called();self.assertTrue((self.root/result["batch_receipt_relative_path"]).is_file())
        (self.root/"SELECTION_LOCK.json").unlink();result,_=self.execute(plan,busy=True)
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        self.assertEqual(result["fit_calls_attempted_now"],0)

    def test_plan_main_does_not_import_model_modules_or_write(self):
        self.extended();before={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        private_before={p.relative_to(self.cache).as_posix():p.read_bytes() for p in self.cache.rglob("*") if p.is_file()}
        modules_before=set(sys.modules)
        with patch.object(bridge.importlib,"import_module",side_effect=AssertionError("LAZY_IMPORT")),patch.object(bridge,"_load_resources",side_effect=AssertionError("RESOURCE_IMPORT")),patch.object(bridge,"_load_runtime",side_effect=AssertionError("MODEL_IMPORT")),patch.object(sys,"stdout",new_callable=io.StringIO) as output:
            bridge.main(["--public-root",str(self.root),"--cache-root",str(self.cache),"--dataset","beijing"])
        self.assertTrue(json.loads(output.getvalue())["metadata_only"])
        self.assertEqual(before,{p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(private_before,{p.relative_to(self.cache).as_posix():p.read_bytes() for p in self.cache.rglob("*") if p.is_file()})
        self.assertFalse(any(n.split('.')[0] in ("torch","numpy","pandas") for n in set(sys.modules)-modules_before))

    def test_nested_lease_CPU_handoff_and_changed_plan_fail_before_import_or_registry(self):
        plan=self.extended();before=(self.root/"EXPERIMENT_REGISTRY.csv").read_bytes()
        for environment,changed,error in (
            ({"INFORMS_GPU_LEASE_ID":"another-owner"},False,"NO_NESTED_GPU_RESERVATION"),
            ({"CUDA_VISIBLE_DEVICES":"-1"},False,"GPU_HANDOFF_REQUIRED"),
            ({"CUDA_VISIBLE_DEVICES":"-1"},True,"PINNED_PLAN_IDENTITY_CHANGED")):
            supplied=copy.deepcopy(plan)
            if changed:supplied["plan_signature"]="changed"
            with patch.dict(bridge.os.environ,environment),patch.object(bridge,"_load_resources",side_effect=AssertionError("RESOURCE_IMPORT")) as resources,patch.object(bridge,"_load_runtime",side_effect=AssertionError("MODEL_IMPORT")) as runtime:
                result=bridge.execute(self.root,self.cache,self.assets,plan=supplied)
            self.assertEqual(result["status"],"EXECUTION_ABORTED")
            self.assertEqual(result["fatal_error_type"],"BridgeGuardError")
            private=json.loads((self.cache/result["private_failure_cache_relative_path"]).read_text())
            self.assertEqual(private["error"],error)
            self.assertTrue((self.root/result["batch_receipt_relative_path"]).is_file())
            resources.assert_not_called();runtime.assert_not_called()
            self.assertEqual((self.root/"EXPERIMENT_REGISTRY.csv").read_bytes(),before)

    def test_complete_fit_reuse_calls_validator_without_new_fits_or_aliases(self):
        plan=self.extended();first,_=self.execute(plan);self.assertEqual(first["completed_new_fits"],4)
        second,events=self.execute(plan)
        self.assertEqual(second["fit_calls_attempted_now"],0)
        self.assertEqual(second["complete_fits_reused"],4)
        self.assertEqual(second["completed_new_aliases"],0)
        self.assertEqual(second["reused_standard_aliases"],7)
        self.assertEqual(sum(e[0]=="validate_complete" for e in events),4)

    def test_partial_existing_output_is_preserved_and_other_routes_continue(self):
        plan=self.extended();run=plan["datasets"][0]["routes"][0]["spec"]["run_id"]
        path=self.root/"combinations"/run/"partial.json";self.write(path,dict(preserve="original"));before=path.read_bytes()
        result,_=self.execute(plan)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(result["fit_calls_attempted_now"],3)
        self.assertEqual(result["routes"][0]["status"],"EXECUTION_FAILED")

    def test_runtime_import_error_still_links_closed_charge_and_public_batch(self):
        plan=self.extended();events,resources,reg,_=self.runtime(plan)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",side_effect=ImportError("PRIVATE_IMPORT_PATH")):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertTrue(result["central_lease_receipt_exists"])
        self.assertTrue((self.root/result["batch_receipt_relative_path"]).is_file())
        self.assertNotIn("PRIVATE_IMPORT_PATH",json.dumps(result))
        self.assertEqual(result["fit_calls_attempted_now"],0)

    def test_V_timeout_defers_unattempted_routes_and_keeps_lease_receipt(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=runtime.postfit.ParentPool.predict
        def predict(pool,run,split="V_select",dense=False):
            if split=="V_select":raise TimeoutError("PRIVATE_DEADLINE_DETAIL")
            return original(pool,run,split,dense)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(runtime.postfit.ParentPool,"predict",predict):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        self.assertEqual(len(result["routes"]),4)
        self.assertTrue(all(r["status"]=="DEFERRED_RESOURCE" for r in result["routes"]))
        self.assertEqual(result["fit_calls_attempted_now"],0)
        self.assertTrue(result["central_lease_receipt_exists"])

    def test_final_source_check_failure_preserves_batch_and_closed_charge(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=runtime.postfit.ParentPool.close
        def close(pool):
            original(pool);(self.root/"training.py").write_text("changed synthetic copy",encoding="utf-8")
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(runtime.postfit.ParentPool,"close",close):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"EXECUTION_ABORTED_SOURCE_CHANGE")
        self.assertFalse(result["source_pins_unchanged"])
        self.assertTrue(result["central_lease_receipt_exists"])
        self.assertTrue((self.root/result["batch_receipt_relative_path"]).is_file())

    def test_last_alpha_fit_export_deadline_preserves_numeric_receipt_and_stops_wrap(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=runtime.combination.fit_combination
        clock=[1000.]
        def fit(spec,*args):
            receipt=original(spec,*args)
            if spec["variant"]=="alpha_family":clock[0]=resources.GPUReservation.deadline+1
            return receipt
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(runtime.combination,"fit_combination",fit),patch.object(bridge.time,"time",side_effect=lambda:clock[0]):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        alpha=next(r for r in result["routes"] if r["variant"]=="alpha_family")
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        self.assertEqual(result["resource_status"],"LIMIT_VIOLATION")
        self.assertEqual(result["completed_new_fits"],4)
        self.assertEqual(result["completed_new_aliases"],3)
        self.assertEqual(alpha["numeric_status"],"COMPLETE_VALID")
        self.assertTrue(alpha["executed_now"])
        self.assertFalse(alpha["training_executed_now"])
        self.assertTrue((self.root/alpha["fit_receipt_relative_path"]).is_file())
        self.assertFalse(any(e[0]=="wrap" and e[1]==alpha["run_id"] for e in events))

    def test_last_wrap_deadline_preserves_all_standard_outputs(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=runtime.audit.wrap_combination_output
        clock=[1000.]
        def wrap(receipt,*args):
            value=original(receipt,*args)
            if receipt["run_id"]==plan["datasets"][0]["routes"][-1]["spec"]["run_id"]:
                clock[0]=resources.GPUReservation.deadline+1
            return value
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(runtime.audit,"wrap_combination_output",wrap),patch.object(bridge.time,"time",side_effect=lambda:clock[0]):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        self.assertEqual(result["completed_new_fits"],4)
        self.assertEqual(result["completed_new_aliases"],7)
        self.assertTrue(result["numeric_completion"]["all_planned_standard_outputs_complete"])
        self.assertTrue(result["routes"][-1]["standard_outputs_complete"])
        self.assertEqual(result["resource_status"],"LIMIT_VIOLATION")
        rows={r["experiment_id"]:r for r in bridge._registry(self.root)[1]}
        for row in plan["datasets"][0]["routes"][-1]["registry_rows"]:
            self.assertEqual(rows[row["experiment_id"]]["validity_status"],"PENDING_INDEPENDENT_VALIDATION")

    def test_closed_lease_overrun_is_separate_from_numeric_completion(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=resources.GPUReservation.__exit__
        def close(lease,*args):
            original(lease,*args)
            path=self.cache/"gpu_receipts/toy_lease.json";value=json.loads(path.read_text())
            value.update(deadline_overrun_seconds=2.75,finished_unix=lease.deadline+2.75);self.write(path,value)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(resources.GPUReservation,"__exit__",close):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"RESOURCE_LIMIT_VIOLATION")
        self.assertEqual(result["resource_status"],"LIMIT_VIOLATION")
        self.assertEqual(result["deadline_overrun_seconds"],2.75)
        self.assertTrue(result["numeric_completion"]["all_planned_fits_complete"])
        self.assertTrue(result["numeric_completion"]["all_planned_standard_outputs_complete"])
        self.assertEqual(result["completed_new_aliases"],7)

    def test_closed_overrun_remains_visible_after_final_source_failure(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=resources.GPUReservation.__exit__
        def close(lease,*args):
            original(lease,*args)
            path=self.cache/"gpu_receipts/toy_lease.json";value=json.loads(path.read_text())
            value.update(deadline_overrun_seconds=3.,finished_unix=lease.deadline+3.);self.write(path,value)
            (self.root/"training.py").write_text("changed synthetic copy",encoding="utf-8")
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(resources.GPUReservation,"__exit__",close):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"EXECUTION_ABORTED_SOURCE_CHANGE")
        self.assertEqual(result["resource_status"],"LIMIT_VIOLATION")
        self.assertEqual(result["deadline_overrun_seconds"],3.)
        self.assertEqual(result["completed_new_fits"],4)
        self.assertEqual(result["completed_new_aliases"],7)

    def test_unverified_closed_charge_cannot_claim_resource_completion(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original=resources.GPUReservation.__exit__
        def close(lease,*args):
            original(lease,*args)
            path=self.cache/"gpu_receipts/toy_lease.json";value=json.loads(path.read_text())
            value["lease_id"]="different_closed_lease";self.write(path,value)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(resources.GPUReservation,"__exit__",close):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"RESOURCE_ACCOUNTING_UNVERIFIED")
        self.assertEqual(result["resource_status"],"UNVERIFIED_CLOSED_LEASE")
        self.assertFalse(result["lease_receipt_verified"])
        self.assertTrue(result["numeric_completion"]["all_planned_standard_outputs_complete"])
        self.assertEqual(result["completed_new_aliases"],7)

    def test_post_fit_storage_fence_preserves_returned_results_before_wrap(self):
        plan=self.extended();events,resources,reg,runtime=self.runtime(plan);original_fit=runtime.combination.fit_combination;original_storage=bridge._storage
        exported=[False];receipts={}
        def fit(spec,*args):
            receipt=original_fit(spec,*args)
            if spec["variant"]=="alpha_family":
                exported[0]=True
                path=self.root/"combinations"/spec["run_id"]/"receipt.json";receipts[path]=path.read_bytes()
            return receipt
        def storage(*args):
            if exported[0]:raise bridge.BridgeGuardError("DEFERRED_COMBINED_STORAGE")
            return original_storage(*args)
        with patch.object(bridge,"_require_gpu_handoff"),patch.object(bridge,"_load_resources",return_value=(resources,reg)),patch.object(bridge,"_load_runtime",return_value=runtime),patch.object(runtime.combination,"fit_combination",fit),patch.object(bridge,"_storage",storage):
            result=bridge.execute(self.root,self.cache,self.assets,plan=plan)
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        self.assertEqual(result["resource_status"],"DEFERRED_RESOURCE")
        self.assertEqual(result["completed_new_fits"],4)
        self.assertEqual(result["completed_new_aliases"],3)
        self.assertEqual(result["routes"][-1]["numeric_status"],"COMPLETE_VALID")
        self.assertTrue(all(path.read_bytes()==value for path,value in receipts.items()))


if __name__=="__main__":unittest.main()
