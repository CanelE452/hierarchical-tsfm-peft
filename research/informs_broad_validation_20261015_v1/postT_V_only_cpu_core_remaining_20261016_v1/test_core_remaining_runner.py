import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import core_remaining_runner as api
import propose_core_remaining_plan as p


def condition(micro=4,width=11,family="B2"):
    return dict(run_id="new_condition",family=family,effective_F_width=width,
                spec=dict(microbatch=micro),preflight_status="SOURCE_METADATA_COMPATIBLE")


def probe(c):
    micro=c["spec"]["microbatch"];width=c["effective_F_width"]
    count=48//micro;calls=api.expected_calls(48,micro,width)
    value=dict(segment=dict(status="SEGMENT_COMPLETE",updates=3,exposures=48),
        F_counter_delta=dict(streams=48*width,row_calls=calls,physical_streams=48*width,physical_calls=calls),
        probe_state_discarded=True,saved_model_or_optimizer=False,base_hash_restored_after_close=True,
        no_foundation_parameter_gradients=True,actual_parameter_gradients_finite=True,actual_parameter_gradient_nonzero=True,
        forward_events=[dict(shape=[micro,512,width],requires_grad=True,grad_enabled=True) for _ in range(count)],
        backward_events=[dict(shape=[micro,512,width],finite=True,absmax=.02) for _ in range(count)])
    if c["family"]=="LORA":value.update(gradient_scope="LORA_adapter_parameters_only",adapter_hooks_removed_after_close=True)
    return value


def fit_evidence(selected=1.):
    import math
    task=dict(H=48,primary_split_ranges=dict(A_fit=[0,10482],A_stop=[10482,12578]),refit_scaler_sha256="refit")
    c=condition();n50=9923;n60=12019;targets=[0]+[math.ceil(e*n50) for e in (.25,1,3)]
    updates=sum(math.ceil((b-a)/16) for a,b in zip(targets,targets[1:]));exposures=math.ceil(selected*n60)
    base=dict(status="COMPLETE_VALID",run_id=c["run_id"],family=c["family"],seed=92601,reused_complete_artifact=False,
        epochs=[.25,1.,3.],selected_epoch=selected,search_exposures=targets[-1],search_updates=updates,
        refit_exposures=exposures,refit_updates=math.ceil(exposures/16),search_fit_end=10482,refit_fit_end=12578,
        normalization_fit_end=12578,refit_scaler_sha="refit")
    search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=targets[-1],updates=updates,train_origins=n50)
    refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([selected] if selected else []),exposures=exposures,
               updates=math.ceil(exposures/16),train_origins=n60)
    return base,search,refit,c,task


class RunnerTests(unittest.TestCase):
    def test_default_no_experiment_reads_imports_or_writes(self):
        with patch.object(api,"proposer",side_effect=AssertionError("unexpected import")),patch.object(Path,"open",side_effect=AssertionError("unexpected open")),patch("sys.stdout",new_callable=io.StringIO) as out:
            self.assertEqual(api.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["writes"],0)

    def test_one_case_selection_ignores_unrelated_malformed_entries(self):
        c=condition();self.assertEqual(api.choose(dict(conditions=[None,[],c]),c["run_id"]),c)
        with self.assertRaises(ValueError):api.choose(dict(conditions=[c,c]),c["run_id"])
        with self.assertRaises(ValueError):api.choose(dict(conditions=[c]),"other")

    def test_micro1_fullC_original_call_counts(self):
        c=condition(1,321);api.validate_probe(probe(c),c)
        self.assertEqual(api.expected_calls(48,1,321),144)
        self.assertEqual(api.expected_calls(50,1,321),150)
        self.assertEqual(api.expected_calls(6,4,321),17)

    def test_detached_F_branch_rejected_even_if_parameter_grads_present(self):
        c=condition();value=probe(c);value["backward_events"]=[]
        with self.assertRaises(ValueError):api.validate_probe(value,c)
        value=probe(c)
        for row in value["backward_events"]:row["absmax"]=0
        with self.assertRaises(ValueError):api.validate_probe(value,c)

    def test_lora_adapter_close_guard_does_not_require_F_input_gradient(self):
        c=condition(1,170,"LORA");value=probe(c);value.pop("forward_events");value.pop("backward_events")
        api.validate_probe(value,c)
        value["adapter_hooks_removed_after_close"]=False
        with self.assertRaises(ValueError):api.validate_probe(value,c)

    def test_base_gradient_or_reused_probe_rejected(self):
        c=condition()
        for name in ("probe_state_discarded","no_foundation_parameter_gradients"):
            value=probe(c);value[name]=False
            with self.assertRaises(ValueError):api.validate_probe(value,c)

    def test_full_endpoint_updates_are_sum_of_segment_ceils(self):
        args=fit_evidence();self.assertEqual(args[0]["search_updates"],1863);api.validate_fit(*args)
        args[1]["updates"]=1861
        with self.assertRaises(ValueError):api.validate_fit(*args)

    def test_selected_zero_refit_is_valid_but_wrong_recipe_not(self):
        args=fit_evidence(0.);api.validate_fit(*args)
        args[0]["family"]="B4"
        with self.assertRaises(ValueError):api.validate_fit(*args)

    def test_existing_attempt_and_T_mode_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            public=Path(d)/"p";cache=Path(d)/"c";public.mkdir();cache.mkdir()
            path=cache/api.NAME/"runs"/"new_condition";path.mkdir(parents=True);sentinel=path/"keep";sentinel.write_bytes(b"original")
            with self.assertRaises(FileExistsError):api.require_fresh(public,cache,"new_condition","fit")
            with self.assertRaises(ValueError):api.require_fresh(public,cache,"new_condition","T_final")
            self.assertEqual(sentinel.read_bytes(),b"original")

    def test_prior_closure_live_and_identity_unknown_refused_dead_allowed(self):
        class Gone(Exception):pass
        class Process:
            def create_time(self):return 10
        class ProcessAPI:
            NoSuchProcess=Gone
            @staticmethod
            def Process(pid):return Process()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"closure.json";p.new(path,dict(status="ALL_PRIOR_CPU_OWNERS_CLOSED",remaining_owned_processes=[],owned_bindings=[dict(pid=42,create_time=10)]))
            digest=p.sha(path)
            with self.assertRaisesRegex(ValueError,"still live"):api.prior_closure(path,digest,p,ProcessAPI)
            with patch.object(ProcessAPI,"Process",side_effect=Gone):api.prior_closure(path,digest,p,ProcessAPI)
            with patch.object(ProcessAPI,"Process",side_effect=PermissionError):
                with self.assertRaises(PermissionError):api.prior_closure(path,digest,p,ProcessAPI)
            with patch.object(Process,"create_time",return_value=11):api.prior_closure(path,digest,p,ProcessAPI)

    def test_saved_variable_H_and_V_contract(self):
        c=condition(1,170);task=dict(H=12,columns=list(range(170)),legal_origin_counts_primary=dict(V_select=148),data_sha256="data",refit_scaler_sha256="scale")
        manifest=dict(run_id=c["run_id"],split="V_select",data_sha="data",scaler_sha="scale",arrays=dict(target_normalized=dict(shape=[148,12,170])))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/"predictions"/c["run_id"]/"V_select";folder.mkdir(parents=True);p.new(folder/"PREDICTION_MANIFEST.json",manifest)
            api.validate_saved(root,c,task,p)
            task["H"]=48
            with self.assertRaises(ValueError):api.validate_saved(root,c,task,p)

    def test_actual_source_contract_bound_to_sealed_science(self):
        c=condition();sources={name:name+"_sha" for name in p.SOURCE6}
        plan=dict(source39=sources,foundation_tensor_sha256="model");task=dict(data_sha256="data");base=dict(source_sha="source")
        source=dict(spec=copy.deepcopy(c["spec"]),files=sources,data_sha="data",model_sha="model",source_sha="source")
        api.validate_source_contract(source,base,c,plan,task)
        source["spec"]["microbatch"]=1
        with self.assertRaises(ValueError):api.validate_source_contract(source,base,c,plan,task)

    def test_budget_limit_is_not_successful_cli_exit(self):
        args=["--execute","--public-root","p","--cache-root","c","--asset-root","a","--output","o",
              "--plan-sha","sha","--run-id","registered","--mode","fit","--deadline","1",
              "--closure-proof","closure","--closure-sha","sha"]
        with patch.object(api,"execute",return_value=dict(status="BUDGET_LIMIT",error_class=None,postflight_error_class=None)),patch("sys.stdout",new_callable=io.StringIO):
            self.assertEqual(api.main(args),1)


if __name__=="__main__":unittest.main()
