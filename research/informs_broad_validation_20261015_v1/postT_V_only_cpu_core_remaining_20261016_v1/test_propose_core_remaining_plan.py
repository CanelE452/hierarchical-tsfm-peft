import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import propose_core_remaining_plan as api


def recipe(dataset="jena", label="B2", k=11, channels=21):
    row=dict(initial_registered_ID="initial_"+dataset+label, dataset=dataset, family=label, registered_K=str(k),
             C=str(channels),L="512",H="12" if dataset=="pems08" else "48",seed="92601",fold="primary")
    task=dict(columns=list(range(channels)),L=512,H=int(row["H"]),data_sha256="data")
    family={"LORA_R8":"LORA","ADAPTS_BOLT_PORT":"ADAPTS"}.get(label,label)
    spec=dict(run_id=row["initial_registered_ID"],dataset=dataset,family=family,latent_k=channels if family=="LORA" else k,
        seed=92601,fold="primary",microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,
        lr=.0001 if family=="LORA" else .001,device="cuda",epochs=[.25],lora_rank=8,lora_alpha=16)
    sources={name:name+"_digest" for name in api.SOURCE6}
    return row,dict(spec=spec,files=copy.deepcopy(sources),data_sha="data"),task,sources


class ProposalTests(unittest.TestCase):
    def test_bare_default_has_zero_input_reads_or_writes(self):
        with patch.object(Path,"read_bytes",side_effect=AssertionError("unexpected read")), patch.object(Path,"open",side_effect=AssertionError("unexpected open")), patch("sys.stdout",new_callable=io.StringIO) as out:
            self.assertEqual(api.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["experiment_inputs_read"],0)

    def test_exact_scope_retains_all_29_and_rejects_duplicate(self):
        rows=[dict(initial_registered_ID="row_"+str(i),dataset=("beijing","jena","electricity","pems08")[i%4],
              remaining_after_all_planned8_success_SCENARIO_ONLY=str(i<29)) for i in range(56)]
        self.assertEqual(len(api.selected_scope(rows)),29)
        rows[-1]["initial_registered_ID"]=rows[0]["initial_registered_ID"]
        with self.assertRaises(ValueError):api.selected_scope(rows)

    def test_scope_does_not_drop_missing_route_to_fit_count(self):
        rows=[dict(initial_registered_ID=str(i),dataset=("beijing","jena","electricity","pems08")[i%4],
              remaining_after_all_planned8_success_SCENARIO_ONLY=str(i<28)) for i in range(56)]
        with self.assertRaises(ValueError):api.selected_scope(rows)

    def test_high_channel_lora_actual_width_micro1_preserves_old_recipe(self):
        args=recipe("electricity","LORA_R8",0,321);old=copy.deepcopy(args[1])
        value=api.condition(*args)
        self.assertEqual((value["registered_latent_k"],value["latent_k"],value["effective_F_width"]),(0,321,321))
        self.assertEqual((value["spec"]["microbatch"],value["spec"]["effective_batch"],value["spec"]["lr"]),(1,16,.0001))
        self.assertIs(value["spec"]["checkpoint_inputs"],False)
        self.assertEqual(args[1],old)
        self.assertEqual(value["preflight_status"],"SOURCE_METADATA_COMPATIBLE")

    def test_small_micro4_and_adapts_mapping(self):
        a=api.condition(*recipe())
        self.assertEqual(a["spec"]["microbatch"],4)
        b=api.condition(*recipe("pems08","ADAPTS_BOLT_PORT",43,170))
        self.assertEqual((b["family"],b["spec"]["microbatch"],b["effective_F_width"]),("ADAPTS",1,43))

    def test_source_mismatch_retained_explicit_and_empty_source_rejected(self):
        args=recipe();args[1]["files"]["fit_base.py"]="changed"
        self.assertEqual(api.condition(*args)["preflight_status"],"BLOCKED_SOURCE_METADATA")
        args[1]["files"]={}
        self.assertEqual(api.condition(*args)["preflight_status"],"BLOCKED_SOURCE_METADATA")

    def test_recipe_or_task_change_rejected(self):
        for field,value in (("lr",.002),("microbatch",1),("checkpoint_inputs",True),("latent_k",7)):
            args=recipe();args[1]["spec"][field]=value
            with self.assertRaises(ValueError):api.condition(*args)
        args=recipe();args[2]["data_sha256"]="different"
        with self.assertRaises(ValueError):api.condition(*args)

    def test_registry_labels_registered_vs_actual_and_safe_relative_path(self):
        args=recipe("electricity","LORA_R8",0,321);c=api.condition(*args)
        rows=list(csv.DictReader(io.StringIO(api.registry_bytes([c],{"electricity":args[2]}).decode())))
        self.assertEqual((rows[0]["family"],rows[0]["latent_k"]),("LORA_R8","0"))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):api.relative(Path(d),"../outside")

    def test_draft_cannot_be_used_for_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/api.NAME).mkdir();path=root/api.NAME/"PLAN.json"
            path.write_text(json.dumps(dict(status="DRAFT_NOT_SEALED")))
            with patch.object(api,"build_proposal",side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(ValueError,"DRAFT"):api.metadata(root,root,root,api.sha(path))


if __name__=="__main__":unittest.main()
