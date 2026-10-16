"""Actual public50-trial schema regression; no model/NPZ/process operations."""
import copy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).parent


def load(name):
    path=HERE/(name+".py")
    if not path.exists():raise AssertionError("RED: family-aware recovery v2 is not implemented")
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def actual(stem):
    folder=HERE/"diagnostics"/("family_initial_jena_"+stem+"_cost_b1_root_v1")
    with gzip.open(folder/"COST_TRIALS.csv.gz","rt",encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    return rows,json.loads((folder/"COST_RECEIPT.json").read_text(encoding="utf-8-sig"))


class FamilyStageSchemaTests(unittest.TestCase):
    def test_actual_lora_old_rejects_and_v2_accepts_four_stage_fields(self):
        old=load("audit_initial_family");rows,receipt=actual("lora_r8");run="postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1"
        with self.assertRaises(ValueError):old.aggregate_trials(rows,run,1,21)
        v2=load("audit_initial_family_recovery_v2");result=v2.aggregate_trials(rows,run,1,21)
        self.assertEqual(result["trial_rows"],50);self.assertEqual(result["F_logical_streams"],1050);self.assertEqual(result["F_calls"],50)
        for got,want in zip(result["summaries"],receipt["summaries"]):
            for field in ("median_seconds","mean_seconds","min_seconds","max_seconds","p90_seconds"):self.assertEqual(got[field],want[field])

    def test_actual_b4_keeps_six_fields_and_exact_frozen_result(self):
        old=load("audit_initial_family");v2=load("audit_initial_family_recovery_v2");rows,_=actual("b4_k6");run="postT_cpu_jena_E1_B4_k6_s92601_family_v1"
        self.assertEqual(v2.aggregate_trials(rows,run,1,6),old.aggregate_trials(rows,run,1,6))

    def test_actual_lora_counter_and_invented_stage_mutations_are_rejected(self):
        a=load("audit_initial_family_recovery_v2");rows,_=actual("lora_r8");run="postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1"
        for change in (lambda v:v[0].update(F_physical_streams="20"),lambda v:v[0].update(stage_P_E_seconds=""),lambda v:v[0].update(whole_seconds="nan"),lambda v:v[0].update(stage_F_seconds="0")):
            changed=copy.deepcopy(rows);change(changed)
            with self.assertRaises(ValueError):a.aggregate_trials(changed,run,1,21)

    def test_actual_b4_missing_stage_and_wrong_family_schema_are_rejected(self):
        a=load("audit_initial_family_recovery_v2");rows,_=actual("b4_k6");run="postT_cpu_jena_E1_B4_k6_s92601_family_v1"
        changed=copy.deepcopy(rows);changed[0].pop("stage_D_seconds")
        with self.assertRaises(ValueError):a.aggregate_trials(changed,run,1,6)
        lora,_=actual("lora_r8")
        with self.assertRaises(ValueError):a.aggregate_trials(lora,"postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1",1,6)

    def test_v2_defaults_remain_zero_io_without_scientific_imports(self):
        for name in ("audit_initial_family_recovery_v2","build_initial_family_recovery_v2"):
            load(name)
            r=subprocess.run([sys.executable,"-B",str(HERE/(name+".py"))],capture_output=True,text=True,check=True)
            v=json.loads(r.stdout);self.assertEqual(v["input_reads"],0);self.assertEqual(v["files_written"],0);self.assertFalse(v["numpy_imported"]);self.assertFalse(v["torch_imported"])


if __name__=="__main__":unittest.main()
