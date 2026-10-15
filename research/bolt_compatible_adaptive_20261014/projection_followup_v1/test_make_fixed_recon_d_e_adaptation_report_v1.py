"""Generated metadata only: E-only freeze, procedure comparison, and separate new spaces."""
from pathlib import Path
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

from test_make_recon_init_sensitivity_report_v1 import fixture_values as prior_fixture

SOURCE=Path(__file__).with_name("make_fixed_recon_d_e_adaptation_report_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("Fixed RECON D report source missing")
    spec=importlib.util.spec_from_file_location("fixed_D_report_fixture",SOURCE)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def fixture_values(folder):
    prior=prior_fixture(folder); r=module(); joint=prior["train"]; train=copy.deepcopy(joint); audit=copy.deepcopy(prior["audit"])
    flags={"trainable_parameter_names":["E"],"D_requires_grad":False,"D_grad_is_None":True,"center_requires_grad":False}
    for seed in r.SEEDS:
        row=train["datasets"]["robin"]["conditions"][str(seed)]
        row.update(D_trainable=False,D_frozen_after_optimizer_step_checks=600,D_and_center_checkpoint_byte_checks=5,fixed_decoder_final_status=flags,initial_D_value_sha256="d"*64,D_bytes_checked_at_checkpoints_and_final_only=True,independent_per_update_D_bytes_replayed=False,gradient_clipping_parameter_scope="E_ONLY")
        for cp in row["checkpoints"]: cp.update(fixed_decoder_status=flags,D_and_center_exact_initial_bytes=True)
        row["probe"]={"status":"PASS","optimizer_updates":0,"component_byte_exact":{k:True for k in ("prediction","latent_history","latent_future","lag_features")},"fixed_decoder_status":flags,"artifact":row["probe"]["artifact"]}
        path=folder/(str(seed)+"_E_ONLY_UPDATES.jsonl")
        source_logs=Path(row["updates_log"]["path"]).read_text(encoding="utf-8").splitlines()
        logs=[{**json.loads(s),**flags,"D_gradient_norm_before_clip":0.,"gradient_clipping_parameter_scope":"E_ONLY"} for s in source_logs]
        path.write_text("".join(json.dumps(p)+"\n" for p in logs),encoding="utf-8"); row["updates_log"]=r.record(path)
        for cell in audit["new_output_space_diagnostics"][str(seed)].values():
            cell.update(selected_E_previously_supervised_by_FIT_future_targets=True,selected_D_previously_supervised_by_FIT_future_targets=False)
    return {"train":train,"audit":audit,"joint":joint,"joint_audit":prior["audit"],"science":prior["science"]}


class ReportTests(unittest.TestCase):
    def setUp(self): self.r=module(); self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parent); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_closed_gate_reads_nothing(self):
        with patch.object(self.r,"FIXED_SHA",{"train":None,"audit":None}),patch.object(self.r,"read",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_verified({"train":"missing","audit":"missing"})
    def test_scores_logs_spaces_and_new_vs_reused_counts(self):
        out=self.r.collect_tables(fixture_values(self.folder)); self.assertEqual([len(out[k]) for k in self.r.TABLES],[72,24,20,2400,4,2,8,24,8,2])
        self.assertEqual(sum(r["new_execution"] for r in out["UPDATES"]),1200)
        self.assertEqual({r["gradient_clipping_parameter_scope"] for r in out["CONDITIONS"]},{"E_ONLY","E_AND_D"})
    def test_frozen_D_flags_or_CP_bytes_cannot_be_asserted_if_false(self):
        for mutate in (lambda row:row["fixed_decoder_final_status"].__setitem__("D_requires_grad",True),lambda row:row["checkpoints"][2].__setitem__("D_and_center_exact_initial_bytes",False)):
            values=fixture_values(self.folder)
            mutate(values["train"]["datasets"]["robin"]["conditions"]["92601"])
            with self.assertRaisesRegex(ValueError,"FIXED_D"): self.r.collect_tables(values)
    def test_nonzero_D_gradient_is_not_silently_reported_as_frozen(self):
        values=fixture_values(self.folder); row=values["train"]["datasets"]["robin"]["conditions"]["92601"]; path=Path(row["updates_log"]["path"])
        logs=[json.loads(s) for s in path.read_text().splitlines()]; logs[0]["D_gradient_norm_before_clip"]=1.
        path.write_text("".join(json.dumps(p)+"\n" for p in logs),encoding="utf-8"); row["updates_log"]=self.r.record(path)
        with self.assertRaisesRegex(ValueError,"E_ONLY_LOG"): self.r.collect_tables(values)
    def test_CP0_zero_optimizer_and_component_bytes_are_required(self):
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"]["92602"]["probe"]["component_byte_exact"]["prediction"]=False
        with self.assertRaisesRegex(ValueError,"CP0"): self.r.collect_tables(values)
    def test_E_and_D_supervision_history_are_distinguished(self):
        values=fixture_values(self.folder); values["audit"]["new_output_space_diagnostics"]["92601"]["tune"]["selected_D_previously_supervised_by_FIT_future_targets"]=True
        with self.assertRaisesRegex(ValueError,"SEPARATE_SELECTED_E"): self.r.collect_tables(values)
    def test_text_preserves_objective_and_clipping_uncertainty(self):
        values=fixture_values(self.folder); text=self.r.make_text(values,self.r.collect_tables(values),{},self.folder)
        for word in ("1,200", "재사용", "RECON CP0", "Raw", "E_ONLY", "clipping", "매 update bytes", "감사 중 새 계산", "known-past offset", "미측정", "독립 일반화", "strict K-only", "F 입력 gradient"):
            self.assertIn(word,text)
    def test_synthetic_images_manifest_and_overwrite_refusal(self):
        values=fixture_values(self.folder/"metadata"); output=self.folder/"reports"/"toy"
        with patch.object(self.r,"HERE",self.folder),patch.object(self.r,"load_verified",return_value=(values,{})):
            manifest=self.r.generate({},output); self.assertEqual((manifest["PNG_count"],manifest["SVG_count"]),(3,3)); self.assertEqual(len(manifest["files"]),17)
            for entry in manifest["files"]: self.assertEqual(self.r.sha(output/entry["path"]),entry["sha256"])
            with self.assertRaisesRegex(ValueError,"FRESH"): self.r.generate({},output)


if __name__=="__main__": unittest.main()
