"""CPU toy checks: no real legacy source arrays, CUDA or foundation calls."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("legacy_evaluation", HERE/"legacy_evaluation.py")
le = importlib.util.module_from_spec(spec) if (HERE/"legacy_evaluation.py").exists() else None
if le is not None:
    spec.loader.exec_module(le)


class ToyWindow:
    def __init__(self):
        self.origins = {"A":np.array([2]), "B":np.array([4, 6]), "C":np.array([8])}
        self.fold = {"bounds":{"A":(0,4), "B":(4,8), "C":(8,10)}}
        self.calls = []
        self.horizon = 2

    def truth(self, origins):
        self.calls.append(("truth",list(origins)))
        if not np.array_equal(origins, self.origins["B"]):
            raise AssertionError("Non-B target access")
        y = np.array([[[1.,0.],[2.,4.]],[[3.,6.],[4.,8.]]], np.float32)
        mask = np.array([[[1,0],[1,1]],[[1,1],[0,1]]], bool)
        return y, mask

    def get(self, origins, length=512):
        self.calls.append(("get",list(origins)))
        if not set(origins).issubset(set(self.origins["B"])):
            raise AssertionError("Non-B forecast access")
        x = torch.tensor(origins, dtype=torch.float32)[:,None,None].expand(-1,3,2)
        return x, None, None


class ToyModel(torch.nn.Module):
    def forward(self, x, foundation):
        return foundation(x)


class ToyFoundation:
    def __init__(self):
        self.calls = self.streams = 0

    def __call__(self,x):
        self.calls += 1
        self.streams += x.shape[0]*x.shape[-1]
        return x[:,-1:].expand(-1,2,-1)


def inventory_fixture(root):
    asset,cache,public=root/"asset",root/"cache",root/"public"
    name="primary_jena_B0_r5_s92601"
    sources={"runtime.py":"frozen"};lock={"statistics_hash":"stats","F_model_hash":"F"}
    states={u:{"w":torch.tensor([float(u)])} for u in (2400,2800,4800)}
    rows=[dict(update=u,B={"mse":float(u),"mae":float(u)},A={},state_sha256=le.lc.tensor_state_hash(states[u])) for u in states]
    oldpath=asset/"research"/le.lc.OLD_NAME/"fits"/f"{name}_2400.json"
    le.lc.write_json(oldpath,dict(id=name,dataset="jena",kind="B0",r=5,seed=92601,updates=2400,checkpoints=[rows[0]]))
    oldlast=le.lc.checkpoint_path(asset,name);oldlast.parent.mkdir(parents=True)
    torch.save({"state":states[2400]},oldlast)
    oldcp=asset/".cache"/le.lc.OLD_NAME/"checkpoints"/name/"cp2400.pt"
    oldcp.parent.mkdir(parents=True);torch.save(states[2400],oldcp)
    cpdir=cache/"legacy_e0/checkpoints"/name;cpdir.mkdir(parents=True)
    for update in (2800,4800):torch.save(states[update],cpdir/f"cp{update}.pt")
    last=cache/"legacy_e0/training"/name/"last_4800.pt";last.parent.mkdir(parents=True)
    torch.save({"state":states[4800],"carry":{"updates":4800,"curve":[{}]*4800,"checkpoints":rows,
        "optimizer":{"state":{0:{"step":4800}}}}},last)
    selected=cache/"legacy_e0/selected"/f"{name}_4800.pt";selected.parent.mkdir(parents=True)
    torch.save(states[2400],selected)
    receipt=dict(id=name,endpoint=4800,status="CONTINUED_ENDPOINT_COMPLETE",source_hashes=sources,
        statistics_sha256="stats",F_model_hash="F",new_runner_source_sha256=le.lc.sha(Path(le.lc.__file__)),
        legacy_receipt_sha256=le.lc.sha(oldpath),legacy_last2400_sha256=le.lc.sha(oldlast),
        last_file_sha256=le.lc.sha(last),selected_file_sha256=le.lc.sha(selected),selected_update=2400,
        selected_state_sha256=rows[0]["state_sha256"],selected_B=rows[0]["B"],checkpoints=rows)
    le.lc.write_json(public/"legacy_e0/fits"/f"{name}_4800.json",receipt)
    return asset,cache,public,name,lock,sources,receipt


class LegacyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(le, "legacy evaluation implementation is required")

    def test_neighbor_prefix_fit_receipts_do_not_invalidate_own_inventory(self):
        for suffix in ("_G_s92601","_B0","_4800_suffix"):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as directory:
                asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                neighbor=dict(receipt,id=name+suffix)
                le.lc.write_json(public/"legacy_e0/fits"/f"{name+suffix}_4800.json",neighbor)
                with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                    route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                self.assertEqual(route["status"],"READY",route)
                self.assertEqual(len(route["endpoint_bindings"]),1)
                self.assertEqual([v["update"] for v in route["checkpoints"]],[2400,2800,4800])

    def test_neighbor_prefix_budget_receipts_do_not_invalidate_own_inventory(self):
        for suffix in ("_G_s92601","_B0","_4800_suffix"):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as directory:
                asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                neighbor=dict(id=name+suffix,endpoint=4800,initialized=True)
                le.lc.write_json(public/"legacy_e0/budgets"/f"{name+suffix}_4800_1792080000000000000.json",neighbor)
                with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                    route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                self.assertEqual(route["status"],"READY",route)
                self.assertEqual(len(route["endpoint_bindings"]),1)

    def test_own_fit_filename_id_endpoint_and_corruption_remain_fail_closed(self):
        cases=(dict(id="primary_jena_B0_r5_s92601_G_s92601"),dict(endpoint=9600),
               dict(status="BUDGET_LIMIT"),dict(selected_state_sha256="tampered"))
        for changes in cases:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as directory:
                asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                (public/"legacy_e0/fits"/f"{name}_4800.json").write_text(json.dumps(dict(receipt,**changes)),encoding="utf-8")
                with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                    route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                self.assertEqual(route["status"],"INVENTORY_FAILED",route)
        with tempfile.TemporaryDirectory() as directory:
            asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
            (public/"legacy_e0/fits"/f"{name}_4800.json").write_text("{broken",encoding="utf-8")
            with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                self.assertEqual(le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]["status"],"INVENTORY_FAILED")

    def test_own_id_in_neighbor_filename_and_malformed_neighbor_are_rejected(self):
        for folder in ("fits","budgets"):
            for malformed in (False,True):
                with self.subTest(folder=folder,malformed=malformed),tempfile.TemporaryDirectory() as directory:
                    asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                    filename=f"{name}_G_s92601_4800"+("_1792080000000000000" if folder=="budgets" else "")+".json"
                    path=public/"legacy_e0"/folder/filename
                    le.lc.write_json(path,dict(receipt,initialized=False))
                    if malformed:path.write_text("{broken",encoding="utf-8")
                    with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                        route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                    self.assertEqual(route["status"],"INVENTORY_FAILED",route)

    def test_own_budget_id_and_endpoint_mismatch_rejected_even_if_uninitialized(self):
        for changes in (dict(id="other"),dict(endpoint=9600)):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as directory:
                asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                rec=dict(id=name,endpoint=4800,initialized=False);rec.update(changes)
                le.lc.write_json(public/"legacy_e0/budgets"/f"{name}_4800_1792080000000000000.json",rec)
                with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                    route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                self.assertEqual(route["status"],"INVENTORY_FAILED",route)

    def test_exact_own_budget_keeps_carry_contract_and_file_hash_validation(self):
        for changed in (None,"source","file_hash"):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                asset,cache,public,name,lock,sources,receipt=inventory_fixture(Path(directory))
                checkpoint=cache/"legacy_e0/training"/name/"budget_u2800_target4800_123.pt"
                checkpoint.write_bytes(b"synthetic budget carry boundary")
                contract=dict(legacy_id=name,endpoint=4800,source_hashes=sources,statistics_sha256="stats",
                    foundation_sha256="F",runner_source_sha256=le.lc.sha(Path(le.lc.__file__)),
                    legacy_receipt_sha256=receipt["legacy_receipt_sha256"],legacy_last2400_sha256=receipt["legacy_last2400_sha256"],
                    resource_guard_source_sha256=le.lc.sha(Path(le.lc.__file__).with_name("resources.py")))
                if changed=="source":contract["source_hashes"]={"runtime.py":"changed"}
                le.lc.write_json(checkpoint.with_suffix(".pt.json"),{"budget_contract":contract})
                budget=dict(id=name,endpoint=4800,initialized=True,
                    budget_checkpoint_cache_relative_path=checkpoint.relative_to(cache).as_posix(),
                    budget_checkpoint_sha256="tampered" if changed=="file_hash" else le.lc.sha(checkpoint))
                le.lc.write_json(public/"legacy_e0/budgets"/f"{name}_4800_123.json",budget)
                with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                    with mock.patch.object(le.lc,"load_budget_checkpoint",return_value={"carry":{
                            "updates":2800,"checkpoints":receipt["checkpoints"][:2]}}):
                        route=le.inventory_checkpoints(asset,cache,public,[name])["routes"][0]
                self.assertEqual(route["status"],"READY" if changed is None else "INVENTORY_FAILED",route)
                if changed is None:
                    self.assertEqual(len(route["endpoint_bindings"]),2)
                    self.assertTrue(route["endpoint_bindings"][1]["partial_endpoint"])
                    self.assertEqual(route["endpoint_bindings"][1]["current_update"],2800)

    def test_fp64_mask_macro_and_public_statistics(self):
        y=np.array([[[1.,np.nan],[3.,2.]],[[4.,6.],[np.nan,8.]]])
        pred=np.array([[[2.,np.nan],[5.,5.]],[[7.,10.],[np.nan,13.]]])
        mask=np.isfinite(y)
        score,rows=le.score_fp64(pred,y,mask,np.array([4,6]),np.array(["a","b"]))
        self.assertEqual(score["counts"],[3,3])
        self.assertAlmostEqual(score["mse"],(14/3+50/3)/2)
        self.assertAlmostEqual(score["mae"],(6/3+12/3)/2)
        self.assertEqual(sum(row["count"] for row in rows),6)
        self.assertEqual(sum(row["sse"] for row in rows),64.)
        with self.assertRaises(ValueError):
            le.score_fp64(pred,y,np.ones_like(mask),np.array([4,6]),np.array(["a","b"]))

    def test_only_B_reference_is_frozen_before_forecasts_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);win=ToyWindow();foundation=ToyFoundation()
            identity={"legacy_id":"toy", "checkpoint_update":2800, "model_file_sha256":"file",
                      "model_state_sha256":"state", "foundation_sha256":"F"}
            result=le.evaluate_checkpoint(ToyModel(),foundation,win,["a","b"],identity,
                        root/"cache",root/"public",expected_B=None,device="cpu",horizon=2,batch_size=1)
            self.assertEqual(win.calls[0],("truth",[4,6]))
            self.assertEqual([call[0] for call in win.calls],["truth","get","get"])
            self.assertEqual(result["cost"]["actual_F_streams"],4)
            self.assertFalse(result["manifest"]["historically_unseen"])
            self.assertFalse(result["manifest"]["independent_training_rerun"])
            self.assertEqual(result["manifest"]["split"],"legacy_B_development")
            le.rescore_checkpoint(root/"cache",root/"public")
            archive=root/"cache"/"predictions.npz"
            with np.load(archive,allow_pickle=False) as value:
                arrays={key:value[key] for key in value.files}
            arrays["target_normalized"][0,0,0] += 1
            with archive.open("wb") as stream:np.savez_compressed(stream,**arrays)
            with self.assertRaisesRegex(ValueError,"archive"):
                le.rescore_checkpoint(root/"cache",root/"public")

    def test_curve_strict_comparison_is_reported_without_fabrication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=le.evaluate_checkpoint(ToyModel(),ToyFoundation(),ToyWindow(),["a","b"],
                {"legacy_id":"toy","checkpoint_update":2800},root/"cache",root/"public",
                expected_B={"mse":0.,"mae":0.},device="cpu",horizon=2)
            self.assertEqual(result["status"],"CURVE_SCORE_MISMATCH")
            self.assertGreater(result["rescore"]["curve_comparison"]["max_abs_difference"],1e-7)
            self.assertFalse(result["rescore"]["curve_comparison"]["passed"])

    def test_invalid_B_origins_rejected_before_any_truth_or_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            win=ToyWindow();win.origins["B"]=np.array([4,8]);f=ToyFoundation()
            with self.assertRaises(ValueError):
                le.evaluate_checkpoint(ToyModel(),f,win,["a","b"],{},Path(directory)/"c",Path(directory)/"p",
                                       device="cpu",horizon=2)
            self.assertEqual(win.calls,[])
            self.assertEqual(f.calls,0)

    def test_gpu_guard_prevents_any_work_without_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            win=ToyWindow();f=ToyFoundation()
            with self.assertRaises(RuntimeError):
                le.evaluate_checkpoint(ToyModel(),f,win,["a","b"],{},Path(directory)/"c",Path(directory)/"p",
                                       device="cuda",horizon=2)
            self.assertEqual(win.calls,[])
            self.assertEqual(f.calls,0)

    def test_reference_cannot_be_created_retroactively(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"c").mkdir();(root/"c/predictions.npz").write_bytes(b"orphan")
            with self.assertRaisesRegex(ValueError,"retrospectively"):
                le.evaluate_checkpoint(ToyModel(),ToyFoundation(),ToyWindow(),["a","b"],{},root/"c",root/"p",
                                       device="cpu",horizon=2)

    def test_inventory_binds_every_new_cp_last_and_selected_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);asset=root/"asset";cache=root/"cache";public=root/"public"
            name="toy";sources={"runtime.py":"frozen"};lock={"statistics_hash":"stats","F_model_hash":"F"}
            states={u:{"w":torch.tensor([float(u)])} for u in (2400,2800,4800)}
            rows=[dict(update=u,B={"mse":float(u),"mae":float(u)},A={},state_sha256=le.lc.tensor_state_hash(states[u])) for u in states]
            oldpath=asset/"research"/le.lc.OLD_NAME/"fits/toy_2400.json"
            old=dict(id=name,dataset="jena",kind="B4",r=1,seed=92601,updates=2400,checkpoints=[rows[0]])
            le.lc.write_json(oldpath,old)
            oldlast=le.lc.checkpoint_path(asset,name);oldlast.parent.mkdir(parents=True);torch.save({"state":states[2400]},oldlast)
            oldcp=asset/".cache"/le.lc.OLD_NAME/"checkpoints"/name/"cp2400.pt"
            oldcp.parent.mkdir(parents=True);torch.save(states[2400],oldcp)
            cpdir=cache/"legacy_e0/checkpoints"/name;cpdir.mkdir(parents=True)
            for u in (2800,4800):torch.save(states[u],cpdir/f"cp{u}.pt")
            last=cache/"legacy_e0/training"/name/"last_4800.pt";last.parent.mkdir(parents=True)
            carry={"updates":4800,"curve":[{}]*4800,"checkpoints":rows,"optimizer":{"state":{0:{"step":4800}}}}
            torch.save({"state":states[4800],"carry":carry},last)
            selected=cache/"legacy_e0/selected/toy_4800.pt";selected.parent.mkdir(parents=True);torch.save(states[2400],selected)
            receipt=dict(id=name,endpoint=4800,status="CONTINUED_ENDPOINT_COMPLETE",source_hashes=sources,
                statistics_sha256="stats",F_model_hash="F",new_runner_source_sha256=le.lc.sha(Path(le.lc.__file__)),
                legacy_receipt_sha256=le.lc.sha(oldpath),legacy_last2400_sha256=le.lc.sha(oldlast),
                last_file_sha256=le.lc.sha(last),selected_file_sha256=le.lc.sha(selected),
                selected_update=2400,selected_state_sha256=rows[0]["state_sha256"],selected_B=rows[0]["B"],checkpoints=rows)
            le.lc.write_json(public/"legacy_e0/fits/toy_4800.json",receipt)
            with mock.patch.object(le.lc,"validate_sources",return_value=(lock,sources)):
                result=le.inventory_checkpoints(asset,cache,public,[name])
                self.assertEqual(result["routes"][0]["status"],"READY")
                self.assertEqual([row["update"] for row in result["routes"][0]["checkpoints"]],[2400,2800,4800])
                self.assertTrue(result["routes"][0]["endpoint_bindings"][0]["last_state_verified"])
                torch.save({"w":torch.tensor([0.])},selected)
                tampered=le.inventory_checkpoints(asset,cache,public,[name])
                self.assertEqual(tampered["routes"][0]["status"],"INVENTORY_FAILED")
                self.assertIn("last/selected",tampered["routes"][0]["reason"])

    def test_B_prefix_does_not_compute_or_extract_C_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);asset=root/"asset";asset.mkdir()
            source=asset/"safe.npz";values=np.arange(52,dtype=np.float32).reshape(26,2)
            values[19:]=np.nan  # C and later observed values would fail a full-slice finite check.
            np.savez(source,physical=values,mask=np.ones_like(values,bool),times=np.arange(26,dtype=np.int64)*1_000_000_000,
                     columns=np.array(["a","b"]))
            mean=values[3:11].astype(np.float64).mean(0).astype(np.float32)
            std=values[3:11].astype(np.float64).std(0).astype(np.float32)
            statpath=asset/"research"/le.lc.OLD_NAME/"FOLD_STATISTICS.json"
            le.lc.write_json(statpath,{"toy":{"primary":{"A_mean":mean.tolist(),"A_std":std.tolist()}}})
            spec=dict(path="safe.npz",bytes=source.stat().st_size,sha256=le.lc.sha(source),permitted_bounds=(3,23),
                      channels=2,kind="physical",interval_seconds=1,contract_sha256="contract")
            fold={"bounds":{"A":(0,8),"B":(8,16),"C":(16,20)},
                  "origins":{"A":np.arange(2,7),"B":np.array([8,10,12,14]),"C":np.array([16,18])}}
            def windows(data,actual_fold):
                self.assertEqual(len(data["physical"]),16)
                self.assertEqual(len(actual_fold["origins"]["C"]),0)
                np.testing.assert_array_equal(data["physical"],values[3:19])
                return types.SimpleNamespace(mean=mean,std=std,origins=actual_fold["origins"])
            dio=types.SimpleNamespace(ASSETS={"toy":spec},_read_contract=lambda value:{"columns":["a","b"],"bounds":{"fit":[3,23]}},
                install_data_access_guard=lambda:None,make_fold=lambda rows,fold:__import__("copy").deepcopy(fold_data))
            fold_data=fold
            modules={"data":dio,"campaign":types.SimpleNamespace(Windows=windows)}
            lease=types.SimpleNamespace(check=lambda:None,remaining=100,deadline=__import__("time").time()+100)
            win,identity,columns=le.load_B_window(modules,asset,{"dataset":"toy","legal_A_origins":5},lease)
            self.assertEqual(columns,["a","b"])
            self.assertEqual(identity["B_bounds"],[8,16])
            self.assertEqual(identity["array_computation_scope"],"old_FIT_A_plus_B_prefix_only")


if __name__ == "__main__":
    unittest.main()
