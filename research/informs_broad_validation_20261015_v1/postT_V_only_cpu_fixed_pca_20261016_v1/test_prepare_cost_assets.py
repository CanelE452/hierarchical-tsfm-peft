"""Opaque synthetic cache bytes only; no NPZ decoding or real cache mutation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class Assets(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE/"prepare_cost_assets.py").is_file(),"portable opaque asset preparation not implemented")
        self.h=load("tested_opaque_cost_assets",HERE/"prepare_cost_assets.py")
        fixture=load("original_PCA_assets_fixture",HERE/"test_fixed_pca_runner.py")
        fixture.FixedPCATests.setUpClass();self.fixtures=fixture.FixedPCATests()

    def fixture(self,folder):
        public,cache,asset,plan=self.fixtures.fixture(Path(folder))
        (cache/"data").mkdir()
        assets=self.h.read(public/"DATA_CONTRACT.json")["assets"]
        records=[]
        for dataset,interval in (("electricity",3600),("pems08",300)):
            task=plan["task_contracts"][dataset]
            task["interval_seconds"]=interval
            task["metadata"]["source_sha256"]="a"*64
            assets[dataset]["interval_seconds"]=interval
            assets[dataset]["metadata"]["source_sha256"]="a"*64
            archive=cache/"data"/(dataset+"_permitted.npz")
            archive.write_bytes(("opaque synthetic archive "+dataset).encode())
            value=dict(name=dataset,shape=task["shape"],metadata=task["metadata"],interval_seconds=interval,
                cache_reference="data/"+archive.name,cache_sha256=self.h.sha(archive),cache_bytes=archive.stat().st_size,source_sha256="a"*64)
            receipt=cache/"data"/(dataset+"_receipt.json")
            receipt.write_text(json.dumps(value),encoding="utf-8")
            for path in (archive,receipt):records.append(dict(dataset=dataset,relative_path=path.relative_to(cache).as_posix(),sha256=self.h.sha(path),bytes=path.stat().st_size))
        (public/"DATA_CONTRACT.json").write_text(json.dumps(dict(assets=assets)),encoding="utf-8")
        plan["immutable_public"]["DATA_CONTRACT.json"]=self.h.sha(public/"DATA_CONTRACT.json")
        path=public/HERE.name/"PLAN.json";path.write_text(json.dumps(plan),encoding="utf-8")
        self.h.PLAN_SHA=self.h.sha(path);self.h.FILE_LOCKS=records
        return public,cache,asset,plan

    def test_metadata_is_no_science_no_write_and_four_exact_paths(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan=self.fixture(folder)
            files={str(p):self.h.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            state=self.h.metadata(public,cache,asset,self.h.PLAN_SHA)
            self.assertEqual(len(state["files"]),4)
            self.assertTrue(all(row["destination_status"]=="ABSENT" for row in state["files"]))
            self.assertEqual(files,{str(p):self.h.sha(p) for p in Path(folder).rglob("*") if p.is_file()})
        self.assertFalse({"numpy","torch"}&(set(sys.modules)-before))

    def test_copy_four_then_verify_existing_four_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan=self.fixture(folder)
            sources={row["relative_path"]:self.h.sha(cache/row["relative_path"]) for row in self.h.FILE_LOCKS}
            first=self.h.execute(public,cache,asset,self.h.PLAN_SHA,public/HERE.name/"first")
            self.assertEqual((first["copied_files"],first["verified_existing_files"]),(4,0))
            second=self.h.execute(public,cache,asset,self.h.PLAN_SHA,public/HERE.name/"second")
            self.assertEqual((second["copied_files"],second["verified_existing_files"],second["copied_bytes"]),(0,4,0))
            self.assertEqual(sources,{r["relative_path"]:self.h.sha(cache/r["relative_path"]) for r in self.h.FILE_LOCKS})
            self.assertTrue(second["original_source_registry_selection_score_cost_T_pins_preserved"])

    def test_existing_mismatch_is_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan=self.fixture(folder)
            target=cache/HERE.name/"data/electricity_permitted.npz"
            target.parent.mkdir(parents=True);target.write_bytes(b"wrong existing opaque bytes")
            with self.assertRaises(ValueError):self.h.execute(public,cache,asset,self.h.PLAN_SHA,public/HERE.name/"fail")
            self.assertEqual(target.read_bytes(),b"wrong existing opaque bytes")
            self.assertEqual(len(list(target.parent.iterdir())),1)

    def test_source_or_public_fingerprint_change_is_rejected(self):
        for change in ("source","scientific_source","plan_sha"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,plan=self.fixture(folder)
                if change=="source":(cache/self.h.FILE_LOCKS[0]["relative_path"]).write_bytes(b"changed source")
                if change=="scientific_source":(public/"data_contract.py").write_text("changed",encoding="utf-8")
                with self.assertRaises(ValueError):self.h.metadata(public,cache,asset,"wrong" if change=="plan_sha" else self.h.PLAN_SHA)

    def test_cache_reference_escape_and_wrong_file_count_are_rejected(self):
        for change in ("reference","count"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,plan=self.fixture(folder)
                if change=="reference":
                    receipt=cache/"data/electricity_receipt.json";value=self.h.read(receipt)
                    value["cache_reference"]="../outside.npz";receipt.write_text(json.dumps(value),encoding="utf-8")
                    for row in self.h.FILE_LOCKS:
                        if row["relative_path"]=="data/electricity_receipt.json":row.update(sha256=self.h.sha(receipt),bytes=receipt.stat().st_size)
                else:self.h.FILE_LOCKS=self.h.FILE_LOCKS[:-1]
                with self.assertRaises(ValueError):self.h.metadata(public,cache,asset,self.h.PLAN_SHA)

    def test_source_change_during_copy_preserves_failure_and_never_claims_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan=self.fixture(folder)
            original=self.h.shutil.copyfileobj
            changed=[False]
            def copying(source,destination,**kwargs):
                original(source,destination,**kwargs)
                if not changed[0]:
                    changed[0]=True
                    (cache/"data/pems08_permitted.npz").write_bytes(b"changed concurrently")
            with patch.object(self.h.shutil,"copyfileobj",side_effect=copying):
                with self.assertRaises(ValueError):self.h.execute(public,cache,asset,self.h.PLAN_SHA,public/HERE.name/"changed")
            result=self.h.read(public/HERE.name/"changed/COST_CACHE_ASSET_ACCESS.json")
            self.assertEqual(result["status"],"FAILED_PRESERVED")

    def test_existing_output_and_invalid_prior_proof_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan=self.fixture(folder)
            output=public/HERE.name/"exists";output.mkdir()
            with self.assertRaises(FileExistsError):self.h.execute(public,cache,asset,self.h.PLAN_SHA,output)
            with self.assertRaises(ValueError):self.h.execute(public,cache,asset,self.h.PLAN_SHA,public/HERE.name/"invalid",prior_proof_sha="unbound")


if __name__=="__main__":unittest.main()
