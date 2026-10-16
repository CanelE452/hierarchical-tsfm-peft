"""Tiny public aggregate CSV/gzip fixtures only; no experiment arrays/models."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class PostTStats(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE/"pack_postT_stats.py").is_file(),"post-T pack helper is not implemented")
        self.h=load(HERE/"pack_postT_stats.py","tested_postT_stats")

    def fixture(self,folder,namespace="fixed_pca"):
        repo=Path(folder)/"repo";public=repo/"research/campaign";cache=Path(folder)/"cache"
        public.mkdir(parents=True);cache.mkdir()
        (public/"stats_pack.py").write_bytes((HERE/"stats_pack.py").read_bytes())
        core=self.h.load_original(public)
        post=public/self.h.NAMES[namespace];post.mkdir()
        sources={}
        for index in range(39):
            name="source%d.py"%index;(public/name).write_text("# synthetic frozen\n",encoding="utf-8")
            sources[name]=core.sha_file(public/name)
        protected={}
        for index in range(9):
            name="immutable%d.json"%index;(public/name).write_text("{}",encoding="utf-8")
            protected[name]=core.sha_file(public/name)
        (post/"REGISTRY.csv").write_text("experiment_id\ntoy\n",encoding="utf-8")
        plan=dict(source39=sources,immutable_public=protected,postT_registry_relative_path="REGISTRY.csv",
            postT_registry_sha256=core.sha_file(post/"REGISTRY.csv"),deadline_unix=1792222550,storage_limit_bytes=30*1024**3)
        (post/"PLAN.json").write_text(json.dumps(plan),encoding="utf-8")
        self.h.PLAN_SHAS[namespace]=core.sha_file(post/"PLAN.json")
        files=[];originals={}
        for index in range(2):
            path=post/"predictions"/("toy%d"%index)/"V_select/PUBLIC_ERROR_STATS.csv.gz"
            path.parent.mkdir(parents=True)
            raw=core.CSV_HEADER+("toy%d,V_select,toy:V,512,c,0,48,0.123456789012345,0.3,48\r\n"%index).encode()
            blob=gzip.compress(raw,compresslevel=9,mtime=0);path.write_bytes(blob)
            name=path.relative_to(repo).as_posix()
            files.append(dict(repository_relative_path=name,gzip_bytes=len(blob),gzip_sha256=core.sha(blob)))
            originals[name]=(blob,raw)
        selection=Path(folder)/"selection.json";selection.write_text(json.dumps(dict(files=files)),encoding="utf-8")
        return repo,public,cache,selection,core,originals

    def test_metadata_does_not_read_statistic_contents_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            repo,public,cache,selection,core,originals=self.fixture(folder)
            before=set(sys.modules)
            original=Path.read_bytes
            def read_bytes(path):
                if path.name=="PUBLIC_ERROR_STATS.csv.gz":raise AssertionError("No statistic content reads in metadata")
                return original(path)
            with patch.object(Path,"read_bytes",read_bytes):state=self.h.metadata(repo,public,cache,"fixed_pca",selection)
            self.assertEqual(len(state["records"]),2)
            self.assertFalse({"numpy","torch"}&(set(sys.modules)-before))
            self.assertFalse((public/"stats_packs").exists())

    def test_nested_pack_restores_original_gzip_and_CSV_bytes_with_original_APIs(self):
        with tempfile.TemporaryDirectory() as folder:
            repo,public,cache,selection,core,originals=self.fixture(folder)
            output=public/"stats_packs/pack_toy"
            result=self.h.execute(repo,public,cache,"fixed_pca",selection,output)
            manifest=core._load_manifest(output/"MANIFEST.json")
            self.assertEqual(manifest["campaign_repository_relative"],"research/campaign/"+self.h.NAMES["fixed_pca"])
            self.assertEqual(set(r["repository_relative_path"] for r in manifest["records"]),set(originals))
            destination=Path(folder)/"clone"
            self.assertEqual(core.restore(output/"MANIFEST.json",destination)["restored"],2)
            self.assertEqual(core.restore(output/"MANIFEST.json",destination)["already_matching"],2)
            for name,(blob,raw) in originals.items():
                self.assertEqual((repo/name).read_bytes(),blob)
                self.assertEqual((destination/name).read_bytes(),blob)
                self.assertEqual(gzip.decompress((destination/name).read_bytes()),raw)
            for archive in manifest["archives"]:self.assertTrue(core.inspect_archive(output/archive["file"]))
            self.assertEqual(result["verified_original_gzip_and_CSV_files"],2)
            self.assertIn("research/campaign/stats_pack.py restore",(output/"README_KO.md").read_text(encoding="utf-8"))

    def test_other_namespace_escape_or_duplicate_is_rejected(self):
        for change in ("namespace","escape","duplicate"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                repo,public,cache,selection,core,originals=self.fixture(folder)
                data=json.loads(selection.read_text(encoding="utf-8"))
                if change=="namespace":data["files"][0]["repository_relative_path"]=data["files"][0]["repository_relative_path"].replace(self.h.NAMES["fixed_pca"],self.h.NAMES["lag"])
                elif change=="escape":data["files"][0]["repository_relative_path"]="../outside.csv.gz"
                else:data["files"].append(dict(data["files"][0]))
                selection.write_text(json.dumps(data),encoding="utf-8")
                with self.assertRaises(ValueError):self.h.metadata(repo,public,cache,"fixed_pca",selection)

    def test_source_SHA_mismatch_and_existing_output_fail_without_overwrite(self):
        for change in ("statistic","core","output"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                repo,public,cache,selection,core,originals=self.fixture(folder)
                output=public/"stats_packs/pack_existing"
                if change=="statistic":(repo/next(iter(originals))).write_bytes(b"same length not required; changed")
                elif change=="core":(public/"stats_pack.py").write_text("changed",encoding="utf-8")
                else:output.mkdir(parents=True)
                with self.assertRaises((ValueError,FileExistsError)):
                    self.h.execute(repo,public,cache,"fixed_pca",selection,output)

    def test_guarded_budget_and_wall_refuse_before_payload_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            repo,public,cache,selection,core,originals=self.fixture(folder)
            for seconds in (0,1101):
                with self.assertRaises(ValueError):self.h.execute(repo,public,cache,"fixed_pca",selection,public/"stats_packs/pack_bad",wall_seconds=seconds)
            with patch.object(self.h,"storage_admission",side_effect=RuntimeError("synthetic storage cap")):
                with self.assertRaises(RuntimeError):self.h.execute(repo,public,cache,"fixed_pca",selection,public/"stats_packs/pack_cap")
            self.assertFalse((public/"stats_packs").exists())

    def test_tampered_archive_or_restore_destination_stays_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            repo,public,cache,selection,core,originals=self.fixture(folder)
            output=public/"stats_packs/pack_toy";self.h.execute(repo,public,cache,"fixed_pca",selection,output)
            manifest=core._load_manifest(output/"MANIFEST.json")
            destination=Path(folder)/"destination"
            wrong=destination/next(iter(originals));wrong.parent.mkdir(parents=True);wrong.write_bytes(b"existing mismatch")
            with self.assertRaises(ValueError):core.restore(output/"MANIFEST.json",destination)
            archive=output/manifest["archives"][0]["file"];archive.write_bytes(archive.read_bytes()+b"tampered")
            with self.assertRaises(ValueError):core.inspect_archive(archive)


if __name__=="__main__":unittest.main()
