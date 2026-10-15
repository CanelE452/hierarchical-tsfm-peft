"""Tiny synthetic report folders only; no real report verification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_methodology_report_files_v1 as verify


class ReportFilesTests(unittest.TestCase):
    def fixture(self,base):
        folder=base/"report";folder.mkdir();(folder/"tables").mkdir();(folder/"figures").mkdir();(base/"source.py").write_text("# synthetic source\n",encoding="utf-8");(base/"input.json").write_text('{"status":"synthetic"}\n',encoding="utf-8")
        (folder/"tables/A.csv").write_text('key,value\n1,"two\nlines"\n2,three\n',encoding="utf-8");(folder/"figures/a.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic");(folder/"figures/a.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>',encoding="utf-8");(folder/"space name.json").write_text("{}",encoding="utf-8");(folder/"REPORT_KO.md").write_text('# fixture\n![png](figures/a.png)\n[table](tables/A.csv#rows)\n[encoded](space%20name.json)\n[remote](https://example.org/not-read)\n[anchor](#local)\n',encoding="utf-8")
        manifest={"status":"COMPLETE","source":verify.record(base/"source.py"),"inputs":{"synthetic":verify.record(base/"input.json")},"table_row_counts":{"A":2},"PNG_count":1,"SVG_count":1};self.repin(folder,manifest);return folder,manifest

    def repin(self,folder,manifest):
        manifest["files"]=[{**verify.record(path),"path":path.relative_to(folder).as_posix()} for path in sorted(folder.rglob("*")) if path.is_file() and path.name!="MANIFEST.json"];(folder/"MANIFEST.json").write_text(json.dumps(manifest),encoding="utf-8")

    def test_payload_hashes_sizes_source_inputs_CSV_and_decoded_local_links(self):
        with tempfile.TemporaryDirectory(dir=verify.HERE,prefix="synthetic_file_check_") as directory:
            folder,_=self.fixture(Path(directory));result=verify.verify(folder,1)
            self.assertEqual(result["status"],"PASS");self.assertEqual(result["payload_files_checked"],5);self.assertEqual(result["CSV_table_row_counts_checked"],{"A":2});self.assertEqual(result["local_links_checked"],3);self.assertEqual(result["bound_metadata_files_checked"],2);self.assertEqual(result["root_viewed_png_count_declared"],1);self.assertFalse(result["scientific_metrics_recomputed"]);self.assertFalse(result["visual_review_performed_by_checker"])

    def test_payload_hash_or_csv_count_change_fails(self):
        for kind in ("hash","rows"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=verify.HERE,prefix="synthetic_file_tamper_") as directory:
                folder,manifest=self.fixture(Path(directory));(folder/"tables/A.csv").write_text("key,value\n1,a\n2,b\n3,c\n",encoding="utf-8")
                if kind=="rows":self.repin(folder,manifest)
                with self.assertRaisesRegex(ValueError,"PAYLOAD_SHA|CSV_ROW_COUNT"):verify.verify(folder)

    def test_missing_link_and_encoded_escape_fail(self):
        for target in ("missing.csv","%2E%2E/%2E%2E/%2E%2E/%2E%2E/%2E%2E/escape.json"):
            with self.subTest(target=target),tempfile.TemporaryDirectory(dir=verify.HERE,prefix="synthetic_file_links_") as directory:
                folder,manifest=self.fixture(Path(directory));(folder/"REPORT_KO.md").write_text("[bad]("+target+")\n",encoding="utf-8");self.repin(folder,manifest)
                with self.assertRaisesRegex(ValueError,"EXISTING_LOCAL_FILE|PATH_CONTAINMENT"):verify.verify(folder)

    def test_source_sha_extra_payload_and_declared_view_count_fail(self):
        for kind in ("source","extra","view"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=verify.HERE,prefix="synthetic_file_bindings_") as directory:
                base=Path(directory);folder,_=self.fixture(base)
                if kind=="source":(base/"source.py").write_text("# altered",encoding="utf-8")
                elif kind=="extra":(folder/"extra.json").write_text("{}",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"BOUND_METADATA_SHA|EXACT_MANIFEST_PAYLOAD_SET|ROOT_VIEWED_PNG_COUNT"):verify.verify(folder,2 if kind=="view" else None)

    def test_meter_before_after_and_new_failure_output_are_preserved(self):
        import continuation_runtime_v1 as runtime
        calls=[]
        class Meter:
            def __init__(self,*args,**kwargs):calls.append("init")
            def check(self):pass
            def protected_check(self):calls.append("protected")
            def close(self,status,error):calls.append(status);return {"status":status,"synthetic_only":True}
        with tempfile.TemporaryDirectory(dir=verify.HERE,prefix="synthetic_file_execute_") as directory:
            base=Path(directory);folder,manifest=self.fixture(base);(folder/"REPORT_KO.md").write_text("[missing](missing.json)",encoding="utf-8");self.repin(folder,manifest);output=base/"check.json"
            with patch.object(runtime,"Meter",Meter):self.assertEqual(verify.execute(folder,output,None),2)
            result=json.loads(output.read_text(encoding="utf-8"));self.assertEqual(result["status"],"FAILED_PRESERVED");self.assertEqual(calls.count("protected"),2);self.assertEqual(result["resource_terminal"]["status"],"FAILED_PRESERVED")


if __name__=="__main__":unittest.main()
