"""Tiny explicit publication fixtures only; no real statistics or experiment API."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tested_numeric_publication",HERE/"publication_check_numeric.py")
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class NumericPublicationTests(unittest.TestCase):
    def setUp(self):self.h=load()

    def fixture(self,temp,run="synthetic_run",legacy=False):
        pc,codec=self.h.load_dependencies();core=codec.load_core()
        repo=Path(temp)/"repo";campaign=repo/"research/campaign";campaign.mkdir(parents=True)
        directory=campaign/"stats_packs"/("pack_legacy_tiny" if legacy else "pack_numeric_tiny");directory.mkdir(parents=True)
        raw=(",".join(core.COLUMNS)+"\r\n"+run+",V_select,primary,512,ch0,0,2,0.1,0.3,2\r\n").encode()
        blob=gzip.compress(raw,compresslevel=9,mtime=0)
        if legacy:
            payload=blob;record=dict(storage_mode="opaque_gzip",gzip_bytes=len(blob),gzip_sha256=core.sha(blob),
                csv_bytes=len(raw),csv_sha256=core.sha(raw),row_count=1,gzip_recipe=None,run_id_prefix=None)
            member="members/"+core.sha(payload)+".csv.gz";archives=[];assignment=core._emit([(member,payload)],directory,archives)
        else:
            record,payload=codec.encode_blob(blob);member="members/"+core.sha(payload)+".numeric"
            archives=[];assignment=codec._emit([(member,payload)],directory,archives)
        record.update(repository_relative_path="research/campaign/predictions/synthetic_run/V_select/PUBLIC_ERROR_STATS.csv.gz",
            member=member,archive=assignment[member])
        manifest=dict(format=core.SCHEMA if legacy else codec.SCHEMA,status="PASS",original_stats_pack_sha256=self.h.CORE_SHA,
            campaign_repository_relative="research/campaign",records=[record],archives=archives)
        if legacy:manifest.update(source_sha256=self.h.CORE_SHA,separate_temporary_restore_all_sha_verified=True)
        (directory/"MANIFEST.json").write_text(json.dumps(manifest),encoding="utf-8")
        archive=directory/archives[0]["file"]
        (campaign/"README.md").write_text("[stats]("+archive.relative_to(campaign).as_posix()+")\n",encoding="utf-8")
        names=[p.relative_to(repo).as_posix() for p in (campaign/"README.md",directory/"MANIFEST.json",archive)]
        return repo,campaign,names,archive,pc,codec

    def transport_fixture(self,temp):
        repo,campaign,names,archive,pc,codec=self.fixture(temp,legacy=True)
        transport=self.h.load_transport();original_manifest=archive.with_name("MANIFEST.json")
        selection=dict(format=transport.SELECTION_SCHEMA,
            original_archive_repository_relative_path=archive.relative_to(repo).as_posix(),
            original_archive_bytes=archive.stat().st_size,original_archive_sha256=self.h.sha(archive),
            original_manifest_repository_relative_path=original_manifest.relative_to(repo).as_posix(),
            original_manifest_bytes=original_manifest.stat().st_size,original_manifest_sha256=self.h.sha(original_manifest))
        selection_path=Path(temp)/"tiny_transport_selection.json"
        selection_path.write_text(json.dumps(selection),encoding="utf-8")
        private_output=Path(temp)/"private_transport_tiny"
        proof=transport.pack(repo,selection_path,self.h.sha(selection_path),private_output,wall_seconds=30)
        self.assertEqual(proof["status"],"PASS_ARCHIVE_BYTE_TRANSPORT")
        destination=campaign/self.h.TRANSPORT_DIRECTORY;destination.mkdir(parents=True)
        for name in ("MANIFEST.json","part_0000.transport.xz"):
            (destination/name).write_bytes((private_output/name).read_bytes())
        transported=destination/"part_0000.transport.xz"
        (campaign/"README.md").write_text("[transport]("+transported.relative_to(campaign).as_posix()+")\n",encoding="utf-8")
        selected=[n for n in names if n!=archive.relative_to(repo).as_posix()]
        selected.extend((destination/name).relative_to(repo).as_posix() for name in ("MANIFEST.json","part_0000.transport.xz"))
        return repo,campaign,selected,transported,original_manifest,archive

    def test_bare_default_reads_nothing_imports_no_science_and_starts_no_child(self):
        before={n:n in sys.modules for n in ("numpy","torch","psutil")}
        with mock.patch.object(self.h,"load_dependencies",side_effect=AssertionError("default cannot load sources")),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.h.main([]),0)
        value=json.loads(out.getvalue());self.assertEqual(value["input_reads"],0);self.assertEqual(value["files_written"],0)
        self.assertEqual(value["model_array_Task_or_child_operations"],0)
        self.assertEqual(before,{n:n in sys.modules for n in before})

    def test_exact_numeric_archive_and_complete_local_link_union_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,campaign,names,archive,_,_=self.fixture(temp)
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"PASS",result["failures"])
            self.assertEqual(result["file_count"],3)
            self.assertEqual(result["selected_bytes"],sum((repo/n).stat().st_size for n in names))
            self.assertEqual(result["numeric_archives"][0]["records"],1)
            self.assertFalse(result["legacy_full_union_PASS_claimed"])
            self.assertEqual(result["links"][0]["target"],archive.relative_to(repo).as_posix())

    def test_legacy_tar_xz_still_uses_the_frozen_original_stats_inspector(self):
        with tempfile.TemporaryDirectory() as temp:
            _,campaign,names,_,_,_=self.fixture(temp,legacy=True)
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"PASS",result["failures"])
            self.assertEqual(result["numeric_archives"],[])

    def test_third_transport_format_real_tiny_rebuild_and_selected_original_manifest_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,campaign,names,archive,original_manifest,original_archive=self.transport_fixture(temp)
            original_bytes=original_archive.read_bytes()
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"PASS",result["failures"])
            self.assertEqual(result["selected_bytes"],sum((repo/n).stat().st_size for n in names))
            self.assertEqual(result["numeric_archives"],[])
            record=result["transport_archives"][0]
            self.assertEqual(record["original_manifest"],original_manifest.relative_to(repo).as_posix())
            self.assertEqual(record["original_archive_sha256"],self.h.sha(original_archive))
            self.assertTrue(record["original_archive_tar_members_and_XZ_exactly_rebuilt"])
            self.assertTrue(record["original_archive_restore_required_before_original_stats_restore"])
            self.assertEqual(original_archive.read_bytes(),original_bytes)
            self.assertEqual(result["links"][0]["target"],archive.relative_to(repo).as_posix())

    def test_transport_missing_original_manifest_or_any_bound_byte_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,campaign,names,archive,original_manifest,_=self.transport_fixture(temp)
            selected=[n for n in names if n!=original_manifest.relative_to(repo).as_posix()]
            self.assertEqual(self.h.inspect(campaign,selected,tracked=set())["status"],"FAIL")
            original_bytes=original_manifest.read_bytes();original_manifest.write_bytes(original_bytes+b" ")
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertIn("TRANSPORT_ARCHIVE_INVALID",{r["rule"] for r in result["failures"]})
            original_manifest.write_bytes(original_bytes);archive.write_bytes(archive.read_bytes()+b"tamper")
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertIn("TRANSPORT_ARCHIVE_INVALID",{r["rule"] for r in result["failures"]})

    def test_numeric_missing_manifest_or_tampered_bytes_never_pass_via_virtual_link(self):
        with tempfile.TemporaryDirectory() as temp:
            _,campaign,names,archive,_,_=self.fixture(temp)
            no_manifest=[n for n in names if not n.endswith("MANIFEST.json")]
            self.assertEqual(self.h.inspect(campaign,no_manifest,tracked=set())["status"],"FAIL")
            archive.write_bytes(archive.read_bytes()+b"tampered")
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"FAIL")
            self.assertIn("NUMERIC_ARCHIVE_INVALID",{f["rule"] for f in result["failures"]})

    def test_private_strings_are_scanned_in_manifest_and_decoded_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            _,campaign,names,_,_,_=self.fixture(temp,run="/home/"+"synthetic/private")
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"FAIL")
            self.assertTrue(any(f["rule"]=="PRIVATE_ABSOLUTE_PATH" and "::" in f["file"] for f in result["failures"]))

    def test_combined_size_limit_includes_numeric_bytes_and_global_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,campaign,names,archive,pc,codec=self.fixture(temp)
            normal=sum((repo/n).stat().st_size for n in names if not n.endswith(".numeric.xz"))
            with mock.patch.object(pc,"MAX_BUNDLE_BYTES",normal+1),mock.patch.object(self.h,"load_dependencies",return_value=(pc,codec)):
                result=self.h.inspect(campaign,names,tracked=set())
            self.assertEqual(result["status"],"FAIL")
            self.assertEqual(result["selected_bytes"],normal+archive.stat().st_size)
            self.assertIn("BUNDLE_SIZE_LIMIT",{f["rule"] for f in result["failures"]})
            duplicated=self.h.inspect(campaign,names+[names[-1].upper()],tracked=set())
            self.assertIn("DUPLICATE_FILE",{f["rule"] for f in duplicated["failures"]})

    def test_wrong_namespace_per_file_limit_and_case_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _,campaign,names,archive,pc,codec=self.fixture(temp)
            with mock.patch.object(codec,"MAX_ARCHIVE",archive.stat().st_size-1),mock.patch.object(self.h,"load_dependencies",return_value=(pc,codec)):
                self.assertEqual(self.h.inspect(campaign,names,tracked=set())["status"],"FAIL")
            (campaign/"README.md").write_text("[case]("+archive.relative_to(campaign).as_posix().upper()+")\n",encoding="utf-8")
            result=self.h.inspect(campaign,names,tracked=set())
            self.assertIn("LINK_CASE_MISMATCH",{f["rule"] for f in result["failures"]})
            wrong=campaign/"wrong.numeric.xz";wrong.write_bytes(archive.read_bytes())
            result=self.h.inspect(campaign,[*names,wrong.relative_to(campaign.parents[1]).as_posix()],tracked=set())
            self.assertEqual(result["status"],"FAIL")

    def test_symlink_and_frozen_source_mismatch_are_rejected_before_inspection(self):
        fake=mock.Mock();fake.parents=[];fake.is_symlink.return_value=True
        with mock.patch.object(self.h.Path,"absolute",return_value=fake):
            with self.assertRaises(ValueError):self.h.plain("synthetic")
        with mock.patch.object(self.h,"sha",return_value="0"*64),mock.patch.object(self.h,"_load",side_effect=AssertionError("pin failure must precede imports")):
            with self.assertRaises(ValueError):self.h.load_dependencies()

    def test_png_svg_rules_and_explicit_index_scope_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,campaign,names,_,pc,codec=self.fixture(temp)
            png=campaign/"bad.png";png.write_bytes(b"not a PNG")
            svg=campaign/"bad.svg";svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>bad</script></svg>',encoding="utf-8")
            result=self.h.inspect(campaign,names+[p.relative_to(repo).as_posix() for p in (png,svg)],tracked=set())
            self.assertTrue({"PNG_SIGNATURE","SVG_ACTIVE_CONTENT"}<={f["rule"] for f in result["failures"]})
            with mock.patch.object(pc,"index_paths",return_value={names[0]}),mock.patch.object(pc,"index_blob",return_value=(repo/names[0]).read_bytes()),mock.patch.object(self.h,"load_dependencies",return_value=(pc,codec)):
                result=self.h.check_index(campaign,names)
            self.assertEqual(result["status"],"FAIL")
            self.assertIn("STAGED_SCOPE",{f["rule"] for f in result["failures"]})


if __name__=="__main__":unittest.main(verbosity=2)
