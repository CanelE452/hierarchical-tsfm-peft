"""Synthetic stdlib-only archive transport fixtures; no experiment inputs."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
import lzma
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load_helper():
    spec = importlib.util.spec_from_file_location("tested_stats_pack_transport", HERE / "stats_pack_transport.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.h = load_helper()
        self.core, self.numeric = self.h.load_dependencies()

    def fixture(self, folder):
        root = Path(folder)
        repo = root / "repo"
        campaign = repo / "research" / "synthetic"
        directory = campaign / "stats_packs" / "pack_fixture"
        directory.mkdir(parents=True)
        common = b"split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count\r\nV_select,primary,512,ch0,0,2,0.10000000000000002,0.3,2\r\n"
        quoted = common.replace(b"ch0", b'"ch,0"')
        opaque_raw = b"run_id," + quoted.splitlines(keepends=True)[0] + b"opaque," + quoted.splitlines(keepends=True)[1]
        opaque = gzip.compress(opaque_raw, mtime=0)
        members = []
        records = []
        for index, (payload, mode) in enumerate(((common, "csv_common9"), (quoted, "csv_common9"), (opaque, "opaque_gzip"))):
            suffix = ".csv" if mode == "csv_common9" else ".csv.gz"
            member = "members/" + self.core.sha(payload) + suffix
            members.append((member, payload))
            prefix = ("run%d," % index).encode()
            raw = b"run_id," + payload.splitlines(keepends=True)[0] + prefix + payload.splitlines(keepends=True)[1] if mode == "csv_common9" else opaque_raw
            blob = gzip.compress(raw, mtime=0) if mode == "csv_common9" else opaque
            records.append(dict(repository_relative_path="research/synthetic/predictions/run%d/V_select/PUBLIC_ERROR_STATS.csv.gz" % index,
                                member=member, storage_mode=mode, archive="part_0000.tar.xz", row_count=1,
                                run_id_prefix=prefix.decode() if mode == "csv_common9" else None,
                                gzip_bytes=len(blob), gzip_sha256=self.core.sha(blob), csv_bytes=len(raw), csv_sha256=self.core.sha(raw)))
        archives = []
        self.core._emit(members, directory, archives)
        original_manifest = dict(format=self.core.SCHEMA, status="PASS", source_sha256=self.h.CORE_SHA,
                                 separate_temporary_restore_all_sha_verified=True, campaign_repository_relative="research/synthetic",
                                 records=records, archives=archives)
        manifest = directory / "MANIFEST.json"
        manifest.write_text(json.dumps(original_manifest), encoding="utf-8")
        archive = directory / "part_0000.tar.xz"
        selection = root / "selection.json"
        selection.write_text(json.dumps(dict(format=self.h.SELECTION_SCHEMA,
            original_archive_repository_relative_path=archive.relative_to(repo).as_posix(),
            original_archive_bytes=archive.stat().st_size, original_archive_sha256=self.core.sha_file(archive),
            original_manifest_repository_relative_path=manifest.relative_to(repo).as_posix(),
            original_manifest_bytes=manifest.stat().st_size, original_manifest_sha256=self.core.sha_file(manifest))), encoding="utf-8")
        return repo, selection, archive, manifest

    def pack_fixture(self, temp):
        repo, selection, archive, original_manifest = self.fixture(temp)
        output = Path(temp) / "transport"
        before = (archive.read_bytes(), original_manifest.read_bytes())
        proof = self.h.pack(repo, selection, self.core.sha_file(selection), output, wall_seconds=60)
        self.assertEqual(before, (archive.read_bytes(), original_manifest.read_bytes()))
        self.assertEqual(proof["status"], "PASS_ARCHIVE_BYTE_TRANSPORT")
        return repo, output / "MANIFEST.json", archive, original_manifest

    def test_exact_original_archive_restores_and_repeat_skips_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, manifest, archive, _ = self.pack_fixture(temp)
            destination = Path(temp) / "clone"
            first = self.h.restore(manifest, destination)
            target = destination / archive.relative_to(repo)
            self.assertEqual(first["restored"], 1)
            self.assertEqual(target.read_bytes(), archive.read_bytes())
            stamp = target.stat().st_mtime_ns
            repeated = self.h.restore(manifest, destination)
            self.assertEqual(repeated["already_matching"], 1)
            self.assertEqual(target.stat().st_mtime_ns, stamp)
            target.write_bytes(b"existing mismatch")
            with self.assertRaises(ValueError):
                self.h.restore(manifest, destination)
            self.assertEqual(target.read_bytes(), b"existing mismatch")

    def test_numeric_raw_common9_and_opaque_members_remain_distinct_and_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            _, manifest, _, _ = self.pack_fixture(temp)
            value = json.loads(manifest.read_text())
            self.assertEqual([v["transport_mode"] for v in value["members"]], ["numeric_common9", "raw_common9", "opaque_gzip"])
            inspected = self.h.inspect_archive(manifest.parent / value["archive"]["file"])
            self.assertEqual(len(inspected), 3)
            self.assertEqual([v["row_count"] for v in inspected], [1, 1, 1])
            self.assertIn("0.10000000000000002", inspected[0]["csv_text"])
            self.assertFalse(value["original_publication_gate_pass_claimed"])

    def test_explicit_selection_metadata_does_not_decompress_statistic_members(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, selection, archive, _ = self.fixture(temp)
            with patch.object(self.h, "load_dependencies", return_value=(self.core, self.numeric)), \
                 patch.object(self.core, "_archive_members", side_effect=AssertionError("decompressed during metadata")) as decoder:
                self.h.metadata(repo, selection, self.core.sha_file(selection))
                decoder.assert_not_called()
                with self.assertRaises(ValueError):
                    self.h.metadata(repo, selection, "0" * 64)
            self.assertTrue(archive.is_file())

    def test_historical_writer_pin_is_distinct_from_compatible_reader_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, selection, _, manifest = self.fixture(temp)
            old = json.loads(manifest.read_text())
            old["source_sha256"] = self.h.LEGACY_PRODUCER_SHA
            manifest.write_text(json.dumps(old), encoding="utf-8")
            chosen = json.loads(selection.read_text())
            chosen.update(original_manifest_bytes=manifest.stat().st_size, original_manifest_sha256=self.core.sha_file(manifest))
            selection.write_text(json.dumps(chosen), encoding="utf-8")
            output = Path(temp) / "historical_writer"
            self.h.pack(repo, selection, self.core.sha_file(selection), output, wall_seconds=60)
            value = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(value["original_writer_source_sha256"], self.h.LEGACY_PRODUCER_SHA)
            self.assertEqual(value["source_pins"]["core"], self.h.CORE_SHA)
            old["source_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self.h._original_manifest(json.dumps(old).encode(), dict(value["binding"], original_manifest_bytes=len(json.dumps(old).encode()),
                                                                       original_manifest_sha256=self.core.sha(json.dumps(old).encode())))

    def test_manifest_path_source_pin_order_and_size_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, manifest, _, _ = self.pack_fixture(temp)
            original = json.loads(manifest.read_text())
            mutations = [lambda v: v["source_pins"].update(numeric="0" * 64),
                         lambda v: v["binding"].update(original_archive_repository_relative_path="../part_0000.tar.xz"),
                         lambda v: v["binding"].update(original_archive_bytes=True),
                         lambda v: v["members"].reverse(),
                         lambda v: v["members"][0].update(original_bytes=self.h.MAX_MEMBER + 1),
                         lambda v: v["archive"].update(tar_bytes=self.h.MAX_TAR + 1),
                         lambda v: v.update(original_tar_sha256="0" * 64),
                         lambda v: v["binding"].update(original_archive_sha256="0" * 64)]
            for mutate in mutations:
                value = copy.deepcopy(original)
                mutate(value)
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.h.restore(manifest, Path(temp) / "rejected")
                self.assertFalse((Path(temp) / "rejected").exists())

    def test_expansion_and_output_budgets_reject_before_decode_or_tar_join(self):
        with tempfile.TemporaryDirectory() as temp:
            _, manifest, archive, _ = self.pack_fixture(temp)
            with patch.object(self.h, "load_dependencies", return_value=(self.core, self.numeric)), \
                 patch.object(self.numeric, "decode_common9", side_effect=AssertionError("decode before budget")) as decode:
                with self.assertRaises(ValueError):
                    self.h.restore(manifest, Path(temp) / "small", max_output_bytes=archive.stat().st_size - 1)
                decode.assert_not_called()
                value = json.loads(manifest.read_text())
                value["members"][0]["original_bytes"] = self.h.MAX_MEMBER
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with patch.object(self.h, "MAX_TAR", 10240), patch.object(self.core, "_tar_bytes", side_effect=AssertionError("join before bound")) as join:
                    with self.assertRaises(ValueError):
                        self.h.restore(manifest, Path(temp) / "huge")
                    join.assert_not_called()
                    decode.assert_not_called()

    def test_transport_byte_tamper_and_deadline_never_write_original_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            _, manifest, _, _ = self.pack_fixture(temp)
            with self.assertRaises(TimeoutError):
                self.h.restore(manifest, Path(temp) / "expired", deadline=0)
            self.assertFalse((Path(temp) / "expired").exists())
            value = json.loads(manifest.read_text())
            archive = manifest.parent / value["archive"]["file"]
            archive.write_bytes(archive.read_bytes() + b"trailing")
            with self.assertRaises(ValueError):
                self.h.restore(manifest, Path(temp) / "tampered")
            self.assertFalse((Path(temp) / "tampered").exists())

    def test_noncanonical_tar_padding_and_concatenated_xz_fail_after_hash_rebinding(self):
        for mutation in ("padding", "concatenated"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                _, manifest, _, _ = self.pack_fixture(temp)
                value = json.loads(manifest.read_text())
                archive = manifest.parent / value["archive"]["file"]
                packed = archive.read_bytes()
                if mutation == "padding":
                    raw = bytearray(self.core.bounded_xz_decompress(packed))
                    raw[-1] = 1
                    packed = lzma.compress(bytes(raw), format=lzma.FORMAT_XZ, preset=6)
                    value["archive"]["tar_sha256"] = self.core.sha(raw)
                else:
                    packed += lzma.compress(b"another stream")
                archive.write_bytes(packed)
                value["archive"].update(bytes=len(packed), sha256=self.core.sha(packed))
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.h.restore(manifest, Path(temp) / "invalid")
                self.assertFalse((Path(temp) / "invalid").exists())

    def test_numeric_declared_expansion_sha_and_rows_bind_before_decode(self):
        common = b"split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count\r\nV_select,primary,512,ch0,0,2,0.1,0.3,2\r\n"
        payload = self.numeric.encode_common9(common)
        for size, digest, rows in ((len(common) - 1, self.core.sha(common), 1),
                                   (len(common), "0" * 64, 1), (len(common), self.core.sha(common), 2)):
            with self.subTest(size=size, digest=digest, rows=rows), \
                 patch.object(self.numeric, "decode_common9", side_effect=AssertionError("decode before declared binding")) as decode:
                with self.assertRaises(ValueError):
                    self.h._decode_numeric(self.numeric, payload, size, digest, rows)
                decode.assert_not_called()

    def test_final_source_recheck_precedes_published_pass_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, selection, archive, _ = self.fixture(temp)
            output = Path(temp) / "late_changed"
            original = self.h._write_json
            def changed(path, value):
                original(path, value)
                if Path(path).name == "PROOF.json":
                    archive.write_bytes(archive.read_bytes() + b"late source change")
            with patch.object(self.h, "_write_json", side_effect=changed):
                with self.assertRaises(ValueError):
                    self.h.pack(repo, selection, self.core.sha_file(selection), output, wall_seconds=60)
            self.assertFalse((output / "MANIFEST.json").exists())

    def test_source_change_during_encode_cannot_publish_pass_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, selection, archive, _ = self.fixture(temp)
            original = self.numeric.encode_common9
            def changed(raw, *, deadline=None):
                payload = original(raw, deadline=deadline)
                archive.write_bytes(archive.read_bytes() + b"changed")
                return payload
            output = Path(temp) / "changed_output"
            with patch.object(self.h, "load_dependencies", return_value=(self.core, self.numeric)), \
                 patch.object(self.numeric, "encode_common9", side_effect=changed):
                with self.assertRaises(ValueError):
                    self.h.pack(repo, selection, self.core.sha_file(selection), output, wall_seconds=60)
            self.assertFalse((output / "MANIFEST.json").exists())

    def test_bare_metadata_has_no_statistics_science_or_writes(self):
        before = {name: name in sys.modules for name in ("numpy", "torch")}
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]), 0)
        value = json.loads(stream.getvalue())
        self.assertEqual(value["files_written"], 0)
        self.assertEqual(value["compressions"], 0)
        self.assertFalse(value["statistics_content_read"])
        self.assertFalse(value["legacy_publication_inspector_compatible"])
        self.assertEqual(before, {name: name in sys.modules for name in before})


if __name__ == "__main__":
    unittest.main(verbosity=2)
