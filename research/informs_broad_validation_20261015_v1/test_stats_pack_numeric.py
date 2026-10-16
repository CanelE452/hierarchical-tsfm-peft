"""Tiny synthetic CSV/gzip fixtures; no experiment arrays, models, or source edits."""
import copy
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load_helper():
    spec = importlib.util.spec_from_file_location("tested_stats_pack_numeric", HERE / "stats_pack_numeric.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class NumericPackTests(unittest.TestCase):
    def setUp(self):
        self.h = load_helper()
        self.core = self.h.load_core()

    def common(self, rows, newline=b"\r\n"):
        return newline.join([b",".join(c.encode("ascii") for c in self.core.COLUMNS[1:]), *rows]) + newline

    def raw(self, common, run=b"synthetic_run"):
        lines = common.splitlines(keepends=True)
        return b"run_id," + lines[0] + b"".join(run + b"," + row for row in lines[1:])

    def fixture(self, folder, raw=None):
        raw = raw or self.raw(self.common([b"V_select,primary,512,ch0,0,2,0.10000000000000002,0.3,2"]))
        blob = gzip.compress(raw, compresslevel=9, mtime=0)
        record, payload = self.h.encode_blob(blob)
        directory = Path(folder) / "stats_packs" / "pack_synthetic"
        directory.mkdir(parents=True)
        suffix = ".numeric" if record["storage_mode"] == "numeric_common9" else ".csv.gz"
        member = "members/" + self.core.sha(payload) + suffix
        archives = []
        assignment = self.h._emit([(member, payload)], directory, archives)
        record.update(repository_relative_path="research/campaign/postT_fixture/predictions/synthetic_run/V_select/PUBLIC_ERROR_STATS.csv.gz",
                      member=member, archive=assignment[member])
        manifest = dict(format=self.h.SCHEMA, status="PASS", original_stats_pack_sha256=self.h.CORE_SHA,
                        campaign_repository_relative="research/campaign/postT_fixture", records=[record], archives=archives)
        path = directory / "MANIFEST.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest, blob, payload

    def test_canonical_float64_and_int64_boundary_tokens_restore_exactly(self):
        values = [0.0, -0.0, 0.1, 0.10000000000000002, 1.0000000000000002,
                  5e-324, 2.2250738585072014e-308, 1e+20, 1e-07, 1.7976931348623157e+308]
        rows = [("V_select,primary,%s,ch0,0,2,%s,0.0,2" % (2**63-1, str(v))).encode("ascii") for v in values]
        raw = self.common(rows)
        encoded = self.h.encode_common9(raw)
        self.assertIsInstance(encoded, bytes)
        decoded = self.h.decode_common9(encoded)
        self.assertEqual(decoded, raw)
        for expected, row in zip(values, decoded.splitlines()[1:]):
            self.assertEqual(struct.pack("<d", float(row.split(b",")[6])), struct.pack("<d", expected))

    def test_noncanonical_utf8_literal_tokens_and_int64_overflow_are_lossless(self):
        rows = [b"V_select,primary,000512,ch0,00,02,1.00,1e+00,02",
                b"V_select,primary,9223372036854775808,ch0,0,2,-0,1E+20,2",
                "V_select,primary,513,온도,0,2,0.000,2.5000,2".encode("utf-8")]
        raw = self.common(rows)
        payload = self.h.encode_common9(raw)
        self.assertIsInstance(payload, bytes)
        self.assertEqual(self.h.decode_common9(payload), raw)
        for token in (b"000512", b"1.00", b"1e+00", b"1E+20", b"-0", b"9223372036854775808"):
            self.assertEqual(self.h._number_token(token)[0], 0)
        self.assertEqual(self.h._number_token(b"9223372036854775807"), (1, struct.pack("<q", 2**63-1)))
        self.assertEqual(self.h._number_token(b"-9223372036854775808"), (1, struct.pack("<q", -(2**63))))
        self.assertEqual(self.h._number_token(b"-0.0"), (2, struct.pack("<d", -0.0)))

    def test_lf_crlf_header_and_row_order_are_not_normalized(self):
        rows = [b"V_select,primary,513,chB,0,2,0.5,0.7,2", b"V_select,primary,512,chA,0,2,0.1,0.3,2"]
        for newline in (b"\n", b"\r\n"):
            raw = self.common(rows, newline)
            self.assertEqual(self.h.decode_common9(self.h.encode_common9(raw)), raw)
        bad = self.common(rows).replace(b"split,cohort", b"cohort,split", 1)
        with self.assertRaises(ValueError):
            self.h.encode_common9(bad)

    def test_quoted_or_embedded_newline_csv_uses_opaque_original_gzip(self):
        for channel in (b'"ch,0"', b'"ch\n0"'):
            raw = (b",".join(c.encode("ascii") for c in self.core.COLUMNS) + b"\r\n" +
                   b"synthetic_run,V_select,primary,512," + channel + b",0,2,0.1,0.3,2\r\n")
            blob = gzip.compress(raw, compresslevel=9, mtime=0)
            record, payload = self.h.encode_blob(blob)
            self.assertEqual(record["storage_mode"], "opaque_gzip")
            self.assertEqual(payload, blob)
            self.assertEqual(self.h.restore_blob(record, payload), blob)

    def test_original_gzip_recipe_and_declared_csv_and_gzip_hashes_are_enforced(self):
        raw = self.raw(self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2"]))
        blob = gzip.compress(raw, compresslevel=9, mtime=1234)
        record, payload = self.h.encode_blob(blob)
        self.assertEqual(record["storage_mode"], "numeric_common9")
        self.assertEqual(self.h.restore_blob(record, payload), blob)
        for key in ("csv_sha256", "gzip_sha256"):
            changed = copy.deepcopy(record)
            changed[key] = "0" * 64
            with self.assertRaises(ValueError):
                self.h.restore_blob(changed, payload)
        with self.assertRaises((ValueError, EOFError, OSError)):
            self.h.restore_blob(record, payload[:-1] + bytes([payload[-1] ^ 1]))

    def test_temporary_restore_repeat_skip_and_mismatch_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, value, blob, _ = self.fixture(temp)
            destination = Path(temp) / "clone"
            result = self.h.restore(manifest, destination)
            self.assertEqual(result["restored"], 1)
            target = destination / value["records"][0]["repository_relative_path"]
            self.assertEqual(target.read_bytes(), blob)
            before = target.stat().st_mtime_ns
            result = self.h.restore(manifest, destination)
            self.assertEqual(result["restored"], 0)
            self.assertEqual(result["already_matching"], 1)
            self.assertEqual(target.stat().st_mtime_ns, before)
            target.write_bytes(b"existing mismatch")
            with self.assertRaises(ValueError):
                self.h.restore(manifest, destination)
            self.assertEqual(target.read_bytes(), b"existing mismatch")

    def test_manifest_path_source_pin_duplicate_and_row_count_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, original, _, _ = self.fixture(temp)
            for mutate in (lambda x: x.update(original_stats_pack_sha256="0"*64),
                           lambda x: x["records"][0].update(repository_relative_path="../PUBLIC_ERROR_STATS.csv.gz"),
                           lambda x: x["records"].append(copy.deepcopy(x["records"][0])),
                           lambda x: x["records"][0].update(row_count=2)):
                value = copy.deepcopy(original)
                mutate(value)
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.h.restore(manifest, Path(temp)/"clone")

    def test_bounded_decode_and_output_budget_prevent_expansion(self):
        raw = self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2"])
        payload = self.h.encode_common9(raw)
        with patch.object(self.h, "MAX_MEMBER", 16):
            with self.assertRaises(ValueError):
                self.h.decode_common9(payload)
        with tempfile.TemporaryDirectory() as temp:
            manifest, _, blob, _ = self.fixture(temp)
            with self.assertRaises(ValueError):
                self.h.restore(manifest, Path(temp)/"clone", max_output_bytes=len(blob)-1)
            self.assertFalse((Path(temp)/"clone").exists())

    def test_numeric_prefix_and_declared_sizes_reject_before_original_csv_join(self):
        raw = self.raw(self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2",
                                    b"V_select,primary,513,ch0,0,2,0.2,0.4,2"]))
        record, payload = self.h.encode_blob(gzip.compress(raw, compresslevel=9, mtime=0))
        self.assertEqual(record["storage_mode"], "numeric_common9")
        mutations = [("run_id_prefix", "r"*32768+","), ("run_id_prefix", "온"*22000+","),
                     ("run_id_prefix", None), ("run_id_prefix", []), ("run_id_prefix", ","),
                     ("run_id_prefix", "bad\n,"), ("run_id_prefix", 'bad",'), ("run_id_prefix", "bad,,"), ("run_id_prefix", "\ud800,"),
                     ("csv_bytes", len(raw)+1), ("csv_bytes", 0), ("csv_bytes", True), ("csv_bytes", 65537),
                     ("gzip_bytes", 0), ("gzip_bytes", True), ("gzip_bytes", self.h.MAX_ARCHIVE+1),
                     ("row_count", 1), ("row_count", 0), ("row_count", True), ("row_count", self.h.MAX_ROWS+1)]
        for key, value in mutations:
            with self.subTest(key=key, value=str(value)[:32]), \
                 patch.object(self.h, "load_core", return_value=self.core), \
                 patch.object(self.h, "MAX_MEMBER", 65536), \
                 patch.object(self.core, "_original_csv", side_effect=AssertionError("Unsafe original CSV join entered")) as join:
                changed = dict(record, **{key:value})
                with self.assertRaises(ValueError):
                    self.h.restore_blob(changed, payload)
                join.assert_not_called()

    def test_prefix_row_product_cannot_exceed_declared_member_budget(self):
        raw = self.raw(self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2",
                                    b"V_select,primary,513,ch0,0,2,0.2,0.4,2"]))
        record, payload = self.h.encode_blob(gzip.compress(raw, compresslevel=9, mtime=0))
        changed = dict(record, run_id_prefix="r"*32768+",", csv_bytes=65536)
        with patch.object(self.h, "load_core", return_value=self.core), \
             patch.object(self.h, "MAX_MEMBER", 65536), \
             patch.object(self.core, "_original_csv", side_effect=AssertionError("Oversized prefix-row product reached original join")) as join:
            with self.assertRaises(ValueError):
                self.h.restore_blob(changed, payload)
            join.assert_not_called()

    def test_utf8_run_prefix_and_unterminated_last_row_restore_exactly(self):
        common = self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2"]).removesuffix(b"\r\n")
        raw = self.raw(common, run="온도_run".encode("utf-8"))
        blob = gzip.compress(raw, compresslevel=9, mtime=0)
        record, payload = self.h.encode_blob(blob)
        self.assertEqual(record["storage_mode"], "numeric_common9")
        self.assertEqual(self.h.restore_blob(record, payload), blob)

    def test_restore_output_budget_rejects_before_original_csv_join(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _, blob, _ = self.fixture(temp)
            with patch.object(self.h, "load_core", return_value=self.core), \
                 patch.object(self.core, "_original_csv", side_effect=AssertionError("Restore budget reached original CSV join")) as join:
                with self.assertRaises(ValueError):
                    self.h.restore(manifest, Path(temp)/"clone", max_output_bytes=len(blob)-1)
                join.assert_not_called()
            self.assertFalse((Path(temp)/"clone").exists())

    def test_archive_byte_tamper_and_expired_deadline_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, value, _, _ = self.fixture(temp)
            original = manifest.read_bytes()
            with self.assertRaises(TimeoutError):
                self.h.restore(manifest, Path(temp)/"expired", deadline=0)
            self.assertEqual(manifest.read_bytes(), original)
            archive = manifest.parent / value["archives"][0]["file"]
            archive.write_bytes(archive.read_bytes() + b"mutated")
            with self.assertRaises(ValueError):
                self.h.restore(manifest, Path(temp)/"clone")
            self.assertFalse((Path(temp)/"clone").exists())

    def test_pack_controller_verifies_all_bytes_and_rejects_changed_source(self):
        for mutate in (False,True):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temp:
                repo=Path(temp);campaign=repo/"research"/"campaign";cache=repo/"cache"
                cache.mkdir();campaign.mkdir(parents=True)
                name="research/campaign/postT_fixture/predictions/synthetic_run/V_select/PUBLIC_ERROR_STATS.csv.gz"
                source=repo/name;source.parent.mkdir(parents=True)
                raw=self.raw(self.common([b"V_select,primary,512,ch0,0,2,0.1,0.3,2"]))
                blob=gzip.compress(raw,compresslevel=9,mtime=0);source.write_bytes(blob)
                selected=dict(repository_relative_path=name,gzip_bytes=len(blob),gzip_sha256=self.core.sha(blob))
                state=dict(repo=repo,public=campaign,cache=cache,core=self.core,records=[selected],prefix="research/campaign/postT_fixture",
                    pins={},selection_sha256="a"*64,wrapper=types.SimpleNamespace(storage_admission=lambda state,pending:0))
                selection=repo/"selection.json";selection.write_text("{}",encoding="utf-8")
                output=campaign/"stats_packs"/("pack_numeric_mutated" if mutate else "pack_numeric_exact")
                original=self.h.encode_blob
                def encode(value,*,deadline=None):
                    result=original(value,deadline=deadline)
                    if mutate:source.write_bytes(blob+b"source mutation")
                    return result
                with patch.object(self.h,"metadata",return_value=state),patch.object(self.h,"encode_blob",side_effect=encode):
                    if mutate:
                        with self.assertRaises(ValueError):
                            self.h.pack(repo,campaign,cache,"fixed_pca",selection,"a"*64,output,wall_seconds=60)
                        self.assertFalse((output/"MANIFEST.json").exists())
                        self.assertEqual(source.read_bytes(),blob+b"source mutation")
                    else:
                        proof=self.h.pack(repo,campaign,cache,"fixed_pca",selection,"a"*64,output,wall_seconds=60)
                        self.assertEqual(proof["status"],"PASS")
                        self.assertEqual(proof["separate_restore"]["restored"],1)
                        self.assertEqual(proof["repeated_restore"]["already_matching"],1)
                        self.assertEqual(source.read_bytes(),blob)
                        self.assertEqual(len(self.h.inspect_archive(output/"part_0000.numeric.xz")),1)

    def test_bare_metadata_has_no_statistics_science_or_writes(self):
        before={name:name in sys.modules for name in ("numpy","torch")}
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["files_written"],0)
        self.assertFalse(result["statistics_content_read"])
        self.assertEqual(result["compressions"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})
        self.assertEqual(result["numpy_imported"],before["numpy"])
        self.assertEqual(result["torch_imported"],before["torch"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
