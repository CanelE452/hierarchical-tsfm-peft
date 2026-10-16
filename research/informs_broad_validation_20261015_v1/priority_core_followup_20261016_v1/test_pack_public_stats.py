"""Synthetic public aggregate CSV/gzip fixtures; no campaign statistics decoded."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pack_public_stats.py"
PREFIX = "research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1"
IDS = ("priority_seed2_jena_B0_k6_s92602_v1", "priority_seed2_jena_B2_k6_s92602_v1",
       "priority_pems08_fixedB2_linear_k43_s92601_v1")


def load():
    spec = importlib.util.spec_from_file_location("tested_priority_stats_adapter", SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def records():
    result = []
    for rid in IDS:
        base = PREFIX + "/predictions/" + rid + "/V_select/"
        result.append(dict(run_id=rid, repository_relative_path=base + "PUBLIC_ERROR_STATS.csv.gz",
            gzip_bytes=1, gzip_sha256="0" * 64,
            prediction_manifest=dict(repository_relative_path=base + "PREDICTION_MANIFEST.json", bytes=1, sha256="0" * 64),
            score_csv=dict(repository_relative_path=base + "SCORES.csv", bytes=1, sha256="0" * 64)))
    return result


class PriorityPackTests(unittest.TestCase):
    def setUp(self):
        self.h = load()

    def fixture(self, folder):
        repo = Path(folder)
        codec = self.h.load_codec(HERE.parent)
        core = codec.load_core()
        selected = records()
        originals = {}
        for index, record in enumerate(selected):
            newline = b"\r\n" if index % 2 else b"\n"
            raw = newline.join([",".join(core.COLUMNS).encode(),
                (record["run_id"] + ",V_select,primary,512,ch0,0,2,0.10000000000000002,0.3,2").encode(),
                (record["run_id"] + ",V_select,primary,513,ch1,0,2,-0.0,1e-07,2").encode()]) + newline
            blob = gzip.compress(raw, compresslevel=9, mtime=0)
            path = repo / record["repository_relative_path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(blob)
            record.update(gzip_bytes=len(blob), gzip_sha256=self.h.sha(path))
            originals[record["repository_relative_path"]] = (blob, raw)
        return repo, codec, selected, originals

    def test_exact_three_rejects_foreign_T_duplicate_and_parent_full_cpu(self):
        self.assertEqual(self.h.validate_records(records(), PREFIX), records())
        variants = []
        for text in ("T_final", "foreign", "parent_full_cpu"):
            value = records()
            value[0]["repository_relative_path"] = value[0]["repository_relative_path"].replace("V_select" if text == "T_final" else "predictions", text)
            variants.append(value)
        variants.extend([records()[:2], [records()[0], records()[0], records()[2]]])
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.h.validate_records(value, PREFIX)

    def test_default_reads_writes_and_science_imports_zero(self):
        before = set(sys.modules)
        with patch("builtins.open", side_effect=AssertionError("Default must not open inputs")), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.h.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["input_reads"], 0)
        self.assertEqual(value["files_written"], 0)
        self.assertFalse({"numpy", "torch"} & (set(sys.modules) - before))

    def test_draft_or_another_runner_cannot_execute(self):
        with self.assertRaises(ValueError):
            self.h.require_sealed(dict(status="DRAFT_NOT_EXECUTABLE", runner_sha256=None))
        with self.assertRaises(ValueError):
            self.h.require_sealed(dict(status="SEALED_BY_ROOT", runner_sha256="0" * 64))
        self.h.require_sealed(dict(status="SEALED_BY_ROOT", runner_sha256=self.h.sha(SOURCE)))

    def test_frozen_reader_accepts_P2_prefix_and_exact_three_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            repo, codec, selected, originals = self.fixture(folder)
            output = repo / "stats_packs" / "pack_fixture"
            manifest = self.h.write_payloads(codec, repo, selected, PREFIX, output, time.time() + 30, lambda: None)
            self.assertEqual(codec.load_manifest(output / "MANIFEST.json")["campaign_repository_relative"], PREFIX)
            destination = repo / "restored"
            result = self.h.verify_restoration(codec, output / "MANIFEST.json", destination, manifest, time.time() + 30, lambda: None)
            self.assertEqual(result["first"]["restored"], 3)
            self.assertEqual(result["repeat"]["already_matching"], 3)
            for name, (blob, raw) in originals.items():
                self.assertEqual((destination / name).read_bytes(), blob)
                self.assertEqual(gzip.decompress((destination / name).read_bytes()), raw)
                self.assertEqual((repo / name).read_bytes(), blob)

    def test_existing_output_and_different_restored_file_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            repo, codec, selected, _ = self.fixture(folder)
            output = repo / "stats_packs" / "pack_fixture"
            manifest = self.h.write_payloads(codec, repo, selected, PREFIX, output, time.time() + 30, lambda: None)
            with self.assertRaises(FileExistsError):
                self.h.write_payloads(codec, repo, selected, PREFIX, output, time.time() + 30, lambda: None)
            destination = repo / "restored"
            target = destination / selected[0]["repository_relative_path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                self.h.verify_restoration(codec, output / "MANIFEST.json", destination, manifest, time.time() + 30, lambda: None)
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_tampered_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            repo, codec, selected, _ = self.fixture(folder)
            output = repo / "stats_packs" / "pack_fixture"
            manifest = self.h.write_payloads(codec, repo, selected, PREFIX, output, time.time() + 30, lambda: None)
            archive = output / manifest["archives"][0]["file"]
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                codec.restore(output / "MANIFEST.json", repo / "restored")

    def test_source_and_sidecar_byte_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            path.write_bytes(b"original")
            record = dict(repository_relative_path="metadata.json", bytes=8, sha256=self.h.sha(path))
            self.h.bound(Path(folder), record)
            path.write_bytes(b"changed!")
            with self.assertRaises(ValueError):
                self.h.bound(Path(folder), record)
            with self.assertRaises(ValueError):
                self.h.check_pins(Path(folder), {"metadata.json": record["sha256"]})
        row = records()[0]
        manifest = dict(run_id=row["run_id"], split="V_select", cohort="jena:primary:V_select:stride48", device="cpu",
            arrays=dict(target_normalized=dict(shape=[43, 48, 21])), score=dict(mse=0.1, n_origins=43))
        self.h.validate_v_metadata(manifest, [dict(mse="0.1", n_origins="43")], row)
        with self.assertRaises(ValueError):
            self.h.validate_v_metadata(dict(manifest, public_stats_relative_path="foreign"), [dict(mse="0.1", n_origins="43")], row)

    def test_wrong_codec_pin_and_expired_deadline_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "stats_pack_numeric.py").write_bytes(b"different")
            with self.assertRaises(ValueError):
                self.h.load_codec(Path(folder))
        with self.assertRaises(TimeoutError):
            self.h.deadline_check(time.time() - 1)


if __name__ == "__main__":
    unittest.main()
