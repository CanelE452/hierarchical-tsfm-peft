import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stats_pack as module


class StatsPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stats_pack_test_")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.public = self.repo / "research" / "campaign"
        self.public.mkdir(parents=True)
        self.sources = []
        for index in range(2):
            path = self.public / "predictions" / f"run{index}" / "V_select" / "PUBLIC_ERROR_STATS.csv.gz"
            path.parent.mkdir(parents=True)
            rows = module.CSV_HEADER + f"run{index},V_select,toy,4,c,0,1,0.25,0.5,1\r\n".encode()
            buffer = io.BytesIO()
            with gzip.GzipFile(filename="PUBLIC_ERROR_STATS.csv", mode="wb", fileobj=buffer, mtime=123456789, compresslevel=9) as stream:
                stream.write(rows)
            path.write_bytes(buffer.getvalue()); self.sources.append(path)

    def make_pack(self, **kwargs):
        return module.pack(self.public, only=[path.relative_to(self.public).as_posix() for path in self.sources], **kwargs)

    def manifest(self, directory):
        path = directory / "MANIFEST.json"
        return path, json.loads(path.read_text())

    def test_original_gzip_and_csv_restore_exact_shared_members_and_payload(self):
        originals = {p: p.read_bytes() for p in self.sources}
        directory = self.make_pack()
        path, manifest = self.manifest(directory)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(len(manifest["records"]), 2)
        self.assertEqual(sum(len(a["members"]) for a in manifest["archives"]), 1)
        destination = self.repo / "restored"
        result = module.restore(path, destination)
        self.assertEqual(result["restored"], 2)
        for source, blob in originals.items():
            restored = destination / source.relative_to(self.repo)
            self.assertEqual(restored.read_bytes(), blob)
            self.assertEqual(gzip.decompress(restored.read_bytes()), gzip.decompress(blob))
            self.assertEqual(source.read_bytes(), blob)
        self.assertTrue(manifest["separate_temporary_restore_all_sha_verified"])
        payload = (directory / "PAYLOAD_FILES.txt").read_text().splitlines()
        self.assertNotIn(self.sources[0].relative_to(self.repo).as_posix(), payload)

    def test_identical_existing_is_read_only_and_mismatch_is_never_overwritten(self):
        directory = self.make_pack(); path, _ = self.manifest(directory)
        result = module.restore(path, self.repo)
        self.assertEqual(result["already_matching"], 2)
        self.assertEqual(result["restored"], 0)
        self.sources[0].write_bytes(b"unrelated existing content")
        with self.assertRaisesRegex(ValueError, "Existing"):
            module.restore(path, self.repo)
        self.assertEqual(self.sources[0].read_bytes(), b"unrelated existing content")

    def test_route_T_final_pack_inspect_restore_and_virtual_publication_are_exact(self):
        import publication_check
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                run_id = "route_opaque" if opaque else "route_common"
                source = self.public/"route_predictions"/run_id/"T_final"/"PUBLIC_ERROR_STATS.csv.gz"
                source.parent.mkdir(parents=True)
                raw = module.CSV_HEADER + f"{run_id},T_final,toy:primary:T_final:stride4,20,c,0,4,0.25,0.5,4\r\n".encode()
                blob = gzip.compress(raw, mtime=0)
                source.write_bytes(blob)
                relative = source.relative_to(self.repo).as_posix()
                (self.public/"README.md").write_text(f"[statistics]({source.relative_to(self.public).as_posix()}#stats)\n", encoding="utf-8")
                options = dict(only=[source.relative_to(self.public).as_posix()])
                if opaque:
                    with patch.object(module, "find_gzip_recipe", return_value=None):
                        directory = module.pack(self.public, **options)
                else:
                    directory = module.pack(self.public, **options)
                path, manifest = self.manifest(directory)
                self.assertEqual(manifest["status"], "PASS")
                self.assertEqual(len(manifest["records"]), 1)
                record = manifest["records"][0]
                self.assertEqual(record["repository_relative_path"], relative)
                self.assertEqual(record["storage_mode"], "opaque_gzip" if opaque else "csv_common9")
                self.assertEqual(record["csv_sha256"], module.sha(raw))
                self.assertEqual(record["gzip_sha256"], module.sha(blob))
                members = module.inspect_archive(directory/record["archive"])
                self.assertEqual(len(members), 1)
                self.assertIn("T_final", members[0]["csv_text"])
                self.assertEqual(source.read_bytes(), blob)
                source.unlink()  # Emulate a clone before restoring this temporary statistic.
                files = (directory/"PAYLOAD_FILES.txt").read_text(encoding="utf-8").splitlines()
                self.assertNotIn(relative, files)
                virtual = {item["repository_relative_path"] for item in manifest["records"]}
                inspected = publication_check.inspect(self.public, files, tracked=set(), virtual_files=virtual)
                self.assertEqual(inspected["status"], "PASS", inspected["failures"])
                destination = self.repo/("restored_opaque" if opaque else "restored_common")
                self.assertEqual(module.restore(path, destination)["restored"], 1)
                restored = destination/relative
                self.assertEqual(restored.read_bytes(), blob)
                self.assertEqual(gzip.decompress(restored.read_bytes()), raw)
                self.assertEqual(module.sha_file(restored), record["gzip_sha256"])
                self.assertEqual(module.sha(gzip.decompress(restored.read_bytes())), record["csv_sha256"])

    def test_legacy_E0_scope_and_seven_column_route_statistics_stay_rejected(self):
        raw = b"origin,channel,horizon_start,horizon_end,sse,sae,count\r\n20,c,0,4,0.25,0.5,4\r\n"
        for relative, message in (("legacy_e0/evaluation/toy/cp2800/PUBLIC_ERROR_STATS.csv.gz", "Only standard"),
                                  ("route_predictions/toy/T_final/PUBLIC_ERROR_STATS.csv.gz", "schema differs")):
            with self.subTest(relative=relative):
                source = self.public/relative; source.parent.mkdir(parents=True)
                source.write_bytes(gzip.compress(raw, mtime=0))
                with self.assertRaisesRegex(ValueError, message):
                    module.pack(self.public, only=[relative])
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        manifest["records"][0]["repository_relative_path"] = "research/campaign/legacy_e0/evaluation/toy/cp2800/PUBLIC_ERROR_STATS.csv.gz"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside public sufficient-statistic scope"):
            module.restore(path, self.repo/"rejected_legacy")

    def test_unreproducible_deflate_uses_opaque_gzip_member(self):
        with patch.object(module, "find_gzip_recipe", return_value=None):
            directory = self.make_pack()
        path, manifest = self.manifest(directory)
        self.assertTrue(all(r["storage_mode"] == "opaque_gzip" for r in manifest["records"]))
        self.assertEqual(module.restore(path, self.repo / "opaque_restore")["restored"], 2)

    def test_path_escape_symlink_and_non_stats_source_are_rejected(self):
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        manifest["records"][0]["repository_relative_path"] = "../escape.csv.gz"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            module.restore(path, self.repo / "no_escape")
        with self.assertRaises(ValueError):
            module.pack(self.public, only=["../../weights.pt"])

    def test_archive_hash_or_declared_size_tampering_is_rejected(self):
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        archive = directory / manifest["archives"][0]["file"]
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "archive|Archive"):
            module.restore(path, self.repo / "corrupt")

    def test_output_cap_and_bomb_limit_fail_before_write(self):
        directory = self.make_pack(); path, _ = self.manifest(directory)
        destination = self.repo / "too_small"
        with self.assertRaisesRegex(ValueError, "limit|budget"):
            module.restore(path, destination, max_output_bytes=1)
        self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, "limit"):
            module.bounded_xz_decompress(module.lzma.compress(b"0" * 10000), 100)

    def test_hidden_trailing_content_is_rejected_even_with_updated_archive_hash(self):
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        info = manifest["archives"][0]; archive = directory / info["file"]
        raw = module.lzma.decompress(archive.read_bytes()) + b"hidden non-statistic content"
        packed = module.lzma.compress(raw); archive.write_bytes(packed)
        info.update(bytes=len(packed), sha256=module.sha(packed), tar_bytes=len(raw))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            module.inspect_archive(archive)

    def test_records_cannot_be_silently_left_without_an_archive(self):
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        manifest["records"][0]["archive"] = "part_9999.tar.xz"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unbound"):
            module.restore(path, self.repo / "unbound")

    def test_publication_inspection_binds_original_csv_sha_before_restore(self):
        directory = self.make_pack(); path, manifest = self.manifest(directory)
        manifest["records"][0]["csv_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "CSV SHA"):
            module.inspect_archive(directory / manifest["archives"][0]["file"])

    def test_header_and_trailer_recipe_reconstruct_original_exactly(self):
        for source in self.sources:
            blob = source.read_bytes(); raw = gzip.decompress(blob)
            recipe = module.find_gzip_recipe(blob, raw)
            self.assertIsNotNone(recipe)
            self.assertEqual(module.regenerate_gzip(raw, recipe), blob)
            self.assertEqual(recipe["zlib_runtime_version"], module.zlib.ZLIB_RUNTIME_VERSION)


if __name__ == "__main__":
    unittest.main()
