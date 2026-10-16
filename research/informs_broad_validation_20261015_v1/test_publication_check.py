"""Temporary-file publication checks; Git index interaction is mocked read-only."""
import base64
import gzip
import io
import json
import lzma
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=")


class PublicationCheckTests(unittest.TestCase):
    def setUp(self):
        import publication_check
        self.m = publication_check
        self.tmp = tempfile.TemporaryDirectory(prefix="informs_publication_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.campaign = self.root / "research" / "campaign"; self.campaign.mkdir(parents=True)
        (self.campaign / "README.md").write_text("![image](plot.png)\n[old](../../old.md)\n")
        (self.campaign / "plot.png").write_bytes(PNG)
        (self.root / "old.md").write_text("Tracked old artifact\n")
        self.files = ["research/campaign/README.md", "research/campaign/plot.png"]

    def check(self, files=None):
        return self.m.inspect(self.campaign, files or self.files, tracked={"old.md"})

    def test_valid_assets_prepare_verify_and_no_self_hash_cycle(self):
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        with patch.object(self.m, "tracked_files", return_value={"old.md"}):
            prepared = self.m.prepare(self.campaign, self.files)
            self.assertEqual(prepared["status"], "PASS")
            verified = self.m.verify(self.campaign)
            self.assertEqual(verified["status"], "PASS")
            allowlist = (self.campaign / "PUBLIC_ALLOWLIST.txt").read_text().splitlines()
            self.assertIn("research/campaign/PUBLIC_ALLOWLIST.txt", allowlist)
            self.assertIn("research/campaign/PUBLICATION_CHECK.json", allowlist)
            self.assertIn("research/campaign/PUBLIC_MANIFEST.sha256", allowlist)
            manifest = (self.campaign / "PUBLIC_MANIFEST.sha256").read_text()
            self.assertNotIn("  research/campaign/PUBLIC_MANIFEST.sha256", manifest)
            with self.assertRaises(FileExistsError): self.m.prepare(self.campaign, self.files)

    def test_escape_symlink_encoded_escape_and_excluded_local_target(self):
        doc = self.campaign / "README.md"
        doc.write_text("[bad](../../../../escape.md)\n")
        self.assertEqual(self.check()["status"], "FAIL")
        doc.write_text("[bad](%2e%2e/%2e%2e/%2e%2e/escape.md)\n")
        self.assertEqual(self.check()["status"], "FAIL")
        (self.campaign / "plan").mkdir()
        (self.campaign / "plan" / "MASTER_PLAN.md").write_text("Private original retained")
        doc.write_text("[local excluded](plan/MASTER_PLAN.md)\n")
        self.assertTrue(any(r["rule"] == "LINK_NOT_PUBLISHED" for r in self.check()["failures"]))

    def test_bad_image_oversize_and_realistic_secret_fail_without_exposing_value(self):
        (self.campaign / "plot.png").write_bytes(b"wrong signature")
        self.assertTrue(any(r["rule"] == "PNG_SIGNATURE" for r in self.check()["failures"]))
        (self.campaign / "plot.png").write_bytes(PNG)
        with patch.object(self.m, "MAX_FILE_BYTES", 10):
            self.assertTrue(any(r["rule"] == "FILE_SIZE_LIMIT" for r in self.check()["failures"]))
        secret = "ghp" + "_" + "q" * 36
        (self.campaign / "README.md").write_text(secret)
        result = self.check()
        self.assertTrue(any(r["rule"] == "SECRET_TOKEN" for r in result["failures"]))
        self.assertNotIn(secret, str(result))

    def test_science_numbers_hashes_and_regex_are_not_phones_or_paths(self):
        text = "0.010123456789 0.07724178633567803 " + "a01025125679b" + "\n"
        text += r"regex [A-Za-z]:[\\/] and SHA256 010251256790abcdef" + "\n"
        text += "Example " + "C" + ":/<workspace>/artifact.md\n"
        (self.campaign / "README.md").write_text(text)
        self.assertEqual(self.check()["status"], "PASS")
        real_path = "C" + ":/Users/person/private/cache"
        (self.campaign / "README.md").write_text(real_path)
        self.assertTrue(any(r["rule"] == "PRIVATE_ABSOLUTE_PATH" for r in self.check()["failures"]))

    def test_svg_external_file_and_wrong_root_rejected(self):
        path = self.campaign / "plot.svg"
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>')
        self.assertEqual(self.check(self.files + ["research/campaign/plot.svg"])["status"], "PASS")
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><image href="../../../outside.png"/></svg>')
        self.assertEqual(self.check(self.files + ["research/campaign/plot.svg"])["status"], "FAIL")
        path.write_text('<html/>')
        self.assertTrue(any(r["rule"] == "SVG_SIGNATURE" for r in self.check(self.files + ["research/campaign/plot.svg"])["failures"]))

    def test_real_file_markdown_wrong_case_is_rejected(self):
        (self.campaign / "README.md").write_text("![image](PLOT.PNG)\n")
        result = self.check()
        failures = [r for r in result["failures"] if r["rule"] == "LINK_CASE_MISMATCH"]
        self.assertEqual(len(failures), 1, result["failures"])
        self.assertEqual(failures[0]["target"], "research/campaign/PLOT.PNG")
        self.assertEqual(failures[0]["expected"], ["research/campaign/plot.png"])

    def test_html_image_and_svg_href_case_match_exact_payload(self):
        svg = self.campaign / "plot.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><image href="PLOT.PNG"/></svg>')
        (self.campaign / "README.md").write_text('<img src="PLOT.PNG">\n')
        files = self.files + ["research/campaign/plot.svg"]
        result = self.check(files)
        self.assertEqual(sum(r["rule"] == "LINK_CASE_MISMATCH" for r in result["failures"]), 2, result["failures"])
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><image href="plot.png#image"/></svg>')
        (self.campaign / "README.md").write_text('<img src="plot.png#image">\n')
        self.assertEqual(self.check(files)["status"], "PASS")

    def test_exact_case_directories_parent_paths_spaces_and_anchors(self):
        docs = self.campaign / "Docs"; docs.mkdir()
        assets = self.campaign / "Assets"; assets.mkdir()
        (assets / "Case Plot.png").write_bytes(PNG)
        doc = docs / "README.md"
        text = "![plot](../Assets/Case%20Plot.png#image)\n[folder](../Assets/)\n[anchor](#local)\n[web](https://example.com/PLOT.PNG)\n[ref][plot]\n[plot]: ../Assets/Case%20Plot.png\n"
        doc.write_text(text)
        files = self.files + ["research/campaign/Docs/README.md", "research/campaign/Assets/Case Plot.png"]
        self.assertEqual(self.check(files)["status"], "PASS")
        doc.write_text(text.replace("../Assets/", "../assets/"))
        result = self.check(files)
        failures = [r for r in result["failures"] if r["rule"] == "LINK_CASE_MISMATCH"]
        self.assertEqual(len(failures), 3, result["failures"])
        self.assertIn(["research/campaign/Assets"], [r["expected"] for r in failures])

    def test_virtual_packed_statistic_link_case_is_exact(self):
        virtual = {"research/campaign/predictions/toy/V_select/PUBLIC_ERROR_STATS.csv.gz"}
        doc = self.campaign / "README.md"
        doc.write_text("[statistics](predictions/toy/V_select/PUBLIC_ERROR_STATS.csv.gz#stats)\n")
        result = self.m.inspect(self.campaign, self.files, tracked={"old.md"}, virtual_files=virtual)
        self.assertEqual(result["status"], "PASS", result["failures"])
        doc.write_text("[statistics](predictions/toy/v_select/public_error_stats.csv.gz)\n")
        result = self.m.inspect(self.campaign, self.files, tracked={"old.md"}, virtual_files=virtual)
        failures = [r for r in result["failures"] if r["rule"] == "LINK_CASE_MISMATCH"]
        self.assertEqual(len(failures), 1, result["failures"])
        self.assertEqual(failures[0]["expected"], sorted(virtual))

    def test_link_symlink_target_stays_rejected(self):
        link = self.campaign / "linked.png"
        try:
            link.symlink_to(self.campaign / "plot.png")
        except OSError:
            return
        (self.campaign / "README.md").write_text("![image](linked.png)\n")
        result = self.check()
        self.assertTrue(any(r["rule"] == "LINK_SYMLINK" for r in result["failures"]), result["failures"])

    def test_mocked_index_scope_and_blob_bytes_must_match_disk(self):
        selected = set(self.files)
        with patch.object(self.m, "index_paths", return_value=selected), patch.object(self.m, "index_blob", side_effect=lambda root, name:(root/name).read_bytes()):
            self.assertEqual(self.m.check_index(self.root, selected)["status"], "PASS")
        with patch.object(self.m, "index_paths", return_value=selected | {"foreign.md"}), patch.object(self.m, "index_blob", return_value=b"wrong"):
            result = self.m.check_index(self.root, selected)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any(r["rule"] == "STAGED_SCOPE" for r in result["failures"]))
        with patch.object(self.m, "index_paths", return_value=selected), patch.object(self.m, "index_blob", return_value=b"wrong"):
            self.assertTrue(any(r["rule"] == "INDEX_DISK_MISMATCH" for r in self.m.check_index(self.root, selected)["failures"]))

    def test_formatted_and_unformatted_contacts_but_not_unix_seconds(self):
        doc = self.campaign / "README.md"
        doc.write_text("1792050350 119608148 103577600\n")
        self.assertEqual(self.check()["status"], "PASS")
        for phone in ("010" + "-1234-5678", "010" + "12345678", "+82" + "-10-1234-5678"):
            doc.write_text(phone)
            result = self.check()
            self.assertTrue(any(r["rule"] == "PHONE_NUMBER" for r in result["failures"]))
            self.assertNotIn(phone, str(result))

    def test_bundle_limit_invalid_utf8_and_explicit_private_asset(self):
        with patch.object(self.m, "MAX_BUNDLE_BYTES", 1):
            self.assertTrue(any(r["rule"] == "BUNDLE_SIZE_LIMIT" for r in self.check()["failures"]))
        (self.campaign / "invalid.json").write_bytes(b"\xff")
        result = self.check(self.files + ["research/campaign/invalid.json"])
        self.assertTrue(any(r["rule"] == "UTF8_OR_GZIP_INVALID" for r in result["failures"]))
        (self.campaign / "weights.npz").write_bytes(b"unpublished")
        result = self.check(self.files + ["research/campaign/weights.npz"])
        self.assertTrue(any(r["rule"] == "DISALLOWED_FILE" for r in result["failures"]))
        private = self.root / ".cache" / "metadata.json"
        private.parent.mkdir(); private.write_text("{}")
        result = self.check(self.files + [".cache/metadata.json"])
        self.assertTrue(any(r["rule"] == "DISALLOWED_FILE" for r in result["failures"]))

    def test_payload_change_and_allowlist_change_are_detected(self):
        with patch.object(self.m, "tracked_files", return_value={"old.md"}):
            self.assertEqual(self.m.prepare(self.campaign, self.files)["status"], "PASS")
            doc = self.campaign / "README.md"
            doc.write_text(doc.read_text() + "Updated payload\n")
            result = self.m.verify(self.campaign)
            self.assertTrue(any(r["rule"] == "MANIFEST_HASH_MISMATCH" for r in result["failures"]))
            self.assertTrue(any(r["rule"] == "PREFLIGHT_PAYLOAD_MISMATCH" for r in result["failures"]))
            allowlist = self.campaign / "PUBLIC_ALLOWLIST.txt"
            allowlist.write_text(allowlist.read_text() + "old.md\n")
            self.assertTrue(any(r["rule"] == "ALLOWLIST_HASH_MISMATCH" for r in self.m.verify(self.campaign)["failures"]))

    def test_symlink_target_and_explicit_traversal_rejected(self):
        result = self.check(self.files + ["../outside.md"])
        self.assertTrue(any(r["rule"] == "FILE_PATH_UNSAFE" for r in result["failures"]))
        link = self.campaign / "linked.md"
        try:
            link.symlink_to(self.root / "old.md")
        except OSError:
            return
        result = self.check(self.files + ["research/campaign/linked.md"])
        self.assertTrue(any(r["rule"] == "FILE_PATH_UNSAFE" for r in result["failures"]))

    def stats_archive(self, relative="stats_packs/pack_sample/part_000.tar.xz"):
        path = self.campaign / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"inspection API fixture")
        manifest = path.parent / "MANIFEST.json"
        manifest.write_text("{}")
        return path, [p.relative_to(self.root).as_posix() for p in (path, manifest)]

    def test_only_manifest_validated_stats_archives_are_selected_and_scanned(self):
        path, files = self.stats_archive()
        entry = {"member_name": "stats/000.csv", "payload_bytes": 17, "sha256": "a" * 64,
                 "csv_text": "origin,mse\n1,0.5\n", "storage_mode": "common_run_id"}
        from unittest.mock import Mock
        validator = Mock(return_value=[entry])
        with patch.dict(sys.modules, {"stats_pack": SimpleNamespace(inspect_archive=validator)}):
            result = self.check(self.files + files)
        selected, _ = self.m.candidate_files(self.campaign)
        self.assertIn(files[0], selected)
        self.assertEqual(result["status"], "PASS", result["failures"])
        validator.assert_called_once_with(path)

    def test_decoded_stats_archive_payload_is_secret_scanned_without_echo(self):
        _, files = self.stats_archive()
        secret = "ghp" + "_" + "q" * 36
        entry = {"member_name": "stats/000.csv", "payload_bytes": 100, "sha256": "b" * 64,
                 "csv_text": "run_id,mse\n" + secret + ",0.5\n", "storage_mode": "opaque_original"}
        with patch.dict(sys.modules, {"stats_pack": SimpleNamespace(inspect_archive=lambda path: [entry])}):
            result = self.check(self.files + files)
        self.assertTrue(any(r["rule"] == "SECRET_TOKEN" and r["file"] == files[0] + "::stats/000.csv" for r in result["failures"]))
        self.assertNotIn(secret, str(result))

    def test_rejected_or_hostile_archive_becomes_failure_without_exception_payload(self):
        _, files = self.stats_archive()
        secret = "ghp" + "_" + "q" * 36
        def reject(path):
            raise ValueError("unsafe member ../../escape.csv " + secret)
        with patch.dict(sys.modules, {"stats_pack": SimpleNamespace(inspect_archive=reject)}):
            result = self.check(self.files + files)
        self.assertTrue(any(r["rule"] == "STATS_ARCHIVE_INVALID" for r in result["failures"]))
        self.assertNotIn(secret, str(result))
        self.assertNotIn("../../escape", str(result))

    def test_archive_outside_stats_pack_scope_is_disallowed_without_validator(self):
        _, files = self.stats_archive("reports/pack_sample/part_000.tar.xz")
        from unittest.mock import Mock
        validator = Mock(side_effect=AssertionError("must not inspect non-stats archive"))
        with patch.dict(sys.modules, {"stats_pack": SimpleNamespace(inspect_archive=validator)}):
            result = self.check(self.files + files)
        self.assertTrue(any(r["rule"] == "DISALLOWED_FILE" for r in result["failures"]))
        validator.assert_not_called()

    def test_benchmark_archive_is_excluded_but_sample_json_is_selected(self):
        archive = self.campaign / "stats_packs" / "sample_benchmark" / "solid_p6.tar.xz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"local benchmark fixture")
        sample = archive.with_name("SAMPLE.json")
        sample.write_text("{}")
        selected, excluded = self.m.candidate_files(self.campaign)
        archive_name = archive.relative_to(self.root).as_posix()
        self.assertNotIn(archive_name, selected)
        self.assertIn(sample.relative_to(self.root).as_posix(), selected)
        self.assertIn({"file": archive_name, "reason": "DISALLOWED_SUFFIX"}, excluded)

    def test_stats_archive_requires_published_sibling_manifest(self):
        _, files = self.stats_archive()
        with patch.dict(sys.modules, {"stats_pack": SimpleNamespace(inspect_archive=lambda path: [])}):
            result = self.check(self.files + files[:1])
        self.assertTrue(any(r["rule"] == "STATS_ARCHIVE_MANIFEST_NOT_PUBLISHED" for r in result["failures"]))

    def test_existing_gzip_scanning_is_preserved(self):
        path = self.campaign / "PUBLIC_ERROR_STATS.csv.gz"
        files = self.files + [path.relative_to(self.root).as_posix()]
        path.write_bytes(gzip.compress(b"origin,mse\n1,0.25\n"))
        self.assertEqual(self.check(files)["status"], "PASS")
        secret = "ghp" + "_" + "q" * 36
        path.write_bytes(gzip.compress(("origin,mse\n" + secret + ",0.25\n").encode()))
        result = self.check(files)
        self.assertTrue(any(r["rule"] == "SECRET_TOKEN" for r in result["failures"]))
        self.assertNotIn(secret, str(result))

    def actual_stats_pack(self, channel="c", opaque=False):
        import stats_pack
        source = self.campaign / "predictions" / "toy" / "V_select" / "PUBLIC_ERROR_STATS.csv.gz"
        source.parent.mkdir(parents=True, exist_ok=True)
        raw = stats_pack.CSV_HEADER + f"toy,V_select,toy,4,{channel},0,1,0.25,0.5,1\r\n".encode()
        source.write_bytes(gzip.compress(raw, mtime=0))
        if opaque:
            with patch.object(stats_pack, "find_gzip_recipe", return_value=None):
                directory = stats_pack.pack(self.campaign, only=[source.relative_to(self.campaign).as_posix()])
        else:
            directory = stats_pack.pack(self.campaign, only=[source.relative_to(self.campaign).as_posix()])
        manifest_path = directory / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        archive = directory / manifest["archives"][0]["file"]
        files = [p.relative_to(self.root).as_posix() for p in (manifest_path, archive)]
        return stats_pack, manifest_path, manifest, archive, files

    def test_actual_stats_pack_passes_archive_inspection(self):
        module, _, _, archive, files = self.actual_stats_pack()
        members = module.inspect_archive(archive)
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["storage_mode"], "csv_common9")
        result = self.check(self.files + files)
        self.assertEqual(result["status"], "PASS", result["failures"])

    def test_actual_stats_csv_secret_is_rejected_for_common_and_opaque_payloads(self):
        secret = "ghp" + "_" + "q" * 36
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                _, _, _, _, files = self.actual_stats_pack(channel=secret, opaque=opaque)
                result = self.check(self.files + files)
                self.assertTrue(any(r["rule"] == "SECRET_TOKEN" and "::members/" in r["file"] for r in result["failures"]))
                self.assertNotIn(secret, str(result))

    def test_actual_tar_escape_and_symlink_are_rejected_with_matching_archive_hashes(self):
        for evil in ("traversal", "symlink"):
            with self.subTest(evil=evil):
                module, manifest_path, manifest, archive, files = self.actual_stats_pack()
                with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive.read_bytes())), mode="r:") as tar:
                    original = tar.next()
                    payload = tar.extractfile(original).read()
                name = "../../escape.csv" if evil == "traversal" else original.name
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                    member = tarfile.TarInfo(name)
                    if evil == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../outside.csv"
                        tar.addfile(member)
                    else:
                        member.size = len(payload)
                        tar.addfile(member, io.BytesIO(payload))
                raw = buffer.getvalue()
                blob = lzma.compress(raw)
                archive.write_bytes(blob)
                declaration = manifest["archives"][0]
                declaration.update(bytes=len(blob), sha256=module.sha(blob), tar_bytes=len(raw))
                declaration["members"][0]["name"] = name
                for record in manifest["records"]:
                    record["member"] = name
                manifest_path.write_text(json.dumps(manifest))
                result = self.check(self.files + files)
                self.assertTrue(any(r["rule"] == "STATS_ARCHIVE_INVALID" for r in result["failures"]))
                self.assertFalse((self.root / "outside.csv").exists())
                self.assertFalse((archive.parent.parent.parent / "escape.csv").exists())


if __name__ == "__main__": unittest.main()
