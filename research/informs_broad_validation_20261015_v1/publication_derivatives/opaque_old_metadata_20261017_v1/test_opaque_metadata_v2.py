"""Tiny stdlib synthetic fixtures only; no current experiment or old payload IO."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack = load("test_opaque_pack_v2", "pack_metadata_v2.py")
inspector = load("test_opaque_inspector_v2", "inspect_opaque_metadata_v2.py")


def manifest_fixture():
    return dict(format="exact_auxiliary_gzip_v1", items=[
        dict(original_file=name, original_bytes=size, original_sha256=sha,
             gzip_file=gzip_name, gzip_bytes=17, gzip_sha256="1" * 64)
        for (name, size, sha), gzip_name in zip(pack.ORIGINALS, pack.GZIP_NAMES)])


def proof_fixture():
    expected = {name: dict(bytes=size, sha256=sha) for name, size, sha in pack.ORIGINALS}
    return dict(schema="OPAQUE_OLD_METADATA_BYTE_PROOF_v2", status="PASS_EXACT_THREE_ALREADY_PUBLIC_METADATA_BYTES",
        old_inventory_sha256=pack.INVENTORY_SHA, existing_restorer_sha256=pack.RESTORER_SHA,
        original_before=expected, original_after=copy.deepcopy(expected), source48_before={str(i): i for i in range(48)},
        source48_after={str(i): i for i in range(48)}, mismatch_refused_without_write=True, original_files_modified=False,
        original_payload_JSON_UTF8_or_numeric_parse=False, new_T_numeric_parse=0, worker_deadline_unix=pack.DEADLINE,
        CPU2_RSS1GiB_external_supervisor_admission="ADMITTED", resource_terminal_admission_independent_of_scientific_success=True,
        scientific_terminal_statuses={}, fresh_restores=[
            dict(destination_label=label, first=dict(status="PASS", restored=3, already_matching=0),
                 repeated=dict(status="PASS", restored=0, already_matching=3),
                 files=[dict(file=name, **expected[name]) for name, _, _ in pack.ORIGINALS])
            for label in ("restore_a", "restore_b")])


def synthetic_terminal_fixture():
    """All receipt IO is mocked; these are not actual or expected run receipts."""
    records = {}

    def record(label, value):
        path = "SYNTHETIC_NOT_REAL/" + label
        records[path] = value
        return dict(path=path, sha256=pack.digest(json.dumps(value, sort_keys=True).encode()))

    def binding(label, count, failed=False):
        spec = record(label + "_spec", dict(schema="CPU_GROUP_SUPERVISOR_v1", deadline_unix=pack.DEADLINE,
            processes=[dict(id=f"{label}_worker_{i}") for i in range(count)]))
        workers = []
        for index in range(count):
            root = dict(pid=1000 + index, create_time=10.0 + index)
            workers.append(dict(id=f"{label}_worker_{index}", root=root, returncode=1 if failed and index == 0 else 0,
                end_unix=90, observed_tree_join_confirmed=True, process_tree_errors=[], remaining_owned_processes=[],
                owned_processes=[dict(root, peak_RSS_bytes=123, last_CPU_seconds=0)]))
        guard = record(label + "_guard", dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1",
            status="STOPPED_RESOURCE_OR_WORKER_FAILURE" if failed else "COMPLETE", helper_sha256=pack.GUARD_HELPER_SHA,
            helper_unchanged=True, spec_file_sha256=spec["sha256"], supervisor_failure_class=None,
            remaining_owned_processes=[], processes=workers))
        closure = record(label + "_closure", dict(schema="ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2",
            status="VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE", guard_sha256=guard["sha256"],
            spec_sha256=spec["sha256"], actual_PID_creation_time_check=True, remaining_exact_owners=[], errors=[],
            checked_unix=100, absent_identities=[dict(pid=1000 + i, create_time=10.0 + i) for i in range(count)]))
        return dict(spec=spec, guard=guard, closure=closure)

    fit_guard = binding("fit", 2, failed=True)
    fit = record("fit_dispatch", dict(status="PARTIAL_TWO_FITS_PRESERVED", error=None,
        pins_before_after_unchanged=True, strong_observed_tree_closure_verified=True,
        changed_or_unreadable_pinned_paths=[], source39_immutable9_count=48, original_worker_deadline=pack.DEADLINE,
        conditions=[dict(run_id=run, outer_status="FAILED_SYNTHETIC" if i == 0 else "COMPLETE_SYNTHETIC",
            worker_science_complete=i == 1, budget_partial_preserved=False) for i, run in enumerate(pack.RUN_IDS)]))
    post_bindings = []
    steps = []
    for i, mode in enumerate(("rescore", "replay", "cost", "cost")):
        item = binding(f"post_{i}", 1)
        item["id"] = f"post_{i}"
        post_bindings.append(item)
        steps.append(dict(id=item["id"], run_id=pack.RUN_IDS[1], mode=mode, batch=None if i < 2 else (1 if i == 2 else 8),
            success=True, outer_status="PASS_SYNTHETIC", strong_observed_tree_closure_verified=True,
            guard_sha256=item["guard"]["sha256"], spec_sha256=item["spec"]["sha256"]))
    post = record("post_result", dict(status="PARTIAL_VALIDATIONS_COSTS_PRESERVED", error=None,
        pins_before_after_unchanged=True, changed_or_unreadable_pinned_paths=[], original_worker_deadline=pack.DEADLINE,
        fit_dispatch_result_sha256=fit["sha256"], steps=steps,
        skipped=[dict(run_id=pack.RUN_IDS[0], reason="FIT_NOT_CLEAN_COMPLETE_PRESERVED", status="FAILED_SYNTHETIC")]))
    value = dict(schema="ROOT_OPAQUE_METADATA_QUIET_ADMISSION_v2", status="ADMITTED", actual_current_process_inspection=True,
        actual_current_campaign_CIM_empty=True, remaining_campaign_model_cost_pack_processes=[], errors=[],
        worker_deadline_unix=pack.DEADLINE, cpu_threads=2, RSS_limit_bytes=1024**3, storage_limit_bytes=pack.STORAGE_LIMIT,
        storage_reserve_bytes=pack.STORAGE_RESERVE, storage_used_bytes=0, checked_unix=100,
        fit_dispatch_result=fit, fit_guard_binding=fit_guard, post_result=post, executed_post_guard_bindings=post_bindings)
    return value, records


class OpaqueMetadataTests(unittest.TestCase):
    def test_both_defaults_have_zero_input_reads_and_writes(self):
        for module in (pack, inspector):
            with self.subTest(module=module.__name__), mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read")), \
                    mock.patch.object(Path, "read_text", side_effect=AssertionError("read")), \
                    mock.patch.object(Path, "open", side_effect=AssertionError("open")), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(module.main([]), 0)
                value = json.loads(output.getvalue())
                self.assertEqual(value["input_reads"], 0)
                self.assertEqual(value["files_written"], 0)

    def test_execute_without_root_trigger_refused_before_io(self):
        for module in (pack, inspector):
            with self.subTest(module=module.__name__), mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                module.main(["--execute"])
            self.assertEqual(error.exception.code, 2)

    def test_draft_is_not_executable_and_deadline_is_original(self):
        with mock.patch.object(pack, "bound", return_value=b'{"schema":"OPAQUE_OLD_METADATA_PLAN_v2","status":"DRAFT_NOT_EXECUTABLE"}'), \
                mock.patch.object(pack.time, "time", return_value=pack.DEADLINE - 1), self.assertRaises(ValueError):
            pack.execute("unused", "unused", "unused", "1" * 64)
        with mock.patch.object(pack, "bound", side_effect=AssertionError("deadline must fail first")), \
                mock.patch.object(pack.time, "time", return_value=pack.DEADLINE), self.assertRaises(ValueError):
            pack.execute("unused", "unused", "unused", "1" * 64)

    def test_relative_paths_and_fresh_write_refuse_traversal_overwrite(self):
        for bad in ("../x", "/x", "a//b", "a/./b", (Path(tempfile.gettempdir()).resolve() / "x").as_posix(), "a\\b", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pack.safe(bad)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output.json"
            pack.json_new(target, dict(synthetic=True))
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                pack.json_new(target, dict(changed=True))
            self.assertEqual(target.read_bytes(), before)

    def test_inventory_requires_all_three_original_sha_bindings(self):
        header = "file,bytes,sha256\n"
        content = header + "".join(f"{name},{size},{sha}\n" for name, size, sha in pack.ORIGINALS)
        self.assertEqual(len(pack.inventory_rows(content.encode())), 3)
        for changed in (content.replace(pack.ORIGINALS[0][2], "0" * 64), header + "".join(f"{name},{size},{sha}\n" for name, size, sha in pack.ORIGINALS[1:])):
            with self.assertRaises(ValueError):
                pack.inventory_rows(changed.encode())

    def test_exact_three_manifest_and_namespace_only(self):
        good = manifest_fixture()
        self.assertIs(pack.validate_manifest(good), good)
        variants = []
        fourth = copy.deepcopy(good)
        fourth["items"].append(copy.deepcopy(fourth["items"][0]))
        variants.append(fourth)
        for key, value in (("gzip_file", "../escape.opaque.json.gz"), ("original_sha256", "0" * 64), ("original_bytes", 1)):
            changed = copy.deepcopy(good)
            changed["items"][0][key] = value
            variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                pack.validate_manifest(changed)

    def test_tiny_bytes_gzip_is_deterministic_and_opaque(self):
        raw = b"\xff\x00SYNTHETIC-BYTES-not-UTF8\r\n\x80"
        first = pack.deterministic_gzip(raw)
        self.assertEqual(first, pack.deterministic_gzip(raw))
        self.assertEqual(gzip.decompress(first), raw)
        with self.assertRaises(UnicodeDecodeError):
            raw.decode("utf-8")

    def test_frozen_restorer_two_fresh_repeat_and_mismatch(self):
        helper = pack.load_restorer(Path(__file__).with_name("restore_originals.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for index in range(3):
                raw = b"\xff\x00SYNTHETIC" + bytes([index])
                blob = pack.deterministic_gzip(raw)
                name = f"tiny_{index}.opaque.json.gz"
                (root / name).write_bytes(blob)
                items.append(dict(original_file=f"synthetic/path_{index}.json", original_bytes=len(raw), original_sha256=pack.digest(raw),
                                  gzip_file=name, gzip_bytes=len(blob), gzip_sha256=pack.digest(blob)))
            manifest_path = root / "GZIP_MANIFEST.json"
            pack.json_new(manifest_path, dict(format="exact_auxiliary_gzip_v1", items=items))
            result = pack.restore_two(helper, manifest_path, items, root / "fresh_private")
            self.assertEqual([row["first"]["restored"] for row in result], [3, 3])
            with self.assertRaises(FileExistsError):
                pack.restore_two(helper, manifest_path, items, root / "fresh_private")
            altered = copy.deepcopy(items)
            altered[0]["gzip_sha256"] = "0" * 64
            pack.json_new(root / "tampered.json", dict(format="exact_auxiliary_gzip_v1", items=altered))
            with self.assertRaises(ValueError):
                helper.restore(root / "tampered.json", root / "another")

    def test_proof_requires_exact_both_restore_receipts_and_unchanged_pins(self):
        good = proof_fixture()
        inspector.check_proof(good, manifest_fixture(), pack)
        variants = []
        for key, value in (("status", "PREP_ONLY"), ("old_inventory_sha256", "0" * 64), ("new_T_numeric_parse", 1), ("original_files_modified", True)):
            changed = copy.deepcopy(good)
            changed[key] = value
            variants.append(changed)
        changed = copy.deepcopy(good)
        changed["fresh_restores"][1]["first"]["restored"] = 0
        variants.append(changed)
        changed = copy.deepcopy(good)
        changed["source48_after"]["0"] = 99
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                inspector.check_proof(changed, manifest_fixture(), pack)

    def test_virtual_facade_preserves_frozen_checks_for_other_files(self):
        calls = []
        frozen = types.SimpleNamespace(inspect=lambda *args, **kwargs: calls.append((args, kwargs)) or dict(status="PASS"))
        facade = inspector.ExactVirtualFacade(frozen, ("a.opaque.json.gz", "b.opaque.json.gz", "c.opaque.json.gz"))
        facade.inspect("campaign", ["normal.md"], tracked={"tracked.py"}, virtual_files=["part.numeric.xz"])
        self.assertEqual(calls[0][1]["virtual_files"], ("part.numeric.xz", "a.opaque.json.gz", "b.opaque.json.gz", "c.opaque.json.gz"))
        self.assertEqual(calls[0][0][1], ["normal.md"])
        with self.assertRaises(ValueError):
            facade.inspect("campaign", ["a.opaque.json.gz"])

    def test_private_dependency_loader_restored_even_on_failure(self):
        frozen = types.SimpleNamespace(inspect=lambda *args, **kwargs: {})
        original = lambda: (frozen, object())
        module = types.SimpleNamespace(load_dependencies=original)
        def inspect(campaign, normal, tracked):
            facade, _ = module.load_dependencies()
            self.assertIsInstance(facade, inspector.ExactVirtualFacade)
            self.assertEqual(tracked, {"tracked", "a", "b", "c"})
            raise ValueError("synthetic failure")
        module.inspect = inspect
        with self.assertRaises(ValueError):
            inspector.delegate_numeric(module, "campaign", ["normal"], ["tracked"], ["a", "b", "c"])
        self.assertIs(module.load_dependencies, original)

    def test_stale_busy_or_resource_contract_admission_is_rejected(self):
        value = dict(schema="ROOT_OPAQUE_METADATA_QUIET_ADMISSION_v2", status="ADMITTED", actual_current_process_inspection=True,
            actual_current_campaign_CIM_empty=True, remaining_campaign_model_cost_pack_processes=[], errors=[], worker_deadline_unix=pack.DEADLINE,
            cpu_threads=2, RSS_limit_bytes=1024**3, storage_limit_bytes=pack.STORAGE_LIMIT,
            storage_reserve_bytes=pack.STORAGE_RESERVE, storage_used_bytes=0, checked_unix=-1000)
        for changed in (value, dict(value, checked_unix=-1000), dict(value, remaining_campaign_model_cost_pack_processes=[123]),
                        dict(value, storage_used_bytes=pack.STORAGE_LIMIT), dict(value, worker_deadline_unix=pack.DEADLINE + 1)):
            with self.assertRaises(ValueError):
                pack.validate_quiet(changed, 101)

    def test_one_scientifically_failed_fit_but_all_resources_closed_is_admitted(self):
        value, records = synthetic_terminal_fixture()
        with mock.patch.object(pack, "read_descriptor", side_effect=lambda row: copy.deepcopy(records[row["path"]])), \
                mock.patch.object(pack, "FIT_SPEC_SHA", value["fit_guard_binding"]["spec"]["sha256"]):
            result = pack.validate_quiet(value, 101)
        science = result["scientific_terminal_statuses"]
        self.assertFalse(science["fit_routes"][0]["worker_science_complete"])
        self.assertEqual(science["fit_routes"][0]["outer_status"], "FAILED_SYNTHETIC")
        self.assertEqual(len(science["post_steps"]), 4)
        self.assertEqual(science["skipped"][0]["run_id"], pack.RUN_IDS[0])

    def test_active_missing_closure_wrong_spec_and_fake_skipped_guard_refused(self):
        for case in ("active", "missing_closure", "wrong_spec", "invented_guard", "changed_source", "live_owner"):
            with self.subTest(case=case):
                value, records = synthetic_terminal_fixture()
                original_spec_sha = value["fit_guard_binding"]["spec"]["sha256"]
                if case == "active":
                    value["remaining_campaign_model_cost_pack_processes"] = [42]
                elif case == "missing_closure":
                    value["executed_post_guard_bindings"].pop()
                elif case == "wrong_spec":
                    value["fit_guard_binding"]["spec"]["sha256"] = "0" * 64
                elif case == "invented_guard":
                    invented = copy.deepcopy(value["executed_post_guard_bindings"][0])
                    invented["id"] = "NEVER_EXECUTED_SKIPPED_ROUTE"
                    value["executed_post_guard_bindings"].append(invented)
                elif case == "changed_source":
                    records[value["fit_dispatch_result"]["path"]]["pins_before_after_unchanged"] = False
                else:
                    records[value["fit_guard_binding"]["closure"]["path"]]["remaining_exact_owners"] = [42]
                with mock.patch.object(pack, "read_descriptor", side_effect=lambda row: copy.deepcopy(records[row["path"]])), \
                        mock.patch.object(pack, "FIT_SPEC_SHA", original_spec_sha), self.assertRaises(ValueError):
                    pack.validate_quiet(value, 101)


if __name__ == "__main__":
    unittest.main()
