"""Rehashed corrupt-CSV fixtures check independent report verification."""
from pathlib import Path
import csv
import json

import pytest

import make_fit_target_report_v1 as generator
import verify_fit_target_report_v1 as verifier
from test_make_fit_target_report_v1 import gate_fixture


def fixture(tmp_path):
    inputs = tmp_path / "inputs"; inputs.mkdir()
    result_path, audit_path = gate_fixture(inputs)
    result, audit = generator.read(result_path), generator.read(audit_path)
    for costs in result["cost"].values():
        for row in costs["rows"]:
            row["paired_RAW_ratios"] = list(row["seconds"])
            row["paired_RAW_median_ratio"] = row["median_seconds"]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    audit["producer"] = generator.record(result_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    folder = tmp_path / "report"; folder.mkdir(); (folder / "tables").mkdir(); (folder / "figures").mkdir()
    tables = generator.collect_tables(result, audit)
    for table, rows in tables.items(): generator.write_csv(folder / "tables" / (table + ".csv"), rows)
    prior = []
    for index in range(3):
        path = tmp_path / f"prior{index}.md"; path.write_text("approved old report", encoding="utf-8"); prior.append((path, generator.sha(path)))
    figures = []
    for index in range(5):
        for suffix in ("png", "svg"):
            path = folder / "figures" / f"fixture{index}.{suffix}"; path.write_text("synthetic artifact bytes", encoding="utf-8"); figures.append(path.relative_to(folder).as_posix())
    (folder / "REPORT_KO.md").write_text("\n".join([f"![fixture](figures/fixture{index}.png)" for index in range(5)] +
        [f"[prior](../prior{index}.md)" for index in range(3)] + [f"[{table}](tables/{table}.csv)" for table in tables]), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": generator.record(Path(generator.__file__)), "producer": generator.record(result_path),
        "audit": generator.record(audit_path), "source_bindings": {"generator": generator.record(Path(generator.__file__))},
        "prior_reports": [{"path": str(path), "sha256": digest} for path, digest in prior], "metadata_only": True,
        "supervised_FIT_future_training": True, "deployment_past_only": True,
        **{flag: False for flag in ("NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "past_only_or_unsupervised_training_claimed",
            "independent_generalization_claimed", "new_model_or_method_selected_posthoc", "fixed_D_MAE_or_physical_MSE_minimum_claimed", "RAW_bound_applied")},
        "CSV_rows": {table: len(rows) for table, rows in tables.items()}, "figures": figures,
        "files": [{"path": path.relative_to(folder).as_posix(), "sha256": generator.sha(path), "bytes": path.stat().st_size} for path in folder.rglob("*") if path.is_file()]}
    (folder / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, prior


def rewrite_csv(path, mutation):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows, columns = list(reader), reader.fieldnames
    mutation(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def rehash(folder, path):
    manifest = generator.read(folder / "MANIFEST.json")
    item = next(item for item in manifest["files"] if item["path"] == path.relative_to(folder).as_posix())
    item.update(sha256=generator.sha(path), bytes=path.stat().st_size)
    (folder / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_full_synthetic_report_and_independent_arithmetic(tmp_path):
    folder, prior = fixture(tmp_path)
    receipt = verifier.verify(folder, prior)
    assert receipt["status"] == "PASS" and receipt["files_checked"] == 23
    assert receipt["CSV_rows"] == verifier.COUNTS and receipt["total_CSV_primitive_fields_checked"] > 50000
    assert receipt["timer_medians_ratios_checkpoint_choice_recomputed_from_metadata"]
    assert not receipt["NPZ_or_model_opened"] and receipt["PNG_count"] == receipt["SVG_count"] == 5


@pytest.mark.parametrize("change", ["score", "curve_loss", "checkpoint_selected", "nested_metadata", "physical_minimum", "adoption", "duplicate"])
def test_corrupt_values_rejected_even_when_artifact_manifest_sha_is_updated(tmp_path, change):
    folder, prior = fixture(tmp_path)
    table = {"score": "COMPARISON", "curve_loss": "LEARNING_CURVE", "checkpoint_selected": "CHECKPOINTS",
             "nested_metadata": "FIT_SPACE_METADATA", "physical_minimum": "MATCHED_COMPLETE_DIAGNOSTIC",
             "adoption": "REGISTERED_ADOPTION", "duplicate": "FIT_TARGET_SPECTRUM"}[change]
    path = folder / "tables" / (table + ".csv")
    def mutation(rows):
        if change == "score": rows[0]["mse"] = "100.0"
        elif change == "curve_loss": rows[0]["loss"] = "123.0"
        elif change == "checkpoint_selected": rows[0]["selected"] = "True"
        elif change == "nested_metadata":
            value = json.loads(rows[0]["fit_metadata"]); value["latent_fit_covariance"][0][0] = 100.; rows[0]["fit_metadata"] = json.dumps(value)
        elif change == "physical_minimum": rows[0]["physical_MSE_is_minimum"] = "True"
        elif change == "adoption": rows[0]["all_registered_conditions_pass"] = "True"
        else: rows[1] = dict(rows[0])
    rewrite_csv(path, mutation); rehash(folder, path)
    with pytest.raises(ValueError, match="CSV_SOURCE_VALUE_MISMATCH|UNKNOWN_OR_DUPLICATE_CSV_ROW"):
        verifier.verify(folder, prior)


def test_wrong_old_D_report_version_is_rejected(tmp_path):
    folder, prior = fixture(tmp_path); changed = list(prior); changed[-1] = (prior[-1][0], "not the approved v2")
    with pytest.raises(ValueError, match="PRIOR_REPORT_VERSION_CHANGED"): verifier.verify(folder, changed)


def test_missing_figure_and_broken_rehashed_markdown_link_are_rejected(tmp_path):
    folder, prior = fixture(tmp_path)
    path = folder / "REPORT_KO.md"; path.write_text(path.read_text(encoding="utf-8") + "\n[broken](tables/missing.csv)", encoding="utf-8"); rehash(folder, path)
    with pytest.raises(ValueError, match="MISSING_LOCAL_LINK"): verifier.verify(folder, prior)


def test_gate_rejects_oracle_use_as_unsupervised_report_scope(tmp_path):
    folder, prior = fixture(tmp_path)
    path = folder / "MANIFEST.json"; value = generator.read(path); value["past_only_or_unsupervised_training_claimed"] = True; path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="REPORT_SCOPE_CHANGED"): verifier.verify(folder, prior)
