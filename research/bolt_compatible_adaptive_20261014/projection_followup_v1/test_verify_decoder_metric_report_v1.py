"""Tampered synthetic report fixtures verify byte and independent leaf checks."""
from pathlib import Path
import csv
import json

import pytest

import make_decoder_metric_report_v1 as generator
import verify_decoder_metric_report_v1 as verifier
from test_make_decoder_metric_report_v1 import gate_fixture, synthetic_metadata


def fixture(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = gate_fixture(inputs)
    decoder, cost = synthetic_metadata()
    d, c = generator.read(paths[0]), generator.read(paths[2])
    d.update(decoder)
    for rows in cost["cost"].values():
        for row in rows["rows"]:
            row["paired_RAW_ratios"] = list(row["seconds"])
            row["paired_RAW_median_ratio"] = row["median_seconds"]
    c.update(cost)
    paths[0].write_text(json.dumps(d), encoding="utf-8")
    da = generator.read(paths[1]); da["diagnostic"] = generator.record(paths[0])
    paths[1].write_text(json.dumps(da), encoding="utf-8")
    c["decoder_diagnostic"], c["decoder_audit"] = generator.record(paths[0]), generator.record(paths[1])
    paths[2].write_text(json.dumps(c), encoding="utf-8")
    ca = generator.read(paths[3]); ca.update(cost=generator.record(paths[2]), decoder_diagnostic=generator.record(paths[0]), decoder_audit=generator.record(paths[1]))
    paths[3].write_text(json.dumps(ca), encoding="utf-8")
    folder = tmp_path / "report"
    folder.mkdir(); (folder / "tables").mkdir(); (folder / "figures").mkdir()
    tables = generator.collect_tables(d, c)
    for table, rows in tables.items(): generator.write_csv(folder / "tables" / (table + ".csv"), rows)
    figures = []
    for index in range(3):
        for suffix in ("png", "svg"):
            path = folder / "figures" / f"fixture{index}.{suffix}"
            path.write_text("synthetic fixture bytes", encoding="utf-8")
            figures.append(path.relative_to(folder).as_posix())
    prior = []
    for index in range(2):
        path = tmp_path / f"prior{index}.md"
        path.write_text("old immutable report", encoding="utf-8")
        prior.append((path, generator.sha(path)))
    text = "\n".join([f"![fixture](figures/fixture{index}.png)" for index in range(3)] +
                     [f"[prior{index}](../prior{index}.md)" for index in range(2)] +
                     [f"[{name}](tables/{name}.csv)" for name in tables])
    (folder / "REPORT_KO.md").write_text(text, encoding="utf-8")
    manifest = {"status": "COMPLETE", "metadata_only": True, "source": generator.record(Path(generator.__file__)),
        "source_bindings": {"source": generator.record(Path(generator.__file__))},
        **{key: generator.record(path) for key, path in zip(("decoder", "decoder_audit", "cost", "cost_audit"), paths)},
        "prior_reports": [{"path": str(path), "sha256": digest} for path, digest in prior],
        "counts": {name: len(rows) for name, rows in tables.items()}, "figures": figures,
        **{key: False for key in ("scientific_NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed",
            "independent_generalization_claimed", "E_Y_full_observed_zero_filled_future_included_in_major_comparison",
            "native_head_reoptimized_for_new_D", "obliqueness_only_causal_effect_claimed", "cost_native_inference_measured")}}
    manifest["files"] = [{"path": path.relative_to(folder).as_posix(), "sha256": generator.sha(path), "bytes": path.stat().st_size}
                         for path in folder.rglob("*") if path.is_file()]
    generator.write_json(folder / "MANIFEST.json", manifest)
    return folder, prior


def update_artifact_manifest(folder, path):
    manifest = generator.read(folder / "MANIFEST.json")
    row = next(row for row in manifest["files"] if row["path"] == path.relative_to(folder).as_posix())
    row.update(sha256=generator.sha(path), bytes=path.stat().st_size)
    (folder / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def mutate_csv(path, mutation):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    mutation(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_complete_synthetic_report_leaf_checks_and_timing_recalculation(tmp_path):
    folder, prior = fixture(tmp_path)
    receipt = verifier.verify(folder, prior)
    assert receipt["status"] == "PASS" and receipt["files_checked"] == 15
    assert receipt["CSV_rows"] == verifier.COUNTS and receipt["total_CSV_primitive_fields_checked"] > 10000
    assert receipt["timer_and_median_arithmetic_recomputed_from_metadata"]
    assert not receipt["scientific_NPZ_or_model_opened"]


@pytest.mark.parametrize("change", ["scalar_score", "nested_array", "nested_source_hash", "adoption_flag", "duplicate_row"])
def test_independent_csv_checks_reject_tampering_even_after_file_manifest_is_rehashed(tmp_path, change):
    folder, prior = fixture(tmp_path)
    table = "SOURCE_LINEAGE" if change == "nested_source_hash" else "REGISTERED_ADOPTION" if change == "adoption_flag" else "DEPLOYMENT_AND_ORACLE_SCORES"
    path = folder / "tables" / (table + ".csv")
    def mutation(rows):
        if change == "scalar_score": rows[0]["normalized_MSE"] = "123.0"
        elif change == "nested_array": rows[0]["normalized_per_channel_MSE"] = "[1.0]"
        elif change == "nested_source_hash":
            source = json.loads(rows[0]["source_input"]); source["sha256"] = "changed"; rows[0]["source_input"] = json.dumps(source)
        elif change == "adoption_flag": rows[0]["all_registered_conditions_pass"] = "True"
        else: rows[1] = dict(rows[0])
    mutate_csv(path, mutation); update_artifact_manifest(folder, path)
    with pytest.raises(ValueError, match="CSV_SOURCE_VALUE_MISMATCH|NESTED_VECTOR_CHANGED|CSV_DUPLICATE"):
        verifier.verify(folder, prior)


def test_deleted_image_and_wrong_prior_version_are_rejected(tmp_path):
    folder, prior = fixture(tmp_path)
    (folder / "figures/fixture0.png").unlink()
    with pytest.raises(ValueError, match="MANIFEST_BYTES_OR_SHA"):
        verifier.verify(folder, prior)


def test_wrong_prior_report_version_rejected(tmp_path):
    folder, prior = fixture(tmp_path)
    altered = list(prior); altered[1] = (prior[1][0], "wrong approved version")
    with pytest.raises(ValueError, match="WRONG_PRIOR_REPORT_VERSION"):
        verifier.verify(folder, altered)


def test_broken_local_link_is_detected_after_md_manifest_rehash(tmp_path):
    folder, prior = fixture(tmp_path)
    path = folder / "REPORT_KO.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n[missing](tables/missing.csv)", encoding="utf-8")
    update_artifact_manifest(folder, path)
    with pytest.raises(ValueError, match="MISSING_LOCAL_LINK"):
        verifier.verify(folder, prior)
