"""Move only cost method-key text above its footer; preserve v1 and v2."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt

import render_readability_v2 as v2


HERE = Path(__file__).resolve().parent
DOCS_V2 = ("REPORT_KO_v2.md","PROFESSOR_BRIEF_v2.md","DECISION_LOG_v2.md")


def main():
    prior_paths = (v2.original_files()
        + [HERE/name for name in DOCS_V2]
        + [HERE/"figures"/f"{name}_v2.{suffix}" for name,_ in v2.original.FIGURES for suffix in v2.SUFFIXES]
        + [HERE/"figures"/"DATA_PROVENANCE_v2.json",HERE/"render_readability_v2.py"])
    prior_paths = list(dict.fromkeys(prior_paths))
    before = v2.hashes(prior_paths)
    output_paths = ([HERE/"figures"/f"05_error_time_memory_v3.{suffix}" for suffix in v2.SUFFIXES]
        + [HERE/name.replace("_v2.md","_v3.md") for name in DOCS_V2])
    manifest_path = HERE/"FINAL_RENDER_PROVENANCE.json"
    if any(path.exists() for path in output_paths+[manifest_path]):
        raise FileExistsError("Preserve existing v3/final outputs")
    previous_provenance = v2.original.read_json(HERE/"figures"/"DATA_PROVENANCE_v2.json")
    bundle = v2.original.load_bundle()
    movements = []
    original_save = v2.original.save_figure

    def save_cost(fig,name):
        if name != "05_error_time_memory":
            raise ValueError("Only the cost figure is authorized for this rendering")
        for axis in fig.axes:
            if not axis.get_title(loc="left").endswith(": method key"):
                continue
            for text in axis.texts:
                if re.match(r"^\d{2}\s",text.get_text()):
                    x,y = text.get_position()
                    new_y = .96-(.96-y)*(.76/.86)
                    text.set_position((x,new_y))
                    movements.append({"key":axis.get_title(loc="left"),"text":text.get_text(),"x":x,"old_y":y,"new_y":new_y})
        original_save(fig,name+"_v3")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,"legend.fontsize":8,
                         "svg.fonttype":"none","savefig.facecolor":"white"})
    v2.original.save_figure = save_cost
    try:
        points = v2.figure_cost_v2(bundle)
    finally:
        v2.original.save_figure = original_save
    if points != previous_provenance["figure05_points"]:
        raise RuntimeError("V3 cost point values differ from preserved v2")
    if not movements or min(row["new_y"] for row in movements) < .199999999:
        raise RuntimeError("Method key was not raised above the unchanged footer")
    document_checks = {}
    for name in DOCS_V2:
        text = (HERE/name).read_text(encoding="utf-8")
        text = text.replace("figures/05_error_time_memory_v2.png","figures/05_error_time_memory_v3.png")
        text = text.replace("figures/05_error_time_memory_v2.svg","figures/05_error_time_memory_v3.svg")
        for target in DOCS_V2:
            text = text.replace("]("+target+")","]("+target.replace("_v2.md","_v3.md")+")")
        target_path = HERE/name.replace("_v2.md","_v3.md")
        with target_path.open("x",encoding="utf-8",newline="\n") as handle:
            handle.write(text)
        document_checks[target_path.name] = "Only cost-image and cross-document link targets changed; other five figure links retain v2"
    after = v2.hashes(prior_paths)
    if before != after:
        raise RuntimeError("A preserved v1/v2 artifact or source changed")
    provenance = {
        "purpose":"Final cost-key vertical-spacing repair only; no new analysis/selection/results",
        "v1_v2_sha256_before":before,"v1_v2_sha256_after":after,"v1_v2_hashes_unchanged":True,
        "driver_sha256":v2.original.sha(Path(__file__)),
        "reused_source_sha256":{name:v2.original.sha(HERE/name) for name in ("generate_report.py","render_readability_v2.py")},
        "v2_provenance_sha256":v2.original.sha(HERE/"figures"/"DATA_PROVENANCE_v2.json"),
        "key_position_formula":"new_y=.96-(.96-old_y)*(.76/.86); x/footer/scatter points/axes/scopes unchanged",
        "text_movements":movements,"cost_points_equal_to_v2":True,"cost_points":points,
        "document_changes":document_checks,
        "failed_presentation_attempt":{
            "reason":"Attempt01 read the center title instead of the left title, moved zero labels, and failed the post-save label-position assertion; no scientific change",
            "preserved_sha256":{str(path.relative_to(HERE)):v2.original.sha(path) for path in
                [HERE/"render_cost_key_v3_attempt01_failed.py"]
                + [HERE/"figures"/f"05_error_time_memory_v3_attempt01_invalid.{suffix}" for suffix in v2.SUFFIXES]
                if path.exists()},
        },
        "output_sha256":{str(path.relative_to(HERE)):v2.original.sha(path) for path in output_paths},
        "new_fits":0,"F_calls":0,"GPU_calls":0,
    }
    with manifest_path.open("x",encoding="utf-8") as handle:
        json.dump(provenance,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":"FINAL_COST_KEY_V3_GENERATED","original_files_unchanged":len(before),
        "cost_points_equal_to_v2":True,"PNG":1,"SVG":1,"MD":3,"minimum_key_y":min(row['new_y'] for row in movements),
        "new_fits":0,"F_calls":0,"GPU_calls":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
