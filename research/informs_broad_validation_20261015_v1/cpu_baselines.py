"""Real CPU-only V inference; parent GPU training and registry are untouched."""
import json
from pathlib import Path
import time

import torch

import campaign
import data_contract
import prediction_cache


def main():
    torch.set_num_threads(2)
    started = time.time()
    outputs = []
    for ds in ("beijing", "jena", "electricity", "pems08"):
        task = data_contract.load_task(ds)
        for family, baseline in (("CURRENT", "CURRENT"), ("SEASONAL_NAIVE", dict(kind="SEASONAL", period=task["period"])),
                                 ("RAW_RIDGE", None)):
            run_id = f"primary_{ds}_E1_{family}_k0_s92601"
            result = prediction_cache.predict_parent(run_id, task, "V_select", campaign.CACHE, campaign.HERE,
                                                      None, baseline=baseline)
            outputs.append(dict(run_id=run_id, score=result["score"], actual_F_streams=result["cost_record"]["actual_F_streams"],
                                prediction_manifest_relative_path=f"predictions/{run_id}/V_select/PREDICTION_MANIFEST.json"))
            print(json.dumps(dict(event="CPU_V_PREDICTION_DONE", **outputs[-1])), flush=True)
    receipt = dict(status="ACTUAL_V_INFERENCE_COMPLETE", device="CPU", torch_threads=2,
                   source_sha256=data_contract.sha256_file(__file__), outputs=outputs,
                   started_unix=started, completed_unix=time.time(), GPU_calls=0, T_final_accessed=False,
                   registry_mutated=False, registry_reconciliation_pending=True)
    folder = campaign.HERE / "execution_batches"
    folder.mkdir(exist_ok=True)
    path = folder / ("cpu_baselines_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
