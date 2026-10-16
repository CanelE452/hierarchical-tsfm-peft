"""Metadata-only exact discovery and SHA-failure checks; no auditor import or arrays."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root",required=True);args=parser.parse_args()
    public=Path(args.public_root).resolve();here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location("sealed_discovery_wrapper",here/"discovery_wrapper.py")
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    metadata=wrapper.metadata(public)
    all_paths={p.relative_to(public).as_posix() for p in public.rglob("PREDICTION_MANIFEST.json")}
    refs_before={p.relative_to(public).as_posix() for p in public.rglob("REFERENCE_CONTRACT.json")}
    other=public/"diagnostics/raw_deep_g_saved_rescore_1792090799000/DERIVED_M_CONTRACTS"
    other_before={p.relative_to(public).as_posix() for p in other.rglob("PREDICTION_MANIFEST.json")}
    calls=[]
    with wrapper.discovery(public,calls):
        canonical={p.relative_to(public).as_posix() for p in public.rglob("PREDICTION_MANIFEST.json")}
        refs_after={p.relative_to(public).as_posix() for p in public.rglob("REFERENCE_CONTRACT.json")}
        other_after={p.relative_to(public).as_posix() for p in other.rglob("PREDICTION_MANIFEST.json")}
    restored={p.relative_to(public).as_posix() for p in public.rglob("PREDICTION_MANIFEST.json")}
    assert len(all_paths)==457 and len(canonical)==453 and all_paths-canonical==set(wrapper.EXCLUDED)
    assert refs_before==refs_after and other_before==other_after and len(other_after)==4 and all_paths==restored
    assert len(calls)==1 and set(calls[0]["excluded"])==set(wrapper.EXCLUDED)
    name=next(iter(wrapper.EXCLUDED));original=wrapper.EXCLUDED[name];rejected=False
    try:
        wrapper.EXCLUDED[name]="0"*64
        try:wrapper.metadata(public)
        except ValueError as error:
            rejected="Pinned manifest bytes changed" in str(error)
    finally:wrapper.EXCLUDED[name]=original
    assert rejected
    assert "numpy" not in sys.modules and "torch" not in sys.modules and "audit_campaign" not in sys.modules
    wrapper.new(here/"DISCOVERY_SCOPE_PROOF.json",dict(status="PASS_METADATA_DISCOVERY_ONLY",actual_original457=len(all_paths),actual_canonical453=len(canonical),actual_exact_exclusions4=len(all_paths-canonical),
        wrong_exclusion_SHA_rejected=True,all_other_root_patterns_unfiltered=True,all_other_subroots_unfiltered=True,Path_rglob_restored_after_context=True,
        original_source39_exact=metadata["source39"],wrapper_sha256=wrapper.sha(here/"discovery_wrapper.py"),original_auditor_imported=False,numpy_imported=False,torch_imported=False,
        actual_FP64_GPU_T_array_model_execution=False,actual_file_corruption_performed=False,created_unix=time.time()))
    print("PASS_METADATA_DISCOVERY_ONLY; exact4 excluded; canonical453; no arrays/auditor/GPU/T")


if __name__=="__main__":main()
