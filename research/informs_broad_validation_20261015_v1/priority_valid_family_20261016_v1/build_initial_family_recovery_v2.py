"""CreateNew original-name reports from the distinct recovery auditor PASS only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

NAME="priority_valid_family_20261016_v1"
ORIGINAL_BUILDER_SHA="1cac20838e49cf5a4c28b3b4715664b4757a3e0dcd9c751105b7e4c02bc383e4"
OUTPUTS=("INITIAL_FAMILY_RESULTS_v1.csv","INITIAL_FAMILY_KO_v1.md","INITIAL_FAMILY_BINDINGS_v1.json")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def load(path,name,want):
    if sha(path)!=want:raise ValueError("Exact ROOT-pinned report/audit producer required")
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def recovery_notice(provenance,audit_relative):
    return f"""원 POST wrapper는 mixed-slash cache-key 조회 때문에 manifest 및 fit guard SHA를 찾지 못해 **PARTIAL_VALIDATIONS_COSTS_PRESERVED**로 끝났다. 원 결과 SHA `{provenance['original_POST_sha256']}`와 네 `success=false` 행·null fit-guard 기록은 그대로 보존했다. 실제 원 네 validation의 science PASS·guard0·STARTED/선택 artifact를 별도 독립 검증하고, 학습·validation을 다시 실행하지 않은 **새 비용4조건**을 결속했다. 별도 recovery RESULT SHA는 `{provenance['recovery_RESULT_sha256']}`다. 원 wrapper를 성공으로 수정한 결과가 아니다. [원 기록·복구 경로의 독립 감사]({audit_relative}).

이번 두 신규 방법과 기존 B2 reference의 학습 metadata는 모두 search 누적3epoch, **선택 fresh refit1epoch**, search/refit **1863/752 update·29769/12020 exposure**로 일치한다. 동일 노출이 LoRA C21 대 B4/B2 K6의 동일 F 연산·동일 학습률·동일 모수 수를 뜻하지 않는다. B2는 기존 selected artifact의 정확도를 재사용한다.
"""


def execute(public,cache,audit_path,audit_sha,auditor_sha,builder_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME;audit_path=Path(audit_path).resolve()
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Existing original-name initial reports preserved")
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Actual recovery audit must be public N5 JSON")
    if Path(__file__).resolve()!=scope/"build_initial_family_recovery_v2.py" or sha(__file__)!=builder_sha or sha(audit_path)!=audit_sha:raise ValueError("Current reviewed builder and actual audit SHA required")
    original=load(scope/"build_initial_family.py","frozen_initial_family_composition",ORIGINAL_BUILDER_SHA)
    a=load(scope/"audit_initial_family_recovery_v2.py","initial_family_recovery_contract",auditor_sha);base=a.load_original(public);_,helper=base.load_helpers(public)
    inputs=helper.Inputs(public,cache)
    for path,want in (("build_initial_family_recovery_v2.py",builder_sha),("audit_initial_family_recovery_v2.py",auditor_sha),("build_initial_family.py",ORIGINAL_BUILDER_SHA),("audit_initial_family.py",a.ORIGINAL_AUDITOR_SHA)):inputs.capture("public",NAME+"/"+path,want)
    audit=inputs.json("public",audit_path.relative_to(public).as_posix(),audit_sha);original.validate_audit(audit)
    provenance=audit.get("recovery_provenance",{});activity=audit.get("matched_training_activity",{})
    if (audit.get("audit_source_sha256")!=auditor_sha or provenance.get("original_POST_sha256")!=a.ORIGINAL_POST_SHA or provenance.get("original_POST_status")!="PARTIAL_VALIDATIONS_COSTS_PRESERVED" or provenance.get("original_false_rows_count")!=4 or provenance.get("original_receipt_mutated") is not False or provenance.get("original_fit_guard_sha256") is not None or provenance.get("new_cost_conditions")!=4 or provenance.get("new_validation_inference_conditions")!=0 or not a.digest(provenance.get("recovery_RESULT_sha256"))):raise ValueError("Actual recovery audit must preserve original partial/false history")
    if any(activity.get(key)!=value for key,value in a.ACTIVITY.items()) or len(activity.get("methods",[]))!=3:raise ValueError("Actual three-method matched activity evidence required")
    for row in audit["inputs"]:inputs.capture(row["root"],row["relative_path"],row["sha256"])
    texts=original.compose(audit["rows"])
    generic="검색0/.25/1/3은 같아도 선택 refit 길이와 exposure/update가 다를 수 있다."
    if generic not in texts[OUTPUTS[1]]:raise ValueError("Frozen composition context changed")
    texts[OUTPUTS[1]]=texts[OUTPUTS[1]].replace(generic,"이번 두 조건과 기존 B2 reference는 search3/refit1 및 실제 update/exposure가 같다. 동일 F 연산·학습률·용량 대조는 아니다.")
    title="말단 wrapper 기록과 과학적 validation·비용을 구분한 복구\n\n"+recovery_notice(provenance,audit_path.relative_to(scope).as_posix())
    title+="\n첫 recovery 감사는 LoRA에 없는 P/E·D stage 열까지 요구해 거절되었으며 [실패 감사](INDEPENDENT_INITIAL_FAMILY_RECOVERY_AUDIT_ROOT_v1.json)를 그대로 보존했다. 새 감사는 실제 LoRA4열/B4 6열 schema와 같은50trial·F counter·정밀 요약을 검증했다. 없는 stage 값을0으로 만들거나 비용을 다시 측정하지 않았다.\n"
    first,rest=texts[OUTPUTS[1]].split("\n",1);texts[OUTPUTS[1]]=first+"\n\n"+title+"\n"+rest
    if not inputs.preserved():raise ValueError("Recovery report inputs changed")
    bindings=dict(schema="INITIAL_FAMILY_REPORT_BINDINGS_v1",status="FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_RECOVERY_AUDIT",audit_relative_path=audit_path.relative_to(public).as_posix(),audit_sha256=audit_sha,auditor_source_sha256=auditor_sha,builder_source_sha256=builder_sha,input_SHA_before_after_equal=True,source48_preserved=True,inputs=[dict(root=root,relative_path=name,sha256=want) for (root,name),want in sorted(inputs.files.items())],outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],new_accuracy_conditions=2,new_cost_conditions=4,recovery_provenance=provenance,matched_training_activity=activity,whole_MASTER_complete=False,independent_generalization_claimed=False,new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0)
    texts[OUTPUTS[2]]=json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
    return dict(status="REPORT_CREATED_FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_RECOVERY_AUDIT",files_written=3,input_count=len(inputs.files))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","audit-path","audit-sha","auditor-sha","builder-sha"):p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,key) is None for key in ("public_root","cache_root","audit_path","audit_sha","auditor_sha","builder_sha")):p.error("ROOT current source and actual complete recovery audit SHAs required")
    print(json.dumps(execute(v.public_root,v.cache_root,v.audit_path,v.audit_sha,v.auditor_sha,v.builder_sha)));return 0


if __name__=="__main__":raise SystemExit(main())
