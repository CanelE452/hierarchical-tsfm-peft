"""Record the followup addendum before learning and verify its local commit."""
import argparse
import csv
import hashlib
import re
import subprocess

from . import common


def git(*args):
    return subprocess.check_output(["git", *args], cwd=common.REPO)


def source_hashes():
    return [{"path": p.relative_to(common.REPO).as_posix(), "sha256": common.sha(p)}
            for p in sorted((common.REPO / "research" / common.NAME).glob("*.py"))]


def no_learning():
    if (common.REPORT / "EXECUTION_CLOCK.json").exists():
        raise RuntimeError("Learning clock already exists")
    for job in common.selected_jobs():
        for root in (common.OLD_REPORT / "jobs", common.OLD_CACHE / "jobs"):
            if (root / job["job_id"]).exists():
                raise RuntimeError(f"Pretraining output collision: {root / job['job_id']}")
    common.guard_preserved()


def csv_rows(name):
    with (common.REPORT / "preflight" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write():
    no_learning()
    validation = common.read(common.REPORT / "VALIDATION_PRETRAINING.json")
    if validation["status"] != "PASS":
        raise RuntimeError("Pretraining validation incomplete")
    instruction = common.SOURCE_DOCUMENT.read_text(encoding="utf-8-sig")
    headings = list(re.finditer(r"^## (\d+)\. .*$", instruction, re.MULTILINE))
    sections = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(instruction)
        if int(match.group(1)) in (0, 3, 5, 6):
            sections[match.group(1)] = instruction[match.start():end]
    if set(sections) != {"0", "3", "5", "6"}:
        raise RuntimeError("Required instruction sections absent")
    budget = common.read(common.REPORT / "BUDGET.json")
    independent = common.read(common.REPORT / "preflight" / "D0_D1_INDEPENDENT_CHECK.json")
    if independent["status"] != "PASS":
        raise RuntimeError("Independent subspace verification incomplete")
    jobs = common.selected_jobs()
    inputs = [common.REPORT / name for name in (
        "BUDGET.json", "EXECUTION_PLAN.json", "PRESERVATION_BASELINE.json",
        "ENVIRONMENT_VALIDATION.json", "VALIDATION_PRETRAINING.json")]
    inputs += sorted((common.REPORT / "preflight").glob("*"))
    inputs = [{"path": p.relative_to(common.REPO).as_posix(), "sha256": common.sha(p)}
              for p in inputs if p.is_file()]
    d0 = csv_rows("D0_SUBSPACE_FLOOR.csv")
    d1 = csv_rows("D1_DECODER_FLOOR_SUMMARY.csv")
    scope = ("The subset was chosen after inspecting the first campaign results, to fill the D-whole-structure "
             "ablation gap and the provisional Jena No-P result under the available time. Job definitions, "
             "seeds, original decision rules, and original training code remain fixed. This is a disclosed "
             "post-hoc scope selection, not an outcome-independent choice of which experiments to run.")
    predictions = {
        "A4_ETTh1_ETTh2": "median V MSE effect versus OURS >= +50 percent",
        "A4_jena_appliances": "absolute median V MSE effect versus OURS < 10 percent",
        "D1_ETTh1_ETTh2_A4_C2_C3": "median per-job floor/actual V MSE ratio >= 0.5",
        "A1_jena_extension": "No prediction",
        "role": "Fixed descriptive predictions, not replacement decision rules",
    }
    lines = [f"# {common.NAME} — 사전등록 추가본", "", f"작성: {common.now_kst()}", "",
             f"기준 커밋: `{common.BASE}`", "", "새 학습 전 작성 및 로컬 커밋. 원격 push는 별도 사용자 확인 후에만 수행한다.", "",
             "## 범위 선택의 시점과 해석", "", scope, "",
             "A4는 D 구조 전체의 효과이며 마지막 관측값과 lag를 분리하지 않는다. V는 개발 평가이며 보호 T 숫자를 읽지 않는다.", "",
             "## 예산의 실측 갱신", "",
             f"A4 40개: {budget['A4_40_estimated_seconds']/3600:.6f}시간. "
             f"A1/Jena 연장 10개: {budget['A1_jena_10_estimated_seconds']/3600:.6f}시간. "
             f"총 {budget['all_50_estimated_seconds']/3600:.6f}시간 [추정].", "",
             "지시문의 약 11.3시간은 이전 K_tau 프로파일의 연장 추정에 근거했다. "
             "이번에는 첫 캠페인 A1 실측 search/refit 시간과 실행 부대 시간을 반영한다. "
             "A4만으로 13시간 이내이므로 FG6은 통과한다. A4 종료 시 원래 13시간 시계와 "
             "2026-10-28 00:00 KST 마감 안에 연장 10개가 모두 끝날 것으로 추정될 때만 묶음으로 진입한다. "
             "그렇지 않으면 10개 모두 NOT_RUN_BUDGET이다. 진행 중인 job은 마친다.", "",
             "## 학습 전 D0 결과", "", "```csv"]
    lines += [(common.REPORT / "preflight" / "D0_SUBSPACE_FLOOR.csv").read_text(encoding="utf-8-sig").rstrip(),
              "```", "", "## 기존 C2/C3의 학습 전 D1 요약", "", "```csv",
              (common.REPORT / "preflight" / "D1_DECODER_FLOOR_SUMMARY.csv").read_text(encoding="utf-8-sig").rstrip(),
              "```", "", "독립 최소제곱 검산과 원 파일 보존 점검은 PASS이며 원본 CSV/검증 영수증 SHA는 JSON에 기록한다.", "",
              "## 실행 목록과 순서", "", "A4는 seed별 4개 데이터를 순회한 뒤 A1/Jena 연장 10개를 조건부 실행한다.", ""]
    lines += [f"- `{job['job_id']}`" for job in jobs]
    lines += ["", "## 지시문 원문: 0·3·5·6절", ""]
    markdown = "\n".join(lines) + "\n" + "\n".join(sections[str(n)] for n in (0, 3, 5, 6))
    record = {
        "campaign": common.NAME, "created_kst": common.now_kst(), "base_commit": common.BASE,
        "training_started": False, "push_authorized": False, "instruction_sha256": common.sha(common.SOURCE_DOCUMENT),
        "verbatim_sections": sections, "markdown": markdown, "scope_selection_disclosure": scope,
        "D0_before_A4": d0, "D1_C2_C3_before_A4": d1, "planner_predictions": predictions,
        "budget": budget, "jobs_in_order": jobs, "input_hashes": inputs,
        "new_source_hashes_at_registration": source_hashes(),
        "preservation": common.guard_preserved(),
    }
    md_path = common.REPORT / "PREREGISTRATION_ADDENDUM.md"
    json_path = common.REPORT / "PREREGISTRATION_ADDENDUM.json"
    if md_path.exists() or json_path.exists():
        raise FileExistsError("Addendum already exists; do not overwrite")
    with md_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    common.write_new(json_path, record)
    print("Addendum written; learning remains blocked pending the exact two-file local commit.")


def seal():
    no_learning()
    commit = git("rev-parse", "HEAD").decode().strip()
    expected = [f"reports/{common.NAME}/PREREGISTRATION_ADDENDUM.{ext}" for ext in ("json", "md")]
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines()
    if sorted(changed) != sorted(expected):
        raise RuntimeError("HEAD must contain exactly the two addendum files")
    files = []
    for path in expected:
        digest = hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest()
        if common.sha(common.REPO / path) != digest:
            raise RuntimeError(f"Committed addendum bytes differ: {path}")
        files.append({"path": path, "sha256": digest})
    receipt = {"commit": commit, "commit_unix": int(git("show", "-s", "--format=%ct", commit)),
               "recorded_kst": common.now_kst(), "files": files, "training_started": False, "push": False}
    common.write_new(common.REPORT / "PREREGISTRATION_COMMIT.json", receipt)
    gates = common.read(common.REPORT / "GATES.json")
    gates["FG5"] = {"status": "PASS", "commit": commit, "commit_unix": receipt["commit_unix"],
                    "training_clock_absent": True}
    gates["D0_D1"] = {"status": "PASS", "independent_evidence": "preflight/D0_D1_INDEPENDENT_CHECK.json"}
    common.atomic(common.REPORT / "GATES.json", gates)
    sources = source_hashes()
    fixed = {"common.py", "controller.py", "job_worker.py", "floors.py"}
    admission = {
        "status": "PASS", "created_kst": common.now_kst(), "commit": commit,
        "instruction_sha256": common.sha(common.SOURCE_DOCUMENT),
        "new_sources": [r for r in sources if r["path"].rsplit("/", 1)[1] in fixed],
        "all_new_sources_at_admission": sources,
        "validation": common.read(common.REPORT / "VALIDATION_PRETRAINING.json"),
        "preservation": common.guard_preserved(),
        "postprocessing_repair_policy": "Only technical report/packaging repairs with recorded before/after evidence; "
            "no change to registered training, fixed scientific definitions, admitted learning harness, or completed runs.",
    }
    common.write_new(common.REPORT / "ADMISSION.json", admission)
    print(common.validate_admission())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "seal"))
    args = parser.parse_args()
    {"write": write, "seal": seal}[args.action]()
