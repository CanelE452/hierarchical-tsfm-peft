"""Publication-only filesystem snapshot; no scientific runtime or assets loaded."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXPECTED_REPORT_MANIFEST = '1573f4ee628fc5598e41178457937c86137198bb4007b11f3790e2659386c6d2'
ALLOWED = {'.py', '.ps1', '.md', '.json', '.csv', '.png', '.svg'}
LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)]+)\)')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def selected_files():
    paths = {p for p in BASE.iterdir() if p.is_file() and 'matched' in p.name.lower() and p.suffix in ALLOWED}
    for folder in (BASE/'report_matched_readout_v2', BASE/'matched_readout_resources_v1'):
        paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ALLOWED and '__pycache__' not in p.parts)
    paths.update(p for p in (BASE/'validation_v1').iterdir() if p.is_file() and 'matched' in p.name.lower() and p.suffix in ALLOWED)
    paths.update(p for p in (BASE/'report_matched_readout_v1').iterdir() if p.is_file() and p.suffix in {'.json','.csv'})
    paths.add(ROOT/'RESEARCH_RESULTS_20261014_KO.md')
    paths.add(ROOT/'_docs/history/2026-10-14.md')
    return paths


def local_target(source, target):
    if target.startswith(('http:','https:','#','mailto:','codex:')):
        return None
    return (source.parent/target.split('#',1)[0]).resolve()


def build():
    canonical = BASE/'report_matched_readout_v2/MANIFEST.json'
    if sha(canonical) != EXPECTED_REPORT_MANIFEST:
        raise RuntimeError('FROZEN_MATCHED_REPORT_MANIFEST_CHANGED')
    selected = selected_files()
    originals = []
    folder = HERE/'reports'
    folder.mkdir(exist_ok=True)
    for stem, source in (('matched',BASE/'report_matched_readout_v2/REPORT_KO.md'),
                         ('interpretation',BASE/'MATCHED_READOUT_INTERPRETATION_KO_v1.md')):
        destination = folder/(stem+'_REPORT_KO.md')
        if destination.exists():
            raise FileExistsError(destination)
        original_sha = sha(source)
        def rewrite(match):
            image, label, target = match.groups()
            path = local_target(source,target)
            if path is None:
                return match.group(0)
            exists = path in selected or (path.is_dir() and any(p.is_relative_to(path) for p in selected))
            if exists:
                return image+'['+label+']('+Path(os.path.relpath(path,destination.parent)).as_posix()+')'
            return label+' — 로컬/후속 범위 자료 `'+(rel(path) if path.is_relative_to(ROOT) else str(path))+'`'
        body = LINK.sub(rewrite,source.read_text(encoding='utf-8-sig'))
        note = '> 게시용 링크 복사본입니다. 원수치·본문과 봉인된 원본은 유지하고 링크만 정리했습니다. 원문 SHA256 `'+original_sha+'`. 원 보고서의 미게시·승인 대기 문구는 작성 당시 기록입니다. 현재 게시 범위는 [최신 색인](../README_KO.md)을 따릅니다.\n\n'
        destination.write_text(note+body,encoding='utf-8',newline='\n')
        if sha(source)!=original_sha:
            raise RuntimeError('ORIGINAL_REPORT_CHANGED')
        originals.append({'path':rel(source),'sha256':original_sha,'publication_copy':rel(destination)})
    text = '''# Matched-readout 완료 결과 main 게시

[최신 보고서·그림 6개](reports/matched_REPORT_KO.md) · [현재 판단과 다음 질문](reports/interpretation_REPORT_KO.md) · [이전 완료 단계 전체](../publication_main_20261014_v1/README_KO.md)

RAW33C와 E512K의 신규 8 fits·56 ridge 후보, FIT/TUNE 후보 예측112개, 선택 예측36개 및 기존 E33K N2 계수4개 재사용을 완료했습니다. 모든 E/D/F는 새로 load·forward·update하지 않았으며, cached 과거 E와 원 과거 C만 사용했습니다.

```text
두 seed 평균 정규화 MSE     Robin        Beijing
RAW-F                      0.34040189   0.58957541
MAIN                       0.56954654   0.66195720
E33K                       0.92409283   0.73662621
RAW33C                     0.69578854   0.74559667
E512K                      0.90270308   0.75732091
```

시간 요약을 없앤 E512K는 Robin에서 소폭 좋아지고 Beijing에서 나빠졌습니다. RAW33C도 RAW-F/MAIN 격차를 해결하지 못했습니다. 입력 차원·계수 수가 다르므로 encoder 단독 효과를 확정하지 않습니다. 반복 VAL30·노출 EVAL146·두 E seed이며 독립 일반화 검증이 아닙니다.

![실제 평가 오차](../report_matched_readout_v2/figures/01_eval_seed_dots.png)

![입력·계수 용량](../report_matched_readout_v2/figures/04_capacity_rank.png)

독립 저장 배열 감사 v2는 PASS/실패0, 최종 파일·링크·CSV 감사는726/0입니다. 이전 감사 v1의 shape/count 오류와 실패 기록도 보존합니다. E/F 순전파·SVD 최적성·새 배포 지연을 독립 재실행한 검사는 아닙니다.

[전체 원수치 CSV](../report_matched_readout_v2/METRICS.csv) · [56 후보](../report_matched_readout_v2/ALPHA_GRID.csv) · [독립 감사](../validation_v1/MATCHED_READOUT_AUDIT_v2_ACTUAL.json) · [최종 726 검사](../validation_v1/MATCHED_REPORT_FINAL_VERIFICATION_v3.json)

실험·검산 코드, 고정 조건, 실제 producer JSON 사본·13 terminal resource receipt, CSV와 PNG/SVG를 게시합니다. NPZ·원데이터·모델·checkpoint와 새 MAIN residual 실험의 준비/실행 소스는 이 snapshot에 포함하지 않습니다. JSON 내부의 캐시·Windows 경로는 로컬 재현 계보이며 GitHub에 존재하는 파일 링크가 아닙니다. 연구 정확도 목표는 미해결이며 다음 승인 범위의 실행은 계속됩니다.

[게시 파일·SHA](FILES.json) · [링크 확인](LINK_AUDIT.json) · [원 보고서 SHA](ORIGINAL_REPORTS.json)
'''
    (HERE/'README_KO.md').write_text(text,encoding='utf-8',newline='\n')
    (HERE/'ORIGINAL_REPORTS.json').write_text(json.dumps(originals,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'FILES.json').write_text('{}\n',encoding='utf-8')
    (HERE/'LINK_AUDIT.json').write_text('{}\n',encoding='utf-8')
    selected.update(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ALLOWED)
    checks=[];failures=[]
    for source in [HERE/'README_KO.md',ROOT/'RESEARCH_RESULTS_20261014_KO.md']+list(folder.glob('*.md')):
        for image,label,target in LINK.findall(source.read_text(encoding='utf-8-sig')):
            path=local_target(source,target)
            if path is None:continue
            tracked=path.is_relative_to(ROOT) and subprocess.run(['git','ls-files','--error-unmatch',rel(path)],capture_output=True).returncode==0
            okay=(path.is_file() and (path in selected or tracked)) or (path.is_dir() and any(p.is_relative_to(path) for p in selected))
            row={'source':rel(source),'target':target,'is_image':bool(image),'pass':okay};checks.append(row)
            if not okay:failures.append(row)
    audit={'status':'PASS' if not failures else 'FAIL','checks':len(checks),'image_links':sum(r['is_image'] for r in checks),'failed_count':len(failures),'links':checks,'failures':failures}
    (HERE/'LINK_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records=[{'path':rel(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(selected) if p!=HERE/'FILES.json']
    manifest={'status':'READY_FOR_EXPLICIT_STAGING' if not failures else 'FAIL','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'excluded_new_MAIN_residual':True,'files':records,'file_count_excluding_manifest':len(records),'bytes':sum(r['bytes'] for r in records)}
    (HERE/'FILES.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'files':len(records)+1,'bytes':manifest['bytes'],'links':len(checks),'images':audit['image_links'],'failures':len(failures)}))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    build()
