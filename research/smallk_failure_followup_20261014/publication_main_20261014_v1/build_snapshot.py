"""Filesystem-only publication inventory; no scientific assets or runtime execution."""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESEARCH = ROOT / 'research'
ROOTS = ('bihash_future_prediction_20261013', 'bihash_future_alignment_20261013',
         'smallk_predictive_alignment_20261013', 'smallk_failure_followup_20261014')
REPORTS = (
    ('bihash', 'Bi-Hash 미래 정렬 전체 결과', 'bihash_future_prediction_20261013/report_canonical_final_v2/REPORT_KO.md'),
    ('bihash_pilot', 'Bi-Hash 파일럿 기록', 'bihash_future_alignment_20261013/report_v1/REPORT_KO.md'),
    ('alignment', '작은 K 예측 목적 정렬·새 자료 비교', 'smallk_predictive_alignment_20261013/report_v2/REPORT_KO.md'),
    ('diagnostics', 'D0–D3 표현·예측기·복원 진단', 'smallk_failure_followup_20261014/report_followup_v3/REPORT_KO.md'),
    ('joint', 'Joint E/D 목표 및 Appliances 손실 비교', 'smallk_failure_followup_20261014/report_joint_followup_v2/REPORT_KO.md'),
    ('sequence', 'RF1/RF3 디코더·균형 순서 비용 비교', 'smallk_failure_followup_20261014/report_sequence_followup_v1/REPORT_KO.md'),
    ('stage', 'E/F/D 분리·native ridge·Oracle 진단', 'smallk_failure_followup_20261014/report_stage_separation_v2/REPORT_KO.md'),
)
EXTENSIONS = {'.py', '.ps1', '.md', '.json', '.csv', '.png', '.svg'}
LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)]+)\)')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def excluded(path):
    return ('matched' in path.name.lower() or any(p in {'__pycache__', '.pytest_cache', '.cache'} for p in path.parts)
            or path.suffix.lower() not in EXTENSIONS or path.stat().st_size >= 100 * 1024**2)


def local_target(source, target):
    target = target.strip().strip('<>')
    if target.startswith(('http:', 'https:', '#', 'mailto:', 'app:', 'codex:')):
        return None
    target = unquote(target.split('#', 1)[0].split('?', 1)[0])
    if not target:
        return None
    if re.match(r'^[A-Za-z]:[/\\]', target):
        return Path(target).resolve()
    return (source.parent / target).resolve()


def select_files():
    selected = set()
    for folder in ROOTS:
        base = RESEARCH / folder
        for path in base.iterdir():
            if path.is_file() and not excluded(path):
                selected.add(path)
        for name in ('validation_v1', 'resources', 'continuation_resources_v1', 'stage_separation_resources_v1'):
            if not (base / name).is_dir():
                continue
            for path in (base / name).rglob('*'):
                if path.is_file() and not excluded(path) and path.stat().st_size <= 6 * 1024**2:
                    selected.add(path)
    for _, _, report in REPORTS:
        folder = (RESEARCH / report).parent
        for path in folder.rglob('*'):
            if path.is_file() and not excluded(path):
                selected.add(path)
    for date in ('2026-10-13', '2026-10-14'):
        selected.add(ROOT / '_docs' / 'history' / (date + '.md'))
    pending = list(selected)
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited or source.suffix != '.md':
            continue
        visited.add(source)
        for _, _, target in LINK.findall(source.read_text(encoding='utf-8-sig')):
            path = local_target(source, target)
            if path is None or not path.is_file() or not path.is_relative_to(RESEARCH):
                continue
            if path.relative_to(RESEARCH).parts[0] not in ROOTS or excluded(path):
                continue
            if path not in selected:
                selected.add(path)
                pending.append(path)
    return selected


def build():
    selected = select_files()
    originals = []
    rewrites = []
    copies = HERE / 'reports'
    copies.mkdir(exist_ok=True)
    for key, label, report in REPORTS:
        source = RESEARCH / report
        destination = copies / (key + '_REPORT_KO.md')
        if destination.exists():
            raise FileExistsError(destination)
        before = sha(source)
        original_text = source.read_text(encoding='utf-8-sig')
        def rewrite(match):
            image, caption, target = match.groups()
            path = local_target(source, target)
            if path is None:
                return match.group(0)
            if path in selected or (path.is_relative_to(ROOT) and path.is_file() and subprocess.run(
                ['git', 'ls-files', '--error-unmatch', relative(path)], cwd=ROOT,
                capture_output=True).returncode == 0):
                new_target = Path(os.path.relpath(path, destination.parent)).as_posix()
                rewrites.append({'report': key, 'target': target, 'published_target': relative(path)})
                return image + '[' + caption + '](' + new_target + ')'
            literal = relative(path) if path.is_relative_to(ROOT) else str(path)
            rewrites.append({'report': key, 'target': target, 'local_only': literal})
            return caption + ' — 로컬 재현 자산 `' + literal + '`'
        copied = LINK.sub(rewrite, original_text)
        note = ('> 게시용 링크 복사본입니다. 아래 원수치·본문은 원 보고서를 유지하고 링크 경로만 정리했습니다. '
                '데이터·모델·NPZ는 로컬 재현 자산입니다. [봉인된 원본](' +
                Path(os.path.relpath(source, destination.parent)).as_posix() + ') SHA256 `' + before + '`.\n\n')
        destination.write_text(note + copied, encoding='utf-8', newline='\n')
        if sha(source) != before:
            raise RuntimeError('ORIGINAL_REPORT_CHANGED')
        originals.append({'key': key, 'label': label, 'path': relative(source), 'sha256': before,
                          'publication_copy': relative(destination)})
    index = HERE / 'README_KO.md'
    body = ['# 완료된 연구 결과 main 게시 색인', '',
        '게시 범위는 Bi-Hash 미래 정렬, 작은 K 예측 목적 정렬, D0–D3 진단, joint E/D·Appliances 손실, RF1/RF3, E/F/D 분리 진단까지입니다. '
        '새 matched-readout 단계는 승인 후 실행 준비 중이며 이번 코드·결과 snapshot에 포함하지 않습니다.', '',
        '**연구 목적이 해결됐다는 뜻은 아닙니다.** 현재 작은 K 경로는 RAW 대비 정확도 격차가 남고, '
        '반복 개발 VAL·노출 EVAL·소수 seed의 제한이 있습니다. Oracle는 실제 미래를 쓰는 진단이며 배포 방법이 아닙니다. '
        'ExoTimer 전체 문장·LLM 시스템을 재현한 결과가 아닙니다.', '',
        '## 보고서와 근거', '']
    for record in originals:
        copy = Path(os.path.relpath(ROOT / record['publication_copy'], HERE)).as_posix()
        original = Path(os.path.relpath(ROOT / record['path'], HERE)).as_posix()
        body.append('- [' + record['label'] + '](' + copy + ') · [봉인 원본](' + original + ')')
    body += ['', '## 최신 핵심 결과', '',
        '```text', '두 seed 평균 정규화 MSE       Robin       Beijing',
        'RAW                         0.340402    0.589575',
        'MAIN                        0.569547    0.661957',
        'N1: 미래 latent 예측 + D    0.914009    0.727793',
        'N2: 과거 E → 미래 C         0.924093    0.736626',
        'N3: 고정 N1 + D 보정        0.946790    0.737978', '```', '',
        'N2는 persistence·FIT 평균 보정보다 좋지만 RAW·MAIN보다 나쁩니다. '
        '현재 hybrid native B1 비용은 약2.66–3.93ms, 같은 측정의 RAW는48.30–51.64ms입니다. '
        '정확도 목표는 미충족이며 이전 측정과 RAW 절대지연이 달라 안정적인 범용 속도 우위를 주장하지 않습니다.', '',
        '![최신 deploy 오차](../report_stage_separation_v2/figures/01_deploy_mse_mae_dots.png)', '',
        '![실제 whole 배포 비용](../report_stage_separation_v2/figures/06_b1_b8_paired_cost.png)', '',
        '## 게시·재현 범위', '',
        '실험·검산 소스, 고정 조건, 최신 보고서·CSV·PNG/SVG, 선택한 terminal/실패 감사와 resource receipt를 게시합니다. '
        '원데이터, NPZ 예측 배열, 모델 가중치·checkpoint, 로그, 실행 중 matched-readout 소스와 오래된 무관한 변경은 제외합니다. '
        '중복 대형 snapshot 전체를 게시하지 않습니다. 원 보고서/manifest/그림의 byte SHA는 바꾸지 않았습니다.', '',
        '대형 JSON/CSV는 GitHub 기본 미리보기 한도를 넘을 수 있습니다. Raw 또는 다운로드로 확인할 수 있으며 이미지·Markdown 링크는 일반 Git blob입니다. '
        'JSON 내부의 `.cache`·Windows 절대경로는 원 실행 계보 기록입니다. 해당 파일이 GitHub에 있는 것으로 해석하면 안 됩니다.', '',
        '[게시 파일·SHA 목록](FILES.json) · [링크 점검](LINK_AUDIT.json) · [원 보고서 SHA](ORIGINAL_REPORTS.json)', '']
    index.write_text('\n'.join(body), encoding='utf-8')
    (HERE / 'ORIGINAL_REPORTS.json').write_text(json.dumps(originals, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE / 'LINK_REWRITES.json').write_text(json.dumps(rewrites, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE / 'FILES.json').write_text('{}\n', encoding='utf-8')
    (HERE / 'LINK_AUDIT.json').write_text('{}\n', encoding='utf-8')
    selected.update(path for path in HERE.rglob('*') if path.is_file() and not excluded(path))
    for path in selected:
        if 'matched' in path.name.lower():
            raise RuntimeError('MATCHED_SOURCE_INCLUDED')
    links, failures = [], []
    for source in [index] + [ROOT / r['publication_copy'] for r in originals]:
        for image, caption, target in LINK.findall(source.read_text(encoding='utf-8')):
            path = local_target(source, target)
            if path is None:
                continue
            exists = path.is_file()
            published = path in selected or (exists and subprocess.run(['git','ls-files','--error-unmatch',relative(path)],
                cwd=ROOT, capture_output=True).returncode == 0)
            record = {'source': relative(source), 'caption': caption, 'target': target,
                      'is_image': bool(image), 'exists': exists, 'in_publication_or_HEAD': published}
            links.append(record)
            if not exists or not published:
                failures.append(record)
    audit = {'status': 'PASS' if not failures else 'FAIL', 'checks': len(links), 'failed_count': len(failures),
             'image_links': sum(r['is_image'] for r in links), 'failures': failures,
             'scope': 'publication-copy Markdown local files and Git selection; external sites not independently checked', 'links': links}
    (HERE / 'LINK_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    selected.add(HERE / 'LINK_AUDIT.json')
    records = [{'path': relative(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(selected) if path != HERE / 'FILES.json']
    manifest = {'status': 'READY_FOR_EXPLICIT_STAGING' if not failures else 'FAILED_LINK_AUDIT',
                'source_HEAD': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'excluded_matched_readout': True, 'excluded_raw_models_arrays_logs': True,
                'file_count_excluding_this_manifest': len(records), 'bytes': sum(r['bytes'] for r in records), 'files': records}
    (HERE / 'FILES.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('status','file_count_excluding_this_manifest','bytes')}))
    print(json.dumps({'link_checks': audit['checks'], 'image_links': audit['image_links'], 'failures': len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    build()
