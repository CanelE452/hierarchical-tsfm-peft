"""Render only the pinned saved-V group statistics; no model or data access."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
START = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = 'patterns/snapshot_20261015T093749Z_1792057069715991800'
EVAL = SNAPSHOT + '/evaluation_20261015T145939Z_1792076379333866900'
PINS = {
    EVAL + '/GROUP_ERRORS.csv': '382f357bf313b28b6308120c1dfe1389cd31105be12f607cec36ce2255497df4',
    EVAL + '/PAIRED_GROUP_DIFFERENCES.csv': '4fd69684a562620662b67a4ad340094cc674efcdfb21c74027daeb38b3aa19be',
    EVAL + '/EVALUATION_MANIFEST.json': '3772a6781249ca4fc2cbeb57230c074d32d8adb2b8d88672dd9ad655e1901780',
    SNAPSHOT + '/PATTERN_MANIFEST.json': '40e329c051b0f843a01007100af2e5ee072650ae8e3ddc52632ee0ddd74bf52b',
}
DATASETS = [('beijing', 'Beijing', 3), ('jena', 'Jena', 6),
            ('electricity', 'Electricity', 81), ('pems08', 'PeMS08', 43)]
PATTERNS = [('variability', 'Variability'), ('seasonality', 'Seasonality'),
            ('delayed_crosschannel', 'Lagged cross-channel\ncorrelation'),
            ('intermittency', 'Observed zero fraction')]
GROUPS = ['low', 'high', 'NA']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / 'reproduce.py'
    if not script_path.exists():
        script_path.write_bytes(Path(__file__).read_bytes())
    elif script_path.resolve() != Path(__file__).resolve():
        raise FileExistsError('Refusing to overwrite an existing reproduction script')
    expected_outputs = ['EXACT_GROUP_DELTAS.csv', 'MSE.png', 'MSE.svg', 'MAE.png',
                        'MAE.svg', 'README_KO.md', 'MANIFEST.json']
    if any((out / name).exists() for name in expected_outputs):
        raise FileExistsError('Refusing to overwrite a figure or result; use a new --output directory')
    if shutil.disk_usage(out).free < 100 * 1024**2:
        raise RuntimeError('Less than the requested 100 MiB pending storage headroom')
    for rel, expected in PINS.items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Pinned input changed: ' + rel)
    pair_rows = load_csv(ROOT / EVAL / 'PAIRED_GROUP_DIFFERENCES.csv')
    groups = load_csv(ROOT / EVAL / 'GROUP_ERRORS.csv')
    group_index = {(g['run_id'], g['pattern'], g['group']): g for g in groups}
    evaluation = json.loads((ROOT / EVAL / 'EVALUATION_MANIFEST.json').read_text())
    evaluated_inputs = {v['file']: v['sha256'] for v in evaluation['input_files'] if v['scope'] == 'public'}
    panels, exact = [], []
    for dataset, label, k in DATASETS:
        left = f'primary_{dataset}_E1_B2_k{k}_s92601'
        right = f'primary_{dataset}_E1_B0_k{k}_s92601'
        selected = [r for r in pair_rows if r['left_run_id'] == left and r['right_run_id'] == right]
        panel = dict(dataset=dataset, label=label, k=k, left_run_id=left, right_run_id=right,
                     status='MISSING', reason='', rows=[], prediction_bindings=[])
        if len(selected) != 12:
            panel['reason'] = f'Exact initial pair has {len(selected)} rows; expected 12; no substitution'
            panels.append(panel)
            continue
        assert {(r['pattern'], r['group']) for r in selected} == {(p, g) for p, _ in PATTERNS for g in GROUPS}
        for r in selected:
            assert r['comparison'] == 'B2 - B0' and r['dataset'] == dataset
            assert r['fold'] == 'primary' and r['split'] == 'V_select'
            assert r['left_seed'] == r['right_seed'] == '92601'
            assert r['left_k'] == r['right_k'] == str(k)
            assert r['left_budget'] == r['right_budget'] == 'A:0.25'
            assert r['left_lr'] == r['right_lr'] == '0.001'
        manifests = []
        for run_id in (left, right):
            rel = f'predictions/{run_id}/V_select/PREDICTION_MANIFEST.json'
            path = ROOT / rel
            assert evaluated_inputs[rel] == sha(path), 'Saved V manifest changed since group evaluation'
            m = json.loads(path.read_text())
            assert m['run_id'] == run_id and m['split'] == 'V_select'
            manifests.append(m)
            panel['prediction_bindings'].append(dict(file=rel, sha256=sha(path),
                model_sha=m['model_sha'], archive_sha256=m['archive_sha256'],
                artifact_sha256=m['metadata']['artifact_sha256'],
                reference_arrays={n: m['arrays'][n] for n in ('target_normalized', 'observed', 'origins', 'channel_ids')}))
        lm, rm = manifests
        assert lm['cohort'] == rm['cohort'] == selected[0]['cohort']
        for name in ('data_sha', 'scaler_sha'):
            assert lm[name] == rm[name]
        for name in ('target_normalized', 'observed', 'origins', 'channel_ids'):
            assert lm['arrays'][name] == rm['arrays'][name]
        panel.update(status='MATCHED', H=lm['arrays']['pred_normalized']['shape'][1],
                     origins=lm['arrays']['pred_normalized']['shape'][0],
                     channels=lm['arrays']['pred_normalized']['shape'][2], cohort=lm['cohort'],
                     data_sha=lm['data_sha'], scaler_sha=lm['scaler_sha'])
        for r in selected:
            l = group_index[(left, r['pattern'], r['group'])]
            b = group_index[(right, r['pattern'], r['group'])]
            for field in ('cohort', 'data_sha', 'scaler_sha', 'metric_scale', 'active_channels', 'observed_target_cells'):
                assert l[field] == b[field]
            assert l['family'] == 'B2' and b['family'] == 'B0'
            assert l['fit_config_sha'] == r['left_fit_config_sha']
            assert b['fit_config_sha'] == r['right_fit_config_sha']
            record = dict(r)
            record.update(left_mse=l['mse'], right_mse=b['mse'], left_mae=l['mae'], right_mae=b['mae'],
                          left_selected_epoch=l['selected_epoch'], right_selected_epoch=b['selected_epoch'],
                          metric_scale=l['metric_scale'], horizon=str(panel['H']), origins=str(panel['origins']),
                          observed_mask_sha256=lm['arrays']['observed']['sha256'],
                          target_sha256=lm['arrays']['target_normalized']['sha256'])
            exact.append(record)
            panel['rows'].append(record)
        panel['selected_epochs'] = [panel['rows'][0]['left_selected_epoch'], panel['rows'][0]['right_selected_epoch']]
        panels.append(panel)
    if exact:
        with (out / 'EXACT_GROUP_DELTAS.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exact[0]))
            writer.writeheader()
            writer.writerows(exact)
    else:
        raise RuntimeError('No exact pairs; no scores or figures fabricated')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'axes.titlesize': 12, 'savefig.dpi': 300})
    cmap = LinearSegmentedColormap.from_list('blue_white_orange', ['#0072B2', '#FFFFFF', '#D55E00'])
    cmap.set_bad('#E0E0E0')
    for metric in ('mse', 'mae'):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
        fig.subplots_adjust(left=.17, right=.965, top=.845, bottom=.23, wspace=.75, hspace=.80)
        fig.suptitle(f'Frozen channel-pattern groups: B2 - B0 ({metric.upper()})', fontsize=17, fontweight='bold', y=.98)
        fig.text(.5, .925, 'Initial A:0.25 search budget | learning rate 0.001 | seed 92601 | primary V_select', ha='center', fontsize=11)
        for letter, ax, panel in zip('ABCD', axes.flat, panels):
            if panel['status'] != 'MATCHED':
                ax.set_facecolor('#E0E0E0')
                ax.text(.5, .5, panel['reason'], ha='center', va='center', wrap=True, transform=ax.transAxes)
                ax.set_title(f"{letter}  {panel['label']} | exact pair unavailable")
                continue
            lookup = {(r['pattern'], r['group']): r for r in panel['rows']}
            matrix = np.full((4, 3), np.nan)
            for i, (pattern, _) in enumerate(PATTERNS):
                for j, group in enumerate(GROUPS):
                    val = lookup[(pattern, group)]['delta_' + metric]
                    if val:
                        matrix[i, j] = float(val)
            finite = matrix[np.isfinite(matrix)]
            max_abs = max(abs(finite.min()), abs(finite.max())) if finite.size else 0.
            scale = float(max_abs) if max_abs else 1.
            image = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), aspect='auto')
            ax.set_xticks(range(3), ['Low', 'High', 'NA'])
            ax.set_yticks(range(4), [label for _, label in PATTERNS])
            ax.tick_params(length=0, pad=6)
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which='minor', color='#999999', linewidth=.55)
            ax.tick_params(which='minor', bottom=False, left=False)
            for i, (pattern, _) in enumerate(PATTERNS):
                for j, group in enumerate(GROUPS):
                    row = lookup[(pattern, group)]
                    value = matrix[i, j]
                    text = f'{value:+.4g}\nn={row["active_channels"]}' if math.isfinite(value) else f'No active channels\nn={row["active_channels"]}'
                    ink = 'white' if math.isfinite(value) and abs(value) > scale * .70 else '#111111'
                    ax.text(j, i, text, ha='center', va='center', color=ink, fontsize=9 if math.isfinite(value) else 8)
            epochs = '/'.join(panel['selected_epochs'])
            note = 'Both selected epoch 0; all deltas 0' if max_abs == 0 else f'Selected epoch B2/B0: {epochs}'
            ax.set_title(f"{letter}  {panel['label']} | K={panel['k']} | H={panel['H']}\nV origins={panel['origins']}, channels={panel['channels']}", loc='left', pad=10)
            ax.text(.5, -.22, note, transform=ax.transAxes, ha='center', va='top', fontsize=9)
            if max_abs:
                cb = fig.colorbar(image, ax=ax, fraction=.05, pad=.04)
                cb.locator = MaxNLocator(nbins=5, symmetric=True)
                cb.update_ticks()
                cb.ax.tick_params(labelsize=8)
                cb.set_label('Delta ' + metric.upper(), fontsize=9)
            else:
                ax.text(1.045, .5, 'Zero\ndelta', transform=ax.transAxes, va='center', fontsize=9)
        fig.text(.5, .075, 'Negative (blue): B2 lower error. Positive (orange): B2 higher error. Gray: empty group; no value imputed.\n'
                 'n = active channels; equal-channel group mean. Each nonzero panel has its own symmetric scale; no cross-dataset pooling.\n'
                 'Matched whole-method comparison, not P-only causation, demonstrated convergence, or independent generalization.',
                 ha='center', va='center', fontsize=9, linespacing=1.7)
        fig.savefig(out / (metric.upper() + '.png'), dpi=300, metadata={'Software': 'Saved V pattern reproducibility script'})
        fig.savefig(out / (metric.upper() + '.svg'))
        plt.close(fig)
    readme = '''# 초기 quarter-K B2−B0의 고정 채널 패턴별 차이

[MSE PNG](MSE.png) · [MSE SVG](MSE.svg) · [MAE PNG](MAE.png) · [MAE SVG](MAE.svg) · [반올림하지 않은 원수치 CSV](EXACT_GROUP_DELTAS.csv) · [재현 스크립트](reproduce.py) · [결박 manifest](MANIFEST.json)

![MSE: B2 minus B0](MSE.png)
![MAE: B2 minus B0](MAE.png)

[확인] 초기 primary E1 B2/B0의 Beijing K=3, Jena K=6, Electricity K=81, PeMS08 K=43만 선택했다. seed=92601, 등록 search budget A:0.25, learning rate=0.001이 같고, 실제 left/right ID를 고정해 후속 G1·장기학습 결과를 포함하지 않았다. 타깃·관측 mask·origin·channel ID의 배열 SHA, cohort, data SHA와 scaler SHA가 각 쌍에서 일치하는 공개 V manifest를 확인했다. 이 그림을 만들 때 배열 파일은 열지 않았다.

CSV의 delta_mse/delta_mae 문자열과 활성 채널수는 기존 PAIRED_GROUP_DIFFERENCES.csv에서 반올림 없이 복사했다. 왼쪽/오른쪽 group MSE·MAE, 실제 선택 epoch, H, V origin 수, mask SHA를 함께 보존했다. delta는 FP64 채널별 오차 차이의 균등평균이므로 CSV의 두 group 평균을 다시 빼지 않았다. 새 재평가·추론·학습이 아닌 저장 V의 기술 분석이다.

각 패널의 low/high/NA는 A_fit에만 적합해 고정한 네 패턴의 그룹이다. n은 V에서 적어도 한 관측 타깃이 있는 활성 채널수이고 독립 반복수나 독립 표본수가 아니다. 48셀 중 16개의 NA는 빈 그룹이며 수치를 0으로 대체하지 않았다. Low/high 채널은 서로 겹치지 않지만 같은 채널이 서로 다른 패턴 행에 반복된다. Observed zero fraction은 간헐 수요를 증명하지 않는다. Jena의 legacy 정규화 자료에서 역변환 FP32의 물리적 exact-zero 보존은 제한된다.

Electricity는 두 방법 모두 선택 epoch 0.0이며 모든 차이가 실제 0이다. 나머지는 양쪽 선택 epoch 0.25이다. 이는 등록 budget이 같다는 사실과 별개이며, Electricity 결과로 학습된 B2의 효과가 없다고 단정할 수 없다. 실제 update 수는 원 group CSV에서 비어 있어 추정해 채우지 않았다.

음수는 B2의 오차 감소, 양수는 증가다. 0 중심의 색각 안전 청색/주황 팔레트와 부호·원수치 주석을 함께 써 회색조에서도 판독 가능하게 했다. 0이 아닌 패널은 각자 대칭 색 범위를 쓰고 Electricity는 색 차이가 없어 별도 색 막대를 생략했다. 자료별 H·정규화·단위가 달라 자료 간 평균이나 색 강도 비교를 하지 않는다. 불확실성 구간이나 유의성 별표는 산출하지 않았다.

같은 recipe/K/cohort/mask 아래 전체 B2와 B0의 차이다. P만의 인과효과, 충분 수렴, 독립적인 일반화 검증으로 해석하면 안 된다. 단일 seed의 초기 짧은 search budget이라는 한계가 있다. 이 파일은 뒤에 도착하는 deep V를 자동 포함하지 않는다.

재현: 이 폴더의 reproduce.py를 원 campaign R 위치에서 실행한다. 기본값은 기존 산출물 덮어쓰기를 거부하므로 새 폴더를 --output으로 지정한다. 입력은 manifest에 결박된 공개 saved-V group CSV와 JSON metadata뿐이며 모델·원자료·F·T에 접근하지 않는다. CUDA=-1, CPU thread=2를 설정한다. PNG는 3840×3000, 300 dpi이고 SVG도 함께 제공한다.
'''
    (out / 'README_KO.md').write_text(readme, encoding='utf-8')
    forbidden = [n for n in sys.modules if n.split('.')[0] in {'torch', 'chronos', 'transformers', 'tensorflow', 'jax'}]
    assert not forbidden
    output_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert output_bytes < 100 * 1024**2
    assert time.monotonic() - START < 600
    manifest = dict(purpose='Saved V descriptive group-difference figures only',
        input_files=[dict(file=rel, sha256=value) for rel, value in PINS.items()],
        panels=panels, exact_pairs=sum(p['status'] == 'MATCHED' for p in panels),
        numerical_rows=len(exact), empty_cells=sum(r['active_channels'] == '0' for r in exact),
        source_script_sha256=sha(Path(__file__)), selected_v_only=True, arrays_opened=False,
        model_imports=forbidden, GPU_execution=False, inference_reexecuted=False,
        pattern_fitting_performed=False, statistical_significance_claimed=False,
        independent_generalization_claimed=False, P_only_causal_claimed=False,
        across_dataset_pooling=False, palette='Okabe-Ito blue-white-orange; signed text redundant encoding',
        normalization='A_normalized_original_channels; panel-specific scales',
        runtime_environment={name: os.environ[name] for name in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        software=dict(python=sys.version.split()[0], matplotlib=matplotlib.__version__, numpy=np.__version__),
        wall_seconds=time.monotonic() - START, output_bytes_before_manifest=output_bytes,
        outputs=[dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), pairs=manifest['exact_pairs'], rows=len(exact),
                         empty_cells=manifest['empty_cells'], bytes=sum(p.stat().st_size for p in out.iterdir()),
                         wall_seconds=manifest['wall_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
