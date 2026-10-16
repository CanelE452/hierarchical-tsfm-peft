"""Strict metadata matching and saved channel-error arithmetic only."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
SNAP = 'patterns/snapshot_20261015T093749Z_1792057069715991800'
EVAL = SNAP + '/evaluation_20261015T145939Z_1792076379333866900'
AUDIT = 'audits/audit_1792075048909619000'
DATASETS = [('beijing', 'Beijing', 3), ('jena', 'Jena', 6), ('electricity', 'Electricity', 81), ('pems08', 'PeMS08', 43)]
PATTERNS = ['variability', 'seasonality', 'delayed_crosschannel', 'intermittency']
START = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def same(a, b, field):
    if a != b:
        raise ValueError('Mismatch: ' + field)


def require(condition, field):
    if not condition:
        raise ValueError('Required condition failed: ' + field)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HOME)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'RESULT.json').exists(), 'new output directory')
    require(shutil.disk_usage(out).free > 100 * 1024**2, '100MiB free headroom')
    if out != HOME:
        for name in ['reproduce.py', 'INPUT_PINS.json']:
            with (out / name).open('xb') as stream:
                stream.write((HOME / name).read_bytes())
    pins = json.loads((HOME / 'INPUT_PINS.json').read_text(encoding='utf-8'))
    for rel, expected in pins.items():
        same(sha(ROOT / rel), expected, 'input SHA ' + rel)
    same(pins[EVAL + '/CHANNEL_ERRORS.csv.gz'], '91b78fe5eda9d24a12c1a399b66bf965c6fe319bedf8426ecad4380f50d5eaee', 'gzip pin')
    restored = gzip.decompress((ROOT / EVAL / 'CHANNEL_ERRORS.csv.gz').read_bytes())
    same(hashlib.sha256(restored).hexdigest(), 'a298855454ab89f75dca84182336f542ec1a9d4669459069d478fdf805c0eacb', 'lossless restored CSV SHA')
    audit_report = load(AUDIT + '/AUDIT_REPORT.json')
    same(audit_report['prediction_rescoring_passed'], 366, '366 audit count')
    audit_records = load(AUDIT + '/PREDICTION_AUDITS.json')
    audit_index = {(x['run_id'], x['split']): x for x in audit_records}
    evaluated_inputs = {x['file']: x['sha256'] for x in load(EVAL + '/EVALUATION_MANIFEST.json')['input_files'] if x['scope'] == 'public'}
    assets = load('DATA_CONTRACT.json')['assets']
    pattern_rows = list(csv.DictReader((ROOT / SNAP / 'CHANNEL_PATTERNS.csv').open(encoding='utf-8-sig', newline='')))
    pattern_index = {(x['dataset'], x['channel']): x for x in pattern_rows if x['fold'] == 'primary'}
    ids = set()
    for dataset, _, k in DATASETS:
        ids.add(f'primary_{dataset}_E2_PAST_CONTROL_k0_s92601_G')
        ids.update(f'primary_{dataset}_E2_CONDITIONAL_k{0 if b == "RAW" else k}_s92601_{b}' for b in ['RAW', 'B0', 'B2'])
    channels = {run_id: {} for run_id in ids}
    for row in csv.DictReader(io.StringIO(restored.decode('utf-8-sig'), newline='')):
        if row['run_id'] in ids:
            require(row['channel'] not in channels[row['run_id']], 'unique channel row')
            channels[row['run_id']][row['channel']] = row
    del restored
    group_source = list(csv.DictReader((ROOT / EVAL / 'GROUP_ERRORS.csv').open(encoding='utf-8-sig', newline='')))
    group_index = {(g['run_id'], g['pattern'], g['group']): g for g in group_source}
    matches, excluded, group_results, overall_results = [], [], [], []
    train_fields = ['seed', 'learning_rate', 'effective_batch', 'microbatch', 'registered_epoch_endpoints',
        'actual_epochs', 'updates', 'exposures', 'unique_M_origins', 'fit_split', 'checkpoint_split',
        'head_parameter_count', 'head_input_width', 'initial_state_sha', 'exposure']
    for dataset, label, k in DATASETS:
        right = f'primary_{dataset}_E2_PAST_CONTROL_k0_s92601_G'
        for baseline in ['RAW', 'B0', 'B2']:
            left = f'primary_{dataset}_E2_CONDITIONAL_k{0 if baseline == "RAW" else k}_s92601_{baseline}'
            try:
                manifests, receipts, precontracts, audit_bindings = [], [], [], []
                for run_id in [left, right]:
                    rel = f'predictions/{run_id}/V_select/PREDICTION_MANIFEST.json'
                    m, rec, pre = load(rel), load(f'combinations/{run_id}/receipt.json'), load(f'predictions/{run_id}/V_select/PRE_TARGET_CONTRACT.json')
                    same(evaluated_inputs[rel], pins[rel], 'saved group input manifest')
                    a = audit_index[(run_id, 'V_select')]
                    same(a['status'], 'PASS', 'audit PASS')
                    same(a['manifest_public_relative_path'], rel, 'audit manifest path')
                    same(a['manifest_sha256'], pins[rel], 'audit manifest SHA')
                    for pkey, parent in m['metadata']['actual_parent_manifests'].items():
                        parent_rel = parent['manifest_public_relative_path']
                        same(parent['manifest_sha256'], pins[parent_rel], 'current parent manifest SHA')
                        actual_parent = load(parent_rel)
                        same(parent['model_sha'], actual_parent['model_sha'], 'parent model SHA')
                        same(parent['archive_sha256'], actual_parent['archive_sha256'], 'parent archive SHA')
                    audit_bindings.append(dict(run_id=run_id, manifest=rel, manifest_sha256=pins[rel], audit_status=a['status']))
                    same(rec['execution_status'], 'COMPLETE_VALID', 'receipt complete')
                    same(rec['contract'], dict(pre['expected_receipt_contract'], parents_sha=rec['contract']['parents_sha']), 'pretarget receipt contract')
                    for key in ['source_sha', 'data_sha', 'scaler_sha']:
                        same(m[key], rec['contract'][key], 'manifest/receipt ' + key)
                    same(pre['arrays'], {name: m['arrays'][name] for name in ['target_normalized', 'observed', 'origins', 'channel_ids']}, 'pretarget arrays')
                    same(m['archive_sha256'], rec['prediction_artifacts']['head']['V_select']['sha256'], 'receipt archive binding')
                    same(m['metadata']['realized_model_binding']['selected_artifact_sha256'], rec['selected_checkpoint_sha256'], 'selected checkpoint metadata binding')
                    require(not rec['T_final_accessed'] and rec['parents_unchanged'] and not rec['parent_forecast_gradients'], 'frozen parent flags')
                    same(rec['foundation_calls_during_H_fit'], 0, 'no F in H fitting')
                    for pkey, parent in rec['parent_models'].items():
                        same(parent['role'], 'base_parent', 'parent role')
                        same(parent['fold'], 'primary', 'parent fold')
                        for boundary in ['parent_fit_end', 'canonical_scaler_fit_end', 'parent_normalization_fit_end']:
                            same(parent[boundary], assets[dataset]['primary_split_ranges']['A_stop'][1], 'A60 parent boundary')
                    manifests.append(m); receipts.append(rec); precontracts.append(pre)
                lm, rm = manifests; lr, rr = receipts
                same(lr['variant'], 'conditional', 'left variant');same(rr['variant'], 'past_control', 'right variant')
                for key in ['split', 'cohort', 'source_sha', 'data_sha', 'scaler_sha']:
                    same(lm[key], rm[key], 'V identity ' + key)
                same(lm['split'], 'V_select', 'V-only')
                for name in ['target_normalized', 'observed', 'origins', 'channel_ids']:
                    same(lm['arrays'][name], rm['arrays'][name], 'reference array ' + name)
                same(lm['metadata']['actual_parent_manifests']['g'], rm['metadata']['actual_parent_manifests']['g'], 'whole actual G parent')
                same(rm['metadata']['actual_parent_manifests']['b'], rm['metadata']['actual_parent_manifests']['g'], 'past control b equals G placeholder')
                for name in ['g_m', 'g_v']:
                    same(lr['parent_cache_fingerprints'][name], rr['parent_cache_fingerprints'][name], 'G fingerprint ' + name)
                for key in ['source_sha', 'data_sha', 'scaler_sha', 'model_sha']:
                    same(lr['contract'][key], rr['contract'][key], 'head template ' + key)
                for key in train_fields:
                    same(lr[key], rr[key], 'head training ' + key)
                same(lr['seed'], 92601, 'seed');same(lr['learning_rate'], .001, 'learning rate')
                same(lr['microbatch'], 4, 'microbatch');same(lr['effective_batch'], 16, 'effective batch')
                same(lr['registered_epoch_endpoints'], [.25, 1., 3.], 'epoch schedule');same(lr['actual_epochs'], 3., 'actual epochs')
                same(lr['head_input_width'], assets[dataset]['L'] + 2 * assets[dataset]['H'], 'head input width')
                same(lr['head_parameter_count'], 17580 if dataset == 'pems08' else 21072, 'head parameter count')
                require(lr['contract']['config_sha'] != rr['contract']['config_sha'], 'variant-specific config SHA')
                same(set(channels[left]), set(channels[right]), 'channel membership')
                same(len(channels[left]), lm['arrays']['channel_ids']['shape'][0], 'channel count')
                values = []
                for channel, l in channels[left].items():
                    b = channels[right][channel]; p = pattern_index[(dataset, channel)]
                    for key in ['dataset', 'fold', 'split', 'cohort', 'data_sha', 'scaler_sha', 'metric_scale', 'count']:
                        same(l[key], b[key], 'channel ' + key)
                    count = int(l['count']);require(count >= 0, 'nonnegative count')
                    for pattern in PATTERNS:
                        same(l[pattern + '_group'], b[pattern + '_group'], 'pair group membership')
                        same(l[pattern + '_group'], p[pattern + '_group'], 'frozen group membership')
                    values.append(dict(channel=channel, count=count, dmse=(float(l['sse'])-float(b['sse']))/count if count else None,
                        dmae=(float(l['sae'])-float(b['sae']))/count if count else None, pattern=p))
                active = [v for v in values if v['count']]
                dmse = math.fsum(v['dmse'] for v in active) / len(active)
                dmae = math.fsum(v['dmae'] for v in active) / len(active)
                err_mse = abs(dmse - (lm['score']['mse'] - rm['score']['mse']))
                err_mae = abs(dmae - (lm['score']['mae'] - rm['score']['mae']))
                require(max(err_mse, err_mae) <= 1e-9, 'overall vs manifest tolerance1e-9')
                common = dict(dataset=dataset, baseline=baseline, left_run_id=left, right_run_id=right,
                    fold='primary', split='V_select', cohort=lm['cohort'], k=0 if baseline=='RAW' else k,
                    seed=92601, learning_rate=.001, head_epochs=lr['actual_epochs'], head_updates=lr['updates'],
                    left_selected_updates=lr['selected_updates'], right_selected_updates=rr['selected_updates'],
                    n_origins=lm['arrays']['origins']['shape'][0], horizon=assets[dataset]['H'],
                    metric_scale=lm['score']['metric_scale'])
                overall_results.append(dict(common, active_channels=len(active), observed_target_cells=sum(v['count'] for v in values),
                    delta_mse=dmse, delta_mae=dmae, left_mse=lm['score']['mse'], right_mse=rm['score']['mse'],
                    left_mae=lm['score']['mae'], right_mae=rm['score']['mae'], mse_manifest_abs_difference=err_mse, mae_manifest_abs_difference=err_mae))
                for pattern in PATTERNS:
                    for group in ['low', 'high', 'NA']:
                        members = [v for v in values if v['pattern'][pattern+'_group']==group]
                        act = [v for v in members if v['count']]
                        for run_id in [left, right]:
                            old = group_index[(run_id, pattern, group)]
                            same(int(old['active_channels']), len(act), 'saved group active count')
                            same(int(old['total_group_channels']), len(members), 'saved group total count')
                            same(int(old['observed_target_cells']), sum(v['count'] for v in members), 'saved group observed count')
                        group_results.append(dict(common, pattern=pattern, group=group, total_group_channels=len(members),
                            active_channels=len(act), observed_target_cells=sum(v['count'] for v in members),
                            delta_mse=math.fsum(v['dmse'] for v in act)/len(act) if act else None,
                            delta_mae=math.fsum(v['dmae'] for v in act)/len(act) if act else None))
                matches.append(dict(dataset=dataset, baseline=baseline, left_run_id=left, right_run_id=right,
                    audit_bindings=audit_bindings, matched_training={key: lr[key] for key in train_fields},
                    G_parent=lm['metadata']['actual_parent_manifests']['g'],
                    excluded_from_equality=['variant', 'b parent', 'config_sha', 'selected_updates', 'selected weights/hash'],
                    selected_updates=[lr['selected_updates'], rr['selected_updates']],
                    maximum_overall_manifest_abs_difference=max(err_mse,err_mae)))
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append(dict(dataset=dataset, baseline=baseline, left_run_id=left, right_run_id=right,
                    reason=str(exc).replace(str(ROOT), '<public root>')))
    write_csv(out/'OVERALL_DIFFERENCES.csv', overall_results)
    write_csv(out/'PAIRED_PATTERN_DIFFERENCES.csv', group_results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    for metric in ['mse','mae']:
        fig, axes = plt.subplots(2,2,figsize=(10.8,9.2))
        fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.26,wspace=.36,hspace=.75)
        fig.suptitle('Forecast-conditioned head minus past-only control: '+metric.upper(),fontsize=16,fontweight='bold',y=.98)
        fig.text(.5,.92,'Same frozen G and head capacity | seed 92601 | head search 0.25 / 1 / 3 epochs | primary V_select',ha='center',fontsize=10)
        for letter,ax,(dataset,label,k) in zip('ABCD',axes.flat,DATASETS):
            rows=[r for r in overall_results if r['dataset']==dataset]
            ax.axhline(0,color='#555555',lw=1)
            for x,baseline in enumerate(['RAW','B0','B2']):
                row=next((r for r in rows if r['baseline']==baseline),None)
                if row is None:
                    ax.text(x,0,'Excluded',ha='center',va='bottom',color='#777777');continue
                value=row['delta_'+metric]
                ax.scatter([x],[value],s=65,c='#0072B2' if value<0 else '#D55E00',edgecolors='black',linewidths=.5,zorder=3)
                ax.annotate(f'{value:+.5g}\nn={row["active_channels"]}',(x,value),xytext=(0,10 if value>=0 else -12),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
            bound=max([abs(r['delta_'+metric]) for r in rows]+[1e-6])*1.8
            ax.set_ylim(-bound,bound);ax.set_xlim(-.5,2.5)
            ax.set_xticks([0,1,2],['Raw Bolt',f'B0 (K={k})',f'B2 (K={k})'])
            ax.set_ylabel('Delta '+metric.upper());ax.yaxis.set_major_locator(MaxNLocator(nbins=5,symmetric=True))
            ax.set_title(f'{letter}  {label} | H={assets[dataset]["H"]}',loc='left',fontsize=12)
            ax.spines[['top','right']].set_visible(False)
            if rows:ax.text(.5,-.27,f'V origins={rows[0]["n_origins"]}; same past control reused for 3 comparisons',transform=ax.transAxes,ha='center',fontsize=8)
        fig.text(.5,.07,'Negative (blue): forecast-conditioned head lower error. Positive (orange): past-only control lower error.\n'
            'n = active channels, not independent replicates; panel-specific symmetric axes; no pooling across datasets.\n'
            'Matched feature-control comparison; no P-only causal, equal-cost, convergence, significance, or generalization claim.',ha='center',fontsize=9,linespacing=1.7)
        fig.savefig(out/(metric.upper()+'.png'),dpi=300);fig.savefig(out/(metric.upper()+'.svg'));plt.close(fig)
    readme = '''# 예측 특징과 추가 과거 특징의 같은 G·동일 용량 비교

[MSE 그림](MSE.png) · [벡터 MSE](MSE.svg) · [MAE 그림](MAE.png) · [벡터 MAE](MAE.svg) · [전체 원수치](OVERALL_DIFFERENCES.csv) · [144 패턴 원수치](PAIRED_PATTERN_DIFFERENCES.csv) · [검증 증빙](RESULT.json) · [입력 SHA](INPUT_PINS.json) · [재현 코드](reproduce.py)

![MSE difference](MSE.png)
![MAE difference](MAE.png)

[확인] 기존 channel_patterns.py:424는 CONDITIONAL과 PAST_CONTROL에도 b·g 전체 부모 서명 일치를 요구해 의도한 비교를 누락했다. networks.py:339–346의 control=True는 b의 shape만 확인하고 값을 사용하지 않으며 최근 H 과거−현재값을 특징으로 쓴다. postfit.py:214–215는 control의 b=g로 지정한다. 이번 별도 진단은 기존 코드를 바꾸지 않고 정확한 G 부모가 같은 쌍을 공개 metadata로 엄격히 검증했다. CONDITIONAL−ALPHA_FIT의 전체 부모 동일 요구는 변경하지 않았다.

예측 head 입력은 같은 과거 입력·g−현재값에 b−g를 추가하며, control은 마지막 부분을 최근 H 과거−현재값으로 교체한다. 같은 G, hidden32, L+2H 입력, head parameter 수, 초기 weights SHA, seed92601, lr0.001, micro4/effective16, 실제 3epoch/updates/노출/기원빈도를 맞춘 비교다. 실제 V 선택 checkpoint는 각자 선택되어 selected_updates와 weights/hash가 다를 수 있다. 최초 Raw/B0 quarter-K/B2 quarter-K만 선택했고 후속 deep 결과는 포함하지 않았다.

366개 독립 prediction 감사 중 해당 16개 V manifest의 exact SHA와 PASS를 확인했다. 타깃·observed mask·origin·channel ID의 shape/dtype/SHA, cohort/data/scaler와 학습 계약, A60에 고정된 부모의 경계·역할을 검증했다. 개별 head artifact, 모델, NPZ 또는 원자료를 열지 않았다. 저장 채널 오류 CSV의 gzip을 실제 해제해 원 byte SHA를 확인하고, 채널별 (SSE_left−SSE_right)/count 및 (SAE_left−SAE_right)/count를 math.fsum으로 활성 채널에 균등 평균했다. 전체 delta는 두 V manifest 점수 차이와 절대 허용오차 1e−9 이내인지 확인했다. 패턴은 기존 A_fit snapshot의 low/high/NA를 재사용했고 빈 그룹은 None(빈 CSV 값)으로 유지했다.

이는 같은 G와 동일 head 용량 아래 예측 차이 특징과 추가 과거 특징의 대조다. P만의 인과효과, 비용 동등, 충분 수렴, 독립 일반화나 통계적 유의성을 의미하지 않는다. 자료별 PAST_CONTROL 하나를 세 번 참조하므로 12개를 독립 반복으로 세면 안 된다. n은 활성 채널수다. 각 자료의 H·정규화가 달라 자료 간 평균을 계산하지 않는다. 데이터별 대칭 축·0선과 색각 안전 청색/주황 및 부호 수치로 표시한다. 과거 특징의 observed exact-zero 그룹은 간헐 수요의 증명이 아니며 Jena의 legacy FP32 역정규화 한계도 남는다.

재현은 같은 campaign 위치에서 reproduce.py --output 새폴더로 실행한다. source·98개 공개 metadata/CSV 입력 SHA를 고정하며 기존 결과는 덮어쓰지 않는다. 새 학습·추론·F·GPU·T 접근은 하지 않는다. PNG 300dpi와 SVG를 함께 제공한다.
'''
    (out/'README_KO.md').write_text(readme,encoding='utf-8')
    for rel,expected in pins.items():same(sha(ROOT/rel),expected,'unchanged after calculation '+rel)
    require(not any(n in sys.modules for n in ['torch','chronos','transformers','tensorflow','jax']), 'no model framework imports')
    require(time.monotonic()-START<600,'10minute soft wall')
    require(sum(p.stat().st_size for p in out.iterdir())<100*1024**2,'100MiB output cap')
    result=dict(status='COMPLETE_MATCHED' if len(matches)==12 and not excluded else 'PARTIAL',
        matched_pairs=len(matches),excluded_pairs=excluded,overall_rows=len(overall_results),paired_group_rows=len(group_results),
        empty_group_rows=sum(x['active_channels']==0 for x in group_results),matched_conditions=matches,
        formula='fsum((SSE_left-SSE_right)/observed_count)/active_channel_count; analogous SAE',
        input_files=[dict(file=rel,sha256=s) for rel,s in pins.items()],
        source_script_sha256=sha(Path(__file__)),new_training=False,inference_reexecuted=False,model_or_array_files_opened=False,
        F_execution=False,GPU_execution=False,T_accessed=False,cost_equivalence_claimed=False,independent_replicates=0,
        software=dict(python=sys.version.split()[0],matplotlib=matplotlib.__version__),
        wall_seconds=time.monotonic()-START,
        outputs=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    with (out/'RESULT.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=result['status'],matched_pairs=len(matches),excluded=excluded,overall=overall_results,
                         paired_group_rows=len(group_results),wall_seconds=result['wall_seconds']),ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,reason=str(exc).replace(str(ROOT),'<public root>')),ensure_ascii=False))
        raise SystemExit(1)
