"""ROOT-only visualization of the completed remaining-family report aggregates."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import types


NAME='priority_valid_family_20261016_v1'
COMMON='priority_valid_minima_20261016_v1'
BUILDER_SHA='792ab964e891e84f04ea021bfed14239c295d9782b1560a14d0a7fe3a672f6d6'
AUDITOR_SHA='3a54591668469213a986228b1c0305710c7ca981b4002110a32e9bf4d164e53d'
INPUT_HELPER_SHA='8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd'
STYLE_SHA='50e25501857a9c3aad0c9f9902397c00075025d1b3556778539b0e400adaf47b'
COLUMNS=('MSE','MAE','B1_median_seconds','B8_median_seconds')
METHODS=(
    dict(dataset='jena',role='B2_K6_REFERENCE',run_id='postT_cpu_jena_E3_B2_k6_s92601_v1',family='B2',registered_k=6,width=6,reused=True,label='B2 K6\n기존',color='#687D8D'),
    dict(dataset='jena',role='B4_K6_REFERENCE',run_id='postT_cpu_jena_E1_B4_k6_s92601_family_v1',family='B4',registered_k=6,width=6,reused=True,label='B4 K6\n기존',color='#0072B2'),
    dict(dataset='jena',role='LORA_R8_C21_REFERENCE',run_id='postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1',family='LORA',registered_k=0,width=21,reused=True,label='LoRA R8 / C21\n기존',color='#D55E00'),
    dict(dataset='jena',role='ADAPTS_BOLT_PORT_K6',run_id='postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1',family='ADAPTS',registered_k=6,width=6,reused=False,label='AdaPTS port K6\n신규',color='#009E73'),
    dict(dataset='beijing',role='B2_K11_REFERENCE',run_id='postT_cpu_beijing_E1_B2_k11_s92601_required_v1',family='B2',registered_k=11,width=11,reused=True,label='B2 K11\n기존',color='#687D8D'),
    dict(dataset='beijing',role='B4_K11',run_id='postT_cpu_beijing_E1_B4_k11_s92601_family_v1',family='B4',registered_k=11,width=11,reused=False,label='B4 K11\n신규',color='#0072B2'))
SVG='figures/REMAINING_FAMILY_TWO_DATASETS_v1.svg'
VIS_MD='REMAINING_FAMILY_VISUALIZATION_KO_v1.md'
VIS_BIND='REMAINING_FAMILY_FIGURE_BINDINGS_v1.json'
CSV='REMAINING_FAMILY_RESULTS_v1.csv'
REPORT_MD='REMAINING_FAMILY_KO_v1.md'
REPORT_BIND='REMAINING_FAMILY_BINDINGS_v1.json'
CAPTION=('데이터셋별 독립 좌표 · 한 seed92601 / 반복 개발 V · 선택 refit 길이는 원 표를 따름(0도 허용) · CI 없음.\n'
    'K/F 폭·LR·노출·용량·기존 학습 ancestry가 달라 동등 compute/순수 encoder 효과가 아님. 수렴·독립 일반화 미확정.')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


def module(path,expected,label):
    if sha(path)!=expected:raise ValueError('Exact frozen input/builder/style source required')
    value=types.ModuleType(label);value.__file__=str(path)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),value.__dict__)
    return value


def number(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError('Finite nonnegative report aggregate required')
    return result


def prepare_data(rows,audit,report):
    if (audit.get('schema')!='INDEPENDENT_REMAINING_FAMILY_AUDIT_v1' or audit.get('status')!='PASS' or audit.get('error') is not None
            or audit.get('audit_source_sha256')!=AUDITOR_SHA or audit.get('input_SHA_before_after_equal') is not True or audit.get('source48_preserved') is not True
            or audit.get('whole_MASTER_complete') is not False or audit.get('independent_generalization_claimed') is not False
            or report.get('schema')!='REMAINING_FAMILY_REPORT_BINDINGS_v1' or report.get('status')!='FROM_ACTUAL_COMPLETE_REMAINING_FAMILY_AUDIT'
            or report.get('builder_source_sha256')!=BUILDER_SHA or report.get('auditor_source_sha256')!=AUDITOR_SHA
            or report.get('input_SHA_before_after_equal') is not True or report.get('source48_preserved') is not True
            or report.get('whole_MASTER_complete') is not False or report.get('independent_generalization_claimed') is not False
            or report.get('new_accuracy_conditions')!=2 or report.get('new_cost_conditions')!=4
            or report.get('reused_Jena_reference_rows')!=3 or report.get('reused_Beijing_reference_rows')!=1
            or report.get('comparison_scope')!='WITHIN_EACH_DATASET_ONLY' or report.get('cross_dataset_MSE_ranking') is not False
            or report.get('reference_audits_rerun') is not False):raise ValueError('Actual complete PASS audit and report with separate dataset scopes required')
    if len(rows)!=6 or {r.get('run_id') for r in rows}!={m['run_id'] for m in METHODS} or len(audit.get('rows',[]))!=2 or len(report.get('references',[]))!=4:raise ValueError('Exact fresh2 and reused4 report rows required')
    data={d:[] for d in ('jena','beijing')}
    for method in METHODS:
        matches=[r for r in rows if r.get('run_id')==method['run_id']]
        if len(matches)!=1:raise ValueError('Duplicate or foreign report identity')
        row=matches[0];dataset=method['dataset']
        expected=dict(dataset=dataset,role=method['role'],family=method['family'],row_scope='REUSED_REFERENCE' if method['reused'] else 'NEW_REMAINING_FAMILY',K=method['registered_k'],registered_K=method['registered_k'],effective_F_width=method['width'],C=21 if dataset=='jena' else 11,L=512,H=48,seed=92601,accuracy_device='cpu',training_device='cpu',CPU_threads=2,V_origins=43,observed_labels=43344 if dataset=='jena' else 20539,accuracy_reused=method['reused'],cost_reused=method['reused'],F_resident_bytes_scope='NOT_SEPARATELY_RECORDED')
        if any(str(row.get(key))!=str(value) for key,value in expected.items()) or row.get('F_resident_bytes') not in ('',None):raise ValueError('Exact dataset/cohort/reuse/F-width and NA memory scope required')
        for column in COLUMNS:number(row[column])
        if method['reused']:
            reference=[r for r in report['references'] if r.get('run_id')==method['run_id']]
            if len(reference)!=1 or any(number(row[c])!=number(reference[0][c]) for c in COLUMNS):raise ValueError('Reused reference display differs from bound report source')
        else:
            record=[r for r in audit['rows'] if r.get('run_id')==method['run_id']]
            if len(record)!=1 or any(number(row[c])!=number(record[0][c.lower()]) for c in ('MSE','MAE')):raise ValueError('Fresh display differs from actual audited accuracy')
            if row.get('selected_refit_epoch') in (None,'') or number(row['selected_refit_epoch']) not in (0,.25,1,3):raise ValueError('Actual selected endpoint including zero required')
            for batch in (1,8):
                timed=[s for s in record[0][f'cost_B{batch}']['summaries'] if s.get('phase')=='timed' and s.get('batch_size')==batch and s.get('trials')==20]
                if len(timed)!=1 or number(row[f'B{batch}_median_seconds'])!=number(timed[0]['median_seconds']):raise ValueError('Timed whole20 median must equal actual audited value')
        data[dataset].append(row)
    return data


def validate_artists(artists,data):
    expected={(dataset,column) for dataset in ('jena','beijing') for column in COLUMNS}
    if len(artists)!=8 or {(a.get('dataset'),a.get('column')) for a in artists}!=expected:raise ValueError('Exact eight independent panels required')
    for artist in artists:
        rows=data[artist['dataset']]
        if artist.get('run_ids')!=[r['run_id'] for r in rows] or artist.get('values')!=[number(r[artist['column']]) for r in rows]:raise ValueError('Actual artist values must equal the 24 unchanged CSV aggregates')


def draw_panels(axes,data):
    artists=[]
    titles=('정규화 MSE','정규화 MAE','CPU2 B1 whole median','CPU2 B8 whole median')
    for row_index,dataset in enumerate(('jena','beijing')):
        rows=data[dataset];methods=[m for m in METHODS if m['dataset']==dataset]
        for column_index,column in enumerate(COLUMNS):
            ax=axes[row_index][column_index];values=[number(r[column]) for r in rows]
            bars=ax.bar(range(len(rows)),values,width=.56,color=[m['color'] for m in methods],zorder=3)
            for bar,method in zip(bars,methods):bar.set_hatch('//' if method['reused'] else '')
            artists.append(dict(dataset=dataset,column=column,run_ids=[r['run_id'] for r in rows],values=[float(b.get_height()) for b in bars]))
            top=max(max(values),1e-12)
            for i,value in enumerate(values):ax.text(i,value+top*.035,f'{value:.4g}',ha='center',va='bottom',fontsize=8.5,fontweight='bold',color='#263238')
            ax.set_title(chr(65+row_index*4+column_index)+'   '+dataset.title()+': '+titles[column_index],loc='left',fontsize=10.3,fontweight='bold',pad=10)
            ax.set_ylabel(column if column_index<2 else '초 / 전체 batch')
            ax.set_xticks(range(len(rows)),[m['label'] for m in methods]);ax.tick_params(axis='both',length=0,labelsize=8)
            ax.set_ylim(0,top*1.28);ax.grid(axis='y',color='#E4E9ED',linewidth=.7,zorder=0);ax.set_axisbelow(True)
    validate_artists(artists,data)
    return artists


def render(data,csv_sha,audit_sha):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path=Path(os.environ['WINDIR'])/'Fonts/malgun.ttf'
    if not font_path.is_file():raise FileNotFoundError('Existing Malgun font required; no installation')
    font=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family':font,'font.size':9,'axes.unicode_minus':False,'svg.fonttype':'none','svg.hashsalt':'REMAINING_FAMILY_TWO_DATASETS_v1','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,4,figsize=(18,10));fig.subplots_adjust(left=.055,right=.985,top=.79,bottom=.25,wspace=.38,hspace=.72)
    try:
        fig.suptitle('남은 방법군 최소 비교: 데이터셋별 정확도와 실제 전체 추론 비용',x=.055,y=.965,ha='left',fontsize=16,fontweight='bold',color='#172B3A')
        fig.text(.055,.921,'Jena C21 / V43 / 관측 43,344 · Beijing C11 / V43 / 관측 20,539 · L512 / H48 / seed92601 / CPU2',fontsize=10,color='#455A64')
        fig.text(.055,.883,'사선=기존 정확도·비용 재사용, 단색=신규 완료 조건 · B1/B8은 warm whole20의 초 / 전체 batch · 모든 축은 0부터',fontsize=9.4,color='#455A64')
        fig.text(.055,.847,'원정밀도 CSV SHA '+csv_sha[:12]+' · 독립 PASS audit SHA '+audit_sha[:12],fontsize=8.5,color='#536471')
        artists=draw_panels(axes,data)
        fig.text(.055,.184,CAPTION,fontsize=8.9,linespacing=1.7,color='#536471')
        fig.text(.055,.115,'AdaPTS는 Bolt 호환 포트의 해당 구현 범위이며 원논문 전체 재현으로 확대하지 않음. LoRA는 원 C21 / effective F 적응 경로.',fontsize=8.7,color='#536471')
        fig.text(.055,.078,'모델·F-pool cold load와 LoRA merge는 warm timer에 포함되지 않음. F-only resident bytes는 NA. p90/CI/독립 seed 분산은 추가하지 않음.',fontsize=8.6,color='#536471')
        fig.text(.055,.042,'원 보고서가 먼저이며 그림 실패는 다른 실험의 중단 gate가 아님. 전체 MASTER 미완료; 데이터셋을 섞은 모델 순위를 만들지 않음.',fontsize=8.6,color='#536471')
        svg,png=io.BytesIO(),io.BytesIO();fig.savefig(svg,format='svg',metadata={'Date':None,'Creator':'Audited remaining family report aggregates'});fig.savefig(png,format='png',dpi=145,metadata={'Software':'Audited remaining family report aggregates'})
        clean=('\n'.join(line.rstrip() for line in svg.getvalue().decode('utf-8').splitlines())+'\n').encode()
        return clean,png.getvalue(),artists,font,sha(font_path)
    finally:plt.close(fig)


def execute(public,cache,plot_sha,report_sha,audit_path,audit_sha,csv_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    svg_path=scope/SVG;md_path=scope/VIS_MD;binding_path=scope/VIS_BIND;preview=cache/NAME/'visualization/REMAINING_FAMILY_TWO_DATASETS_v1.png'
    if any(p.exists() for p in (svg_path,md_path,binding_path,preview)):raise FileExistsError('Existing visualization preserved; fresh outputs only')
    if Path(__file__).resolve()!=scope/'plot_remaining_family_v2.py' or sha(__file__)!=plot_sha:raise ValueError('ROOT current plot source SHA required')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')):raise ValueError('Explicit CPU/BLAS1 visualization environment required')
    helper=module(public/COMMON/'build_beijing_report.py',INPUT_HELPER_SHA,'remaining_plot_inputs')
    builder=module(scope/'build_remaining_family_report_v2.py',BUILDER_SHA,'remaining_plot_builder')
    style=module(public/COMMON/'plot_beijing.py',STYLE_SHA,'remaining_plot_resource_style')
    if time.time()>=style.DEADLINE:raise TimeoutError('Original worker deadline reached')
    inputs=helper.Inputs(public,cache)
    for name,want in ((NAME+'/plot_remaining_family_v2.py',plot_sha),(NAME+'/build_remaining_family_report_v2.py',BUILDER_SHA),(NAME+'/audit_remaining_family.py',AUDITOR_SHA),(COMMON+'/build_beijing_report.py',INPUT_HELPER_SHA),(COMMON+'/plot_beijing.py',STYLE_SHA)):
        inputs.capture('public',name,want)
    report=inputs.json('public',NAME+'/'+REPORT_BIND,report_sha)
    audit_path=Path(audit_path).resolve()
    if not audit_path.is_relative_to(scope) or audit_path.suffix!='.json':raise ValueError('ROOT explicit actual N5 remaining audit JSON required')
    audit_relative=audit_path.relative_to(public).as_posix();audit=inputs.json('public',audit_relative,audit_sha);builder.validate_audit(audit)
    if report.get('audit_sha256')!=audit_sha or report.get('audit_relative_path')!=audit_relative:raise ValueError('Report must bind the same actual complete PASS audit')
    outputs=report.get('outputs',[])
    if len(outputs)!=2 or {r['relative_path'] for r in outputs}!={NAME+'/'+CSV,NAME+'/'+REPORT_MD}:raise ValueError('Actual experiment MD and CSV must precede visualization')
    for row in (*report.get('inputs',[]),*audit.get('inputs',[])):inputs.capture(row['root'],row['relative_path'],row['sha256'])
    for row in outputs:inputs.capture('public',row['relative_path'],row['sha256'])
    csv_relative=NAME+'/'+CSV
    if inputs.files[('public',csv_relative)]!=csv_sha:raise ValueError('ROOT actual CSV SHA must equal report output binding')
    csv_blob=inputs.path('public',csv_relative).read_bytes()
    expected=builder.compose(audit,report['references'],audit_path.relative_to(scope).as_posix())
    if csv_blob!=expected[CSV].encode('utf-8'):raise ValueError('CSV bytes must preserve exact frozen builder aggregates and reused reference rows')
    text=csv_blob.decode('utf-8')
    reader=csv.DictReader(io.StringIO(text))
    if reader.fieldnames!=list(builder.FIELDS):raise ValueError('Exact frozen 57-column CSV header required')
    rows=list(reader);data=prepare_data(rows,audit,report)
    if not inputs.preserved():raise ValueError('Inputs changed before renderer')
    started=time.perf_counter()
    with style.ResourceCheck() as resources:
        svg,png,artists,font,font_sha=render(data,csv_sha,audit_sha)
        if not inputs.preserved():raise ValueError('Inputs changed during renderer')
        resources.check()
    activities='; '.join(r['dataset']+'/'+r['role']+': selected refit '+str(r.get('selected_refit_epoch') or 'NA')+' epoch, '+str(r.get('refit_updates') or 'NA')+' updates, '+str(r.get('refit_exposures') or 'NA')+' exposures' for r in rows)
    md=('# 남은 방법군: 데이터셋별 시각자료\n\n![Jena와 Beijing을 분리한 정확도와 CPU2 warm whole 비용](figures/REMAINING_FAMILY_TWO_DATASETS_v1.svg)\n\n'
        '[실험 보고](REMAINING_FAMILY_KO_v1.md) · [원정밀도 CSV](REMAINING_FAMILY_RESULTS_v1.csv) · [그림 결속](REMAINING_FAMILY_FIGURE_BINDINGS_v1.json) · [코드](plot_remaining_family_v2.py) · [실제 독립 감사]('+audit_path.relative_to(scope).as_posix()+')\n\n'
        'Jena는 기존 B2 K6/B4 K6/LoRA R8와 신규 AdaPTS Bolt port K6를, Beijing은 기존 B2 K11과 신규 B4 K11을 표시했다. 기존 네 행의 정확도·비용은 재사용이며 새 독립 학습이나 재측정이 아니다. 각 데이터셋은 독립 좌표이고 표시는 명시한 역할 순서다. 사선은 재사용, 단색은 신규이다. 두 데이터셋을 합친 MSE 순위를 만들지 않는다.\n\n'
        'MSE가 주지표이며 MAE와 CPU2 B1/B8 ordinary whole20 median을 함께 표시했다. B8은 8개 입력 batch 전체의 초이며 8로 나누지 않았다. 모든 축은 0부터이며 표시 라벨만 반올림했다. 8개 panel의 artist24개 값은 CSV의 원 수치와 정확히 대사했다. p90·CI·seed 분산을 새로 추가하지 않았다.\n\n'
        +CAPTION.replace('\n',' ')+'\n\n'+activities+'.\n\n'
        'AdaPTS는 현재 Bolt 호환 포트 구현 범위의 최소 비교이고 원논문 전체 충실 재현으로 확대하지 않는다. LoRA는 원 C21 stream/effective F 적응 경로여서 K6 압축과 F 폭·모수·연산이 다르다. LR·초기화·선택 refit 길이(0도 허용)·실제 노출·학습 ancestry는 원 표의 조건을 그대로 유지한다. 동등 compute, 동일 표현/디코더, 순수 encoder의 인과 효과 또는 수렴·독립 일반화를 주장하지 않는다.\n\n'
        'warm whole timer에는 모델/F-pool cold load 및 LoRA merge 준비가 포함되지 않는다. F-only resident bytes는 별도 미기록(NA)이며 Route 전체 weights/RSS로 대입하지 않는다. 그림을 위해 새 통계/trial 재집계, Task/원자료/NPZ/CP/모델 해석, 학습·추론·비용 재측정·프로세스 제어·Git·network를 수행하지 않았다. SVG는 공개용이며 PNG는 private preview이다. 보고서가 먼저이며 그림 실패는 다른 경로를 중단하는 gate가 아니다. 전체 MASTER는 미완료다.\n')
    svg_path.parent.mkdir(exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    for path,blob in ((svg_path,svg),(preview,png),(md_path,md.encode())):
        with path.open('xb') as handle:handle.write(blob)
    proof=dict(schema='REMAINING_FAMILY_TWO_DATASET_FIGURE_BINDINGS_v1',status='PASS_ACTUAL_AUDITED_REMAINING_AGGREGATE_VISUALIZATION',plot_source_sha256=plot_sha,builder_source_sha256=BUILDER_SHA,auditor_source_sha256=AUDITOR_SHA,actual_audit_sha256=audit_sha,report_bindings_sha256=report_sha,actual_csv_sha256=csv_sha,input_SHA_before_after_equal=True,
        inputs=[dict(root=root,relative_path=name,sha256=value) for (root,name),value in sorted(inputs.files.items())],outputs=[dict(relative_path=p.relative_to(scope).as_posix(),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)) for p,blob in ((svg_path,svg),(md_path,md.encode()))],private_preview=dict(published=False,cache_relative_path=preview.relative_to(cache).as_posix(),sha256=hashlib.sha256(png).hexdigest(),bytes=len(png)),artist_values_from_unchanged_CSV=artists,artist_groups=8,artist_scalar_count=24,CSV_precision_preserved=True,new_rows=2,reused_rows=4,dataset_axes_independent=True,p90_or_CI_added=False,font=font,font_file_sha256=font_sha,CPU_BLAS_threads=1,outer_CPU_maximum=2,outer_RSS_maximum_bytes=3*1024**3,own_RSS_limit_bytes=style.MEMORY_LIMIT,sampled_own_peak_RSS_bytes=resources.peak,own_OS_peak_working_set_bytes=resources.os_peak,deadline_unix=style.DEADLINE,elapsed_seconds=time.perf_counter()-started,actual_renderer_executed=True,new_accuracy_or_cost_measurements=0,report_first=True,plot_failure_blocks_other_paths=False,whole_MASTER_complete=False,independent_generalization_claimed=False,self_hash_policy='Bindings excludes itself')
    with binding_path.open('x',encoding='utf-8',newline='\n') as handle:json.dump(proof,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    return dict(status=proof['status'],public_files_written=3,private_previews_written=1,artist_groups=8,artist_scalar_count=24)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--root-triggered',action='store_true')
    for key in ('public-root','cache-root','plot-sha','report-binding-sha','audit','audit-sha','csv-sha'):parser.add_argument('--'+key)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='PREPARATION_ONLY_NOT_EXECUTED',input_reads=0,files_written=0,actual_render=0,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,matplotlib_imported='matplotlib' in sys.modules,future_actual_audit_sha256=None,future_report_binding_sha256=None,future_CSV_sha256=None)));return 0
    if not args.root_triggered or any(getattr(args,k) is None for k in ('public_root','cache_root','plot_sha','report_binding_sha','audit','audit_sha','csv_sha')):parser.error('ROOT exact current plot/audit/report-binding/CSV SHA, explicit roots and fresh outputs required')
    print(json.dumps(execute(args.public_root,args.cache_root,args.plot_sha,args.report_binding_sha,args.audit,args.audit_sha,args.csv_sha)));return 0


if __name__=='__main__':raise SystemExit(main())
