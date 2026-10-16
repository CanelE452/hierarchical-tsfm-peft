"""Prepare the two audited Jena families with the existing exact B2 reference.

Default0 input/output and lazy renderer imports. ROOT alone renders after reports.
No statistic/trial reaggregation, models, NPZ, Task, inference or cost operations.
"""
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
BUILDER_SHA='ca4958e5b1486afb211a84555b19d3750284f7e625403cc12ab0f843089d4267'
AUDITOR_SHA='11d19c69d562784847131e14429150e44a719e6384937a364db20d70615098b8'
INPUT_HELPER_SHA='8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd'
STYLE_SHA='50e25501857a9c3aad0c9f9902397c00075025d1b3556778539b0e400adaf47b'
B2_FILES={
    'B2_REFERENCE_VALIDATION_KO_v1.md':'362b7b970cef2783ed8af416939cb50797e06eec720e7f4a1982cafd43551206',
    'B2_REFERENCE_COSTS_v1.csv':'e87b84a1639322d6f8cee91f6cf2dc7c048e8e0121101002ffbcf82cdde121fa',
    'B2_REFERENCE_REPORT_BINDINGS_v1.json':'a58fdf1b85a4ddaa07942edc7ab1d1e0f22992d718f5a1fdfbb749c8d7403749'}
B2_AUDIT_SHA='346ae638277afc669facfcc98cc22f7e01820c79cd3bea60486913cdb700dae8'
B2_RUN='postT_cpu_jena_E3_B2_k6_s92601_v1'
B2_ARTIFACT='020d7d980e685c38e9b94472676a5d6b7f8b188ab573a16bd367aa20712a5963'
B2_MANIFEST='6a885f3c8f298009ae93c731409baff116263a41c9c17ecaff713cf5e9499148'
ROLES=('LORA_R8_C21','B4_K6')
IDS=('postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1','postT_cpu_jena_E1_B4_k6_s92601_family_v1')
LABELS=('B2 K6\n기존 가중치','LoRA R8\n원 C21','B4 K6')
COLUMNS=('MSE','MAE','B1_median_seconds','B8_median_seconds')
CAPTION=('세 방법 search3/refit1의 update·exposure 동일. B2/B4 frozen F, LoRA effective F 적응.\n'
    'B2 원 학습/정규화 ancestry 유지 · 한 seed92601/반복 개발 V · 수렴·독립 일반화 미확정.')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def module(path,expected,label):
    if sha(path)!=expected:raise ValueError('Exact reviewed stdlib/style source changed')
    value=types.ModuleType(label);value.__file__=str(path)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),value.__dict__)
    return value


def number(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError('Finite nonnegative source aggregate required')
    return result


def prepare_data(rows,audit,report,reference_rows,reference_audit,reference_binding):
    if (audit.get('schema')!='INDEPENDENT_INITIAL_FAMILY_AUDIT_v1' or audit.get('status')!='PASS' or audit.get('error') is not None
            or audit.get('audit_source_sha256')!=AUDITOR_SHA or audit.get('input_SHA_before_after_equal') is not True
            or audit.get('whole_MASTER_complete') is not False or audit.get('independent_generalization_claimed') is not False or audit.get('baseline_comparison_included') is not False
            or report.get('schema')!='INITIAL_FAMILY_REPORT_BINDINGS_v1' or report.get('status')!='FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_RECOVERY_AUDIT'
            or report.get('builder_source_sha256')!=BUILDER_SHA or report.get('auditor_source_sha256')!=AUDITOR_SHA
            or report.get('source48_preserved') is not True or report.get('input_SHA_before_after_equal') is not True or report.get('whole_MASTER_complete') is not False
            or report.get('new_accuracy_conditions')!=2 or report.get('new_cost_conditions')!=4):raise ValueError('Actual complete source-bound family audit and report required')
    activity=audit.get('matched_training_activity',{})
    expected_activity=dict(selected_epoch=1,search_updates=1863,refit_updates=752,search_exposures=29769,refit_exposures=12020,search_fit_end=10482,refit_fit_end=12579)
    provenance=audit.get('recovery_provenance',{})
    if (any(activity.get(k)!=v for k,v in expected_activity.items()) or activity.get('methods')!=[*IDS,B2_RUN]
            or report.get('matched_training_activity')!=activity or report.get('recovery_provenance')!=provenance
            or provenance.get('original_POST_sha256')!='28911ed451a95ce4af157dd863914f9078f5b0f3f11ab82d0ee5ae2efd6d6080'
            or provenance.get('original_POST_status')!='PARTIAL_VALIDATIONS_COSTS_PRESERVED'
            or provenance.get('original_false_rows_count')!=4 or provenance.get('original_receipt_mutated') is not False):
        raise ValueError('Separate recovery provenance and actual matched three-method training activity required')
    actual=audit.get('rows',[])
    if len(rows)!=2 or len(actual)!=2 or tuple(r.get('role') for r in rows)!=ROLES or tuple(r.get('role') for r in actual)!=ROLES:raise ValueError('Only exact two completed family CSV rows allowed')
    metadata=('search_max_epoch','selected_refit_epoch','search_updates','refit_updates','search_exposures','refit_exposures','search_seconds','refit_seconds','training_status')
    for row,record,identifier,k,registered in zip(rows,actual,IDS,(21,6),(0,6)):
        expected=dict(run_id=identifier,effective_F_width=k,registered_K=registered,accuracy_device='cpu',training_device='cpu',CPU_threads=2,observed_labels=43344,V_origins=43,search_max_epoch=3,accuracy_reused=False,F_resident_bytes_scope='NOT_SEPARATELY_RECORDED')
        if any(str(row.get(key))!=str(value) or record.get(key)!=value for key,value in expected.items()) or any(str(row.get(key))!=str(record.get(key)) for key in metadata):raise ValueError('Same CPU2 cohort/registered route and actual selected training metadata required')
        if record.get('F_resident_bytes') is not None or row.get('F_resident_bytes') not in ('',None):raise ValueError('Unrecorded F-only resident memory must stay NA')
        for key in COLUMNS:
            if number(row[key])!=number(record[key]):raise ValueError('Displayed source CSV differs from actual audited value')
    if (reference_audit.get('schema')!='INDEPENDENT_EXISTING_B2_REFERENCE_ACTUAL_VALIDATION_v1' or reference_audit.get('status')!='PASS'
            or reference_audit.get('new_training')!=0 or reference_audit.get('accuracy_reused') is not True or reference_audit.get('input_SHA_before_after_equal') is not True
            or reference_audit.get('new_T_target_operations')!=0 or reference_audit.get('whole_MASTER_complete') is not False or reference_audit.get('independent_generalization_claimed') is not False
            or reference_binding.get('schema')!='EXISTING_JENA_B2_REFERENCE_REPORT_BINDINGS_v1' or reference_binding.get('status')!='REPORT_FROM_ACTUAL_AUDIT_PASS'
            or reference_binding.get('audit_sha256')!=B2_AUDIT_SHA or reference_binding.get('new_training')!=0 or reference_binding.get('accuracy_reused') is not True
            or reference_binding.get('input_SHA_before_after_equal') is not True or reference_binding.get('whole_MASTER_complete') is not False):raise ValueError('Exact completed existing B2 validation/cost report required')
    if len(reference_rows)!=4 or {(r.get('batch_size'),r.get('phase')) for r in reference_rows}!={('1','timed'),('1','stage_probe'),('8','timed'),('8','stage_probe')}:raise ValueError('Exact B2 four summary CSV rows required; stage probes cannot replace timed medians')
    accuracy=reference_audit['original_reference_fullV_FP64']
    if accuracy.get('run_id')!=B2_RUN or accuracy.get('n_observed')!=43344 or accuracy.get('full_V_origins')!=43 or accuracy.get('artifact_sha256')!=B2_ARTIFACT or accuracy.get('manifest_sha256')!=B2_MANIFEST:raise ValueError('Same existing B2 V43 selected artifact required')
    expected=dict(run_id=B2_RUN,dataset='jena',family='B2',seed='92601',C='21',K='6',L='512',H='48',accuracy_reused='True',new_training='0',n_origins='43',n_observed='43344',device='cpu',CPU_threads='2',worker_RSS_limit_bytes='3221225472',trials='20',artifact_sha256=B2_ARTIFACT,accuracy_manifest_sha256=B2_MANIFEST,audit_sha256=B2_AUDIT_SHA,F_resident_bytes_scope='NOT_SEPARATELY_RECORDED',model_loading_in_whole_trial_timer='False')
    for row in reference_rows:
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Existing B2 same CPU2/RSS3GiB cohort/provenance required')
        if number(row['normalized_MSE'])!=number(accuracy['mse']) or number(row['normalized_MAE'])!=number(accuracy['mae']):raise ValueError('B2 source CSV accuracy differs from actual reused fullV audit')
    baseline=dict(role='B2_REFERENCE',run_id=B2_RUN,MSE=reference_rows[0]['normalized_MSE'],MAE=reference_rows[0]['normalized_MAE'])
    for batch in (1,8):
        timed=[r for r in reference_rows if r['batch_size']==str(batch) and r['phase']=='timed']
        costs=[r for r in reference_audit.get('costs',[]) if r.get('batch')==batch]
        if len(timed)!=1 or len(costs)!=1:raise ValueError('Exact B2 timed batch and audited cost required')
        summaries=[s for s in costs[0]['summaries'] if s.get('phase')=='timed' and s.get('batch_size')==batch and s.get('trials')==20]
        if len(summaries)!=1 or number(timed[0]['median_seconds'])!=number(summaries[0]['median_seconds']):raise ValueError('B2 timed median must copy actual audited whole20 summary')
        baseline[f'B{batch}_median_seconds']=timed[0]['median_seconds']
    return dict(rows=[baseline,*rows],p90_confirmed={1:False,8:False},pca_reference=None)


def validate_artists(artists,data):
    if len(artists)!=4 or {a.get('column') for a in artists}!=set(COLUMNS):raise ValueError('Only four requested error/median panels allowed')
    for artist in artists:
        if artist['values']!=[number(row[artist['column']]) for row in data['rows']]:raise ValueError('Artist values must equal the twelve unchanged CSV numbers')


def render(data,style,csv_sha,audit_sha):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path=Path(os.environ['WINDIR'])/'Fonts/malgun.ttf'
    if not font_path.is_file():raise FileNotFoundError('Existing Malgun font required; no installation')
    font=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','svg.hashsalt':'INITIAL_FAMILY_WITH_B2_REFERENCE_v1','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    style.LABELS=LABELS;style.COLORS=('#687D8D','#D55E00','#0072B2')
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.9));fig.subplots_adjust(left=.08,right=.97,bottom=.255,top=.79,wspace=.28,hspace=.55)
    try:
        fig.suptitle('Jena: 기존 B2와 LoRA / B4의 개발 V 오차·전체 추론 비용',x=.08,y=.966,ha='left',fontsize=15,fontweight='bold',color='#172B3A')
        fig.text(.08,.922,'C21 / L512 / H48 · CPU2 · V43 / observed labels43,344 · seed92601 · 모든 축0부터',fontsize=9,color='#455A64')
        fig.text(.08,.882,'B2 정확도=기존 CPU fullV 재사용, 새 검증/비용 · LoRA/B4=새 완료 조건 · warm whole20 median / batch',fontsize=8.7,color='#455A64')
        fig.text(.08,.842,'새 CSV SHA '+csv_sha[:12]+' · 실제 PASS audit SHA '+audit_sha[:12],fontsize=8.3,color='#536471')
        artists=style.draw_panels(list(axes.flat),data);validate_artists(artists,data)
        fig.text(.08,.196,'B8 = 8개 입력 한 batch 전체 시간. LoRA merge·모델/F pool cold loading은 warm whole timer에서 제외.',fontsize=8.4)
        fig.text(.08,.157,CAPTION,fontsize=8.3,linespacing=1.7,color='#536471')
        fig.text(.08,.088,'F-only resident bytes=별도 기록 없음(NA). 표시 숫자만 반올림; 원 CSV 수치·단위 보존. 전체 MASTER 미완료.',fontsize=8.3,color='#536471')
        fig.text(.08,.05,'노출량은 같고 F 폭·학습률·모수는 다름: 동일 표현/동등 compute 또는 순수 encoder 효과 비교가 아님.',fontsize=8.3,color='#536471')
        svg,png=io.BytesIO(),io.BytesIO();fig.savefig(svg,format='svg',metadata={'Date':None,'Creator':'Audited Jena initial families and B2 reference'});fig.savefig(png,format='png',dpi=145,metadata={'Software':'Audited Jena initial families and B2 reference'})
        clean=('\n'.join(line.rstrip() for line in svg.getvalue().decode().splitlines())+'\n').encode()
        return clean,png.getvalue(),artists,font,sha(font_path)
    finally:plt.close(fig)


def execute(public,cache,plot_sha,report_sha,audit_path,audit_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    svg_path=scope/'figures/INITIAL_FAMILY_WITH_B2_REFERENCE_v1.svg';preview=cache/NAME/'visualization/INITIAL_FAMILY_WITH_B2_REFERENCE_v1.png'
    binding_path=scope/'INITIAL_FAMILY_FIGURE_BINDINGS_v1.json';md_path=scope/'INITIAL_FAMILY_VISUALIZATION_KO_v1.md'
    if any(p.exists() for p in (svg_path,preview,binding_path,md_path)):raise FileExistsError('Existing visualization preserved; CreateNew outputs only')
    if Path(__file__).resolve()!=scope/'plot_initial_family_recovery_v2.py' or sha(__file__)!=plot_sha:raise ValueError('ROOT current reviewed plot source required')
    if any(os.environ.get(key)!='1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')):raise ValueError('Explicit CPU/BLAS1 visualization environment required')
    helper=module(public/COMMON/'build_beijing_report.py',INPUT_HELPER_SHA,'family_plot_inputs')
    builder=module(scope/'build_initial_family_recovery_v2.py',BUILDER_SHA,'family_plot_builder')
    style=module(public/COMMON/'plot_beijing.py',STYLE_SHA,'family_plot_style')
    if time.time()>=style.DEADLINE:raise RuntimeError('Original worker deadline reached')
    inputs=helper.Inputs(public,cache)
    for name,digest in ((NAME+'/plot_initial_family_recovery_v2.py',plot_sha),(NAME+'/build_initial_family_recovery_v2.py',BUILDER_SHA),(NAME+'/audit_initial_family_recovery_v2.py',AUDITOR_SHA),(COMMON+'/build_beijing_report.py',INPUT_HELPER_SHA),(COMMON+'/plot_beijing.py',STYLE_SHA)):
        inputs.capture('public',name,digest)
    report=inputs.json('public',NAME+'/INITIAL_FAMILY_BINDINGS_v1.json',report_sha)
    audit_path=Path(audit_path).resolve()
    if not audit_path.is_relative_to(scope) or audit_path.suffix!='.json':raise ValueError('ROOT explicit actual N5 audit JSON required')
    relative=audit_path.relative_to(public).as_posix();audit=inputs.json('public',relative,audit_sha)
    original_builder=builder.load(scope/'build_initial_family.py','frozen_family_plot_validation',builder.ORIGINAL_BUILDER_SHA)
    original_builder.validate_audit(audit)
    if report.get('audit_sha256')!=audit_sha or report.get('audit_relative_path')!=relative:raise ValueError('Report must bind the same actual complete family PASS audit')
    outputs=report.get('outputs',[]);csv_relative=NAME+'/INITIAL_FAMILY_RESULTS_v1.csv'
    if len(outputs)!=2 or {r['relative_path'] for r in outputs}!={csv_relative,NAME+'/INITIAL_FAMILY_KO_v1.md'}:raise ValueError('Completed experiment report MD must precede visualization')
    for row in (*report.get('inputs',[]),*audit.get('inputs',[])):inputs.capture(row['root'],row['relative_path'],row['sha256'])
    for row in outputs:inputs.capture('public',row['relative_path'],row['sha256'])
    with inputs.path('public',csv_relative).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for name,digest in B2_FILES.items():inputs.capture('public',NAME+'/'+name,digest)
    reference_binding=inputs.json('public',NAME+'/B2_REFERENCE_REPORT_BINDINGS_v1.json',B2_FILES['B2_REFERENCE_REPORT_BINDINGS_v1.json'])
    reference_audit=inputs.json('public',NAME+'/INDEPENDENT_B2_REFERENCE_AUDIT_ROOT_v1.json',B2_AUDIT_SHA)
    for row in reference_binding['inputs']:inputs.capture(row['root'],row['relative_path'],row['sha256'])
    for row in reference_binding['outputs']:
        if row['relative_path'] not in (NAME+'/B2_REFERENCE_VALIDATION_KO_v1.md',NAME+'/B2_REFERENCE_COSTS_v1.csv'):raise ValueError('Only original B2 report MD/CSV output bindings allowed')
        inputs.capture('public',row['relative_path'],row['sha256'])
    with inputs.path('public',NAME+'/B2_REFERENCE_COSTS_v1.csv').open(encoding='utf-8-sig',newline='') as f:reference_rows=list(csv.DictReader(f))
    data=prepare_data(rows,audit,report,reference_rows,reference_audit,reference_binding);csv_sha=inputs.files[('public',csv_relative)]
    if not inputs.preserved():raise ValueError('Input bytes changed before renderer')
    started=time.perf_counter()
    with style.ResourceCheck() as resources:
        svg,png,artists,font,font_sha=render(data,style,csv_sha,audit_sha)
        if not inputs.preserved():raise ValueError('Input bytes changed during renderer')
        resources.check()
    exposures='; '.join(row['role']+': selected refit '+str(row['selected_refit_epoch'])+' epoch / '+str(row['refit_exposures'])+' exposures' for row in rows)
    md=('# Jena 초기 방법군과 기존 B2: 시각자료\n\n![Jena 세 방법의 전체 V 오차 및 CPU2 warm whole batch 지연](figures/INITIAL_FAMILY_WITH_B2_REFERENCE_v1.svg)\n\n'
        '[원 실험 보고](INITIAL_FAMILY_KO_v1.md) · [새 두 조건 원정밀도 CSV](INITIAL_FAMILY_RESULTS_v1.csv) · [기존 B2 검증 보고](B2_REFERENCE_VALIDATION_KO_v1.md) · [B2 비용 CSV](B2_REFERENCE_COSTS_v1.csv) · [그림 SHA 결속](INITIAL_FAMILY_FIGURE_BINDINGS_v1.json)\n\n'
        '같은 Jena C21/L512/H48·seed92601·CPU2·반복 개발 V43/43344 관측 라벨의 두 오차와 B1/B8 warm whole20 median을 원 CSV에서 그대로 복사했다. B2 정확도는 기존 fullV 재사용이고 새 CPU6/비용이 연결되었다. LoRA R8와 B4 K6는 실제 신규 완료된 학습/추론/비용이다. p90·CI·seed 분산은 그리지 않았다. 모든 축은0부터이며 표시 숫자만 반올림한다. B8은8개 입력 한 batch의 전체 초 단위 시간이다.\n\n'
        +CAPTION.replace('\n',' ')+' '+exposures+'. 실제 B2·LoRA·B4 모두 search1863/refit752 update·29769/12020 exposure다. LoRA는 원 C21의 effective F를 적응하고 merge하며 B2/B4의 frozen F 경로와 다르다.\n\n'
        '모델 로드·LoRA merge·앞서 생성한 F pool cold load는 warm whole timer에 포함되지 않는다. cold-start 전체 latency나 checkpoint 비용 비교가 아니다. B2 원4GiB 학습 guard/정규화 ancestry를 새3GiB 학습으로 소급 변경하지 않는다. F-only resident bytes는 별도 기록 없음(NA)이며 전체 Route weights/RSS로 대신하지 않는다.\n\n'
        '이 그림은 동일 표현·동일 decoder·동등 compute의 효과 분리가 아니다. 선택 refit1 및 update/exposure는 같고 방법·모수·K/F width·학습률은 다르다. 검사 통과를 우위·수렴·독립 일반화 또는 치료 효과로 해석하지 않는다. 전체 MASTER 미완료다. 원 MD 보고가 먼저이며 그림 실패는 다른 독립 경로 중단 gate가 아니다. 새 통계/trial 재집계·모델/NPZ/CP/Task·학습·추론·비용·프로세스 제어·Git·network는0이다. 공개 SVG와 private PNG를 만들고 입력 전후 SHA 및 실제 artist12개 값을 대사했다.\n')
    svg_path.parent.mkdir(exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    for path,blob in ((svg_path,svg),(preview,png),(md_path,md.encode())):
        with path.open('xb') as f:f.write(blob)
    proof=dict(schema='INITIAL_FAMILY_WITH_B2_REFERENCE_FIGURE_BINDINGS_v1',status='PASS_ACTUAL_AUDITED_AGGREGATE_VISUALIZATION',plot_source_sha256=plot_sha,builder_source_sha256=BUILDER_SHA,auditor_source_sha256=AUDITOR_SHA,actual_family_audit_sha256=audit_sha,report_bindings_sha256=report_sha,reference_audit_sha256=B2_AUDIT_SHA,reference_files=B2_FILES,input_SHA_before_after_equal=True,
        inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],outputs=[dict(relative_path=p.relative_to(scope).as_posix(),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)) for p,blob in ((svg_path,svg),(md_path,md.encode()))],private_preview=dict(published=False,cache_relative_path=preview.relative_to(cache).as_posix(),sha256=hashlib.sha256(png).hexdigest(),bytes=len(png)),artist_values_from_unchanged_CSV=artists,artist_scalar_count=12,CSV_precision_preserved=True,p90_or_CI_added=False,font=font,font_file_sha256=font_sha,CPU_BLAS_threads=1,own_RSS_limit_bytes=style.MEMORY_LIMIT,sampled_own_peak_RSS_bytes=resources.peak,own_OS_peak_working_set_bytes=resources.os_peak,deadline_unix=style.DEADLINE,elapsed_seconds=time.perf_counter()-started,actual_renderer_executed=True,new_accuracy_or_cost_measurements=0,report_first=True,plot_failure_blocks_other_paths=False,whole_MASTER_complete=False,independent_generalization_claimed=False,self_hash_policy='Bindings excludes itself')
    with binding_path.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return dict(status=proof['status'],public_files_written=3,private_previews_written=1,artist_scalar_count=12)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--root-triggered',action='store_true')
    for name in ('public-root','cache-root','plot-sha','report-binding-sha','audit','audit-sha'):p.add_argument('--'+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status='PREPARATION_ONLY_NOT_EXECUTED',input_reads=0,files_written=0,actual_render=0,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,matplotlib_imported='matplotlib' in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,k) is None for k in ('public_root','cache_root','plot_sha','report_binding_sha','audit','audit_sha')):p.error('ROOT explicit current source, actual completed family report/audit SHA, pinned B2 reference and fresh outputs required')
    print(json.dumps(execute(v.public_root,v.cache_root,v.plot_sha,v.report_binding_sha,v.audit,v.audit_sha)));return 0


if __name__=='__main__':raise SystemExit(main())
