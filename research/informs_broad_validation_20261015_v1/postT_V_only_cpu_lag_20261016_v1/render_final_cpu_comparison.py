"""Render a ROOT-audited seven-route Jena aggregate CSV; never open raw assets.

Default invocation is stdlib metadata only. Rendering requires --execute, an
explicit aggregate CSV SHA and a new output directory. This is figure generation,
not a new scientific audit, checkpoint selection, inference or cost measurement.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import textwrap

METHODS=(
    ('primary_jena_E1_RAW_BOLT_SMALL_k0_s92601','Raw Bolt-small','RAW',21),
    ('primary_jena_E1_RAW_RIDGE_k0_s92601','Native ridge','RAW_RIDGE',0),
    ('primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3','A-only CPU fresh G','DIRECT_G',0),
    ('postT_cpu_jena_E3_B0_k6_s92601_v1','B0: pointwise','B0',6),
    ('postT_cpu_jena_E3_B2_k6_s92601_v1','B2: full-window mixer','B2',6),
    ('postT_cpu_jena_E3_LAG_CAUSAL_k6_s92601_v1','Past-directed lags','LAG_CAUSAL',6),
    ('postT_cpu_jena_E3_LAG_BIDIR_k6_s92601_v1','Within-window bidirectional lags','LAG_BIDIR',6),
)
BASE_FIELDS=('run_id','label','kind','dataset','split','cohort','data_sha256','scaler_sha256','origins_sha256',
    'K','C','L','H','V_origins','seed','device','threads','selected_epoch','search_max_epoch','refit_epochs','refit_updates',
    'selection_rule','metric_scope','MSE','MAE','FP64_rescore_status','replay_status','replay_device','replay_origin_count',
    'replay_actual_forward','cost_scope','cost_timer_scope','cost_warmups','cost_timed_trials','cost_stage_trials','row_microbatch','checkpoint_inputs')
COST_FIELDS=('status','median_seconds','p90_seconds','min_seconds','max_seconds','parity_status','reason')
FIELDS=BASE_FIELDS+tuple('cost_B%d_%s'%(batch,name) for batch in (1,8) for name in COST_FIELDS)
COLORS=('#666666','#000000','#E69F00','#0072B2','#56B4E9','#D55E00','#009E73')
MARKERS=('o','s','^','D','v','P','X')
SCOPE=('Jena L512/H48/C21, V43, seed92601; CPU2 deployment costs and checkpoint replay.\n'
    'Post-T exploratory comparison; no CI or significance claim. Architectures and training budgets differ.\n'
    'Saved full V43 scores and six-origin CPU checkpoint replay are distinct; this renderer performs neither audit nor inference.\n'
    'A_stop selects neural checkpoints; V is evaluation only. Search depth and selected fresh refit are reported separately.')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def number(value,*,integer=False,positive=False):
    try:result=int(value) if integer else float(value)
    except (ValueError,TypeError,OverflowError):raise ValueError('Expected a numeric aggregate scalar') from None
    if not math.isfinite(result) or result<0 or (positive and result==0):raise ValueError('Nonfinite/negative or zero positive scalar')
    return result


def boolean(value,*,optional=False):
    if optional and value=='':return None
    if value in ('true','True','1'):return True
    if value in ('false','False','0'):return False
    raise ValueError('Expected an explicit CSV boolean')


def validate_rows(records):
    expected={row[0]:row[1:] for row in METHODS}
    if len(records)!=7 or len({row.get('run_id') for row in records})!=7 or {row.get('run_id') for row in records}!=set(expected):
        raise ValueError('Exactly the four approved encoders plus explicit Raw/ridge/A-only CPU-fresh G are required')
    rows={};identity=None
    for row in records:
        if set(row)!=set(FIELDS) or any(not isinstance(value,str) for value in row.values()):raise ValueError('Exact aggregate CSV schema/string cells required')
        label,kind,k=expected[row['run_id']]
        if (row['label'],row['kind'],number(row['K'],integer=True))!=(label,kind,k):raise ValueError('Explicit run ID, role, label or F input width differs')
        for field in ('data_sha256','scaler_sha256','origins_sha256'):
            if re.fullmatch('[0-9a-f]{64}',row[field]) is None:raise ValueError('Data/scaler/origin identity SHA required')
        same=tuple(row[field] for field in ('dataset','split','cohort','data_sha256','scaler_sha256','origins_sha256','device'))
        if identity is None:identity=same
        if same!=identity or row['dataset']!='jena' or row['split']!='V_select' or row['cohort']!='jena:primary:V_select:stride48' or row['device']!='cpu':
            raise ValueError('Mixed data/scaler/origin cohort, metric split or deployment device')
        for field,value in dict(C=21,L=512,H=48,V_origins=43,seed=92601,threads=2,cost_warmups=10,cost_timed_trials=20,
                cost_stage_trials=20,row_microbatch=128).items():
            if number(row[field],integer=True)!=value:raise ValueError('Dimensions or fixed CPU2 whole-cost protocol differs')
        if (boolean(row['checkpoint_inputs']) is not False or row['metric_scope']!='NORMALIZED_CHANNEL_MACRO'
                or row['cost_scope']!='fresh_process_single_route' or row['cost_timer_scope']!='physical_pipeline'
                or row['FP64_rescore_status']!='PASS'):raise ValueError('Audited full-V metrics or fresh whole-pipeline timer contract differs')
        value=dict(row,MSE=number(row['MSE']),MAE=number(row['MAE']),K=k,refit_updates=number(row['refit_updates'],integer=True))
        if kind in {'RAW','RAW_RIDGE'}:
            rule='NONE_FIXED' if kind=='RAW' else 'ORIGINAL_RIDGE'
            if row['selection_rule']!=rule or any(row[field]!='' for field in ('selected_epoch','search_max_epoch','refit_epochs')) or value['refit_updates']!=0:
                raise ValueError('Raw/analytic ridge must not fabricate neural epochs or updates')
            value.update(selected_epoch=None,search_max_epoch=None,refit_epochs=None)
        else:
            selected,search,refit=(number(row[field]) for field in ('selected_epoch','search_max_epoch','refit_epochs'))
            if (row['selection_rule']!='A_stop' or selected not in (0.,.25,1.,3.) or search!=3. or refit!=selected
                    or (value['refit_updates']==0)!=(selected==0.)):raise ValueError('Completed search3 and original A_stop-selected fresh refit must remain distinct')
            value.update(selected_epoch=selected,search_max_epoch=search,refit_epochs=refit)
        replay=row['replay_status'];forward=boolean(row['replay_actual_forward'],optional=True);count=number(row['replay_origin_count'],integer=True)
        if replay not in {'PASS','FAIL','NOT_EXECUTED'} or count not in (0,6):raise ValueError('Explicit six-origin replay status required')
        if replay=='PASS' and (row['replay_device']!='cpu' or forward is not True or count!=6):raise ValueError('PASS requires actual fresh CPU six-origin forward evidence')
        if replay=='NOT_EXECUTED' and (count!=0 or forward is True):raise ValueError('Unexecuted replay cannot claim a fresh forward')
        value['costs']={}
        for batch in (1,8):
            prefix='cost_B%d_'%batch;cost={name:row[prefix+name] for name in COST_FIELDS};status=cost['status']
            if status not in {'COMPLETE_PROTOCOL','MEASURED','FAILED','NOT_MEASURED'}:raise ValueError('Unknown whole-cost status')
            if cost['parity_status'] not in {'PASS','NOT_REQUIRED','FAIL','NOT_EXECUTED'}:raise ValueError('Explicit cost parity status required')
            if status in {'COMPLETE_PROTOCOL','MEASURED'}:
                if replay!='PASS' or cost['parity_status'] not in {'PASS','NOT_REQUIRED'} or (kind=='RAW' and cost['parity_status']!='PASS') or cost['reason']!='':
                    raise ValueError('Measured matched cost requires successful CPU replay/parity and no failure reason')
                for field in ('median_seconds','p90_seconds','min_seconds','max_seconds'):cost[field]=number(cost[field],positive=True)
                if not cost['min_seconds']<=cost['median_seconds']<=cost['p90_seconds']<=cost['max_seconds']:raise ValueError('Whole timer min/median/p90/max order differs')
                cost['measured']=True
            else:
                if any(cost[field]!='' for field in ('median_seconds','p90_seconds','min_seconds','max_seconds')):raise ValueError('Failed/unmeasured costs must be empty, never zero or surrogate timings')
                if re.fullmatch('[A-Za-z0-9 _.()+-]{1,100}',cost['reason']) is None:raise ValueError('Unavailable cost requires a short safe explicit reason')
                cost.update({field:None for field in ('median_seconds','p90_seconds','min_seconds','max_seconds')});cost['measured']=False
            value['costs'][batch]=cost
        rows[row['run_id']]=value
    return [rows[run_id] for run_id,_,_,_ in METHODS]


def plain_path(path):
    path=Path(path).absolute()
    if any(part.is_symlink() for part in (path,*path.parents)):raise ValueError('Original input/output path and parents must not be symlinks')
    return path


def read_rows(path):
    path=plain_path(path)
    if path.suffix.lower()!='.csv' or path.name=='PUBLIC_ERROR_STATS.csv' or path.stat().st_size>1024**2:
        raise ValueError('Only a small explicit aggregate CSV, not per-target statistics or linked raw assets, is supported')
    raw=path.read_bytes();reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
    if reader.fieldnames!=list(FIELDS):raise ValueError('Aggregate CSV header/order differs; use metadata schema')
    return validate_rows(list(reader)),raw


def phase_label(row):
    if row['kind']=='RAW':phase='fixed pretrained baseline'
    elif row['kind']=='RAW_RIDGE':phase='analytic ridge; no neural epoch'
    else:phase='search 3 -> select %g / refit %g ep; %d updates'%(row['selected_epoch'],row['refit_epochs'],row['refit_updates'])
    replay='' if row['replay_status']=='PASS' else ' | CPU replay '+row['replay_status']
    return '%s | K=%d\n%s%s'%(row['label'],row['K'],phase,replay)


def draw(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    outputs=[]
    def save(fig,stem):
        try:
            for suffix in ('svg','png'):
                path=output/(stem+'.'+suffix)
                with path.open('xb') as stream:
                    options=dict(format=suffix,bbox_inches='tight')
                    if suffix=='svg':options['metadata']={'Date':None}
                    else:options.update(dpi=300,metadata={'Software':'Matplotlib'})
                    fig.savefig(stream,**options)
                outputs.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
        finally:plt.close(fig)
    def style(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18,zorder=0)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','svg.hashsalt':'audited-final-cpu-comparison-v1'}):
        y=list(range(len(rows)));labels=[phase_label(row) for row in rows]
        fig,axes=plt.subplots(1,2,figsize=(15,7.4),sharey=True)
        fig.subplots_adjust(left=.31,right=.97,top=.84,bottom=.19,wspace=.19)
        for panel,(ax,metric) in enumerate(zip(axes,('MSE','MAE'))):
            ax.barh(y,[row[metric] for row in rows],color=COLORS,edgecolor='#333333',linewidth=.6,zorder=3)
            ax.set_yticks(y)
            if panel==0:ax.set_yticklabels(labels)
            else:ax.tick_params(labelleft=False)
            ax.set_ylim(len(rows)-.4,-.6);ax.set_xlim(0,max(row[metric] for row in rows)*1.23 or 1)
            for index,row in enumerate(rows):ax.annotate('%.6g'%row[metric],(row[metric],index),xytext=(4,0),textcoords='offset points',va='center',fontsize=8.5)
            ax.set_xlabel('Normalized channel-macro '+metric+' (lower is better)');ax.set_title(('A' if panel==0 else 'B')+'  Saved full V43 '+metric,loc='left',fontweight='bold');style(ax)
        fig.suptitle('Jena: audited saved accuracy and explicit search/refit budgets',x=.035,ha='left',fontsize=14,fontweight='bold')
        fig.text(.035,.027,SCOPE,fontsize=8.5,linespacing=1.45);save(fig,'ACCURACY_MSE_MAE')

        fig,axes=plt.subplots(1,2,figsize=(14,6.8),sharey=True)
        fig.subplots_adjust(left=.27,right=.98,top=.82,bottom=.2,wspace=.18)
        for panel,(ax,batch) in enumerate(zip(axes,(1,8))):
            measured=[row['costs'][batch] for row in rows if row['costs'][batch]['measured']]
            ax.set_xscale('log');ax.set_xlim(min(c['min_seconds'] for c in measured)*.6 if measured else .001,max(c['max_seconds'] for c in measured)*4 if measured else 1)
            for index,row in enumerate(rows):
                cost=row['costs'][batch]
                if cost['measured']:
                    ax.hlines(index,cost['min_seconds'],cost['max_seconds'],color=COLORS[index],linewidth=2)
                    ax.scatter(cost['median_seconds'],index,color=COLORS[index],marker='o',s=40,zorder=4)
                    ax.scatter(cost['p90_seconds'],index,color=COLORS[index],marker='^',s=40,zorder=4)
                    ax.annotate('%.4g / %.4g'%(cost['median_seconds'],cost['p90_seconds']),(cost['max_seconds'],index),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
                else:
                    ax.axhspan(index-.25,index+.25,color='#eeeeee',zorder=0)
                    ax.text(.025,index,cost['status']+': '+cost['reason'],transform=ax.get_yaxis_transform(),va='center',color='#555555',fontsize=8)
            ax.set_yticks(y)
            if panel==0:ax.set_yticklabels([row['label']+' | K=%d'%row['K'] for row in rows])
            else:ax.tick_params(labelleft=False)
            ax.set_ylim(len(rows)-.4,-.6);style(ax)
            ax.set_title(('A' if panel==0 else 'B')+'  Batch %d'%batch,loc='left',fontweight='bold');ax.set_xlabel('Whole physical pipeline (seconds per batch; log scale)')
        fig.suptitle('Jena CPU2 fresh-process cost: median, p90 and observed trial range',x=.035,ha='left',fontsize=14,fontweight='bold')
        fig.text(.035,.055,'Circle: median; triangle: p90; line: min-max timed-trial spread, not a confidence interval. Failed/missing costs have no numeric point.\n'
            'Same CPU2 protocol: warmup 10 / timed 20 / separate stage 20 / row microbatch 128 / checkpoint inputs off. No speedup ratios.\n'
            'Different architectures and learned refit lengths; no equal-parameter or equal-compute claim. Post-T, one seed; no CI.',fontsize=8.5,linespacing=1.45)
        save(fig,'CPU_WHOLE_COST_B1_B8')

        fig,axes=plt.subplots(1,2,figsize=(13.5,7))
        fig.subplots_adjust(left=.07,right=.97,top=.76,bottom=.3,wspace=.23)
        for panel,(ax,batch) in enumerate(zip(axes,(1,8))):
            measured=[row['costs'][batch] for row in rows if row['costs'][batch]['measured']]
            ax.set_xscale('log');ax.set_xlim(min(c['median_seconds'] for c in measured)*.6 if measured else .001,max(c['median_seconds'] for c in measured)*1.7 if measured else 1)
            missing=[]
            for index,row in enumerate(rows):
                cost=row['costs'][batch];code=chr(65+index)
                if cost['measured']:
                    ax.scatter(cost['median_seconds'],row['MSE'],color=COLORS[index],marker=MARKERS[index],s=60,zorder=3)
                    ax.annotate(code,(cost['median_seconds'],row['MSE']),xytext=(5,5 if index%2 else -11),textcoords='offset points',fontsize=8)
                else:missing.append(code+' '+cost['status']+' ('+cost['reason']+')')
            ax.set_ylim(0,max(row['MSE'] for row in rows)*1.18 or 1);ax.set_title(('A' if panel==0 else 'B')+'  Batch %d'%batch,loc='left',fontweight='bold')
            ax.set_xlabel('Median whole CPU2 seconds per batch (log scale)');ax.set_ylabel('Saved V43 normalized MSE (lower is better)');style(ax)
            text='Unavailable costs: '+('; '.join(missing) if missing else 'none')
            ax.text(0,-.23,'\n'.join(textwrap.wrap(text,width=67)),transform=ax.transAxes,va='top',fontsize=8,color='#555555')
        handles=[Line2D([],[],color=COLORS[index],marker=MARKERS[index],linestyle='None',label=chr(65+index)+' '+row['label']) for index,row in enumerate(rows)]
        fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,.93),frameon=False,fontsize=8)
        fig.suptitle('Jena saved accuracy versus matched CPU whole-pipeline cost',x=.07,ha='left',fontsize=14,fontweight='bold')
        fig.text(.07,.025,'No speedup ratios, statistical significance, equal-compute or independent-generalization claim. Missing costs remain listed.\n'
            'Post-T exploratory development, one seed; saved full V43 and six-origin CPU replay are different scopes. Exact IDs and source CSV: FIGURE_BINDING.json.',fontsize=8.3,linespacing=1.45)
        save(fig,'ACCURACY_CPU_COST_SCATTER')
    return outputs


def render(csv_path,output,expected_csv_sha):
    csv_path,output=plain_path(csv_path),plain_path(output)
    csv_path,output=csv_path.resolve(),output.resolve()
    if output.exists():raise FileExistsError('Previous figure directory must be preserved')
    if re.fullmatch('[0-9a-f]{64}',expected_csv_sha or '') is None or sha(csv_path)!=expected_csv_sha:raise ValueError('Explicit aggregate CSV SHA differs')
    rows,raw=read_rows(csv_path);source_sha=sha(__file__)
    if hashlib.sha256(raw).hexdigest()!=expected_csv_sha:raise ValueError('Input aggregate changed during read')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'INPUT_SNAPSHOT.csv').open('xb') as stream:stream.write(raw)
    files=draw(rows,output)
    if sha(csv_path)!=expected_csv_sha or sha(__file__)!=source_sha:raise ValueError('Input/source changed while rendering; partial output preserved')
    binding=dict(status='RENDERED_EXPLICIT_AGGREGATE_ONLY',source_file=Path(__file__).name,source_sha256=source_sha,
        input_csv_file=csv_path.name,input_csv_sha256=expected_csv_sha,input_copy_file='INPUT_SNAPSHOT.csv',input_copy_sha256=sha(output/'INPUT_SNAPSHOT.csv'),
        row_count=7,explicit_methods=[dict(run_id=row['run_id'],label=row['label'],kind=row['kind'],K=row['K']) for row in rows],
        matched_identity={name:rows[0][name] for name in ('dataset','split','cohort','data_sha256','scaler_sha256','origins_sha256','device','threads')},
        exact_schema=list(FIELDS),files=files,PNG_dpi=300,SVG_vector=True,source_and_input_bytes_unchanged=True,
        accuracy_source='ROOT-provided FP64-rescored full V43 aggregate; no original statistics or arrays read here',
        checkpoint_replay_scope='ROOT-provided six-origin CPU replay status; not a CPU full-V reinference claim',
        device_field_scope='CPU deployment costs and checkpoint replay; original saved full-V inference device is not inferred',
        missing_cost_states=[dict(run_id=row['run_id'],batch=batch,status=row['costs'][batch]['status'],reason=row['costs'][batch]['reason'])
            for row in rows for batch in (1,8) if not row['costs'][batch]['measured']],
        speedup_ratios_computed=False,new_scientific_validation_performed=False,model_or_Task_or_NPZ_accessed=False,
        checkpoint_selection_performed=False,original_campaign_T_or_selection_modified=False,
        post_T_exploratory_comparison=True,single_seed=92601,confidence_intervals_or_significance_computed=False,
        equal_parameters_or_training_compute_claimed=False,figure_generation_is_not_scientific_campaign_completion=True)
    with (output/'FIGURE_BINDING.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(binding,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    readme=('# Jena 최종 CPU 비교 집계 그림\n\nROOT가 별도로 검산한 집계 CSV의 정확한 bytes만 사용했다. 렌더러는 Task·NPZ·가중치·개별 정답/오차 로그·원 비용 자료를 열지 않으며 새 검산이나 선택을 수행하지 않는다.\n\n'
        '- [정확한 원 CSV 복사](INPUT_SNAPSHOT.csv)\n- [run ID·입력·그림 SHA와 적용 범위](FIGURE_BINDING.json)\n'
        '- [정확도 SVG](ACCURACY_MSE_MAE.svg) / [PNG](ACCURACY_MSE_MAE.png)\n'
        '- [CPU B1/B8 비용 SVG](CPU_WHOLE_COST_B1_B8.svg) / [PNG](CPU_WHOLE_COST_B1_B8.png)\n'
        '- [정확도-비용 SVG](ACCURACY_CPU_COST_SCATTER.svg) / [PNG](ACCURACY_CPU_COST_SCATTER.png)\n\n'
        '정확도는 저장된 전체 V43 정규화 channel-macro MSE/MAE다. CPU의 6 origin checkpoint 재추론과 구분한다. A_stop 선택 epoch, 검색3epoch, 선택한 fresh A60 refit 길이는 같은 개념이 아니다. Raw·ridge에는 신경망 epoch를 붙이지 않았다.\n\n'
        'FAILED/NOT_MEASURED 비용은 0이나 다른 조건의 시간으로 대체하지 않았다. 비용 선은 timed20의 min–max, 원은 median, 삼각형은 p90이며 CI가 아니다. CPU2·warm10/timed20/separate-stage20·row128·checkpoint off의 fresh whole physical pipeline 조건만 허용한다. 속도 배율은 계산하지 않았다.\n\n'
        'T 관측 이후의 탐색적 V 재사용 비교, seed 하나다. 서로 다른 구조·학습량의 비교이며 동일 파라미터/연산량·충분 수렴·독립 일반화나 원 연구 전체 완료를 뜻하지 않는다. G는 CSV의 명시적 A-only CPU fresh ID만 허용한다.\n')
    with (output/'README_KO.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write(readme)
    return binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--csv');parser.add_argument('--csv-sha');parser.add_argument('--output')
    args=parser.parse_args(argv)
    if args.execute:
        if not all((args.csv,args.csv_sha,args.output)):parser.error('Explicit audited aggregate CSV, SHA and new output are required')
        result=render(Path(args.csv),Path(args.output),args.csv_sha)
        print(json.dumps(dict(status=result['status'],rows=result['row_count'],figures=3,new_scientific_validation=False)))
    else:
        if any((args.csv,args.csv_sha,args.output)):parser.error('Input files are read only with explicit --execute')
        print(json.dumps(dict(status='METADATA_ONLY_FINAL_CPU_COMPARISON_RENDERER',files_written=0,input_files_read=0,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,matplotlib_imported='matplotlib' in sys.modules,
            schema=list(FIELDS),expected_methods=[dict(run_id=r,label=label,kind=kind,K=k) for r,label,kind,k in METHODS],
            missing_cost_numeric_cells='empty, never zero',cost_statuses=['COMPLETE_PROTOCOL','MEASURED','FAILED','NOT_MEASURED'],
            measured_protocol='CPU2, fresh process, physical_pipeline seconds, 10 warmups / 20 timed / 20 separate stage / row128 / checkpoint false',
            new_model_array_Task_inference_or_cost_operations=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
