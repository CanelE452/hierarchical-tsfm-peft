"""Render v7 evidence figures from completed result JSON only."""
import csv
import json
from pathlib import Path
from runtime_v7 import HERE, Job, save_json, digest

LABELS={'level_res':'LEVEL','old_res':'OLD residual','level_raw':'Matched RAW','level_only':'LEVEL only',
        'compress':'Compress','f0':'F0','lora':'LoRA','direct_nlinear':'Direct NLinear',
        'level_linear_res':'Latent linear residual','shared':'Shared ridge'}
COLORS={'level_res':'#0072B2','old_res':'#999999','level_raw':'#56B4E9','level_only':'#009E73',
        'compress':'#777777','f0':'#E69F00','lora':'#D55E00','direct_nlinear':'#CC79A7',
        'level_linear_res':'#009E73','shared':'#000000'}

def load(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))

def save_figure(fig,name,output):
    entries=[]
    for ext in ('png','pdf','svg'):
        path=output/(name+'.'+ext)
        fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        entries.append({'path':str(path),'sha256':digest(path)})
    plt.close(fig)
    return entries

with Job('evidence_figures01',category='cpu_analysis',reserve_seconds=120):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    out=HERE/'figures'
    out.mkdir(exist_ok=True)
    robin,jena=load('robin_eval01.json'),load('jena_eval01.json')
    manifest={'inputs':{n:digest(HERE/n) for n in ('robin_eval01.json','jena_eval01.json','cost_summary.json','cost_gpu_rows.json')},'figures':{}}
    order=['level_res','old_res','level_raw','level_only','compress','f0','lora','direct_nlinear']
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for axis,data,title in zip(axes,[robin,jena],['A  Robin: exposed follow-up','B  Jena 2024: exposed domain extension']):
        for y,role in enumerate(order):
            scores=data['role_scores'][role]
            center=scores['combined']['mse']
            ids=scores['combined']['run_ids']
            seeds=[data['models'][i]['scores']['combined']['mse'] for i in ids]
            axis.plot([min(seeds),max(seeds)],[y,y],color=COLORS[role],lw=3)
            axis.plot(center,y,'o',color=COLORS[role],ms=6)
            axis.plot(scores['test_a']['mse'],y-.12,'|',color=COLORS[role],ms=10)
            axis.plot(scores['test_b']['mse'],y+.12,'x',color=COLORS[role],ms=5)
            axis.annotate(f'{center:.4f}',(center,y),xytext=(6,4),textcoords='offset points',fontsize=8)
        axis.set_yticks(range(len(order)),[LABELS[k] for k in order])
        axis.invert_yaxis()
        axis.set_xlabel('Masked channel-macro MSE (TRAIN-standardized)')
        axis.set_title(title,loc='left',fontsize=11)
        axis.grid(axis='x',alpha=.2)
        axis.set_xlim(left=0)
    fig.legend(handles=[Line2D([],[],marker='o',color='black',ls='-',label='Mean individual-seed loss; segment = seed range'),
                        Line2D([],[],marker='|',color='black',ls='',label='Period A'),
                        Line2D([],[],marker='x',color='black',ls='',label='Period B')],
               loc='outside lower center',ncol=3,frameon=False)
    manifest['figures']['accuracy']=save_figure(fig,'accuracy_fixed_comparisons',out)
    space_path=HERE/'diagnostics_corrected/robin_complete_row_space_summary_corrected.csv'
    manifest['inputs'][str(space_path.relative_to(HERE))]=digest(space_path)
    with space_path.open(newline='',encoding='utf-8') as stream:
        space=[r for r in csv.DictReader(stream) if r['member_id']=='mean_of_member_losses']
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    space_roles=['level_res','old_res','level_raw','level_only','f0','lora']
    for axis,period,title in zip(axes,['test_a','test_b'],['A  Complete targets: 1920/1920 rows','B  Complete targets: 1918/1920 rows']):
        rows={r['role']:r for r in space if r['period']==period}
        retained=[float(rows[r]['p_retained_space_mse']) for r in space_roles]
        discarded=[float(rows[r]['q_residual_space_mse']) for r in space_roles]
        axis.bar(np.arange(len(space_roles)),retained,color='#E69F00',label='Retained PCA space P')
        axis.bar(np.arange(len(space_roles)),discarded,bottom=retained,color='#0072B2',label='Discarded space Q')
        axis.set_xticks(np.arange(len(space_roles)),[LABELS[r] for r in space_roles],rotation=35,ha='right')
        axis.set_title(title,loc='left',fontsize=10)
        axis.set_ylabel('Complete-row error energy per channel')
        axis.set_ylim(bottom=0)
        axis.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Robin diagnosis: mean seed errors; distinct from masked macro MSE',fontsize=11)
    manifest['figures']['diagnostics']=save_figure(fig,'robin_complete_row_error_spaces',out)
    cost=load('cost_summary.json')
    groups=[r for r in cost['groups'] if r['device']=='cuda' and r['batch']==4]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for row in groups:
        family=row['family']
        y=robin['role_scores'][family]['combined']['mse']
        variant,micro=row['variant'],row['micro']
        marker='o' if variant=='original' else 's' if micro in (68,20) and (micro==68 or family=='level_res') else '^' if micro==20 else 'v'
        mem=row['mean_seed_median_peak_allocated_bytes']/1024**2
        low,high=np.asarray(row['observed_peak_allocated_range_bytes'])/1024**2
        throughput=row['mean_seed_median_throughput']
        axes[0].errorbar(mem,y,xerr=[[max(0,mem-low)],[max(0,high-mem)]],fmt=marker,color=COLORS[family],
                         mfc='white' if variant=='original' else COLORS[family],ms=7,capsize=2)
        vals=[f['origins_per_second'] for f in cost['fits'] if f['family']==family and f['batch']==4 and
              f['variant']==variant and f['micro']==micro and f['device']=='cuda']
        tlo=min(v['min'] for v in vals);thi=max(v['max'] for v in vals)
        axes[1].errorbar(throughput,y,xerr=[[max(0,throughput-tlo)],[max(0,thi-throughput)]],fmt=marker,color=COLORS[family],
                         mfc='white' if variant=='original' else COLORS[family],ms=7,capsize=2)
    for ax in axes:
        ax.set_ylabel('Robin masked channel-macro MSE')
        ax.grid(alpha=.2)
        ax.set_xscale('log')
    axes[0].set_xlabel('Batch 4 peak allocated GPU memory (MiB, log scale)')
    axes[1].set_xlabel('Batch 4 throughput (origins/s, log scale)')
    axes[0].set_title('A  Accuracy–memory',loc='left')
    axes[1].set_title('B  Accuracy–throughput',loc='left')
    legend=[Line2D([],[],marker='o',ls='',color=COLORS[k],label=LABELS[k]) for k in ('level_res','f0','lora','direct_nlinear')]
    legend += [Line2D([],[],marker=m,ls='',color='black',mfc='white' if m=='o' else 'black',label=label)
               for m,label in [('o','Legacy'),('s','Generic full'),('^','M=20'),('v','M=5')]]
    fig.legend(handles=legend,loc='outside lower center',ncol=4,frameon=False)
    manifest['figures']['tradeoff']=save_figure(fig,'robin_microbatch_tradeoffs',out)
    manifest['scope']='Seed-mean MSE, not ensemble; memory whiskers observed all-block/seed range; throughput whiskers range of block medians, not confidence intervals. CPU costs in tables, not same-device plot. No cross-domain pooled MSE.'
    save_json(HERE/'figure_manifest.json',manifest)

