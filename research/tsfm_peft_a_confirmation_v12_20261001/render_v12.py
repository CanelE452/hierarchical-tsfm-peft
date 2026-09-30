"""Draw recorded v12 Q ablations; no model, target access or fitting."""
import argparse
import numpy as np
from runtime_v12 import HERE, Job, read_json, save_json, artifact


def render(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = read_json(source)
    assert result['status'] == 'complete'
    families = ['A_MAIN','A_MAIN_LAST','A_FULL','base_level','f0','full_lora_mse','lora_native','direct_nlinear']
    labels = ['Main only','Main + last','A full Q','LEVEL','F0','LoRA MSE','LoRA native','NLinear']
    colors = ['#96A8B2','#DC9B35','#156D7A','#68778D','#B5BBC1','#5C507A','#8A7498','#82985C']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), constrained_layout=True)
    plotted = []
    for col, dataset in enumerate(('robin','jena','hog')):
        data = result['datasets'][dataset]
        for row, metric in enumerate(('mse','mae')):
            ax = axes[row,col]
            means = [data['summary'][family]['periods']['combined'][metric] for family in families]
            ax.bar(np.arange(len(families)), means, color=colors, width=.68)
            for x, (family, mean) in enumerate(zip(families, means)):
                models = [m for m in data['models'].values() if m['family'] == family]
                seed_values = [m['periods']['combined'][metric] for m in models]
                assert abs(float(np.mean(seed_values))-mean) < 1e-12
                jitter = np.linspace(-.12,.12,len(seed_values)) if len(seed_values)>1 else [0]
                ax.scatter(x+np.asarray(jitter),seed_values,s=19,c='#152632',zorder=3)
                ax.text(x, max(seed_values)+max(means)*.018, f'{mean:.3f}',ha='center',va='bottom',fontsize=8)
                plotted.append({'dataset':dataset,'metric':metric,'family':family,'mean_seed_loss':mean,'seed_values':seed_values})
            ax.set_xticks(np.arange(len(families)),labels,rotation=38,ha='right',fontsize=9)
            ax.set_ylabel(metric.upper()+' (TRAIN-standardized)')
            ax.set_title(dataset.title()+' — '+metric.upper(),loc='left',fontweight='bold')
            ax.set_ylim(0,max(means)*1.2)
            ax.grid(axis='y',alpha=.16)
            ax.set_axisbelow(True)
            for side in ('top','right'):
                ax.spines[side].set_visible(False)
    fig.suptitle('Fixed latent-LoRA A: what does the Q forecast add?',fontsize=19,fontweight='bold')
    fig.supxlabel('Previously exposed development data. Bars: mean seed losses; dots: selected seeds (not confidence intervals).\nMain-only and last-only use the same trained A main path. No independent no-Q fitting or new confirmation.',fontsize=10)
    directory=HERE/'figures'
    directory.mkdir(exist_ok=True)
    outputs=[]
    for ext in ('png','svg'):
        path=directory/('fixed_a_q_contribution.'+ext)
        if path.exists():
            raise FileExistsError('Preserve prior render; use a documented repair if needed')
        fig.savefig(path,dpi=180)
        outputs.append(artifact(path))
    plt.close(fig)
    save_json(HERE/'figure_receipt.json',{'source':artifact(source),'outputs':outputs,'plotted_values':plotted,'new_model_runs':0,'same_checkpoint_ablations':True,'independent_confirmation':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    with Job('cpu_check','render_q01',reserve_s=90):
        render(HERE/args.source)
