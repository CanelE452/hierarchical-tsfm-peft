"""Public M_fit/M_select heterogeneity summaries; no V loader."""
import numpy as np

from . import common,data_contract


def build(ctx):
    rows,heterogeneity,spectra=[],[],[]
    for dataset in common.DATASETS:
        base=data_contract.load(ctx,dataset,common.SEEDS[0])
        probe=common.read(common.CACHE/'probes'/dataset/'RESULT.json')
        rows.extend(probe['rows'])
        diagnostics=common.read(common.CACHE/'probes'/dataset/'DIAGNOSTICS.json')
        common.atomic(common.REPORT/f'GEOMETRY_{dataset}.json',diagnostics)
        for j,(v,cumulative) in enumerate(zip(diagnostics['spectrum'],diagnostics['cumulative_variance'])):
            spectra.append(dict(dataset=dataset,component=j+1,eigenvalue=v,cumulative_variance=cumulative,
                scope='M_fit standardized past-window covariance; causal-filled inputs; targets remain original observed-only'))
        predictions=[('G',None,base['g']),('RAW',None,base['raw'])]
        for kind in ('MEAN','OWN','ALL'):
            with np.load(common.CACHE/'probes'/dataset/(kind+'_search_predictions.npz'),allow_pickle=False) as a:
                predictions.append((kind,None,base['g']+a['correction']))
        for seed in common.SEEDS:
            data=data_contract.load(ctx,dataset,seed)
            alpha=common.read(common.CACHE/'probes'/dataset/f'alpha_s{seed}.json')
            for row in alpha['rows']:
                for split in ('M_fit','M_select'):
                    rows.append(dict(dataset=dataset,method='S_DIFF_'+row['label'],seed=seed,fit_split='M_fit',
                        report_split=split,alpha=row['alpha'],mse=row[split+'_mse'],mae=row[split+'_mae'],
                        n_observed=row[split+'_n_observed'],refit_full_Mcal_alpha=alpha['refit_alpha']['alpha']))
            predictions.extend([('S',seed,data['s']),('ALPHA_fitted',seed,data['g']+alpha['fit_alpha']['alpha']*(data['s']-data['g']))])
        for split,idx in (('M_fit',base['split']['fit_indices']),('M_select',base['split']['select_indices'])):
            y,mask=base['y'][idx],base['mask'][idx]
            for method,seed,p in predictions:
                delta=np.where(mask,p[idx].astype(np.float64)-y.astype(np.float64),0.)
                for c,name in enumerate(base['task']['columns']):
                    n=int(mask[:,:,c].sum())
                    heterogeneity.append(dict(dataset=dataset,split=split,method=method,seed=seed,axis='channel',index=c,label=name,
                        mse=float(np.square(delta[:,:,c]).sum()/n),mae=float(np.abs(delta[:,:,c]).sum()/n),count=n))
                for h in range(y.shape[1]):
                    counts=mask[:,h].sum(0)
                    active=counts>0
                    heterogeneity.append(dict(dataset=dataset,split=split,method=method,seed=seed,axis='horizon',index=h,label=h+1,
                        mse=float((np.square(delta[:,h]).sum(0)[active]/counts[active]).mean()),
                        mae=float((np.abs(delta[:,h]).sum(0)[active]/counts[active]).mean()),count=int(counts.sum())))
                for b,ix in enumerate(np.array_split(np.arange(len(idx)),4)):
                    score=common.score(p[idx][ix],y[ix],mask[ix])
                    heterogeneity.append(dict(dataset=dataset,split=split,method=method,seed=seed,axis='time_block',index=b,label=b+1,
                        mse=score['mse'],mae=score['mae'],count=score['n_observed'],
                        first_origin=int(base['origins'][idx[ix[0]]]),last_origin=int(base['origins'][idx[ix[-1]]])))
    common.csv_write(common.REPORT/'RESIDUAL_PROBES.csv',rows)
    common.csv_write(common.REPORT/'HETEROGENEITY.csv',heterogeneity)
    common.csv_write(common.REPORT/'COVARIANCE_SPECTRUM.csv',spectra)
    common.event('M_SELECT_DIAGNOSTICS',status='COMPLETE',rows=len(heterogeneity),new_V_used=False)
