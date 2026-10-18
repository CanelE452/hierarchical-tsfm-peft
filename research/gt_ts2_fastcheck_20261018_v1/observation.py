"""Mask-only descriptive lookback checks; the primary cohort is never replaced."""
import csv
import io
import json

import numpy as np

from . import common as c
from .analysis import paired, score


def complete_history_mask(mask,origins,length_a,length_b):
    mask=np.asarray(mask);origins=np.asarray(origins)
    if (mask.ndim!=2 or mask.dtype!=np.bool_ or origins.ndim!=1 or origins.dtype.kind not in 'iu'
            or min(length_a,length_b)<1 or (origins<max(length_a,length_b)).any() or (origins>len(mask)).any()):
        raise ValueError('History completeness requires legal past-only windows')
    return np.asarray([mask[o-length_a:o].all() and mask[o-length_b:o].all() for o in origins],bool)


def subset_scores(candidate,reference,selected):
    full,_=paired(candidate,reference)
    selected=np.asarray(selected)
    if selected.dtype!=np.bool_ or selected.shape!=(len(candidate['origins']),):
        raise ValueError('Diagnostic subset must select the same paired origins')
    result=dict(status='COMPLETE' if selected.any() else 'NO_FULLY_OBSERVED_HISTORY_CASES',
        primary_origin_count=len(selected),diagnostic_origin_count=int(selected.sum()),
        primary_cohort_replaced=False,diagnostic_origins_sha256=c.array_sha(candidate['origins'][selected]),
        full_candidate_MSE=full['candidate_mse'],full_reference_MSE=full['reference_mse'],
        full_MSE_relative_percent=full['mse_relative_percent'])
    if selected.any():
        cand=score(candidate['prediction'][selected],candidate['target'][selected],candidate['mask'][selected])
        ref=score(reference['prediction'][selected],reference['target'][selected],reference['mask'][selected])
        result.update(candidate_MSE=cand['mse'],reference_MSE=ref['mse'],candidate_MAE=cand['mae'],reference_MAE=ref['mae'],
            MSE_relative_percent=(cand['mse']-ref['mse'])/ref['mse']*100,
            MAE_relative_percent=(cand['mae']-ref['mae'])/ref['mae']*100)
    return result


def generate():
    from .execution import context
    from .reporting import _comparisons
    data=context()['data'];rows=[]
    for name,seed,candidate,reference in _comparisons():
        if name not in ('RAW_LOOKBACK','FROZEN_GT_LOOKBACK','ADAPTED_GT_LOOKBACK','PRE_VS_GT','PRE_VS_RAND'):
            continue
        metadata=[c.read_json(c.REPORT/'receipts'/(run_id+'.json')) for run_id in (candidate,reference)]
        bundles=[]
        for run_id,receipt in zip((candidate,reference),metadata):
            path=c.CACHE/'predictions'/(run_id+'.npz')
            if c.file_sha(path)!=receipt['artifact_sha256']:
                raise ValueError('Changed prediction in observation diagnostic')
            with np.load(path,allow_pickle=False) as archive:
                bundles.append({key:archive[key].copy() for key in ('prediction','target','mask','origins','mean','std')})
        selected=complete_history_mask(data['mask'],data['val_origins'],metadata[0]['L'],metadata[1]['L'])
        rows.append(dict(comparison=name,seed=seed,candidate=candidate,reference=reference,
                         candidate_L=metadata[0]['L'],reference_L=metadata[1]['L'],
                         criterion='every physical history cell observed in BOTH windows; mask-only, no error threshold',
                         criterion_used_for_selection=False,registered_before_new_VAL_scores=False,
                         role='supplementary descriptive diagnostic required by specification; not a replacement primary cohort',
                         **subset_scores(bundles[0],bundles[1],selected)))
    stream=io.StringIO(newline='')
    fields=sorted({key for row in rows for key in row})
    writer=csv.DictWriter(stream,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    path=c.REPORT/'OBSERVATION_CONTEXT_STATS.csv'
    with path.open('xb') as output:output.write(stream.getvalue().encode('utf-8'))
    c.seal_json(c.REPORT/'OBSERVATION_DIAGNOSTIC.json',dict(status='COMPLETE',rows=len(rows),
        CSV_sha256=c.file_sha(path),source_sha256=c.file_sha(c.CODE/'observation.py'),
        criterion_registration='Mask-only strict complete-history rule recorded after Phase A; post-hoc descriptive, never score selection',
        primary_origin_count=371,primary_cohort_changed=False,new_model_fits=0))
    return dict(status='COMPLETE',rows=len(rows))


if __name__=='__main__':
    print(json.dumps(generate()))
