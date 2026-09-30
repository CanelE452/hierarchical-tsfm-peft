"""Prepare the frozen fresh unit without exposing TEST statistics or predictions."""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from runtime_confirmation_v12 import (HERE, ROOT, CACHE, Job, protocol, digest, artifact,
                                      import_file, save_json, configure)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_new(path, arrays):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return artifact(path)


def prepare():
    with Job('cpu_analysis','confirmation_prepare01',reserve_s=150,
             metadata={'purpose':'frozen new-unit TRAIN statistics and sealed array packaging','fit_count':0}) as job:
        configure()
        p = protocol()
        manifest_path = HERE/'confirmation_data_manifest.json'
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest = {}
        pca = import_file(ROOT/'research/tsfm_peft_development_20260926/linear_fulltrain.py',
                          'confirmation_train_pca_primitive').pca_basis
        for dataset,unit in p['units'].items():
            contract_path = HERE/f'data_contract_confirmation_{dataset}.json'
            if contract_path.exists():
                raise FileExistsError(contract_path)
            source=unit['source']
            raw_receipt=artifact(source['raw_electricity_csv'],source['raw_electricity_sha256'])
            metadata_receipt=artifact(source['raw_metadata_csv'],source['raw_metadata_sha256'])
            columns=unit['columns']
            meta=pd.read_csv(metadata_receipt['path'])
            selected=meta[meta['building_id'].isin(columns)]
            assert len(selected)==unit['C']
            assert (selected['site_id']==unit['site_id']).all()
            assert (selected['primaryspaceusage']==unit['primaryspaceusage']).all()
            assert (selected['electricity'].astype(str).str.strip()=='Yes').all()
            frame=pd.read_csv(raw_receipt['path'],usecols=['timestamp',*columns])
            times=pd.to_datetime(frame['timestamp'],errors='raise')
            assert not times.duplicated().any() and times.is_monotonic_increasing
            choose=(times>=pd.Timestamp(unit['context_start']))&(times<pd.Timestamp(unit['splits']['test_b'][1]))
            frame=frame.loc[choose].reset_index(drop=True)
            times=times.loc[choose].reset_index(drop=True)
            assert (times.diff().dropna()==pd.Timedelta(seconds=unit['interval_seconds'])).all()
            assert times.iloc[0]==pd.Timestamp(unit['context_start'])
            assert times.iloc[-1]+pd.Timedelta(seconds=unit['interval_seconds'])==pd.Timestamp(unit['splits']['test_b'][1])
            stamps=times.to_numpy(dtype='datetime64[ns]')
            bounds={name:tuple(int(np.searchsorted(stamps,np.datetime64(t,'ns'))) for t in limits)
                    for name,limits in unit['splits'].items()}
            raw=frame[columns].to_numpy(dtype=np.float32)
            finite=np.isfinite(raw)
            start,end=bounds['train']
            train=raw[start:end]
            observed=np.isfinite(train).mean(0)
            variation=np.nanstd(train,axis=0)
            assert len(train)==1992 and (observed>=.95).all() and (variation>0).all()
            tick=time.perf_counter()
            mean=np.nanmean(train,axis=0).astype(np.float32)
            std=np.nanstd(train,axis=0).astype(np.float32)
            std=np.where(std<1e-6,1.,std).astype(np.float32)
            x=np.where(finite,(raw-mean)/std,0.).astype(np.float32)
            basis=pca(x[start:end],unit['K']).astype(np.float32)
            statistic_seconds=time.perf_counter()-tick
            assert basis.shape==(unit['C'],unit['K'])
            assert np.max(np.abs(basis.T@basis-np.eye(unit['K'])))<=1e-5
            origins={name:np.arange(max(a,512),b-48+1,1 if name=='train' else unit['eval_stride'],dtype=np.int64)
                     for name,(a,b) in bounds.items()}
            assert {name:len(value) for name,value in origins.items()}==unit['expected_origins']
            for name,values in origins.items():
                a,b=bounds[name]
                assert np.all(values>=max(a,512)) and np.all(values+48<=b)
            common={'basis':basis,'columns':np.asarray(columns,dtype='U'), 'mean':mean,'std':std,
                    'train_start':np.array(start),'train_end':np.array(end),
                    **{name+'_origins':values for name,values in origins.items()}}
            trainval_end=bounds['val'][1]
            local=CACHE/'confirmation_data'/dataset
            trainval=save_new(local/'trainval.npz',{**common,'x':x[:trainval_end],
                            'finite':finite[:trainval_end],'times':stamps[:trainval_end].astype('int64')})
            test=save_new(local/'test.npz',{**common,'x':x,'finite':finite,'times':stamps.astype('int64')})
            contract={'dataset':dataset,'unit':unit,'source':{'raw':raw_receipt,'metadata':metadata_receipt,
                'variant':'BDG2 raw electricity; no cleaned-source substitution',
                'license_title_at_pinned_revision':'Attribution-ShareAlike 4.0 Unported',
                'license_url':'https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/9b97ccbe90096aff42ed4fd6493bf7ae692d7118/LICENSE'},
                'trainval':trainval,'test':test,'bounds':bounds,'origin_counts':unit['expected_origins'],
                'origin_hashes':{name:array_hash(value) for name,value in origins.items()},
                'basis_sha256':array_hash(basis),'statistic_fit_scope':'TRAIN only; nonpredictive preparation statistics',
                'statistic_operations':{'standardization':1,'PCA':1,'seconds':statistic_seconds},
                'train_observed_fraction':observed.tolist(),
                'normalization':'TRAIN nanmean/nanstd ddof0; std<1e-6 to1; missing input to standardized0',
                'target_mask':'raw finite mask before input replacement; observed zero retained',
                'time_basis':'source-naive hourly grid; metadata US/Eastern retained, no invented DST/UTC mapping',
                'test_access_scope':'Values mechanically packaged with TRAIN statistics; no TEST statistics/EDA/performance output',
                'protocol_sha256':digest(HERE/'confirmation_protocol.json')}
            save_json(contract_path,contract)
            manifest[dataset]={'contract_path':str(contract_path),'contract_sha256':digest(contract_path),
                               'trainval':trainval,'test':test}
            job.heartbeat(dataset+' fixed arrays packaged without TEST statistics')
        save_json(manifest_path,manifest)
        print('Fixed Peacock Education C13/K4 arrays prepared; TRAIN1992, origins1945/30/40/40. TEST statistics not inspected.',flush=True)


if __name__=='__main__':
    prepare()
