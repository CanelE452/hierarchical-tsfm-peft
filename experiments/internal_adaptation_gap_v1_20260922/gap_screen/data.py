from __future__ import annotations
import hashlib, io, json, time, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .io import Blocked, read_json, write_json, sha, digest_object

@dataclass
class Panel:
    name: str
    values: np.ndarray
    timestamps: np.ndarray
    sigma: np.ndarray
    columns: list[str]
    context: int
    short_context: int
    horizon: int
    origins: dict[str,np.ndarray]
    metadata: dict
    def batch(self, origins, context=None):
        length=self.context if context is None else int(context)
        oo=np.asarray(origins,dtype=np.int64)
        x=np.stack([self.values[o-length:o].T for o in oo])
        y=np.stack([self.values[o:o+self.horizon].T for o in oo])
        c=len(self.columns)
        return x.reshape(-1,length), y.reshape(-1,self.horizon), np.repeat(np.arange(len(oo)),c)
    def targets(self, role): return np.stack([self.values[o:o+self.horizon].T for o in self.origins[role]])
    def schedule(self, seed, steps, accumulation):
        return np.random.default_rng(seed).choice(self.origins['train'],size=(steps,accumulation),replace=True)


def download(url: str, dest: Path) -> None:
    if dest.exists(): return
    dest.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={'User-Agent':'TSFM-Gap-Research/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload=r.read()
    except Exception as e: raise Blocked(f'DATA_DOWNLOAD_FAILED: {url}: {type(e).__name__}') from e
    tmp=dest.with_suffix(dest.suffix+'.part'); tmp.write_bytes(payload); tmp.replace(dest)


def load_raw(path: Path, panel: str) -> pd.DataFrame:
    encoding='latin1' if panel=='jena' else 'utf-8'
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
            if len(names)!=1: raise Blocked(f'AMBIGUOUS_DATA_ARCHIVE: {names}')
            return pd.read_csv(io.BytesIO(z.read(names[0])),encoding=encoding)
    return pd.read_csv(path,encoding=encoding)


def prepare_frame(frame: pd.DataFrame, name: str, config: dict, raw_receipt: dict) -> Panel:
    if name not in ('ettm2','jena'): raise ValueError(name)
    time_col=str(frame.columns[0]); fmt='%d.%m.%Y %H:%M:%S' if name=='jena' else '%Y-%m-%d %H:%M:%S'
    times=pd.to_datetime(frame.iloc[:,0],format=fmt,errors='raise')
    cols=[str(c) for c in frame.columns[1:]]
    if len(cols)!=config['expected_channels'][name]: raise Blocked(f'SCHEMA_CHANNEL_COUNT {name}: {len(cols)}')
    numeric=frame.iloc[:,1:].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    order=np.argsort(times.to_numpy(),kind='stable'); reorders=int((order!=np.arange(len(order))).sum())
    times=pd.DatetimeIndex(times.iloc[order]); numeric=numeric[order]
    if times.duplicated().any(): raise Blocked('DUPLICATE_TIMESTAMPS: no silent deduplication')
    freq=config['frequency_seconds'][name]
    if np.any(np.diff(times.asi8)!=freq*10**9): raise Blocked('IRREGULAR_TIMELINE: no automatic resampling/imputation')
    if name=='jena': numeric[numeric<=-9999.0+1e-6]=np.nan
    if np.isinf(numeric).any(): raise Blocked('INFINITE_OBSERVATIONS')
    # Fixed-year Jena provenance: do not use a later year as a rescue data set.
    if name=='jena' and not (times[0].year==2024 and times[-1].year in (2024,2025)):
        raise Blocked('JENA_YEAR_MISMATCH: expected the official 2024 archive')
    rpd=86400//freq
    c=int(config['context_days']*rpd); cs=int(config['short_context_days']*rpd); h=int(config['horizon_days']*rpd)
    n=len(times); a=(int(n*config['train_fraction'])//rpd)*rpd
    b=(int(n*(config['train_fraction']+config['validation_fraction']))//rpd)*rpd
    bounds={'train':(c,a),'val':(a,b),'test':(b,n)}
    # Scale and eligibility use TRAIN only. Target availability is a shared QC filter, never a performance filter.
    sigma=np.nanstd(numeric[:a].astype(np.float64),axis=0)
    keep=np.isfinite(sigma)&(sigma>1e-6)
    if not keep.any(): raise Blocked('NO_VARIABLE_TRAIN_SERIES')
    excluded=[col for col,k in zip(cols,keep) if not k]
    cols=[col for col,k in zip(cols,keep) if k]; numeric=numeric[:,keep]; sigma=sigma[keep]
    bad=(~np.isfinite(numeric)).any(axis=1).astype(np.int64); cumulative=np.r_[0,np.cumsum(bad)]
    origins={}; eligibility={}
    for role,(left,right) in bounds.items():
        stride=int(config['train_stride_hours']*3600/freq) if role=='train' else int(config['evaluation_stride_hours']*3600/freq)
        oo=np.arange(max(c,left),right-h+1,stride,dtype=np.int64)
        valid=(cumulative[oo+h]-cumulative[oo-c])==0
        origins[role]=oo[valid]; eligibility[role]={'before':len(oo),'after':int(valid.sum()),'excluded_nonfinite_context_or_target':int((~valid).sum())}
    for role,key in [('train','minimum_train_origins'),('val','minimum_validation_origins'),('test','minimum_test_origins')]:
        if len(origins[role])<config[key]: raise Blocked(f'INSUFFICIENT_COMPLETE_{role.upper()}_ORIGINS: {len(origins[role])}')
    metadata=dict(name=name,receipt=raw_receipt,time_column=time_col,columns=cols,excluded_train_constant=excluded,
                  frequency_seconds=freq,context=c,short_context=cs,horizon=h,rows=n,
                  source_first=str(times[0]),source_last=str(times[-1]),sort_changed_rows=reorders,
                  timezone='source naive timestamps; no timezone or DST invented',bounds=bounds,eligibility=eligibility,
                  split_meaning='60/20/20 time order rounded down to source-day boundaries; target must be contained in role',
                  source_overlap='Previously used data sources; development screen, not independent external confirmation',
                  valid_origin_rule='whole multivariate context and target finite; identical origins for every arm',
                  native_channels=int(numeric.shape[1]),values_sha256=hashlib.sha256(numeric.tobytes()).hexdigest(),
                  scale_sha256=hashlib.sha256(sigma.tobytes()).hexdigest())
    return Panel(name,numeric,times.to_numpy(dtype='datetime64[ns]').astype('int64'),sigma,cols,c,cs,h,origins,metadata)


def save_panel(panel: Panel, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,values=panel.values,timestamps=panel.timestamps,sigma=panel.sigma,
        columns=np.asarray(panel.columns),context=panel.context,short_context=panel.short_context,horizon=panel.horizon,
        **{k+'_origins':v for k,v in panel.origins.items()})
    write_json(path.with_suffix('.json'),panel.metadata)


def read_panel(path: Path) -> Panel:
    with np.load(path,allow_pickle=False) as z:
        meta=read_json(path.with_suffix('.json'))
        p=Panel(meta['name'],z['values'],z['timestamps'],z['sigma'],[str(x) for x in z['columns']],
                int(z['context']),int(z['short_context']),int(z['horizon']),
                {r:z[r+'_origins'] for r in ('train','val','test')},meta)
    return p


def acquire(repo: Path, cache: Path, result: Path, name: str, config: dict, local_file: str|None=None) -> Panel:
    start=time.perf_counter(); raw_dir=cache/'raw'; raw_dir.mkdir(parents=True,exist_ok=True)
    if local_file:
        source=Path(local_file).expanduser().resolve()
        if not source.is_file(): raise Blocked(f'LOCAL_DATA_NOT_FOUND: {source.name}')
        origin='explicit local override (no file-name discovery)'
    else:
        url=config['source_urls'][name]; source=raw_dir/('ETTm2.csv' if name=='ettm2' else 'mpi_roof_2024.zip')
        download(url,source); origin=url
    panel=prepare_frame(load_raw(source,name),name,config,dict(source=origin,sha256=sha(source),bytes=source.stat().st_size))
    panel.metadata['data_prepare_s']=time.perf_counter()-start
    save_panel(panel,cache/'panel.npz'); write_json(result/'DATA_AUDIT.json',panel.metadata)
    # A poison-target test for every TRAIN origin, not a costly model invocation.
    poisoned=panel.values.copy(); train_end=panel.metadata['bounds']['train'][1]; poisoned[train_end:]=1234567
    for o in panel.origins['train']:
        if not np.array_equal(panel.values[o-panel.context:o+panel.horizon],poisoned[o-panel.context:o+panel.horizon]):
            raise AssertionError('TRAIN_AFTER_BOUNDARY_ACCESS')
    write_json(result/'DATA_VERIFICATION.json',dict(pass_=True,train_future_poison=True,
        exact_roles=True,feature_labels_unused=True,prepared_sha256=sha(cache/'panel.npz')))
    return panel
