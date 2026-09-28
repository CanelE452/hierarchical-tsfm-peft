import json, hashlib, time
from pathlib import Path
here=Path(__file__).resolve().parent
if (here/'ledger.json').exists():
 raise RuntimeError('Bootstrap is one-time only; never reset an existing budget ledger')
root=here.parents[1]
v6=root/'research/tsfm_peft_level_confirmation_v6_20260928'
old=json.loads((v6/'protocol.json').read_text(encoding='utf-8'))
limits=dict(old['limits'],real_fit=20)
p={k:old[k] for k in ('context','horizon','residual_rank','seeds','epochs','patience','epoch_origins','effective_batch_origins','microbatch_origins','eval_batch_origins','optimizer','weight_decay','clip_grad_norm','scheduler','checkpoint_selection','backbone_revision','precision','tf32','cpu_threads','extension','level_statistic')}
p.update(schema_version=1,stage='v7_practical_controls',base_commit='a3ccb0a3ae25c075c79015545a0b8d7013ad6550',limits=limits,base_real_fits=16,reserve_real_fits=4,references=['f0','compress','level_only'],direct_lrs=[1e-3,1e-4],direct_lr_selection='mean two seed minimum full VAL macro MSE; exact tie listed first',test_selection_prohibited=True)
p['datasets']={
 'robin':dict(channels=17,latent=5,phase_period=24,interval_minutes=60,bootstrap_block_origins=7,expected_origins=old['expected_origins'],splits=old['splits'],evaluation_stride=24,status='v6 exposed; v7 follow-up direct comparison'),
 'jena':dict(channels=21,latent=6,phase_period=144,interval_minutes=10,bootstrap_block_origins=42,expected_origins=dict(train=25648,val=371,test_a=365,test_b=365),splits={'train':['2024-01-01T00:10','2024-07-01T00:00'],'val':['2024-07-01T00:00','2024-09-01T00:00'],'test_a':['2024-09-01T00:00','2024-11-01T00:00'],'test_b':['2024-11-01T00:00','2025-01-01T00:10']},evaluation_stride=24,status='previously exposed source; fixed other-domain extension, not independent confirmation')}
p['formulas']={k:old['formulas'][k] for k in ('level_res','level_raw','old_res','level_only','compress')}
p['formulas']['direct_nlinear']='P_H(X)+T(X-P_L(X)); shared affine Linear(512,48); last input detached'
p['bootstrap']=dict(draws=2000,seed=9262026,stratify_periods=True,channels_and_methods_paired=True)
p['cost']=dict(val_origins=24,batches=[1,4],warmup_calls=10,passes=20,blocks=3,primary_gpu_rows=117,cpu_rows=12,atol=1e-5,rtol=1e-4,scope='standardized CPU FP32 input to complete CPU [B,48,C] output; load/disk/scoring/ledger excluded',generic_output_policy='one preallocated GPU median buffer; no retained raw quantile outputs, no per-chunk model reload or cache flush',conditional_diagnostic_max_gpu_seconds=600,conditional_diagnostic_sessions=1)
p['fits']=[]
for dataset in ('robin','jena'):
 c,k=p['datasets'][dataset]['channels'],p['datasets'][dataset]['latent']
 families=['direct_nlinear'] if dataset=='robin' else ['level_res','old_res','level_raw','lora','direct_nlinear']
 for family in families:
  for lr in ([.001,.0001] if family=='direct_nlinear' else [.0001] if family=='lora' else [.001]):
   for seed in p['seeds']:
    suffix=f'_lr{lr:g}' if family=='direct_nlinear' else ''
    p['fits'].append(dict(id=f'v7_{dataset}_{family}{suffix}_{seed}',dataset=dataset,family=family,arm={'level_raw':'raw','lora':'lora','direct_nlinear':'direct'}.get(family,'residual'),ed_mode='none' if family=='direct_nlinear' else 'fixed_ed',seed=seed,channels=c,latent=0 if family=='direct_nlinear' else k,residual_rank=32,normalization='train_standardization_only' if family=='direct_nlinear' else 'context',lr=lr,t_lr=lr,epochs=120))
def save(name,obj):
 (here/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
save('protocol.json',p)
save('ledger.json',dict(schema_version=1,created_utc=time.time(),limits=limits,jobs=[]))
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
save('provenance.json',dict(base_commit=p['base_commit'],plan_sha256=h(here/'PLAN.md'),approval_sha256=h(here/'APPROVAL.txt'),protocol_sha256=h(here/'protocol.json'),plan_source='verbatim final assistant plan from turn 01a0e873-9b59-7851-b29c-f0e4d7d6698e; retrieved via read_thread',approval_source='verbatim active goal objective',bootstrapping='bookkeeping only; no model import, fit, forecast, score, benchmark or check executed'))
source=(v6/'runtime_v6.py').read_text(encoding='utf-8')
source=source.replace('V6','V7').replace('tsfm_peft_level_confirmation_v6_20260928','tsfm_peft_practical_controls_v7_20260928').replace("'real_fit': 16","'real_fit': 20")
source=source.replace("V3 = ROOT / 'research/tsfm_peft_backbone_value_v3_20260927'","V3 = ROOT / 'research/tsfm_peft_backbone_value_v3_20260927'\nV6 = ROOT / 'research/tsfm_peft_level_confirmation_v6_20260928'\nV6_CACHE = ROOT / '.cache/tsfm_peft_level_confirmation_v6_20260928'")
source=source.replace("HERE / 'test_exposure.json'","HERE / f'test_exposure_{self.metadata.get(\"dataset\", \"unknown\")}.json'",1)
start=source.index('def load_data(')
end=source.index('\ndef tensors(',start)
source=source[:start]+'''def load_data(dataset, include_test=False):
    import numpy as np
    if dataset not in ('robin', 'jena'):
        raise ValueError('Only the two approved datasets are allowed')
    contract = json.loads((HERE / f'data_contract_{dataset}.json').read_text(encoding='utf-8'))
    if include_test:
        seal_path = HERE / f'evaluation_seal_{dataset}.json'
        exposure_path = HERE / f'test_exposure_{dataset}.json'
        if not seal_path.exists() or not exposure_path.exists():
            raise RuntimeError('TEST requires dataset-specific selection seal and exposure record')
        exposure = json.loads(exposure_path.read_text(encoding='utf-8'))
        if exposure.get('seal_sha256') != digest(seal_path):
            raise RuntimeError('TEST exposure/selection seal mismatch')
    entry = contract['test' if include_test else 'trainval']
    path = Path(entry['path'])
    if not path.is_absolute():
        path = ROOT / path
    if digest(path) != entry['sha256']:
        raise RuntimeError('Sealed data content changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['_path'] = path
    data['_dataset'] = dataset
    data['_phase_period'] = 24 if dataset == 'robin' else 144
    for name in ('train_start', 'train_end'):
        if name in data:
            data[name] = int(data[name])
    expected = 17 if dataset == 'robin' else 21
    if data['x'].shape[1] != expected:
        raise ValueError('Approved channel count changed')
    return data

''' +source[end:]
source=source.replace('def phase_sample(origins, seed, epoch, n=512):','def phase_sample(origins, seed, epoch, n=512, period=24):').replace('for phase in range(24):','for phase in range(period):').replace('origins % 24 == phase','origins % period == phase').replace('n // 24 + int(phase < n % 24)','n // period + int(phase < n % period)')
source=source.replace("paths = list(HERE.glob('*.py')) + [HERE / 'PLAN.md', HERE / 'protocol.json', HERE / 'data_contract.json']","paths = list(HERE.glob('*.py')) + list(HERE.glob('data_contract*.json')) + [HERE / 'PLAN.md', HERE / 'protocol.json']")
(here/'runtime_v7.py').write_text(source,encoding='utf-8')
print('Bookkeeping initialized: protocol 16 base fits +4 reserve; empty ledger; approved text hashes; runtime source.')

