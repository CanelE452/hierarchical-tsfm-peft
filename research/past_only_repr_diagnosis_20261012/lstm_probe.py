"""Conditional small latent LSTM; frozen E/D, no foundation-model retraining."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from common import (HERE,CACHE,L,H,SEEDS,Budget,BudgetHeld,access_guard,read,write,sha,ahash,
                    load_data,windows,masks,score,physical_score)


class SmallLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(16,32,num_layers=1,dropout=0,bidirectional=False,batch_first=True)
        self.output = nn.Linear(32,H*16)

    def forward(self,x):
        if x.ndim != 3 or x.shape[1:] != (33,16):
            raise ValueError('LSTM input must be last33 latent ticks [B,33,16]')
        hidden,_ = self.recurrent(x)
        return self.output(hidden[:,-1]).reshape(len(x),H,16)


def state_hash(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pins():
    return {str(p):sha(p) for p in [Path(__file__),HERE/'common.py',HERE/'models_io.py',
                                    HERE/'e34.py',HERE/'APPROVED_INSTRUCTIONS.md']}


def latent_loss(p,y,m):
    if not bool(m.any()):
        raise ValueError('No observed FIT latent target')
    return ((p-y).square()*m).sum()/m.sum()


def remaining_cpu():
    return 3600-sum(read(p)['wall_seconds'] for p in (HERE/'resources').glob('*.json') if not read(p)['gpu'])


def cost_only(budget):
    started = time.perf_counter()
    torch.manual_seed(92601)
    model = SmallLSTM()
    x = torch.randn(8,33,16); target = torch.randn(8,H,16); valid = torch.ones_like(target,dtype=torch.bool)
    values = []
    for i in range(5):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        latent_loss(model(x),target,valid).backward()
        values.append(time.perf_counter()-start)
        budget.check()
        if time.perf_counter()-started > 1.8:
            raise BudgetHeld('LSTM synthetic cost-only 2-second local limit reached')
    step = float(np.median(values[1:]))
    return {'status':'COST_ONLY_NOT_SCIENCE','actual_optimizer_updates':0,'science_updates':0,'gpu_calls':0,
            'synthetic_shape':{'history':[8,33,16],'future':[8,H,16]},'samples_seconds':values,
            'median_forward_backward_seconds':step,'estimated_balanced_wall_seconds':2*2*600*step+90,
            'margin_multiplier':2,'fixed_extra_seconds_for_io_curves_bench':90,
            'actual_cost_only_wall_seconds':time.perf_counter()-started,
            'estimate_not_completion_guarantee':True,'shared_CPU_remaining_seconds':remaining_cpu()}


def verify_row(row):
    path = Path(row['prediction_path'])
    if sha(path) != row['prediction_hash']:
        raise ValueError('E34 prediction bytes changed')
    with np.load(path,allow_pickle=False) as z:
        prediction,target,mask,origins,mean,std = [z[k] for k in ('prediction','target','mask','origins','mean','std')]
    metric = score(prediction*std+mean,target*std+mean,mask)
    delta = max(abs(metric[k]-row['metrics'][k]) for k in ('mse','mae'))
    if delta > 1e-7:
        raise ValueError(f'E34 saved-array verification failed: {row["method"]}: {delta}')
    return {'path':str(path),'sha256':sha(path),'method':row['method'],'metric_max_abs_delta':delta}, (target,mask,origins,mean,std)


def check_pair(gate):
    safety = read(HERE/'E1_GATE.json')
    if safety.get('safety_pass') is not True:
        raise ValueError('E1 safety gate not passed')
    records = gate.get('e34_verified',[])
    if len(records) != 2:
        raise ValueError('Both verified E34 seed records are required')
    checked = []
    for binding in records:
        path = Path(binding['path'])
        if sha(path) != binding['sha256']:
            raise ValueError('Root-verified E34 case bytes changed')
        case = read(path)
        if case.get('dataset') != 'jena' or case.get('dim') != 16 or case.get('seed') not in SEEDS:
            raise ValueError('Only the balanced Jena d16 pair is authorized')
        if case.get('scope') != 'DIAG_SUBSET_JENA_D16_PAIR' or case.get('status') != 'COMPLETE_UNVERIFIED':
            raise ValueError('Unexpected E34 actual scope/status')
        if case.get('safety_gate_sha') != sha(HERE/'E1_GATE.json'):
            raise ValueError('E34 safety receipt binding differs')
        if case.get('source_hashes') != case.get('source_hashes_at_start'):
            raise ValueError('E34 implementation changed during actual execution')
        metrics = {r['method']:r['metrics'] for r in case['E4']}
        required = ('REFERENCE_LATENT_DIAGNOSTIC','LATENT_PERSISTENCE','LATENT_JOINT_RIDGE','LATENT_SELF_RIDGE')
        if any(k not in metrics for k in required):
            raise ValueError('Required E4 methods are missing')
        comparisons = {f'reference_better_{k}':metrics['REFERENCE_LATENT_DIAGNOSTIC'][k]<metrics['LATENT_PERSISTENCE'][k]
                       for k in ('mse','mae')}
        comparisons.update({f'joint_better_self_{k}':metrics['LATENT_JOINT_RIDGE'][k]<metrics['LATENT_SELF_RIDGE'][k]
                            for k in ('mse','mae')})
        checks,common_arrays = [],None
        for row in case['E3']+case['E4']:
            check,arrays = verify_row(row); checks.append(check)
            if common_arrays is None:
                common_arrays = arrays
            elif any(not np.array_equal(a,b) for a,b in zip(common_arrays,arrays)):
                raise ValueError('E34 target/mask/origin/normalization mismatch')
        folder = Path(case['folder'])
        decoder_row = next(r for r in case['E3'] if r['method']=='PREFIX_BOLT_REC_D')
        checkpoint = Path(case['checkpoint_record']['checkpoint'])
        if sha(checkpoint) != case['checkpoint_record']['checkpoint_sha256']:
            raise ValueError('Historical fixed encoder bytes changed after E34')
        asset_paths = [folder/'PREFIX'/'EXPORTS.npz',folder/'E4'/'REFERENCE_LATENT_DIAGNOSTIC.npz',
                       folder/'PREFIX'/f'D_{decoder_row["decoder"]["selected_update"]}.pt',checkpoint]
        checked.append({'seed':case['seed'],'case_path':str(path),'case_sha256':binding['sha256'],
                        'scope':case['scope'],'comparisons':comparisons,'scientific_gate':all(comparisons.values()),
                        'raw_metrics':{k:metrics[k] for k in required},'array_checks':checks,
                        'asset_pins':{str(p):sha(p) for p in asset_paths}})
    if sorted(c['seed'] for c in checked) != SEEDS:
        raise ValueError('Two distinct declared seeds are required')
    return sorted(checked,key=lambda c:c['seed'])


def create_plan(gate_path,gate_sha,tag):
    if sha(gate_path) != gate_sha:
        raise ValueError('Gate bytes changed')
    gate = read(gate_path)
    if gate.get('experiment') != 'LSTM':
        raise ValueError('Wrong experiment gate')
    checked = check_pair(gate)
    allocation = float(gate.get('cpu_allocation_seconds',0))
    allowed = gate.get('execute') is True and all(c['scientific_gate'] for c in checked) and allocation > 0
    plan = {'experiment':'LSTM','status':'PLANNED_NOT_EXECUTED' if allowed else 'NOT_APPLICABLE_WITH_REASON',
            'execute':allowed,'reason':None if allowed else 'Required paired reference/joint evidence or allocation did not pass',
            'purpose':'Check a small joint latent predictor after reference reconstruction and joint ridge evidence',
            'simple_alternative':'Frozen-D latent persistence/self/joint ridge',
            'fixed':{'input_ticks':33,'latent_dim':16,'hidden':32,'layers':1,'bidirectional':False,'dropout':0,
                     'horizon':H,'seeds':SEEDS,'max_updates':600,'optimizer':'Adam','lr':.001,'batch':8,
                     'selection':'TUNE original-space physical macroMSE through frozen PREFIX reconstruction D at0/100..600',
                     'fit_origins':64,'tune_origins':32,'val_origins_max':32,'no_E_or_D_update':True},
            'conditional_scientific_gate':checked,'gate_path':str(gate_path),'gate_sha256':gate_sha,
            'cpu_allocation_seconds':allocation,'source_pins':pins(),
            'interpretation':'Operational evidence gate; not significance or independent generalization; frozen E historically exposed TRAIN/VAL',
            'decisions':{'better':'Retain one configuration as a conditional candidate',
                         'unchanged_or_worse':'Current small-LSTM configuration did not establish added value',
                         'stop':'Asset/source/target/budget/nonfinite mismatch; preserve partial attempt'},
            'scope':'DIAG_SUBSET_JENA_D16_PAIR','device':'cpu'}
    path = HERE/'plans'/f'LSTM_{tag}.json'
    write(path,plan)
    return path,plan


def decode(decoder,latent):
    with torch.no_grad():
        return np.concatenate([decoder(torch.as_tensor(latent[i:i+8],dtype=torch.float32)).numpy()
                               for i in range(0,len(latent),8)])


def load_assets(binding):
    from models_io import ReconstructionDecoder
    case = read(binding['case_path'])
    data = load_data('jena',frozen=True,sizes=(64,32,32))
    if any(ahash(data['origins'][s]) != case['origin_hashes'][s] for s in ('fit','tune','val')):
        raise ValueError('E34 origins changed')
    folder = Path(case['folder'])
    export_path = folder/'PREFIX'/'EXPORTS.npz'
    reference_path = folder/'E4'/'REFERENCE_LATENT_DIAGNOSTIC.npz'
    with np.load(export_path,allow_pickle=False) as z:
        latent = {s:z[s] for s in ('fit','tune','val')}
    with np.load(reference_path,allow_pickle=False) as z:
        reference = {s:z[s] for s in ('fit','tune','val')}
    decoder_row = next(r for r in case['E3'] if r['method']=='PREFIX_BOLT_REC_D')
    info = decoder_row['decoder']
    decoder_path = folder/'PREFIX'/f'D_{info["selected_update"]}.pt'
    decoder = ReconstructionDecoder(16,len(data['columns']))
    saved = torch.load(decoder_path,map_location='cpu',weights_only=True)
    decoder.load_state_dict(saved['state_dict']); decoder.eval().requires_grad_(False)
    if state_hash(decoder) != info['selected_D_hash']:
        raise ValueError('Fixed D tensor binding differs')
    for s in latent:
        if latent[s].shape != (len(data['origins'][s]),L,16) or reference[s].shape != (len(data['origins'][s]),H,16):
            raise ValueError('E34 latent asset shape mismatch')
    # The reference is supervision on FIT, selection on TUNE, and diagnostics on VAL.
    valid = {s:np.repeat(masks(data,s).all(-1,keepdims=True),16,-1) for s in latent}
    prediction = decode(decoder,reference['val'])
    metric = physical_score(data,prediction,'val')
    row = next(r for r in case['E4'] if r['method']=='REFERENCE_LATENT_DIAGNOSTIC')
    if max(abs(metric[k]-row['metrics'][k]) for k in ('mse','mae')) > 1e-7:
        raise ValueError('Reference->fixed D recomputation differs')
    assets = {str(p):sha(p) for p in [Path(binding['case_path']),export_path,reference_path,decoder_path,
                                     Path(case['checkpoint_record']['checkpoint'])]}
    return data,latent,reference,valid,decoder,assets


def save_checkpoint(path,value):
    with path.open('xb') as f:
        torch.save(value,f)
    return sha(path)


def save_npz(path,**value):
    with path.open('xb') as f:
        np.savez_compressed(f,**value)
    return sha(path)


def evaluate(model,decoder,latent,reference,valid,data,split):
    outputs = []
    with torch.no_grad():
        for i in range(0,len(latent),8):
            outputs.append(model(torch.from_numpy(latent[i:i+8,-33:])).numpy())
    predicted = np.concatenate(outputs)
    raw = decode(decoder,predicted)
    return predicted,raw,physical_score(data,raw,split),score(predicted,reference,valid)


def benchmark(model,decoder,latent,budget,local_check):
    rows = []
    model.eval()
    for n in (1,32):
        indices = np.arange(n)%len(latent)
        inputs = latent[indices,-33:]
        samples = []
        for rep in range(25):
            local_check(); budget.check()
            start = time.perf_counter()
            with torch.no_grad():
                p = model(torch.from_numpy(inputs.copy()))
                raw = decoder(p).numpy().copy()
            elapsed = time.perf_counter()-start
            if rep >= 5:
                samples.append(elapsed)
        rows.append({'batch':n,'unique_origins':min(n,len(latent)),'repeated_origins':max(0,n-len(latent)),
                     'warmups':5,'timed':20,'samples_seconds':samples,'p50_seconds':float(np.median(samples)),
                     'scope':'HEAD_ONLY: cached last33 latent CPU copy -> LSTM -> fixed D -> CPU raw output',
                     'excludes':'raw-window preparation, E export, source/model load, guards, disk, scoring',
                     'device':'cpu','not_end_to_end_E_P_D':True,'raw_shape':list(raw.shape)})
    return rows


def run_pair(plan_path,plan_sha,tag):
    if sha(plan_path) != plan_sha:
        raise ValueError('Plan bytes changed')
    plan = read(plan_path)
    for p,digest in plan['source_pins'].items():
        if sha(p) != digest:
            raise ValueError(f'Frozen implementation changed: {p}')
    if not plan['execute']:
        raise ValueError('NOT_APPLICABLE: conditional gate did not pass')
    if sha(plan['gate_path']) != plan['gate_sha256']:
        raise ValueError('Root gate bytes changed')
    fresh = check_pair(read(plan['gate_path']))
    if any(new['asset_pins'] != old['asset_pins'] for new,old in zip(fresh,plan['conditional_scientific_gate'])):
        raise ValueError('E34 exports/reference/D/E bytes differ from the frozen plan')
    if not all(r['scientific_gate'] for r in fresh):
        raise ValueError('Conditional numerical gate no longer passes')
    output = HERE/'results'/f'LSTM_{tag}'; output.mkdir(parents=True,exist_ok=False)
    cache = CACHE/f'LSTM_{tag}'; cache.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    def local_check():
        if time.perf_counter()-started >= plan['cpu_allocation_seconds']:
            raise BudgetHeld('Root allocated conditional-LSTM CPU wall exhausted')
    result = {'experiment':'LSTM','status':'RUNNING_CONFIRMED','scope':plan['scope'],'device':'cpu',
              'plan_path':str(plan_path),'plan_sha256':plan_sha,'source_pins':pins(),'conditions':[]}
    try:
        with Budget(2,gpu=False,label='Conditional balanced latent LSTM CPU '+tag) as budget:
            cost = cost_only(budget); write(output/'COST_PREFLIGHT.json',cost)
            result['cost'] = cost
            if cost['estimated_balanced_wall_seconds'] >= min(remaining_cpu(),plan['cpu_allocation_seconds']):
                raise BudgetHeld('Balanced LSTM cost forecast exceeds remaining global/local CPU wall')
            for binding in fresh:
                seed = binding['seed']; local_check(); budget.check()
                data,latent,reference,valid,decoder,assets = load_assets(binding)
                before_d = state_hash(decoder)
                torch.manual_seed(seed)
                model = SmallLSTM(); initial_hash = state_hash(model)
                optimizer = torch.optim.Adam(model.parameters(),lr=.001)
                rng = np.random.default_rng(seed+33)
                order = np.stack([rng.choice(len(latent['fit']),size=8,replace=False) for _ in range(600)])
                fit_x = torch.from_numpy(latent['fit'][:,-33:])
                fit_y = torch.from_numpy(reference['fit']); fit_m = torch.from_numpy(valid['fit'])
                curves,states,best = [],[],None
                sub = cache/f'seed{seed}'; sub.mkdir()
                for step in range(601):
                    local_check(); budget.check()
                    if step%100 == 0:
                        model.eval()
                        _,_,metric,lmetric = evaluate(model,decoder,latent['tune'],reference['tune'],valid['tune'],data,'tune')
                        curves.append({'update':step,'tune_raw':metric,'tune_latent':lmetric})
                        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                        cp = sub/f'LSTM_u{step:04}.pt'
                        digest = save_checkpoint(cp,{'state_dict':state,'optimizer':optimizer.state_dict(),
                                                      'step':step,'seed':seed,'batch_order_hash':ahash(order)})
                        states.append({'update':step,'path':str(cp),'sha256':digest})
                        if best is None or metric['mse'] < best[0]:
                            best = (metric['mse'],step,state)
                    if step == 600:
                        break
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    js = order[step]
                    loss = latent_loss(model(fit_x[js]),fit_y[js],fit_m[js])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite LSTM training loss')
                    loss.backward(); optimizer.step()
                model.load_state_dict(best[2]); model.eval()
                latent_pred,raw,metric,lmetric = evaluate(model,decoder,latent['val'],reference['val'],valid['val'],data,'val')
                latency = benchmark(model,decoder,latent['val'],budget,local_check)
                arrays = sub/'LSTM_VAL.npz'
                digest = save_npz(arrays,prediction=raw,target=windows(data,'val',True),mask=masks(data,'val'),
                                  origins=data['origins']['val'],mean=data['mean'],std=data['std'],
                                  latent_prediction=latent_pred,latent_reference=reference['val'],latent_mask=valid['val'],
                                  fit_origins=data['origins']['fit'],tune_origins=data['origins']['tune'],batch_order=order)
                if state_hash(decoder) != before_d or any(sha(p)!=digest for p,digest in assets.items()):
                    raise ValueError('Frozen E/D/export/reference/source asset changed')
                row = {'condition':'LATENT_LSTM_FIXED_D','seed':seed,'actual_updates':600,'selected_update':best[1],
                       'selected_at_boundary':best[1]==600,'parameters':sum(p.numel() for p in model.parameters()),
                       'initial_hash':initial_hash,'selected_hash':state_hash(model),'fixed_D_hash':before_d,
                       'batch':8,'batch_order_hash':ahash(order),'actual_sample_exposures':4800,
                       'actual_input_latent_timesteps':4800*33,'actual_future_latent_timestamps':4800*H,
                       'actual_observed_future_latent_values':int(valid['fit'][order].sum()),
                       'selection':'TUNE physical macroMSE through frozen D at0/100..600; no VAL reselection',
                       'val':metric,'latent_metrics':lmetric,'curves':curves,'checkpoints':states,'cost_HEAD_ONLY':latency,
                       'prediction_path':str(arrays),'prediction_sha256':digest,'asset_pins_before_after':assets,
                       'reference_isolation':'FIT future latent supervision/TUNE selection/VAL diagnostics; not prediction input',
                       'information_difference':'LSTM/self/joint use last33 latent ticks; Bolt uses512; E itself has same512-window history',
                       'E34_case_path':binding['case_path'],'E34_case_sha256':binding['case_sha256']}
                path = output/f'SEED_{seed}.json'; write(path,row)
                result['conditions'].append({'seed':seed,'condition':row['condition'],'path':str(path),
                                             'sha256':sha(path),'val':metric,'actual_updates':600})
                write(output/'PROGRESS.json',result,replace=True)
            result['budget_ledger'] = str(budget.path)
        result.update({'status':'COMPLETE_BALANCED','budget_ledger_sha256':sha(budget.path),
                       'actual_wall_seconds':time.perf_counter()-started,
                       'limitations':['DIAG_SUBSET','historically exposed frozen E','repeated development VAL',
                                      'two seeds not independent generalization','fixed600 not convergence','HEAD_ONLY costs not full E-P-D']})
        write(output/'RESULT.json',result)
        return output/'RESULT.json'
    except BaseException as error:
        result.update({'status':'BUDGET_HELD_PARTIAL' if isinstance(error,BudgetHeld) else 'FAILED_PRESERVED',
                       'failure_reason':str(error)})
        write(output/'FAILURE.json',result)
        raise


def unit_test():
    assert 'SmallLSTM' in globals(), 'Small latent LSTM not implemented'
    torch.manual_seed(13)
    model = SmallLSTM()
    x = torch.randn(2,33,16)
    output = model(x)
    assert output.shape == (2,48,16)
    assert model.recurrent.num_layers == 1 and not model.recurrent.bidirectional
    assert model.recurrent.dropout == 0
    modified = x.clone(); modified[:,20:] += 20
    with torch.no_grad():
        a,_ = model.recurrent(x); b,_ = model.recurrent(modified)
    torch.testing.assert_close(a[:,:20],b[:,:20],rtol=0,atol=0)
    loss = output.square().mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    return {'tests':4,'pass':True,'actual_optimizer_updates':0,'science_updates':0,'gpu_calls':0}


def main():
    parser = ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--unit',action='store_true'); options.add_argument('--cost-only',action='store_true')
    options.add_argument('--plan',action='store_true'); options.add_argument('--run',action='store_true')
    parser.add_argument('--tag'); parser.add_argument('--gate',type=Path); parser.add_argument('--gate-sha256')
    parser.add_argument('--frozen-plan',type=Path); parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    access_guard()
    torch.set_num_threads(2)
    if args.unit or args.cost_only:
        with Budget(2,gpu=False,label='LSTM CPU unit' if args.unit else 'LSTM CPU COST_ONLY') as budget:
            result = unit_test() if args.unit else cost_only(budget)
            result['budget_ledger'] = str(budget.path)
            result['source_pins'] = pins()
        result['budget_ledger_sha256'] = sha(budget.path)
        path = HERE/'receipts'/f'LSTM_{"UNIT" if args.unit else "COST_ONLY"}_{budget.path.stem}.json'
        write(path,result)
    else:
        if not args.tag or not args.tag.replace('_','').replace('-','').isalnum():
            parser.error('--tag must be a fresh simple attempt name')
        if args.plan:
            if args.gate is None or args.gate_sha256 is None:
                parser.error('--plan requires a Root gate and SHA256')
            with Budget(2,gpu=False,label='LSTM conditional plan '+args.tag):
                path,result = create_plan(args.gate,args.gate_sha256,args.tag)
        else:
            if args.frozen_plan is None or args.plan_sha256 is None:
                parser.error('--run requires a frozen plan and SHA256')
            path = run_pair(args.frozen_plan,args.plan_sha256,args.tag)
            result = read(path)
    print(json.dumps({'path':str(path),'sha256':sha(path),'status':result.get('status','UNIT_PASS')},ensure_ascii=False))


if __name__ == '__main__':
    main()
