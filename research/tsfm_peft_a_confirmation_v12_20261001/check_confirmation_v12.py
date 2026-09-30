"""Bounded CPU checks; no pretrained-model load and no optimizer update."""
import ast
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from runtime_confirmation_v12 import (HERE, Job, configure, load_data, protocol,
                                      masked_macro_loss, phase_sample, save_json, digest)
from model_confirmation_v12 import ConfirmationModel


class PersistenceBackbone(nn.Module):
    chronos_config = SimpleNamespace(quantiles=[0.5])

    def forward(self, context):
        return SimpleNamespace(quantile_preds=context[:, -1:, None].expand(-1, 1, 48))


def run():
    with Job('cpu_check', 'confirmation_cpu_checks01', reserve_s=60,
             metadata={'optimizer_updates': 0, 'pretrained_model_loads': 0}) as job:
        configure()
        parsed = []
        for path in HERE.glob('*confirmation*v12.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            parsed.append({'path': str(path), 'sha256': digest(path)})
        results, decisions = {}, {}
        for dataset, unit in protocol()['units'].items():
            data = load_data(dataset)
            basis = torch.as_tensor(data['basis'])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(19301)
                x = torch.randn(4, 512, unit['C'])
            direct = ConfirmationModel(dataset, 'direct', 92601, basis)
            actual = direct(x)
            last = x[:, -1:, :].detach()
            expected = direct.temporal((x-last).transpose(1, 2)).transpose(1, 2)+last
            assert torch.equal(actual, expected)
            assert sum(p.numel() for p in direct.parameters() if p.requires_grad) == 24624
            level = ConfirmationModel(dataset, 'level', 92601, basis, PersistenceBackbone())
            assert torch.count_nonzero(level.residual.up.weight) == 0
            assert torch.allclose(level(x), last.expand(-1, 48, -1), atol=1e-5, rtol=1e-4)
            assert {n for n,p in level.named_parameters() if p.requires_grad} == {'residual.down.weight','residual.up.weight'}
            y = torch.zeros_like(actual)
            mask = torch.ones_like(y, dtype=torch.bool)
            mask[0, :8, 0] = False
            score = masked_macro_loss(actual, y, mask)
            manual = torch.stack([actual[:,:,c][mask[:,:,c]].square().mean() for c in range(unit['C'])]).mean()
            assert torch.allclose(score, manual, atol=1e-7, rtol=1e-6)
            schedule = phase_sample(data['train_origins'],92601,1,512,unit['phase_period'])
            assert len(np.unique(schedule)) == 512 and np.isin(schedule,data['train_origins']).all()
            assert np.array_equal(schedule,phase_sample(data['train_origins'],92601,1,512,unit['phase_period']))
            blocked = False
            try:
                load_data(dataset, include_test=True)
            except FileNotFoundError:
                blocked = True
            assert blocked, 'TEST loader must remain closed before the joint seal'
            counts = {}
            for split in ('train','val'):
                count = np.zeros(unit['C'],dtype=np.int64)
                complete = 0
                for o in data[split+'_origins']:
                    m = data['finite'][o:o+48]
                    count += m.sum(0)
                    complete += int(m.all(axis=1).sum())
                assert (count>0).all()
                counts[split] = {'channel_observed_target_occurrences':count.tolist(),
                    'all_observed':bool(np.all(count==48*len(data[split+'_origins']))),
                    'complete_vector_occurrences':complete,
                    'total_vector_occurrences':48*len(data[split+'_origins'])}
            decisions[dataset] = {'branch':'E1_same_checkpoint_claim_only','no_q_fit':False,
                'additional_fit_count':0,'test_error_used_for_decision':False,'mask_contract':counts,
                'reason':'Claim is Q contribution at the same selected A main, not superiority to independently trained compressed LoRA.',
                'full_training_policy_equivalence':'not_established; masked weights and relative plateau decisions need not preserve equivalence',
                'additional_paired_backward':0}
            results[dataset] = {'direct_formula':True,'direct_trainable':24624,'level_zero_G':True,
                'level_initial_surrogate_identity':True,'masked_loss':True,'sampling':True,'test_guard':True,
                'scope':'Synthetic algebra/shape checks only; real Chronos gradient and replay verified inside scheduled fits'}
            job.heartbeat(dataset+' finite CPU checks complete')
        save_json(HERE/'confirmation_no_q_decision.json',{'datasets':decisions,'decision_before_TEST':True})
        save_json(HERE/'confirmation_cpu_checks01.json',{'status':'pass','files':parsed,'units':results,
                 'optimizer_updates':0,'synthetic_optimizer_sessions':0,'model_execution':'CPU surrogate and direct formula only'})


if __name__=='__main__':
    run()
