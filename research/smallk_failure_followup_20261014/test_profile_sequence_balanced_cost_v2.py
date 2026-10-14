"""Verification-device repair; CPU source/math invariants, no real assets."""
import ast
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent


def test_v2_only_moves_verification_tensors_to_cpu_and_preserves_entire_other_source():
    old=(HERE/'profile_sequence_balanced_cost_v1.py').read_text(encoding='utf-8')
    new=(HERE/'profile_sequence_balanced_cost_v2.py').read_text(encoding='utf-8')
    before="torch.equal(new_codec.state_dict()['fixed_basis'],initial['fixed_basis'])"
    after="torch.equal(new_codec.state_dict()['fixed_basis'].detach().cpu(),initial['fixed_basis'].detach().cpu())"
    assert old.count(before)==1 and new.count(after)==1
    assert new.replace(after,before)==old
    ast.parse(new)


def test_verification_is_exact_and_still_detects_changed_fixed_basis():
    original=torch.tensor([[.25,.5],[.75,1.]],requires_grad=True)
    selected=original.detach().clone()
    assert torch.equal(selected.detach().cpu(),original.detach().cpu())
    selected[0,0]+=.01
    assert not torch.equal(selected.detach().cpu(),original.detach().cpu())
