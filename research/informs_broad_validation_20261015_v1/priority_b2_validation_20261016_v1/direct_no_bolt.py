"""Jena R5a/R5b models only; no Task, checkpoint, fit, F or cost runner.

R5a receives the exact R4 adapter and freezes its P/E/D. ROOT must bind that
artifact and A60 scaler. Its head uses a predeclared fixed A60 three epochs;
A_stop is not an unseen head-selection set. R5b creates fresh original B2
P/E/D from the caller's A50/A60 initialization, retaining the original search
0/.25/1/3 -> A_stop selection -> fresh selected-length A60 refit procedure.
These are different training comparisons, never identical final weights.

Both replace only F with one channel-shared Linear(512,48,bias=True), initialized
to latent persistence. ROOT owns AdamW/lr1e-3/wd0/effective16/micro4/clip1/
seed92601 and safe checkpointing; this module implements no training loop.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

NETWORKS_SHA="787f9caf91f32a001f929f324c7bcb570b9270331f6f60f0f56872529f86dc54"
L,H,C,K,SEED=512,48,21,6,92601
_CLASS=None


def model_class():
    global _CLASS
    if _CLASS is not None:return _CLASS
    original=Path(__file__).resolve().parents[1]/"networks.py"
    if hashlib.sha256(original.read_bytes()).hexdigest()!=NETWORKS_SHA:raise ValueError("Original networks source changed")
    sys.path.insert(0,str(original.parent))
    import torch
    networks=importlib.import_module("networks")
    if Path(networks.__file__).resolve()!=original:raise ValueError("Original networks module root required")

    class DirectNoBolt(torch.nn.Module):
        """Original B2 adapter plus a shared full-context temporal linear head."""
        def __init__(self,adapter,mode):
            super().__init__()
            if (mode not in {"R5a","R5b"} or not isinstance(adapter,networks.ForecastAdapter)
                    or (adapter.length,adapter.horizon,adapter.channels,adapter.kmax)!=(L,H,C,K)
                    or adapter.front.kind!="B2" or adapter.encoder_kind!="pointwise" or adapter.k_grid is not None):
                raise ValueError("Exact Jena B2 pointwise P/E/lag-D L512/H48/C21/K6 required")
            self.adapter,self.mode=adapter,mode
            self.adapter.requires_grad_(mode=="R5b")
            for parameter in self.adapter.parameters():parameter.grad=None
            self.head=torch.nn.Linear(L,H,bias=True).to(device=adapter.E.device,dtype=adapter.E.dtype)
            with torch.no_grad():
                self.head.weight.zero_();self.head.weight[:,-1]=1.;self.head.bias.zero_()
            if mode=="R5a":self.adapter.eval()

        def train(self,mode=True):
            super().train(mode)
            if self.mode=="R5a":self.adapter.eval()
            return self

        def optimizer_parameters(self):
            return [parameter for parameter in self.parameters() if parameter.requires_grad]

        def forward(self,x):
            if x.ndim!=3 or tuple(x.shape[1:])!=(L,C):raise ValueError("Full normalized history [B,512,21] required")
            if self.mode=="R5a":
                with torch.no_grad():z=self.adapter.encode(x,K)
            else:z=self.adapter.encode(x,K)
            if tuple(z.shape)!=(x.shape[0],L,K):raise ValueError("Full latent history [B,512,6] required")
            future=self.head(z.transpose(1,2)).transpose(1,2)
            # Frozen D is still differentiable with respect to the trainable head.
            output=self.adapter.decode(x,z,future,K)
            if tuple(output.shape)!=(x.shape[0],H,C):raise ValueError("Original anchored lag-D output shape differs")
            return output

    _CLASS=DirectNoBolt
    return _CLASS


def fixed_model(exact_R4_adapter):
    """R5a: caller supplies the actual selected adapter, without a state copy."""
    return model_class()(exact_R4_adapter,"R5a")


def fresh_model(pca_center,basis,seed=SEED):
    """R5b: fresh original architecture; caller supplies legal stage PCA/statistics."""
    if seed!=SEED:raise ValueError("Only the predeclared seed92601 is supported")
    cls=model_class()
    import torch
    networks=importlib.import_module("networks")
    torch.manual_seed(seed)
    adapter=networks.ForecastAdapter("B2",L,H,C,K,pca_center,basis,"pointwise",None)
    return cls(adapter,"R5b")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(dict(status="MODEL_LIBRARY_ONLY_NOT_EXECUTED",input_reads=0,writes=0,
        actual_Task_checkpoint_foundation_fit_cost_or_child_operations=0,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
        roles=["R5a_fixed_exact_R4_PED_head_only","R5b_fresh_same_architecture_PED_and_head"],
        L=L,H=H,C=C,K=K,head_parameters=H*L+H,full_latent_history=True,foundation_constructed=False)))
    return 0


if __name__=="__main__":raise SystemExit(main())
