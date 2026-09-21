import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6
CAP_RATIO = 0.1


def rms(x):
    return torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + EPS ** 2)


def context_embeddings(embeddings, arm, parents, children):
    if arm == "LORA_SELF":
        return torch.zeros_like(embeddings)
    if arm == "LORA_POOL":
        if len(embeddings) < 2:
            raise ValueError("POOL needs at least two nodes")
        return (embeddings.sum(dim=0, keepdim=True) - embeddings) / (len(embeddings) - 1)
    if arm != "LORA_HIER":
        raise ValueError(arm)
    contexts = []
    for i in range(len(embeddings)):
        groups = [embeddings[list(indices)].mean(dim=0) for indices in (parents[i], children[i]) if len(indices)]
        if not groups:
            raise ValueError("Isolated hierarchy node")
        contexts.append(torch.stack(groups).mean(dim=0))
    return torch.stack(contexts)


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 8)
        self.up = nn.Linear(8, 512)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        assert sum(p.numel() for p in self.parameters()) == 8712

    def forward(self, h, context):
        ratio = (rms(h) / (rms(context) + EPS)).detach()
        u = F.layer_norm(h + context * ratio, (512,), eps=EPS)
        delta = CAP_RATIO * rms(h).detach() * torch.tanh(self.up(F.gelu(self.down(u))))
        return h + delta
