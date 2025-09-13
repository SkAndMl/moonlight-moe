import torch, math

from pydantic import BaseModel
from torch import nn, Tensor
from torch.nn import functional as F

class ModelConfig(BaseModel):
    base: int = 10000
    vocab_size: int = 50304
    ctx_size: int = 256
    embed_dim: int = 512
    n_heads: int = 8
    ffn_dim: int = 512 * 4
    eps: float = 1e-8
    n_blocks: int = 8
    n_experts: int = 8
    capacity_factor: float = 1.0
    alpha_aux_loss: float = 1e-2

def get_freqs_cis(cfg: ModelConfig) -> Tensor:
    head_dim = cfg.embed_dim // cfg.n_heads
    
    thetas = cfg.base ** (-2 * (torch.arange(0, head_dim // 2) / head_dim))
    pos = torch.arange(0, cfg.ctx_size)
    freqs = torch.outer(pos, thetas)
    
    _real = torch.cos(freqs)
    _img = torch.sin(freqs)

    return torch.complex(_real, _img)

def apply_rot_emb(x: Tensor, freq_cis: Tensor) -> Tensor:
    # x -> bsz, n_heads, seq_len, head_dim
    bsz, n_heads, seq_len, head_dim = x.shape
    _x = x.view(bsz, n_heads, seq_len, head_dim // 2, 2)
    _x = torch.view_as_complex(_x.to(torch.float32)) # bsz, n_heads, seq_len, head_dim / 2
    freqs = freq_cis[..., :seq_len, :]
    _x_rot = _x * freqs.to(_x.dtype) # bsz, n_heads, seq_len, head_dim / 2
    _x_rot = torch.view_as_real(_x_rot) # bsz, n_heads, seq_len, head_dim / 2, 2
    _x_rot = _x_rot.view(bsz, n_heads, seq_len, head_dim)
    return _x_rot


class RMSNorm(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.ones(cfg.embed_dim))
        self.eps = cfg.eps
    
    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim = True).add(self.eps).rsqrt()
        return self.W * x * rms


class MHA(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        
        super().__init__()
        self.cfg = cfg
        self.QKV = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias = False)
        self.O = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias = False)
        
        # get freqs_cis and register it
        freqs_cis = get_freqs_cis(cfg).unsqueeze(0).unsqueeze(0)
        self.register_buffer('freqs_cis', freqs_cis)
        
        # construct mask and register it
        mask = float("-inf") * torch.triu(torch.ones(1, 1, cfg.ctx_size, cfg.ctx_size), diagonal=1)
        self.register_buffer("mask", mask)


    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, embed_dim = x.shape
        n_heads, head_dim = self.cfg.n_heads, embed_dim // self.cfg.n_heads
        qkv = self.QKV(x) #
        q, k, v = qkv.split(embed_dim, -1)
        q: Tensor = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2) # bsz, n_heads, seq_len, head_dim
        k: Tensor = k.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
        v: Tensor = v.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

        q, k = apply_rot_emb(q, self.freqs_cis), apply_rot_emb(k, self.freqs_cis)

        wts = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        wts = wts + self.mask[:, :, :seq_len, :seq_len]
        wts = F.softmax(wts, dim = -1)

        y = wts @ v # bsz, n_heads, seq_len, head_dim
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return self.O(y)


class Router(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.W = nn.Linear(cfg.embed_dim, cfg.n_experts, bias = False)
    
    def forward(self, x: Tensor) -> Tensor:
        logits = self.W(x.to(torch.float32)) # T, n_experts
        return F.softmax(logits, dim = -1)

class FFN(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim, bias = False),
            nn.ReLU(),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim, bias = False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class MoE(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.capacity_factor = cfg.capacity_factor
        self.n_experts = cfg.n_experts
        self.router = Router(cfg)
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(cfg.n_experts)])
    
    def forward(self, x: Tensor) -> Tensor:
        # bsz, seq_len, embed_dim
        bsz, seq_len, embed_dim = x.shape
        t = bsz * seq_len

        x_flat = x.view(t, embed_dim) # t, embed_dim
        expert_probs: Tensor = self.router(x_flat) # t, e

        # calculate aux loss
        per_token_expert_idxs = torch.argmax(expert_probs, dim=-1) # t
        per_token_expert_prob = expert_probs.gather(1, per_token_expert_idxs.unsqueeze(-1)).squeeze(1) # t
        # fraction of tokens routed to an expert
        f_e = per_token_expert_idxs.bincount(minlength=self.n_experts) / t # e
        # total probability of tokens routed to an expert
        p_e = expert_probs.mean(dim=0) # e
        self.aux_loss = (
            self.cfg.alpha_aux_loss * self.n_experts * (f_e * p_e).sum()
        ).to(x.device)

        # calculate max. num of tokens to be routed to an expert
        cap = max(
            math.ceil(t * self.capacity_factor / self.n_experts),
            1
        )
        y_out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = per_token_expert_idxs == e
            idxs_e = torch.nonzero(mask, as_tuple=False).squeeze(1)[:cap]
            y_out[idxs_e, :] = per_token_expert_prob[idxs_e] * self.experts[e](x_flat[idxs_e, :])
        
        return y_out.view(bsz, seq_len, embed_dim)


class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mha = MHA(cfg)
        self.ffn = MoE(cfg)

        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tok_emb(x)
        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.norm(x))

    def moe_aux_loss(self) -> Tensor:
        _loss = torch.tensor(0.0, device=self.lm_head.weight.device)
        for block in self.blocks:
            if hasattr(block.ffn, 'aux_loss'):
                _loss += block.ffn.aux_loss
        return _loss    