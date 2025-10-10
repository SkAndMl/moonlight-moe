import torch, math

from config import ModelConfig
from torch import nn, Tensor
from torch.nn import functional as F

def fan_in(module: nn.Module) -> int | None:
    w = getattr(module, "weight", None)
    if w is None or w.data is None:
        raise ValueError(f"{module.__class__.__name__} has not weight")
    shape = w.shape
    if w.ndim < 2:
        return 0.02
    assert w.ndim == 2 # since we have only linear layers
    return shape[1]

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
        mask = torch.triu(torch.ones(1, 1, cfg.ctx_size, cfg.ctx_size) * float("-inf"), diagonal=1)
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

        # wts = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # wts = wts + self.mask[:, :, :seq_len, :seq_len]
        # wts = F.softmax(wts, dim = -1)
        # y = wts @ v # bsz, n_heads, seq_len, head_dim 
        # use flash attention instead
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        topk_probs, topk_experts = expert_probs.topk(k=self.cfg.k, dim=-1)
        # calculate max. num of tokens to be routed to an expert
        cap = max(
            math.ceil(t * self.cfg.k * self.capacity_factor / self.n_experts),
            1
        )
        y_out = torch.zeros_like(x_flat)
        self.drop_rate, assigned_e, accepted = 0, torch.zeros(size=(self.n_experts,), device=x.device), 0
        for e in range(self.n_experts):
            mask = topk_experts == e
            rows, cols = torch.nonzero(mask, as_tuple=True)
            if rows.numel() == 0:
                continue
            gates_e = topk_probs[rows, cols]
            if rows.numel() > cap:
                keep = gates_e.topk(cap).indices
                rows = rows[keep]
                gates_e = gates_e[keep]

            y_out[rows, :] += gates_e.unsqueeze(-1) * self.experts[e](x_flat[rows, :]) # TODO: replace this with index_add_
            accepted += rows.numel()
            assigned_e[e] += rows.numel()
        
        self.drop_rate = 1 - (accepted / (self.cfg.k * t))
        self.f_e = assigned_e / (self.cfg.k * t)
        self.p_e = expert_probs.mean(dim=0)
        self.aux_loss = (self.cfg.alpha_aux_loss * self.n_experts * (self.f_e * self.p_e).sum()).to(x.device)

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

class GPTMoE(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        # add weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(0.1 / fan_in(module))
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

    def router_stats(self):
        device = self.lm_head.weight.device
        n_e = self.cfg.n_experts
        n_b = len(self.blocks)

        f_sum = torch.zeros(n_e, device=device)
        p_sum = torch.zeros(n_e, device=device)
        drop_sum = torch.tensor(0.0, device=device)

        counted = 0
        for block in self.blocks:
            moe = block.ffn
            if hasattr(moe, 'f_e') and hasattr(moe, "p_e") and hasattr(moe, "drop_rate"):
                f_sum += moe.f_e
                p_sum += moe.p_e
                drop_sum += torch.as_tensor(moe.drop_rate, device=device, dtype=torch.float32)
                counted += 1
        
        if counted == 0:
            return None
        
        f_avg = (f_sum / counted).detach().cpu()
        p_avg = (p_sum / counted).detach().cpu()
        drop_avg = float((drop_sum / counted).item())

        return {
            "f_e": f_avg,
            "p_e": p_avg,
            "drop_rate": drop_avg
        }
    
    def generate(self, 
                 x: Tensor, 
                 max_tokens: int=20, 
                 temperature: float=0.7, 
                 top_p: float=0.9,
                 repetition_penalty: float=1.15,
                 repetition_penalty_length: int=10,
                 stop_token_id: int = 50256) -> list[int]:
        assert x.ndim == 1
        input_len = x.shape[0]
        x = x.view(1, x.shape[0])
        for _ in range(max_tokens):
            logits = self(x)
            # apply repetition_penalty
            recent_tokens = x[:, -repetition_penalty_length:].flatten()
            freqs = torch.bincount(recent_tokens, minlength=self.cfg.vocab_size)
            logits[:, -1, :] = logits[:, -1, :] / torch.pow(torch.tensor(repetition_penalty, device=x.device), freqs)
            # apply temperature
            next_token_probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            # top-p sampling
            sorted_probs, sorted_token_idxs = next_token_probs.sort(dim=-1, descending=True)
            cap_idx = (torch.cumsum(sorted_probs, dim=-1) > top_p).nonzero()
            cap_idx = cap_idx[0, 1].item() + 1 if cap_idx.numel() > 0 else next_token_probs.shape[-1]
            top_p_probs = sorted_probs[:, :cap_idx]
            # renormalize
            top_p_probs /= top_p_probs.sum(dim=-1, keepdim=True)
            # sample
            sampled_idx = torch.multinomial(top_p_probs, num_samples=1)
            next_token = sorted_token_idxs.gather(1, sampled_idx)
            x = torch.cat([x, next_token], dim=1)

            if next_token[0, 0].item() == stop_token_id:
                break
        
        return x[0].tolist()[input_len:]