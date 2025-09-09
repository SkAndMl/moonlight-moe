import torch

from pydantic import BaseModel
from torch import Tensor

class ModelConfig(BaseModel):
    base: int = 10000
    ctx_size: int = 256
    embed_dim: int = 512
    n_heads: int = 8

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
    _x = x.clone()
    _x = _x.view(bsz, n_heads, seq_len, head_dim // 2, 2)
    _x = torch.view_as_complex(_x) # bsz, n_heads, seq_len, head_dim / 2
    _x_rot = _x * freq_cis.unsqueeze(0).unsqueeze(0)[:, :, :seq_len, :] # bsz, n_heads, seq_len, head_dim / 2
    _x_rot = torch.view_as_real(_x_rot) # bsz, n_heads, seq_len, head_dim / 2, 2
    _x_rot = _x_rot.view(bsz, n_heads, seq_len, head_dim)
    return _x_rot