from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict

class ModelConfig(BaseModel):
    base: int = 10000
    vocab_size: int = 50304
    ctx_size: int = 1024
    embed_dim: int = 512
    n_heads: int = 8
    ffn_dim: int = 512 * 4
    eps: float = 1e-8
    n_blocks: int = 8
    n_experts: int = 6
    k: int = 2 # top-k experts to route to
    capacity_factor: float = 1.25
    alpha_aux_loss: float = 1e-2

class TrainingConfig(BaseModel):
    ctx_size: int = 1024
    batch_size: int = 16
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    weight_decay: float = 1e-2
    accumulation_steps: int = 8
    device: str = "cpu"