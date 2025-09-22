from pydantic import BaseModel

class ModelConfig(BaseModel):
    base: int = 10000
    vocab_size: int = 50304
    ctx_size: int = 256
    embed_dim: int = 384
    n_heads: int = 6
    ffn_dim: int = 384 * 4
    eps: float = 1e-8
    n_blocks: int = 8
    n_experts: int = 8
    capacity_factor: float = 1.0
    alpha_aux_loss: float = 1e-2


class TrainingConfig(BaseModel):
    ctx_size: int = 256
    batch_size: int = 16
    lr: float = 1e-3