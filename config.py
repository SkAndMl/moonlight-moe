from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict

class ModelConfig(BaseModel):
    base: int = 10000
    vocab_size: int = 50304
    ctx_size: int = 512
    embed_dim: int = 384
    n_heads: int = 6
    ffn_dim: int = 384 * 4
    eps: float = 1e-8
    n_blocks: int = 6
    n_experts: int = 8
    k: int = 2 # top-k experts to route to
    capacity_factor: float = 1.25
    alpha_aux_loss: float = 1e-2

class TrainingConfig(BaseModel):
    ctx_size: int = 512
    batch_size: int = 16
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    weight_decay: float = 1e-2
    accumulation_steps: int = 8
    device: str = "cpu"


@dataclass
class SFTConfig:
    
    total_samples: int
    token_limit: int = 513
    val_split: float = 0.1
    random_seed: int = 2406
    
    task_distribution: Dict[str, float] = field(default_factory=lambda: {
        "generate_long": 0.40,   
        "generate_creative": 0.35,   
        "summarize": 0.10,
        "generate_short": 0.10,
        "inference": 0.05
    })
    
    def __post_init__(self):
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be between 0.0 and 1.0, got {self.val_split}")
        
        total = sum(self.task_distribution.values())
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"task_distribution must sum to 1.0, got {total}")
        
        for task, pct in self.task_distribution.items():
            if not 0.0 <= pct <= 1.0:
                raise ValueError(f"Invalid percentage for {task}: {pct}")
    
    def get_target_counts(self) -> Dict[str, int]:
        train_samples = int(self.total_samples * (1 - self.val_split))
        
        targets = {}
        for task, pct in self.task_distribution.items():
            targets[task] = int(train_samples * pct)
        
        return targets
    
    def get_train_val_counts(self) -> tuple[int, int]:
        train_count = int(self.total_samples * (1 - self.val_split))
        val_count = self.total_samples - train_count
        return train_count, val_count


@dataclass
class DatasetInfo:
    dataset_id: str
    split: str
    task_type: str
    subset: str = None