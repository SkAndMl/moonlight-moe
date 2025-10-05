from pathlib import Path
from models import TrainingConfig
from typing import Literal
from data import ShardedDataset
from torch.utils.data import DataLoader
import torch, random, numpy as np

def set_seed(seed=2406):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = ShardedDataset(config=training_cfg, shards_dir=train_dir)
    test_ds = ShardedDataset(config=training_cfg, shards_dir=test_dir)

    train_dl = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=training_cfg.batch_size)
    return train_dl, test_dl

def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def format_data_for_sft(system_prompt, user_content, assistant_content) -> str:
    formatted_content = f"""
<|system|>
{system_prompt}
<|user|>
{user_content}
<|assistant|>
{assistant_content}
<|endoftext|>
"""
    return formatted_content