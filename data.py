from torch.utils.data import Dataset
from pathlib import Path
from models import TrainingConfig
import numpy as np, torch, typing

class ShardedDataset(Dataset):

    def __init__(self, config: TrainingConfig, shards_dir: Path) -> None:
        self.config = config
        self.shards_dir = shards_dir
        self.shard_paths = list(shards_dir.glob("shard_*.bin"))

        self.memmaps = [np.memmap(shard_path, dtype=np.uint8) for i, shard_path in enumerate(self.shard_paths)]
        self.cum_seqs = np.cumsum([len(self.memmaps[i]) // (self.config.ctx_size + 1) for i in range(len(self.memmaps))])
    
    def __len__(self):
        return self.cum_seqs[-1]
    
    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor]:
        for i in range(len(self.cum_seqs)):
            if index < self.cum_seqs[i]:
                index = index - self.cum_seqs[i - 1] if i > 0 else index
                token_slice = self.memmaps[i][index * (self.config.ctx_size + 1): (index + 1) * (self.config.ctx_size + 1)]
                x, y = torch.from_numpy(token_slice[:-1]), torch.from_numpy(token_slice[1:])
                return x.long(), y.long()