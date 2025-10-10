from torch.utils.data import Dataset
from pathlib import Path
from config import TrainingConfig
import numpy as np, torch, typing

class ShardedDataset(Dataset):

    def __init__(self, config: TrainingConfig, shards_dir: Path) -> None:
        self.config = config
        self.shards_dir = shards_dir
        self.shard_paths = list(shards_dir.glob("shard_*.bin"))

        self.memmaps = [np.memmap(shard_path, dtype=np.uint16) for _, shard_path in enumerate(self.shard_paths)]
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


class SFTShardedDataset(Dataset):
    
    def __init__(self, shards_dir: Path) -> None:
        self.shards_dir = Path(shards_dir)
        
        if not self.shards_dir.exists():
            raise ValueError(f"Shards directory does not exist: {self.shards_dir}")
        
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.bin"))
        
        if len(self.shard_paths) == 0:
            raise ValueError(f"No shard files found in {self.shards_dir}")
        
        print(f"Found {len(self.shard_paths)} shards in {self.shards_dir}")
        
        self.memmaps = []
        for shard_path in self.shard_paths:
            memmap = np.memmap(shard_path, dtype=np.uint16, mode='r')
            
            num_rows = len(memmap) // 515
            if len(memmap) % 515 != 0:
                print(f"Warning: {shard_path.name} size is not a multiple of 515, truncating")
            
            shaped = memmap[:num_rows * 515].reshape(-1, 515)
            self.memmaps.append(shaped)
        
        shard_sizes = [len(memmap) for memmap in self.memmaps]
        self.cum_seqs = np.cumsum(shard_sizes)
        
        print(f"Total samples: {self.cum_seqs[-1]:,}")
    
    def __len__(self) -> int:
        return int(self.cum_seqs[-1])
    
    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, int, int]:
        shard_idx = int(np.searchsorted(self.cum_seqs, idx, side='right'))
        
        if shard_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - int(self.cum_seqs[shard_idx - 1])
        
        row = self.memmaps[shard_idx][local_idx]
        
        tokens = torch.from_numpy(row[:513].copy()).long()
        prompt_len = int(row[513])
        content_len = int(row[514])
        
        return tokens, prompt_len, content_len