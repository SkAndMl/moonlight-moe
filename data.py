from torch.utils.data import Dataset
from pathlib import Path
from models import TrainingConfig
import numpy as np, torch, typing, json, prompts, util

class ShardedDataset(Dataset):

    def __init__(self, config: TrainingConfig, shards_dir: Path) -> None:
        self.config = config
        self.shards_dir = shards_dir
        self.shard_paths = list(shards_dir.glob("shard_*.bin"))

        self.memmaps = [np.memmap(shard_path, dtype=np.uint16) for i, shard_path in enumerate(self.shard_paths)]
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


class SFTDataset(Dataset):

    def __init__(self, sft_data_path: Path, tokenizer) -> None:

        with open(sft_data_path, "r") as f:
            self.sft_data = json.loads(f.read())

        self.tokenizer = tokenizer

    def __len__(self) -> int: 
        return len(self.sft_data["user_content"])

    def __getitem__(self, idx: int):
        
        task: typing.Literal["generate_long", "generate_short", "summarize", "inference"] = self.sft_data["task"][idx]
        match task:
            case "inference":
                user_content = prompts.INFERENCE_USER_PROMPT.format(**self.sft_data["user_content"][idx])
                assistant_content = self.sft_data["assistant_content"][idx]
                x_str, y_str = util.get_x_y_for_sft(prompts.SYSTEM_PROMPT, user_content, assistant_content)
            case "generate_long":
                user_content = prompts.STORY_GENERATION_LONG_USER_PROMPT.format(user_content=self.sft_data["user_content"][idx])
                x_str, y_str = util.get_x_y_for_sft(prompts.SYSTEM_PROMPT, user_content, self.sft_data["assistant_content"][idx])
            case "generate_short":
                user_content = prompts.STORY_GENERATION_SHORT_USER_PROMPT.format(user_content=self.sft_data["user_content"][idx])
                x_str, y_str = util.get_x_y_for_sft(prompts.SYSTEM_PROMPT, user_content, self.sft_data["assistant_content"][idx])
            case "summarize":
                user_content = prompts.SUMMARIZE_USER_PROMPT.format(story=self.sft_data["user_content"][idx])
                x_str, y_str = util.get_x_y_for_sft(prompts.SYSTEM_PROMPT, user_content, self.sft_data["assistant_content"][idx])
        
        x_tokens = self.tokenizer.encode(x_str, allowed_special="all")
        y_tokens = self.tokenizer.encode(y_str, allowed_special="all")

        tokens = x_tokens + y_tokens
        if len(tokens) < 513:
            tokens.extend([50256] * (513 - len(tokens)))
        
        return torch.tensor(tokens).long(), len(x_tokens), len(x_tokens + y_tokens)