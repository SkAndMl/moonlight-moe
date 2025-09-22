from data import ShardedDataset
from torch.utils.data import DataLoader
from models import TrainingConfig, ModelConfig
from moe import GPTMoE
from pathlib import Path
from torch.optim import AdamW

import torch, tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = ShardedDataset(config=training_cfg, shards_dir=train_dir)
    test_ds = ShardedDataset(config=training_cfg, shards_dir=test_dir)

    train_dl = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=training_cfg.batch_size)
    return train_dl, test_dl


training_cfg, model_cfg = TrainingConfig(), ModelConfig()
train_dl, test_dl = get_dataloaders(Path("shards/train"), Path("shards/test"), training_cfg)

model = GPTMoE(model_cfg)
optimizer = AdamW(params=model.parameters(), lr=training_cfg.lr)


for step, (x, y) in enumerate(train_dl):

    bsz, seq_len = x.shape
    logits: torch.Tensor = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(bsz * seq_len, -1), y.view(-1,)) + model.moe_aux_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step: {step + 1} | loss: {loss.item():.4f}")
    
    if (step + 1) % 100 == 0:
        start = "There was a"
        gen = model.generate(torch.tensor(tokenizer.encode(start)))
        print(f"step: {step + 1} | generation: {tokenizer.decode(gen)}")
    
    if step == 500:
        break