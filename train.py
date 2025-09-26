from data import ShardedDataset
from torch.utils.data import DataLoader
from models import TrainingConfig, ModelConfig
from moe import GPTMoE
from pathlib import Path
from torch.optim import AdamW
from typing import Literal

import torch, tiktoken, time, math

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

device = get_device()
print(f"training on {device}...")
tokenizer = tiktoken.get_encoding("gpt2")
training_cfg, model_cfg = TrainingConfig(device=device), ModelConfig()
train_dl, test_dl = get_dataloaders(Path("shards/train"), Path("shards/test"), training_cfg)

total_steps = len(train_dl) // training_cfg.accumulation_steps
warmup_steps = int(0.02 * total_steps)

def get_lr(it: int):

    min_lr, max_lr = training_cfg.min_lr, training_cfg.max_lr
    if it < warmup_steps:
        return ((it + 1) / warmup_steps) * max_lr
    
    elif warmup_steps <= it <= total_steps:
        step_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (1/2) * (max_lr - min_lr) * (1 + math.cos(math.pi * step_ratio))
    else:
        return min_lr


model = GPTMoE(model_cfg).to(device)
model = torch.compile(model)
print(f"compiled model...")

param_groups = {
    "decay_params": [p for name, p in model.named_parameters() if 'bias' not in name],
    "non_decay_params": [p for name, p in model.named_parameters() if 'bias' in name]
}
optimizer = AdamW(
    params=[
        {"params": param_groups["decay_params"], "weight_decay": training_cfg.weight_decay},
        {"params": param_groups["non_decay_params"], "weight_decay": 0}
    ]
)

train_dl = iter(train_dl)
for step in range(total_steps):

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum, moe_aux_loss_accum = 0.0, 0.0
    for _ in range(training_cfg.accumulation_steps):
        x, y = next(train_dl)
        x, y = x.to(device), y.to(device)
        bsz, seq_len = x.shape
        logits: torch.Tensor = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(bsz * seq_len, -1), y.view(-1,)) + model.moe_aux_loss()
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach()
        moe_aux_loss_accum += model.moe_aux_loss().detach() / training_cfg.accumulation_steps
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    throughput = (bsz * seq_len * training_cfg.accumulation_steps) / (time.time() - t0)
    print(f"step: {step + 1:>5} | loss: {loss_accum.item():.4f} | moe_aux_loss: {moe_aux_loss_accum.item():.4f} | lr: {lr:.6f} | tokens per sec: {throughput:.4f} | norm: {norm.item():.4f}")
    
    if (step + 1) % 100 == 0:
        start = "There was a"
        gen = model.generate(torch.tensor(tokenizer.encode(start), device=device))
        print(f"step: {step + 1} | generation: {tokenizer.decode(gen)}")
    
    if step == 500:
        break