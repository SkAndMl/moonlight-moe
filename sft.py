import pathlib, tiktoken, models, util, torch, math
from data import SFTDataset
from torch.utils.data import DataLoader
from moe import GPTMoE

device = util.get_device()
cp = torch.load("checkpoints/pretrain.pt", map_location=device)
model_cfg = models.ModelConfig(**cp["model_config"])
training_cfg = models.TrainingConfig(device=device)
tokenizer = tiktoken.get_encoding("gpt2")
ds = SFTDataset(pathlib.Path("sft_data.json"), tokenizer)
train_dl = DataLoader(ds, batch_size=training_cfg.batch_size, shuffle=True)


model = torch.compile(GPTMoE(model_cfg))
model.load_state_dict(cp["model"])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

total_steps = len(training_cfg) // training_cfg.accumulation_steps
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

prompt_mask = torch.tensor([torch.arange(0, 512) for _ in range(training_cfg.batch_size)]).to(device)

train_dl = iter(train_dl)
for step in range(total_steps):
    optimizer.zero_grad()
    loss_accum = 0
    for _ in training_cfg.accumulation_steps:
        tokens, prompt_lens, eot_lens = next(train_dl)
        tokens, prompt_lens, eot_lens = tokens.to(device), prompt_lens.to(device), eot_lens.to(device)
        x, y = tokens[:, -1], tokens[:, 1:] # bsz, 512; bsz, 512
        logits: torch.Tensor = model(x)
        y_masked = torch.where(prompt_mask < prompt_lens-1, -100, y)
        loss = torch.nn.functional.cross_entropy(logits.view(training_cfg.batch_size * 512, -1), y_masked.view(-1,), ignore_index=-100)
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach().item()
    optimizer.step()

    print(f"step: {step + 1:>5} | loss: {loss_accum:.4f}")