import pathlib, tiktoken, models, util, torch, math
from data import SFTDataset
from torch.utils.data import DataLoader
from moe import GPTMoE

device = util.get_device()
cp = torch.load("checkpoints/pretrain.pt", map_location=device)
model_cfg = models.ModelConfig(**cp["model_cfg"])
training_cfg = models.TrainingConfig(device=device)
tokenizer = tiktoken.get_encoding("gpt2")
ds = SFTDataset(pathlib.Path("sft_data.json"), tokenizer)
train_dl = DataLoader(ds, batch_size=training_cfg.batch_size, shuffle=True)


model = GPTMoE(model_cfg).to(device)
model = torch.compile(model)
model.load_state_dict(cp["model"])

non_decay_names = ["norm", "bias"]
param_groups = {
    "non_decay": [p for name, p in model.named_parameters() if any(_ in name for _ in non_decay_names)],
    "decay": [p for name, p in model.named_parameters() if not any(_ in name for _ in non_decay_names)]
}

optimizer = torch.optim.AdamW(
    params=[
        {"params": param_groups["non_decay"], "weight_decay": 0},
        {"params": param_groups["decay"], "weight_decay": training_cfg.weight_decay}
    ]
)

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

prompt_mask = torch.vstack([torch.arange(0, 512) for _ in range(training_cfg.batch_size)]).to(device)

print("training...")
train_dl = iter(train_dl)
for step in range(total_steps):
    optimizer.zero_grad()
    loss_accum = 0
    for _ in range(training_cfg.accumulation_steps):
        tokens, prompt_lens, eot_lens = next(train_dl)
        tokens, prompt_lens, eot_lens = tokens.to(device), prompt_lens.unsqueeze(-1).to(device), eot_lens.to(device)
        x, y = tokens[:, :-1], tokens[:, 1:] # bsz, 512; bsz, 512
        logits: torch.Tensor = model(x)
        y_masked = torch.where(prompt_mask < prompt_lens - 1, -100, y)
        loss = torch.nn.functional.cross_entropy(logits.view(training_cfg.batch_size * 512, -1), y_masked.view(-1,), ignore_index=-100)
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach().item()
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    print(f"step: {step + 1:>5} | loss: {loss_accum:.4f}")