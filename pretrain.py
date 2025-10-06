from models import TrainingConfig, ModelConfig
from moe import GPTMoE
from pathlib import Path
from torch.optim import AdamW
from datetime import datetime
from log import logger
from torch.utils.data import DataLoader
from data import ShardedDataset

import torch, tiktoken, time, math, wandb, util, json

util.set_seed()

device = util.get_device()
logger.info(f"training on {device}...")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = ShardedDataset(config=training_cfg, shards_dir=train_dir)
    test_ds = ShardedDataset(config=training_cfg, shards_dir=test_dir)

    train_dl = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=training_cfg.batch_size)
    return train_dl, test_dl


tokenizer = tiktoken.get_encoding("gpt2")
training_cfg, model_cfg = TrainingConfig(device=device), ModelConfig()
train_dl, test_dl = get_dataloaders(Path("shards/train"), Path("shards/validation"), training_cfg)

total_steps = len(train_dl) // training_cfg.accumulation_steps
warmup_steps = int(0.02 * total_steps)

wandb.init(
    project=f"gpt-moe-pretraining-{datetime.now().strftime('%d_%m_%Y')}",
    config={
        "model_cfg": model_cfg.model_dump(),
        "training_cfg": training_cfg.model_dump(),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps
    }
)

def get_lr(it: int):

    min_lr, max_lr = training_cfg.min_lr, training_cfg.max_lr
    if it < warmup_steps:
        return ((it + 1) / warmup_steps) * max_lr
    
    elif warmup_steps <= it <= total_steps:
        step_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (1/2) * (max_lr - min_lr) * (1 + math.cos(math.pi * step_ratio))
    else:
        return min_lr

@torch.inference_mode()
def evaluate() -> torch.Tensor:
    model.eval()
    loss_accum = 0
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        bsz, seq_len = x.shape
        logits: torch.Tensor = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(bsz*seq_len, -1), y.view(-1,))
        loss_accum += loss.detach()
    model.train()
    return loss_accum / len(test_dl)


model = GPTMoE(model_cfg).to(device)
print(f"Total params: {sum(p.numel() for p in model.parameters())}")
model = torch.compile(model)
logger.info(f"compiled model...")

nondecay_group = ['bias', 'norm']
param_groups = {
    "decay_params": [p for name, p in model.named_parameters() if not any(_ in name for _ in nondecay_group)],
    "non_decay_params": [p for name, p in model.named_parameters() if any(_ in name for _ in nondecay_group)]
}
optimizer = AdamW(
    params=[
        {"params": param_groups["decay_params"], "weight_decay": training_cfg.weight_decay},
        {"params": param_groups["non_decay_params"], "weight_decay": 0}
    ]
)

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
best_test = float("inf")
def save_ckpt(tag: str, step: int, val_loss: float=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "training_cfg": training_cfg.model_dump(),
        "model_cfg": model_cfg.model_dump(),
    }
    path = CKPT_DIR / f"{tag}.pt"
    torch.save(state, path)
    meta = {"tag": tag, "step": step, "val_loss": float(val_loss) if val_loss is not None else None}
    with open(CKPT_DIR / f"{tag}.json", "w") as f:
        json.dump(meta, f)
    return path


train_dl = iter(train_dl)
for step in range(total_steps):

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum, moe_aux_loss_accum = 0.0, 0.0
    f_accum, p_accum, drop_accum = torch.zeros(model_cfg.n_experts, device="cpu"), torch.zeros(model_cfg.n_experts, device="cpu"), 0.0
    counted = 0
    for _ in range(training_cfg.accumulation_steps):
        x, y = next(train_dl)
        x, y = x.to(device), y.to(device)
        bsz, seq_len = x.shape
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits: torch.Tensor = model(x)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(bsz * seq_len, -1), y.view(-1,)
                )
        else:
            logits: torch.Tensor = model(x)
            ce = torch.nn.functional.cross_entropy(
                logits.view(bsz * seq_len, -1), y.view(-1,)
            ) 
        
        loss = ce + model.moe_aux_loss()
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach()
        moe_aux_loss_accum += model.moe_aux_loss().detach() / training_cfg.accumulation_steps
    
        stats = model.router_stats()
        if stats is not None:
            f_accum += stats["f_e"]
            p_accum += stats["p_e"]
            drop_accum += stats["drop_rate"]
            counted += 1

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    throughput = (bsz * seq_len * training_cfg.accumulation_steps) / (time.time() - t0)
    logger.info(f"step: {step + 1:>5} | loss: {loss_accum.item():.4f} | moe_aux_loss: {moe_aux_loss_accum.item():.4f} | lr: {lr:.6f} | tokens per sec: {throughput:.4f} | norm: {norm.item():.4f}")
    
    wandb.log({
        "train/loss": loss_accum.item(),
        "train/moe_aux_loss": moe_aux_loss_accum.item(),
        "train/lr": lr,
        "train/throughput": throughput,
        "train/grad_norm": norm.item(),
        "step": step + 1
    })

    if (step + 1) % 100 == 0:
        if counted == 0:
            logger.info(f"step: {step + 1:>5} | no router stats")
        else:
            f_avg = (f_accum / counted)
            p_avg = (p_accum / counted)
            drop_avg = drop_accum / counted
            f_top = float(f_avg.max().item())
            f_min = float(f_avg.min().item())
            p_entropy = float(-(p_avg * (p_avg.clamp(1e-12)).log()).sum().item())
            logger.info(f"step: {step + 1:>5} | drop_rate: {drop_avg:.4f} | f_top: {f_top:.4f} | f_min: {f_min:.4f} | p_entropy: {p_entropy:.4f}")

            wandb.log({
                "moe/drop_rate": drop_avg,
                "moe/f_top": f_top,
                "moe/f_min": f_min,
                "moe/p_entropy": p_entropy,
                "moe/expert_frequencies": wandb.Histogram(f_avg.numpy()),
                "moe/expert_probabilities": wandb.Histogram(p_avg.numpy()),
                "step": step + 1
            })

    if (step + 1) % 200 == 0 or step == total_steps - 1:
        test_loss = evaluate()
        val = float(test_loss.item())
        logger.info(f"step: {step + 1:>5} | val_loss: {val:.4f}")
        wandb.log({
            "test/loss": test_loss.item()
        })

        save_ckpt("last", step + 1, val)

        if val < best_test:
            best_test = val
            save_ckpt("best", step + 1, best_test)
            logger.info(f"✅ new best @ step {step+1}: {best_test:.4f}")

    if (step + 1) % 200 == 0:
        start = "There was a"
        gen = model.generate(torch.tensor(tokenizer.encode(start), device=device))
        gen_text = tokenizer.decode(gen)
        logger.info(f"step: {step + 1:>5} | generation: {gen_text}")
        wandb.log({
            "generation": wandb.Html(f"<p><b>Prompt:</d> {start}<br><b>Generated:</b> {gen_text}</p>"),
            "step": step + 1
        })