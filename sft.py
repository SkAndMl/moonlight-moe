import pathlib, tiktoken, config, util, torch, math
from data import SFTDataset
from torch.utils.data import DataLoader
from hf_utils import load_model_from_hf, upload_to_hf
from log import logger
from torch.nn import functional as F

device = util.get_device()
autocast_dtype = util.get_autocast_dtype(device)
training_cfg = config.TrainingConfig(device=device)

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# setup tokenizer and dataset
tokenizer = tiktoken.get_encoding("gpt2")
ds = SFTDataset(pathlib.Path("sft_data.json"), tokenizer)
train_dl = DataLoader(ds, batch_size=training_cfg.batch_size, shuffle=True)

# setup model and optimizer
model = load_model_from_hf(repo_id="SkAndMl/moonlight-moe-pretrain", filename="pretrain.pt", device=device)
model.train()
logger.info(f"loaded pretrained checkpoint from HF")
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

CKPT_DIR = pathlib.Path("checkpoints/sft")
CKPT_DIR.mkdir(exist_ok=True, parents=True)
def save_ckpt(tag: str):
    ckpt_path = CKPT_DIR / f"{tag}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_cfg": training_cfg.model_dump(),
            "model_cfg": model.cfg.model_dump()
        },
        ckpt_path
    )
    logger.info(f"ckpt saved to {ckpt_path}")

prompt_mask = torch.vstack([torch.arange(0, 512) for _ in range(training_cfg.batch_size)]).to(device)

logger.info("training...")
train_dl = iter(train_dl)
for step in range(total_steps):
    optimizer.zero_grad()
    loss_accum = 0
    for _ in range(training_cfg.accumulation_steps):
        tokens, prompt_lens, eot_lens = next(train_dl)
        tokens, prompt_lens, eot_lens = tokens.to(device), prompt_lens.unsqueeze(-1).to(device), eot_lens.to(device)
        x, y = tokens[:, :-1], tokens[:, 1:] # bsz, 512; bsz, 512
        bsz, seq_len = x.shape
        y_masked = torch.where(prompt_mask[:bsz, :] < prompt_lens - 1, -100, y)
        if autocast_dtype is not None:
            with torch.amp.autocast(device, autocast_dtype):
                logits: torch.Tensor = model(x)
                ce_loss = F.cross_entropy(
                    logits.view(bsz * seq_len, -1),
                    y_masked.view(-1,),
                    ignore_index=-100
                )
        else:
            logits: torch.Tensor = model(x)
            ce_loss = F.cross_entropy(
                logits.view(bsz * seq_len, -1),
                y_masked.view(-1,),
                ignore_index=-100
            )
        
        loss = ce_loss + model.moe_aux_loss()
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach().item()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    optimizer.step()

    if (step + 1) % 1000 == 0 or step == total_steps - 1:
        save_ckpt("sft")

    logger.info(f"step: {step + 1:>5} | loss: {loss_accum:.4f} | lr: {lr:2e} | norm: {norm.item():.4f}")


upload_to_hf(pathlib.Path("checkpoints/sft/sft.pt"), "SkAndMl/moonlight-moe-it")