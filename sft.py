import pathlib, tiktoken, config, util, torch, math, time, wandb, prompts
from data import SFTShardedDataset
from torch.utils.data import DataLoader
from hf_utils import load_model_from_hf, upload_to_hf
from log import logger
from torch.nn import functional as F
from config import TrainingConfig
from pathlib import Path
from datetime import datetime

device = util.get_device()
autocast_dtype = util.get_autocast_dtype(device)
logger.info(f"[sft] dtype: {autocast_dtype}")
training_cfg = config.TrainingConfig(device=device)

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = SFTShardedDataset(shards_dir=train_dir)
    test_ds = SFTShardedDataset(shards_dir=test_dir)

    train_dl = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=training_cfg.batch_size)
    return train_dl, test_dl


# setup tokenizer and dataloaders
tokenizer = tiktoken.get_encoding("gpt2")
train_dl, test_dl = get_dataloaders(Path("sft_data/shards/train"), Path("sft_data/shards/val"), training_cfg)
# setup model and optimizer
model = load_model_from_hf(repo_id="SkAndMl/moonlight-moe-pretrain", filename="pretrain.pt", device=device)
model.train()
logger.info(f"loaded pretrained checkpoint from HF")
non_decay_names = ["norm", "bias"]
param_groups = {
    "non_decay": [p for name, p in model.named_parameters() if any(_ in name for _ in non_decay_names)],
    "decay": [p for name, p in model.named_parameters() if not any(_ in name for _ in non_decay_names)]
}
logger.info(f"[sft] model params: {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"[sft] decay params: {sum(p.numel() for p in param_groups['decay']):,}")
logger.info(f"[sft] non_decay params: {sum(p.numel() for p in param_groups['non_decay']):,}")

optimizer = torch.optim.AdamW(
    params=[
        {"params": param_groups["non_decay"], "weight_decay": 0},
        {"params": param_groups["decay"], "weight_decay": training_cfg.weight_decay}
    ]
)

total_steps = len(train_dl) // training_cfg.accumulation_steps
warmup_steps = int(0.02 * total_steps)
logger.info(f"[sft] total_steps: {total_steps} | warmup_steps: {warmup_steps}")

wandb.init(
    project=f"gpt-moe-sft-{datetime.now().strftime('%d_%m_%Y')}",
    config={
        "model_cfg": model.cfg.model_dump(),
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

@torch.inference_mode()
def eval():
    model.eval()
    loss_accum = 0
    for tokens, prompt_lens, content_lens in test_dl:
        tokens, prompt_lens, content_lens = tokens.to(device), prompt_lens.to(device), content_lens.to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        prompt_lens, content_lens = prompt_lens.unsqueeze(-1), content_lens.unsqueeze(-1)
        assert x.shape == y.shape
        bsz, seq_len = x.shape
        # some more shape asserts
        assert prompt_lens.shape == content_lens.shape == (bsz, 1)
        # prepare mask
        mask = (prompt_mask[:bsz, :] < prompt_lens - 1) | (prompt_mask[:bsz, :] >= content_lens - 1)
        y_masked = torch.where(mask, -100, y)
        # mixed precision training
        if autocast_dtype is not None:
            with torch.amp.autocast(device, autocast_dtype):
                logits: torch.Tensor = model(x)
                loss = F.cross_entropy(
                    logits.view(bsz*seq_len, -1),
                    y_masked.view(-1,),
                    ignore_index=-100
                )
        else:
            logits: torch.Tensor = model(x)
            loss = F.cross_entropy(
                logits.view(bsz*seq_len, -1),
                y_masked.view(-1,),
                ignore_index=-100
            )
        loss_accum += loss.item()
    
    model.train()

    return loss_accum / len(test_dl)


@torch.inference_mode()
def generate_samples():
    model.eval()
    test_prompts = {
        "generate_long": {
            "user_content": "Features: Lily, garden, butterfly\nWords: colorful, flying, happy",
            "task": "generate_long"
        },
        "generate_creative": {
            "user_content": "You wake up one day to find that you can hear what animals are thinking.",
            "task": "generate_creative"
        },
        "generate_short": {
            "user_content": "The Lost Treasure",
            "task": "generate_short"
        },
        "summarize": {
            "user_content": "Story: Once upon a time, there was a little girl named Lucy. She loved to play in the park with her friends. One day, she found a beautiful red ball under a big tree. Lucy was so happy! She played with the ball all day long. When it was time to go home, Lucy took the ball with her and showed it to her mom. Her mom smiled and said it was a very special ball.",
            "task": "summarize"
        },
        "inference": {
            "user_content": {
                "paragraph": "Tom was very excited. Today was his birthday and all his friends were coming to his party.",
                "option_1": "Tom felt sad and went to his room.",
                "option_2": "Tom smiled and helped his mom decorate the house."
            },
            "task": "inference"
        }
    }
    
    logger.info("\n" + "="*70)
    logger.info("GENERATION SAMPLES")
    logger.info("="*70)
    
    for task_name, prompt_data in test_prompts.items():
        user_content = prompt_data["user_content"]
        task = prompt_data["task"]
        
        user_content_formatted = prompts.format_user_content(task, user_content)
        x_part, _ = util.get_x_y_for_sft(prompts.SYSTEM_PROMPT, user_content_formatted, "")
        
        input_ids = torch.tensor(tokenizer.encode(x_part), device=device)
        generated_ids = model.generate(input_ids)
        generated_text = tokenizer.decode(generated_ids)
        
        logger.info(f"\nTask: {task_name}")
        if isinstance(user_content, dict):
            logger.info(f"Prompt: {user_content['paragraph']}")
        else:
            logger.info(f"Prompt: {user_content[:100]}...") 
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 70)
    
    logger.info("="*70 + "\n")
    model.train()


logger.info("training...")
train_dl = iter(train_dl)
best_test = float("inf")
for step in range(total_steps):
    optimizer.zero_grad()
    loss_accum, moe_aux_loss_accum = 0., 0.
    t0 = time.time()
    for _ in range(training_cfg.accumulation_steps):
        tokens, prompt_lens, content_lens = next(train_dl)
        tokens, prompt_lens, content_lens = tokens.to(device), prompt_lens.unsqueeze(-1).to(device), content_lens.unsqueeze(-1).to(device)
        x, y = tokens[:, :-1], tokens[:, 1:] # bsz, 512; bsz, 512
        assert x.shape == y.shape
        bsz, seq_len = x.shape
        # some more shape asserts
        assert prompt_lens.shape == content_lens.shape == (bsz, 1)
        # mask positions where
        # pos < prompt_len - 1 (system/user prompt)
        # pos >= content_len - 1 (padding tokens)
        # keeps only positions where we are predicting the assistant response
        mask = (prompt_mask[:bsz, :] < prompt_lens - 1) | (prompt_mask[:bsz, :] >= content_lens - 1)
        y_masked = torch.where(mask, -100, y)
        # mixed precision training
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
        # add auxilary loss
        moe_aux_loss = model.moe_aux_loss()
        loss = ce_loss + moe_aux_loss
        # divide by accumulation steps to maintain the average
        loss /= training_cfg.accumulation_steps
        loss.backward()
        # accumulate loss
        loss_accum += loss.detach().item()
        moe_aux_loss_accum += moe_aux_loss.detach().item() / training_cfg.accumulation_steps
    # throughput calculation
    total_time = time.time() - t0 # seconds
    num_tokens = training_cfg.batch_size * training_cfg.ctx_size * training_cfg.accumulation_steps
    token_throughput = num_tokens / total_time
    # gradient clipping; to avoid exploding gradients problem
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # apply cosine lr decay
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr    
    optimizer.step()

    logger.info(f"step: {step + 1:>5} | loss: {loss_accum:.4f} | lr: {lr:2e} | norm: {norm.item():.4f} | token throughput: {token_throughput:.4f}")

    wandb.log({
        "train/loss": loss_accum,
        "train/moe_aux_loss": moe_aux_loss_accum,
        "train/lr": lr,
        "train/grad_norm": norm.item(),
        "step": step + 1
    })

    if (step + 1) % 200 == 0 or step == total_steps - 1:
        val_loss = eval()
        logger.info(f"step: {step + 1:>5} | val_loss: {val_loss:.4f}")
        wandb.log({
            "test/loss": val_loss
        })

        generate_samples()
        if val_loss < best_test:
            best_test = val_loss
            save_ckpt("sft_best")
            logger.info(f"✅ new best @ step {step+1}: {best_test:.4f}")
    
    if (step + 1) % 1000 == 0 or step == total_steps - 1:
        save_ckpt("sft")  # Periodic checkpoint for recover

upload_to_hf(pathlib.Path("checkpoints/sft/sft_best.pt"), "SkAndMl/moonlight-moe-it")