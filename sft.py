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
training_cfg = config.TrainingConfig(device=device, max_lr=5e-5, min_lr=5e-6)

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = SFTShardedDataset(shards_dir=train_dir, config=training_cfg)
    test_ds = SFTShardedDataset(shards_dir=test_dir, config=training_cfg)

    train_dl = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=training_cfg.batch_size)
    return train_dl, test_dl


# setup tokenizer and dataloaders
tokenizer = tiktoken.get_encoding("gpt2")
train_dl, test_dl = get_dataloaders(Path("sft_data/train"), Path("sft_data/val"), training_cfg)
# setup model and optimizer
model = load_model_from_hf(repo_id="SkAndMl/moonlight-moe-pretrain", filename="best.pt", device=device)
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

prompt_mask = torch.vstack([torch.arange(0, training_cfg.ctx_size) for _ in range(training_cfg.batch_size)]).to(device)

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
        mask = (prompt_mask[:bsz, :] < prompt_lens) | (prompt_mask[:bsz, :] >= (content_lens - 2))
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
def generate_samples(step):
    """Generate samples matching the types of data in SFT training"""
    model.eval()
    
    SYSTEM_ID = 50257
    USER_ID = 50258
    ASSISTANT_ID = 50259
    
    system_prompt = "You are a creative storyteller who writes engaging stories."
    
    # Test prompts matching your training data distribution
    test_prompts = [
        # WritingPrompts style (50% of training data)
        {
            "name": "WritingPrompts - Fantasy",
            "user_prompt": "You discover a door in your basement that wasn't there yesterday. When you open it, you find yourself in a magical library where books write themselves."
        },
        {
            "name": "WritingPrompts - Sci-Fi",
            "user_prompt": "In the future, humans can download skills directly into their brains. You just downloaded the wrong skill by mistake."
        },
        # TinyStories style (30% of training data) - Simple, child-friendly
        {
            "name": "TinyStories - Simple",
            "user_prompt": "Write a short story."
        },
        {
            "name": "TinyStories - Characters",
            "user_prompt": "Tell me a story about a cat and a dog who become friends."
        },
        # ROCStories style (20% of training data) - Very short, 5 sentences
        {
            "name": "ROCStories - Short Narrative",
            "user_prompt": "Create a short narrative."
        },
        # Mixed styles
        {
            "name": "Creative Adventure",
            "user_prompt": "Write a creative story about adventure."
        }
    ]
    
    logger.info("\n" + "="*70)
    logger.info(f"GENERATION SAMPLES - Step {step + 1}")
    logger.info("="*70)
    
    generations_html = []
    
    for prompt_data in test_prompts:
        prompt_name = prompt_data["name"]
        user_prompt = prompt_data["user_prompt"]
        
        # Build input tokens
        system_tokens = [SYSTEM_ID] + tokenizer.encode(system_prompt)
        user_tokens = [USER_ID] + tokenizer.encode(user_prompt)
        assistant_start = [ASSISTANT_ID]
        
        input_tokens = system_tokens + user_tokens + assistant_start
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=device)
        
        try:
            generated_tokens = model.generate(
                input_ids,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9
            )
            generated_text = tokenizer.decode(generated_tokens)
        except Exception as e:
            generated_text = f"[Generation failed: {e}]"
        
        # Log to console
        logger.info(f"\n{'-'*70}")
        logger.info(f"Task: {prompt_name}")
        logger.info(f"Prompt: {user_prompt}")
        logger.info(f"\nGenerated:")
        logger.info(generated_text)
        logger.info("-"*70)
        
        # Collect for wandb (single log)
        generations_html.append(
            f"<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ddd;'>"
            f"<h3 style='color: #333;'>{prompt_name}</h3>"
            f"<p><strong>Prompt:</strong> {user_prompt}</p>"
            f"<p><strong>Generated:</strong></p>"
            f"<p style='white-space: pre-wrap;'>{generated_text}</p>"
            f"</div>"
        )
    
    # Log all generations to wandb at once
    wandb.log({
        "generations": wandb.Html("".join(generations_html)),
        "step": step + 1
    })
    
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
        mask = (prompt_mask[:bsz, :] < prompt_lens) | (prompt_mask[:bsz, :] >= (content_lens - 2))
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

        generate_samples(step)
        if val_loss < best_test:
            best_test = val_loss
            save_ckpt("sft_best")
            logger.info(f"✅ new best @ step {step+1}: {best_test:.4f}")
    
    if (step + 1) % 1000 == 0 or step == total_steps - 1:
        save_ckpt("sft")

upload_to_hf(pathlib.Path("checkpoints/sft/sft_best.pt"), "SkAndMl/moonlight-moe-it")