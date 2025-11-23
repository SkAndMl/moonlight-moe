from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn import functional as F
from datasets import load_dataset
from typing import Literal
from log import logger
from hf_utils import load_model_from_hf, upload_to_hf
from datetime import datetime
from config import tokenizer

import torch, util, config, operator, wandb, math, pathlib, time
import re, random


SYSTEM_PROMPT = """
You are a math reasoning assistant.
Solve the following problem. Think step by step, but put your reasoning inside <thought>...</thought>
and the final result inside <answer>...</answer>.

Problem: {question}
"""

class GSM8KDS(Dataset):

    def __init__(self, split: Literal["train", "test"], ctx_size: int):
        self.rows = self._get_valid_rows(split, ctx_size)
    
    def _format_gsm8k_answer(self, answer: str) -> tuple[str, str]:
        thought, final_answer = answer.split("####")
        return thought.strip(), final_answer.strip()

    def _get_valid_rows(self, split: Literal["train", "test"], ctx_size) -> list[int]:
        ds = load_dataset("openai/gsm8k", "main")[split]
        rows = []
        for row in ds:
            question, answer = row["question"], row["answer"]
            thought, final_answer = self._format_gsm8k_answer(answer)

            system_prompt = SYSTEM_PROMPT.format(question=question)
            assistant_content = f"<thought>{thought}</thought><answer>{final_answer}</answer>"

            tokens = [config.SYSTEM_TOKEN_ID] + \
                    tokenizer.encode(system_prompt) + \
                    [config.ASSISTANT_TOKEN_ID] + \
                    tokenizer.encode(assistant_content) + \
                    [config.EOT_TOKEN_ID]
            
            if len(tokens) < ctx_size + 1:
                tokens += [config.EOT_TOKEN_ID] * (ctx_size + 1 - len(tokens))
            
            if len(tokens) > ctx_size + 1:
                continue
            rows.append(tokens)
        return rows

    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert idx < len(self.rows)
        row = self.rows[idx]
        return torch.tensor(row[:-1]).long(), torch.tensor(row[1:]).long()


class MetaMathQA(Dataset):

    def __init__(self, ctx_size: int, num_samples: int):
        self.rows = self._get_valid_rows(ctx_size=ctx_size, num_samples=num_samples)
    
    def _get_valid_rows(self, ctx_size: int, num_samples: int):
        ds = load_dataset("meta-math/MetaMathQA")["train"]
        assert num_samples <= ds.num_rows
        random.seed(2406)
        random_idxs = random.sample(range(ds.num_rows), k=num_samples)
        rows = []
        for random_idx in random_idxs:
            row = ds[random_idx]
            question, answer = row["original_question"], row["response"]

            parts = answer.rsplit("The answer is: ", 1)
            if len(parts) != 2 or re.fullmatch(r"\d+", parts[-1].strip()) is None:
                continue
            
            thought, final_answer = parts
            system_prompt = SYSTEM_PROMPT.format(question=question.strip())
            assistant_content = f"<thought>{thought}</thought><answer>{final_answer}</answer>"

            tokens = [config.SYSTEM_TOKEN_ID] + \
                     tokenizer.encode(system_prompt) + \
                     [config.ASSISTANT_TOKEN_ID] + \
                     tokenizer.encode(assistant_content) + \
                     [config.EOT_TOKEN_ID]
            
            if len(tokens) > ctx_size + 1:
                continue
            if len(tokens) < ctx_size + 1:
                tokens += [config.EOT_TOKEN_ID] * (ctx_size + 1 - len(tokens))
            
            rows.append(tokens)
        
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        return torch.tensor(row[:-1]).long(), torch.tensor(row[1:]).long()

device = util.get_device()
autocast_dtype = util.get_autocast_dtype(device)
logger.info(f"[math-sft] dtype: {autocast_dtype}")
training_cfg = config.TrainingConfig(device=device, max_lr=1e-4, min_lr=2e-5)

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_dataloaders(training_cfg: config.TrainingConfig):
    train_ds_1 = GSM8KDS(split="train", ctx_size=training_cfg.ctx_size)
    train_ds_2 = MetaMathQA(ctx_size=training_cfg.ctx_size, num_samples=20000)
    test_ds = GSM8KDS(split="test", ctx_size=training_cfg.ctx_size)

    train_dl = DataLoader(
        ConcatDataset([train_ds_1, train_ds_2]), 
        batch_size=training_cfg.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=training_cfg.batch_size,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    return train_dl, test_dl


# setup dataloaders
train_dl, test_dl = get_dataloaders(training_cfg)
# setup model and optimizer
model = load_model_from_hf(repo_id="SkAndMl/moonlight-moe-it", filename="it_best.pt", device=device)
model.train()
logger.info(f"loaded pretrained checkpoint from HF")
non_decay_names = ["norm", "bias"]
param_groups = {
    "non_decay": [p for name, p in model.named_parameters() if any(_ in name for _ in non_decay_names)],
    "decay": [p for name, p in model.named_parameters() if not any(_ in name for _ in non_decay_names)]
}
logger.info(f"[math-sft] model params: {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"[math-sft] decay params: {sum(p.numel() for p in param_groups['decay']):,}")
logger.info(f"[math-sft] non_decay params: {sum(p.numel() for p in param_groups['non_decay']):,}")

optimizer = torch.optim.AdamW(
    params=[
        {"params": param_groups["non_decay"], "weight_decay": 0},
        {"params": param_groups["decay"], "weight_decay": training_cfg.weight_decay}
    ],
    lr=training_cfg.max_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=device.startswith("cuda")
)

n_epochs = 5
total_steps = n_epochs * len(train_dl) // training_cfg.accumulation_steps
warmup_steps = int(0.10 * total_steps)
logger.info(f"[math-sft] total_steps: {total_steps} | warmup_steps: {warmup_steps}")

wandb.init(
    project=f"gpt-moe-math-sft-{datetime.now().strftime('%d_%m_%Y')}",
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
    

CKPT_DIR = pathlib.Path("checkpoints/grpo")
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
def construct_mask(y: torch.Tensor) -> torch.Tensor:

    def _construct_mask(token_id, op=operator.le):
        _mask = y == token_id
        indices = _mask.long().argmax(dim=1).unsqueeze(1)
        mask = op(prompt_mask[:y.shape[0], :], indices)
        return mask

    content_mask = _construct_mask(config.ASSISTANT_TOKEN_ID, operator.le)
    eot_mask = _construct_mask(config.EOT_TOKEN_ID, operator.gt)
    return content_mask | eot_mask

@torch.inference_mode()
def eval():
    model.eval()
    loss_accum = 0
    for x, y in test_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        assert x.shape == y.shape
        bsz, seq_len = x.shape
        mask = construct_mask(y)
        y_masked = torch.where(mask, -100, y)
        # mixed precision training
        if autocast_dtype is not None:
            with torch.amp.autocast(device, autocast_dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(bsz*seq_len, -1),
                    y_masked.view(-1,),
                    ignore_index=-100
                )
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(bsz*seq_len, -1),
                y_masked.view(-1,),
                ignore_index=-100
            )
        loss_accum += loss.item()
    
    model.train()

    return loss_accum / len(test_dl)


logger.info("training...")
train_dl_iter = iter(train_dl)
best_test = float("inf")

for step in range(total_steps):
    optimizer.zero_grad()
    loss_accum, moe_aux_loss_accum = 0., 0.
    t0 = time.time()
    for _ in range(training_cfg.accumulation_steps):
        try:
            x, y = next(train_dl_iter)
        except StopIteration:
            train_dl_iter = iter(train_dl)
            x, y = next(train_dl_iter)

        assert x.shape == y.shape
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bsz, seq_len = x.shape
        mask = construct_mask(y)
        y_masked = torch.where(mask, -100, y)
        # mixed precision training
        if autocast_dtype is not None:
            with torch.amp.autocast(device, autocast_dtype):
                logits, _ = model(x)
                ce_loss = F.cross_entropy(
                    logits.view(bsz * seq_len, -1),
                    y_masked.view(-1,),
                    ignore_index=-100
                )
        else:
            logits, _ = model(x)
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

    logger.info(f"[math-sft] step: {step + 1:>5} | loss: {loss_accum:.4f} | lr: {lr:2e} | norm: {norm.item():.4f} | token throughput: {token_throughput:.4f}")

    wandb.log({
        "train/loss": loss_accum,
        "train/moe_aux_loss": moe_aux_loss_accum,
        "train/lr": lr,
        "train/grad_norm": norm.item(),
        "step": step + 1
    })

    if (step + 1) % 500 == 0 or step == total_steps - 1:
        val_loss = eval()
        logger.info(f"step: {step + 1:>5} | val_loss: {val_loss:.4f}")
        wandb.log({
            "test/loss": val_loss
        })

        # generate_samples(step)
        if val_loss < best_test:
            best_test = val_loss
            save_ckpt("math_sft_best")
            logger.info(f"new best @ step {step+1}: {best_test:.4f}")
    
    if (step + 1) % 500 == 0 or step == total_steps - 1:
        save_ckpt("math_sft")

upload_to_hf(pathlib.Path("checkpoints/grpo/math_sft_best.pt"), "SkAndMl/moonlight-moe-math-sft")