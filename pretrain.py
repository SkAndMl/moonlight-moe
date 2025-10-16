from config import TrainingConfig, ModelConfig
from moe import GPTMoE
from pathlib import Path
from datetime import datetime
from log import logger
from torch.utils.data import DataLoader
from data import ShardedDataset
from hf_utils import upload_to_hf
from adamw import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch, tiktoken, time, math, wandb, util, json, os
import torch.distributed as dist


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size= int(os.environ.get("WORLD_SIZE", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

 
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

if "RANK" in os.environ:
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"
    is_distributed = True
else:
    rank, local_rank, world_size = 0, 0, 1
    device = util.get_device()
    is_distributed = False


util.set_seed(2406 + rank)

autocast_dtype = util.get_autocast_dtype(device)
if is_main_process():
    logger.info(f"training on {world_size} GPUs...")

if device.startswith("cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def get_dataloaders(train_dir: Path, test_dir: Path, training_cfg: TrainingConfig):
    train_ds = ShardedDataset(config=training_cfg, shards_dir=train_dir)
    test_ds = ShardedDataset(config=training_cfg, shards_dir=test_dir)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed else None

    train_dl = DataLoader(
        train_ds, 
        batch_size=training_cfg.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=training_cfg.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return train_dl, test_dl, train_sampler, test_sampler


tokenizer = tiktoken.get_encoding("gpt2")
training_cfg, model_cfg = TrainingConfig(device=device), ModelConfig()

if is_distributed:
    training_cfg.batch_size = training_cfg.batch_size // world_size

train_dl, test_dl, train_sampler, test_sampler = get_dataloaders(
    Path("pretrain_data/train"), 
    Path("pretrain_data/validation"), 
    training_cfg
)

TOTAL_TRAIN_TOKENS = 3_000_000_000
TOKENS_PER_STEP = world_size * training_cfg.accumulation_steps * training_cfg.batch_size * training_cfg.ctx_size

total_steps = math.ceil(TOTAL_TRAIN_TOKENS / TOKENS_PER_STEP)
warmup_steps = int(0.05 * total_steps)

if is_main_process():
    wandb.init(
        project=f"gpt-moe-pretraining-{datetime.now().strftime('%d_%m_%Y')}",
        config={
            "model_cfg": model_cfg.model_dump(),
            "training_cfg": training_cfg.model_dump(),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "world_size": world_size
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
    loss_accum, num_batches = 0, 0
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        bsz, seq_len = x.shape
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                logits: torch.Tensor = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(bsz*seq_len, -1), y.view(-1,))
        else:
            logits: torch.Tensor = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(bsz*seq_len, -1), y.view(-1,))
        
        loss_accum += loss.detach()
        num_batches += 1
    
    if is_distributed:
        loss_tensor = loss_accum.clone()
        batch_tensor = torch.tensor(num_batches, dtype=torch.float32, device=device)

        dist.all_reduce(loss_tensor, dist.ReduceOp.SUM)
        dist.all_reduce(batch_tensor, dist.ReduceOp.SUM)

        loss_accum = loss_tensor
        num_batches = int(batch_tensor.item())


    model.train()
    return loss_accum / num_batches


model = GPTMoE(model_cfg).to(device)
if is_main_process():
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters())}")
model = torch.compile(model)

if is_distributed:
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True
    )
    if is_main_process():
        logger.info(f"wrapped model with DDP...")

raw_model = model.module if hasattr(model, 'module') else model
nondecay_group = ['bias', 'norm']
param_groups = {
    "decay_params": [p for name, p in raw_model.named_parameters() if not any(_ in name for _ in nondecay_group)],
    "non_decay_params": [p for name, p in raw_model.named_parameters() if any(_ in name for _ in nondecay_group)]
}
optimizer = AdamW(
    params=[
        {"params": param_groups["decay_params"], "weight_decay": training_cfg.weight_decay},
        {"params": param_groups["non_decay_params"], "weight_decay": 0}
    ]
)

CKPT_DIR = Path("checkpoints/pretrain")
if is_main_process():
    CKPT_DIR.mkdir(exist_ok=True)

best_test = float("inf")

def save_ckpt(tag: str, step: int, val_loss: float=None):
    
    if not is_main_process():
        return None

    state = {
        "model": raw_model.state_dict(),
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


train_dl_iter = iter(train_dl)
for step in range(total_steps):

    if train_sampler is not None:
        train_sampler.set_epoch(step)

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum, moe_aux_loss_accum = 0.0, 0.0
    f_accum, p_accum, drop_accum = torch.zeros(model_cfg.n_experts, device="cpu"), torch.zeros(model_cfg.n_experts, device="cpu"), 0.0
    counted = 0
    for _ in range(training_cfg.accumulation_steps):
        try:
            x, y = next(train_dl_iter)
        except StopIteration:
            train_dl_iter = iter(train_dl)
            x, y = next(train_dl_iter)

        x, y = x.to(device), y.to(device)
        bsz, seq_len = x.shape
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits: torch.Tensor = model(x)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(bsz * seq_len, -1), y.view(-1,)
                )
        else:
            logits: torch.Tensor = model(x)
            ce = torch.nn.functional.cross_entropy(
                logits.view(bsz * seq_len, -1), y.view(-1,)
            ) 
        
        loss = ce + raw_model.moe_aux_loss()
        loss /= training_cfg.accumulation_steps
        loss.backward()
        loss_accum += loss.detach()
        moe_aux_loss_accum += raw_model.moe_aux_loss().detach() / training_cfg.accumulation_steps
    
        stats = raw_model.router_stats()
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

    throughput = (world_size * training_cfg.batch_size * training_cfg.ctx_size * training_cfg.accumulation_steps) / (time.time() - t0)

    if is_distributed:
        loss_tensor = loss_accum.clone()
        moe_aux_loss_tensor = moe_aux_loss_accum.clone()

        dist.all_reduce(loss_tensor, dist.ReduceOp.AVG)
        dist.all_reduce(moe_aux_loss_tensor, dist.ReduceOp.AVG)

        loss_accum = loss_tensor
        moe_aux_loss_accum = moe_aux_loss_tensor

    if is_main_process():
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
            if is_main_process():
                logger.info(f"step: {step + 1:>5} | no router stats")
        else:
            if is_distributed:
                f_tensor = f_accum.to(device)
                p_tensor = p_accum.to(device)
                drop_tensor = torch.tensor(drop_accum, device=device)
                count_tensor = torch.tensor(counted, device=device)

                dist.all_reduce(f_tensor, dist.ReduceOp.SUM)
                dist.all_reduce(p_tensor, dist.ReduceOp.SUM)
                dist.all_reduce(drop_tensor, dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, dist.ReduceOp.SUM)

                f_avg = (f_tensor / count_tensor).cpu()
                p_avg= (p_tensor / count_tensor).cpu()
                drop_avg = (drop_tensor / count_tensor).item()
            else:
                f_avg = (f_accum / counted)
                p_avg = (p_accum / counted)
                drop_avg = drop_accum / counted

            if is_main_process():
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

        if is_main_process():
            logger.info(f"step: {step + 1:>5} | val_loss: {val:.4f}")
            wandb.log({
                "test/loss": test_loss.item()
            })

            save_ckpt("last", step + 1, val)

            if val < best_test:
                best_test = val
                save_ckpt("best", step + 1, best_test)
                logger.info(f"✅ new best @ step {step+1}: {best_test:.4f}")

    if (step + 1) % 200 == 0 and is_main_process():
        raw_model.eval()
        start = "There was a"
        with torch.no_grad():
            gen = raw_model.generate(torch.tensor(tokenizer.encode(start), device=device))
        gen_text = tokenizer.decode(gen)
        logger.info(f"step: {step + 1:>5} | generation: {gen_text}")
        wandb.log({
            "generation": wandb.Html(f"<p><b>Prompt:</d> {start}<br><b>Generated:</b> {gen_text}</p>"),
            "step": step + 1
        })
        raw_model.train()

if is_main_process():
    upload_to_hf(
        CKPT_DIR / "best.pt",
        "SkAndMl/moonlight-moe-pretrain"
    )

cleanup()