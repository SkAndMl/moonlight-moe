from collections import deque
from dotenv import load_dotenv, find_dotenv
from hf_utils import stream_hf_dataset
from typing import Literal
import tiktoken, pathlib, hashlib, numpy as np

load_dotenv(find_dotenv())
tokenizer = tiktoken.get_encoding("gpt2")
EOT_ID = tokenizer.eot_token
CTX_LENGTH = 1024
TARGET_TOKENS = {
    "train": 3_000_000_000,
    "validation": int(3_000_000_000 * 0.05) # 5% of train tokens
}
NUM_TOKENS_PER_SHARD = 50_000_000
NUM_SEQUENCES_PER_SHARD = NUM_TOKENS_PER_SHARD // CTX_LENGTH
SPLIT_THRESHOLDS = {
    "train": TARGET_TOKENS["train"] / sum(TARGET_TOKENS.values()),
    "validation": 1 - TARGET_TOKENS["train"] / sum(TARGET_TOKENS.values())
}
SPLIT_SALT = "moonlight"
HF_ID = "HuggingFaceFW/fineweb-edu"
HF_NAME = "sample-10BT"

def assign_split(doc_key: str) -> Literal["train", "validation"]:
    h = hashlib.sha256((doc_key + str(SPLIT_SALT)).encode()).digest()
    u = int.from_bytes(h[:8], 'big') / 2**64
    return "train" if u < SPLIT_THRESHOLDS["train"] else "validation"

def create_shards():
    ds_stream = stream_hf_dataset(HF_ID, HF_NAME, "train")
    
    # Create directories for both splits
    shard_dirs = {
        "train": pathlib.Path("pretrain_data/train"),
        "validation": pathlib.Path("pretrain_data/validation")
    }
    for dir_path in shard_dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize state for both splits
    shard_ids = {"train": 0, "validation": 0}
    tokens_consumed = {"train": 0, "validation": 0}
    buffers = {"train": deque(), "validation": deque()}
    
    for row in ds_stream:
        try:
            split = assign_split(row["id"])
            
            # Check if this split has already reached its target
            if tokens_consumed[split] >= TARGET_TOKENS[split]:
                # Check if both splits are done
                if all(tokens_consumed[s] >= TARGET_TOKENS[s] for s in ["train", "validation"]):
                    break
                continue
            
            text = row["text"]
            tokens = tokenizer.encode(text, allowed_special="all")
            buffers[split].extend(tokens)
            buffers[split].append(EOT_ID)
            
            if len(buffers[split]) // CTX_LENGTH >= NUM_SEQUENCES_PER_SHARD:
                take = NUM_SEQUENCES_PER_SHARD * CTX_LENGTH
                block = np.fromiter((buffers[split].popleft() for _ in range(take)), count=take, dtype=np.uint16)
                
                with open(shard_dirs[split] / f"shard_{shard_ids[split]:05d}.bin", "wb") as f:
                    block.tofile(f)
                
                shard_ids[split] += 1
                tokens_consumed[split] += NUM_SEQUENCES_PER_SHARD * CTX_LENGTH
        except:
            continue
    
    for split in ["train", "validation"]:
        print(f"{split}: wrote {shard_ids[split]} shards to {shard_dirs[split]}")

create_shards()