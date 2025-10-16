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


def create_split_shards(split: Literal["train", "validation"]):
    ds_stream = stream_hf_dataset(HF_ID, HF_NAME, "train")
    SHARD_DIR = pathlib.Path(f"pretrain/{split}")
    SHARD_DIR.mkdir(exist_ok=True, parents=True)
    shard_id = 0
    tokens_consumed = 0
    buf = deque()

    for row in ds_stream:
        if assign_split(row["id"]) != split:
            continue

        text = row["text"]
        tokens = tokenizer.encode(text)
        buf.extend(tokens)
        buf.append(EOT_ID)

        if len(buf) // CTX_LENGTH >= NUM_SEQUENCES_PER_SHARD:
            take = NUM_SEQUENCES_PER_SHARD * CTX_LENGTH
            block = np.fromiter((buf.popleft() for _ in range(take)), count=take, dtype=np.uint16)
            
            with open(SHARD_DIR / f"shard_{shard_id:05d}.bin", "wb") as f:
                block.tofile(f)
            
            shard_id += 1
            tokens_consumed += NUM_SEQUENCES_PER_SHARD * CTX_LENGTH
        
        if tokens_consumed >= TARGET_TOKENS[split]:
            break
    
    print(f"wrote {shard_id} shards to {SHARD_DIR}")

create_split_shards("train")
create_split_shards("validation")