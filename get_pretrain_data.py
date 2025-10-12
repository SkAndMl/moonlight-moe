from datasets import load_dataset
from collections import deque
from dotenv import load_dotenv, find_dotenv
import tiktoken, pathlib, numpy as np, random, os

load_dotenv(find_dotenv())

tokenizer = tiktoken.get_encoding("gpt2")

EOT_ID = tokenizer.eot_token
CTX_LENGTH = 1024
TARGET_TOKENS = {
    "train": 1_000_000_000,
    "validation": 50_000_000
}
NUM_TOKENS_PER_SHARD = 25_000_000
NUM_SEQUENCES_PER_SHARD = NUM_TOKENS_PER_SHARD // CTX_LENGTH

HF_DATASETS = {
    "roneneldan/TinyStories": {
        "splits": ["train", "validation"],
        "weight": {
            "train": 0.30,
            "validation": 0.45
        },
        "target_column": "text"
    },
    "euclaise/writingprompts": {
        "splits" : ["train", "validation"],
        "weight": {
            "train": 0.30,
            "validation": 0.45
        },
        "target_column": "story"
    },
    "ajibawa-2023/Children-Stories-Collection": {
        "splits": ["train"],
        "weight": {
            "train": 0.30,
            "validation": 0.0
        },
        "target_column": "text"
    },
    "mintujupally/ROCStories": {
        "splits": ["train", "test"],
        "weight": {
            "train": 0.30,
            "validation": 0.10
        },
        "target_column": "text"
    }
}

def stream_hf_dataset(hf_id: str, split: str):
    ds = load_dataset(hf_id, split=split, streaming=True, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    while True: # makes the data infinite, by repeatedly iterating
        for row in ds:
            yield row
        
def get_split(hf_id: str, split: str):
    if split == "train":
        return split
    elif split == "validation":
        if "validation" in HF_DATASETS[hf_id]["splits"]:
            return "validation"
        elif "test" in HF_DATASETS[hf_id]["splits"]:
            return "test"
    return None


HF_DATASET_NAMES = list(HF_DATASETS.keys())

for split in {"train", "validation"}:

    DS_TOKENS_CONSUMED = {k: 0 for k in HF_DATASET_NAMES}
    DS_STREAMS = {}
    for k in HF_DATASET_NAMES:
        _split = get_split(k, split)
        if _split is None:
            continue
        DS_STREAMS[k] = stream_hf_dataset(k, _split)

    shard_id = 0
    buf = deque()
    out_dir = pathlib.Path(f"pretrain_data/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    while sum(DS_TOKENS_CONSUMED.values()) < TARGET_TOKENS[split]:

        ds_name = random.choice(list(DS_STREAMS.keys()))
        if DS_TOKENS_CONSUMED[ds_name] >= int(HF_DATASETS[ds_name]["weight"][split] * TARGET_TOKENS[split]):
            continue

        row = next(DS_STREAMS[ds_name])
        text = row[HF_DATASETS[ds_name]["target_column"]]
        tokens = tokenizer.encode(text)
        if len(tokens) > CTX_LENGTH:
            continue
        
        DS_TOKENS_CONSUMED[ds_name] += len(tokens)
        buf.extend(tokens)
        buf.append(EOT_ID)

        if len(buf) // CTX_LENGTH >= NUM_SEQUENCES_PER_SHARD:
            take = NUM_SEQUENCES_PER_SHARD * CTX_LENGTH
            block = np.fromiter((buf.popleft() for _ in range(take)), dtype=np.uint16, count=take)

            with open(out_dir / f"shard_{shard_id:05d}.bin", "wb") as f:
                block.tofile(f)
            
            shard_id += 1
        
    if shard_id % 5 == 0:
        print(f"[{split}] Shard {shard_id} | Tokens: {sum(DS_TOKENS_CONSUMED.values()):,}/{TARGET_TOKENS[split]:,}")