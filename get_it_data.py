from hf_utils import stream_hf_dataset
from pathlib import Path
from typing import Literal
from collections import deque
import hashlib, random, tiktoken, numpy as np

tokenizer = tiktoken.get_encoding("gpt2")
EOT_TOKEN_ID = tokenizer.eot_token
SYSTEM_TOKEN_ID = 50257
USER_TOKEN_ID = 50258
ASSISTANT_TOKEN_ID = 50259

CTX_SIZE = 1024
NUM_SAMPLES_PER_SHARD = 10000

HF_ID = "HuggingFaceTB/smoltalk"
SPLIT_SALT = "moonlight"

IT_TARGET_TOKENS = {
    "train": 20_000_000,
    "test": int(20_000_000 * 0.05)
}

IT_SPLIT_THRESHOLDS = {
    "train": IT_TARGET_TOKENS["train"] / sum(IT_TARGET_TOKENS.values()),
    "test": 1 - (IT_TARGET_TOKENS["train"] / sum(IT_TARGET_TOKENS.values()))
}

IT_SUBSET_SPLIT = {
    "smol-summarize": 0.05,
    "smol-constraints": 0.05,
    "self-oss-instruct": 0.3,
    "metamathqa-50k": 0.3,
    "openhermes-100k": 0.3
}


SFT_DIR = Path("sft_data/it")
SFT_DIR.mkdir(exist_ok=True, parents=True)
(SFT_DIR / "train").mkdir(exist_ok=True)
(SFT_DIR / "test").mkdir(exist_ok=True)

# create IT data
it_streams = {
    subset: stream_hf_dataset(HF_ID, subset, "train") for subset in IT_SUBSET_SPLIT
}

it_tokens_consumed = {
    split: {
        subset: 0 for subset in IT_SUBSET_SPLIT
    } for split in ["train", "test"]
}

it_buffers = {"train": deque(), "test": deque()}

def assign_split(doc_key: str) -> Literal["train", "test"]:
    h = hashlib.sha256((doc_key + str(SPLIT_SALT)).encode()).digest()
    u = int.from_bytes(h[:8], 'big') / 2**64
    return "train" if u < IT_SPLIT_THRESHOLDS["train"] else "test"


def format(messages: list[dict], subset_name=None) -> list[int]:
    if len(messages) not in {2, 3}:
        return None
    system_content, user_content, assistant_content = "", "", ""
    for row in messages:
        match row["role"]:
            case "system":
                system_content += row["content"].strip()
            case "user":
                user_content += row["content"].strip()
            case "assistant":
                assistant_content += row["content"].strip()

    SYS_DEFAULT = (
        "Be correct and concise. Prefer bullet points. "
        "Say \"I don't know\" if unsure. Follow instructions precisely."
    )
    SYS_BY_SUBSET = {
        "smol-constraints": (
            "Be correct and concise. When a JSON schema is implied, output ONLY the JSON—no extra text."
        ),
        "smol-summarize": (
            "Summarize briefly. Aim for 3–5 bullet points or 1–3 sentences. No repetition."
        ),
        "metamathqa-50k": (
            "Solve briefly. If steps are needed, keep them ≤3. Return ONLY the final answer."
        ),
        "self-oss-instruct": (
            "Follow the instruction precisely. If code is requested, provide a minimal, runnable snippet with no extra commentary."
        ),
        "openhermes-100k": SYS_DEFAULT,
    }

    if len(system_content) == 0:
        system_content += SYS_BY_SUBSET.get(subset_name or "", SYS_DEFAULT)

    x = (
        [SYSTEM_TOKEN_ID] + 
        tokenizer.encode(system_content) + 
        [USER_TOKEN_ID] + 
        tokenizer.encode(user_content) + 
        [ASSISTANT_TOKEN_ID]
    )
    y = tokenizer.encode(assistant_content) + [EOT_TOKEN_ID]

    tokens = x + y
    if len(tokens) > CTX_SIZE + 1:
        return None

    if len(tokens) < CTX_SIZE + 1:
        tokens += [EOT_TOKEN_ID] * (CTX_SIZE + 1 - len(tokens))

    return tokens[:CTX_SIZE + 1], len(x + y)


shard_id = {"train": 0, "test": 0}

while True:

    if sum(it_tokens_consumed["train"].values()) >= IT_TARGET_TOKENS["train"] and sum(it_tokens_consumed["test"].values()) >= IT_TARGET_TOKENS["test"]:
        break

    subset = random.choices(list(IT_SUBSET_SPLIT.keys()), weights=IT_SUBSET_SPLIT.values(), k=1)[0]
    row = next(it_streams[subset])
    split = assign_split(str(row))
    if it_tokens_consumed[split][subset] >= int(IT_SUBSET_SPLIT[subset] * IT_TARGET_TOKENS[split]):
        continue

    formatted_content = format(row["messages"], subset)
    if formatted_content is None:
        continue

    tokens, content_len = formatted_content
    it_buffers[split].extend(tokens)
    it_tokens_consumed[split][subset] += content_len

    if len(it_buffers[split]) // (CTX_SIZE + 1) >= NUM_SAMPLES_PER_SHARD:
        take = NUM_SAMPLES_PER_SHARD * (CTX_SIZE +1)
        block = np.fromiter((it_buffers[split].popleft() for _ in range(take)), dtype=np.uint16, count=take)
        with open(SFT_DIR / split / f"shard_{shard_id[split]:05d}.bin", "wb") as f:
            block.tofile(f)

        shard_id[split] += 1

for split in it_buffers:
    if len(it_buffers[split]) > 0:
        take = len(it_buffers[split])
        block = np.fromiter((it_buffers[split].popleft() for _ in range(take)), dtype=np.uint16, count=take)
        with open(SFT_DIR / split / f"shard_{shard_id[split]:05d}.bin", "wb") as f:
            block.tofile(f)
        
        shard_id[split] += 1