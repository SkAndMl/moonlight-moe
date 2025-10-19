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

CHAT_TARGET_TOKENS = {
    "train": 30_000_000,
    "test": int(30_000_000 * 0.05)
}

CHAT_SPLIT_THRESHOLDS = {
    "train": CHAT_TARGET_TOKENS["train"] / sum(CHAT_TARGET_TOKENS.values()),
    "test": 1 - (CHAT_TARGET_TOKENS["train"] / sum(CHAT_TARGET_TOKENS.values()))
}

CHAT_SUBSET_SPLIT = {
    "everyday-conversations": 0.5,
    "systemchats-30k": 0.5
}


SFT_DIR = Path("sft_data/chat")
SFT_DIR.mkdir(exist_ok=True, parents=True)
(SFT_DIR / "train").mkdir(exist_ok=True)
(SFT_DIR / "test").mkdir(exist_ok=True)

# create IT data
chat_streams = {
    subset: stream_hf_dataset(HF_ID, subset, "train") for subset in CHAT_SUBSET_SPLIT
}

chat_tokens_consumed = {
    split: {
        subset: 0 for subset in CHAT_SUBSET_SPLIT
    } for split in ["train", "test"]
}

chat_buffers = {"train": deque(), "test": deque()}

def assign_split(doc_key: str) -> Literal["train", "test"]:
    h = hashlib.sha256((doc_key + str(SPLIT_SALT)).encode()).digest()
    u = int.from_bytes(h[:8], 'big') / 2**64
    return "train" if u < CHAT_SPLIT_THRESHOLDS["train"] else "test"


def format(messages: list[dict]) -> tuple[list[list[int]], int] | None:
    system_content, user_contents, assistant_contents = "", [], []
    for row in messages:
        match row["role"]:
            case "system":
                system_content += row["content"].strip()
            case "user":
                user_contents.append(row["content"].strip())
            case "assistant":
                assistant_contents.append(row["content"].strip())

    if len(user_contents) == 0 or len(assistant_contents) == 0:
        return None

    if len(user_contents) != len(assistant_contents):
        return None

    SYS_UNIVERSAL = (
        "You are a helpful, accurate, and concise AI assistant. "
        "Follow instructions precisely. "
        "If you're unsure, say so."
    )

    if len(system_content) == 0:
        system_content += SYS_UNIVERSAL

    final_content_len = 0
    chat_turns = []
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        if len(chat_turns) > 0:
            prev_turn = chat_turns[-1][:-1] # ignoring eot token
        else:
            prev_turn = [SYSTEM_TOKEN_ID] + tokenizer.encode(system_content)

        x = (
            prev_turn + 
            [USER_TOKEN_ID] + 
            tokenizer.encode(user_content) + 
            [ASSISTANT_TOKEN_ID]
        )
        y = tokenizer.encode(assistant_content) + [EOT_TOKEN_ID]

        tokens = x + y
        if len(tokens) > CTX_SIZE + 1:
            break

        chat_turns.append(tokens)
        final_content_len = len(tokens)
    
    if len(chat_turns) == 0:
        return None

    for i in range(len(chat_turns)):
        if len(chat_turns[i]) < CTX_SIZE + 1:
            chat_turns[i] += [EOT_TOKEN_ID] * (CTX_SIZE + 1 - len(chat_turns[i]))

    return chat_turns, final_content_len


shard_id = {"train": 0, "test": 0}

while True:

    if sum(chat_tokens_consumed["train"].values()) >= CHAT_TARGET_TOKENS["train"] and sum(chat_tokens_consumed["test"].values()) >= CHAT_TARGET_TOKENS["test"]:
        break

    subset = random.choices(list(CHAT_SUBSET_SPLIT.keys()), weights=CHAT_SUBSET_SPLIT.values(), k=1)[0]
    row = next(chat_streams[subset])
    split = assign_split(str(row))
    if chat_tokens_consumed[split][subset] >= int(CHAT_SUBSET_SPLIT[subset] * CHAT_TARGET_TOKENS[split]):
        continue

    formatted_content = format(row["messages"])
    if formatted_content is None:
        continue

    chat_turns, content_len = formatted_content
    for turn in chat_turns:
        chat_buffers[split].extend(turn)
    chat_tokens_consumed[split][subset] += content_len

    if len(chat_buffers[split]) // (CTX_SIZE + 1) >= NUM_SAMPLES_PER_SHARD:
        take = NUM_SAMPLES_PER_SHARD * (CTX_SIZE +1)
        block = np.fromiter((chat_buffers[split].popleft() for _ in range(take)), dtype=np.uint16, count=take)
        with open(SFT_DIR / split / f"shard_{shard_id[split]:05d}.bin", "wb") as f:
            block.tofile(f)

        shard_id[split] += 1

for split in chat_buffers:
    if len(chat_buffers[split]) > 0:
        take = len(chat_buffers[split])
        block = np.fromiter((chat_buffers[split].popleft() for _ in range(take)), dtype=np.uint16, count=take)
        with open(SFT_DIR / split / f"shard_{shard_id[split]:05d}.bin", "wb") as f:
            block.tofile(f)
        
        shard_id[split] += 1