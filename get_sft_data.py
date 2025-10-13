from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import tiktoken
import pathlib
import numpy as np
import random
import os

load_dotenv(find_dotenv())

tokenizer = tiktoken.get_encoding("gpt2")
EOT_ID = tokenizer.eot_token  # 50256

# special tokens for chat format
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# These will be added to the model's vocabulary
SYSTEM_ID = 50257
USER_ID = 50258
ASSISTANT_ID = 50259

CTX_LENGTH = 1024
PAD_ID = EOT_ID

NUM_SAMPLES_PER_SHARD = 10_000
TARGET_SAMPLES = {
    "train": 250_000,   # Down from 1M
    "val": 12_500       # Down from 50K
}


SFT_DATASETS = {
    "euclaise/writingprompts": {
        "splits": ["train", "validation"],
        "weight": {"train": 0.50, "val": 0.50},
        "prompt_column": "prompt",
        "story_column": "story",
        "has_prompts": True
    },
    "roneneldan/TinyStories": {
        "splits": ["train", "validation"],
        "weight": {"train": 0.30, "val": 0.30},
        "story_column": "text",
        "has_prompts": False
    },
    "mintujupally/ROCStories": {
        "splits": ["train", "test"],
        "weight": {"train": 0.20, "val": 0.20},
        "story_column": "text",
        "has_prompts": False
    }
}

SYSTEM_PROMPTS = [
    "You are a creative storyteller who writes engaging stories.",
    "You are an AI assistant specialized in creative writing and storytelling.",
    "You are a helpful assistant that creates imaginative stories.",
]

PROMPT_TEMPLATES = [
    "Write a short story.",
    "Tell me a story.",
    "Create an interesting story.",
    "Write a creative story.",
    "Generate a story for me.",
    "Write a story about adventure.",
    "Tell me an engaging tale.",
    "Create a short narrative.",
]


def encode_with_special_tokens(text, token_type="text"):
    tokens = tokenizer.encode(text)
    
    if token_type == "system":
        return [SYSTEM_ID] + tokens
    elif token_type == "user":
        return [USER_ID] + tokens
    elif token_type == "assistant":
        return [ASSISTANT_ID] + tokens
    return tokens


def create_sft_sample(system_prompt, user_prompt, assistant_response):

    system_tokens = encode_with_special_tokens(system_prompt, "system")
    user_tokens = encode_with_special_tokens(user_prompt, "user")
    assistant_tokens = encode_with_special_tokens(assistant_response, "assistant")
    prompt_len = len(system_tokens) + len(user_tokens)
    tokens = system_tokens + user_tokens + assistant_tokens
    tokens.append(EOT_ID)
    content_len = len(tokens)
    
    return tokens, prompt_len, content_len


def generate_prompt_for_story(story_text):
    return random.choice(PROMPT_TEMPLATES)


def stream_hf_dataset(hf_id: str, split: str):
    ds = load_dataset(
        hf_id, 
        split=split, 
        streaming=True, 
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    while True:
        for row in ds:
            yield row


def get_split(hf_id: str, split: str):
    if split == "train":
        return "train"
    elif split == "val":
        if "validation" in SFT_DATASETS[hf_id]["splits"]:
            return "validation"
        elif "test" in SFT_DATASETS[hf_id]["splits"]:
            return "test"
    return None


def process_sample(row, dataset_config, dataset_name):
    story = row[dataset_config["story_column"]]
    
    if dataset_config.get("has_prompts", False):
        user_prompt = row[dataset_config["prompt_column"]]
        if dataset_name == "euclaise/writingprompts":
            user_prompt = user_prompt.strip()
            if user_prompt.startswith("[") and "]" in user_prompt:
                user_prompt = user_prompt[user_prompt.index("]")+1:].strip()
    else:
        user_prompt = generate_prompt_for_story(story)
    
    system_prompt = random.choice(SYSTEM_PROMPTS)
    
    tokens, prompt_len, content_len = create_sft_sample(
        system_prompt, 
        user_prompt, 
        story
    )
    
    return tokens, prompt_len, content_len


dataset_names = list(SFT_DATASETS.keys())

for split in ["train", "val"]:
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}\n")
    
    ds_samples_consumed = {k: 0 for k in dataset_names}
    ds_streams = {}
    
    for dataset_name in dataset_names:
        _split = get_split(dataset_name, split)
        if _split is None:
            continue
        ds_streams[dataset_name] = stream_hf_dataset(dataset_name, _split)
    
    if not ds_streams:
        print(f"No datasets available for {split} split. Skipping...")
        continue
    
    shard_id = 0
    samples_buffer = []
    
    out_dir = pathlib.Path(f"sft_data/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = sum(ds_samples_consumed.values())
    target = TARGET_SAMPLES[split]
    
    while total_samples < target:
        available_datasets = [
            name for name in ds_streams.keys()
            if ds_samples_consumed[name] < int(SFT_DATASETS[name]["weight"][split] * target)
        ]
        
        if not available_datasets:
            break
        
        dataset_name = random.choice(available_datasets)
        
        try:
            row = next(ds_streams[dataset_name])
            tokens, prompt_len, content_len = process_sample(
                row, 
                SFT_DATASETS[dataset_name], 
                dataset_name
            )
            
            if len(tokens) > CTX_LENGTH + 1:
                continue
            
            # Pad to CTX_LENGTH + 1
            if len(tokens) < CTX_LENGTH + 1:
                tokens = tokens + [PAD_ID] * (CTX_LENGTH + 1 - len(tokens))
            else:
                tokens = tokens[:CTX_LENGTH + 1]
                # Adjust content_len if we truncated
                content_len = min(content_len, CTX_LENGTH + 1)
            
            # Create row: [tokens (1025), prompt_len (1), content_len (1)]
            sample_row = tokens + [prompt_len, content_len]
            samples_buffer.append(sample_row)
            
            ds_samples_consumed[dataset_name] += 1
            total_samples += 1
            
            if len(samples_buffer) >= NUM_SAMPLES_PER_SHARD:
                shard_array = np.array(samples_buffer, dtype=np.uint16)
                shard_path = out_dir / f"shard_{shard_id:05d}.bin"
                with open(shard_path, "wb") as f:
                    shard_array.tofile(f)
                
                print(f"[{split}] Saved shard {shard_id} | "
                        f"Samples: {total_samples:,}/{target:,} | "
                        f"Shape: {shard_array.shape}")
                
                shard_id += 1
                samples_buffer = []
                
        except StopIteration:
            print(f"Dataset {dataset_name} exhausted")
            continue
        except Exception as e:
            print(f"Error processing sample from {dataset_name}: {e}")
            continue
    
    if samples_buffer:
        shard_array = np.array(samples_buffer, dtype=np.uint16)
        shard_path = out_dir / f"shard_{shard_id:05d}.bin"
        with open(shard_path, "wb") as f:
            shard_array.tofile(f)
        
        print(f"[{split}] Saved final shard {shard_id} | "
                f"Samples: {total_samples:,}/{target:,} | "
                f"Shape: {shard_array.shape}")
    
    print(f"\n[{split}] Completed! Total samples: {total_samples:,}")
    print(f"Dataset distribution:")
    for name, count in ds_samples_consumed.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {name}: {count:,} ({percentage:.1f}%)")