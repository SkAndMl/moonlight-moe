import argparse
import json
import random
import numpy as np
import tiktoken
from pathlib import Path
from config import SFTConfig
from data_processors import (
    TinyStoriesProcessor,
    ROCStoriesProcessor,
    StoryClozeProcessor,
    WritingPromptsProcessor,
)
from sampler import sample_all_tasks
import prompts
import util


# Sharding configuration
SAMPLES_PER_SHARD = 10000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data from multiple datasets with train/val split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--total_samples",
        type=int,
        required=True,
        help="Total number of samples to generate (train + val combined)"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--token_limit",
        type=int,
        default=513,
        help="Maximum token count per sample (513 allows for 512-token sequences after x/y shift)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2406,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="sft_data",
        help="Prefix for output filenames"
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        default="json",
        choices=["json", "shards", "both"],
        help="Output format: 'json' (dict format), 'shards' (binary shards), or 'both'"
    )
    
    return parser.parse_args()


def print_config_summary(config: SFTConfig):
    print("=" * 70)
    print("SFT DATA PREPARATION")
    print("=" * 70)
    print(f"Total samples: {config.total_samples:,}")
    print(f"Token limit: {config.token_limit}")
    print(f"Validation split: {config.val_split:.1%}")
    print(f"Random seed: {config.random_seed}")
    print()
    
    train_count, val_count = config.get_train_val_counts()
    print(f"Train samples: {train_count:,}")
    print(f"Val samples: {val_count:,}")
    print()
    
    print("Task distribution:")
    targets = config.get_target_counts()
    for task, count in targets.items():
        pct = config.task_distribution[task]
        print(f"  {task:20s}: {count:6,} samples ({pct:.1%})")
    print()


def print_final_summary(train_data: dict, val_data: dict, all_stats: dict):
    print("=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    
    train_total = len(train_data["user_content"])
    val_total = len(val_data["user_content"])
    
    print(f"Total train samples: {train_total:,}")
    print(f"Total val samples: {val_total:,}")
    print(f"Total samples: {train_total + val_total:,}")
    print()
    
    print("Detailed statistics by task:")
    for task_type, processors_stats in all_stats.items():
        print(f"\n  {task_type}:")
        for processor_name, stats in processors_stats.items():
            print(f"    {processor_name}:")
            print(f"      Target: {stats['target_count']:,}")
            print(f"      Actual: {stats['actual_count']:,}")
            print(f"      Train: {stats['train_count']:,}")
            print(f"      Val: {stats['val_count']:,}")
            print(f"      Tokens rejected: {stats['tokens_rejected']:,}")
            print(f"      Parse failed: {stats['parse_failed']:,}")
            if stats['dataset_exhausted']:
                print(f"      ⚠️  WARNING: Dataset exhausted!")
    
    print()


def save_json_data(data: dict, filepath: Path):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to: {filepath}")


def process_sample_for_sharding(
    user_content,
    assistant_content,
    task_type: str,
    tokenizer,
    token_limit: int = 513
) -> tuple:
    
    user_content_formatted = prompts.format_user_content(task_type, user_content)
    
    formatted = util.format_data_for_sft(
        prompts.SYSTEM_PROMPT,
        user_content_formatted,
        assistant_content
    )
    
    tokens = tokenizer.encode(formatted, allowed_special="all")
    content_len = len(tokens)
    
    x_part, _ = util.get_x_y_for_sft(
        prompts.SYSTEM_PROMPT,
        user_content_formatted,
        assistant_content
    )
    prompt_tokens = tokenizer.encode(x_part, allowed_special="all")
    prompt_len = len(prompt_tokens)
    
    if content_len > token_limit:
        raise ValueError(f"Sample exceeds token limit: {content_len} > {token_limit}")
    
    if prompt_len >= content_len:
        raise ValueError(f"Invalid lengths: prompt_len={prompt_len} >= content_len={content_len}")
    
    if content_len < token_limit:
        padding = [tokenizer.eot_token] * (token_limit - content_len)
        tokens.extend(padding)
    
    tokens_array = np.array(tokens, dtype=np.uint16)
    
    return tokens_array, prompt_len, content_len


def write_shards(data: dict, output_dir: Path, split_name: str, tokenizer):
    print(f"\nWriting {split_name} shards...")
    
    shards_dir = output_dir / "shards" / split_name
    shards_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(data["user_content"])
    shard_buffer = []
    shard_id = 0
    
    for i in range(num_samples):
        user_content = data["user_content"][i]
        assistant_content = data["assistant_content"][i]
        task_type = data["task"][i]
        
        try:
            tokens_array, prompt_len, content_len = process_sample_for_sharding(
                user_content,
                assistant_content,
                task_type,
                tokenizer
            )
            
            row = np.concatenate([tokens_array, [prompt_len, content_len]]).astype(np.uint16)
            shard_buffer.append(row)
            
            if len(shard_buffer) >= SAMPLES_PER_SHARD:
                shard_array = np.array(shard_buffer, dtype=np.uint16)  # Shape: (N, 515)
                shard_path = shards_dir / f"shard_{shard_id:05d}.bin"
                shard_array.tofile(shard_path)
                print(f"  Wrote {len(shard_buffer)} samples to {shard_path.name}")
                
                shard_buffer = []
                shard_id += 1
        
        except Exception as e:
            print(f"  Warning: Failed to process sample {i} for sharding: {e}")
            continue
    
    if len(shard_buffer) > 0:
        shard_array = np.array(shard_buffer, dtype=np.uint16)
        shard_path = shards_dir / f"shard_{shard_id:05d}.bin"
        shard_array.tofile(shard_path)
        print(f"  Wrote {len(shard_buffer)} samples to {shard_path.name}")
        shard_id += 1
    
    print(f"Finished writing {shard_id} shards to {shards_dir}")
    
    metadata = {
        "split": split_name,
        "total_samples": num_samples,
        "samples_per_shard": SAMPLES_PER_SHARD,
        "num_shards": shard_id,
        "token_limit": 513,
        "row_size": 515,
        "eot_token_id": tokenizer.eot_token,
        "format": "Each row: [513 tokens (uint16), prompt_len (uint16), content_len (uint16)]"
    }
    
    metadata_path = shards_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")


def main():
    args = parse_args()
    
    # Create config
    config = SFTConfig(
        total_samples=args.total_samples,
        token_limit=args.token_limit,
        val_split=args.val_split,
        random_seed=args.random_seed
    )
    
    # Print configuration
    print_config_summary(config)
    
    processors_map = {
        "generate_long": [
            TinyStoriesProcessor()
        ],
        "generate_creative": [
            WritingPromptsProcessor()
        ],
        "summarize": [
            TinyStoriesProcessor()
        ],
        "generate_short": [
            ROCStoriesProcessor()
        ],
        "inference": [
            StoryClozeProcessor(random_seed=config.random_seed)
        ]
    }
    
    print("Initialized processors:")
    for task, processors in processors_map.items():
        processor_names = [p.__class__.__name__ for p in processors]
        print(f"  {task}: {', '.join(processor_names)}")
    print()
    print("=" * 70)
    print()
    
    train_data, val_data, all_stats = sample_all_tasks(config, processors_map)
    
    print_final_summary(train_data, val_data, all_stats)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.output_format in ["json", "both"]:
        print("\nSaving JSON files...")
        train_path = output_dir / f"{args.output_prefix}_train.json"
        val_path = output_dir / f"{args.output_prefix}_val.json"
        
        save_json_data(train_data, train_path)
        save_json_data(val_data, val_path)
        
        stats_path = output_dir / f"{args.output_prefix}_stats.json"
        save_json_data(all_stats, stats_path)
    
    if args.output_format in ["shards", "both"]:
        print("\nCreating binary shards...")
        tokenizer = tiktoken.get_encoding("gpt2")
        
        write_shards(train_data, output_dir, "train", tokenizer)
        write_shards(val_data, output_dir, "val", tokenizer)
    
    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    
    if args.output_format == "json":
        print(f"\nJSON files saved to: {output_dir}")
    elif args.output_format == "shards":
        print(f"\nBinary shards saved to: {output_dir / 'shards'}")
    else:  # both
        print(f"\nJSON files saved to: {output_dir}")
        print(f"Binary shards saved to: {output_dir / 'shards'}")


if __name__ == "__main__":
    main()

# python3.11 prepare_sft_data.py --total_samples 300000 --val_split 0.1 --token_limit 513 --random_seed 2406 --output_dir sft_data --output_prefix sft --output_format both