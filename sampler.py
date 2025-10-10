import random
from typing import Dict, List, Tuple, Any
import tiktoken
from dataclasses import dataclass
import prompts
import util
from data_processors.base import BaseDatasetProcessor, MultiTaskDatasetProcessor


@dataclass
class SamplingResult:
    train_samples: List[Dict[str, Any]]
    val_samples: List[Dict[str, Any]]
    stats: Dict[str, Any]


class StratifiedSampler:
    def __init__(
        self,
        token_limit: int = 513,
        val_split: float = 0.1,
        random_seed: int = 42
    ):
        self.token_limit = token_limit
        self.val_split = val_split
        self.random_seed = random_seed
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        random.seed(random_seed)
    
    def sample_from_processor(
        self,
        processor: BaseDatasetProcessor,
        target_count: int,
        task_type: str = None
    ) -> SamplingResult:
        if not processor.is_loaded():
            processor.load_dataset()
        
        is_multi_task = isinstance(processor, MultiTaskDatasetProcessor)
        
        if is_multi_task and task_type is None:
            raise ValueError("task_type must be specified for MultiTaskDatasetProcessor")
        
        total_size = processor.get_dataset_size()
        indices = list(range(total_size))
        random.shuffle(indices)
        
        samples = []
        tokens_rejected = 0
        parse_failed = 0
        
        for idx in indices:
            if len(samples) >= target_count:
                break
            
            result = processor.extract_sample(idx)
            
            if result is None:
                parse_failed += 1
                continue
            
            # Handle multi-task vs single-task processors
            if is_multi_task:
                user_content, assistant_content, sample_task = result
                # Skip if this sample is for a different task
                if sample_task != task_type:
                    continue
            else:
                user_content, assistant_content = result
                sample_task = processor.get_task_type()
            
            # Validate token count with the correct task-specific prompt
            if not self._validate_tokens(user_content, assistant_content, sample_task):
                tokens_rejected += 1
                continue
            
            # Add to samples
            samples.append({
                "user_content": user_content,
                "assistant_content": assistant_content,
                "task": sample_task
            })
        
        # Split into train and val
        train_samples, val_samples = self._split_train_val(samples)
        
        # Compile statistics
        stats = {
            "target_count": target_count,
            "actual_count": len(samples),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "tokens_rejected": tokens_rejected,
            "parse_failed": parse_failed,
            "indices_checked": min(len(indices), len(samples) + tokens_rejected + parse_failed),
            "dataset_exhausted": len(samples) < target_count
        }
        
        return SamplingResult(
            train_samples=train_samples,
            val_samples=val_samples,
            stats=stats
        )
    
    def _validate_tokens(self, user_content: Any, assistant_content: str, task_type: str) -> bool:
        """
        Validate that the formatted sample is within token limit.
        
        Args:
            user_content: User content (str or dict)
            assistant_content: Assistant response
            task_type: The task type to apply correct prompt
            
        Returns:
            True if within token limit, False otherwise
        """
        # Format user_content with the appropriate prompt for this task
        try:
            user_content_str = prompts.format_user_content(task_type, user_content)
        except (KeyError, ValueError) as e:
            # If formatting fails, reject the sample
            return False
        
        # Format the complete conversation
        formatted = util.format_data_for_sft(
            prompts.SYSTEM_PROMPT,
            user_content_str,
            assistant_content
        )
        
        # Tokenize and check length
        tokens = self.tokenizer.encode(formatted, allowed_special="all")
        return len(tokens) <= self.token_limit
    
    def _split_train_val(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split samples into train and validation sets.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Tuple of (train_samples, val_samples)
        """
        if len(samples) == 0:
            return [], []
        
        # Shuffle samples before splitting
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        # Calculate split point
        val_count = int(len(samples_copy) * self.val_split)
        
        # Split
        val_samples = samples_copy[:val_count]
        train_samples = samples_copy[val_count:]
        
        return train_samples, val_samples


def sample_all_tasks(
    config,
    processors_map: Dict[str, List[BaseDatasetProcessor]]
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, Dict]]:
    """
    Sample from all processors for all tasks.
    
    Args:
        config: SFTConfig object with sampling configuration
        processors_map: Dictionary mapping task types to list of processor instances
        
    Returns:
        Tuple of (train_data_dict, val_data_dict, all_stats_dict)
    """
    sampler = StratifiedSampler(
        token_limit=config.token_limit,
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    
    # Get target counts for each task
    targets = config.get_target_counts()
    
    # Initialize data structures
    train_data = {
        "user_content": [],
        "task": [],
        "assistant_content": []
    }
    val_data = {
        "user_content": [],
        "task": [],
        "assistant_content": []
    }
    all_stats = {}
    
    # Sample from each task
    for task_type, target_count in targets.items():
        if task_type not in processors_map:
            print(f"Warning: No processors found for task type '{task_type}'")
            continue
        
        processors = processors_map[task_type]
        
        # Distribute target count across processors
        count_per_processor = target_count // len(processors)
        remainder = target_count % len(processors)
        
        task_train_samples = []
        task_val_samples = []
        task_stats = {}
        
        for i, processor in enumerate(processors):
            # Add remainder to last processor
            processor_target = count_per_processor + (remainder if i == len(processors) - 1 else 0)
            
            print(f"Sampling {processor_target} samples from {processor.__class__.__name__} for task '{task_type}'...")
            
            result = sampler.sample_from_processor(
                processor=processor,
                target_count=processor_target,
                task_type=task_type if isinstance(processor, MultiTaskDatasetProcessor) else None
            )
            
            task_train_samples.extend(result.train_samples)
            task_val_samples.extend(result.val_samples)
            task_stats[processor.__class__.__name__] = result.stats
            
            # Print warnings
            if result.stats["dataset_exhausted"]:
                print(f"  WARNING: Dataset exhausted! Got {result.stats['actual_count']} / {processor_target} samples")
            if result.stats["tokens_rejected"] > 0:
                rejection_rate = result.stats["tokens_rejected"] / result.stats["indices_checked"] * 100
                print(f"  Token rejection rate: {rejection_rate:.1f}%")
        
        # Add to main data structures
        for sample in task_train_samples:
            train_data["user_content"].append(sample["user_content"])
            train_data["task"].append(sample["task"])
            train_data["assistant_content"].append(sample["assistant_content"])
        
        for sample in task_val_samples:
            val_data["user_content"].append(sample["user_content"])
            val_data["task"].append(sample["task"])
            val_data["assistant_content"].append(sample["assistant_content"])
        
        all_stats[task_type] = task_stats
        
        print(f"Task '{task_type}': {len(task_train_samples)} train, {len(task_val_samples)} val samples")
        print()
    
    return train_data, val_data, all_stats