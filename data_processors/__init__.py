from .base import BaseDatasetProcessor, MultiTaskDatasetProcessor
from .tinystories import TinyStoriesProcessor
from .roc_stories import ROCStoriesProcessor
from .story_cloze import StoryClozeProcessor
from .writing_prompts import WritingPromptsProcessor


__all__ = [
    "BaseDatasetProcessor",
    "MultiTaskDatasetProcessor",
    "TinyStoriesProcessor",
    "ROCStoriesProcessor",
    "StoryClozeProcessor",
    "WritingPromptsProcessor",
]


TASK_TO_PROCESSORS = {
    "generate_long": [TinyStoriesProcessor],
    "generate_creative": [WritingPromptsProcessor],  # Separate task now
    "summarize": [TinyStoriesProcessor],
    "generate_short": [ROCStoriesProcessor],
    "inference": [StoryClozeProcessor],
}


def get_processors_for_task(task_type: str) -> list:
    if task_type not in TASK_TO_PROCESSORS:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Available tasks: {list(TASK_TO_PROCESSORS.keys())}"
        )
    return TASK_TO_PROCESSORS[task_type]


def get_all_processors() -> list:
    return [
        TinyStoriesProcessor,
        ROCStoriesProcessor,
        StoryClozeProcessor,
        WritingPromptsProcessor,
    ]