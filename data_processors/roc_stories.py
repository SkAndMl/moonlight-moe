from typing import Optional, Tuple
import datasets
from .base import BaseDatasetProcessor


class ROCStoriesProcessor(BaseDatasetProcessor):
    
    DATASET_ID = "igormorgado/ROCStories2018"
    SPLIT = "train"
    TASK_TYPE = "generate_short"
    
    def load_dataset(self) -> None:
        self.dataset = datasets.load_dataset(
            self.DATASET_ID,
            split=self.SPLIT,
            token=self.token
        )
        self._size = len(self.dataset)
    
    def extract_sample(self, idx: int) -> Optional[Tuple[str, str]]:
        if not self.is_loaded():
            self.load_dataset()
        
        try:
            sample = self.dataset[idx]
            user_content = sample["storytitle"]
            sentences = [sample[f"sentence{i}"] for i in range(1, 6)]
            assistant_content = "\n".join(sentences)
            
            if not user_content or not all(sentences):
                return None
            
            return user_content, assistant_content
            
        except (KeyError, IndexError, Exception):
            return None
    
    def get_task_type(self) -> str:
        return self.TASK_TYPE