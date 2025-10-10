from typing import Optional, Tuple, Dict
import random
import datasets
from .base import BaseDatasetProcessor


class StoryClozeProcessor(BaseDatasetProcessor):
    DATASET_ID = "lecslab/story_cloze"
    TASK_TYPE = "inference"
    
    def __init__(self, token: str = None, random_seed: int = 2406):
        super().__init__(token)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
    
    def load_dataset(self) -> None:
        ds = datasets.load_dataset(
            self.DATASET_ID,
            token=self.token
        )
        
        self.dataset = datasets.concatenate_datasets([ds["train"], ds["test"]])
        self._size = len(self.dataset)
    
    def extract_sample(self, idx: int) -> Optional[Tuple[Dict[str, str], str]]:
        if not self.is_loaded():
            self.load_dataset()
        
        try:
            sample = self.dataset[idx]
            
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]
            
            if not prompt or not chosen or not rejected:
                return None
            
            if self._rng.random() > 0.5:
                option_1 = rejected
                option_2 = chosen
            else:
                option_1 = chosen
                option_2 = rejected
            
            user_content = {
                "paragraph": prompt,
                "option_1": option_1,
                "option_2": option_2
            }
            
            assistant_content = chosen
            
            return user_content, assistant_content
            
        except (KeyError, IndexError, Exception):
            return None
    
    def get_task_type(self) -> str:
        return self.TASK_TYPE