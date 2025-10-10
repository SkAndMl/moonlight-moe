from typing import Optional, Tuple
import datasets
from .base import BaseDatasetProcessor


class WritingPromptsProcessor(BaseDatasetProcessor):
    DATASET_ID = "euclaise/writingprompts"
    SPLIT = "train"
    TASK_TYPE = "generate_creative"
    
    def load_dataset(self) -> None:
        """Load WritingPrompts dataset from HuggingFace"""
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
            
            prompt = sample.get("prompt", "")
            story = sample.get("story", "")
            
            prompt = self._clean_prompt(prompt)
            
            if not prompt or not story:
                return None
            
            if len(story.strip()) < 100:
                return None
            
            user_content = prompt
            assistant_content = story
            
            return user_content, assistant_content
            
        except (KeyError, IndexError, Exception):
            return None
    
    def _clean_prompt(self, prompt: str) -> str:
        tags = ["[WP]", "[EU]", "[CW]", "[RF]", "[MP]", "[TT]", "[PI]", "[IP]"]
        for tag in tags:
            prompt = prompt.replace(tag, "")
        
        prompt = " ".join(prompt.split())
        return prompt.strip()
    
    def get_task_type(self) -> str:
        return self.TASK_TYPE